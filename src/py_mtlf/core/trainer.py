import copy
import inspect
import math
import tarfile
import tempfile
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from py_mtlf.config import TrainingSettings
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.training_data import ScopeTrainingData, TrainingDataError, TrainingDataset


class TrainingError(RuntimeError):
    pass


@dataclass
class LoadedBundle:
    manifest: dict[str, object]
    model: torch.nn.Module
    scaler: StandardScaler
    model_source: bytes
    scaler_source: bytes = b""


@dataclass(frozen=True)
class WapeMetric:
    error_sum: float
    actual_sum: float
    value: float


@dataclass(frozen=True)
class ScopeEvaluation:
    scope_digest: str
    triggering_scope: bool
    current: WapeMetric
    candidate: WapeMetric
    delta: float


@dataclass(frozen=True)
class CandidateEvaluation:
    scopes: tuple[ScopeEvaluation, ...]
    aggregate_current: WapeMetric
    aggregate_candidate: WapeMetric
    aggregate_delta: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass
class TrainingResult:
    model: torch.nn.Module
    scaler: StandardScaler
    model_source: bytes
    manifest: dict[str, object]
    evaluation: CandidateEvaluation
    final_loss: float


class TrustedBundleLoader:
    def load(self, artifact: ArtifactMetadata) -> LoadedBundle:
        with tempfile.TemporaryDirectory(prefix="py-mtlf-bundle-") as temporary:
            root = Path(temporary)
            self._extract(artifact.path, root)
            manifest = self._manifest(root / "config.json")
            model_source = (root / "model.py").read_bytes()
            scaler_source = (root / "scaler.pkl").read_bytes()
            model_class = self._model_class(model_source, root / "model.py")
            model = self._instantiate(model_class, manifest)
            self._load_weights(model, root / "model.npy")
            scaler = joblib.load(root / "scaler.pkl")
        self._validate_scaler(scaler, manifest)
        model.to("cpu")
        model.eval()
        return LoadedBundle(
            manifest=manifest,
            model=model,
            scaler=scaler,
            model_source=model_source,
            scaler_source=scaler_source,
        )

    @staticmethod
    def _extract(path: Path, destination: Path) -> None:
        try:
            with tarfile.open(path, "r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isreg() or "/" in member.name or member.name.startswith("."):
                        raise TrainingError("trusted bundle contains an unsafe entry")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise TrainingError("trusted bundle entry cannot be read")
                    (destination / member.name).write_bytes(stream.read())
        except (OSError, tarfile.TarError) as error:
            raise TrainingError("trusted bundle cannot be extracted") from error

    @staticmethod
    def _manifest(path: Path) -> dict[str, object]:
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingError("trusted bundle manifest cannot be read") from error
        if not isinstance(value, dict):
            raise TrainingError("trusted bundle manifest must be an object")
        return value

    @staticmethod
    def _model_class(source: bytes, path: Path):
        module = types.ModuleType(f"trusted_model_{hash(source)}")
        module.__file__ = str(path)
        try:
            exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
            return module.Model
        except Exception as error:
            raise TrainingError("trusted bundle Model entry point cannot be loaded") from error

    @staticmethod
    def _instantiate(model_class, manifest: dict[str, object]) -> torch.nn.Module:
        model_config = manifest.get("model")
        if not isinstance(model_config, dict):
            raise TrainingError("trusted bundle model configuration is missing")
        signature = inspect.signature(model_class)
        kwargs = {
            name: model_config[name]
            for name in (
                "input_size",
                "output_size",
                "num_channels",
                "kernel_size",
                "dropout",
            )
            if name in signature.parameters and name in model_config
        }
        try:
            return model_class(**kwargs)
        except Exception as error:
            raise TrainingError("trusted model cannot be instantiated") from error

    @staticmethod
    def _load_weights(model: torch.nn.Module, path: Path) -> None:
        try:
            weights = np.load(path, allow_pickle=True)
            parameters = zip(model.state_dict().keys(), weights, strict=True)
            state = OrderedDict(
                (name, torch.as_tensor(np.asarray(value))) for name, value in parameters
            )
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise TrainingError("trusted model weights cannot be loaded") from error

    @staticmethod
    def _validate_scaler(scaler: object, manifest: dict[str, object]) -> None:
        inference = manifest.get("inference")
        features = inference.get("feature_order") if isinstance(inference, dict) else None
        if not isinstance(scaler, StandardScaler):
            raise TrainingError("trusted bundle scaler is not a StandardScaler")
        if not isinstance(features, list) or scaler.n_features_in_ != len(features):
            raise TrainingError("trusted bundle scaler shape is incompatible")


class LocalTrainer:
    def __init__(self, settings: TrainingSettings) -> None:
        self._settings = settings

    def train(
        self,
        current: LoadedBundle,
        candidate_base: LoadedBundle,
        dataset: TrainingDataset,
    ) -> TrainingResult:
        self._seed()
        scaler = self._fit_scaler(dataset)
        inputs, targets = self._training_tensors(dataset, scaler)
        model = candidate_base.model.to("cpu")
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._settings.learning_rate,
        )
        loss_function = torch.nn.HuberLoss()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._settings.random_seed)
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=self._settings.batch_size,
            shuffle=True,
            generator=generator,
        )
        final_loss = math.nan
        for _epoch in range(self._settings.epochs):
            for features, expected in loader:
                optimizer.zero_grad(set_to_none=True)
                actual = model(features)
                if actual.shape != expected.shape:
                    raise TrainingError("candidate model output shape is incompatible")
                loss = loss_function(actual, expected)
                if not torch.isfinite(loss):
                    raise TrainingError("training loss is not finite")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
        if not math.isfinite(final_loss):
            raise TrainingError("training produced no finite loss")
        model.eval()
        evaluation = self._evaluate(current, model, scaler, dataset)
        manifest = copy.deepcopy(candidate_base.manifest)
        return TrainingResult(
            model=model,
            scaler=scaler,
            model_source=candidate_base.model_source,
            manifest=manifest,
            evaluation=evaluation,
            final_loss=final_loss,
        )

    def _seed(self) -> None:
        np.random.seed(self._settings.random_seed)
        torch.manual_seed(self._settings.random_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def _fit_scaler(dataset: TrainingDataset) -> StandardScaler:
        observations = np.concatenate(
            [
                scope.training_observations
                for scope in dataset.training_scopes
                if scope.training_observations is not None
            ],
            axis=0,
        )
        transformed = np.log1p(observations)
        if not np.isfinite(transformed).all():
            raise TrainingDataError("training observations are not finite")
        return StandardScaler().fit(transformed)

    @staticmethod
    def _training_tensors(
        dataset: TrainingDataset,
        scaler: StandardScaler,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.concatenate(
            [
                scope.training_inputs
                for scope in dataset.training_scopes
                if scope.training_inputs is not None
            ]
        )
        targets = np.concatenate(
            [
                scope.training_targets
                for scope in dataset.training_scopes
                if scope.training_targets is not None
            ]
        )
        transformed_inputs = scaler.transform(
            np.log1p(inputs.reshape(-1, len(dataset.feature_order)))
        ).reshape(inputs.shape)
        output_indices = np.asarray(dataset.output_indices)
        repeated_mean = np.tile(scaler.mean_[output_indices], dataset.out_seq_len)
        repeated_scale = np.tile(scaler.scale_[output_indices], dataset.out_seq_len)
        transformed_targets = (np.log1p(targets) - repeated_mean) / repeated_scale
        if not (np.isfinite(transformed_inputs).all() and np.isfinite(transformed_targets).all()):
            raise TrainingDataError("training tensors are not finite")
        return (
            torch.as_tensor(
                transformed_inputs.transpose(0, 2, 1),
                dtype=torch.float32,
            ),
            torch.as_tensor(transformed_targets, dtype=torch.float32),
        )

    def _evaluate(
        self,
        current: LoadedBundle,
        candidate: torch.nn.Module,
        candidate_scaler: StandardScaler,
        dataset: TrainingDataset,
    ) -> CandidateEvaluation:
        evaluations = []
        current_error = current_actual = 0.0
        candidate_error = candidate_actual = 0.0
        for scope in dataset.evaluation_scopes:
            current_prediction = self._predict(
                current.model,
                current.scaler,
                scope,
                dataset,
            )
            candidate_prediction = self._predict(
                candidate,
                candidate_scaler,
                scope,
                dataset,
            )
            expected = scope.validation_targets
            if expected is None:
                continue
            current_metric = wape(expected, current_prediction)
            candidate_metric = wape(expected, candidate_prediction)
            current_error += current_metric.error_sum
            current_actual += current_metric.actual_sum
            candidate_error += candidate_metric.error_sum
            candidate_actual += candidate_metric.actual_sum
            evaluations.append(
                ScopeEvaluation(
                    scope_digest=scope.scope_digest,
                    triggering_scope=scope.scope_key == dataset.triggering_scope_key,
                    current=current_metric,
                    candidate=candidate_metric,
                    delta=candidate_metric.value - current_metric.value,
                )
            )
        if not any(item.triggering_scope for item in evaluations):
            raise TrainingDataError("triggering scope has no evaluation result")
        aggregate_current = wape_sums(current_error, current_actual)
        aggregate_candidate = wape_sums(candidate_error, candidate_actual)
        reasons = []
        if self._settings.enforce_performance_gate:
            trigger = next(item for item in evaluations if item.triggering_scope)
            if trigger.candidate.value >= trigger.current.value:
                reasons.append("triggering_scope_not_improved")
            if aggregate_candidate.value >= aggregate_current.value:
                reasons.append("aggregate_not_improved")
            for item in evaluations:
                if (
                    not item.triggering_scope
                    and item.delta > self._settings.max_scope_wape_regression
                ):
                    reasons.append(f"scope_regression_exceeded:{item.scope_digest}")
        return CandidateEvaluation(
            scopes=tuple(evaluations),
            aggregate_current=aggregate_current,
            aggregate_candidate=aggregate_candidate,
            aggregate_delta=aggregate_candidate.value - aggregate_current.value,
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _predict(
        model: torch.nn.Module,
        scaler: StandardScaler,
        scope: ScopeTrainingData,
        dataset: TrainingDataset,
    ) -> np.ndarray:
        inputs = scope.validation_inputs
        if inputs is None:
            raise TrainingDataError("evaluation scope has no validation inputs")
        transformed = scaler.transform(
            np.log1p(inputs.reshape(-1, len(dataset.feature_order)))
        ).reshape(inputs.shape)
        tensor = torch.as_tensor(
            transformed.transpose(0, 2, 1),
            dtype=torch.float32,
        )
        model.eval()
        with torch.no_grad():
            output = model(tensor).detach().cpu().numpy()
        output_indices = np.asarray(dataset.output_indices)
        repeated_mean = np.tile(scaler.mean_[output_indices], dataset.out_seq_len)
        repeated_scale = np.tile(scaler.scale_[output_indices], dataset.out_seq_len)
        prediction = np.expm1(output * repeated_scale + repeated_mean)
        if not np.isfinite(prediction).all():
            raise TrainingError("model evaluation produced non-finite predictions")
        return np.maximum(prediction, 0)


def wape(actual: np.ndarray, predicted: np.ndarray) -> WapeMetric:
    if actual.shape != predicted.shape:
        raise TrainingError("WAPE inputs have different shapes")
    return wape_sums(
        float(np.abs(actual - predicted).sum()),
        float(np.abs(actual).sum()),
    )


def wape_sums(error_sum: float, actual_sum: float) -> WapeMetric:
    if not math.isfinite(error_sum) or not math.isfinite(actual_sum):
        raise TrainingError("WAPE inputs are not finite")
    if actual_sum > 0:
        value = error_sum / actual_sum
    elif error_sum == 0:
        value = 0.0
    else:
        value = 1.0
    return WapeMetric(error_sum=error_sum, actual_sum=actual_sum, value=value)
