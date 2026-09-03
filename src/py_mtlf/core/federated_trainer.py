import copy
import math
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from py_mtlf.config import FittingRuntimeSettings
from py_mtlf.core.trainer import LoadedBundle, LocalTrainer, TrainingError, resolve_device
from py_mtlf.core.training_data import TrainingDataset


@dataclass(frozen=True)
class FederatedTrainingResult:
    model: torch.nn.Module
    training_sample_count: int
    final_loss: float


class FederatedTrainer:
    """Train full local weights while preserving the Server-provided scaler."""

    def __init__(self, settings: FittingRuntimeSettings) -> None:
        self._settings = settings
        self._device = resolve_device(settings.device)

    def train(
        self,
        base: LoadedBundle,
        dataset: TrainingDataset,
        *,
        epochs: int,
        proximal_mu: float | None = None,
    ) -> FederatedTrainingResult:
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if proximal_mu is not None and (not math.isfinite(proximal_mu) or proximal_mu < 0):
            raise ValueError("proximal_mu must be finite and non-negative")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        np.random.seed(self._settings.random_seed)
        torch.manual_seed(self._settings.random_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        inputs, targets = LocalTrainer._training_tensors(dataset, base.scaler)
        model = copy.deepcopy(base.model).to(self._device)
        global_reference = {
            name: value.detach().clone().to(self._device)
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self._settings.learning_rate)
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
            for _epoch in range(epochs):
                for features, expected in loader:
                    features = features.to(self._device)
                    expected = expected.to(self._device)
                    optimizer.zero_grad(set_to_none=True)
                    actual = model(features)
                    if actual.shape != expected.shape:
                        raise TrainingError("federated model output shape is incompatible")
                    loss = loss_function(actual, expected)
                    if proximal_mu is not None:
                        penalty = sum(
                            torch.sum((parameter - global_reference[name]) ** 2)
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad
                        )
                        loss = loss + (proximal_mu / 2) * penalty
                    if not torch.isfinite(loss):
                        raise TrainingError("federated training loss is not finite")
                    loss.backward()
                    optimizer.step()
                    final_loss = float(loss.detach().cpu().item())
            if not math.isfinite(final_loss):
                raise TrainingError("federated training produced no finite loss")
        finally:
            model.to("cpu")
        model.eval()
        return FederatedTrainingResult(
            model=model,
            training_sample_count=sum(
                scope.training_sample_count for scope in dataset.training_scopes
            ),
            final_loss=final_loss,
        )

    @staticmethod
    def aggregate(
        base: LoadedBundle,
        participants: tuple[tuple[LoadedBundle, int], ...],
    ) -> torch.nn.Module:
        if not participants:
            raise TrainingError("FedAvg requires at least one participant")
        total = sum(sample_count for _bundle, sample_count in participants)
        if total <= 0:
            raise TrainingError("FedAvg sample count must be positive")
        base_state = base.model.state_dict()
        aggregate = OrderedDict()
        for name, base_value in base_state.items():
            values = [bundle.model.state_dict()[name] for bundle, _count in participants]
            if any(
                value.shape != base_value.shape or value.dtype != base_value.dtype
                for value in values
            ):
                raise TrainingError("FedAvg tensor contract does not match the global model")
            if base_value.is_floating_point():
                weighted = sum(
                    value.to(torch.float64) * (count / total)
                    for (bundle, count), value in zip(participants, values, strict=True)
                )
                aggregate[name] = weighted.to(base_value.dtype)
            else:
                if any(not torch.equal(value, base_value) for value in values):
                    raise TrainingError("FedAvg non-floating tensors must remain unchanged")
                aggregate[name] = base_value.detach().clone()
        model = copy.deepcopy(base.model)
        model.load_state_dict(aggregate, strict=True)
        model.eval()
        return model
