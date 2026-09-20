from __future__ import annotations

import filecmp
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

from py_mtlf.config import ExperimentRecordingSettings
from py_mtlf.core.fl_hierarchy import normalize_nf_instance_id, normalize_plan_id
from py_mtlf.core.image_classification import (
    ImageClassificationDataset,
    ImageClassificationEvaluator,
    ImageDatasetLoader,
)
from py_mtlf.core.trainer import resolve_device
from py_mtlf.core.workloads import validate_image_manifest

EvaluationStage = Literal[
    "ROOT_INITIAL",
    "ROOT_GLOBAL",
    "BRANCH_DOMAIN",
    "LEAF_LOCAL",
]

FINAL_MODEL_FILENAME = "final-model.tar.gz"
_REQUEST_FIELDS = frozenset(
    {
        "notifCorreId",
        "suppFeats",
        "mLEventSubscs",
        "mLModelTrainInfos",
        "mLPreFlag",
        "mLTrainRepInfo",
        "roundInd",
        "mLModelInfos",
        "x-flTopology",
        "skipFlInd",
        "mLAccChkFlg",
    }
)
_NOTIFICATION_FIELDS = frozenset(
    {
        "notifCorreId",
        "roundInd",
        "x-flTopologyReport",
        "mLModelInfos",
        "statusReport",
        "termTrainReq",
        "delayEventNotif",
    }
)


class ExperimentRecorder:
    """Persist node-local hierarchical FL observations as append-only JSONL."""

    def __init__(
        self,
        settings: ExperimentRecordingSettings,
        nf_instance_id_provider: Callable[[], str] | None = None,
    ) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._opened = False
        self._nf_instance_id: str | None = None
        self._nf_instance_id_provider = nf_instance_id_provider
        self._validation_dataset: ImageClassificationDataset | None = None
        self._evaluator = ImageClassificationEvaluator()

    @property
    def validation_enabled(self) -> bool:
        return self._settings.validation is not None

    def open(self, nf_instance_id: str | None = None) -> None:
        normalized_nf_instance_id = (
            normalize_nf_instance_id(nf_instance_id)
            if nf_instance_id is not None
            else None
        )
        directory = self._settings.directory
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise RuntimeError("experiment recording directory is not a directory")
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=".pymtlf-write-probe-",
            ):
                pass
        except OSError as error:
            raise RuntimeError("experiment recording directory is not writable") from error
        validation = self._settings.validation
        dataset = None
        if validation is not None:
            dataset = ImageDatasetLoader().load(validation.path, validation.dataset)
        with self._lock:
            if self._opened:
                raise RuntimeError("experiment recorder is already open")
            self._opened = True
            self._nf_instance_id = normalized_nf_instance_id
            self._validation_dataset = dataset

    def start_procedure(self, ml_correlation_id: str) -> None:
        with self._lock:
            self._required_nf_instance_id()
            path = self._observation_path(normalize_plan_id(ml_correlation_id))
            path.touch(exist_ok=True)

    def close(self) -> None:
        with self._lock:
            self._opened = False
            self._nf_instance_id = None
            self._validation_dataset = None

    def record_model_evaluation(
        self,
        *,
        ml_correlation_id: str,
        evaluation_stage: EvaluationStage,
        model: torch.nn.Module,
        manifest: Mapping[str, object],
        round_indicator: int | None = None,
        recorded_at: datetime | None = None,
    ) -> None:
        validation = self._settings.validation
        dataset = self._validation_dataset
        if validation is None:
            return
        if dataset is None:
            raise RuntimeError("experiment recorder is not open")
        if evaluation_stage == "ROOT_INITIAL":
            if round_indicator is not None:
                raise ValueError("ROOT_INITIAL must not have a round indicator")
        elif round_indicator is None:
            raise ValueError("model evaluation requires a round indicator")
        else:
            round_indicator = _round_indicator(round_indicator)
        contract = validate_image_manifest(manifest)
        if contract.name != dataset.dataset or contract.name.value != validation.dataset:
            raise ValueError("validation dataset does not match model bundle dataset")
        result = self._evaluator.evaluate_metrics(
            model,
            dataset,
            device=resolve_device(validation.device),
            batch_size=validation.batch_size,
        )
        record: dict[str, object] = {
            "recordedAt": _recorded_at(recorded_at),
            "recordType": "MODEL_EVALUATION",
            "mlCorreId": normalize_plan_id(ml_correlation_id),
            "nfInstanceId": self._required_nf_instance_id(),
            "evaluationStage": evaluation_stage,
            "dataSplit": "VALIDATION",
            "dataset": result.dataset.value,
            "sampleCount": result.sample_count,
            "loss": result.mean_cross_entropy_loss,
            "accuracy": result.accuracy,
        }
        if round_indicator is not None:
            record["roundInd"] = round_indicator
        self._append(record)

    def save_final_model(
        self,
        *,
        ml_correlation_id: str,
        round_indicator: int,
        artifact_path: Path,
        artifact_digest: str,
        recorded_at: datetime | None = None,
    ) -> Path:
        normalized_correlation_id = normalize_plan_id(ml_correlation_id)
        normalized_round_indicator = _round_indicator(round_indicator)
        source = Path(artifact_path)
        if not source.is_file():
            raise RuntimeError("final model artifact is not a file")
        if (
            len(artifact_digest) != 64
            or artifact_digest != artifact_digest.lower()
            or any(character not in "0123456789abcdef" for character in artifact_digest)
        ):
            raise ValueError("final model artifact digest must be lowercase SHA-256")

        with self._lock:
            self._required_nf_instance_id()
            procedure_directory = self._procedure_directory(normalized_correlation_id)
            destination = procedure_directory / FINAL_MODEL_FILENAME
            if destination.exists():
                if not destination.is_file() or not filecmp.cmp(
                    source,
                    destination,
                    shallow=False,
                ):
                    raise RuntimeError("saved final model conflicts with existing artifact")
                return destination

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".final-model-",
                suffix=".tar.gz",
                dir=procedure_directory,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

            self._append_base(
                normalized_correlation_id,
                "MODEL_ARTIFACT_SAVED",
                recorded_at,
                roundInd=normalized_round_indicator,
                artifactFile=FINAL_MODEL_FILENAME,
            )
            return destination

    def record_root_round_outcome(
        self,
        *,
        ml_correlation_id: str,
        round_indicator: int,
        accepted: bool,
        selected_nf_instance_ids: tuple[str, ...],
        successful_nf_instance_ids: tuple[str, ...],
        failed_nf_instance_ids: tuple[str, ...],
        recorded_at: datetime | None = None,
    ) -> None:
        self._append_base(
            ml_correlation_id,
            "ROUND_AGGREGATION",
            recorded_at,
            roundInd=_round_indicator(round_indicator),
            accepted=accepted,
            selectedNfInstanceIds=_nf_instance_ids(selected_nf_instance_ids),
            successfulNfInstanceIds=_nf_instance_ids(successful_nf_instance_ids),
            failedNfInstanceIds=_nf_instance_ids(failed_nf_instance_ids),
        )

    def record_training_operation(
        self,
        *,
        ml_correlation_id: str,
        operation: Literal["CREATE", "PUT", "PATCH", "DELETE", "NOTIFY"],
        direction: Literal["SENT", "RECEIVED"],
        started_at: datetime,
        outcome: Literal["SUCCESS", "REJECTED", "FAILED"],
        message: Mapping[str, object] | None = None,
        subscription_id: str | None = None,
        target_nf_instance_id: str | None = None,
        source_nf_instance_id: str | None = None,
        cause: str | None = None,
        recorded_at: datetime | None = None,
    ) -> None:
        fields: dict[str, object] = {
            "operation": operation,
            "direction": direction,
            "startedAt": _recorded_at(started_at),
            "outcome": outcome,
        }
        if subscription_id:
            fields["subscriptionId"] = subscription_id
        if target_nf_instance_id:
            fields["targetNfInstanceId"] = normalize_nf_instance_id(target_nf_instance_id)
        if source_nf_instance_id:
            fields["sourceNfInstanceId"] = normalize_nf_instance_id(source_nf_instance_id)
        if cause:
            fields["cause"] = cause
        if message is not None and operation != "DELETE":
            fields["message"] = _message_excerpt(message, operation)
        self._append_base(ml_correlation_id, "MODEL_TRAINING_OPERATION", recorded_at, **fields)

    def record_decision(
        self,
        *,
        ml_correlation_id: str,
        record_type: Literal[
            "CANDIDATE_SELECTION",
            "EDGE_CONFIRMED",
            "EDGE_UNAVAILABLE",
            "REPAIR_SELECTION",
            "TOPOLOGY_ACCEPTANCE",
            "ROUND_AGGREGATION",
        ],
        recorded_at: datetime | None = None,
        **fields: object,
    ) -> None:
        self._append_base(ml_correlation_id, record_type, recorded_at, **fields)

    def _append_base(
        self,
        ml_correlation_id: str,
        record_type: str,
        recorded_at: datetime | None,
        **fields: object,
    ) -> None:
        self._append(
            {
                "recordedAt": _recorded_at(recorded_at),
                "recordType": record_type,
                "mlCorreId": normalize_plan_id(ml_correlation_id),
                "nfInstanceId": self._required_nf_instance_id(),
                **fields,
            }
        )

    def _append(self, record: Mapping[str, object]) -> None:
        ml_correlation_id = normalize_plan_id(str(record["mlCorreId"]))
        serialized = json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._required_nf_instance_id()
            path = self._observation_path(ml_correlation_id)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
                stream.flush()

    def _observation_path(self, ml_correlation_id: str) -> Path:
        return self._procedure_directory(ml_correlation_id) / "observations.jsonl"

    def _procedure_directory(self, ml_correlation_id: str) -> Path:
        procedure_directory = self._settings.directory / ml_correlation_id
        procedure_directory.mkdir(parents=True, exist_ok=True)
        return procedure_directory

    def _required_nf_instance_id(self) -> str:
        if not self._opened:
            raise RuntimeError("experiment recorder is not open")
        if self._nf_instance_id is None:
            if self._nf_instance_id_provider is None:
                raise RuntimeError("experiment recorder has no NF instance identity")
            self._nf_instance_id = normalize_nf_instance_id(
                self._nf_instance_id_provider()
            )
        return self._nf_instance_id


def _recorded_at(value: datetime | None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("recordedAt must include a timezone")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _round_indicator(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("round indicator must be a non-negative integer")
    return value


def _nf_instance_ids(values: tuple[str, ...]) -> list[str]:
    return [normalize_nf_instance_id(value) for value in values]


def _message_excerpt(message: Mapping[str, object], operation: str) -> dict[str, object]:
    allowed = _NOTIFICATION_FIELDS if operation == "NOTIFY" else _REQUEST_FIELDS
    excerpt = {key: value for key, value in message.items() if key in allowed}
    infos = excerpt.get("mLModelInfos")
    if isinstance(infos, list):
        excerpt["mLModelInfos"] = [
            {
                key: value
                for key, value in info.items()
                if key in {"event", "modelUniqueId", "mLFileAddr", "mLModelAdrf"}
            }
            for info in infos
            if isinstance(info, dict)
        ]
    return excerpt
