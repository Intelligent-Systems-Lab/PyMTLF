import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from py_mtlf.config import TrainingSettings
from py_mtlf.core.dataset import DatasetSnapshot

logger = logging.getLogger(__name__)

FEATURE_ORDER = (
    "total_vol",
    "ul_vol",
    "dl_vol",
    "total_nb_pkts",
    "ul_nb_pkts",
    "dl_nb_pkts",
    "ul_thr",
    "dl_thr",
    "ul_pkt_thr",
    "dl_pkt_thr",
)

_VOLUME_FIELDS = (
    ("total_vol", "totalVolume"),
    ("ul_vol", "ulVolume"),
    ("dl_vol", "dlVolume"),
    ("total_nb_pkts", "totalNbOfPackets"),
    ("ul_nb_pkts", "ulNbOfPackets"),
    ("dl_nb_pkts", "dlNbOfPackets"),
)
_RATE_FIELDS = (
    ("ul_thr", "ulThroughput", "bit"),
    ("dl_thr", "dlThroughput", "bit"),
    ("ul_pkt_thr", "ulPacketThroughput", "packet"),
    ("dl_pkt_thr", "dlPacketThroughput", "packet"),
)


class TrainingDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScopeTrainingData:
    scope_key: str
    scope_digest: str
    observation_count: int
    training_inputs: np.ndarray | None
    training_targets: np.ndarray | None
    validation_inputs: np.ndarray | None
    validation_targets: np.ndarray | None
    training_observations: np.ndarray | None
    training_eligible: bool
    evaluation_eligible: bool
    exclusion_reason: str

    @property
    def training_sample_count(self) -> int:
        return 0 if self.training_inputs is None else int(len(self.training_inputs))

    @property
    def validation_sample_count(self) -> int:
        return 0 if self.validation_inputs is None else int(len(self.validation_inputs))


@dataclass(frozen=True)
class TrainingDataset:
    feature_order: tuple[str, ...]
    output_fields: tuple[str, ...]
    output_indices: tuple[int, ...]
    seq_length: int
    out_seq_len: int
    triggering_scope_key: str
    scopes: tuple[ScopeTrainingData, ...]

    @property
    def training_scopes(self) -> tuple[ScopeTrainingData, ...]:
        return tuple(scope for scope in self.scopes if scope.training_eligible)

    @property
    def evaluation_scopes(self) -> tuple[ScopeTrainingData, ...]:
        return tuple(scope for scope in self.scopes if scope.evaluation_eligible)


@dataclass
class _Bucket:
    sums: dict[str, float]
    rate_sums: dict[str, float]
    rate_counts: dict[str, int]


class TrainingDatasetBuilder:
    def __init__(self, settings: TrainingSettings) -> None:
        self._settings = settings

    def build(
        self,
        snapshot: DatasetSnapshot,
        current_manifest: dict[str, object],
    ) -> TrainingDataset:
        inference = current_manifest.get("inference")
        model = current_manifest.get("model")
        if not isinstance(inference, dict) or not isinstance(model, dict):
            raise TrainingDataError("current bundle has no model/inference configuration")
        feature_order = tuple(inference.get("feature_order") or ())
        output_fields = tuple(inference.get("output_fields") or ())
        seq_length = inference.get("seq_length")
        out_seq_len = inference.get("out_seq_len", 1)
        if feature_order != FEATURE_ORDER:
            raise TrainingDataError("current bundle feature order is not supported")
        if not output_fields or any(field not in feature_order for field in output_fields):
            raise TrainingDataError("current bundle output fields are invalid")
        if not isinstance(seq_length, int) or seq_length <= 0:
            raise TrainingDataError("current bundle seq_length is invalid")
        if not isinstance(out_seq_len, int) or out_seq_len <= 0:
            raise TrainingDataError("current bundle out_seq_len is invalid")
        expected_output = len(output_fields) * out_seq_len
        if model.get("input_size") != len(feature_order):
            raise TrainingDataError("current model input_size does not match feature order")
        if model.get("output_size") != expected_output:
            raise TrainingDataError("current model output_size does not match prediction target")
        if inference.get("preprocessing") != "log1p_standard_scaler":
            raise TrainingDataError("current bundle preprocessing is not supported")

        observations = self._observations(snapshot)
        output_indices = tuple(feature_order.index(field) for field in output_fields)
        purge = seq_length + out_seq_len - 1
        scopes = []
        for scope_key in snapshot.required_scope_keys:
            values = observations.get(scope_key, ())
            scope = self._scope_dataset(
                scope_key,
                values,
                seq_length,
                out_seq_len,
                output_indices,
                purge,
            )
            scopes.append(scope)
            if scope_key == snapshot.triggering_scope_key and (
                not scope.training_eligible or not scope.evaluation_eligible
            ):
                raise TrainingDataError(
                    "triggering scope has insufficient training/reference-validation data"
                )
            if scope_key != snapshot.triggering_scope_key and scope.exclusion_reason:
                logger.warning(
                    "Training scope degraded scope=%s reason=%s observations=%s "
                    "training_samples=%s validation_samples=%s",
                    scope.scope_digest,
                    scope.exclusion_reason,
                    scope.observation_count,
                    scope.training_sample_count,
                    scope.validation_sample_count,
                )
        if not any(scope.scope_key == snapshot.triggering_scope_key for scope in scopes):
            raise TrainingDataError("triggering scope is not present in dataset snapshot")
        return TrainingDataset(
            feature_order=feature_order,
            output_fields=output_fields,
            output_indices=output_indices,
            seq_length=seq_length,
            out_seq_len=out_seq_len,
            triggering_scope_key=snapshot.triggering_scope_key,
            scopes=tuple(scopes),
        )

    def _scope_dataset(
        self,
        scope_key: str,
        observations: tuple[np.ndarray, ...],
        seq_length: int,
        out_seq_len: int,
        output_indices: tuple[int, ...],
        purge: int,
    ) -> ScopeTrainingData:
        values = (
            np.asarray(observations, dtype=np.float64)
            if observations
            else np.empty((0, len(FEATURE_ORDER)), dtype=np.float64)
        )
        split = int(math.floor(len(values) * self._settings.validation_ratio))
        validation_observations = values[: max(split - purge, 0)]
        training_observations = values[split:]
        validation_inputs, validation_targets = self._windows(
            validation_observations,
            seq_length,
            out_seq_len,
            output_indices,
        )
        training_inputs, training_targets = self._windows(
            training_observations,
            seq_length,
            out_seq_len,
            output_indices,
        )
        training_eligible = len(training_inputs) > 0
        evaluation_eligible = len(validation_inputs) > 0
        reason = ""
        if not training_eligible:
            reason = "insufficient_training_data"
        elif not evaluation_eligible:
            reason = "insufficient_reference_validation_data"
        return ScopeTrainingData(
            scope_key=scope_key,
            scope_digest=hashlib.sha256(scope_key.encode()).hexdigest(),
            observation_count=len(values),
            training_inputs=training_inputs if training_eligible else None,
            training_targets=training_targets if training_eligible else None,
            validation_inputs=validation_inputs if evaluation_eligible else None,
            validation_targets=validation_targets if evaluation_eligible else None,
            training_observations=(training_observations if training_eligible else None),
            training_eligible=training_eligible,
            evaluation_eligible=evaluation_eligible,
            exclusion_reason=reason,
        )

    @staticmethod
    def _windows(
        observations: np.ndarray,
        seq_length: int,
        out_seq_len: int,
        output_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(observations) - seq_length - out_seq_len + 1
        if count <= 0:
            return (
                np.empty((0, seq_length, len(FEATURE_ORDER)), dtype=np.float64),
                np.empty((0, out_seq_len * len(output_indices)), dtype=np.float64),
            )
        inputs = []
        targets = []
        for start in range(count):
            target_start = start + seq_length
            inputs.append(observations[start:target_start])
            targets.append(
                observations[
                    target_start : target_start + out_seq_len,
                    output_indices,
                ].reshape(-1)
            )
        return np.asarray(inputs), np.asarray(targets)

    def _observations(
        self,
        snapshot: DatasetSnapshot,
    ) -> dict[str, tuple[np.ndarray, ...]]:
        scope_buckets: dict[str, dict[datetime, _Bucket]] = defaultdict(dict)
        malformed_items = 0
        invalid_timestamps = 0
        outside_window = 0
        invalid_measurements = 0
        for dataset_record in snapshot.records:
            record = dataset_record.record
            data_notif = record.get("dataNotif") if isinstance(record, dict) else None
            notifications = (
                data_notif.get("upfEventNotifs", []) if isinstance(data_notif, dict) else []
            )
            for notification in notifications:
                items = (
                    notification.get("notificationItems", [])
                    if isinstance(notification, dict)
                    else []
                )
                for item in items:
                    if not isinstance(item, dict):
                        malformed_items += 1
                        continue
                    timestamp = self._timestamp(
                        item.get("startTime") or item.get("timeStamp"),
                        dataset_record.measurement_time,
                    )
                    if timestamp is None:
                        invalid_timestamps += 1
                        continue
                    if not (
                        snapshot.time_window.start_time
                        <= timestamp
                        <= snapshot.time_window.stop_time
                    ):
                        outside_window += 1
                        continue
                    measurements = item.get("userDataUsageMeasurements", [])
                    if not isinstance(measurements, list):
                        malformed_items += 1
                        continue
                    for measurement in measurements:
                        values, rates = self._measurement(measurement)
                        if values is None:
                            invalid_measurements += 1
                            continue
                        for scope_key in dataset_record.scope_keys:
                            bucket = scope_buckets[scope_key].setdefault(
                                timestamp,
                                _Bucket(
                                    sums={field: 0.0 for field, _source in _VOLUME_FIELDS},
                                    rate_sums={
                                        field: 0.0 for field, _source, _kind in _RATE_FIELDS
                                    },
                                    rate_counts={
                                        field: 0 for field, _source, _kind in _RATE_FIELDS
                                    },
                                ),
                            )
                            for field, value in values.items():
                                bucket.sums[field] += value
                            for field, value in rates.items():
                                if value is not None:
                                    bucket.rate_sums[field] += value
                                    bucket.rate_counts[field] += 1
        malformed = malformed_items + invalid_timestamps + outside_window + invalid_measurements
        if malformed:
            logger.warning(
                "Training record conversion skipped observations count=%s "
                "malformed_items=%s invalid_timestamps=%s outside_window=%s "
                "invalid_measurements=%s",
                malformed,
                malformed_items,
                invalid_timestamps,
                outside_window,
                invalid_measurements,
            )
        output: dict[str, tuple[np.ndarray, ...]] = {}
        for scope_key, buckets in scope_buckets.items():
            rows = []
            for _timestamp, bucket in sorted(buckets.items()):
                values = dict(bucket.sums)
                for field in bucket.rate_sums:
                    count = bucket.rate_counts[field]
                    values[field] = bucket.rate_sums[field] / count if count else 0.0
                row = np.asarray([values[field] for field in FEATURE_ORDER], dtype=np.float64)
                if np.isfinite(row).all() and (row >= 0).all():
                    rows.append(row)
            output[scope_key] = tuple(rows)
        return output

    @staticmethod
    def _measurement(
        measurement: object,
    ) -> tuple[dict[str, float] | None, dict[str, float | None]]:
        if not isinstance(measurement, dict):
            return None, {}
        volume = measurement.get("volumeMeasurement")
        throughput = measurement.get("throughputMeasurement")
        volume = volume if isinstance(volume, dict) else {}
        throughput = throughput if isinstance(throughput, dict) else {}
        values: dict[str, float] = {}
        for field, source in _VOLUME_FIELDS:
            try:
                value = float(volume.get(source, 0))
            except (TypeError, ValueError):
                return None, {}
            if not math.isfinite(value) or value < 0:
                return None, {}
            values[field] = value
        rates: dict[str, float | None] = {}
        for field, source, kind in _RATE_FIELDS:
            try:
                rates[field] = _parse_rate(throughput.get(source, ""), kind)
            except (TypeError, ValueError):
                rates[field] = None
        return values, rates

    @staticmethod
    def _timestamp(value: object, fallback: datetime | None) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            parsed = fallback
        if parsed is None:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)


def _parse_rate(value: object, kind: str) -> float | None:
    if value in {None, ""}:
        return None
    parts = str(value).split(" ", 1)
    number = float(parts[0])
    if not math.isfinite(number) or number < 0:
        raise ValueError("rate must be finite and non-negative")
    if len(parts) == 1:
        return number
    units = {
        "bit": {"bps": 1, "Kbps": 1e3, "Mbps": 1e6, "Gbps": 1e9, "Tbps": 1e12},
        "packet": {"pps": 1, "kpps": 1e3, "Mpps": 1e6, "Gpps": 1e9, "Tpps": 1e12},
    }
    multiplier = units[kind].get(parts[1])
    if multiplier is None:
        raise ValueError(f"unknown {kind} rate unit")
    return number * multiplier


def eligibility_manifest(dataset: TrainingDataset) -> list[dict[str, object]]:
    return [
        {
            "scope_key_sha256": scope.scope_digest,
            "triggering_scope": scope.scope_key == dataset.triggering_scope_key,
            "observation_count": scope.observation_count,
            "training_sample_count": scope.training_sample_count,
            "validation_sample_count": scope.validation_sample_count,
            "training_eligible": scope.training_eligible,
            "evaluation_eligible": scope.evaluation_eligible,
            "exclusion_reason": scope.exclusion_reason,
        }
        for scope in dataset.scopes
    ]
