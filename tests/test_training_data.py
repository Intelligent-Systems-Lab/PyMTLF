from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from py_mtlf.config import FittingSettings
from py_mtlf.core.dataset import DatasetRecord, DatasetSnapshot
from py_mtlf.core.training_data import (
    FEATURE_ORDER,
    TrainingDataError,
    TrainingDatasetBuilder,
    dataset_evidence,
)
from py_mtlf.wire.adrf import TimeWindow


def snapshot(*, second_scope_observations: int = 400) -> DatasetSnapshot:
    start = datetime(2026, 7, 25, tzinfo=UTC)
    items = []
    for index in range(400):
        items.append(
            {
                "eventType": "USER_DATA_USAGE_MEASURES",
                "ueIpv4Addr": "10.0.0.1",
                "timeStamp": (start + timedelta(seconds=index)).isoformat(),
                "userDataUsageMeasurements": [
                    {
                        "volumeMeasurement": {
                            "totalVolume": 1,
                            "ulVolume": 2,
                            "dlVolume": 3,
                            "totalNbOfPackets": 4,
                            "ulNbOfPackets": 5,
                            "dlNbOfPackets": 6,
                        },
                        "throughputMeasurement": {
                            "ulThroughput": "1 Mbps",
                            "dlThroughput": "2 Mbps",
                            "ulPacketThroughput": "3 kpps",
                            "dlPacketThroughput": "4 kpps",
                        },
                    },
                    {
                        "volumeMeasurement": {
                            "totalVolume": 2,
                            "ulVolume": 4,
                            "dlVolume": 6,
                            "totalNbOfPackets": 8,
                            "ulNbOfPackets": 10,
                            "dlNbOfPackets": 12,
                        },
                        "throughputMeasurement": {
                            "ulThroughput": "3 Mbps",
                            "dlThroughput": "4 Mbps",
                            "ulPacketThroughput": "5 kpps",
                            "dlPacketThroughput": "6 kpps",
                        },
                    },
                ],
            }
        )
    records = [
        DatasetRecord(
            identity="record-a",
            scope_keys=("scope-a",),
            source="mongodb",
            measurement_time=None,
            record={
                "dataSub": [{"smfDataSub": {"supi": "imsi-a"}}],
                "dataNotif": {
                    "upfEventNotifs": [
                        {
                            "correlationId": "corr-a",
                            "notificationItems": items,
                        }
                    ]
                },
            },
        ),
        DatasetRecord(
            identity="record-b",
            scope_keys=("scope-b",),
            source="mongodb",
            measurement_time=None,
            record={
                "dataSub": [{"smfDataSub": {"supi": "imsi-b"}}],
                "dataNotif": {
                    "upfEventNotifs": [
                        {
                            "correlationId": "corr-b",
                            "notificationItems": items[:second_scope_observations],
                        }
                    ]
                },
            },
        ),
    ]
    return DatasetSnapshot(
        job_id="dataset-a",
        family_key="ue-communication-default",
        triggering_scope_key="scope-a",
        required_scope_keys=("scope-a", "scope-b"),
        time_window=TimeWindow(
            startTime=start,
            stopTime=start + timedelta(seconds=500),
        ),
        source="mongodb",
        records=tuple(records),
        scope_record_counts={"scope-a": 1, "scope-b": 1},
    )


def manifest() -> dict[str, object]:
    return {
        "model": {"input_size": 10, "output_size": 2},
        "inference": {
            "seq_length": 30,
            "out_seq_len": 1,
            "feature_order": list(FEATURE_ORDER),
            "output_fields": ["ul_vol", "dl_vol"],
            "preprocessing": "log1p_standard_scaler",
        },
    }


def test_builder_preserves_feature_aggregation_and_purged_chronological_split():
    dataset = TrainingDatasetBuilder(FittingSettings()).build(snapshot(), manifest())

    scope = dataset.scopes[0]
    assert scope.observation_count == 400
    assert scope.validation_sample_count == 34
    assert scope.training_sample_count == 306
    assert scope.training_inputs is not None
    assert scope.training_inputs[0, 0].tolist() == [
        3,
        6,
        9,
        12,
        15,
        18,
        2_000_000,
        3_000_000,
        4_000,
        5_000,
    ]
    assert scope.training_eligible
    assert scope.evaluation_eligible


def test_non_triggering_scope_can_train_without_reference_validation():
    dataset = TrainingDatasetBuilder(FittingSettings()).build(
        snapshot(second_scope_observations=60),
        manifest(),
    )

    secondary = dataset.scopes[1]
    assert secondary.training_eligible
    assert not secondary.evaluation_eligible
    assert secondary.exclusion_reason == "insufficient_reference_validation_data"
    assert secondary in dataset.training_scopes
    assert secondary not in dataset.evaluation_scopes


def test_window_first_split_keeps_validation_and_training_observations_disjoint():
    observations = tuple(
        np.full(len(FEATURE_ORDER), index, dtype=float) for index in range(62)
    )

    scope = TrainingDatasetBuilder(FittingSettings())._scope_dataset(
        "scope-a",
        observations,
        seq_length=30,
        out_seq_len=1,
        output_indices=(1, 2),
        purge=30,
        observation_timestamps=tuple(
            datetime(2026, 7, 25, tzinfo=UTC) + timedelta(seconds=index)
            for index in range(len(observations))
        ),
    )

    assert scope.validation_sample_count == 1
    assert scope.training_sample_count == 1
    assert scope.validation_inputs is not None
    assert scope.validation_targets is not None
    assert scope.training_inputs is not None
    assert scope.training_targets is not None
    assert scope.training_observations is not None
    assert scope.validation_inputs[0, :, 0].tolist() == list(range(30))
    assert scope.validation_targets[0].tolist() == [30, 30]
    assert scope.training_inputs[0, :, 0].tolist() == list(range(31, 61))
    assert scope.training_targets[0].tolist() == [61, 61]
    assert scope.training_observations[0, 0] == 31


def test_builder_uses_only_observations_inside_requested_interval():
    value = snapshot()
    bounded = replace(
        value,
        time_window=TimeWindow(
            startTime=value.time_window.start_time + timedelta(seconds=100),
            stopTime=value.time_window.start_time + timedelta(seconds=199),
        ),
    )

    dataset = TrainingDatasetBuilder(FittingSettings()).build(bounded, manifest())

    assert {scope.observation_count for scope in dataset.scopes} == {100}
    assert all(
        bounded.time_window.start_time <= timestamp <= bounded.time_window.stop_time
        for scope in dataset.scopes
        for timestamp in scope.observation_timestamps
    )


def test_triggering_scope_requires_training_and_reference_validation():
    value = snapshot()
    insufficient = DatasetSnapshot(
        **{
            **value.__dict__,
            "records": tuple(
                record for record in value.records if record.scope_keys == ("scope-b",)
            ),
        }
    )

    with pytest.raises(TrainingDataError, match="triggering scope"):
        TrainingDatasetBuilder(FittingSettings()).build(insufficient, manifest())


def test_dataset_evidence_reports_counts_independent_of_scope_order_and_endianness():
    dataset = TrainingDatasetBuilder(FittingSettings()).build(snapshot(), manifest())
    reordered = replace(dataset, scopes=tuple(reversed(dataset.scopes)))
    big_endian = replace(
        dataset,
        scopes=tuple(
            replace(
                scope,
                observations=scope.observations.astype(">f8"),
                training_inputs=(
                    None
                    if scope.training_inputs is None
                    else scope.training_inputs.astype(">f8")
                ),
                training_targets=(
                    None
                    if scope.training_targets is None
                    else scope.training_targets.astype(">f8")
                ),
                validation_inputs=(
                    None
                    if scope.validation_inputs is None
                    else scope.validation_inputs.astype(">f8")
                ),
                validation_targets=(
                    None
                    if scope.validation_targets is None
                    else scope.validation_targets.astype(">f8")
                ),
            )
            for scope in dataset.scopes
        ),
    )

    expected = dataset_evidence(dataset)

    assert dataset_evidence(reordered) == expected
    assert dataset_evidence(big_endian) == expected


def test_dataset_evidence_tracks_explicit_observation_and_split_counts():
    dataset = TrainingDatasetBuilder(FittingSettings()).build(snapshot(), manifest())
    changed_split = TrainingDatasetBuilder(
        FittingSettings(validation_ratio=0.2)
    ).build(snapshot(), manifest())

    assert dataset_evidence(changed_split).observation_count == 800
    assert dataset_evidence(changed_split).validation_sample_count != dataset_evidence(
        dataset
    ).validation_sample_count


def test_dataset_evidence_contains_only_counts():
    evidence = dataset_evidence(
        TrainingDatasetBuilder(FittingSettings()).build(snapshot(), manifest())
    )
    payload = evidence.as_dict()

    assert set(payload) == {
        "observation_count",
        "training_sample_count",
        "validation_sample_count",
    }
