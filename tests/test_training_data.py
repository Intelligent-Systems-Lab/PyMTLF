from datetime import UTC, datetime, timedelta

import pytest

from py_mtlf.config import TrainingSettings
from py_mtlf.core.dataset import DatasetRecord, DatasetSnapshot
from py_mtlf.core.training_data import FEATURE_ORDER, TrainingDataError, TrainingDatasetBuilder
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
        family_key=("local-mtlf", "ue-communication-default"),
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
    dataset = TrainingDatasetBuilder(TrainingSettings()).build(snapshot(), manifest())

    scope = dataset.scopes[0]
    assert scope.observation_count == 400
    assert scope.validation_sample_count == 20
    assert scope.training_sample_count == 290
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
    dataset = TrainingDatasetBuilder(TrainingSettings()).build(
        snapshot(second_scope_observations=100),
        manifest(),
    )

    secondary = dataset.scopes[1]
    assert secondary.training_eligible
    assert not secondary.evaluation_eligible
    assert secondary.exclusion_reason == "insufficient_reference_validation_data"
    assert secondary in dataset.training_scopes
    assert secondary not in dataset.evaluation_scopes


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
        TrainingDatasetBuilder(TrainingSettings()).build(insufficient, manifest())
