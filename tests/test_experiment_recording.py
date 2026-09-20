import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from py_mtlf.config import ExperimentRecordingSettings
from py_mtlf.core.experiment_recording import ExperimentRecorder

PLAN_ID = "550e8400-e29b-41d4-a716-446655440000"
NF_INSTANCE_ID = "10000000-0000-4000-8000-000000000001"


def _recorder(tmp_path, *, validation=False):
    payload = {"directory": tmp_path / "records"}
    if validation:
        validation_path = tmp_path / "validation.npz"
        np.savez(
            validation_path,
            images=np.zeros((2, 1, 28, 28), dtype=np.uint8),
            labels=np.asarray([0, 1], dtype=np.int64),
        )
        payload["validation"] = {
            "dataset": "mnist",
            "path": validation_path,
            "batch_size": 2,
        }
    recorder = ExperimentRecorder(ExperimentRecordingSettings.model_validate(payload))
    recorder.open(NF_INSTANCE_ID)
    return recorder


def _records(tmp_path):
    path = tmp_path / "records" / PLAN_ID / "observations.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_appends_reopens_and_keeps_each_line_valid_json(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_root_round_outcome(
        ml_correlation_id=PLAN_ID,
        round_indicator=0,
        accepted=False,
        selected_nf_instance_ids=(NF_INSTANCE_ID,),
        successful_nf_instance_ids=(),
        failed_nf_instance_ids=(NF_INSTANCE_ID,),
    )
    recorder.close()

    recorder = _recorder(tmp_path)
    recorder.record_decision(
        ml_correlation_id=PLAN_ID,
        record_type="EDGE_UNAVAILABLE",
        roundInd=0,
        childNfInstanceId=NF_INSTANCE_ID,
    )
    recorder.close()

    assert [record["recordType"] for record in _records(tmp_path)] == [
        "ROUND_AGGREGATION",
        "EDGE_UNAVAILABLE",
    ]


def test_start_procedure_creates_an_empty_append_target(tmp_path):
    recorder = _recorder(tmp_path)

    recorder.start_procedure(PLAN_ID)
    recorder.close()

    path = tmp_path / "records" / PLAN_ID / "observations.jsonl"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == ""


def test_recorder_persists_final_model_atomically_and_rejects_conflicts(tmp_path):
    recorder = _recorder(tmp_path)
    source = tmp_path / "round-global.tar.gz"
    source.write_bytes(b"final-model-bundle")

    saved = recorder.save_final_model(
        ml_correlation_id=PLAN_ID,
        round_indicator=3,
        artifact_path=source,
        artifact_digest="a" * 64,
    )

    assert saved == tmp_path / "records" / PLAN_ID / "final-model.tar.gz"
    assert saved.read_bytes() == source.read_bytes()
    records = _records(tmp_path)
    assert records == [
        {
            "recordedAt": records[0]["recordedAt"],
            "recordType": "MODEL_ARTIFACT_SAVED",
            "mlCorreId": PLAN_ID,
            "nfInstanceId": NF_INSTANCE_ID,
            "roundInd": 3,
            "artifactFile": "final-model.tar.gz",
        }
    ]

    assert (
        recorder.save_final_model(
            ml_correlation_id=PLAN_ID,
            round_indicator=3,
            artifact_path=source,
            artifact_digest="a" * 64,
        )
        == saved
    )
    assert len(_records(tmp_path)) == 1
    source.unlink()
    assert saved.read_bytes() == b"final-model-bundle"

    conflict = tmp_path / "conflict.tar.gz"
    conflict.write_bytes(b"different-model")
    with pytest.raises(RuntimeError, match="conflicts"):
        recorder.save_final_model(
            ml_correlation_id=PLAN_ID,
            round_indicator=4,
            artifact_path=conflict,
            artifact_digest="b" * 64,
        )
    recorder.close()


def test_recorder_defers_nf_identity_lookup_until_the_first_record(tmp_path):
    requested = []
    recorder = ExperimentRecorder(
        ExperimentRecordingSettings(directory=tmp_path / "records"),
        nf_instance_id_provider=lambda: requested.append(True) or NF_INSTANCE_ID,
    )

    recorder.open()
    assert requested == []
    recorder.start_procedure(PLAN_ID)
    recorder.record_root_round_outcome(
        ml_correlation_id=PLAN_ID,
        round_indicator=0,
        accepted=True,
        selected_nf_instance_ids=(NF_INSTANCE_ID,),
        successful_nf_instance_ids=(NF_INSTANCE_ID,),
        failed_nf_instance_ids=(),
    )
    recorder.close()

    assert requested == [True]
    assert _records(tmp_path)[0]["nfInstanceId"] == NF_INSTANCE_ID


def test_recorder_rejects_writes_before_open_and_duplicate_open(tmp_path):
    recorder = ExperimentRecorder(
        ExperimentRecordingSettings(directory=tmp_path / "records"),
        nf_instance_id_provider=lambda: NF_INSTANCE_ID,
    )

    with pytest.raises(RuntimeError, match="not open"):
        recorder.start_procedure(PLAN_ID)
    recorder.open()
    with pytest.raises(RuntimeError, match="already open"):
        recorder.open()
    recorder.close()


def test_recorder_serializes_concurrent_writers_without_corrupting_lines(tmp_path):
    recorder = _recorder(tmp_path)

    def write(round_indicator):
        recorder.record_root_round_outcome(
            ml_correlation_id=PLAN_ID,
            round_indicator=round_indicator,
            accepted=True,
            selected_nf_instance_ids=(NF_INSTANCE_ID,),
            successful_nf_instance_ids=(NF_INSTANCE_ID,),
            failed_nf_instance_ids=(),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(40)))
    recorder.close()

    records = _records(tmp_path)
    assert len(records) == 40
    assert {record["roundInd"] for record in records} == set(range(40))


def test_training_operation_keeps_wire_topology_but_not_callback_uri_or_model_payload(tmp_path):
    recorder = _recorder(tmp_path)
    started = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    child = "10000000-0000-4000-8000-000000000002"
    subscription_id = "22222222-2222-4222-8222-222222222222"
    recorder.record_training_operation(
        ml_correlation_id=PLAN_ID,
        operation="CREATE",
        direction="SENT",
        started_at=started,
        outcome="SUCCESS",
        target_nf_instance_id=child,
        subscription_id=subscription_id,
        recorded_at=started + timedelta(seconds=2),
        message={
            "notifUri": "http://callback.example/private",
            "notifCorreId": "root-to-branch",
            "x-flTopology": {
                "nfInstanceId": child,
                "children": [{"nfInstanceId": NF_INSTANCE_ID, "priority": 50}],
            },
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://adrf.example/model"},
                    "mlFile": {"secret": "not-for-recording"},
                }
            ],
        },
    )
    recorder.close()

    record = _records(tmp_path)[0]
    assert record["startedAt"] == "2026-09-21T10:00:00Z"
    assert record["recordedAt"] >= record["startedAt"]
    assert record["subscriptionId"] == subscription_id
    assert record["targetNfInstanceId"] == child
    assert record["message"]["x-flTopology"]["children"][0]["priority"] == 50
    assert "notifUri" not in record["message"]
    assert "mlFile" not in record["message"]["mLModelInfos"][0]


def test_training_operation_rejected_create_does_not_invent_subscription_id(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_training_operation(
        ml_correlation_id=PLAN_ID,
        operation="CREATE",
        direction="RECEIVED",
        started_at=datetime.now(UTC),
        outcome="REJECTED",
        cause="ML_MODEL_TRAINING_REQS_NOT_MET",
        message={"notifCorreId": "root-to-branch"},
    )
    recorder.close()
    record = _records(tmp_path)[0]
    assert record["outcome"] == "REJECTED"
    assert record["cause"] == "ML_MODEL_TRAINING_REQS_NOT_MET"
    assert "subscriptionId" not in record


def test_recorder_write_failure_propagates_and_preserves_prior_evidence(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    recorder.record_decision(
        ml_correlation_id=PLAN_ID,
        record_type="TOPOLOGY_ACCEPTANCE",
        accepted=True,
        realizedTopology={"nfInstanceId": NF_INSTANCE_ID, "children": []},
    )
    original_open = type(tmp_path).open

    def fail_observation_open(path, *args, **kwargs):
        if path.name == "observations.jsonl" and args and args[0] == "a":
            raise OSError("recording disk unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", fail_observation_open)
    with pytest.raises(OSError, match="recording disk unavailable"):
        recorder.record_decision(
            ml_correlation_id=PLAN_ID,
            record_type="EDGE_CONFIRMED",
            childNfInstanceId=NF_INSTANCE_ID,
            subscriptionId="22222222-2222-4222-8222-222222222222",
        )
    monkeypatch.setattr(type(tmp_path), "open", original_open)
    assert [record["recordType"] for record in _records(tmp_path)] == ["TOPOLOGY_ACCEPTANCE"]
    recorder.close()


def test_recorder_evaluates_real_model_and_rejects_dataset_mismatch(tmp_path):
    recorder = _recorder(tmp_path, validation=True)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))
    manifest = {
        "dataset": "mnist",
        "model": {"input_channels": 1, "num_classes": 10},
        "inference": {
            "input_shape": [1, 28, 28],
            "class_count": 10,
            "normalization": "uint8_to_float32_div_255",
        },
    }
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    recorder.record_model_evaluation(
        ml_correlation_id=PLAN_ID,
        evaluation_stage="ROOT_INITIAL",
        model=model,
        manifest=manifest,
    )

    record = _records(tmp_path)[0]
    assert record["sampleCount"] == 2
    assert np.isfinite(record["loss"])
    assert 0 <= record["accuracy"] <= 1
    assert record["dataSplit"] == "VALIDATION"
    assert "roundInd" not in record
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )
    assert model.training is True
    with pytest.raises(ValueError, match="does not match"):
        recorder.record_model_evaluation(
            ml_correlation_id=PLAN_ID,
            evaluation_stage="ROOT_GLOBAL",
            round_indicator=0,
            model=model,
            manifest={
                "dataset": "cifar10",
                "model": {"input_channels": 3, "num_classes": 10},
                "inference": {
                    "input_shape": [3, 32, 32],
                    "class_count": 10,
                    "normalization": "uint8_to_float32_div_255",
                },
            },
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        recorder.record_model_evaluation(
            ml_correlation_id=PLAN_ID,
            evaluation_stage="ROOT_GLOBAL",
            round_indicator=True,
            model=model,
            manifest=manifest,
        )
    recorder.close()


def test_recorder_startup_rejects_missing_validation_dataset(tmp_path):
    settings = ExperimentRecordingSettings.model_validate(
        {
            "directory": tmp_path / "records",
            "validation": {
                "dataset": "mnist",
                "path": tmp_path / "missing.npz",
            },
        }
    )

    with pytest.raises(RuntimeError, match="not found"):
        ExperimentRecorder(settings).open(NF_INSTANCE_ID)
