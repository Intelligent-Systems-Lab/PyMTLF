import hashlib
import io
import json
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from py_mtlf.config import (
    ModelProvisionSettings,
    SeedModelSettings,
    TrainingSettings,
)
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.dataset import DatasetRecord, DatasetSnapshot
from py_mtlf.core.provision_store import ProvisionResourceStore
from py_mtlf.core.seed_catalog import ModelCatalog
from py_mtlf.core.training_data import FEATURE_ORDER
from py_mtlf.core.training_jobs import TrainingCoordinator, TrainingJobState
from py_mtlf.wire.adrf import TimeWindow
from py_mtlf.wire.ml_model import MLModelProvisionSubscription

MODEL_SOURCE = b"""\
import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        input_size=10,
        output_size=2,
        num_channels=None,
        kernel_size=2,
        dropout=0.0,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, value):
        return self.linear(value[:, :, -1])
"""


class SeedRuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def make_seed_bundle(path: Path) -> None:
    torch.manual_seed(7)
    model = SeedRuntimeModel()
    weights = np.empty(len(model.state_dict()), dtype=object)
    weights[:] = [value.detach().numpy() for value in model.state_dict().values()]
    weight_stream = io.BytesIO()
    np.save(weight_stream, weights, allow_pickle=True)
    scaler_stream = io.BytesIO()
    joblib.dump(
        StandardScaler().fit(np.log1p(np.arange(1000).reshape(100, 10) + 1)),
        scaler_stream,
    )
    components = {
        "model.py": MODEL_SOURCE,
        "model.npy": weight_stream.getvalue(),
        "scaler.pkl": scaler_stream.getvalue(),
    }
    manifest = {
        "bundle_schema_version": "1.0",
        "model_identity": {
            "provider_id": "local",
            "model_unique_id": 1,
        },
        "model_generation": 1,
        "analytics_event": "UE_COMMUNICATION",
        "created_at": "2026-07-25T00:00:00Z",
        "producer": {"name": "test", "version": "1"},
        "runtime_compatibility": {
            "python": ">=3.12",
            "framework": "torch",
        },
        "MODEL_SCRIPT": "model.py",
        "MODEL_PATH": "model.npy",
        "SCALER_PATH": "scaler.pkl",
        "model": {
            "input_size": 10,
            "output_size": 2,
            "num_channels": [8],
            "kernel_size": 2,
            "dropout": 0.0,
        },
        "inference": {
            "seq_length": 4,
            "out_seq_len": 1,
            "feature_order": list(FEATURE_ORDER),
            "output_fields": ["ul_vol", "dl_vol"],
            "preprocessing": "log1p_standard_scaler",
        },
        "file_digests": {
            name: hashlib.sha256(content).hexdigest() for name, content in components.items()
        },
    }
    files = {
        "config.json": json.dumps(manifest, sort_keys=True).encode(),
        **components,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def snapshot(record_count: int) -> DatasetSnapshot:
    start = datetime(2026, 7, 25, tzinfo=UTC)
    stop = start + timedelta(minutes=max(record_count - 1, 0))
    items = []
    for index in range(record_count):
        value = index + 1
        items.append(
            {
                "startTime": (start + timedelta(minutes=index)).isoformat(),
                "userDataUsageMeasurements": [
                    {
                        "volumeMeasurement": {
                            "totalVolume": 3 * value,
                            "ulVolume": value,
                            "dlVolume": 2 * value,
                            "totalNbOfPackets": 6 * value,
                            "ulNbOfPackets": 2 * value,
                            "dlNbOfPackets": 4 * value,
                        },
                        "throughputMeasurement": {
                            "ulThroughput": f"{value} Kbps",
                            "dlThroughput": f"{2 * value} Kbps",
                            "ulPacketThroughput": f"{value} pps",
                            "dlPacketThroughput": f"{2 * value} pps",
                        },
                    }
                ],
            }
        )
    return DatasetSnapshot(
        job_id="dataset-1",
        family_key=("local", "ue-communication-default"),
        triggering_scope_key="scope-a",
        required_scope_keys=("scope-a",),
        time_window=TimeWindow(startTime=start, stopTime=stop),
        source="mongodb",
        records=(
            DatasetRecord(
                identity="record-1",
                scope_keys=("scope-a",),
                source="mongodb",
                measurement_time=None,
                record={"dataNotif": {"upfEventNotifs": [{"notificationItems": items}]}},
            ),
        ),
        scope_record_counts={"scope-a": 1},
    )


class DatasetStub:
    def __init__(self, value: DatasetSnapshot) -> None:
        self.snapshot = value
        self.claimed = False
        self.handler = None
        self.outcome = None

    def set_ready_handler(self, handler) -> None:
        self.handler = handler

    def claim_ready(self, job_id):
        if job_id != self.snapshot.job_id or self.claimed:
            return None
        self.claimed = True
        return self.snapshot

    def jobs(self):
        return (SimpleNamespace(job_id=self.snapshot.job_id, snapshot=self.snapshot),)

    def finish_claim(self, _job_id, **outcome) -> None:
        self.outcome = outcome


class PolicyStub:
    def __init__(self) -> None:
        self.generation = None

    @staticmethod
    def active_scope_keys(_model_key):
        return ("scope-a",)

    @staticmethod
    def scope_reference(_model_key, _scope_key):
        return object()

    def begin_generation(self, family_key, previous_version, current_version, scope_keys):
        self.generation = (
            family_key,
            previous_version,
            current_version,
            scope_keys,
        )


class NotificationStub:
    def __init__(self) -> None:
        self.models = []

    def enqueue(self, resource):
        self.models.extend(key for key in resource.family_keys if key is not None)


def coordinator_subject(settings, tmp_path, record_count):
    seed_path = tmp_path / "seed.tar.gz"
    make_seed_bundle(seed_path)
    artifacts = ArtifactRepository(
        settings.storage.artifact_root,
        settings.artifact,
    )
    artifacts.open()
    seed = artifacts.publish(seed_path)
    catalog = ModelCatalog(
        ModelProvisionSettings(
            provider_namespace="local",
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=1,
                    artifact_key=seed.key,
                    event="UE_COMMUNICATION",
                ),
            ),
        ),
        artifacts,
    )
    catalog.open()
    provisions = ProvisionResourceStore(catalog)
    provisions.create(
        MLModelProvisionSubscription.model_validate(
            {
                "mLEventSubscs": [
                    {
                        "mLEvent": "UE_COMMUNICATION",
                        "mLEventFilter": {},
                    }
                ],
                "notifUri": "http://go.internal/model-update",
            }
        )
    )
    datasets = DatasetStub(snapshot(record_count))
    policy = PolicyStub()
    notifications = NotificationStub()
    coordinator = TrainingCoordinator(
        TrainingSettings(
            epochs=1,
            batch_size=16,
            enforce_performance_gate=False,
        ),
        datasets,
        catalog,
        artifacts,
        provisions,
        notifications,
        policy,
    )
    return coordinator, datasets, catalog, artifacts, notifications, policy


def test_ready_snapshot_runs_one_local_training_and_promotes_candidate(
    settings,
    tmp_path,
):
    coordinator, datasets, catalog, artifacts, notifications, policy = coordinator_subject(
        settings, tmp_path, record_count=80
    )
    coordinator.open()
    coordinator.submit("dataset-1")
    wait_until(
        lambda: (
            (job := coordinator.job_for_dataset("dataset-1")) is not None
            and job.state == TrainingJobState.COMPLETED
        )
    )
    coordinator.shutdown()

    job = coordinator.job_for_dataset("dataset-1")
    assert job is not None
    assert job.promoted_generation == 2
    assert job.candidate_artifact_key
    family_key = ("local", "ue-communication-default")
    assert catalog.current(family_key).artifact.key == job.candidate_artifact_key
    assert catalog.current(family_key).model_id == 2
    assert artifacts.manifest(job.candidate_artifact_key)["model_generation"] == 2
    assert datasets.outcome == {
        "success": True,
        "failure": "",
        "cancelled": False,
    }
    assert notifications.models == [family_key]
    assert policy.generation == (
        family_key,
        ("local", 1),
        ("local", 2),
        ("scope-a",),
    )


def test_insufficient_triggering_scope_fails_and_releases_claim(
    settings,
    tmp_path,
):
    coordinator, datasets, catalog, _artifacts, notifications, policy = coordinator_subject(
        settings, tmp_path, record_count=12
    )
    original = catalog.current(("local", "ue-communication-default"))
    coordinator.open()
    coordinator.submit("dataset-1")
    wait_until(
        lambda: (
            (job := coordinator.job_for_dataset("dataset-1")) is not None
            and job.state == TrainingJobState.FAILED
        )
    )
    coordinator.shutdown()

    job = coordinator.job_for_dataset("dataset-1")
    assert job is not None
    assert "insufficient" in job.failure
    assert catalog.current(("local", "ue-communication-default")) == original
    assert datasets.outcome["success"] is False
    assert notifications.models == []
    assert policy.generation is None
