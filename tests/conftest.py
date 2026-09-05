import io
import json
import tarfile
from pathlib import Path

import pytest

from py_mtlf.config import (
    ArtifactSettings,
    FederatedLearningSettings,
    ModelStateSettings,
    PublicationSettings,
    Settings,
    StorageSettings,
)
from py_mtlf.core.training_scope import TrainingScopeDescriptor


def training_scope_descriptor(name: str = "scope-a") -> TrainingScopeDescriptor:
    return TrainingScopeDescriptor(
        event_subscription={"mLEvent": "UE_COMMUNICATION", "scopeName": name},
        target_reporting_ue={"intGroupIds": ["group-G"]},
        requested_time_windows=[
            {
                "startTime": "2026-07-01T00:00:00Z",
                "stopTime": "2026-07-27T00:00:00Z",
            }
        ],
    )


def build_bundle(
    path: Path,
    *,
    extra_entries: list[tuple[tarfile.TarInfo, bytes]] | None = None,
    mutate_manifest: dict[str, object] | None = None,
    manifest_value: object | None = None,
) -> bytes:
    components = {
        "model.py": b"class Model:\n    pass\n",
        "model.npy": b"model-weights",
        "scaler.pkl": b"trusted-local-scaler",
    }
    config = {
        "model_identity": {"model_unique_id": 1},
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "created_at": "2026-07-17T00:00:00Z",
        "producer": {"name": "py_mtlf", "version": "0.1.0"},
        "runtime_compatibility": {"python": ">=3.12", "framework": "torch"},
        "MODEL_SCRIPT": "model.py",
        "MODEL_PATH": "model.npy",
        "SCALER_PATH": "scaler.pkl",
        "model": {"input_size": 10, "output_size": 2, "num_channels": [1]},
        "inference": {
            "seq_length": 30,
            "feature_order": ["total_vol"],
            "feature_units": {"total_vol": "bytes"},
            "output_fields": ["ul_vol", "dl_vol"],
        },
    }
    if mutate_manifest:
        config.update(mutate_manifest)
    manifest = config if manifest_value is None else manifest_value
    files = {"config.json": json.dumps(manifest, sort_keys=True).encode(), **components}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        for info, content in extra_entries or []:
            archive.addfile(info, io.BytesIO(content))
    data = output.getvalue()
    path.write_bytes(data)
    return data


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        storage=StorageSettings(
            artifact_root=tmp_path / "artifacts",
        ),
        model_state=ModelStateSettings(directory=tmp_path / "model-state"),
        publication=PublicationSettings(directory=tmp_path / "publications"),
        artifact=ArtifactSettings(public_base_url="http://127.0.0.1:9092"),
        federated_learning=FederatedLearningSettings(
            workspace_root=tmp_path / "fl-workspaces",
        ),
    )


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.tar.gz"
    build_bundle(path)
    return path
