import hashlib
import tarfile

import pytest
from conftest import build_bundle

from py_mtlf.config import ArtifactSettings, FederatedLearningSettings
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    model_contract_digest,
    preprocessing_contract_digest,
)


def workspace(tmp_path) -> FLWorkspace:
    return FLWorkspace(
        FederatedLearningSettings(workspace_root=tmp_path / "fl-workspaces"),
        ArtifactSettings(),
    )


def test_download_validation_rejects_component_digest_mismatch(tmp_path):
    path = tmp_path / "bad-digest.tar.gz"
    build_bundle(path, mutate_manifest={"file_digests": {}})

    with pytest.raises(RuntimeError, match="digest inventory"):
        workspace(tmp_path)._validate_archive(path)


def test_download_validation_rejects_unexpected_bundle_file(tmp_path):
    path = tmp_path / "extra-file.tar.gz"
    info = tarfile.TarInfo("notes.txt")
    info.size = 5
    build_bundle(path, extra_entries=[(info, b"notes")])

    with pytest.raises(RuntimeError, match="file set"):
        workspace(tmp_path)._validate_archive(path)


def test_download_validation_rejects_invalid_artifact_role_contract(tmp_path):
    path = tmp_path / "invalid-role.tar.gz"
    build_bundle(path, mutate_manifest={"artifact_role": "ROUND_LOCAL"})

    with pytest.raises(RuntimeError, match="role contract"):
        workspace(tmp_path)._validate_archive(path)


def test_contract_digests_cover_executable_model_and_scaler_components():
    base = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 10},
        "inference": {"seq_length": 30},
        "file_digests": {
            "model.py": hashlib.sha256(b"model-a").hexdigest(),
            "model.npy": hashlib.sha256(b"weights").hexdigest(),
            "scaler.pkl": hashlib.sha256(b"scaler-a").hexdigest(),
        },
    }
    changed_model = {
        **base,
        "file_digests": {
            **base["file_digests"],
            "model.py": hashlib.sha256(b"model-b").hexdigest(),
        },
    }
    changed_scaler = {
        **base,
        "file_digests": {
            **base["file_digests"],
            "scaler.pkl": hashlib.sha256(b"scaler-b").hexdigest(),
        },
    }

    assert model_contract_digest(base) != model_contract_digest(changed_model)
    assert preprocessing_contract_digest(base) == preprocessing_contract_digest(changed_model)
    assert preprocessing_contract_digest(base) != preprocessing_contract_digest(changed_scaler)
