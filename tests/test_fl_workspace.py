import tarfile
from unittest.mock import Mock

import pytest
import torch
from conftest import build_bundle

from py_mtlf.config import ArtifactSettings, FederatedLearningSettings
from py_mtlf.core.fl_workspace import (
    FLArtifactContractError,
    FLWorkspace,
    validate_model_compatibility,
)


def workspace(tmp_path) -> FLWorkspace:
    return FLWorkspace(
        FederatedLearningSettings(workspace_root=tmp_path / "fl-workspaces"),
        ArtifactSettings(),
    )


def test_download_validation_rejects_removed_component_digest_inventory(tmp_path):
    path = tmp_path / "removed-digest.tar.gz"
    build_bundle(path, mutate_manifest={"file_digests": {"model.py": "invalid"}})

    with pytest.raises(RuntimeError, match="unsupported manifest field"):
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


def test_model_compatibility_checks_typed_contract_and_parameter_shape():
    manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 10},
        "inference": {"seq_length": 30},
    }
    base = Mock(manifest=manifest, model=torch.nn.Linear(10, 2))
    compatible = Mock(manifest=manifest.copy(), model=torch.nn.Linear(10, 2))
    incompatible = Mock(manifest=manifest.copy(), model=torch.nn.Linear(11, 2))

    validate_model_compatibility(base, compatible)
    with pytest.raises(FLArtifactContractError, match="shape"):
        validate_model_compatibility(base, incompatible)


def test_model_compatibility_compares_parameter_keys_without_ordering():
    manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2},
        "inference": {"seq_length": 1},
    }
    weight = torch.zeros((1, 2))
    bias = torch.zeros((1,))
    base_model = Mock()
    base_model.state_dict.return_value = {"weight": weight, "bias": bias}
    candidate_model = Mock()
    candidate_model.state_dict.return_value = {"bias": bias, "weight": weight}

    validate_model_compatibility(
        Mock(manifest=manifest, model=base_model),
        Mock(manifest=manifest.copy(), model=candidate_model),
    )


def test_model_compatibility_rejects_parameter_key_and_dtype_mismatch():
    manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2},
        "inference": {"seq_length": 1},
    }
    base_model = Mock()
    base_model.state_dict.return_value = {"weight": torch.zeros((1, 2))}
    wrong_key = Mock()
    wrong_key.state_dict.return_value = {"other": torch.zeros((1, 2))}
    wrong_dtype = Mock()
    wrong_dtype.state_dict.return_value = {
        "weight": torch.zeros((1, 2), dtype=torch.float64)
    }
    base = Mock(manifest=manifest, model=base_model)

    with pytest.raises(FLArtifactContractError, match="keys"):
        validate_model_compatibility(
            base,
            Mock(manifest=manifest.copy(), model=wrong_key),
        )
    with pytest.raises(FLArtifactContractError, match="dtype"):
        validate_model_compatibility(
            base,
            Mock(manifest=manifest.copy(), model=wrong_dtype),
        )
