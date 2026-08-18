from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.config import (
    AdrfSettings,
    ArtifactSettings,
    FederatedLearningSettings,
    FittingSettings,
    FLClientSettings,
    FLServerSettings,
    ModelProvisionSettings,
    RuntimeSettings,
    SeedModelSettings,
    Settings,
    StorageSettings,
    load_settings,
)


def test_defaults_use_confirmed_phase_one_values():
    settings = Settings()

    assert settings.server.binding_host == "127.0.0.1"
    assert settings.server.port == 9092
    assert settings.storage.artifact_root == Path("data/artifacts")
    assert settings.artifact.max_compressed_bytes == 256 * 1024 * 1024
    assert settings.artifact.max_extracted_bytes == 1024 * 1024 * 1024
    assert settings.artifact.max_single_file_bytes == 512 * 1024 * 1024
    assert settings.artifact.max_entries == 32
    assert FittingSettings().validation_ratio == 0.10
    assert FLServerSettings().preparation_data_window_seconds == 3600


def test_load_settings_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  port: 9092\n  unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(path)


@pytest.mark.parametrize(
    "value",
    [
        "file:///tmp/artifacts",
        "http://user@example.com",
        "http://example.com/path",
        "http://example.com?query=yes",
        "http://example.com:invalid",
        "http://example.com:0",
        "http://example.com:65536",
    ],
)
def test_artifact_public_base_url_requires_origin(value):
    with pytest.raises(ValidationError):
        ArtifactSettings(public_base_url=value)


def test_single_file_limit_must_not_exceed_total_limit():
    with pytest.raises(ValidationError):
        ArtifactSettings(max_single_file_bytes=11, max_extracted_bytes=10)


def test_storage_path_must_not_be_blank():
    with pytest.raises(ValidationError):
        StorageSettings(artifact_root=" ")


def test_seed_model_ids_and_artifact_keys_are_unique():
    seed = SeedModelSettings(
        family_id="ue-communication-default",
        model_id=1,
        artifact_key="a" * 64,
        event="UE_COMMUNICATION",
    )
    with pytest.raises(ValidationError, match="IDs must be unique"):
        ModelProvisionSettings(seed_models=(seed, seed))


@pytest.mark.parametrize(
    "payload",
    [
        {"storage": {"database_path": "data/old.sqlite3"}},
        {"reconciliation": {"retry_interval_seconds": 1}},
    ],
)
def test_removed_state_machine_config_is_rejected(payload):
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "other"},
        {"mode": "configured", "configured_endpoint": ""},
        {"mode": "configured", "configured_endpoint": "http://example.com/path"},
        {"mode": "configured", "configured_endpoint": "http://example.com:invalid"},
        {"retry_initial_backoff_seconds": 3, "retry_max_backoff_seconds": 2},
    ],
)
def test_adrf_settings_reject_invalid_mode_endpoint_and_backoff(payload):
    with pytest.raises(ValidationError):
        AdrfSettings.model_validate(payload)


def test_adrf_configured_endpoint_is_normalized():
    settings = AdrfSettings(
        mode="configured",
        configured_endpoint="http://adrf.example:9888/",
        configured_nf_instance_id="00000000-0000-4000-8000-000000000010",
    )

    assert settings.configured_endpoint == "http://adrf.example:9888"


@pytest.mark.parametrize("mode", ["local", "federated"])
def test_runtime_accepts_supported_modes(mode):
    assert RuntimeSettings(mode=mode.upper()).mode == mode


@pytest.mark.parametrize(
    "mode",
    ["coordinator", "fl_server", "fl_client", "root", "branch", "leaf"],
)
def test_runtime_rejects_unknown_and_role_modes(mode):
    with pytest.raises(ValidationError, match="runtime.mode"):
        RuntimeSettings(mode=mode)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("cpu", "cpu"), ("CUDA", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:12", "cuda:12")],
)
def test_training_device_is_validated_and_canonicalized(value, expected):
    assert FittingSettings(device=value).device == expected


@pytest.mark.parametrize("value", ["", "auto", "gpu", "cuda:-1", "cuda:01", "mps"])
def test_invalid_training_device_is_rejected(value):
    with pytest.raises(ValidationError, match="training.device"):
        FittingSettings(device=value)


@pytest.mark.parametrize(
    "payload",
    [
        {"workspace_root": ""},
        {"workspace_ttl_seconds": 0},
        {"public_base_url": "relative"},
        {"artifact_download": {"timeout_seconds": 0}},
        {"artifact_download": {"allowed_origins": ["http://peer.example", "http://peer.example/"]}},
    ],
)
def test_federated_learning_settings_fail_fast(payload):
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "server", "client", "local_training", "valid"),
    [
        ("local", None, None, {}, True),
        ("local", FLServerSettings(), None, None, False),
        ("local", None, FLClientSettings(), None, False),
        ("federated", None, None, None, False),
        ("federated", FLServerSettings(), None, None, True),
        ("federated", None, FLClientSettings(), None, True),
        ("federated", FLServerSettings(), FLClientSettings(), None, True),
        ("federated", FLServerSettings(), None, {}, False),
    ],
)
def test_runtime_engine_configuration_matrix(
    mode,
    server,
    client,
    local_training,
    valid,
):
    payload = {
        "runtime": {"mode": mode},
        "federated_learning": {"server": server, "client": client},
        "local_training": local_training,
    }
    if valid:
        settings = Settings.model_validate(payload)
        assert (settings.federated_learning.server is not None) is (server is not None)
        assert (settings.federated_learning.client is not None) is (client is not None)
    else:
        with pytest.raises(ValidationError):
            Settings.model_validate(payload)


def test_federated_server_requires_single_active_process():
    with pytest.raises(ValidationError, match="max_active_processes"):
        Settings(
            runtime=RuntimeSettings(mode="federated"),
            federated_learning=FederatedLearningSettings(
                server=FLServerSettings(max_active_processes=2),
            ),
        )


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        ("local.yaml", "local"),
        ("fl-server.yaml", "federated"),
        ("fl-client.yaml", "federated"),
        ("fl-server-client.yaml", "federated"),
    ],
)
def test_tracked_engine_profiles_are_valid(profile, mode):
    path = Path(__file__).parents[1] / "config" / profile

    assert load_settings(path).runtime.mode == mode


def test_removed_flat_training_and_fl_role_keys_are_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate({"training": {"epochs": 1}})
    with pytest.raises(ValidationError):
        FederatedLearningSettings.model_validate({"round_count": 2})
