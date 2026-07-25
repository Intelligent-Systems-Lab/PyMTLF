from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.config import (
    AdrfSettings,
    ArtifactSettings,
    ModelProvisionSettings,
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
    )

    assert settings.configured_endpoint == "http://adrf.example:9888"
