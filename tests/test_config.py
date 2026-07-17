from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.config import ArtifactSettings, Settings, StorageSettings, load_settings


def test_defaults_use_confirmed_phase_one_values():
    settings = Settings()

    assert settings.server.binding_host == "127.0.0.1"
    assert settings.server.port == 9092
    assert settings.storage.artifact_root == Path("data/artifacts")
    assert settings.data_source.storage_mode == "mongodb"
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


@pytest.mark.parametrize("value", ["invalid", "", "MONGODB"])
def test_storage_mode_rejects_unknown_value(value):
    with pytest.raises(ValidationError):
        Settings.model_validate({"data_source": {"storage_mode": value}})


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
