from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerSettings(FrozenSettings):
    binding_host: str = "127.0.0.1"
    port: int = Field(default=9092, ge=1, le=65535)

    @field_validator("binding_host")
    @classmethod
    def host_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("binding_host must not be blank")
        return value


class StorageSettings(FrozenSettings):
    database_path: Path = Path("data/mtlf-state.sqlite3")
    artifact_root: Path = Path("data/artifacts")

    @field_validator("database_path", "artifact_root", mode="before")
    @classmethod
    def path_must_not_be_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("storage path must not be blank")
        return value


def _validate_http_base_url(value: str, field_name: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain userinfo, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{field_name} must not contain a path")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{field_name} contains an invalid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field_name} port must be between 1 and 65535")
    return value


class ArtifactSettings(FrozenSettings):
    public_base_url: str = "http://127.0.0.1:9092"
    max_compressed_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_extracted_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)
    max_single_file_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    max_entries: int = Field(default=32, gt=0)

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, value: str) -> str:
        return _validate_http_base_url(value, "artifact.public_base_url")

    @model_validator(mode="after")
    def validate_limits(self) -> "ArtifactSettings":
        if self.max_single_file_bytes > self.max_extracted_bytes:
            raise ValueError("max_single_file_bytes must not exceed max_extracted_bytes")
        return self


class ReconciliationSettings(FrozenSettings):
    retry_interval_seconds: float = Field(default=1, gt=0)
    shutdown_timeout_seconds: float = Field(default=5, gt=0)


class LogSettings(FrozenSettings):
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return value


class Settings(FrozenSettings):
    server: ServerSettings = ServerSettings()
    storage: StorageSettings = StorageSettings()
    artifact: ArtifactSettings = ArtifactSettings()
    reconciliation: ReconciliationSettings = ReconciliationSettings()
    log: LogSettings = LogSettings()


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return Settings.model_validate(raw)
