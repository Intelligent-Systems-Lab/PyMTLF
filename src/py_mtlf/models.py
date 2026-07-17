import re
from datetime import datetime
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentity(DomainModel):
    provider_id: str = Field(min_length=1)
    model_unique_id: int = Field(ge=0, le=9223372036854775807)

    @field_validator("provider_id")
    @classmethod
    def provider_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("provider_id must not be blank")
        return value


class ArtifactRecord(DomainModel):
    url: str
    digest: str
    size_bytes: int = Field(gt=0)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("artifact URL must be absolute HTTP(S)")
        return value

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("digest must be lowercase SHA-256 hex")
        return value


class ApplyStatus(StrEnum):
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    STALE = "STALE"
    NO_MATCH = "NO_MATCH"
    CONFLICT = "CONFLICT"


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


class ApplyEvidence(DomainModel):
    event_id: str = Field(min_length=1)
    model_identity: ModelIdentity
    target_generation: int = Field(gt=0)
    status: ApplyStatus
    active_generation: int | None = Field(default=None, ge=0)
    active_artifact_digest: str | None = None
    affected_runtime_count: int = Field(ge=0)
    failure_code: str = ""
    failure_detail: str = ""
    completed_at: datetime

    _completed_at_timezone = field_validator("completed_at")(_require_timezone)

    @field_validator("active_artifact_digest")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("active_artifact_digest must be lowercase SHA-256 hex")
        return value


class ObservedModelStatus(StrEnum):
    CONSISTENT = "CONSISTENT"
    NO_MATCH = "NO_MATCH"
    DIVERGED = "DIVERGED"
    UNAVAILABLE = "UNAVAILABLE"


class ObservedModelState(DomainModel):
    model_identity: ModelIdentity
    status: ObservedModelStatus
    active_generation: int | None = Field(default=None, ge=0)
    active_artifact_digest: str | None = None
    matching_runtime_count: int = Field(ge=0)
    observed_at: datetime
    divergence_summary: str = ""

    _observed_at_timezone = field_validator("observed_at")(_require_timezone)

    @field_validator("active_artifact_digest")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("active_artifact_digest must be lowercase SHA-256 hex")
        return value


class PrivateError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    correlation_id: str = ""
