import re

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


class PrivateError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    correlation_id: str = ""
