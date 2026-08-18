from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from py_mtlf.core.fl_root import RootRequestState


class HierarchyPrivateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class HierarchicalTrainingRequest(HierarchyPrivateModel):
    request_id: str = Field(alias="requestId")
    model_family_id: str = Field(min_length=1, alias="modelFamilyId")

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("requestId must be a canonical UUIDv4") from error
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("requestId must be a canonical UUIDv4")
        return value

    @field_validator("model_family_id")
    @classmethod
    def validate_model_family_id(cls, value: str) -> str:
        if value.strip() != value or not value:
            raise ValueError("modelFamilyId must be a canonical non-empty value")
        return value


class HierarchicalTrainingStatus(HierarchyPrivateModel):
    request_id: str = Field(alias="requestId")
    plan_id: str = Field(alias="planId")
    model_family_id: str = Field(alias="modelFamilyId")
    state: RootRequestState
    failure_cause: str | None = Field(default=None, alias="failureCause")
    failure_detail: str | None = Field(default=None, alias="failureDetail")
