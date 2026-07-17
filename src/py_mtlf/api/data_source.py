from typing import Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from py_mtlf.models import PrivateError

router = APIRouter(prefix="/internal/v1", tags=["data-source"])

DataSource = Literal["adrf", "mongodb"]
StorageMode = Literal["adrf", "mongodb", "dual"]


class DataSourceSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    available_data_sources: list[DataSource] = Field(alias="availableDataSources")

    @field_validator("available_data_sources")
    @classmethod
    def reject_duplicates(cls, value: list[DataSource]) -> list[DataSource]:
        if len(value) != len(set(value)):
            raise ValueError("availableDataSources must not contain duplicates")
        return value


class DataSourceSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    storage_mode: StorageMode = Field(alias="storageMode")


@router.post(
    "/data-source-selection",
    response_model=DataSourceSelectionResponse,
)
async def select_data_source(
    payload: DataSourceSelectionRequest,
    request: Request,
) -> DataSourceSelectionResponse | JSONResponse:
    mode: StorageMode = request.app.state.settings.data_source.storage_mode
    required_sources = {
        "adrf": {"adrf"},
        "mongodb": {"mongodb"},
        "dual": {"adrf", "mongodb"},
    }[mode]
    if not required_sources.issubset(payload.available_data_sources):
        error = PrivateError(
            code="DATA_SOURCE_REQUIREMENT_UNSATISFIED",
            message="configured storage mode requires unavailable data source",
            retryable=True,
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=error.model_dump(mode="json", exclude={"correlation_id"}),
        )
    request.app.state.runtime.selected_storage_mode = mode
    return DataSourceSelectionResponse(storage_mode=mode)
