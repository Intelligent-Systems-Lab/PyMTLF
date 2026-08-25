from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from py_mtlf.api.problems import problem_response
from py_mtlf.core.training_data_collection import CollectionManagerError
from py_mtlf.wire.training_data_collection import (
    TrainingDataCollectionRequest,
    TrainingDataCollectionStatus,
)

router = APIRouter(
    prefix="/internal/v1/training-data-collections",
    tags=["training-data-collection"],
)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_collection(
    payload: TrainingDataCollectionRequest,
    request: Request,
) -> JSONResponse:
    try:
        snapshot = request.app.state.training_data_collection_manager.create(payload)
    except CollectionManagerError as error:
        return _problem(error)
    location = f"/internal/v1/training-data-collections/{payload.request_id}"
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Location": location},
        content=_content(snapshot),
    )


@router.get("/{request_id}")
def get_collection(request_id: str, request: Request) -> JSONResponse:
    try:
        snapshot = request.app.state.training_data_collection_manager.get(request_id)
    except CollectionManagerError as error:
        return _problem(error)
    return JSONResponse(status_code=status.HTTP_200_OK, content=_content(snapshot))


@router.delete("/{request_id}", status_code=status.HTTP_202_ACCEPTED)
def delete_collection(request_id: str, request: Request) -> JSONResponse:
    try:
        snapshot = request.app.state.training_data_collection_manager.delete(request_id)
    except CollectionManagerError as error:
        return _problem(error)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=_content(snapshot))


def _content(snapshot: TrainingDataCollectionStatus) -> dict:
    return snapshot.model_dump(by_alias=True, exclude_none=True, mode="json")


def _problem(error: CollectionManagerError) -> JSONResponse:
    titles = {
        404: "Not Found",
        409: "Conflict",
        503: "Service Unavailable",
    }
    return problem_response(
        error.status_code,
        titles.get(error.status_code, "Bad Request"),
        error.detail,
        cause=error.cause,
    )
