from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response

from py_mtlf.api.problems import problem_response
from py_mtlf.core.fl_root import (
    RootCoordinatorUnavailableError,
    RootModelFamilyNotFoundError,
    RootRequestConflictError,
    RootRequestSnapshot,
)
from py_mtlf.wire.hierarchical_fl import (
    HierarchicalTrainingRequest,
    HierarchicalTrainingStatus,
)

router = APIRouter(
    prefix="/internal/v1/hierarchical-fl",
    tags=["hierarchical-fl"],
)


@router.post("/training-requests", status_code=status.HTTP_202_ACCEPTED)
def create_hierarchical_training_request(
    payload: HierarchicalTrainingRequest,
    request: Request,
) -> Response:
    try:
        snapshot = request.app.state.fl_root.submit_manual(
            request_id=payload.request_id,
            model_family_id=payload.model_family_id,
        )
    except RootRequestConflictError as error:
        return problem_response(
            status.HTTP_409_CONFLICT,
            "Conflict",
            str(error),
            cause="CONFLICTING_REQUEST",
        )
    except RootModelFamilyNotFoundError as error:
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            str(error),
            cause="RESOURCE_NOT_FOUND",
        )
    except RootCoordinatorUnavailableError as error:
        return problem_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Service Unavailable",
            str(error),
            cause="SERVICE_NOT_AVAILABLE",
        )
    location = str(
        request.url_for(
            "get_hierarchical_training_request",
            request_id=snapshot.request_id,
        )
    )
    return _status_response(snapshot, status.HTTP_202_ACCEPTED, location=location)


@router.get("/training-requests/{request_id}")
def get_hierarchical_training_request(request_id: str, request: Request) -> Response:
    snapshot = request.app.state.fl_root.get(request_id)
    if snapshot is None:
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            f"hierarchical training request {request_id} was not found",
            cause="RESOURCE_NOT_FOUND",
        )
    return _status_response(snapshot, status.HTTP_200_OK)


def _status_response(
    snapshot: RootRequestSnapshot,
    status_code: int,
    *,
    location: str = "",
) -> JSONResponse:
    value = HierarchicalTrainingStatus(
        requestId=snapshot.request_id,
        planId=snapshot.plan_id,
        modelFamilyId=snapshot.model_family_id,
        state=snapshot.state,
        currentRound=snapshot.current_round,
        completedRounds=snapshot.completed_rounds or None,
        candidateUrl=snapshot.candidate_url or None,
        candidateDigest=snapshot.candidate_digest or None,
        failureCause=snapshot.failure_cause or None,
        failureDetail=snapshot.failure_detail or None,
    )
    headers = {"Location": location} if location else None
    return JSONResponse(
        status_code=status_code,
        content=value.model_dump(by_alias=True, exclude_none=True, mode="json"),
        headers=headers,
    )
