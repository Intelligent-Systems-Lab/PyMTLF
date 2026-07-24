from fastapi import APIRouter, Request, status
from fastapi.responses import Response

from py_mtlf.api.problems import problem_response
from py_mtlf.wire.adrf import NadrfDataRetrievalNotification

router = APIRouter(
    prefix="/internal/v1/adrf-data-management",
    tags=["adrf-data-management"],
)


@router.post(
    "/retrieval-notifications",
    status_code=status.HTTP_204_NO_CONTENT,
)
def receive_retrieval_notification(
    payload: NadrfDataRetrievalNotification,
    request: Request,
) -> Response:
    try:
        request.app.state.dataset_coordinator.receive_notification(payload)
    except KeyError:
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            "notifCorrId does not identify an active retrieval subscription",
            cause="RESOURCE_NOT_FOUND",
        )
    except RuntimeError as error:
        return problem_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Service Unavailable",
            str(error),
            cause="SYSTEM_FAILURE",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
