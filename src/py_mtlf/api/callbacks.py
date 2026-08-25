from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse

from py_mtlf.api.problems import problem_response
from py_mtlf.core.training_data_collection import CollectionManagerError
from py_mtlf.wire.upf_event_exposure import NotificationData

router = APIRouter(prefix="/callbacks", tags=["event-exposure-callback"])


@router.post(
    "/upf-event-exposure",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def upf_event_exposure_notification(
    payload: NotificationData,
    request: Request,
) -> Response | JSONResponse:
    try:
        request.app.state.training_data_collection_manager.accept_callback(payload)
    except CollectionManagerError as error:
        return problem_response(
            error.status_code,
            "Event Exposure notification rejected",
            error.detail,
            cause=error.cause,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
