import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response

from py_mtlf.api.problems import problem_response
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorNotification,
    MLModelMonitorRegistration,
)

router = APIRouter(
    prefix="/internal/v1/ml-model-monitor",
    tags=["ml-model-monitor"],
)
logger = logging.getLogger(__name__)


@router.post("/registrations", status_code=status.HTTP_201_CREATED)
def create_monitor_registration(
    payload: MLModelMonitorRegistration,
    request: Request,
) -> Response:
    if payload.consumer_set_id:
        return problem_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Service Unavailable",
            "consumerSetId resolution is not supported by this MTLF backend",
            cause="SERVICE_NOT_AVAILABLE",
        )
    resource = request.app.state.monitor_registrations.create(payload)
    request.app.state.model_catalog.observe_external_model_ids({payload.model_id})
    request.app.state.accuracy_policy.record_registration(payload)
    request.app.state.monitor_reconciler.refresh()
    location = str(
        request.url_for(
            "delete_monitor_registration",
            registration_id=resource.registration_id,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=resource.representation.model_dump(
            by_alias=True,
            exclude_none=True,
            exclude_defaults=True,
            mode="json",
        ),
        headers={"Location": location},
    )


@router.delete(
    "/registrations/{registration_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_monitor_registration(
    registration_id: str,
    request: Request,
) -> Response:
    resource = request.app.state.monitor_registrations.get(registration_id)
    if resource is None or not request.app.state.monitor_registrations.delete(registration_id):
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            f"ML Model Monitor registration {registration_id} was not found",
            cause="RESOURCE_NOT_FOUND",
        )
    request.app.state.accuracy_policy.remove_registration(resource.representation)
    request.app.state.monitor_reconciler.refresh()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/notifications",
    status_code=status.HTTP_204_NO_CONTENT,
)
def receive_monitor_notification(
    payload: MLModelMonitorNotification,
    request: Request,
) -> Response:
    with request.app.state.state_lock:
        subscription = request.app.state.monitor_subscriptions.find_by_correlation(
            payload.notification_id
        )
        if subscription is None:
            return problem_response(
                status.HTTP_404_NOT_FOUND,
                "Not Found",
                "notifCorrId does not identify an active monitor subscription",
                cause="RESOURCE_NOT_FOUND",
            )
        registration = request.app.state.monitor_registrations.get(
            subscription.owner_registration_id
        )
        if registration is None or not request.app.state.monitor_reconciler.owns(
            registration.registration_id,
            subscription.subscription_id,
        ):
            return problem_response(
                status.HTTP_404_NOT_FOUND,
                "Not Found",
                "monitor subscription has no active registration owner",
                cause="RESOURCE_NOT_FOUND",
            )
        decisions = request.app.state.accuracy_policy.observe(
            subscription.representation,
            payload,
            registration.representation,
        )
        logger.info(
            "ML Model accuracy report processed correlation_id=%s model_ids=%s "
            "evaluated=%s triggered=%s",
            payload.notification_id,
            [info.model_id for info in payload.model_accuracy_info],
            [decision.evaluated for decision in decisions],
            [decision.triggered for decision in decisions],
        )
    _dispatch_retrain_intents(request.app.state, decisions)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _dispatch_retrain_intents(state, decisions) -> None:
    if not any(decision.triggered for decision in decisions):
        return
    if state.runtime.mode == "local":
        state.dataset_coordinator.accept_policy_intents()
    elif state.runtime.mode == "fl_server":
        state.fl_server.accept_policy_intents()
