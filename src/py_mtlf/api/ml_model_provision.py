from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response

from py_mtlf.api.problems import problem_response
from py_mtlf.core.provision_store import ProvisionResource
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.wire.ml_model import MLModelProvisionSubscription

router = APIRouter(
    prefix="/internal/v1/ml-model-provision",
    tags=["ml-model-provision"],
)


def _representation(
    resource: ProvisionResource,
    *,
    include_immediate_report: bool,
) -> MLModelProvisionSubscription:
    update: dict[str, object] = {}
    if include_immediate_report:
        notifications = SeedCatalog.notifications(
            resource.seeds,
            resource.representation.notification_correlation_id,
        )
        if notifications:
            update["ml_event_notifications"] = notifications
    else:
        update["ml_event_notifications"] = None
    return resource.representation.model_copy(update=update, deep=True)


def _json_response(
    representation: MLModelProvisionSubscription,
    status_code: int,
    *,
    location: str = "",
) -> JSONResponse:
    headers = {"Location": location} if location else None
    return JSONResponse(
        status_code=status_code,
        content=representation.model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
        ),
        headers=headers,
    )


@router.post("/subscriptions", status_code=status.HTTP_201_CREATED)
def create_ml_model_provision_subscription(
    payload: MLModelProvisionSubscription,
    request: Request,
) -> JSONResponse:
    resource = request.app.state.provision_store.create(payload)
    immediate = bool(
        payload.event_request is not None
        and payload.event_request.immediate_report
    )
    response = _representation(resource, include_immediate_report=immediate)
    if not immediate:
        request.app.state.provision_notifications.enqueue(resource)
    location = str(
        request.url_for(
            "replace_ml_model_provision_subscription",
            subscription_id=resource.subscription_id,
        )
    )
    return _json_response(response, status.HTTP_201_CREATED, location=location)


@router.put("/subscriptions/{subscription_id}")
def replace_ml_model_provision_subscription(
    subscription_id: str,
    payload: MLModelProvisionSubscription,
    request: Request,
) -> Response:
    resource = request.app.state.provision_store.replace(subscription_id, payload)
    if resource is None:
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            f"ML Model Provision subscription {subscription_id} was not found",
            cause="RESOURCE_NOT_FOUND",
        )
    immediate = bool(
        payload.event_request is not None
        and payload.event_request.immediate_report
    )
    response = _representation(resource, include_immediate_report=immediate)
    if not immediate:
        request.app.state.provision_notifications.enqueue(resource)
    return _json_response(response, status.HTTP_200_OK)


@router.delete(
    "/subscriptions/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_ml_model_provision_subscription(
    subscription_id: str,
    request: Request,
) -> Response:
    if not request.app.state.provision_store.delete(subscription_id):
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            f"ML Model Provision subscription {subscription_id} was not found",
            cause="RESOURCE_NOT_FOUND",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
