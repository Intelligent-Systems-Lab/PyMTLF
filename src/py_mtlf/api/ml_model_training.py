import inspect
import json
from datetime import UTC, datetime
from functools import wraps
from uuid import UUID

from fastapi import APIRouter, Header, Request, status
from fastapi.responses import JSONResponse, Response

from py_mtlf.api.problems import problem_response
from py_mtlf.core.fl_client import FLClientCapacityError, FLClientResource
from py_mtlf.wire.ml_model_training import (
    InvalidMessageError,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
)

router = APIRouter(
    prefix="/internal/v1/ml-model-training",
    tags=["ml-model-training"],
)


def _record_received(operation: str):
    def decorate(handler):
        signature = inspect.signature(handler)

        @wraps(handler)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            request = arguments["request"]
            recorder = getattr(request.app.state, "experiment_recorder", None)
            started_at = datetime.now(UTC)
            payload = arguments.get("payload")
            subscription_id = arguments.get("subscription_id")
            ml_correlation_id = getattr(payload, "ml_correlation_id", None)
            source_nf_instance_id = None
            if operation in {"PUT", "PATCH", "DELETE"}:
                client = getattr(request.app.state, "fl_client", None)
                try:
                    resource = client.get(subscription_id) if client is not None else None
                except KeyError:
                    resource = None
                ml_correlation_id = resource.representation.ml_correlation_id if resource else None
            elif operation == "NOTIFY":
                server = getattr(request.app.state, "fl_server", None)
                context = (
                    server.notification_record_context(payload.notification_correlation_id)
                    if server is not None
                    else None
                )
                if context is not None:
                    ml_correlation_id, subscription_id, source_nf_instance_id = context
            try:
                response = handler(*args, **kwargs)
            except Exception as error:
                if recorder is not None and ml_correlation_id:
                    recorder.record_training_operation(
                        ml_correlation_id=ml_correlation_id,
                        operation=operation,
                        direction="RECEIVED",
                        started_at=started_at,
                        outcome="FAILED",
                        subscription_id=subscription_id,
                        source_nf_instance_id=source_nf_instance_id,
                        message=(
                            payload.model_dump(
                                by_alias=True, exclude_none=True, exclude_unset=True, mode="json"
                            )
                            if payload is not None
                            else None
                        ),
                        cause=str(error),
                    )
                raise
            if recorder is not None and ml_correlation_id:
                cause = None
                if response.status_code >= 400:
                    try:
                        cause = json.loads(response.body).get("cause") if response.body else None
                    except (ValueError, AttributeError):
                        cause = None
                recorder.record_training_operation(
                    ml_correlation_id=ml_correlation_id,
                    operation=operation,
                    direction="RECEIVED",
                    started_at=started_at,
                    outcome=(
                        "SUCCESS"
                        if response.status_code < 400
                        else "REJECTED"
                        if response.status_code < 500
                        else "FAILED"
                    ),
                    subscription_id=(
                        subscription_id
                        if operation != "CREATE" or response.status_code == 201
                        else None
                    ),
                    source_nf_instance_id=source_nf_instance_id,
                    message=(
                        payload.model_dump(
                            by_alias=True, exclude_none=True, exclude_unset=True, mode="json"
                        )
                        if payload is not None
                        else None
                    ),
                    cause=cause,
                )
            return response

        return wrapped

    return decorate


def _resource_response(
    resource: FLClientResource,
    status_code: int,
    *,
    location: str = "",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=resource.representation.model_dump(by_alias=True, exclude_none=True, mode="json"),
        headers={"Location": location} if location else None,
    )


@router.post("/subscriptions", status_code=status.HTTP_201_CREATED)
@_record_received("CREATE")
def create_training_subscription(
    payload: NwdafMLModelTrainSubsc,
    request: Request,
    subscription_id: str | None = Header(default=None, alias="X-NWDAF-Subscription-Id"),
) -> Response:
    if request.app.state.fl_client is None:
        return _role_unavailable("FL Client")
    try:
        parsed_id = UUID(subscription_id or "")
        if parsed_id.version != 4 or str(parsed_id) != subscription_id:
            raise ValueError("invalid subscription ID")
    except ValueError:
        return problem_response(
            status.HTTP_400_BAD_REQUEST,
            "Bad Request",
            "X-NWDAF-Subscription-Id must be a UUIDv4",
            cause="INVALID_MSG_FORMAT",
        )
    try:
        resource = request.app.state.fl_client.create(subscription_id, payload)
    except InvalidMessageError as error:
        return _invalid_message(error)
    except RequirementsError as error:
        return _requirements_not_met(error)
    except ValueError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="ML_MODEL_TRAINING_REQS_NOT_MET",
        )
    except FLClientCapacityError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="OVERLOAD",
        )
    except RuntimeError as error:
        return problem_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Service Unavailable",
            str(error),
            cause="UNAVAILABLE_ML_MODEL_TRAINING_FOR_ALLEVENTS",
        )
    location = str(
        request.url_for(
            "replace_training_subscription",
            subscription_id=resource.subscription_id,
        )
    )
    return _resource_response(resource, status.HTTP_201_CREATED, location=location)


@router.put("/subscriptions/{subscription_id}")
@_record_received("PUT")
def replace_training_subscription(
    subscription_id: str,
    payload: NwdafMLModelTrainSubsc,
    request: Request,
) -> Response:
    if request.app.state.fl_client is None:
        return _role_unavailable("FL Client")
    try:
        resource = request.app.state.fl_client.replace(subscription_id, payload)
    except KeyError:
        return _not_found(subscription_id)
    except InvalidMessageError as error:
        return _invalid_message(error)
    except RequirementsError as error:
        return _requirements_not_met(error)
    except FLClientCapacityError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="OVERLOAD",
        )
    except RuntimeError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="ML_TRAINING_NOT_COMPLETE",
        )
    return _resource_response(resource, status.HTTP_200_OK)


@router.patch("/subscriptions/{subscription_id}")
@_record_received("PATCH")
def patch_training_subscription(
    subscription_id: str,
    payload: NwdafMLModelTrainSubscPatch,
    request: Request,
) -> Response:
    if request.app.state.fl_client is None:
        return _role_unavailable("FL Client")
    try:
        resource = request.app.state.fl_client.patch(subscription_id, payload)
    except KeyError:
        return _not_found(subscription_id)
    except InvalidMessageError as error:
        return _invalid_message(error)
    except RequirementsError as error:
        return _requirements_not_met(error)
    except FLClientCapacityError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="OVERLOAD",
        )
    except RuntimeError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="ML_TRAINING_NOT_COMPLETE",
        )
    return _resource_response(resource, status.HTTP_200_OK)


@router.delete(
    "/subscriptions/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@_record_received("DELETE")
def delete_training_subscription(subscription_id: str, request: Request) -> Response:
    if request.app.state.fl_client is None:
        return _role_unavailable("FL Client")
    try:
        request.app.state.fl_client.delete(subscription_id)
    except KeyError:
        return _not_found(subscription_id)
    except RuntimeError as error:
        return problem_response(
            status.HTTP_403_FORBIDDEN,
            "Forbidden",
            str(error),
            cause="ML_TRAINING_NOT_COMPLETE",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/notifications", status_code=status.HTTP_204_NO_CONTENT)
@_record_received("NOTIFY")
def receive_training_notification(
    payload: NwdafMLModelTrainNotif,
    request: Request,
) -> Response:
    if request.app.state.fl_server is None:
        return _role_unavailable("FL Server")
    try:
        request.app.state.fl_server.receive_notification(payload)
    except KeyError:
        return problem_response(
            status.HTTP_404_NOT_FOUND,
            "Not Found",
            "ML Model Training callback route was not found",
            cause="RESOURCE_NOT_FOUND",
        )
    except InvalidMessageError as error:
        return _invalid_message(error)
    except (RequirementsError, ValueError) as error:
        return problem_response(
            status.HTTP_400_BAD_REQUEST,
            "Bad Request",
            str(error),
            cause="INVALID_MSG_FORMAT",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _not_found(subscription_id: str) -> Response:
    return problem_response(
        status.HTTP_404_NOT_FOUND,
        "Not Found",
        f"ML Model Training subscription {subscription_id} was not found",
        cause="RESOURCE_NOT_FOUND",
    )


def _role_unavailable(role: str) -> Response:
    return problem_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Service Unavailable",
        f"{role} role is not enabled",
        cause="SERVICE_NOT_AVAILABLE",
    )


def _requirements_not_met(error: RequirementsError) -> Response:
    return problem_response(
        status.HTTP_403_FORBIDDEN,
        "Forbidden",
        "federated learning requirements are not met",
        cause="ML_MODEL_TRAINING_REQS_NOT_MET",
        invalid_params=[
            {"param": violation.parameter, "reason": violation.reason}
            for violation in error.violations
        ],
    )


def _invalid_message(error: InvalidMessageError) -> Response:
    return problem_response(
        status.HTTP_400_BAD_REQUEST,
        "Bad Request",
        str(error),
        cause="INVALID_MSG_FORMAT",
        invalid_params=[
            {"param": violation.parameter, "reason": violation.reason}
            for violation in error.violations
        ],
    )
