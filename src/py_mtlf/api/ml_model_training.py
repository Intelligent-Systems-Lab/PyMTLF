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
