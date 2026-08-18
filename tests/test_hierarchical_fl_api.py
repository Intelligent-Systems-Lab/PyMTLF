from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from py_mtlf.api import hierarchical_fl
from py_mtlf.core.fl_root import (
    RootCoordinatorUnavailableError,
    RootModelFamilyNotFoundError,
    RootRequestConflictError,
    RootRequestSnapshot,
    RootRequestState,
)

REQUEST_ID = "00000000-0000-4000-8000-000000000701"
PLAN_ID = "00000000-0000-4000-8000-000000000801"


def app_with(coordinator) -> FastAPI:
    app = FastAPI()
    app.state.fl_root = coordinator
    app.include_router(hierarchical_fl.router)
    return app


def snapshot(state=RootRequestState.ACCEPTED) -> RootRequestSnapshot:
    return RootRequestSnapshot(
        request_id=REQUEST_ID,
        plan_id=PLAN_ID,
        model_family_id="ue-communication-default",
        state=state,
    )


def test_private_hierarchy_request_returns_async_resource_and_status():
    coordinator = Mock()
    coordinator.submit_manual.return_value = snapshot()
    coordinator.get.return_value = snapshot(RootRequestState.PREPARATION_WAITING)

    with TestClient(app_with(coordinator)) as client:
        created = client.post(
            "/internal/v1/hierarchical-fl/training-requests",
            json={
                "requestId": REQUEST_ID,
                "modelFamilyId": "ue-communication-default",
            },
        )
        status = client.get(created.headers["Location"])

    assert created.status_code == 202
    assert created.json() == {
        "requestId": REQUEST_ID,
        "planId": PLAN_ID,
        "modelFamilyId": "ue-communication-default",
        "state": "ACCEPTED",
    }
    assert status.status_code == 200
    assert status.json()["state"] == "PREPARATION_WAITING"
    coordinator.submit_manual.assert_called_once_with(
        request_id=REQUEST_ID,
        model_family_id="ue-communication-default",
    )


def test_private_hierarchy_request_maps_conflict_missing_family_and_unavailable():
    exceptions = (
        (RootRequestConflictError("active"), 409),
        (RootModelFamilyNotFoundError("missing"), 404),
        (RootCoordinatorUnavailableError("closing"), 503),
    )
    for error, expected_status in exceptions:
        coordinator = Mock()
        coordinator.submit_manual.side_effect = error
        with TestClient(app_with(coordinator)) as client:
            response = client.post(
                "/internal/v1/hierarchical-fl/training-requests",
                json={
                    "requestId": REQUEST_ID,
                    "modelFamilyId": "ue-communication-default",
                },
            )
        assert response.status_code == expected_status
        assert response.headers["content-type"].startswith("application/problem+json")


def test_private_hierarchy_status_returns_not_found():
    coordinator = Mock()
    coordinator.get.return_value = None
    with TestClient(app_with(coordinator)) as client:
        response = client.get(
            f"/internal/v1/hierarchical-fl/training-requests/{REQUEST_ID}"
        )

    assert response.status_code == 404
    assert response.json()["cause"] == "RESOURCE_NOT_FOUND"
