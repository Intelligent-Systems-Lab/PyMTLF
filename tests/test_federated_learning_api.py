from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from py_mtlf.api import federated_learning
from py_mtlf.config import OrchestrationSettings
from py_mtlf.core.fl_flat import FlatFLCoordinator
from py_mtlf.core.fl_orchestration import (
    TopLevelCoordinatorUnavailableError,
    TopLevelModelFamilyNotFoundError,
    TopLevelRequestConflictError,
)
from py_mtlf.core.fl_server import FLProcess

REQUEST_ID = "00000000-0000-4000-8000-000000000701"
PLAN_ID = "550e8400-e29b-41d4-a716-446655440000"


def app_with(coordinator) -> FastAPI:
    app = FastAPI()
    app.state.fl_coordinator = coordinator
    app.include_router(federated_learning.router)
    return app


def snapshot(
    state: str = "ACCEPTED",
    *,
    mode: str = "hierarchical",
    participant_source: str = "static",
):
    return SimpleNamespace(
        request_id=REQUEST_ID,
        plan_id=None,
        model_family_id="ue-communication-default",
        state=state,
        mode=mode,
        participant_source=participant_source,
        trigger_source="private_api",
        current_round=None,
        completed_rounds=0,
        candidate_digest="",
        failure_cause="",
        failure_detail="",
    )


def test_private_training_request_returns_common_async_resource_and_status():
    coordinator = Mock()
    coordinator.submit_manual.return_value = snapshot()
    coordinator.get.return_value = SimpleNamespace(
        **{
            **snapshot(
                "CANDIDATE_READY",
                mode="flat",
                participant_source="static",
            ).__dict__,
            "current_round": 1,
            "completed_rounds": 2,
            "candidate_digest": "a" * 64,
        }
    )

    with TestClient(app_with(coordinator)) as client:
        created = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={
                "requestId": REQUEST_ID,
                "modelFamilyId": "ue-communication-default",
            },
        )
        request_status = client.get(created.headers["Location"])
        repeated_status = client.get(created.headers["Location"])

    assert created.status_code == 202
    assert created.json() == {
        "requestId": REQUEST_ID,
        "modelFamilyId": "ue-communication-default",
        "mode": "hierarchical",
        "participantSource": "static",
        "triggerSource": "private_api",
        "state": "ACCEPTED",
    }
    assert request_status.status_code == 200
    assert request_status.json() == {
        "requestId": REQUEST_ID,
        "modelFamilyId": "ue-communication-default",
        "mode": "flat",
        "participantSource": "static",
        "triggerSource": "private_api",
        "state": "CANDIDATE_READY",
        "currentRound": 1,
        "completedRounds": 2,
        "candidateDigest": "a" * 64,
    }
    assert repeated_status.json() == request_status.json()
    coordinator.submit_manual.assert_called_once_with(
        request_id=REQUEST_ID,
        model_family_id="ue-communication-default",
    )
    assert coordinator.get.call_count == 2


def test_hierarchical_training_status_exposes_the_record_correlation_id():
    coordinator = Mock()
    coordinator.submit_manual.return_value = SimpleNamespace(
        **{**snapshot().__dict__, "plan_id": PLAN_ID},
    )

    with TestClient(app_with(coordinator)) as client:
        response = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={
                "requestId": REQUEST_ID,
                "modelFamilyId": "ue-communication-default",
            },
        )

    assert response.status_code == 202
    assert response.json()["planId"] == PLAN_ID


def test_private_training_request_maps_conflict_missing_family_and_unavailable():
    exceptions = (
        (TopLevelRequestConflictError("active"), 409),
        (TopLevelModelFamilyNotFoundError("missing"), 404),
        (TopLevelCoordinatorUnavailableError("closing"), 503),
    )
    for error, expected_status in exceptions:
        coordinator = Mock()
        coordinator.submit_manual.side_effect = error
        with TestClient(app_with(coordinator)) as client:
            response = client.post(
                "/internal/v1/federated-learning/training-requests",
                json={
                    "requestId": REQUEST_ID,
                    "modelFamilyId": "ue-communication-default",
                },
            )
        assert response.status_code == expected_status
        assert response.headers["content-type"].startswith("application/problem+json")


def test_private_training_status_returns_not_found():
    coordinator = Mock()
    coordinator.get.return_value = None
    with TestClient(app_with(coordinator)) as client:
        response = client.get(
            f"/internal/v1/federated-learning/training-requests/{REQUEST_ID}"
        )

    assert response.status_code == 404
    assert response.json()["cause"] == "RESOURCE_NOT_FOUND"


def test_generation_abort_makes_old_common_training_request_return_not_found():
    server = Mock()
    server.start_flat.return_value = FLProcess(process_id="process-1", intent=None)
    catalog = Mock()
    catalog.current.return_value = SimpleNamespace(
        model_id=1,
        descriptor=SimpleNamespace(
            event="UE_COMMUNICATION",
            event_filter={},
            target_ue=None,
        ),
    )
    tracking_area = Mock()
    tracking_area.wire_value.return_value = {
        "plmnId": {"mcc": "466", "mnc": "92"},
        "tac": "001101",
    }
    planner = Mock()
    planner.build.return_value = SimpleNamespace(
        topology_version=1,
        clients=(
            SimpleNamespace(
                nf_instance_id="00000000-0000-4000-8000-000000000301",
                tracking_areas=(tracking_area,),
            ),
            SimpleNamespace(
                nf_instance_id="00000000-0000-4000-8000-000000000302",
                tracking_areas=(tracking_area,),
            ),
        ),
    )
    nwdaf_context = Mock()
    nwdaf_context.get.return_value.nf_instance_id = (
        "00000000-0000-4000-8000-000000000001"
    )
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(
            mode="flat",
            participant_source="static",
        ),
        server=server,
        policy=Mock(),
        catalog=catalog,
        nwdaf_context=nwdaf_context,
        planner=planner,
        terminal_status_ttl_seconds=60,
    )

    with TestClient(app_with(coordinator)) as client:
        created = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={
                "requestId": REQUEST_ID,
                "modelFamilyId": "ue-communication-default",
            },
        )
        coordinator.abort_generation("containing NWDAF generation changed")
        coordinator.abort_generation("duplicate reset")
        old_status = client.get(created.headers["Location"])

    assert created.status_code == 202
    assert old_status.status_code == 404
    assert old_status.json()["cause"] == "RESOURCE_NOT_FOUND"
