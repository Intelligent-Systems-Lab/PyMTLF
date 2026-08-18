from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient
from nwdaf_context import verified_capability_checker

from py_mtlf.api.ml_model_monitor import _dispatch_retrain_intents
from py_mtlf.app import create_app
from py_mtlf.config import (
    FLClientSettings,
    FLServerSettings,
)
from py_mtlf.core.accuracy_policy import PolicyDecision
from py_mtlf.core.fl_experiment import ExperimentRole


def with_engines(
    settings,
    workspace: Path,
    *,
    server: bool = False,
    client: bool = False,
):
    payload = settings.model_dump(mode="python")
    federated = server or client
    payload["runtime"] = {"mode": "federated" if federated else "local"}
    payload["local_training"] = None if federated else payload["local_training"]
    payload["federated_learning"] = {
        "workspace_root": workspace,
        "public_base_url": settings.artifact.public_base_url,
        "server": FLServerSettings() if server else None,
        "client": (
            FLClientSettings(model_interoperability_ids=("001122",))
            if client
            else None
        ),
    }
    return settings.__class__.model_validate(payload)


def test_local_mode_preserves_local_training_lifecycle(settings, tmp_path):
    app = create_app(
        with_engines(settings, tmp_path / "local"),
        capability_checker=verified_capability_checker(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "local"
        assert app.state.fl_server is None
        assert app.state.fl_client is None
        assert app.state.training_coordinator is not None
        assert app.state.training_coordinator._workers
        assert "/internal/v1/ml-model-provision/subscriptions" in app.openapi()["paths"]
        assert "/internal/v1/ml-model-training/subscriptions" not in app.openapi()["paths"]


def test_fl_server_owns_model_services_without_local_training(settings, tmp_path):
    app = create_app(
        with_engines(settings, tmp_path / "server", server=True),
        capability_checker=verified_capability_checker(server=True),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is not None
        assert app.state.fl_client is None
        assert app.state.training_coordinator is None
        paths = app.openapi()["paths"]
        assert "/internal/v1/ml-model-provision/subscriptions" in paths
        assert "/internal/v1/ml-model-monitor/registrations" in paths
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            json=_training_subscription_body(),
        )
        assert response.status_code == 503
    assert app.state.fl_server._closing.is_set()


def test_fl_client_starts_foundation_without_server_coordinators(settings, tmp_path):
    workspace = tmp_path / "client"
    app = create_app(
        with_engines(settings, workspace, client=True),
        capability_checker=verified_capability_checker(client=True),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is None
        assert app.state.fl_client is not None
        assert app.state.training_coordinator is None
        paths = app.openapi()["paths"]
        assert "/internal/v1/artifacts/{artifact_key}" in paths
        assert "/internal/v1/ml-model-provision/subscriptions" not in paths
        assert "/internal/v1/ml-model-monitor/registrations" not in paths
        assert "/internal/v1/ml-model-training/subscriptions" in paths
        response = client.post(
            "/internal/v1/ml-model-training/notifications",
            json={"notifCorreId": "unknown", "termTrainReq": "STOP"},
        )
        assert response.status_code == 503
    assert workspace.is_dir()
    assert app.state.fl_client._closing.is_set()


def test_combined_profile_enables_both_fl_engines(settings, tmp_path):
    app = create_app(
        with_engines(settings, tmp_path / "combined", server=True, client=True),
        capability_checker=verified_capability_checker(server=True, client=True),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is not None
        assert app.state.fl_client is not None
        assert app.state.fl_client._experiments is app.state.fl_experiments
        assert app.state.fl_server._experiments is app.state.fl_experiments
        assert app.state.training_coordinator is None
        paths = app.openapi()["paths"]
        assert "/internal/v1/ml-model-provision/subscriptions" in paths
        assert "/internal/v1/ml-model-monitor/registrations" in paths
        assert "/internal/v1/ml-model-training/subscriptions" in paths
        assert "/internal/v1/ml-model-training/notifications" in paths
        subscription_response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            json=_training_subscription_body(),
        )
        notification_response = client.post(
            "/internal/v1/ml-model-training/notifications",
            json={"notifCorreId": "unknown", "termTrainReq": "STOP"},
        )
        assert subscription_response.status_code == 403
        assert notification_response.status_code == 404
    assert app.state.fl_server._closing.is_set()
    assert app.state.fl_client._closing.is_set()


def test_combined_profile_exposes_upper_client_lower_server_pairing_seam(
    settings, tmp_path
):
    app = create_app(
        with_engines(settings, tmp_path / "combined", server=True, client=True),
        capability_checker=verified_capability_checker(server=True, client=True),
    )
    with TestClient(app):
        reservation = app.state.fl_experiments.reserve_client(
            "upper-subscription", "upper-correlation"
        )
        plan_id = str(uuid4())
        app.state.fl_experiments.bind_plan(
            reservation.reservation_id,
            plan_id,
            ExperimentRole.BRANCH,
        )
        attached = app.state.fl_experiments.attach_server(
            reservation.reservation_id,
            plan_id,
            "lower-process",
        )

        assert attached.upper_client_subscription_ids == frozenset(
            {"upper-subscription"}
        )
        assert attached.server_process_id == "lower-process"


def test_new_app_construction_uses_a_fresh_experiment_registry(settings, tmp_path):
    first = create_app(
        with_engines(settings, tmp_path / "first", client=True),
        capability_checker=verified_capability_checker(client=True),
    )
    with TestClient(first):
        first.state.fl_experiments.reserve_client("subscription-a", "correlation-a")
    second = create_app(
        with_engines(settings, tmp_path / "second", client=True),
        capability_checker=verified_capability_checker(client=True),
    )
    with TestClient(second):
        assert second.state.fl_experiments is not first.state.fl_experiments
        assert second.state.fl_experiments.active() is None


def test_registry_fences_admission_before_server_and_publication_stop(settings, tmp_path):
    app = create_app(with_engines(settings, tmp_path / "server", server=True))
    shutdown_order = []
    shutdown_registry = app.state.fl_experiments.shutdown
    close_server = app.state.fl_server.close
    close_publication = app.state.publication.close

    def record_registry_shutdown():
        shutdown_order.append("registry")
        shutdown_registry()

    def record_server_close():
        shutdown_order.append("server")
        close_server()

    def record_publication_close():
        shutdown_order.append("publication")
        close_publication()

    app.state.fl_experiments.shutdown = record_registry_shutdown
    app.state.fl_server.close = record_server_close
    app.state.publication.close = record_publication_close

    with TestClient(app):
        pass

    assert shutdown_order == ["registry", "server", "publication"]


def test_only_local_mode_dispatches_current_dataset_training_path():
    triggered = [PolicyDecision(evaluated=True, triggered=True)]
    local_dataset = Mock()
    server_dataset = Mock()
    fl_server = Mock()

    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="local"),
            dataset_coordinator=local_dataset,
            fl_server=None,
        ),
        triggered,
    )
    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="federated"),
            dataset_coordinator=server_dataset,
            fl_server=fl_server,
        ),
        triggered,
    )

    local_dataset.accept_policy_intents.assert_called_once_with()
    server_dataset.accept_policy_intents.assert_not_called()
    fl_server.accept_policy_intents.assert_called_once_with()


def _training_subscription_body():
    return {
        "mLEventSubscs": [{"mLEvent": "UE_COMMUNICATION", "mLEventFilter": {}}],
        "notifUri": "http://server.example/callback",
        "notifCorreId": "correlation-1",
        "mLPreFlag": True,
    }
