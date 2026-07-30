from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from py_mtlf.api.ml_model_monitor import _dispatch_retrain_intents
from py_mtlf.app import create_app
from py_mtlf.config import FederatedLearningSettings, RuntimeSettings
from py_mtlf.core.accuracy_policy import PolicyDecision


def with_mode(settings, mode: str, workspace: Path):
    return settings.model_copy(
        update={
            "runtime": RuntimeSettings(mode=mode),
            "federated_learning": FederatedLearningSettings(
                workspace_root=workspace,
                public_base_url=settings.artifact.public_base_url,
            ),
        }
    )


def test_local_mode_preserves_local_training_lifecycle(settings, tmp_path):
    app = create_app(with_mode(settings, "local", tmp_path / "local"))
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "local"
        assert app.state.training_coordinator._workers
        assert "/internal/v1/ml-model-provision/subscriptions" in app.openapi()["paths"]


def test_fl_server_owns_model_services_without_local_training(settings, tmp_path):
    app = create_app(with_mode(settings, "fl_server", tmp_path / "server"))
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "fl_server"
        assert not app.state.training_coordinator._workers
        paths = app.openapi()["paths"]
        assert "/internal/v1/ml-model-provision/subscriptions" in paths
        assert "/internal/v1/ml-model-monitor/registrations" in paths


def test_fl_client_starts_foundation_without_server_coordinators(settings, tmp_path):
    workspace = tmp_path / "client"
    app = create_app(with_mode(settings, "fl_client", workspace))
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "fl_client"
        assert not app.state.training_coordinator._workers
        paths = app.openapi()["paths"]
        assert "/internal/v1/artifacts/{artifact_key}" in paths
        assert "/internal/v1/ml-model-provision/subscriptions" not in paths
        assert "/internal/v1/ml-model-monitor/registrations" not in paths
        assert "/internal/v1/ml-model-training/subscriptions" in paths
    assert workspace.is_dir()


def test_only_local_mode_dispatches_current_dataset_training_path():
    triggered = [PolicyDecision(evaluated=True, triggered=True)]
    local_dataset = Mock()
    server_dataset = Mock()
    fl_server = Mock()

    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="local"),
            dataset_coordinator=local_dataset,
        ),
        triggered,
    )
    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="fl_server"),
            dataset_coordinator=server_dataset,
            fl_server=fl_server,
        ),
        triggered,
    )

    local_dataset.accept_policy_intents.assert_called_once_with()
    server_dataset.accept_policy_intents.assert_not_called()
    fl_server.accept_policy_intents.assert_called_once_with()
