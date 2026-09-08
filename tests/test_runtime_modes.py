from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from nwdaf_context import context_client, verified_capability_checker
from pydantic import ValidationError

from py_mtlf.api.ml_model_monitor import _dispatch_retrain_intents
from py_mtlf.app import create_app
from py_mtlf.config import (
    FLClientSettings,
    FLServerSettings,
)
from py_mtlf.core.accuracy_policy import PolicyDecision
from py_mtlf.core.fl_experiment import ExperimentRole
from py_mtlf.core.nwdaf_context import NwdafContext


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
            FLClientSettings(
                training_data={"collection_trigger": "consumer_subscription"},
                model_interoperability_ids=("001122",),
            )
            if client
            else None
        ),
    }
    if server and not client:
        payload["federated_learning"]["orchestration"] = {
            "mode": "flat",
            "participant_source": "monitor_scopes",
        }
        payload["federated_learning"]["training_trigger"] = {
            "degradation": {"enabled": True},
            "private_api": {"enabled": False},
        }
    return settings.__class__.model_validate(payload)


def with_hierarchy(settings, workspace: Path, topology_path: Path, *, private_api: bool):
    payload = with_engines(settings, workspace, server=True).model_dump(mode="python")
    payload["federated_learning"]["orchestration"] = {
        "mode": "hierarchical",
        "participant_source": "static",
    }
    payload["federated_learning"]["topology"] = {
        "strategy": "static",
        "config_file": topology_path,
    }
    payload["federated_learning"]["training_trigger"] = {
        "degradation": {"enabled": True},
        "private_api": {"enabled": private_api}
    }
    return settings.__class__.model_validate(payload)


def with_static_flat(settings, workspace: Path, topology_path: Path):
    payload = with_engines(settings, workspace, server=True).model_dump(mode="python")
    payload["federated_learning"]["orchestration"] = {
        "mode": "flat",
        "participant_source": "static",
    }
    payload["federated_learning"]["topology"] = {
        "strategy": "static",
        "config_file": topology_path,
    }
    payload["federated_learning"]["training_trigger"] = {
        "degradation": {"enabled": False},
        "private_api": {"enabled": True},
    }
    return settings.__class__.model_validate(payload)


def write_topology(path: Path) -> None:
    path.write_text(
        """
admission:
  mode: complete_required
policy: &policy
  allow_additional_candidates: false
  additional_candidate_priority: 0
  selection_method: priority
  min_available_nodes: 1
  fraction_train: 1.0
  min_train_nodes: 1
  accept_failures: false
  min_completion_rate: 1.0
strategy: &strategy
  method: fedProx
  aggregation: sampleWeighted
  method_parameters: {proximal_mu: 0.01}
branch_groups:
  - branches:
      - nf_instance_id: 00000000-0000-4000-8000-000000000010
        priority: 100
        report_after: {count: 1, unit: round}
    policy: *policy
    strategy: *strategy
    leaves:
      - nf_instance_id: 00000000-0000-4000-8000-000000000101
        priority: 100
        report_after: {count: 1, unit: epoch}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_flat_topology(path: Path) -> None:
    path.write_text(
        """
version: 1
clients:
  - nf_instance_id: 00000000-0000-4000-8000-000000000101
    scope:
      tracking_areas:
        - plmn_id: {mcc: "466", mnc: "92"}
          tac: "001101"
  - nf_instance_id: 00000000-0000-4000-8000-000000000102
    scope:
      tracking_areas:
        - plmn_id: {mcc: "466", mnc: "92"}
          tac: "001102"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_local_mode_preserves_local_training_lifecycle(settings, tmp_path):
    app = create_app(
        with_engines(settings, tmp_path / "local"),
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "local"
        assert app.state.fl_server is None
        assert app.state.fl_client is None
        assert app.state.fl_coordinator is None
        assert app.state.training_coordinator is not None
        assert app.state.training_coordinator._workers
        assert "/internal/v1/ml-model-provision/subscriptions" in app.openapi()["paths"]
        assert "/internal/v1/ml-model-training/subscriptions" not in app.openapi()["paths"]


def test_fl_server_owns_model_services_without_local_training(settings, tmp_path):
    app = create_app(
        with_engines(settings, tmp_path / "server", server=True),
        capability_checker=verified_capability_checker(server=True),
        nwdaf_context_client=context_client(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is not None
        assert app.state.fl_client is None
        assert app.state.fl_branch is None
        assert app.state.fl_coordinator is not None
        assert app.state.fl_root is None
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
        nwdaf_context_client=context_client(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is None
        assert app.state.fl_client is not None
        assert app.state.fl_branch is None
        assert app.state.fl_coordinator is None
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
        nwdaf_context_client=context_client(),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["runtimeMode"] == "federated"
        assert app.state.fl_server is not None
        assert app.state.fl_client is not None
        assert app.state.fl_branch is not None
        assert app.state.fl_coordinator is None
        assert app.state.fl_client._branch_coordinator is app.state.fl_branch
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
    assert app.state.fl_branch._closing is True


def test_generic_training_api_is_mounted_only_when_enabled(settings, tmp_path):
    topology_path = tmp_path / "topology.yaml"
    write_topology(topology_path)
    disabled = create_app(
        with_hierarchy(settings, tmp_path / "disabled", topology_path, private_api=False),
        capability_checker=verified_capability_checker(server=True),
        nwdaf_context_client=context_client(),
    )
    enabled = create_app(
        with_hierarchy(settings, tmp_path / "enabled", topology_path, private_api=True),
        capability_checker=verified_capability_checker(server=True),
        nwdaf_context_client=context_client(),
    )

    with TestClient(disabled) as client:
        assert disabled.state.fl_root is not None
        assert disabled.state.fl_coordinator is disabled.state.fl_root
        assert (
            client.post(
                "/internal/v1/federated-learning/training-requests",
                json={},
            ).status_code
            == 404
        )
    with TestClient(enabled) as client:
        malformed = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={},
        )
        missing_family = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={
                "requestId": "00000000-0000-4000-8000-000000000701",
                "modelFamilyId": "missing-family",
            },
        )
        deployment_overrides = [
            {"mode": "flat"},
            {"participants": []},
            {"topology": {"clients": []}},
            {"strategy": {"algorithm": "fedavg"}},
            {"collectionTrigger": "consumer_subscription"},
            {"profile": "controlled"},
            {"timeWindow": {"startTime": "2026-08-01T00:00:00Z"}},
            {"callbackUri": "http://override.example"},
            {"artifactUrl": "http://override.example/model.tar.gz"},
        ]
        override_responses = [
            client.post(
                "/internal/v1/federated-learning/training-requests",
                json={
                    "requestId": "00000000-0000-4000-8000-000000000702",
                    "modelFamilyId": "ue-communication-default",
                    **override,
                },
            )
            for override in deployment_overrides
        ]
        assert malformed.status_code == 400
        assert malformed.headers["content-type"].startswith("application/problem+json")
        assert missing_family.status_code == 404
        assert all(response.status_code == 400 for response in override_responses)
        assert all(
            response.headers["content-type"].startswith("application/problem+json")
            for response in override_responses
        )


def test_static_flat_constructs_the_only_top_level_owner_and_generic_route(
    settings, tmp_path
):
    topology_path = tmp_path / "flat-topology.yaml"
    write_flat_topology(topology_path)
    app = create_app(
        with_static_flat(settings, tmp_path / "static-flat", topology_path),
        capability_checker=verified_capability_checker(server=True),
        nwdaf_context_client=context_client(),
    )

    with TestClient(app) as client:
        assert type(app.state.fl_coordinator).__name__ == "FlatFLCoordinator"
        assert app.state.fl_root is None
        assert app.state.fl_branch is None
        paths = app.openapi()["paths"]
        assert "/internal/v1/federated-learning/training-requests" in paths
        assert "/internal/v1/hierarchical-fl/training-requests" not in paths
        assert (
            client.post(
                "/internal/v1/hierarchical-fl/training-requests",
                json={},
            ).status_code
            == 404
        )
        app.state.fl_coordinator.close()
        unavailable = client.post(
            "/internal/v1/federated-learning/training-requests",
            json={
                "requestId": "00000000-0000-4000-8000-000000000701",
                "modelFamilyId": "ue-communication-default",
            },
        )
        assert unavailable.status_code == 503
        assert unavailable.headers["content-type"].startswith(
            "application/problem+json"
        )


def test_hierarchy_app_construction_rejects_invalid_topology(settings, tmp_path):
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text("version: 2\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        create_app(
            with_hierarchy(settings, tmp_path / "invalid", topology_path, private_api=False)
        )


def test_combined_profile_exposes_upper_client_lower_server_pairing_seam(
    settings, tmp_path
):
    app = create_app(
        with_engines(settings, tmp_path / "combined", server=True, client=True),
        capability_checker=verified_capability_checker(server=True, client=True),
        nwdaf_context_client=context_client(),
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
        nwdaf_context_client=context_client(),
    )
    with TestClient(first):
        first.state.fl_experiments.reserve_client("subscription-a", "correlation-a")
    second = create_app(
        with_engines(settings, tmp_path / "second", client=True),
        capability_checker=verified_capability_checker(client=True),
        nwdaf_context_client=context_client(),
    )
    with TestClient(second):
        assert second.state.fl_experiments is not first.state.fl_experiments
        assert second.state.fl_experiments.active() is None


def test_app_generation_change_clears_old_slot_and_workspace_before_reopening(
    settings,
):
    context = MutableContextClient()
    app = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context,
    )
    old_plan = str(uuid4())
    new_plan = str(uuid4())

    with TestClient(app):
        app.state.fl_experiments.reserve_root(old_plan)
        old_directory = settings.federated_learning.workspace_root / old_plan
        old_directory.mkdir()
        (old_directory / "old-artifact").write_bytes(b"old")
        context.process_instance_id = "33333333-3333-4333-8333-333333333333"

        snapshot = app.state.generation_monitor.refresh_once()

        assert snapshot.ready is True
        assert app.state.fl_experiments.active() is None
        assert not old_directory.exists()
        assert app.state.fl_experiments.reserve_root(new_plan).plan_id == new_plan


def test_registry_fences_admission_before_server_and_publication_stop(settings, tmp_path):
    app = create_app(with_engines(settings, tmp_path / "server", server=True))
    shutdown_order = []
    shutdown_registry = app.state.fl_experiments.shutdown
    close_server = app.state.fl_server.close
    stop_publication = app.state.publication.stop
    close_publication = app.state.publication.close
    stop_generation_monitor = app.state.generation_monitor.stop

    def record_registry_shutdown():
        shutdown_order.append("registry")
        shutdown_registry()

    def record_server_close():
        shutdown_order.append("server")
        close_server()

    def record_publication_stop():
        shutdown_order.append("publication-stop")
        stop_publication()

    def record_publication_close():
        shutdown_order.append("publication-close")
        app.state.publication.stop = stop_publication
        close_publication()

    def record_generation_monitor_stop():
        shutdown_order.append("generation-monitor")
        stop_generation_monitor()

    app.state.fl_experiments.shutdown = record_registry_shutdown
    app.state.fl_server.close = record_server_close
    app.state.publication.stop = record_publication_stop
    app.state.publication.close = record_publication_close
    app.state.generation_monitor.stop = record_generation_monitor_stop

    with TestClient(app):
        pass

    assert shutdown_order == [
        "generation-monitor",
        "publication-stop",
        "registry",
        "server",
        "publication-close",
    ]


class MutableContextClient:
    def __init__(self):
        self.process_instance_id = "22222222-2222-4222-8222-222222222222"

    def open(self):
        return None

    def close(self):
        return None

    def get(self, *, refresh=False):
        del refresh
        return NwdafContext(
            nf_instance_id="11111111-1111-4111-8111-111111111111",
            containing_nwdaf_process_instance_id=self.process_instance_id,
            api_root="http://go.example",
            internal_api_root="http://go-internal.example",
        )


def test_local_and_federated_modes_dispatch_to_selected_owners():
    triggered = [PolicyDecision(evaluated=True, triggered=True)]
    local_dataset = Mock()
    server_dataset = Mock()
    fl_coordinator = Mock()

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
            fl_coordinator=fl_coordinator,
            federated_degradation_enabled=True,
        ),
        triggered,
    )

    local_dataset.accept_policy_intents.assert_called_once_with()
    server_dataset.accept_policy_intents.assert_not_called()
    fl_coordinator.accept_policy_intents.assert_called_once_with()


def test_hierarchy_dispatches_degradation_through_common_coordinator():
    triggered = [PolicyDecision(evaluated=True, triggered=True)]
    fl_root = Mock()
    fl_server = Mock()

    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="federated"),
            dataset_coordinator=Mock(),
            fl_coordinator=fl_root,
            federated_degradation_enabled=True,
        ),
        triggered,
    )

    fl_root.accept_policy_intents.assert_called_once_with()
    fl_server.accept_policy_intents.assert_not_called()


def test_non_owner_or_disabled_federated_profile_atomically_discards_intents():
    policy = Mock()

    _dispatch_retrain_intents(
        SimpleNamespace(
            runtime=SimpleNamespace(mode="federated"),
            dataset_coordinator=Mock(),
            fl_coordinator=None,
            federated_degradation_enabled=False,
            accuracy_policy=policy,
        ),
        [PolicyDecision(evaluated=True, triggered=True)],
    )

    policy.discard_intents.assert_called_once_with()


def _training_subscription_body():
    return {
        "mLEventSubscs": [{"mLEvent": "UE_COMMUNICATION", "mLEventFilter": {}}],
        "notifUri": "http://server.example/callback",
        "notifCorreId": "correlation-1",
        "mLPreFlag": True,
    }
