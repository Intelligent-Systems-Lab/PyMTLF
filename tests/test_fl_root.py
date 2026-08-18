from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from py_mtlf.config import FederatedStrategySettings
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_root import (
    FLRootCoordinator,
    RootRequestConflictError,
    RootRequestState,
)
from py_mtlf.core.fl_topology import StaticTopologyPlanner
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.wire.private import SelectedTarget

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_ID = "00000000-0000-4000-8000-000000000010"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
REQUEST_A_ID = "00000000-0000-4000-8000-000000000701"
REQUEST_B_ID = "00000000-0000-4000-8000-000000000702"


def root_coordinator(tmp_path: Path, *, resolver_side_effect=None):
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        f"""
version: 1
admission:
  mode: complete_required
branches:
  - nf_instance_id: {BRANCH_ID}
    leaves:
      - nf_instance_id: {LEAF_B_ID}
      - nf_instance_id: {LEAF_A_ID}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    planner = StaticTopologyPlanner.load(topology_path)
    resolver = Mock()

    def resolve(*, nf_instance_id, role, **_kwargs):
        if resolver_side_effect is not None:
            resolver_side_effect(nf_instance_id, role)
        return ResolvedHierarchyNode(
            nf_instance_id=nf_instance_id,
            role=role,
            target=SelectedTarget(
                nfInstanceId=nf_instance_id,
                nfServiceInstanceId=f"training-{nf_instance_id}",
                serviceName="nnwdaf-mlmodeltraining",
                apiRoot=f"http://{nf_instance_id}.example",
                selectionSource="NRF",
            ),
        )

    resolver.resolve.side_effect = resolve
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=ROOT_ID,
        api_root="http://root.example",
        internal_api_root="http://go.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=FLCapabilityType.SERVER,
            ),
        ),
    )
    model = SimpleNamespace(
        model_id=1,
        artifact=SimpleNamespace(key="a" * 64, url="http://root.example/base"),
        descriptor=SimpleNamespace(
            family_id="ue-communication-default",
            event="UE_COMMUNICATION",
            event_filter={"networkArea": {"tais": []}},
            target_ue=None,
            model_interoperability="001122",
        ),
    )
    catalog = Mock()
    catalog.current.side_effect = lambda family: (
        model if family == "ue-communication-default" else None
    )
    artifact_service = Mock()
    artifact_service.publish_branch_assignment.return_value = SimpleNamespace(
        url="http://root.example/assignments/branch"
    )
    workspace = Mock()
    registry = FLExperimentRegistry()
    server = Mock()

    def start_hierarchy_preparation(**kwargs):
        registry.attach_server(
            kwargs["reservation_id"],
            kwargs["plan_id"],
            "server-process",
        )
        return SimpleNamespace(process_id="server-process")

    server.start_hierarchy_preparation.side_effect = start_hierarchy_preparation
    policy = Mock()
    policy.take_intents.return_value = ()
    loader = Mock()
    loader.load.return_value = SimpleNamespace()
    coordinator = FLRootCoordinator(
        strategy=FederatedStrategySettings.model_validate(
            {
                "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
                "participant_selection": "all",
                "waiting_policy": "all",
                "aggregation": "sample_weighted",
            }
        ),
        planner=planner,
        resolver=resolver,
        nwdaf_context=context,
        catalog=catalog,
        artifact_service=artifact_service,
        workspace=workspace,
        server=server,
        policy=policy,
        experiments=registry,
        loader=loader,
    )
    return (
        coordinator,
        resolver,
        artifact_service,
        workspace,
        server,
        registry,
        policy,
        catalog,
        model,
    )


def test_root_request_validates_tree_publishes_assignment_and_waits(tmp_path):
    (
        coordinator,
        resolver,
        artifacts,
        _workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    try:
        accepted = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        waiting = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.PREPARATION_WAITING, RootRequestState.FAILED},
            timeout=2,
        )

        assert accepted.state is RootRequestState.ACCEPTED
        assert waiting.state is RootRequestState.PREPARATION_WAITING
        assert waiting.plan_id == accepted.plan_id
        assert [call.kwargs["role"] for call in resolver.resolve.call_args_list] == [
            HierarchyNodeRole.BRANCH,
            HierarchyNodeRole.LEAF,
            HierarchyNodeRole.LEAF,
        ]
        publication = artifacts.publish_branch_assignment.call_args.kwargs
        assert publication["plan_id"] == accepted.plan_id
        assert publication["publisher_nf_instance_id"] == ROOT_ID
        assert publication["branch_nf_instance_id"] == BRANCH_ID
        assert publication["assigned_leaf_nf_instance_ids"] == (LEAF_A_ID, LEAF_B_ID)
        dispatched = server.start_hierarchy_preparation.call_args.kwargs
        assert dispatched["plan_id"] == accepted.plan_id
        assert dispatched["reservation_id"] == registry.active().reservation_id
        assert dispatched["targets"][0].branch_nf_instance_id == BRANCH_ID
        assert dispatched["targets"][0].assignment_url.endswith("/branch")
        assert registry.active().server_process_id == "server-process"
    finally:
        coordinator.close()


def test_root_request_is_idempotent_and_rejects_conflicting_active_request(tmp_path):
    coordinator, *_ = root_coordinator(tmp_path)
    try:
        first = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        replay = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        assert replay.plan_id == first.plan_id

        with pytest.raises(RootRequestConflictError):
            coordinator.submit_manual(
                request_id=REQUEST_A_ID,
                model_family_id="different-family",
            )
        with pytest.raises(RootRequestConflictError):
            coordinator.submit_manual(
                request_id=REQUEST_B_ID,
                model_family_id="ue-communication-default",
            )
    finally:
        coordinator.close()


def test_failed_root_attempt_releases_slot_and_allows_explicit_new_request(tmp_path):
    failed_once = False

    def discovery_failure(nf_instance_id, role):
        nonlocal failed_once
        if not failed_once and role is HierarchyNodeRole.BRANCH:
            failed_once = True
            raise HierarchyDiscoveryError(f"failed {nf_instance_id}")

    (
        coordinator,
        _resolver,
        _artifacts,
        workspace,
        _server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path, resolver_side_effect=discovery_failure)
    try:
        first = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        failed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.FAILED},
            timeout=2,
        )
        assert failed.failure_cause == "DISCOVERY_FAILED"
        assert registry.active() is None
        workspace.release_plan.assert_called_once_with(first.plan_id)

        second = coordinator.submit_manual(
            request_id=REQUEST_B_ID,
            model_family_id="ue-communication-default",
        )
        waiting = coordinator.wait_for_state(
            REQUEST_B_ID,
            {RootRequestState.PREPARATION_WAITING, RootRequestState.FAILED},
            timeout=2,
        )
        assert waiting.state is RootRequestState.PREPARATION_WAITING
        assert second.plan_id != first.plan_id
    finally:
        coordinator.close()


def test_degradation_failure_latches_without_consuming_another_policy_intent(tmp_path):
    def discovery_failure(nf_instance_id, _role):
        raise HierarchyDiscoveryError(f"failed {nf_instance_id}")

    (
        coordinator,
        _resolver,
        _artifacts,
        _workspace,
        _server,
        _registry,
        policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path, resolver_side_effect=discovery_failure)
    intent = SimpleNamespace(
        family_key="ue-communication-default",
        active_scopes=(),
    )
    policy.take_intents.return_value = (intent,)
    try:
        coordinator.accept_policy_intents()
        request = coordinator.requests()[0]
        failed = coordinator.wait_for_state(
            request.request_id,
            {RootRequestState.FAILED},
            timeout=2,
        )
        assert failed.failure_cause == "DISCOVERY_FAILED"

        coordinator.accept_policy_intents()

        policy.take_intents.assert_called_once_with()
        policy.complete_retrain.assert_not_called()
    finally:
        coordinator.close()


def test_root_status_does_not_expose_peer_response_body(tmp_path):
    (
        coordinator,
        _resolver,
        _artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    server.start_hierarchy_preparation.side_effect = RuntimeError(
        "participant preparation create failed with 503: private-peer-body"
    )
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        failed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.FAILED},
            timeout=2,
        )

        assert failed.failure_cause == "PREPARATION_DISPATCH_FAILED"
        assert failed.failure_detail == "upper-tier preparation dispatch failed"
        assert "private-peer-body" not in failed.failure_detail
    finally:
        coordinator.close()


def test_root_rejects_base_model_change_after_assignment_publication(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        catalog,
        model,
    ) = root_coordinator(tmp_path)
    changed_model = SimpleNamespace(
        **{
            **model.__dict__,
            "artifact": SimpleNamespace(key="b" * 64, url="http://root.example/new-base"),
        }
    )
    catalog.current.side_effect = [model, model, model, changed_model]
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        failed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.FAILED},
            timeout=2,
        )

        assert failed.failure_cause == "VALIDATION_FAILED"
        artifacts.publish_branch_assignment.assert_called_once()
        server.start_hierarchy_preparation.assert_not_called()
    finally:
        coordinator.close()


def test_root_rejects_generated_plan_identity_retired_in_this_process(tmp_path, monkeypatch):
    coordinator, *_rest, registry, _policy, _catalog, _model = root_coordinator(tmp_path)
    retired_plan_id = "00000000-0000-4000-8000-000000000900"
    reservation = registry.reserve_root(retired_plan_id)
    registry.mark_terminal(reservation.reservation_id, "FAILED")
    registry.begin_cleanup(reservation.reservation_id)
    registry.release(reservation.reservation_id)
    monkeypatch.setattr(
        "py_mtlf.core.fl_root.uuid4",
        lambda: UUID(retired_plan_id),
    )
    try:
        with pytest.raises(RootRequestConflictError, match="retired"):
            coordinator.submit_manual(
                request_id=REQUEST_A_ID,
                model_family_id="ue-communication-default",
            )
        assert coordinator.requests() == ()
    finally:
        coordinator.close()
