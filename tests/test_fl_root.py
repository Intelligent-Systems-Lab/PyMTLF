import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from py_mtlf.config import (
    ArtifactSettings,
    FederatedLearningSettings,
    FederatedStrategySettings,
    FLServerSettings,
)
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import HierarchyPreparationResultArtifact
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy import PreparationOutcome
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
from py_mtlf.core.fl_server import (
    FLParticipant,
    FLProcess,
    FLServerEngine,
    FLServerState,
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
)
from py_mtlf.core.fl_topology import StaticTopologyPlanner
from py_mtlf.core.fl_workspace import FLWorkspace, ValidatedHierarchyArtifact
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainNotif
from py_mtlf.wire.private import SelectedTarget

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_ID = "00000000-0000-4000-8000-000000000010"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
REQUEST_A_ID = "00000000-0000-4000-8000-000000000701"
REQUEST_B_ID = "00000000-0000-4000-8000-000000000702"


def root_coordinator(
    tmp_path: Path,
    *,
    resolver_side_effect=None,
    round_count: int = 1,
    terminal_status_ttl_seconds: int = 3600,
    clock=None,
    workspace_override=None,
):
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
        containing_nwdaf_process_instance_id="22222222-2222-4222-8222-222222222222",
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
    workspace = workspace_override or Mock()
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
    collection_cancelled = threading.Event()

    def collect_hierarchy_preparation(_process_id):
        collection_cancelled.wait()
        raise RuntimeError("test hierarchy collection cancelled")

    def cancel_hierarchy_preparation(_process_id, _reason):
        collection_cancelled.set()

    server.collect_hierarchy_preparation.side_effect = collect_hierarchy_preparation
    server.cancel_hierarchy_preparation.side_effect = cancel_hierarchy_preparation
    policy = Mock()
    policy.take_intents.return_value = ()
    loader = Mock()
    loader.load.return_value = SimpleNamespace()
    root_kwargs = {}
    if clock is not None:
        root_kwargs["clock"] = clock
    coordinator = FLRootCoordinator(
        strategy=FederatedStrategySettings.model_validate(
            {
                "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
                "participant_selection": "all",
                "waiting_policy": "all",
                "aggregation": "sample_weighted",
            }
        ),
        server_settings=FLServerSettings(
            round_count=round_count,
            client_training={"epochs": 3},
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
        terminal_status_ttl_seconds=terminal_status_ttl_seconds,
        **root_kwargs,
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


def test_go_generation_reset_discards_root_status_and_releases_slot(tmp_path):
    coordinator, _resolver, _artifacts, _workspace, server, registry, *_ = (
        root_coordinator(tmp_path)
    )
    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.PREPARATION_WAITING},
        timeout=2,
    )

    coordinator.abort_generation("containing NWDAF process generation changed")

    assert coordinator.get(REQUEST_A_ID) is None
    assert registry.active() is None
    replacement = coordinator.submit_manual(
        request_id=REQUEST_B_ID,
        model_family_id="ue-communication-default",
    )
    assert replacement.plan_id
    coordinator.close()
    server.cancel_hierarchy_preparation.assert_called()


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
        assert dispatched["targets"][0].participant_nf_instance_id == BRANCH_ID
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


def test_root_classifies_post_dispatch_collection_failure(tmp_path):
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
    server.collect_hierarchy_preparation.side_effect = RuntimeError(
        "lower collection failed"
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

        assert failed.failure_cause == "PREPARATION_FAILED"
        assert failed.failure_detail == "one or more Branch preparations failed"
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


def _configure_branch_result(
    *,
    tmp_path,
    artifacts,
    workspace,
    server,
    registry,
    outcome: PreparationOutcome,
):
    file_digests = {
        "model.py": "1" * 64,
        "model.npy": "2" * 64,
        "scaler.pkl": "3" * 64,
    }
    artifacts.publish_branch_assignment.return_value = SimpleNamespace(
        url="http://root.example/assignments/branch",
        contract=SimpleNamespace(file_digests=file_digests),
    )
    result_url = "http://branch.example/artifacts/" + "d" * 64

    def collect(_process_id):
        plan_id = registry.active().plan_id
        notification = {
            "notifCorreId": "branch-result",
            "mlCorreId": "server-process",
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": result_url},
                }
            ],
        }
        if outcome is PreparationOutcome.FAILED:
            notification["termTrainReq"] = "NOT_AVAILABLE_ML_TRAIN"
        return HierarchyPreparationCollection(
            process_id="server-process",
            plan_id=plan_id,
            participants=(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_ID,
                    resource_location="http://root.example/subscriptions/branch",
                    assignment_url=result_url,
                    notification=NwdafMLModelTrainNotif.model_validate(notification),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
            ),
            timed_out_participant_nf_instance_ids=(),
        )

    def download_result(_url, **kwargs):
        plan_id = kwargs["expected_plan_id"]
        prepared = (
            [{"nf_instance_id": LEAF_A_ID}, {"nf_instance_id": LEAF_B_ID}]
            if outcome is PreparationOutcome.READY
            else [{"nf_instance_id": LEAF_A_ID}]
        )
        failed = (
            []
            if outcome is PreparationOutcome.READY
            else [
                {
                    "nf_instance_id": LEAF_B_ID,
                    "cause": "NOT_AVAILABLE_ML_TRAIN",
                }
            ]
        )
        contract = HierarchyPreparationResultArtifact.model_validate(
            {
                "artifact_role": "HIERARCHY_PREPARATION_RESULT",
                "bundle_schema_version": "1.0",
                "file_digests": file_digests,
                "hierarchy_metadata": {
                    "contract_version": "1.0",
                    "message_type": "PREPARATION_RESULT",
                    "plan_id": plan_id,
                    "publisher_nf_instance_id": BRANCH_ID,
                    "intended_recipient_nf_instance_id": ROOT_ID,
                    "outcome": outcome.value,
                    "assigned_client_nf_instance_ids": [LEAF_A_ID, LEAF_B_ID],
                    "prepared_clients": prepared,
                    "failed_clients": failed,
                    "timed_out_client_nf_instance_ids": [],
                },
            }
        )
        path = tmp_path / f"result-{outcome.value}.tar.gz"
        path.write_bytes(b"result")
        return ValidatedHierarchyArtifact(
            metadata=ArtifactMetadata(
                key="d" * 64,
                size_bytes=path.stat().st_size,
                path=path,
                url=result_url,
            ),
            manifest={},
            contract=contract,
        )

    server.collect_hierarchy_preparation.side_effect = collect
    workspace.download_hierarchy.side_effect = download_result
    artifacts.publish_round_input.return_value = SimpleNamespace(
        url="http://root.example/round-input/0",
        digest="e" * 64,
    )
    aggregate_path = tmp_path / "root-round-global.tar.gz"
    aggregate_path.write_bytes(b"root-global")
    server.execute_hierarchy_round.return_value = SimpleNamespace(
        digest="f" * 64,
        path=aggregate_path,
        url="http://root.example/round-global/0",
    )

    def finalize_hierarchy_candidate(**kwargs):
        observer = kwargs["state_observer"]
        for state in (
            FLServerState.FINAL_VALIDATION_DISPATCH,
            FLServerState.FINAL_VALIDATION_WAITING,
            FLServerState.FINAL_VALIDATION_EVALUATING,
            FLServerState.CANDIDATE_READY,
            FLServerState.PUBLISHING,
            FLServerState.COMPLETE,
        ):
            observer(state)
        return SimpleNamespace(state=FLServerState.COMPLETE, published_model_id=2)

    server.finalize_hierarchy_candidate.side_effect = finalize_hierarchy_candidate


def test_root_admits_only_complete_ready_branch_results(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    aggregate = server.execute_hierarchy_round.return_value
    observed_round_states = []

    def execute_round(**kwargs):
        observer = kwargs["state_observer"]
        for state in (
            FLServerState.ROUND_DISPATCH,
            FLServerState.ROUND_WAITING,
            FLServerState.ROUND_EVALUATING,
            FLServerState.AGGREGATING,
        ):
            observer(state)
            snapshot = coordinator.get(REQUEST_A_ID)
            observed_round_states.append((snapshot.state, snapshot.current_round))
        return aggregate

    server.execute_hierarchy_round.side_effect = execute_round
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        admitted = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.COMPLETE, RootRequestState.FAILED},
            timeout=2,
        )

        assert admitted.state is RootRequestState.COMPLETE
        assert admitted.admission is not None
        assert admitted.admission.plan_id == admitted.plan_id
        assert admitted.admission.branches[0].branch_nf_instance_id == BRANCH_ID
        assert admitted.admission.branches[0].prepared_leaf_nf_instance_ids == (
            LEAF_A_ID,
            LEAF_B_ID,
        )
        assert registry.active() is None
        workspace.release_plan.assert_called_once_with(admitted.plan_id)
        server.close_hierarchy_training.assert_called_once_with(
            "server-process",
            retain_for_adoption=False,
        )
        assert admitted.completed_rounds == 1
        assert admitted.current_round == 0
        assert admitted.candidate_digest == "f" * 64
        assert admitted.published_model_id == 2
        assert observed_round_states == [
            (RootRequestState.ROUND_DISPATCH, 0),
            (RootRequestState.ROUND_WAITING, 0),
            (RootRequestState.ROUND_WAITING, 0),
            (RootRequestState.AGGREGATING, 0),
        ]
        round_input = artifacts.publish_round_input.call_args.kwargs
        assert round_input["epochs"] == 3
        upper_round = server.execute_hierarchy_round.call_args.kwargs
        assert upper_round["expected_subordinates"] == {
            BRANCH_ID: (LEAF_A_ID, LEAF_B_ID)
        }
        finalization = server.finalize_hierarchy_candidate.call_args.kwargs
        assert finalization["process_id"] == "server-process"
        assert finalization["validation_round"] == 1
        assert finalization["candidate"] is aggregate
        assert finalization["expected_subordinates"] == {
            BRANCH_ID: (LEAF_A_ID, LEAF_B_ID)
        }
    finally:
        coordinator.close()


def test_root_terminal_closure_invalidates_real_candidate_artifact(tmp_path):
    workspace = FLWorkspace(
        FederatedLearningSettings(workspace_root=tmp_path / "workspace"),
        ArtifactSettings(),
    )
    workspace.open()
    workspace.download_hierarchy = Mock()
    cleanup_complete = threading.Event()
    release_plan = workspace.release_plan

    def release_and_signal(plan_id):
        try:
            release_plan(plan_id)
        finally:
            cleanup_complete.set()

    workspace.release_plan = release_and_signal
    (
        coordinator,
        _resolver,
        artifacts,
        _workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path, workspace_override=workspace)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    original_finalize = server.finalize_hierarchy_candidate.side_effect
    candidate_path = (
        workspace._root
        / "server-process"
        / ROOT_ID
        / "0"
        / "ROUND_GLOBAL"
        / f"{'f' * 64}.tar.gz"
    )

    def finalize_with_owned_candidate(**kwargs):
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_bytes(b"candidate")
        workspace.claim_artifact(
            registry.active().plan_id,
            ArtifactMetadata(
                key="f" * 64,
                size_bytes=candidate_path.stat().st_size,
                path=candidate_path,
                url="http://root.example/round-global/0",
            ),
        )
        return original_finalize(**kwargs)

    server.finalize_hierarchy_candidate.side_effect = finalize_with_owned_candidate
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        completed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.COMPLETE, RootRequestState.FAILED},
            timeout=2,
        )

        assert completed.state is RootRequestState.COMPLETE
        assert completed.candidate_digest == "f" * 64
        assert cleanup_complete.wait(1)
        assert candidate_path.exists() is False
        assert workspace.open_artifact(
            "server-process",
            ROOT_ID,
            0,
            "ROUND_GLOBAL",
            "f" * 64,
        ) is None
    finally:
        coordinator.close()
        workspace.close()


def test_root_reuses_upper_process_and_feeds_previous_global_into_next_round(
    tmp_path,
):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path, round_count=2)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    initial_base = SimpleNamespace(name="initial-base")
    first_global = SimpleNamespace(name="round-0-global")
    final_global = SimpleNamespace(name="round-1-global")
    coordinator._loader.load.side_effect = [
        initial_base,
        first_global,
        final_global,
    ]
    artifacts.publish_round_input.side_effect = [
        SimpleNamespace(url="http://root.example/round-input/0", digest="e" * 64),
        SimpleNamespace(url="http://root.example/round-input/1", digest="1" * 64),
    ]
    aggregates = []
    for round_indicator, digest in enumerate(("f" * 64, "2" * 64)):
        path = tmp_path / f"root-round-global-{round_indicator}.tar.gz"
        path.write_bytes(f"root-global-{round_indicator}".encode())
        aggregates.append(
            SimpleNamespace(
                digest=digest,
                path=path,
                url=f"http://root.example/round-global/{round_indicator}",
            )
        )
    server.execute_hierarchy_round.side_effect = aggregates
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        completed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.COMPLETE, RootRequestState.FAILED},
            timeout=2,
        )

        assert completed.state is RootRequestState.COMPLETE
        assert completed.completed_rounds == 2
        assert completed.current_round == 1
        assert completed.candidate_url == ""
        assert completed.candidate_digest == "2" * 64
        publications = artifacts.publish_round_input.call_args_list
        assert [item.kwargs["base"] for item in publications] == [
            initial_base,
            first_global,
        ]
        assert [item.kwargs["round_indicator"] for item in publications] == [0, 1]
        assert [item.kwargs["epochs"] for item in publications] == [3, 3]
        executions = server.execute_hierarchy_round.call_args_list
        assert [item.kwargs["process_id"] for item in executions] == [
            "server-process",
            "server-process",
        ]
        assert [item.kwargs["round_indicator"] for item in executions] == [0, 1]
        server.start_hierarchy_preparation.assert_called_once()
    finally:
        coordinator.close()


def test_duplicate_status_queries_during_publication_do_not_restart_validation(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    publication_started = threading.Event()
    release_publication = threading.Event()
    finalization_observer = None

    def finalize(**kwargs):
        nonlocal finalization_observer
        observer = kwargs["state_observer"]
        finalization_observer = observer
        for state in (
            FLServerState.FINAL_VALIDATION_DISPATCH,
            FLServerState.FINAL_VALIDATION_WAITING,
            FLServerState.FINAL_VALIDATION_EVALUATING,
            FLServerState.CANDIDATE_READY,
            FLServerState.PUBLISHING,
        ):
            observer(state)
        publication_started.set()
        if not release_publication.wait(1):
            raise AssertionError("test did not release publication")
        observer(FLServerState.CUTOVER_PENDING)
        return SimpleNamespace(state=FLServerState.CUTOVER_PENDING)

    server.finalize_hierarchy_candidate.side_effect = finalize
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        assert publication_started.wait(1) is True

        first = coordinator.get(REQUEST_A_ID)
        second = coordinator.get(REQUEST_A_ID)

        assert first is not None
        assert second is not None
        assert first.state is RootRequestState.PUBLISHING
        assert second.state is RootRequestState.PUBLISHING
        server.finalize_hierarchy_candidate.assert_called_once()
        server.execute_hierarchy_round.assert_called_once()

        release_publication.set()
        completed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.CUTOVER_PENDING, RootRequestState.FAILED},
            timeout=1,
        )
        assert completed.state is RootRequestState.CUTOVER_PENDING
        assert registry.active() is not None
        workspace.release_plan.assert_called_once_with(completed.plan_id)
        server.close_hierarchy_training.assert_called_once_with(
            "server-process",
            retain_for_adoption=True,
        )
        with pytest.raises(RootRequestConflictError):
            coordinator.submit_manual(
                request_id=REQUEST_B_ID,
                model_family_id="ue-communication-default",
            )

        assert finalization_observer is not None
        finalization_observer(FLServerState.COMPLETE)

        assert registry.active() is None
        server.close_hierarchy_training.assert_called_with(
            "server-process",
            retain_for_adoption=False,
        )
    finally:
        release_publication.set()
        coordinator.close()


def test_validation_rejection_closes_training_and_releases_slot(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )

    def reject(**kwargs):
        observer = kwargs["state_observer"]
        for state in (
            FLServerState.FINAL_VALIDATION_DISPATCH,
            FLServerState.FINAL_VALIDATION_WAITING,
            FLServerState.FINAL_VALIDATION_EVALUATING,
            FLServerState.VALIDATION_REJECTED,
        ):
            observer(state)
        return SimpleNamespace(state=FLServerState.VALIDATION_REJECTED)

    server.finalize_hierarchy_candidate.side_effect = reject
    try:
        accepted = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        rejected = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.VALIDATION_REJECTED, RootRequestState.FAILED},
            timeout=2,
        )

        assert rejected.state is RootRequestState.VALIDATION_REJECTED
        assert registry.active() is None
        workspace.release_plan.assert_called_once_with(accepted.plan_id)
        server.close_hierarchy_training.assert_called_once_with(
            "server-process",
            retain_for_adoption=False,
        )
    finally:
        coordinator.close()


def test_terminal_root_status_is_pruned_lazily_after_retention_ttl(tmp_path):
    now = [10.0]
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        terminal_status_ttl_seconds=5,
        clock=lambda: now[0],
    )
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        completed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.COMPLETE, RootRequestState.FAILED},
            timeout=2,
        )
        assert completed.state is RootRequestState.COMPLETE

        now[0] = 16.0

        barrier = threading.Barrier(3)
        results = []

        def get_expired_status():
            barrier.wait()
            results.append(coordinator.get(REQUEST_A_ID))

        threads = [threading.Thread(target=get_expired_status) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1)

        assert results == [None, None]
        assert coordinator.requests() == ()
    finally:
        coordinator.close()


def test_root_round_failure_cancels_upper_tier_and_releases_experiment(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    server.execute_hierarchy_round.side_effect = RuntimeError(
        "required hierarchy participants terminated"
    )
    try:
        accepted = coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        failed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.FAILED},
            timeout=2,
        )

        assert failed.failure_cause == "ROUND_FAILED"
        assert failed.completed_rounds == 0
        assert failed.candidate_url == ""
        server.cancel_hierarchy_preparation.assert_called_once_with(
            "server-process",
            "required hierarchy participants terminated",
        )
        workspace.release_plan.assert_called_once_with(accepted.plan_id)
        assert registry.active() is None
    finally:
        coordinator.close()


def test_root_shutdown_wakes_round_waiter_while_branch_callback_is_pending(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    round_waiting = threading.Event()
    cancellation_received = threading.Event()
    close_completed = threading.Event()

    def execute_round(**_kwargs):
        round_waiting.set()
        if not cancellation_received.wait(1):
            raise AssertionError("Root shutdown did not cancel the upper Server process")
        raise RuntimeError("Root coordinator is closing")

    def cancel_round(_process_id, _reason):
        cancellation_received.set()

    server.execute_hierarchy_round.side_effect = execute_round
    server.cancel_hierarchy_preparation.side_effect = cancel_round
    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    assert round_waiting.wait(1) is True

    def close_root():
        coordinator.close()
        close_completed.set()

    thread = threading.Thread(target=close_root)
    thread.start()
    try:
        assert close_completed.wait(1) is True
        thread.join(timeout=1)
        snapshot = coordinator.get(REQUEST_A_ID)

        assert not thread.is_alive()
        assert snapshot.state is RootRequestState.FAILED
        assert snapshot.failure_cause == "SHUTDOWN"
        assert cancellation_received.is_set()
        assert any(
            call.args == ("server-process", "Root coordinator is closing")
            for call in server.cancel_hierarchy_preparation.call_args_list
        )
        assert registry.active() is None
    finally:
        cancellation_received.set()
        thread.join(timeout=1)


def test_root_shutdown_after_final_aggregate_fences_finalization(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    aggregate = server.execute_hierarchy_round.return_value
    aggregate_ready = threading.Event()
    allow_return = threading.Event()

    def finish_aggregate(**_kwargs):
        aggregate_ready.set()
        assert allow_return.wait(1)
        return aggregate

    server.execute_hierarchy_round.side_effect = finish_aggregate
    server.cancel_hierarchy_preparation.side_effect = (
        lambda _process_id, _reason: allow_return.set()
    )
    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    assert aggregate_ready.wait(1)
    thread = threading.Thread(target=coordinator.close)
    thread.start()
    thread.join(timeout=1)

    try:
        assert thread.is_alive() is False
        snapshot = coordinator.get(REQUEST_A_ID)
        assert snapshot.state is RootRequestState.FAILED
        assert snapshot.failure_cause == "SHUTDOWN"
        server.finalize_hierarchy_candidate.assert_not_called()
        assert registry.active() is None
    finally:
        allow_return.set()
        thread.join(timeout=1)


def test_root_shutdown_during_final_validation_fences_publication(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.READY,
    )
    publication = Mock()
    actual_server = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path / "server"),
        FLServerSettings(round_timeout_seconds=30),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
        publication=publication,
    )
    actual_server._patch_validation = Mock()
    actual_server._validate_hierarchy_candidate = Mock()
    actual_process = FLProcess(
        process_id="server-process",
        intent=None,
        state=FLServerState.READY,
        participants=[
            FLParticipant(
                scope=Mock(),
                candidate=Mock(target=Mock(nf_instance_id=BRANCH_ID)),
                notification_correlation_id="validation-branch-a",
            )
        ],
        hierarchy_plan_id="11111111-1111-4111-8111-111111111111",
        hierarchy_family_key="ue-communication-default",
    )
    actual_server._processes[actual_process.process_id] = actual_process
    validation_waiting = threading.Event()
    close_completed = threading.Event()
    original_finalize = actual_server.finalize_hierarchy_candidate

    def finalize(**kwargs):
        observer = kwargs["state_observer"]

        def observe(state):
            observer(state)
            if state is FLServerState.FINAL_VALIDATION_WAITING:
                validation_waiting.set()

        return original_finalize(**{**kwargs, "state_observer": observe})

    server.finalize_hierarchy_candidate.side_effect = finalize
    server.cancel_hierarchy_preparation.side_effect = (
        actual_server.cancel_hierarchy_preparation
    )
    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    assert validation_waiting.wait(1) is True

    thread = threading.Thread(
        target=lambda: (coordinator.close(), close_completed.set())
    )
    thread.start()
    try:
        assert close_completed.wait(1) is True
        thread.join(timeout=1)
        snapshot = coordinator.get(REQUEST_A_ID)

        assert not thread.is_alive()
        assert snapshot.state is RootRequestState.FAILED
        assert snapshot.failure_cause == "SHUTDOWN"
        publication.publish.assert_not_called()
        assert actual_process.state is FLServerState.FAILED
        assert registry.active() is None
    finally:
        thread.join(timeout=1)
        actual_server.close()


def test_root_validates_failure_result_before_rejecting_admission(tmp_path):
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(tmp_path)
    _configure_branch_result(
        tmp_path=tmp_path,
        artifacts=artifacts,
        workspace=workspace,
        server=server,
        registry=registry,
        outcome=PreparationOutcome.FAILED,
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

        assert failed.failure_cause == "ADMISSION_REJECTED"
        workspace.download_hierarchy.assert_called_once()
        assert registry.active() is None
    finally:
        coordinator.close()
