import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from py_mtlf.config import FederatedStrategySettings, FLServerSettings
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy_discovery import ResolvedHierarchyNode
from py_mtlf.core.fl_root import FLRootCoordinator, RootRequestState
from py_mtlf.core.fl_server import (
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
)
from py_mtlf.core.fl_topology import StaticTopologyPlanner
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.wire.ml_model import MLModelAdrf
from py_mtlf.wire.ml_model_training import FlTopologyReport, NwdafMLModelTrainNotif
from py_mtlf.wire.private import SelectedTarget

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_ID = "00000000-0000-4000-8000-000000000010"
BRANCH_B_ID = "00000000-0000-4000-8000-000000000020"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
LEAF_C_ID = "00000000-0000-4000-8000-000000000201"
LEAF_D_ID = "00000000-0000-4000-8000-000000000202"
REQUEST_A_ID = "00000000-0000-4000-8000-000000000701"
REQUEST_B_ID = "00000000-0000-4000-8000-000000000702"


def test_protocol_root_requires_round_model_distribution_owner():
    with pytest.raises(ValueError, match="requires round model distribution"):
        FLRootCoordinator(
            strategy=Mock(),
            server_settings=Mock(),
            planner=Mock(),
            resolver=Mock(),
            nwdaf_context=Mock(),
            catalog=Mock(),
            artifact_service=Mock(),
            workspace=Mock(),
            server=Mock(),
            policy=Mock(),
            experiments=Mock(),
            round_model_distribution=None,
        )


def root_coordinator(
    tmp_path: Path,
    *,
    resolver_side_effect=None,
    round_count: int = 1,
    terminal_status_ttl_seconds: int = 3600,
    clock=None,
    workspace_override=None,
    round_model_distribution=None,
    ml_event: str = "X_IMAGE_CLASSIFICATION",
    model_interoperability: str = "pymtlf-image-classification-mnist",
    topology_text: str | None = None,
):
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        topology_text
        or f"""
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
                    ml_analytics_ids=(ml_event,),
                fl_capability_type=FLCapabilityType.SERVER,
            ),
        ),
    )
    model = SimpleNamespace(
        model_id=1,
        artifact=SimpleNamespace(key="a" * 64, url="http://root.example/base"),
        descriptor=SimpleNamespace(
            family_id="ue-communication-default",
            event=ml_event,
            event_filter={"networkArea": {"tais": []}},
            target_ue=None,
            model_interoperability=model_interoperability,
        ),
    )
    catalog = Mock()
    catalog.current.side_effect = lambda family: (
        model if family == "ue-communication-default" else None
    )
    artifact_service = Mock()
    workspace = workspace_override or Mock()
    registry = FLExperimentRegistry()
    server = Mock()

    def start_protocol_preparation(**kwargs):
        plan_id = kwargs.get("plan_id", kwargs.get("ml_correlation_id"))
        registry.attach_server(
            kwargs["reservation_id"],
            plan_id,
            "server-process",
        )
        return SimpleNamespace(process_id="server-process")

    server.start_protocol_preparation.side_effect = start_protocol_preparation
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
        round_model_distribution=round_model_distribution or Mock(),
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


def test_protocol_root_dispatches_model_free_recursive_preparation(tmp_path):
    distribution = Mock()
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_model_distribution=distribution,
        ml_event="X_IMAGE_CLASSIFICATION",
        model_interoperability="pymtlf-image-classification-mnist",
    )

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

        assert waiting.state is RootRequestState.PREPARATION_WAITING
        kwargs = server.start_protocol_preparation.call_args.kwargs
        assert kwargs["ml_correlation_id"] == accepted.plan_id
        assert UUID(kwargs["ml_correlation_id"]).version == 4
        assert kwargs["ml_event"] == "X_IMAGE_CLASSIFICATION"
        assert kwargs["model_interoperability"] == (
            "pymtlf-image-classification-mnist"
        )
        branch = kwargs["targets"][0].topology
        assert branch.nf_instance_id == BRANCH_ID
        assert tuple(child.nf_instance_id for child in branch.children) == (
            LEAF_A_ID,
            LEAF_B_ID,
        )
        assert branch.report_after.unit == "round"
        assert all(child.report_after.unit == "epoch" for child in branch.children)
        artifacts.publish_branch_assignment.assert_not_called()
        distribution.store.assert_not_called()
    finally:
        coordinator.close()


def test_protocol_root_dispatches_two_explicit_branch_subtrees(tmp_path):
    topology_text = f"""
version: 1
admission:
  mode: complete_required
branches:
  - nf_instance_id: {BRANCH_ID}
    leaves:
      - nf_instance_id: {LEAF_A_ID}
      - nf_instance_id: {LEAF_B_ID}
  - nf_instance_id: {BRANCH_B_ID}
    leaves:
      - nf_instance_id: {LEAF_C_ID}
      - nf_instance_id: {LEAF_D_ID}
""".strip() + "\n"
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
    ) = root_coordinator(
        tmp_path,
        round_model_distribution=Mock(),
        ml_event="X_IMAGE_CLASSIFICATION",
        model_interoperability="pymtlf-image-classification-mnist",
        topology_text=topology_text,
    )

    try:
        coordinator.submit_manual(
            request_id=REQUEST_A_ID,
            model_family_id="ue-communication-default",
        )
        waiting = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.PREPARATION_WAITING, RootRequestState.FAILED},
            timeout=2,
        )

        assert waiting.state is RootRequestState.PREPARATION_WAITING
        targets = server.start_protocol_preparation.call_args.kwargs["targets"]
        assert tuple(target.participant_nf_instance_id for target in targets) == (
            BRANCH_ID,
            BRANCH_B_ID,
        )
        assert tuple(
            child.nf_instance_id for child in targets[0].topology.children
        ) == (LEAF_A_ID, LEAF_B_ID)
        assert tuple(
            child.nf_instance_id for child in targets[1].topology.children
        ) == (LEAF_C_ID, LEAF_D_ID)
    finally:
        coordinator.close()


def test_protocol_root_stores_before_round_dispatch_and_cleans_record(tmp_path):
    topology_text = f"""
version: 1
admission:
  mode: complete_required
branches:
  - nf_instance_id: {BRANCH_ID}
    leaves:
      - nf_instance_id: {LEAF_A_ID}
      - nf_instance_id: {LEAF_B_ID}
  - nf_instance_id: {BRANCH_B_ID}
    leaves:
      - nf_instance_id: {LEAF_C_ID}
      - nf_instance_id: {LEAF_D_ID}
""".strip() + "\n"
    distribution = Mock()
    stored = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    distribution.store.return_value = stored
    (
        coordinator,
        _resolver,
        artifacts,
        workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_model_distribution=distribution,
        ml_event="X_IMAGE_CLASSIFICATION",
        model_interoperability="pymtlf-image-classification-mnist",
        topology_text=topology_text,
    )
    round_path = tmp_path / "round.tar.gz"
    round_path.write_bytes(b"round")
    aggregate_path = tmp_path / "aggregate.tar.gz"
    aggregate_path.write_bytes(b"aggregate")
    round_input = SimpleNamespace(
        digest="b" * 64,
        path=round_path,
        url="http://root.example/round",
    )
    aggregate = SimpleNamespace(
        digest="c" * 64,
        path=aggregate_path,
        url="http://branch.example/aggregate",
    )
    handoff = SimpleNamespace(
        digest=aggregate.digest,
        path=tmp_path / "root-handoff.tar.gz",
        url="http://root.example/final-handoff",
    )
    workspace.republish_validation_candidate.return_value = handoff
    artifacts.publish_round_input.return_value = round_input
    process_id = None

    def collect_protocol(process):
        nonlocal process_id
        process_id = process
        reports = {
            BRANCH_ID: (LEAF_A_ID, LEAF_B_ID),
            BRANCH_B_ID: (LEAF_C_ID, LEAF_D_ID),
        }
        return HierarchyPreparationCollection(
            process_id=process,
            plan_id=server.start_protocol_preparation.call_args.kwargs[
                "ml_correlation_id"
            ],
            participants=tuple(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=branch_id,
                    resource_location=f"http://{branch_id}.example/subscriptions/1",
                    notification=NwdafMLModelTrainNotif(
                        notifCorreId=f"branch-callback-{index}",
                        mlCorreId=server.start_protocol_preparation.call_args.kwargs[
                            "ml_correlation_id"
                        ],
                        **{
                            "x-flTopologyReport": FlTopologyReport.model_validate(
                                {
                                    "nfInstanceId": branch_id,
                                    "children": [
                                        {
                                            "nfInstanceId": leaf_id,
                                            "status": "ACTIVE",
                                            "statusTimestamp": "2026-09-05T12:00:00Z",
                                        }
                                        for leaf_id in leaf_ids
                                    ],
                                }
                            )
                        },
                    ),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                )
                for index, (branch_id, leaf_ids) in enumerate(
                    reports.items(),
                    start=1,
                )
            ),
            timed_out_participant_nf_instance_ids=(),
        )

    server.collect_hierarchy_preparation.side_effect = collect_protocol
    def execute_round(**_kwargs):
        assert distribution.store.called
        return aggregate

    server.execute_hierarchy_round.side_effect = execute_round
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
    assert completed.candidate_url == handoff.url
    assert completed.candidate_digest == handoff.digest
    assert process_id is not None
    store_kwargs = distribution.store.call_args.kwargs
    assert store_kwargs["ml_correlation_id"] == completed.plan_id
    assert store_kwargs["round_indicator"] == 0
    assert store_kwargs["allowed_consumer_ids"] == (
        BRANCH_ID,
        BRANCH_B_ID,
        LEAF_A_ID,
        LEAF_B_ID,
        LEAF_C_ID,
        LEAF_D_ID,
    )
    round_kwargs = server.execute_hierarchy_round.call_args.kwargs
    assert round_kwargs["expected_subordinates"] == {
        BRANCH_ID: (LEAF_A_ID, LEAF_B_ID),
        BRANCH_B_ID: (LEAF_C_ID, LEAF_D_ID),
    }
    assert round_kwargs["round_input_model"].model_unique_id == 77
    assert round_kwargs["round_input_model"].model_adrf == stored.wire_reference
    distribution.cleanup.assert_called_once_with(completed.plan_id, 0)
    workspace.republish_validation_candidate.assert_called_once()
    server.close_hierarchy_training.assert_called_once_with(process_id)
    workspace.release_plan.assert_not_called()
    coordinator.close()
    workspace.release_plan.assert_called_once_with(completed.plan_id)


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
