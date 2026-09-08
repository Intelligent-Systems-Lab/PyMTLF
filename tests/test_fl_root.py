import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from py_mtlf.config import FLServerSettings
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_root import FLRootCoordinator, RootRequestState
from py_mtlf.core.fl_server import (
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
    HierarchyRoundOutcome,
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
BRANCH_REPLACEMENT_ID = "00000000-0000-4000-8000-000000000011"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
LEAF_C_ID = "00000000-0000-4000-8000-000000000201"
LEAF_D_ID = "00000000-0000-4000-8000-000000000202"
REQUEST_A_ID = "00000000-0000-4000-8000-000000000701"
REQUEST_B_ID = "00000000-0000-4000-8000-000000000702"


def hierarchy_topology(*groups: tuple[str, tuple[str, ...]]) -> str:
    lines = [
        "admission:",
        "  mode: complete_required",
        "policy:",
        "  allow_additional_candidates: false",
        "  additional_candidate_priority: 0",
        "  selection_method: priority",
        f"  min_available_nodes: {len(groups)}",
        "  fraction_train: 1.0",
        f"  min_train_nodes: {len(groups)}",
        "  accept_failures: false",
        "  min_completion_rate: 1.0",
        "strategy:",
        "  method: fedProx",
        "  aggregation: sampleWeighted",
        "  method_parameters: {proximal_mu: 0.01}",
        "branch_groups:",
    ]
    for branch_id, leaf_ids in groups:
        lines.extend(
            [
                "  - branches:",
                f"      - nf_instance_id: {branch_id}",
                "        priority: 100",
                "        report_after: {count: 1, unit: round}",
                "    policy:",
                "      allow_additional_candidates: false",
                "      additional_candidate_priority: 0",
                "      selection_method: priority",
                f"      min_available_nodes: {len(leaf_ids)}",
                "      fraction_train: 1.0",
                f"      min_train_nodes: {len(leaf_ids)}",
                "      accept_failures: false",
                "      min_completion_rate: 1.0",
                "    strategy:",
                "      method: fedProx",
                "      aggregation: sampleWeighted",
                "      method_parameters: {proximal_mu: 0.01}",
                "    leaves:",
            ]
        )
        for leaf_id in leaf_ids:
            lines.extend(
                [
                    f"      - nf_instance_id: {leaf_id}",
                    "        priority: 100",
                    "        report_after: {count: 3, unit: epoch}",
                ]
            )
    return "\n".join(lines) + "\n"


def branch_replacement_topology(*, root_minimum: int = 2) -> str:
    return f"""
admission:
  mode: complete_required
policy: &root_policy
  allow_additional_candidates: false
  additional_candidate_priority: 0
  selection_method: priority
  min_available_nodes: {root_minimum}
  fraction_train: 1.0
  min_train_nodes: {root_minimum}
  accept_failures: true
  min_completion_rate: 0.5
strategy: &strategy
  method: fedProx
  aggregation: sampleWeighted
  method_parameters: {{proximal_mu: 0.01}}
branch_groups:
  - branches:
      - nf_instance_id: {BRANCH_ID}
        priority: 100
        report_after: {{count: 1, unit: round}}
      - nf_instance_id: {BRANCH_REPLACEMENT_ID}
        priority: 50
        report_after: {{count: 1, unit: round}}
    policy: &leaf_policy
      allow_additional_candidates: false
      additional_candidate_priority: 0
      selection_method: priority
      min_available_nodes: 2
      fraction_train: 1.0
      min_train_nodes: 2
      accept_failures: false
      min_completion_rate: 1.0
    strategy: *strategy
    leaves:
      - nf_instance_id: {LEAF_A_ID}
        priority: 100
        report_after: {{count: 2, unit: epoch}}
      - nf_instance_id: {LEAF_B_ID}
        priority: 90
        report_after: {{count: 2, unit: epoch}}
  - branches:
      - nf_instance_id: {BRANCH_B_ID}
        priority: 100
        report_after: {{count: 1, unit: round}}
    policy: *leaf_policy
    strategy: *strategy
    leaves:
      - nf_instance_id: {LEAF_C_ID}
        priority: 100
        report_after: {{count: 2, unit: epoch}}
      - nf_instance_id: {LEAF_D_ID}
        priority: 90
        report_after: {{count: 2, unit: epoch}}
""".strip() + "\n"


def test_protocol_root_requires_round_model_distribution_owner():
    with pytest.raises(ValueError, match="requires round model distribution"):
        FLRootCoordinator(
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
        or hierarchy_topology((BRANCH_ID, (LEAF_B_ID, LEAF_A_ID))),
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


def branch_preparation_outcome(
    server,
    branch_id,
    leaf_ids,
    callback_id,
    *,
    resource_location=None,
    failure="",
):
    plan_id = server.start_protocol_preparation.call_args.kwargs["ml_correlation_id"]
    notification = None
    if not failure:
        notification = NwdafMLModelTrainNotif(
            notifCorreId=callback_id,
            mlCorreId=plan_id,
            **{
                "x-flTopologyReport": FlTopologyReport.model_validate(
                    {
                        "nfInstanceId": branch_id,
                        "children": [
                            {
                                "nfInstanceId": leaf_id,
                                "status": "ACTIVE",
                                "statusTimestamp": "2026-09-08T12:00:00Z",
                            }
                            for leaf_id in leaf_ids
                        ],
                    }
                )
            },
        )
    return HierarchyParticipantPreparationOutcome(
        participant_nf_instance_id=branch_id,
        resource_location=(
            resource_location
            if resource_location is not None
            else f"http://{branch_id}.example/subscriptions/1"
        ),
        notification=notification,
        failure=failure,
        delay_extensions=0,
        granted_extension_seconds=0,
    )


def two_group_preparation_collection(server, process_id):
    plan_id = server.start_protocol_preparation.call_args.kwargs["ml_correlation_id"]
    return HierarchyPreparationCollection(
        process_id=process_id,
        plan_id=plan_id,
        participants=(
            branch_preparation_outcome(
                server,
                BRANCH_ID,
                (LEAF_A_ID, LEAF_B_ID),
                "callback-primary",
            ),
            branch_preparation_outcome(
                server,
                BRANCH_B_ID,
                (LEAF_C_ID, LEAF_D_ID),
                "callback-secondary",
            ),
        ),
        timed_out_participant_nf_instance_ids=(),
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
    topology_text = hierarchy_topology(
        (BRANCH_ID, (LEAF_A_ID, LEAF_B_ID)),
        (BRANCH_B_ID, (LEAF_C_ID, LEAF_D_ID)),
    )
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


def test_protocol_root_initial_discovery_uses_the_next_branch_candidate(tmp_path):
    def fail_primary(nf_instance_id, _role):
        if nf_instance_id == BRANCH_ID:
            raise HierarchyDiscoveryError("primary Branch is unavailable")

    (
        coordinator,
        resolver,
        _artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        resolver_side_effect=fail_primary,
        round_model_distribution=Mock(),
        topology_text=branch_replacement_topology(),
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
        assert [
            call.kwargs["nf_instance_id"] for call in resolver.resolve.call_args_list
        ] == [BRANCH_ID, BRANCH_REPLACEMENT_ID, BRANCH_B_ID]
        assert tuple(
            target.participant_nf_instance_id
            for target in server.start_protocol_preparation.call_args.kwargs["targets"]
        ) == (BRANCH_REPLACEMENT_ID, BRANCH_B_ID)
    finally:
        coordinator.close()


def test_protocol_root_initial_preparation_failure_uses_next_branch_candidate(
    tmp_path,
):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    (
        coordinator,
        resolver,
        artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(),
    )
    round_path = tmp_path / "fallback-round.tar.gz"
    aggregate_path = tmp_path / "fallback-aggregate.tar.gz"
    round_path.write_bytes(b"round")
    aggregate_path.write_bytes(b"aggregate")
    artifacts.publish_round_input.return_value = SimpleNamespace(
        digest="2" * 64,
        path=round_path,
        url="http://root.example/fallback-round",
    )
    aggregate = SimpleNamespace(
        digest="5" * 64,
        path=aggregate_path,
        url="http://root.example/fallback-aggregate",
    )
    collection_count = 0

    def collect(process_id):
        nonlocal collection_count
        collection_count += 1
        participants = [
            branch_preparation_outcome(
                server,
                BRANCH_B_ID,
                (LEAF_C_ID, LEAF_D_ID),
                "callback-secondary",
            )
        ]
        if collection_count == 1:
            participants.insert(
                0,
                branch_preparation_outcome(
                    server,
                    BRANCH_ID,
                    (LEAF_A_ID, LEAF_B_ID),
                    "callback-primary",
                    failure="preparation deadline expired",
                ),
            )
        else:
            participants.insert(
                0,
                branch_preparation_outcome(
                    server,
                    BRANCH_REPLACEMENT_ID,
                    (LEAF_A_ID, LEAF_B_ID),
                    "callback-replacement",
                ),
            )
        return HierarchyPreparationCollection(
            process_id=process_id,
            plan_id=server.start_protocol_preparation.call_args.kwargs[
                "ml_correlation_id"
            ],
            participants=tuple(participants),
            timed_out_participant_nf_instance_ids=(),
        )

    server.collect_hierarchy_preparation.side_effect = collect
    server.remove_protocol_participant.return_value = ""
    server.execute_hierarchy_round.return_value = HierarchyRoundOutcome(
        aggregate=aggregate,
        accepted=True,
        selected_participant_nf_instance_ids=(
            BRANCH_REPLACEMENT_ID,
            BRANCH_B_ID,
        ),
        successful_participant_nf_instance_ids=(
            BRANCH_REPLACEMENT_ID,
            BRANCH_B_ID,
        ),
        failed_participant_nf_instance_ids=(),
    )

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    completed = coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.COMPLETE, RootRequestState.FAILED},
        timeout=3,
    )

    assert completed.state is RootRequestState.COMPLETE
    assert collection_count == 2
    assert server.add_protocol_preparation_targets.call_count == 1
    assert server.add_protocol_preparation_targets.call_args.kwargs[
        "targets"
    ][0].participant_nf_instance_id == BRANCH_REPLACEMENT_ID
    assert server.remove_protocol_participant.call_args_list[0].args[1] == BRANCH_ID
    assert resolver.resolve.call_args_list[-1].kwargs["nf_instance_id"] == (
        BRANCH_REPLACEMENT_ID
    )
    assert server.execute_hierarchy_round.call_args.kwargs[
        "selected_participant_nf_instance_ids"
    ] == (BRANCH_REPLACEMENT_ID, BRANCH_B_ID)
    coordinator.close()


def test_protocol_root_stores_before_round_dispatch_and_cleans_record(tmp_path):
    topology_text = hierarchy_topology(
        (BRANCH_ID, (LEAF_A_ID, LEAF_B_ID)),
        (BRANCH_B_ID, (LEAF_C_ID, LEAF_D_ID)),
    )
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
        return HierarchyRoundOutcome(
            aggregate=aggregate,
            accepted=True,
            selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
            successful_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
            failed_participant_nf_instance_ids=(),
        )

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
    assert completed.candidate_url == aggregate.url
    assert completed.candidate_digest == aggregate.digest
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
    workspace.republish_validation_candidate.assert_not_called()
    server.close_hierarchy_training.assert_called_once_with(process_id)
    workspace.release_plan.assert_not_called()
    coordinator.close()
    workspace.release_plan.assert_called_once_with(completed.plan_id)


def test_protocol_root_replaces_failed_branch_without_retained_result(tmp_path):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    (
        coordinator,
        resolver,
        artifacts,
        workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_count=2,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(),
    )
    round_paths = [tmp_path / f"round-{index}.tar.gz" for index in range(2)]
    aggregate_paths = [tmp_path / f"aggregate-{index}.tar.gz" for index in range(2)]
    for path in (*round_paths, *aggregate_paths):
        path.write_bytes(path.name.encode())
    round_inputs = [
        SimpleNamespace(
            digest=str(index + 2) * 64,
            path=path,
            url=f"http://root.example/round-{index}",
        )
        for index, path in enumerate(round_paths)
    ]
    aggregates = [
        SimpleNamespace(
            digest=str(index + 5) * 64,
            path=path,
            url=f"http://root.example/aggregate-{index}",
        )
        for index, path in enumerate(aggregate_paths)
    ]
    artifacts.publish_round_input.side_effect = round_inputs
    workspace.republish_validation_candidate.return_value = SimpleNamespace(
        digest="9" * 64,
        path=tmp_path / "handoff.tar.gz",
        url="http://root.example/handoff",
    )

    def report(branch_id, leaf_ids):
        return NwdafMLModelTrainNotif(
            notifCorreId=f"callback-{branch_id}",
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
                                "statusTimestamp": "2026-09-08T12:00:00Z",
                            }
                            for leaf_id in leaf_ids
                        ],
                    }
                )
            },
        )

    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        HierarchyPreparationCollection(
            process_id=process_id,
            plan_id=server.start_protocol_preparation.call_args.kwargs[
                "ml_correlation_id"
            ],
            participants=(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_ID,
                    resource_location="http://primary.example/subscriptions/1",
                    notification=report(BRANCH_ID, (LEAF_A_ID, LEAF_B_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_B_ID,
                    resource_location="http://branch-b.example/subscriptions/1",
                    notification=report(BRANCH_B_ID, (LEAF_C_ID, LEAF_D_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
            ),
            timed_out_participant_nf_instance_ids=(),
        )
    )
    server.execute_hierarchy_round.side_effect = (
        HierarchyRoundOutcome(
            aggregate=aggregates[0],
            accepted=True,
            selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
            successful_participant_nf_instance_ids=(BRANCH_B_ID,),
            failed_participant_nf_instance_ids=(BRANCH_ID,),
        ),
        HierarchyRoundOutcome(
            aggregate=aggregates[1],
            accepted=True,
            selected_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            successful_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            failed_participant_nf_instance_ids=(),
        ),
    )
    server.remove_protocol_participant.return_value = ""
    server.prepare_protocol_replacement_target.side_effect = lambda **_kwargs: (
        HierarchyParticipantPreparationOutcome(
            participant_nf_instance_id=BRANCH_REPLACEMENT_ID,
            resource_location="http://replacement.example/subscriptions/2",
            notification=report(
                BRANCH_REPLACEMENT_ID,
                (LEAF_A_ID, LEAF_B_ID),
            ),
            failure="",
            delay_extensions=0,
            granted_extension_seconds=0,
        )
    )

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    completed = coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.COMPLETE, RootRequestState.FAILED},
        timeout=3,
    )

    assert completed.state is RootRequestState.COMPLETE
    assert completed.completed_rounds == 2
    replacement_call = server.prepare_protocol_replacement_target.call_args.kwargs
    assert replacement_call["process_id"] == "server-process"
    assert replacement_call["target"].participant_nf_instance_id == (
        BRANCH_REPLACEMENT_ID
    )
    assert replacement_call["target"].topology.retained_result_request is None
    assert tuple(
        child.nf_instance_id
        for child in replacement_call["target"].topology.children
    ) == (LEAF_A_ID, LEAF_B_ID)
    assert server.execute_hierarchy_round.call_args_list[1].kwargs[
        "selected_participant_nf_instance_ids"
    ] == (BRANCH_REPLACEMENT_ID, BRANCH_B_ID)
    assert resolver.resolve.call_args_list[-1].kwargs["nf_instance_id"] == (
        BRANCH_REPLACEMENT_ID
    )
    assert distribution.cleanup.call_count == 2
    coordinator.close()


def test_protocol_root_rejected_attempt_reuses_last_committed_model_and_final_aggregate(
    tmp_path,
):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
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
        round_count=1,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(),
    )
    round_paths = [tmp_path / f"rejected-round-{index}.tar.gz" for index in range(2)]
    aggregate_path = tmp_path / "accepted-aggregate.tar.gz"
    for path in (*round_paths, aggregate_path):
        path.write_bytes(path.name.encode())
    artifacts.publish_round_input.side_effect = [
        SimpleNamespace(
            digest=str(index + 2) * 64,
            path=path,
            url=f"http://root.example/rejected-round-{index}",
        )
        for index, path in enumerate(round_paths)
    ]
    aggregate = SimpleNamespace(
        digest="8" * 64,
        path=aggregate_path,
        url="http://root.example/accepted-aggregate",
    )

    def report(branch_id, leaf_ids):
        return NwdafMLModelTrainNotif(
            notifCorreId=f"callback-{branch_id}",
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
                                "statusTimestamp": "2026-09-08T12:00:00Z",
                            }
                            for leaf_id in leaf_ids
                        ],
                    }
                )
            },
        )

    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        HierarchyPreparationCollection(
            process_id=process_id,
            plan_id=server.start_protocol_preparation.call_args.kwargs[
                "ml_correlation_id"
            ],
            participants=(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_ID,
                    resource_location="http://primary.example/subscriptions/1",
                    notification=report(BRANCH_ID, (LEAF_A_ID, LEAF_B_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_B_ID,
                    resource_location="http://branch-b.example/subscriptions/1",
                    notification=report(BRANCH_B_ID, (LEAF_C_ID, LEAF_D_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
            ),
            timed_out_participant_nf_instance_ids=(),
        )
    )
    server.execute_hierarchy_round.side_effect = (
        HierarchyRoundOutcome(
            aggregate=None,
            accepted=False,
            selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
            successful_participant_nf_instance_ids=(BRANCH_B_ID,),
            failed_participant_nf_instance_ids=(BRANCH_ID,),
        ),
        HierarchyRoundOutcome(
            aggregate=aggregate,
            accepted=True,
            selected_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            successful_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            failed_participant_nf_instance_ids=(),
        ),
    )
    server.remove_protocol_participant.return_value = ""
    server.prepare_protocol_replacement_target.side_effect = lambda **_kwargs: (
        HierarchyParticipantPreparationOutcome(
            participant_nf_instance_id=BRANCH_REPLACEMENT_ID,
            resource_location="http://replacement.example/subscriptions/2",
            notification=report(
                BRANCH_REPLACEMENT_ID,
                (LEAF_A_ID, LEAF_B_ID),
            ),
            failure="",
            delay_extensions=0,
            granted_extension_seconds=0,
        )
    )

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    completed = coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.COMPLETE, RootRequestState.FAILED},
        timeout=3,
    )

    assert completed.state is RootRequestState.COMPLETE
    assert completed.completed_rounds == 1
    assert completed.candidate_url == aggregate.url
    assert completed.candidate_digest == aggregate.digest
    assert [
        call.kwargs["round_indicator"]
        for call in server.execute_hierarchy_round.call_args_list
    ] == [0, 1]
    first_source = artifacts.publish_round_input.call_args_list[0].kwargs["base"]
    second_source = artifacts.publish_round_input.call_args_list[1].kwargs["base"]
    assert second_source is first_source
    workspace.republish_validation_candidate.assert_not_called()
    assert [call.args[1] for call in distribution.cleanup.call_args_list] == [0, 1]
    coordinator.close()


def test_protocol_root_continues_while_replacement_prepares_and_adopts_next_cohort(
    tmp_path,
):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
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
        round_count=3,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(root_minimum=1),
    )
    round_inputs = []
    aggregates = []
    for index in range(3):
        round_path = tmp_path / f"parallel-round-{index}.tar.gz"
        aggregate_path = tmp_path / f"parallel-aggregate-{index}.tar.gz"
        round_path.write_bytes(round_path.name.encode())
        aggregate_path.write_bytes(aggregate_path.name.encode())
        round_inputs.append(
            SimpleNamespace(
                digest=str(index + 2) * 64,
                path=round_path,
                url=f"http://root.example/parallel-round-{index}",
            )
        )
        aggregates.append(
            SimpleNamespace(
                digest=str(index + 5) * 64,
                path=aggregate_path,
                url=f"http://root.example/parallel-aggregate-{index}",
            )
        )
    artifacts.publish_round_input.side_effect = round_inputs
    workspace.republish_validation_candidate.return_value = SimpleNamespace(
        digest="9" * 64,
        path=tmp_path / "parallel-handoff.tar.gz",
        url="http://root.example/parallel-handoff",
    )

    def report(branch_id, leaf_ids):
        return NwdafMLModelTrainNotif(
            notifCorreId=f"callback-{branch_id}",
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
                                "statusTimestamp": "2026-09-08T12:00:00Z",
                            }
                            for leaf_id in leaf_ids
                        ],
                    }
                )
            },
        )

    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        HierarchyPreparationCollection(
            process_id=process_id,
            plan_id=server.start_protocol_preparation.call_args.kwargs[
                "ml_correlation_id"
            ],
            participants=(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_ID,
                    resource_location="http://primary.example/subscriptions/1",
                    notification=report(BRANCH_ID, (LEAF_A_ID, LEAF_B_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=BRANCH_B_ID,
                    resource_location="http://branch-b.example/subscriptions/1",
                    notification=report(BRANCH_B_ID, (LEAF_C_ID, LEAF_D_ID)),
                    failure="",
                    delay_extensions=0,
                    granted_extension_seconds=0,
                ),
            ),
            timed_out_participant_nf_instance_ids=(),
        )
    )
    second_round_dispatched = threading.Event()
    allow_second_round_completion = threading.Event()
    replacement_started = threading.Event()
    allow_replacement_completion = threading.Event()
    replacement_completed = threading.Event()
    execution_count = 0

    def execute_round(**kwargs):
        nonlocal execution_count
        invocation = execution_count
        execution_count += 1
        if invocation == 0:
            return HierarchyRoundOutcome(
                aggregate=aggregates[0],
                accepted=True,
                selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
                successful_participant_nf_instance_ids=(BRANCH_B_ID,),
                failed_participant_nf_instance_ids=(BRANCH_ID,),
            )
        if invocation == 1:
            assert kwargs["selected_participant_nf_instance_ids"] == (BRANCH_B_ID,)
            second_round_dispatched.set()
            assert allow_second_round_completion.wait(2)
            return HierarchyRoundOutcome(
                aggregate=aggregates[1],
                accepted=True,
                selected_participant_nf_instance_ids=(BRANCH_B_ID,),
                successful_participant_nf_instance_ids=(BRANCH_B_ID,),
                failed_participant_nf_instance_ids=(),
            )
        return HierarchyRoundOutcome(
            aggregate=aggregates[2],
            accepted=True,
            selected_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            successful_participant_nf_instance_ids=(
                BRANCH_REPLACEMENT_ID,
                BRANCH_B_ID,
            ),
            failed_participant_nf_instance_ids=(),
        )

    def prepare_replacement(**_kwargs):
        replacement_started.set()
        assert allow_replacement_completion.wait(2)
        replacement_completed.set()
        return HierarchyParticipantPreparationOutcome(
            participant_nf_instance_id=BRANCH_REPLACEMENT_ID,
            resource_location="http://replacement.example/subscriptions/2",
            notification=report(
                BRANCH_REPLACEMENT_ID,
                (LEAF_A_ID, LEAF_B_ID),
            ),
            failure="",
            delay_extensions=0,
            granted_extension_seconds=0,
        )

    server.execute_hierarchy_round.side_effect = execute_round
    server.remove_protocol_participant.return_value = ""
    server.prepare_protocol_replacement_target.side_effect = prepare_replacement

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    try:
        assert replacement_started.wait(2)
        assert second_round_dispatched.wait(2)
        assert replacement_completed.is_set() is False
        replacing = coordinator.get(REQUEST_A_ID)
        assert replacing is not None
        assert replacing.branch_groups[0].state == "BRANCH_REPLACING"
        assert replacing.branch_groups[0].active_branch_nf_instance_id == ""
        allow_replacement_completion.set()
        assert replacement_completed.wait(2)
        allow_second_round_completion.set()
        completed = coordinator.wait_for_state(
            REQUEST_A_ID,
            {RootRequestState.COMPLETE, RootRequestState.FAILED},
            timeout=3,
        )

        assert completed.state is RootRequestState.COMPLETE
        assert completed.completed_rounds == 3
        assert completed.branch_groups[0].state == "ACTIVE"
        assert (
            completed.branch_groups[0].active_branch_nf_instance_id
            == BRANCH_REPLACEMENT_ID
        )
        assert server.execute_hierarchy_round.call_args_list[2].kwargs[
            "selected_participant_nf_instance_ids"
        ] == (BRANCH_REPLACEMENT_ID, BRANCH_B_ID)
    finally:
        allow_replacement_completion.set()
        allow_second_round_completion.set()
        coordinator.close()


@pytest.mark.parametrize(
    ("root_minimum", "expected_state"),
    [
        (1, RootRequestState.COMPLETE),
        (2, RootRequestState.FAILED),
    ],
)
def test_protocol_root_candidate_exhaustion_follows_remaining_readiness(
    tmp_path,
    root_minimum,
    expected_state,
):
    def fail_replacement(nf_instance_id, _role):
        if nf_instance_id == BRANCH_REPLACEMENT_ID:
            raise HierarchyDiscoveryError("replacement Branch is unavailable")

    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    (
        coordinator,
        resolver,
        artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        resolver_side_effect=fail_replacement,
        round_count=2,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(root_minimum=root_minimum),
    )
    round_inputs = []
    aggregates = []
    for index in range(2):
        round_path = tmp_path / f"exhausted-round-{index}.tar.gz"
        aggregate_path = tmp_path / f"exhausted-aggregate-{index}.tar.gz"
        round_path.write_bytes(round_path.name.encode())
        aggregate_path.write_bytes(aggregate_path.name.encode())
        round_inputs.append(
            SimpleNamespace(
                digest=str(index + 2) * 64,
                path=round_path,
                url=f"http://root.example/exhausted-round-{index}",
            )
        )
        aggregates.append(
            SimpleNamespace(
                digest=str(index + 5) * 64,
                path=aggregate_path,
                url=f"http://root.example/exhausted-aggregate-{index}",
            )
        )
    artifacts.publish_round_input.side_effect = round_inputs
    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        two_group_preparation_collection(server, process_id)
    )
    server.execute_hierarchy_round.side_effect = (
        HierarchyRoundOutcome(
            aggregate=aggregates[0],
            accepted=True,
            selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
            successful_participant_nf_instance_ids=(BRANCH_B_ID,),
            failed_participant_nf_instance_ids=(BRANCH_ID,),
        ),
        HierarchyRoundOutcome(
            aggregate=aggregates[1],
            accepted=True,
            selected_participant_nf_instance_ids=(BRANCH_B_ID,),
            successful_participant_nf_instance_ids=(BRANCH_B_ID,),
            failed_participant_nf_instance_ids=(),
        ),
    )
    server.remove_protocol_participant.return_value = ""

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    terminal = coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.COMPLETE, RootRequestState.FAILED},
        timeout=3,
    )

    assert terminal.state is expected_state
    assert any(
        call.kwargs["nf_instance_id"] == BRANCH_REPLACEMENT_ID
        for call in resolver.resolve.call_args_list
    )
    assert server.prepare_protocol_replacement_target.call_count == 0
    if expected_state is RootRequestState.COMPLETE:
        assert terminal.completed_rounds == 2
        assert server.execute_hierarchy_round.call_count == 2
        assert server.execute_hierarchy_round.call_args_list[1].kwargs[
            "selected_participant_nf_instance_ids"
        ] == (BRANCH_B_ID,)
    else:
        assert terminal.completed_rounds == 1
        assert server.execute_hierarchy_round.call_count == 1
        assert terminal.failure_cause == "ROUND_FAILED"
    coordinator.close()


def test_protocol_root_simultaneous_branch_failures_are_terminal(tmp_path):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    (
        coordinator,
        _resolver,
        artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_count=2,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(),
    )
    round_path = tmp_path / "simultaneous-round.tar.gz"
    round_path.write_bytes(b"round")
    artifacts.publish_round_input.return_value = SimpleNamespace(
        digest="2" * 64,
        path=round_path,
        url="http://root.example/simultaneous-round",
    )
    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        two_group_preparation_collection(server, process_id)
    )
    server.execute_hierarchy_round.return_value = HierarchyRoundOutcome(
        aggregate=None,
        accepted=False,
        selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
        successful_participant_nf_instance_ids=(),
        failed_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
    )

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    failed = coordinator.wait_for_state(
        REQUEST_A_ID,
        {RootRequestState.COMPLETE, RootRequestState.FAILED},
        timeout=3,
    )

    assert failed.state is RootRequestState.FAILED
    assert failed.failure_cause == "ROUND_FAILED"
    assert server.remove_protocol_participant.call_count == 0
    assert server.prepare_protocol_replacement_target.call_count == 0
    distribution.cleanup.assert_called_once_with(failed.plan_id, 0)
    coordinator.close()


def test_protocol_root_generation_reset_cleans_replacement_created_in_flight(
    tmp_path,
):
    distribution = Mock()
    distribution.store.return_value = SimpleNamespace(
        model_unique_id=77,
        wire_reference=MLModelAdrf(
            adrfId="00000000-0000-4000-8000-000000000900",
            storTransId="round-store",
        ),
    )
    (
        coordinator,
        _resolver,
        artifacts,
        _workspace,
        server,
        _registry,
        _policy,
        _catalog,
        _model,
    ) = root_coordinator(
        tmp_path,
        round_count=3,
        round_model_distribution=distribution,
        topology_text=branch_replacement_topology(root_minimum=1),
    )
    round_paths = []
    aggregate_paths = []
    for index in range(2):
        round_path = tmp_path / f"generation-round-{index}.tar.gz"
        aggregate_path = tmp_path / f"generation-aggregate-{index}.tar.gz"
        round_path.write_bytes(round_path.name.encode())
        aggregate_path.write_bytes(aggregate_path.name.encode())
        round_paths.append(round_path)
        aggregate_paths.append(aggregate_path)
    artifacts.publish_round_input.side_effect = [
        SimpleNamespace(
            digest=str(index + 2) * 64,
            path=path,
            url=f"http://root.example/generation-round-{index}",
        )
        for index, path in enumerate(round_paths)
    ]
    aggregates = [
        SimpleNamespace(
            digest=str(index + 5) * 64,
            path=path,
            url=f"http://root.example/generation-aggregate-{index}",
        )
        for index, path in enumerate(aggregate_paths)
    ]
    server.collect_hierarchy_preparation.side_effect = lambda process_id: (
        two_group_preparation_collection(server, process_id)
    )
    replacement_started = threading.Event()
    allow_replacement = threading.Event()
    replacement_returned = threading.Event()
    degraded_round_started = threading.Event()
    allow_degraded_round = threading.Event()
    execution_count = 0

    def execute_round(**_kwargs):
        nonlocal execution_count
        invocation = execution_count
        execution_count += 1
        if invocation == 0:
            return HierarchyRoundOutcome(
                aggregate=aggregates[0],
                accepted=True,
                selected_participant_nf_instance_ids=(BRANCH_ID, BRANCH_B_ID),
                successful_participant_nf_instance_ids=(BRANCH_B_ID,),
                failed_participant_nf_instance_ids=(BRANCH_ID,),
            )
        degraded_round_started.set()
        assert allow_degraded_round.wait(2)
        return HierarchyRoundOutcome(
            aggregate=aggregates[1],
            accepted=True,
            selected_participant_nf_instance_ids=(BRANCH_B_ID,),
            successful_participant_nf_instance_ids=(BRANCH_B_ID,),
            failed_participant_nf_instance_ids=(),
        )

    def prepare_replacement(**_kwargs):
        replacement_started.set()
        assert allow_replacement.wait(2)
        plan_id = server.start_protocol_preparation.call_args.kwargs[
            "ml_correlation_id"
        ]
        result = HierarchyParticipantPreparationOutcome(
            participant_nf_instance_id=BRANCH_REPLACEMENT_ID,
            resource_location="http://replacement.example/subscriptions/2",
            notification=NwdafMLModelTrainNotif(
                notifCorreId="callback-replacement",
                mlCorreId=plan_id,
                **{
                    "x-flTopologyReport": FlTopologyReport.model_validate(
                        {
                            "nfInstanceId": BRANCH_REPLACEMENT_ID,
                            "children": [
                                {
                                    "nfInstanceId": leaf_id,
                                    "status": "ACTIVE",
                                    "statusTimestamp": "2026-09-08T12:00:00Z",
                                }
                                for leaf_id in (LEAF_A_ID, LEAF_B_ID)
                            ],
                        }
                    )
                },
            ),
            failure="",
            delay_extensions=0,
            granted_extension_seconds=0,
        )
        replacement_returned.set()
        return result

    server.execute_hierarchy_round.side_effect = execute_round
    server.remove_protocol_participant.return_value = ""
    server.prepare_protocol_replacement_target.side_effect = prepare_replacement

    coordinator.submit_manual(
        request_id=REQUEST_A_ID,
        model_family_id="ue-communication-default",
    )
    try:
        assert replacement_started.wait(2)
        assert degraded_round_started.wait(2)
        coordinator.abort_generation("containing NWDAF process generation changed")
        allow_replacement.set()
        allow_degraded_round.set()
        assert replacement_returned.wait(2)
        coordinator.close()

        assert coordinator.get(REQUEST_A_ID) is None
        retired_ids = [
            call.args[1] for call in server.remove_protocol_participant.call_args_list
        ]
        assert BRANCH_ID in retired_ids
        assert BRANCH_REPLACEMENT_ID in retired_ids
    finally:
        allow_replacement.set()
        allow_degraded_round.set()
        coordinator.close()


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
