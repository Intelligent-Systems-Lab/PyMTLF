import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, call

import pytest
from conftest import training_scope_descriptor

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    HierarchyAssignmentArtifact,
    RoundGlobalArtifact,
    ValidationSummary,
    WapeComponents,
)
from py_mtlf.core.fl_branch import (
    BranchPreparationCancelled,
    BranchPreparationExecution,
    FLBranchPreparationCoordinator,
)
from py_mtlf.core.fl_candidate_orchestration import IntermediateLocalWork
from py_mtlf.core.fl_hierarchy import PreparationFailureCause, PreparationOutcome
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_server import (
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
    HierarchyValidationCollection,
)
from py_mtlf.core.fl_workspace import ValidatedHierarchyArtifact
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainNotif, NwdafMLModelTrainSubsc
from py_mtlf.wire.private import SelectedTarget

ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEAF_A = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEAF_B = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PLAN = "11111111-1111-4111-8111-111111111111"


def _assignment(tmp_path) -> ValidatedHierarchyArtifact:
    path = tmp_path / "branch-assignment.tar.gz"
    path.write_bytes(b"assignment")
    contract = HierarchyAssignmentArtifact.model_validate(
        {
            "artifact_role": "HIERARCHY_ASSIGNMENT",
            "hierarchy_metadata": {
                "message_type": "BRANCH_ASSIGNMENT",
                "plan_id": PLAN,
                "publisher_nf_instance_id": ROOT,
                "intended_recipient_nf_instance_id": BRANCH,
                "assigned_leaf_nf_instance_ids": [LEAF_A, LEAF_B],
                "admission": {"mode": "complete_required"},
                "strategy": {
                    "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
                    "participant_selection": "all",
                    "waiting_policy": "all",
                    "aggregation": "sample_weighted",
                },
            },
        }
    )
    return ValidatedHierarchyArtifact(
        metadata=ArtifactMetadata(
            key="a" * 64,
            size_bytes=path.stat().st_size,
            path=path,
            url="http://root.example/artifacts/" + "a" * 64,
        ),
        manifest={
            "analytics_event": "UE_COMMUNICATION",
            "model_interoperability": "001122",
        },
        contract=contract,
    )


def _representation() -> NwdafMLModelTrainSubsc:
    return NwdafMLModelTrainSubsc.model_validate(
        {
            "mLEventSubscs": [
                {
                    "mLEvent": "UE_COMMUNICATION",
                    "mLEventFilter": {"networkArea": {"tais": []}},
                    "modelInterInfo": "001122",
                }
            ],
            "notifUri": "http://root.example/callback",
            "notifCorreId": "branch-callback",
            "mlCorreId": "root-process",
            "mLPreFlag": True,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://root.example/artifacts/" + "a" * 64
                    },
                }
            ],
            "eventReq": {"notifMethod": "ON_EVENT_DETECTION"},
            "mLModelTrainInfos": [
                {
                    "dataAvReq": {
                        "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                        "minNumSamples": 1,
                        "timeWindows": [
                            {
                                "startTime": "2026-07-01T00:00:00Z",
                                "stopTime": "2026-07-02T00:00:00Z",
                            }
                        ],
                    },
                    "timeAvReq": "PT5M",
                }
            ],
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )


def _node(nf_instance_id: str) -> ResolvedHierarchyNode:
    return ResolvedHierarchyNode(
        nf_instance_id=nf_instance_id,
        role=HierarchyNodeRole.LEAF,
        target=SelectedTarget(
            nfInstanceId=nf_instance_id,
            nfServiceInstanceId="training-" + nf_instance_id[:4],
            serviceName="nnwdaf-mlmodeltraining",
            apiRoot="http://leaf-" + nf_instance_id[:4] + ".example",
            selectionSource="NRF",
        ),
    )


def _coordinator(resolver, artifacts, server, **kwargs) -> FLBranchPreparationCoordinator:
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=BRANCH,
        containing_nwdaf_process_instance_id="22222222-2222-4222-8222-222222222222",
        api_root="http://branch.example",
        internal_api_root="http://branch-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    return FLBranchPreparationCoordinator(
        resolver=resolver,
        nwdaf_context=context,
        artifact_service=artifacts,
        server=server,
        **kwargs,
    )


def _lower_global(tmp_path, round_indicator: int):
    path = tmp_path / f"lower-global-{round_indicator}.tar.gz"
    path.write_bytes(f"lower-{round_indicator}".encode())
    contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "ml_corre_id": "lower-process",
                "round_ind": round_indicator,
                "participants": [
                    {
                        "participant_nf_instance_id": LEAF_A,
                        "training_sample_count": 1,
                        "local_artifact_digest": "8" * 64,
                    }
                ],
                "aggregated_training_sample_count": 1,
            },
        }
    )
    return Mock(
        url=f"http://branch.example/global-{round_indicator}",
        digest=str(round_indicator + 4) * 64,
        path=path,
        contract=contract,
    )


def test_branch_resolves_every_leaf_before_republish_and_lower_dispatch(tmp_path):
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    published_a = Mock(url="http://branch.example/artifacts/a")
    published_b = Mock(url="http://branch.example/artifacts/b")
    artifacts = Mock()
    artifacts.republish_leaf_assignment.side_effect = [published_a, published_b]
    server = Mock()
    server.start_hierarchy_preparation.return_value = Mock(process_id="lower-process")
    coordinator = _coordinator(resolver, artifacts, server)

    execution = coordinator.dispatch(
        assignment=_assignment(tmp_path),
        representation=_representation(),
        reservation_id="reservation-1",
    )

    assert execution.process_id == "lower-process"
    assert tuple(node.nf_instance_id for node in execution.leaf_nodes) == (LEAF_A, LEAF_B)
    assert resolver.resolve.call_args_list == [
        call(
            nf_instance_id=LEAF_A,
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
        ),
        call(
            nf_instance_id=LEAF_B,
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
        ),
    ]
    dispatched = server.start_hierarchy_preparation.call_args.kwargs
    assert [item.participant_nf_instance_id for item in dispatched["targets"]] == [
        LEAF_A,
        LEAF_B,
    ]
    assert [item.assignment_url for item in dispatched["targets"]] == [
        published_a.url,
        published_b.url,
    ]


def test_branch_resolution_failure_creates_no_leaf_bundle_or_lower_resource(tmp_path):
    resolver = Mock()
    resolver.resolve.side_effect = [
        _node(LEAF_A),
        RuntimeError("Leaf unavailable"),
    ]
    artifacts = Mock()
    server = Mock()
    coordinator = _coordinator(resolver, artifacts, server)

    with pytest.raises(RuntimeError, match="assigned Leaf discovery failed"):
        coordinator.dispatch(
            assignment=_assignment(tmp_path),
            representation=_representation(),
            reservation_id="reservation-1",
        )

    artifacts.republish_leaf_assignment.assert_not_called()
    server.start_hierarchy_preparation.assert_not_called()


def test_pre_dispatch_failure_publishes_complete_failed_partition(tmp_path):
    resolver = Mock()
    resolver.resolve.side_effect = [
        _node(LEAF_A),
        RuntimeError("Leaf unavailable"),
    ]
    result_artifact = Mock(url="http://branch.example/artifacts/result")
    artifacts = Mock()
    artifacts.publish_preparation_result.return_value = result_artifact
    server = Mock()
    coordinator = _coordinator(resolver, artifacts, server)

    result = coordinator.prepare(
        assignment=_assignment(tmp_path),
        representation=_representation(),
        reservation_id="reservation-1",
    )

    assert result.outcome is PreparationOutcome.FAILED
    published = artifacts.publish_preparation_result.call_args.kwargs
    assert [item.nf_instance_id for item in published["failed_clients"]] == [
        LEAF_A,
        LEAF_B,
    ]
    assert [item.cause for item in published["failed_clients"]] == [
        PreparationFailureCause.NOT_AVAILABLE_ML_TRAIN,
        PreparationFailureCause.DISCOVERY_FAILED,
    ]
    assert published["timed_out_client_nf_instance_ids"] == ()
    server.start_hierarchy_preparation.assert_not_called()
    server.collect_hierarchy_preparation.assert_not_called()


def test_successful_branch_preparation_admits_lower_server_process(tmp_path):
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    published_a = Mock(url="http://branch.example/artifacts/a")
    published_b = Mock(url="http://branch.example/artifacts/b")
    artifacts = Mock()
    artifacts.republish_leaf_assignment.side_effect = [published_a, published_b]
    artifacts.publish_preparation_result.return_value = Mock(
        url="http://branch.example/artifacts/result"
    )
    server = Mock()
    server.start_hierarchy_preparation.return_value = Mock(process_id="lower-process")
    server.collect_hierarchy_preparation.return_value = HierarchyPreparationCollection(
        process_id="lower-process",
        plan_id=PLAN,
        participants=tuple(
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=leaf_id,
                resource_location=f"http://branch.example/subscriptions/{leaf_id}",
                assignment_url=published.url,
                notification=NwdafMLModelTrainNotif.model_validate(
                    {
                        "notifCorreId": leaf_id,
                        "mlCorreId": "lower-process",
                        "mLModelInfos": [
                            {
                                "event": "UE_COMMUNICATION",
                                "mLFileAddr": {"mLModelUrl": published.url},
                            }
                        ],
                    }
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            )
            for leaf_id, published in ((LEAF_A, published_a), (LEAF_B, published_b))
        ),
        timed_out_participant_nf_instance_ids=(),
    )
    coordinator = _coordinator(resolver, artifacts, server)

    result = coordinator.prepare(
        assignment=_assignment(tmp_path),
        representation=_representation(),
        reservation_id="reservation-1",
    )

    assert result.outcome is PreparationOutcome.READY
    server.admit_hierarchy_preparation.assert_called_once_with("lower-process")


def test_parent_cancellation_fences_pre_dispatch_publication(tmp_path):
    resolver = Mock()
    artifacts = Mock()
    server = Mock()
    coordinator = _coordinator(resolver, artifacts, server)

    def cancel_during_resolution(**_kwargs):
        coordinator.cancel(PLAN, "parent cancelled")
        return _node(LEAF_A)

    resolver.resolve.side_effect = cancel_during_resolution

    with pytest.raises(BranchPreparationCancelled):
        coordinator.prepare(
            assignment=_assignment(tmp_path),
            representation=_representation(),
            reservation_id="reservation-1",
        )

    artifacts.republish_leaf_assignment.assert_not_called()
    artifacts.publish_preparation_result.assert_not_called()
    server.start_hierarchy_preparation.assert_not_called()


def test_go_generation_reset_discards_branch_mappings_without_closing(tmp_path):
    server = Mock()
    coordinator = _coordinator(Mock(), Mock(), server)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=_assignment(tmp_path),
        leaf_nodes=(_node(LEAF_A),),
        leaf_assignments=(Mock(),),
        process_id="lower-process",
    )
    coordinator._rounds[(PLAN, "upper-process", 0)] = Mock()
    coordinator._cancelled_plan_ids[PLAN] = 9999999999

    coordinator.abort_generation("containing NWDAF process generation changed")

    assert coordinator._executions == {}
    assert coordinator._rounds == {}
    assert coordinator._cancelled_plan_ids == {}
    assert coordinator._closing is False
    server.cancel_hierarchy_preparation.assert_called_once_with(
        "lower-process",
        "containing NWDAF process generation changed",
    )


def test_cancelled_branch_plan_tombstone_is_pruned_lazily():
    now = [10.0]
    coordinator = _coordinator(
        Mock(),
        Mock(),
        Mock(),
        tombstone_ttl_seconds=5,
        clock=lambda: now[0],
    )
    coordinator.cancel(PLAN, "parent cancelled")

    with pytest.raises(BranchPreparationCancelled):
        coordinator._ensure_dispatch_active(PLAN)

    now[0] = 16.0

    coordinator._ensure_dispatch_active(PLAN)


def test_parent_cancellation_after_upper_validation_fences_lower_round_publication(
    tmp_path,
):
    artifacts = Mock()
    server = Mock()
    coordinator = _coordinator(Mock(), artifacts, server)
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    context_entered = threading.Event()
    release_context = threading.Event()
    context = coordinator._nwdaf_context.get.return_value

    def get_context(**_kwargs):
        context_entered.set()
        if not release_context.wait(1):
            raise AssertionError("test did not release context validation")
        return context

    coordinator._nwdaf_context.get.side_effect = get_context
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    failures = []

    def execute():
        try:
            coordinator.execute_round(
                assignment=assignment,
                representation=representation,
                upper_input=Mock(
                    manifest={"fl_metadata": {"client_training": {"epochs": 7}}}
                ),
                upper_client_subscription_id="upper-resource",
                upper_resource_revision=2,
                upper_input_artifact_digest="3" * 64,
                upper_training_scope=training_scope_descriptor(),
                callback_margin_seconds=5,
            )
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert context_entered.wait(1)
        coordinator.cancel(PLAN, "parent cancelled")
        release_context.set()
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], BranchPreparationCancelled)
        artifacts.publish_round_input.assert_not_called()
        artifacts.publish_hierarchy_aggregate.assert_not_called()
        server.execute_hierarchy_round.assert_not_called()
        server.cancel_hierarchy_preparation.assert_called_once_with(
            "lower-process",
            "parent cancelled",
        )
    finally:
        release_context.set()
        thread.join(timeout=1)


def test_branch_round_preserves_root_epochs_and_maps_upper_to_lower(tmp_path):
    resolver = Mock()
    artifacts = Mock()
    lower_input = Mock(
        url="http://branch.example/round-input",
        digest="4" * 64,
    )
    lower_contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "ml_corre_id": "lower-process",
                "round_ind": 0,
                "participants": [
                    {
                        "participant_nf_instance_id": LEAF_A,
                        "training_sample_count": 1,
                        "local_artifact_digest": "8" * 64,
                    }
                ],
                "aggregated_training_sample_count": 1,
            },
        }
    )
    lower_global = Mock(
        url="http://branch.example/round-global",
        digest="5" * 64,
        contract=lower_contract,
    )
    upper_result = Mock(url="http://branch.example/upper-result", digest="6" * 64)
    artifacts.publish_hierarchy_aggregate.return_value = upper_result
    server = Mock()
    server.execute_hierarchy_round.return_value = lower_global
    coordinator = _coordinator(resolver, artifacts, server)
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )

    def publish_lower_input(**_kwargs):
        mapping = coordinator._rounds.get((PLAN, "root-process", 4))
        assert mapping is not None
        assert mapping.state == "RUNNING"
        return lower_input

    artifacts.publish_round_input.side_effect = publish_lower_input
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    upper_input = Mock(
        manifest={"fl_metadata": {"client_training": {"epochs": 7}}}
    )

    first = coordinator.execute_round(
        assignment=assignment,
        representation=representation,
        upper_input=upper_input,
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=2,
        upper_input_artifact_digest="3" * 64,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=5,
    )
    replay = coordinator.execute_round(
        assignment=assignment,
        representation=representation,
        upper_input=upper_input,
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=2,
        upper_input_artifact_digest="3" * 64,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=5,
    )

    assert first is upper_result
    assert replay is upper_result
    lower_publication = artifacts.publish_round_input.call_args.kwargs
    assert lower_publication["process_id"] == "lower-process"
    assert lower_publication["round_indicator"] == 0
    assert lower_publication["epochs"] == 7
    lower_execution = server.execute_hierarchy_round.call_args.kwargs
    assert lower_execution["timeout_seconds"] == 295
    assert lower_execution["round_input_artifact"] is lower_input
    artifacts.publish_round_input.assert_called_once()
    artifacts.publish_hierarchy_aggregate.assert_called_once()
    mapping = coordinator._rounds[(PLAN, "root-process", 4)]
    assert mapping.upper_client_subscription_id == "upper-resource"
    assert mapping.upper_resource_revision == 2
    assert mapping.upper_input_artifact_digest == "3" * 64
    assert mapping.lower_server_process_id == "lower-process"
    assert mapping.lower_ml_corre_id == "lower-process"
    assert mapping.lower_round_indicator == 0
    assert mapping.state == "COMPLETE"

    with pytest.raises(RuntimeError, match="conflicting duplicate"):
        coordinator.execute_round(
            assignment=assignment,
            representation=representation,
            upper_input=upper_input,
            upper_client_subscription_id="upper-resource",
            upper_resource_revision=2,
            upper_input_artifact_digest="8" * 64,
            upper_training_scope=training_scope_descriptor(),
            callback_margin_seconds=5,
        )
    assert (PLAN, "root-process", 4) not in coordinator._rounds
    server.cancel_hierarchy_preparation.assert_called_once_with(
        "lower-process",
        "conflicting duplicate Branch upper round command",
    )


def test_branch_report_after_runs_sequential_lower_rounds_before_upper_report(tmp_path):
    artifacts = Mock()
    server = Mock()
    coordinator = _coordinator(Mock(), artifacts, server)
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    lower_inputs = [
        Mock(url=f"http://branch.example/input-{index}", digest=str(index + 1) * 64)
        for index in range(3)
    ]
    artifacts.publish_round_input.side_effect = lower_inputs
    lower_globals = [_lower_global(tmp_path, index) for index in range(3)]
    server.execute_hierarchy_round.side_effect = lower_globals
    loaded_intermediates = (Mock(name="lower-aggregate-0"), Mock(name="lower-aggregate-1"))
    coordinator._loader = Mock()
    coordinator._loader.load.side_effect = loaded_intermediates
    upper_result = Mock(url="http://branch.example/upper-result")
    artifacts.publish_hierarchy_aggregate.return_value = upper_result
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    upper_input = Mock(manifest={"fl_metadata": {"client_training": {"epochs": 7}}})

    result = coordinator.execute_round(
        assignment=assignment,
        representation=NwdafMLModelTrainSubsc.model_validate(payload),
        upper_input=upper_input,
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=2,
        upper_input_artifact_digest="3" * 64,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=6,
        local_work=IntermediateLocalWork(lower_round_count=3),
    )

    assert result is upper_result
    assert [
        item.kwargs["round_indicator"]
        for item in artifacts.publish_round_input.call_args_list
    ] == [0, 1, 2]
    assert [item.kwargs["base"] for item in artifacts.publish_round_input.call_args_list] == [
        upper_input,
        *loaded_intermediates,
    ]
    assert [
        item.kwargs["timeout_seconds"]
        for item in server.execute_hierarchy_round.call_args_list
    ] == [98, 98, 98]
    assert artifacts.publish_hierarchy_aggregate.call_args.kwargs["lower_global"] is (
        lower_globals[-1]
    )
    assert coordinator._rounds[(PLAN, "root-process", 4)].lower_round_indicator == 2


def test_branch_report_after_failure_does_not_publish_earlier_partial_aggregate(tmp_path):
    artifacts = Mock()
    artifacts.publish_round_input.side_effect = [
        Mock(url="http://branch.example/input-0", digest="1" * 64),
        Mock(url="http://branch.example/input-1", digest="2" * 64),
    ]
    server = Mock()
    server.execute_hierarchy_round.side_effect = [
        _lower_global(tmp_path, 0),
        RuntimeError("lower round failed"),
    ]
    coordinator = _coordinator(Mock(), artifacts, server)
    coordinator._loader = Mock(return_value=Mock(name="lower-aggregate-0"))
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )

    with pytest.raises(RuntimeError, match="lower round failed"):
        coordinator.execute_round(
            assignment=assignment,
            representation=NwdafMLModelTrainSubsc.model_validate(payload),
            upper_input=Mock(
                manifest={"fl_metadata": {"client_training": {"epochs": 7}}}
            ),
            upper_client_subscription_id="upper-resource",
            upper_resource_revision=2,
            upper_input_artifact_digest="3" * 64,
            upper_training_scope=training_scope_descriptor(),
            callback_margin_seconds=6,
            local_work=IntermediateLocalWork(lower_round_count=3),
        )

    assert server.execute_hierarchy_round.call_count == 2
    artifacts.publish_hierarchy_aggregate.assert_not_called()
    assert coordinator._rounds[(PLAN, "root-process", 4)].state == "FAILED"


def test_branch_validation_republishes_before_lower_dispatch_and_returns_leaf_evidence(
    tmp_path,
):
    assignment = _assignment(tmp_path)
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    candidate_artifact = ArtifactMetadata(
        key="3" * 64,
        size_bytes=candidate_path.stat().st_size,
        path=candidate_path,
        url="http://root.example/candidate/" + "3" * 64,
    )
    republished = Mock(
        url="http://branch.example/candidate/" + "3" * 64,
        digest="3" * 64,
    )
    upper_result = Mock(url="http://branch.example/validation-result")
    artifacts = Mock()
    artifacts.republish_validation_candidate.return_value = republished
    artifacts.publish_hierarchy_validation_result.return_value = upper_result
    start = datetime(2026, 8, 20, tzinfo=UTC)
    summaries = tuple(
        ValidationSummary(
            participant_nf_instance_id=leaf_id,
            training_scope=training_scope_descriptor(f"scope-{index}"),
            evaluation_sample_count=10,
            start_time=start,
            end_time=start + timedelta(minutes=1),
            base=WapeComponents(
                absolute_error_sum=10,
                absolute_actual_sum=100,
            ),
            candidate=WapeComponents(
                absolute_error_sum=5,
                absolute_actual_sum=100,
            ),
        )
        for index, leaf_id in enumerate((LEAF_A, LEAF_B), start=1)
    )
    server = Mock()
    server.execute_hierarchy_validation.return_value = HierarchyValidationCollection(
        candidate_artifact=candidate_artifact,
        validation_summaries=summaries,
    )
    coordinator = _coordinator(Mock(), artifacts, server)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    coordinator._next_lower_round[PLAN] = 2
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    upper_candidate = Mock()

    first = coordinator.execute_validation(
        assignment=assignment,
        representation=representation,
        upper_candidate=upper_candidate,
        upper_candidate_artifact=candidate_artifact,
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=3,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=5,
    )
    replay = coordinator.execute_validation(
        assignment=assignment,
        representation=representation,
        upper_candidate=upper_candidate,
        upper_candidate_artifact=candidate_artifact,
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=3,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=5,
    )

    assert first is upper_result
    assert replay is upper_result
    artifacts.republish_validation_candidate.assert_called_once()
    lower = server.execute_hierarchy_validation.call_args.kwargs
    assert lower["process_id"] == "lower-process"
    assert lower["validation_round"] == 2
    assert lower["candidate"] is republished
    assert lower["expected_candidate_process_id"] == "root-process"
    assert lower["expected_candidate_round"] == 1
    assert lower["timeout_seconds"] == 295
    upper = artifacts.publish_hierarchy_validation_result.call_args.kwargs
    assert upper["subordinate_summaries"] == summaries
    assert coordinator._validations[(PLAN, "root-process", 2)].state == "COMPLETE"


def test_branch_validation_republish_failure_does_not_dispatch_leaves(tmp_path):
    assignment = _assignment(tmp_path)
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    candidate_artifact = ArtifactMetadata(
        key="3" * 64,
        size_bytes=candidate_path.stat().st_size,
        path=candidate_path,
        url="http://root.example/candidate/" + "3" * 64,
    )
    artifacts = Mock()
    artifacts.republish_validation_candidate.side_effect = RuntimeError(
        "republish failed"
    )
    server = Mock()
    coordinator = _coordinator(Mock(), artifacts, server)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )

    with pytest.raises(RuntimeError, match="republish failed"):
        coordinator.execute_validation(
            assignment=assignment,
            representation=NwdafMLModelTrainSubsc.model_validate(payload),
            upper_candidate=Mock(),
            upper_candidate_artifact=candidate_artifact,
            upper_client_subscription_id="upper-resource",
            upper_resource_revision=3,
            upper_training_scope=training_scope_descriptor(),
            callback_margin_seconds=5,
        )

    server.execute_hierarchy_validation.assert_not_called()
    artifacts.publish_hierarchy_validation_result.assert_not_called()


def test_branch_cancellation_during_validation_fences_late_lower_result(tmp_path):
    assignment = _assignment(tmp_path)
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    candidate_artifact = ArtifactMetadata(
        key="3" * 64,
        size_bytes=candidate_path.stat().st_size,
        path=candidate_path,
        url="http://root.example/candidate/" + "3" * 64,
    )
    republished = Mock(
        url="http://branch.example/candidate/" + "3" * 64,
        digest="3" * 64,
    )
    artifacts = Mock()
    artifacts.republish_validation_candidate.return_value = republished
    server = Mock()
    lower_started = threading.Event()
    release_lower = threading.Event()

    def execute_lower(**_kwargs):
        lower_started.set()
        if not release_lower.wait(1):
            raise AssertionError("test did not release the lower validation")
        return HierarchyValidationCollection(
            candidate_artifact=candidate_artifact,
            validation_summaries=(),
        )

    def cancel_lower(_process_id, _reason):
        release_lower.set()

    server.execute_hierarchy_validation.side_effect = execute_lower
    server.cancel_hierarchy_preparation.side_effect = cancel_lower
    coordinator = _coordinator(Mock(), artifacts, server)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    failures = []

    def execute():
        try:
            coordinator.execute_validation(
                assignment=assignment,
                representation=representation,
                upper_candidate=Mock(),
                upper_candidate_artifact=candidate_artifact,
                upper_client_subscription_id="upper-resource",
                upper_resource_revision=3,
                upper_training_scope=training_scope_descriptor(),
                callback_margin_seconds=5,
            )
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert lower_started.wait(1) is True
        coordinator.cancel(PLAN, "parent deleted upper validation resource")
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], BranchPreparationCancelled)
        assert (PLAN, "root-process", 2) not in coordinator._validations
        artifacts.publish_hierarchy_validation_result.assert_not_called()
        server.cancel_hierarchy_preparation.assert_called_once_with(
            "lower-process",
            "parent deleted upper validation resource",
        )
    finally:
        release_lower.set()
        thread.join(timeout=1)
        coordinator.close()


def test_concurrent_exact_branch_round_replay_waits_for_one_lower_execution(tmp_path):
    lower_input = Mock(url="http://branch.example/round-input", digest="4" * 64)
    lower_contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "ml_corre_id": "lower-process",
                "round_ind": 0,
                "participants": [
                    {
                        "participant_nf_instance_id": LEAF_A,
                        "training_sample_count": 1,
                        "local_artifact_digest": "8" * 64,
                    }
                ],
                "aggregated_training_sample_count": 1,
            },
        }
    )
    lower_global = Mock(
        url="http://branch.example/round-global",
        digest="5" * 64,
        contract=lower_contract,
    )
    upper_result = Mock(url="http://branch.example/upper-result", digest="6" * 64)
    publication_entered = threading.Event()
    release_publication = threading.Event()
    artifacts = Mock()

    def publish_lower_input(**_kwargs):
        publication_entered.set()
        if not release_publication.wait(1):
            raise AssertionError("test did not release lower input publication")
        return lower_input

    artifacts.publish_round_input.side_effect = publish_lower_input
    artifacts.publish_hierarchy_aggregate.return_value = upper_result
    server = Mock()
    server.execute_hierarchy_round.return_value = lower_global
    coordinator = _coordinator(Mock(), artifacts, server)
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    upper_input = Mock(
        manifest={"fl_metadata": {"client_training": {"epochs": 7}}}
    )
    results = []
    failures = []

    def execute():
        try:
            results.append(
                coordinator.execute_round(
                    assignment=assignment,
                    representation=representation,
                    upper_input=upper_input,
                    upper_client_subscription_id="upper-resource",
                    upper_resource_revision=2,
                    upper_input_artifact_digest="3" * 64,
                    upper_training_scope=training_scope_descriptor(),
                    callback_margin_seconds=5,
                )
            )
        except Exception as error:
            failures.append(error)

    first = threading.Thread(target=execute)
    replay = threading.Thread(target=execute)
    first.start()
    assert publication_entered.wait(1)
    replay.start()
    assert replay.is_alive()
    release_publication.set()
    first.join(timeout=1)
    replay.join(timeout=1)

    assert not first.is_alive()
    assert not replay.is_alive()
    assert failures == []
    assert results == [upper_result, upper_result]
    artifacts.publish_round_input.assert_called_once()
    server.execute_hierarchy_round.assert_called_once()
    assert server.execute_hierarchy_round.call_args.kwargs["round_indicator"] == 0
    artifacts.publish_hierarchy_aggregate.assert_called_once()


def test_branch_shutdown_wakes_lower_round_waiter_and_fences_upper_callback(tmp_path):
    lower_contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "ml_corre_id": "lower-process",
                "round_ind": 0,
                "participants": [
                    {
                        "participant_nf_instance_id": LEAF_A,
                        "training_sample_count": 1,
                        "local_artifact_digest": "8" * 64,
                    }
                ],
                "aggregated_training_sample_count": 1,
            },
        }
    )
    lower_global = Mock(
        url="http://branch.example/round-global",
        digest="5" * 64,
        contract=lower_contract,
    )
    lower_waiting = threading.Event()
    release_lower = threading.Event()
    artifacts = Mock()
    artifacts.publish_round_input.return_value = Mock(
        url="http://branch.example/round-input",
        digest="4" * 64,
    )
    server = Mock()

    def execute_lower(**_kwargs):
        lower_waiting.set()
        if not release_lower.wait(1):
            raise AssertionError("Branch shutdown did not cancel the lower Server process")
        return lower_global

    def cancel_lower(_process_id, _reason):
        release_lower.set()

    server.execute_hierarchy_round.side_effect = execute_lower
    server.cancel_hierarchy_preparation.side_effect = cancel_lower
    coordinator = _coordinator(Mock(), artifacts, server)
    assignment = _assignment(tmp_path)
    coordinator._executions[PLAN] = BranchPreparationExecution(
        plan_id=PLAN,
        parent_assignment=assignment,
        leaf_nodes=(_node(LEAF_A), _node(LEAF_B)),
        leaf_assignments=(Mock(), Mock()),
        process_id="lower-process",
    )
    payload = _representation().model_dump(by_alias=True, exclude_none=True, mode="json")
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 4,
            "mLTrainRepInfo": {"maxResTime": 300},
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    failures = []

    def execute():
        try:
            coordinator.execute_round(
                assignment=assignment,
                representation=representation,
                upper_input=Mock(
                    manifest={"fl_metadata": {"client_training": {"epochs": 7}}}
                ),
                upper_client_subscription_id="upper-resource",
                upper_resource_revision=2,
                upper_input_artifact_digest="3" * 64,
                upper_training_scope=training_scope_descriptor(),
                callback_margin_seconds=5,
            )
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert lower_waiting.wait(1) is True
        coordinator.close()
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], BranchPreparationCancelled)
        artifacts.publish_hierarchy_aggregate.assert_not_called()
        server.cancel_hierarchy_preparation.assert_called_once_with(
            "lower-process",
            "Branch preparation coordinator is closing",
        )
    finally:
        release_lower.set()
        thread.join(timeout=1)


@pytest.mark.parametrize("duplicate_model_info", [False, True])
def test_branch_result_partitions_all_leaf_outcomes_once(
    tmp_path,
    duplicate_model_info,
):
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    published_a = Mock(url="http://branch.example/artifacts/a")
    published_b = Mock(url="http://branch.example/artifacts/b")
    result_artifact = Mock(url="http://branch.example/artifacts/result")
    artifacts = Mock()
    artifacts.republish_leaf_assignment.side_effect = [published_a, published_b]
    artifacts.publish_preparation_result.return_value = result_artifact
    server = Mock()
    server.start_hierarchy_preparation.return_value = Mock(process_id="lower-process")
    server.collect_hierarchy_preparation.return_value = HierarchyPreparationCollection(
        process_id="lower-process",
        plan_id=PLAN,
        participants=(
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=LEAF_A,
                resource_location="http://branch.example/subscriptions/leaf-a",
                assignment_url="",
                notification=NwdafMLModelTrainNotif(
                    notifCorreId="leaf-a",
                    mlCorreId="lower-process",
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            ),
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=LEAF_B,
                resource_location="http://branch.example/subscriptions/leaf-b",
                assignment_url=published_b.url,
                notification=NwdafMLModelTrainNotif.model_validate(
                    {
                        "notifCorreId": "leaf-b",
                        "mlCorreId": "lower-process",
                        "mLModelInfos": [
                            {
                                "event": "UE_COMMUNICATION",
                                "mLFileAddr": {"mLModelUrl": published_b.url},
                            }
                        ]
                        * (2 if duplicate_model_info else 1),
                    }
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            ),
        ),
        timed_out_participant_nf_instance_ids=(),
    )
    coordinator = _coordinator(resolver, artifacts, server)

    result = coordinator.prepare(
        assignment=_assignment(tmp_path),
        representation=_representation(),
        reservation_id="reservation-1",
    )

    assert result.artifact is result_artifact
    assert result.outcome is PreparationOutcome.FAILED
    published = artifacts.publish_preparation_result.call_args.kwargs
    assert [item.nf_instance_id for item in published["prepared_clients"]] == (
        [] if duplicate_model_info else [LEAF_B]
    )
    assert [item.nf_instance_id for item in published["failed_clients"]] == (
        [LEAF_A, LEAF_B] if duplicate_model_info else [LEAF_A]
    )
    assert published["failed_clients"][0].cause is (
        PreparationFailureCause.NOT_AVAILABLE_ML_TRAIN
    )
    assert published["timed_out_client_nf_instance_ids"] == ()
    server.admit_hierarchy_preparation.assert_not_called()
