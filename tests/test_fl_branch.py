from unittest.mock import Mock, call

import pytest

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import HierarchyAssignmentArtifact
from py_mtlf.core.fl_branch import (
    BranchPreparationCancelled,
    FLBranchPreparationCoordinator,
)
from py_mtlf.core.fl_hierarchy import PreparationFailureCause, PreparationOutcome
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_server import (
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
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
            "bundle_schema_version": "1.0",
            "file_digests": {
                "model.py": "1" * 64,
                "model.npy": "2" * 64,
                "scaler.pkl": "3" * 64,
            },
            "hierarchy_metadata": {
                "contract_version": "1.0",
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


def _coordinator(resolver, artifacts, server) -> FLBranchPreparationCoordinator:
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=BRANCH,
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
