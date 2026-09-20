import json
from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import torch
from conftest import training_scope_descriptor

from py_mtlf.config import ExperimentRecordingSettings
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.experiment_recording import ExperimentRecorder
from py_mtlf.core.fl_artifacts import RoundGlobalArtifact
from py_mtlf.core.fl_branch import FLBranchPreparationCoordinator
from py_mtlf.core.fl_candidate_orchestration import IntermediateLocalWork
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryScope,
    HierarchyDiscoverySnapshot,
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_server import (
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
    HierarchyRoundOutcome,
)
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.wire.ml_model import MLEventNotification, MLModelAdrf
from py_mtlf.wire.ml_model_training import (
    FlTopologyReport,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
)
from py_mtlf.wire.private import SelectedTarget

ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEAF_A = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEAF_B = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PLAN = "11111111-1111-4111-8111-111111111111"


def _protocol_representation() -> NwdafMLModelTrainSubsc:
    return NwdafMLModelTrainSubsc.model_validate(
        {
            "mLEventSubscs": [
                {
                    "mLEvent": "X_IMAGE_CLASSIFICATION",
                    "mLEventFilter": {},
                    "modelInterInfo": "pymtlf-image-classification-mnist",
                }
            ],
            "notifUri": "http://root.example/callback",
            "notifCorreId": "branch-callback",
            "mlCorreId": PLAN,
            "mLPreFlag": True,
            "suppFeats": "4",
            "x-flTopology": {
                "nfInstanceId": BRANCH,
                "policy": {
                    "allowAdditionalCandidates": False,
                    "additionalCandidatePriority": 0,
                    "selectionMethod": "priority",
                    "minAvailableNodes": 2,
                    "fractionTrain": 1.0,
                    "minTrainNodes": 2,
                    "acceptFailures": False,
                    "minCompletionRate": 1.0,
                },
                "strategy": {
                    "method": "fedProx",
                    "aggregation": "sampleWeighted",
                    "methodParameters": {"proximalMu": 0.01},
                },
                "reportAfter": {"count": 1, "unit": "round"},
                "children": [
                    {
                        "nfInstanceId": LEAF_A,
                        "priority": 100,
                        "strategy": {
                            "method": "fedProx",
                            "aggregation": "sampleWeighted",
                            "methodParameters": {"proximalMu": 0.01},
                        },
                        "reportAfter": {"count": 2, "unit": "epoch"},
                    },
                    {
                        "nfInstanceId": LEAF_B,
                        "priority": 90,
                        "strategy": {
                            "method": "fedProx",
                            "aggregation": "sampleWeighted",
                            "methodParameters": {"proximalMu": 0.01},
                        },
                        "reportAfter": {"count": 2, "unit": "epoch"},
                    },
                ],
            },
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


def _protocol_collection(
    participant_ids: tuple[str, ...],
) -> HierarchyPreparationCollection:
    return HierarchyPreparationCollection(
        process_id=PLAN,
        plan_id=PLAN,
        participants=tuple(
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=leaf_id,
                resource_location=f"http://{leaf_id}.example/subscriptions/1",
                notification=NwdafMLModelTrainNotif(
                    notifCorreId=f"callback-{leaf_id}",
                    mlCorreId=PLAN,
                    **{
                        "x-flTopologyReport": FlTopologyReport(
                            nfInstanceId=leaf_id
                        )
                    },
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            )
            for leaf_id in participant_ids
        ),
        timed_out_participant_nf_instance_ids=(),
    )


def _coordinator(
    resolver,
    artifacts,
    server,
    *,
    ml_event="UE_COMMUNICATION",
    **kwargs,
) -> FLBranchPreparationCoordinator:
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=BRANCH,
        containing_nwdaf_process_instance_id="22222222-2222-4222-8222-222222222222",
        api_root="http://branch.example",
        internal_api_root="http://branch-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=(ml_event,),
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


def test_protocol_branch_establishes_children_and_composes_topology_report():
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    server = Mock()
    server.start_protocol_preparation.return_value = Mock(process_id=PLAN)
    reports = {
        leaf_id: FlTopologyReport(nfInstanceId=leaf_id)
        for leaf_id in (LEAF_A, LEAF_B)
    }
    server.collect_hierarchy_preparation.return_value = HierarchyPreparationCollection(
        process_id=PLAN,
        plan_id=PLAN,
        participants=tuple(
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=leaf_id,
                resource_location=f"http://{leaf_id}.example/subscriptions/1",
                notification=NwdafMLModelTrainNotif(
                    notifCorreId=f"callback-{leaf_id}",
                    mlCorreId=PLAN,
                    **{"x-flTopologyReport": reports[leaf_id]},
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            )
            for leaf_id in (LEAF_A, LEAF_B)
        ),
        timed_out_participant_nf_instance_ids=(),
    )
    coordinator = _coordinator(
        resolver,
        Mock(),
        server,
        ml_event="X_IMAGE_CLASSIFICATION",
    )

    report = coordinator.prepare_protocol(
        representation=_protocol_representation(),
        reservation_id="reservation-1",
    )

    assert report.nf_instance_id == BRANCH
    assert tuple(child.nf_instance_id for child in report.children) == (
        LEAF_A,
        LEAF_B,
    )
    assert all(child.status == "ACTIVE" for child in report.children)
    targets = server.start_protocol_preparation.call_args.kwargs["targets"]
    assert tuple(target.participant_nf_instance_id for target in targets) == (LEAF_A,)
    added = server.add_protocol_preparation_targets.call_args.kwargs["targets"]
    assert tuple(target.participant_nf_instance_id for target in added) == (LEAF_B,)
    server.admit_hierarchy_preparation.assert_called_once_with(PLAN)
    coordinator.close()


def test_protocol_branch_combines_explicit_and_discovered_candidates_until_ready():
    leaf_c = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    resolver = Mock()
    resolver.resolve.return_value = _node(LEAF_A)
    discovered_at = datetime.now(UTC)
    discovered_b = _node(LEAF_B)
    discovered_c = _node(leaf_c)
    scope = HierarchyDiscoveryScope(
        containing_nf_instance_id=BRANCH,
        role=HierarchyNodeRole.LEAF,
        ml_event="X_IMAGE_CLASSIFICATION",
        model_interoperability="pymtlf-image-classification-mnist",
        tracking_areas=(("466", "92", "000001"),),
    )
    resolver.discovery_scope_for_subscription.return_value = scope
    resolver.discover_for_subscription.return_value = HierarchyDiscoverySnapshot(
        scope=scope,
        nodes=(discovered_b, discovered_c),
        observed_at=discovered_at,
        validity_period=300,
        returned_count=2,
        complete_nf_instance_count=2,
    )
    server = Mock()
    server.default_protocol_client_epochs = 2
    server.start_protocol_preparation.return_value = Mock(process_id=PLAN)
    server.collect_hierarchy_preparation.side_effect = [
        _protocol_collection((LEAF_A,)),
        _protocol_collection((LEAF_A, LEAF_B)),
    ]
    payload = _protocol_representation().model_dump(
        by_alias=True,
        exclude_none=True,
        mode="json",
    )
    payload["mLEventSubscs"][0]["mLEventFilter"] = {
        "networkArea": {
            "tais": [
                {
                    "plmnId": {"mcc": "466", "mnc": "92"},
                    "tac": "000001",
                }
            ]
        }
    }
    topology = payload["x-flTopology"]
    topology["policy"].update(
        {
            "allowAdditionalCandidates": True,
            "additionalCandidatePriority": 10,
            "minAvailableNodes": 2,
            "minTrainNodes": 2,
        }
    )
    topology["children"] = [topology["children"][0]]
    representation = NwdafMLModelTrainSubsc.model_validate(payload)
    coordinator = _coordinator(
        resolver,
        Mock(),
        server,
        ml_event="X_IMAGE_CLASSIFICATION",
    )

    report = coordinator.prepare_protocol(
        representation=representation,
        reservation_id="reservation-1",
    )

    statuses = {child.nf_instance_id: child.status for child in report.children}
    assert statuses == {
        LEAF_A: "ACTIVE",
        LEAF_B: "ACTIVE",
        leaf_c: "UNCONFIRMED",
    }
    resolver.discover_for_subscription.assert_called_once()
    first_target = server.start_protocol_preparation.call_args.kwargs["targets"]
    assert tuple(item.participant_nf_instance_id for item in first_target) == (LEAF_A,)
    added = server.add_protocol_preparation_targets.call_args.kwargs["targets"]
    assert tuple(item.participant_nf_instance_id for item in added) == (LEAF_B,)
    assert added[0].topology.strategy.method == "fedProx"
    assert added[0].topology.report_after.count == 2
    assert added[0].topology.report_after.unit == "epoch"
    repeated = coordinator.prepare_protocol(
        representation=representation,
        reservation_id="reservation-1",
    )
    assert repeated == report
    resolver.discover_for_subscription.assert_called_once()
    server.start_protocol_preparation.assert_called_once()
    server.add_protocol_preparation_targets.assert_called_once()
    coordinator.close()


def test_protocol_branch_patch_deletes_disabled_child_and_keeps_existing_process():
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    server = Mock()
    server.default_protocol_client_epochs = 2
    server.start_protocol_preparation.return_value = Mock(process_id=PLAN)
    server.collect_hierarchy_preparation.return_value = _protocol_collection(
        (LEAF_A, LEAF_B)
    )
    coordinator = _coordinator(
        resolver,
        Mock(),
        server,
        ml_event="X_IMAGE_CLASSIFICATION",
    )
    coordinator.prepare_protocol(
        representation=_protocol_representation(),
        reservation_id="reservation-1",
    )
    payload = _protocol_representation().model_dump(
        by_alias=True,
        exclude_none=True,
        mode="json",
    )
    payload["x-flTopology"]["policy"]["minAvailableNodes"] = 1
    payload["x-flTopology"]["policy"]["minTrainNodes"] = 1
    payload["x-flTopology"]["children"][0]["enabled"] = False

    report = coordinator.prepare_protocol(
        representation=NwdafMLModelTrainSubsc.model_validate(payload),
        reservation_id="reservation-1",
    )

    statuses = {child.nf_instance_id: child.status for child in report.children}
    assert statuses[LEAF_A] == "INACTIVE"
    assert statuses[LEAF_B] == "ACTIVE"
    server.remove_protocol_participant.assert_called_once_with(PLAN, LEAF_A)
    server.start_protocol_preparation.assert_called_once()
    coordinator.close()


def test_protocol_branch_reuses_adrf_then_uses_local_round_input(tmp_path):
    resolver = Mock()
    resolver.resolve.side_effect = [_node(LEAF_A), _node(LEAF_B)]
    artifacts = Mock()
    server = Mock()
    server.start_protocol_preparation.return_value = Mock(process_id=PLAN)
    server.collect_hierarchy_preparation.return_value = HierarchyPreparationCollection(
        process_id=PLAN,
        plan_id=PLAN,
        participants=tuple(
            HierarchyParticipantPreparationOutcome(
                participant_nf_instance_id=leaf_id,
                resource_location=f"http://{leaf_id}.example/subscriptions/1",
                notification=NwdafMLModelTrainNotif(
                    notifCorreId=f"callback-{leaf_id}",
                    mlCorreId=PLAN,
                    **{
                        "x-flTopologyReport": FlTopologyReport(
                            nfInstanceId=leaf_id
                        )
                    },
                ),
                failure="",
                delay_extensions=0,
                granted_extension_seconds=0,
            )
            for leaf_id in (LEAF_A, LEAF_B)
        ),
        timed_out_participant_nf_instance_ids=(),
    )
    validation_path = tmp_path / "validation.npz"
    np.savez(
        validation_path,
        images=np.zeros((2, 1, 28, 28), dtype=np.uint8),
        labels=np.asarray([0, 1], dtype=np.int64),
    )
    recorder = ExperimentRecorder(
        ExperimentRecordingSettings.model_validate(
            {
                "directory": tmp_path / "experiment-records",
                "validation": {
                    "dataset": "mnist",
                    "path": validation_path,
                    "batch_size": 2,
                },
            }
        )
    )
    recorder.open(BRANCH)
    coordinator = _coordinator(
        resolver,
        artifacts,
        server,
        ml_event="X_IMAGE_CLASSIFICATION",
        experiment_recorder=recorder,
    )
    coordinator.prepare_protocol(
        representation=_protocol_representation(),
        reservation_id="reservation-1",
    )
    upper_path = tmp_path / "root-round.tar.gz"
    upper_path.write_bytes(b"root-round")
    upper_manifest = {
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "X_IMAGE_CLASSIFICATION",
        "model_interoperability": "pymtlf-image-classification-mnist",
        "runtime_compatibility": {"framework": "torch"},
        "dataset": "mnist",
        "model": {"input_channels": 1, "num_classes": 10},
        "inference": {
            "input_shape": [1, 28, 28],
            "class_count": 10,
            "normalization": "uint8_to_float32_div_255",
        },
        "fl_metadata": {
            "ml_corre_id": PLAN,
            "round_ind": 3,
            "client_training": {"epochs": 2},
        },
    }
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))
    upper_input = Mock(manifest=upper_manifest, model=model)
    upper_artifact = ArtifactMetadata(
        key="1" * 64,
        size_bytes=upper_path.stat().st_size,
        path=upper_path,
        url="http://adrf.example/store/root-round/model",
    )
    first_global = _lower_global(tmp_path, 0, process_id=PLAN)
    second_global = _lower_global(tmp_path, 1, process_id=PLAN)
    server.execute_hierarchy_round.side_effect = [
        HierarchyRoundOutcome(
            aggregate=first_global,
            accepted=True,
            selected_participant_nf_instance_ids=(LEAF_A, LEAF_B),
            successful_participant_nf_instance_ids=(LEAF_A, LEAF_B),
            failed_participant_nf_instance_ids=(),
        ),
        HierarchyRoundOutcome(
            aggregate=second_global,
            accepted=True,
            selected_participant_nf_instance_ids=(LEAF_A, LEAF_B),
            successful_participant_nf_instance_ids=(LEAF_A, LEAF_B),
            failed_participant_nf_instance_ids=(),
        ),
    ]
    coordinator._loader = Mock()
    coordinator._loader.load.return_value = Mock(manifest=upper_manifest, model=model)
    local_input = Mock(
        url="http://branch.example/lower-input",
        digest="7" * 64,
        path=upper_path,
        contract=Mock(fl_metadata=Mock(round_ind=1)),
    )
    artifacts.publish_round_input.return_value = local_input
    upper_result = Mock(url="http://branch.example/upper-result")
    artifacts.publish_hierarchy_aggregate.return_value = upper_result
    payload = _protocol_representation().model_dump(
        by_alias=True,
        exclude_none=True,
        mode="json",
    )
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 3,
            "mLTrainRepInfo": {"maxResTime": 300},
            "mLModelInfos": [
                {
                    "event": "X_IMAGE_CLASSIFICATION",
                    "modelUniqueId": 91,
                    "mLModelAdrf": {
                        "adrfId": "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                        "storTransId": "root-round",
                    },
                }
            ],
        }
    )
    representation = NwdafMLModelTrainSubsc.model_validate(payload)

    result = coordinator.execute_protocol_round(
        representation=representation,
        upper_input=upper_input,
        upper_input_artifact=upper_artifact,
        upper_model=MLEventNotification(
            event="X_IMAGE_CLASSIFICATION",
            modelUniqueId=91,
            mLModelAdrf=MLModelAdrf(
                adrfId="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                storTransId="root-round",
            ),
        ),
        upper_client_subscription_id="upper-resource",
        upper_resource_revision=2,
        upper_training_scope=training_scope_descriptor(),
        callback_margin_seconds=5,
        local_work=IntermediateLocalWork(lower_round_count=2),
    )

    assert result is upper_result
    first, second = server.execute_hierarchy_round.call_args_list
    assert first.kwargs["round_input_model"].model_adrf is not None
    assert second.kwargs["round_input_model"] is None
    assert second.kwargs["round_input_artifact"] is local_input
    artifacts.publish_round_input.assert_called_once()
    coordinator.close()
    recorder.close()
    records = [
        json.loads(line)
        for line in (
            tmp_path / "experiment-records" / PLAN / "observations.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        record["evaluationStage"]
        for record in records
        if record["recordType"] == "MODEL_EVALUATION"
    ] == [
        "BRANCH_DOMAIN",
        "BRANCH_DOMAIN",
    ]
    assert [
        record["roundInd"] for record in records if record["recordType"] == "ROUND_AGGREGATION"
    ] == [0, 1]


def _lower_global(tmp_path, round_indicator: int, *, process_id="lower-process"):
    path = tmp_path / f"lower-global-{round_indicator}.tar.gz"
    path.write_bytes(f"lower-{round_indicator}".encode())
    contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "ml_corre_id": process_id,
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
