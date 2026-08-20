import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
import torch
from nwdaf_context import context_client

from py_mtlf.config import FederatedLearningSettings, FLServerSettings
from py_mtlf.core.accuracy_policy import ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    HierarchyBranchValidation,
    RoundGlobalArtifact,
    RoundLocalArtifact,
    RoundLocalResultType,
    ValidationSummary,
    WapeComponents,
)
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLClientResolver,
    FLParticipant,
    FLProcess,
    FLServerEngine,
    FLServerState,
    HierarchyPreparationTarget,
    HierarchyValidationCollection,
    _assign,
)
from py_mtlf.core.fl_workspace import (
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainNotif
from py_mtlf.wire.private import SelectedTarget


def candidate(nf_id: str, tac: str) -> FLClientCandidate:
    return FLClientCandidate(
        target=SelectedTarget(
            nfInstanceId=nf_id,
            nfServiceInstanceId=f"service-{tac}",
            serviceName="nnwdaf-mlmodeltraining",
            apiRoot=f"http://{tac}.example",
            selectionSource="NRF",
        ),
        tracking_areas=(f"466-92-{tac}",),
    )


def scope(name: str, tac: str, owner_id: str) -> ScopeReference:
    return ScopeReference(
        scope_key=name,
        consumer_id=owner_id,
        model_ids=(1,),
        ml_event="UE_COMMUNICATION",
        ml_event_filter={
            "networkArea": {
                "tais": [
                    {
                        "plmnId": {"mcc": "466", "mnc": "92"},
                        "tac": tac,
                    }
                ]
            }
        },
        target_ue={"intGroupIds": ["group-G"]},
    )


def validation_summary(
    participant_id: str,
    *,
    base_error: float,
    candidate_error: float,
) -> ValidationSummary:
    start = datetime(2026, 8, 20, tzinfo=UTC)
    return ValidationSummary(
        participant_nf_instance_id=participant_id,
        scope_digest="a" * 64,
        evaluation_sample_count=10,
        start_time=start,
        end_time=start + timedelta(minutes=1),
        base_model_weights_digest="b" * 64,
        candidate_weights_digest="c" * 64,
        base=WapeComponents(
            absolute_error_sum=base_error,
            absolute_actual_sum=100,
        ),
        candidate=WapeComponents(
            absolute_error_sum=candidate_error,
            absolute_actual_sum=100,
        ),
    )


def test_first_profile_freezes_one_distinct_client_per_tai_scope():
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    assignments = _assign(
        (
            scope("scope-b", "000002", second_id),
            scope("scope-a", "000001", first_id),
        ),
        (
            candidate("00000000-0000-4000-8000-000000000000", "000001"),
            candidate(second_id, "000002"),
            candidate(first_id, "000001"),
        ),
    )

    assert [
        (item_scope.scope_key, item_client.target.nf_instance_id)
        for item_scope, item_client in assignments
    ] == [("scope-a", first_id), ("scope-b", second_id)]


def test_assignment_does_not_substitute_another_eligible_same_tai_client():
    owner_id = "11111111-1111-4111-8111-111111111111"
    decoy_id = "00000000-0000-4000-8000-000000000000"

    with pytest.raises(RuntimeError, match=f"monitor owner {owner_id}"):
        _assign(
            (scope("scope-a", "000001", owner_id),),
            (candidate(decoy_id, "000001"),),
        )


def test_fl_client_discovery_requests_training_capability_for_scope_tai():
    target_id = "11111111-1111-4111-8111-111111111111"
    projection = context_client(
        nf_instance_id="33333333-3333-4333-8333-333333333333",
        api_root="http://go-c.example",
        internal_api_root="http://go-c-internal.example",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["target-nf-type"] == "NWDAF"
        assert params["requester-nf-type"] == "NWDAF"
        assert params["target-nf-instance-id"] == target_id
        assert params["service-names"] == "nnwdaf-mlmodeltraining"
        assert json.loads(params["ml-analytics-info-list"]) == [
            {
                "flCapabilityType": "FL_CLIENT",
                "mlAnalyticsIds": ["UE_COMMUNICATION"],
                "mlModelInterInfo": {"vendorList": ["001122"]},
                "trackingAreaList": [
                    {
                        "plmnId": {"mcc": "466", "mnc": "92"},
                        "tac": "000001",
                    }
                ],
            }
        ]
        return httpx.Response(
            200,
            json={
                "nfInstances": [
                    {
                        "nfInstanceId": target_id,
                        "nfStatus": "REGISTERED",
                        "nwdafInfo": {
                            "mlAnalyticsList": [
                                {
                                    "mlAnalyticsIds": ["UE_COMMUNICATION"],
                                    "trackingAreaList": [
                                        {
                                            "plmnId": {"mcc": "466", "mnc": "92"},
                                            "tac": "000001",
                                        }
                                    ],
                                    "mlModelInterInfo": {"vendorList": ["001122"]},
                                    "flCapabilityType": "FL_CLIENT",
                                }
                            ]
                        },
                        "nfServices": [
                            {
                                "serviceInstanceId": "training-a",
                                "serviceName": "nnwdaf-mlmodeltraining",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://nwdaf-a.example",
                            }
                        ],
                    }
                ]
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = FLClientResolver(FederatedLearningSettings(), projection, client)

    assert resolver.discover(scope("scope-a", "000001", target_id), "001122") == (
        FLClientCandidate(
            target=SelectedTarget(
                nfInstanceId=target_id,
                nfServiceInstanceId="training-a",
                serviceName="nnwdaf-mlmodeltraining",
                apiRoot="http://nwdaf-a.example",
                selectionSource="NRF",
            ),
            tracking_areas=("466-92-000001",),
        ),
    )
    client.close()


def test_server_process_reserves_shared_slot_until_cleanup_finishes(tmp_path):
    registry = FLExperimentRegistry()
    policy = Mock()
    intent = Mock(family_key=("UE_COMMUNICATION", "001122"))
    policy.take_intents.return_value = (intent,)
    catalog = Mock()
    catalog_entered = threading.Event()
    catalog_release = threading.Event()

    def current(_family_key):
        catalog_entered.set()
        assert catalog_release.wait(timeout=2)
        return None

    catalog.current.side_effect = current
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        policy,
        catalog,
        Mock(),
        Mock(),
        client=Mock(),
        experiments=registry,
    )

    try:
        orchestrator.accept_policy_intents()
        assert catalog_entered.wait(timeout=2)
        process = orchestrator.processes()[0]
        active = registry.active()
        assert active is not None
        assert active.server_process_id == process.process_id

        catalog_release.set()
        orchestrator.close()

        assert process.state is FLServerState.FAILED
        assert registry.active() is None
        policy.complete_retrain.assert_called_once_with(intent.family_key)
    finally:
        catalog_release.set()
        if not orchestrator._closing.is_set():
            orchestrator.close()


def test_server_process_rejects_conflict_with_active_client_group(tmp_path):
    registry = FLExperimentRegistry()
    registry.reserve_client("subscription-a", "correlation-a")
    policy = Mock()
    intent = Mock(family_key=("UE_COMMUNICATION", "001122"))
    policy.take_intents.return_value = (intent,)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        policy,
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
        experiments=registry,
    )

    try:
        orchestrator.accept_policy_intents()

        process = orchestrator.processes()[0]
        assert process.state is FLServerState.FAILED
        assert "top-level experiment" in process.failure
        assert registry.active().upper_client_subscription_ids == frozenset(
            {"subscription-a"}
        )
        policy.complete_retrain.assert_called_once_with(intent.family_key)
    finally:
        orchestrator.close()


def test_flat_server_publishes_round_input_with_server_owned_epochs(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    second_owner_id = "22222222-2222-4222-8222-222222222222"
    active_scopes = (
        scope("scope-a", "000001", owner_id),
        scope("scope-b", "000002", second_owner_id),
    )
    intent = SimpleNamespace(
        family_key="ue-communication-default",
        active_scope_keys=tuple(item.scope_key for item in active_scopes),
        active_scopes=active_scopes,
    )
    current = SimpleNamespace(
        artifact=SimpleNamespace(
            key="a" * 64,
            url="http://root.example/base.tar.gz",
        ),
        descriptor=SimpleNamespace(model_interoperability="001122"),
    )
    catalog = Mock()
    catalog.current.return_value = current
    resolver = Mock()
    resolver.discover.return_value = (
        candidate(owner_id, "000001"),
        candidate(second_owner_id, "000002"),
    )
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "33333333-3333-4333-8333-333333333333"
    )
    workspace = Mock()
    source = SimpleNamespace(name="flat-base")
    round_input = SimpleNamespace(url="http://root.example/round-input/0")
    workspace.publish_round_input.return_value = round_input
    policy = Mock()
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(
            round_count=1,
            client_training={"epochs": 9},
            cleanup={"max_attempts": 1},
        ),
        context,
        policy,
        catalog,
        workspace,
        resolver,
        client=client,
    )
    orchestrator._loader = Mock()
    orchestrator._loader.load.return_value = source

    def prepare(_process, participant, _model_interoperability, _base_url):
        participant.resource_location = (
            "http://go.example/subscriptions/"
            + participant.candidate.target.nf_instance_id
        )
        participant.preparation_complete = True

    def patch_round(_process, participant, *_args, **_kwargs):
        participant.notification = Mock()

    def patch_validation(_process, participant, *_args, **_kwargs):
        participant.notification = Mock()

    def evaluate_validation(process, *_args):
        process.gate_would_accept = True

    orchestrator._create_preparation = Mock(side_effect=prepare)
    orchestrator._wait = Mock()
    orchestrator._patch_round = Mock(side_effect=patch_round)
    orchestrator._aggregate_round = Mock(
        return_value=SimpleNamespace(url="http://root.example/round-global/0")
    )
    orchestrator._patch_validation = Mock(side_effect=patch_validation)
    orchestrator._evaluate_final_validation = Mock(side_effect=evaluate_validation)
    process = FLProcess(process_id="process-1", intent=intent)
    try:
        orchestrator._run(process)

        publication = workspace.publish_round_input.call_args.kwargs
        assert publication["base"] is source
        assert publication["process_id"] == process.process_id
        assert publication["round_indicator"] == 0
        assert publication["epochs"] == 9
        assert orchestrator._patch_round.call_args.args[3] == round_input.url
        assert process.candidate_url == "http://root.example/round-global/0"
        assert process.state is FLServerState.CANDIDATE_READY
    finally:
        orchestrator.close()


def test_hierarchy_preparation_attaches_root_process_and_uses_branch_bundle_urls(tmp_path):
    plan_id = "00000000-0000-4000-8000-000000000900"
    registry = FLExperimentRegistry()
    reservation = registry.reserve_root(plan_id)
    posts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(request)
            target_id = request.headers["X-NWDAF-Target-Nf-Instance-Id"]
            return httpx.Response(
                201,
                headers={"Location": f"http://go.example/subscriptions/{target_id}"},
                request=request,
            )
        return httpx.Response(204, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    context = context_client(
        nf_instance_id="00000000-0000-4000-8000-000000000001",
        internal_api_root="http://go.example",
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        context,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    targets = tuple(
        HierarchyPreparationTarget(
            participant_nf_instance_id=branch_id,
            candidate=FLClientCandidate(
                target=SelectedTarget(
                    nfInstanceId=branch_id,
                    nfServiceInstanceId=f"training-{index}",
                    serviceName="nnwdaf-mlmodeltraining",
                    apiRoot=f"http://branch-{index}.example",
                    selectionSource="NRF",
                ),
                tracking_areas=(),
            ),
            assignment_url=f"http://root.example/artifacts/assignment-{index}",
        )
        for index, branch_id in enumerate(
            (
                "00000000-0000-4000-8000-000000000010",
                "00000000-0000-4000-8000-000000000020",
            ),
            start=1,
        )
    )

    process = orchestrator.start_hierarchy_preparation(
        plan_id=plan_id,
        reservation_id=reservation.reservation_id,
        family_key="ue-communication-default",
        model_id=1,
        ml_event="UE_COMMUNICATION",
        ml_event_filter={"networkArea": {"tais": []}},
        target_ue=None,
        model_interoperability="001122",
        targets=targets,
    )

    assert process.state is FLServerState.PREPARATION_WAITING
    assert process.process_id != plan_id
    assert process.hierarchy_plan_id == plan_id
    assert registry.active().server_process_id == process.process_id
    assert len(posts) == 2
    for request, target in zip(posts, targets, strict=True):
        payload = json.loads(request.content)
        assert payload["mLPreFlag"] is True
        assert payload["mlCorreId"] == process.process_id
        assert payload["mLModelInfos"][0]["mLFileAddr"]["mLModelUrl"] == target.assignment_url
        assert (
            request.headers["X-NWDAF-Target-Nf-Instance-Id"]
            == target.participant_nf_instance_id
        )

    orchestrator.cancel_hierarchy_preparation(process.process_id, "test cleanup")
    orchestrator.close()
    client.close()


def test_hierarchy_preparation_rolls_back_partial_dispatch(tmp_path):
    plan_id = "00000000-0000-4000-8000-000000000900"
    registry = FLExperimentRegistry()
    reservation = registry.reserve_root(plan_id)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "POST" and len([item for item in calls if item[0] == "POST"]) == 1:
            return httpx.Response(
                201,
                headers={"Location": "http://go.example/subscriptions/first"},
                request=request,
            )
        if request.method == "POST":
            return httpx.Response(503, request=request)
        return httpx.Response(204, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        context_client(internal_api_root="http://go.example"),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    targets = tuple(
        HierarchyPreparationTarget(
            participant_nf_instance_id=branch_id,
            candidate=FLClientCandidate(
                target=SelectedTarget(
                    nfInstanceId=branch_id,
                    nfServiceInstanceId=f"training-{index}",
                    serviceName="nnwdaf-mlmodeltraining",
                    apiRoot=f"http://branch-{index}.example",
                    selectionSource="NRF",
                ),
                tracking_areas=(),
            ),
            assignment_url=f"http://root.example/artifacts/assignment-{index}",
        )
        for index, branch_id in enumerate(
            (
                "00000000-0000-4000-8000-000000000010",
                "00000000-0000-4000-8000-000000000020",
            ),
            start=1,
        )
    )

    with pytest.raises(RuntimeError, match="preparation create failed"):
        orchestrator.start_hierarchy_preparation(
            plan_id=plan_id,
            reservation_id=reservation.reservation_id,
            family_key="ue-communication-default",
            model_id=1,
            ml_event="UE_COMMUNICATION",
            ml_event_filter={},
            target_ue=None,
            model_interoperability="001122",
            targets=targets,
        )

    process = orchestrator.processes()[-1]
    assert process.state is FLServerState.FAILED
    assert process.hierarchy_cleanup_complete is True
    assert calls[-1] == ("DELETE", "http://go.example/subscriptions/first")
    assert registry.active().server_process_id is None
    orchestrator.close()
    client.close()


def test_cutover_pending_process_releases_slot_only_after_scope_adoption(tmp_path):
    registry = FLExperimentRegistry()
    policy = Mock()
    publication = Mock()
    publication.mark_scope_adopted.return_value = True
    intent = Mock(family_key=("UE_COMMUNICATION", "001122"))
    process = FLProcess(process_id="process-1", intent=intent)
    process.state = FLServerState.CUTOVER_PENDING
    process.published_model_id = 7
    reservation = registry.reserve_server(process.process_id)
    process.experiment_reservation_id = reservation.reservation_id
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        policy,
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
        publication=publication,
        experiments=registry,
    )
    orchestrator._processes[process.process_id] = process

    try:
        assert registry.active() is not None
        assert orchestrator.mark_scope_adopted(intent.family_key, 7, "scope-a") is True
        assert process.state is FLServerState.COMPLETE
        assert registry.active() is None
        policy.complete_retrain.assert_called_once_with(intent.family_key)
    finally:
        orchestrator.close()


def test_duplicate_delay_callback_is_acknowledged_without_second_extension(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="delay-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(process_id="process-1", intent=Mock())
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = [participant]
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": participant.notification_correlation_id,
            "mlCorreId": process.process_id,
            "delayEventNotif": {
                "delayEventInd": True,
                "delayCause": "NEED_MORE_TIME",
                "expCompTime": 30,
            },
        }
    )
    try:
        orchestrator.receive_notification(notification)
        first_digest = participant.accepted_delay_notification_digest
        orchestrator.receive_notification(notification)
        process.state = FLServerState.READY
        orchestrator.receive_notification(notification)

        assert participant.requested_extension == 30
        assert participant.accepted_delay_notification_digest == first_digest
        assert process.failure == ""
    finally:
        orchestrator.close()


def test_preparation_model_callback_is_stage_aware_success(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="preparation-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(process_id="process-1", intent=Mock())
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = [participant]
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": participant.notification_correlation_id,
            "mlCorreId": process.process_id,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://server.example/base.tar.gz"},
                }
            ],
        }
    )
    try:
        orchestrator.receive_notification(notification)

        assert participant.preparation_complete is True
        assert participant.preparation_notification == notification
        assert participant.notification is None
        assert process.failure == ""
    finally:
        orchestrator.close()


def test_preparation_result_is_recorded_before_termination(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="preparation-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
    )
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = [participant]
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": participant.notification_correlation_id,
            "mlCorreId": process.process_id,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://branch.example/preparation-result.tar.gz"
                    },
                }
            ],
            "termTrainReq": "NOT_AVAILABLE_ML_TRAIN",
        }
    )
    try:
        orchestrator.receive_notification(notification)

        assert participant.preparation_notification == notification
        assert participant.preparation_complete is True
        assert participant.preparation_notification.termination_request == (
            "NOT_AVAILABLE_ML_TRAIN"
        )
        assert process.failure == ""
    finally:
        orchestrator.close()


def test_flat_preparation_termination_still_fails_the_process(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="preparation-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(process_id="process-1", intent=Mock())
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = [participant]
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    try:
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participant.notification_correlation_id,
                mlCorreId=process.process_id,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )

        assert participant.preparation_complete is True
        assert "NOT_AVAILABLE_ML_TRAIN" in process.failure
    finally:
        orchestrator.close()


def test_hierarchy_collection_waits_for_every_leaf_after_first_failure(tmp_path):
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"preparation-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate((first_id, second_id), start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
    )
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = participants
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(preparation_timeout_seconds=2),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    completed = threading.Event()
    result = []

    def collect():
        result.append(orchestrator.collect_hierarchy_preparation(process.process_id))
        completed.set()

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participants[0].notification_correlation_id,
                mlCorreId=process.process_id,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )
        assert completed.wait(0.05) is False

        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[1].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://branch.example/leaf-assignment"
                            },
                        }
                    ],
                }
            )
        )
        assert completed.wait(1) is True
        assert result[0].timed_out_participant_nf_instance_ids == ()
        assert [
            item.notification.termination_request if item.notification is not None else None
            for item in result[0].participants
        ] == ["NOT_AVAILABLE_ML_TRAIN", None]
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[1].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://branch.example/leaf-assignment"
                            },
                        }
                    ],
                }
            )
        )
        assert process.state is FLServerState.PREPARATION_EVALUATING
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_validation_uses_existing_resources_and_next_round(tmp_path):
    branch_id = "11111111-1111-4111-8111-111111111111"
    process = FLProcess(
        process_id="process-1",
        intent=None,
        state=FLServerState.READY,
        hierarchy_plan_id="00000000-0000-4000-8000-000000000901",
        hierarchy_cleanup_complete=True,
    )
    participant = FLParticipant(
        scope=scope("branch-a", "000001", branch_id),
        candidate=candidate(branch_id, "000001"),
        notification_correlation_id="validation-branch-a",
        resource_location="http://go.example/subscriptions/branch-a",
    )
    process.participants = [participant]
    candidate_artifact = Mock(
        url="http://root.example/round-global/" + "c" * 64,
        digest="c" * 64,
    )
    collection = HierarchyValidationCollection(
        candidate_artifact=Mock(),
        validation_summaries=(
            validation_summary(branch_id, base_error=10, candidate_error=5),
        ),
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=300),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._validate_hierarchy_candidate = Mock()
    orchestrator._patch_validation = Mock()
    orchestrator._wait = Mock()
    orchestrator._collect_hierarchy_validation = Mock(return_value=collection)
    observed = []
    try:
        result = orchestrator.execute_hierarchy_validation(
            process_id=process.process_id,
            validation_round=2,
            candidate=candidate_artifact,
            base_artifact=Mock(),
            expected_candidate_process_id="process-1",
            expected_candidate_round=1,
            expected_subordinates={branch_id: ("leaf-a",)},
            state_observer=observed.append,
        )

        assert result is collection
        assert observed == [
            FLServerState.FINAL_VALIDATION_DISPATCH,
            FLServerState.FINAL_VALIDATION_WAITING,
            FLServerState.FINAL_VALIDATION_EVALUATING,
        ]
        patch = orchestrator._patch_validation.call_args
        assert patch.args[:3] == (process, participant, 2)
        assert patch.args[3] == candidate_artifact.url
        assert patch.kwargs["timeout_seconds"] == 300
        collected = orchestrator._collect_hierarchy_validation.call_args.kwargs
        assert collected["expected_candidate_process_id"] == "process-1"
        assert collected["expected_candidate_round"] == 1
        assert collected["expected_subordinates"] == {branch_id: ("leaf-a",)}
    finally:
        orchestrator.close()


def test_hierarchy_validation_rejects_invalid_candidate_before_branch_dispatch(
    tmp_path,
):
    branch_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("branch-a", "000001", branch_id),
        candidate=candidate(branch_id, "000001"),
        notification_correlation_id="validation-branch-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        state=FLServerState.READY,
        participants=[participant],
        hierarchy_plan_id="00000000-0000-4000-8000-000000000901",
        hierarchy_cleanup_complete=True,
    )
    invalid_candidate = Mock(
        url="http://root.example/not-round-global",
        contract=Mock(),
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=300),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._patch_validation = Mock()
    orchestrator._wait = Mock()
    try:
        with pytest.raises(RuntimeError, match="not a ROUND_GLOBAL"):
            orchestrator.execute_hierarchy_validation(
                process_id=process.process_id,
                validation_round=2,
                candidate=invalid_candidate,
                base_artifact=Mock(),
                expected_candidate_process_id=process.process_id,
                expected_candidate_round=1,
            )

        orchestrator._patch_validation.assert_not_called()
    finally:
        orchestrator.close()


def test_hierarchy_validation_rejects_mismatched_candidate_identity_before_dispatch(
    tmp_path,
    monkeypatch,
):
    branch_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("branch-a", "000001", branch_id),
        candidate=candidate(branch_id, "000001"),
        notification_correlation_id="validation-branch-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        state=FLServerState.READY,
        participants=[participant],
        hierarchy_plan_id="00000000-0000-4000-8000-000000000901",
        hierarchy_cleanup_complete=True,
    )
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    contract = RoundGlobalArtifact.model_validate(
        {
            "artifact_role": "ROUND_GLOBAL",
            "bundle_schema_version": "1.0",
            "file_digests": {
                "model.py": "1" * 64,
                "model.npy": "2" * 64,
                "scaler.pkl": "3" * 64,
            },
            "fl_metadata": {
                "contract_version": "1.0",
                "ml_corre_id": "different-process",
                "round_ind": 1,
                "model_contract_digest": "4" * 64,
                "preprocessing_contract_digest": "5" * 64,
                "base_weights_digest": "6" * 64,
                "weights_digest": "7" * 64,
                "participants": [
                    {
                        "participant_nf_instance_id": branch_id,
                        "training_sample_count": 10,
                        "local_artifact_digest": "8" * 64,
                    }
                ],
                "aggregated_training_sample_count": 10,
            },
        }
    )
    candidate_artifact = SimpleNamespace(
        contract=contract,
        digest="9" * 64,
        path=candidate_path,
        url="http://root.example/candidate",
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=300),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._patch_validation = Mock()
    orchestrator._loader = Mock()
    orchestrator._loader.load.side_effect = [
        SimpleNamespace(
            manifest={"bundle": "base"},
            model=SimpleNamespace(digest="6" * 64),
        ),
        SimpleNamespace(
            manifest={"bundle": "candidate"},
            model=SimpleNamespace(digest="7" * 64),
        ),
    ]
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.model_contract_digest",
        lambda _manifest: "4" * 64,
    )
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.preprocessing_contract_digest",
        lambda _manifest: "5" * 64,
    )
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.weights_digest",
        lambda model: model.digest,
    )
    try:
        with pytest.raises(RuntimeError, match="identity does not match"):
            orchestrator.execute_hierarchy_validation(
                process_id=process.process_id,
                validation_round=2,
                candidate=candidate_artifact,
                base_artifact=Mock(),
                expected_candidate_process_id=process.process_id,
                expected_candidate_round=1,
            )

        orchestrator._patch_validation.assert_not_called()
    finally:
        orchestrator.close()


def test_hierarchy_validation_deadline_records_every_missing_participant(tmp_path):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"branch-{index}", f"00000{index}", participant_id),
            candidate=candidate(participant_id, f"00000{index}"),
            notification_correlation_id=f"validation-branch-{index}",
            resource_location=f"http://go.example/subscriptions/branch-{index}",
        )
        for index, participant_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        state=FLServerState.READY,
        participants=participants,
        hierarchy_plan_id="00000000-0000-4000-8000-000000000901",
    )
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=1),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._validate_hierarchy_candidate = Mock()
    orchestrator._patch_validation = Mock()
    orchestrator._wait = Mock(
        side_effect=RuntimeError("federated stage deadline expired")
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            orchestrator.execute_hierarchy_validation(
                process_id=process.process_id,
                validation_round=2,
                candidate=Mock(url="http://root.example/candidate"),
                base_artifact=Mock(),
                expected_candidate_process_id="process-1",
                expected_candidate_round=1,
            )

        assert str(caught.value).endswith(",".join(participant_ids))
        assert [item.round_failure for item in participants] == [
            "validation deadline expired",
            "validation deadline expired",
        ]
        assert process.state is FLServerState.FAILED
        assert client.delete.call_count == 2
    finally:
        orchestrator.close()


def test_root_rejects_branch_validation_evidence_for_unassigned_leaf(
    tmp_path,
    monkeypatch,
):
    branch_id = "11111111-1111-4111-8111-111111111111"
    admitted_leaf = "22222222-2222-4222-8222-222222222222"
    wrong_leaf = "33333333-3333-4333-8333-333333333333"
    base_digest = "b" * 64
    candidate_digest = "c" * 64
    model_digest = "d" * 64
    preprocessing_digest = "e" * 64
    archive_digest = "f" * 64
    start = datetime(2026, 8, 20, tzinfo=UTC)
    subordinate = ValidationSummary(
        participant_nf_instance_id=wrong_leaf,
        scope_digest="1" * 64,
        evaluation_sample_count=10,
        start_time=start,
        end_time=start + timedelta(minutes=1),
        base_model_weights_digest=base_digest,
        candidate_weights_digest=candidate_digest,
        base=WapeComponents(absolute_error_sum=10, absolute_actual_sum=100),
        candidate=WapeComponents(absolute_error_sum=5, absolute_actual_sum=100),
    )
    candidate_contract = RoundGlobalArtifact.model_validate(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_GLOBAL",
            "file_digests": {
                "model.py": "1" * 64,
                "model.npy": "2" * 64,
                "scaler.pkl": "3" * 64,
            },
            "fl_metadata": {
                "contract_version": "1.0",
                "ml_corre_id": "root-process",
                "round_ind": 1,
                "model_contract_digest": model_digest,
                "preprocessing_contract_digest": preprocessing_digest,
                "base_weights_digest": base_digest,
                "weights_digest": candidate_digest,
                "participants": [
                    {
                        "participant_nf_instance_id": branch_id,
                        "training_sample_count": 20,
                        "local_artifact_digest": "4" * 64,
                    }
                ],
                "aggregated_training_sample_count": 20,
            },
        }
    )
    local_contract = RoundLocalArtifact.model_validate(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_LOCAL",
            "result_type": "ACCURACY_CHECK",
            "file_digests": {
                "model.py": "1" * 64,
                "model.npy": "2" * 64,
                "scaler.pkl": "3" * 64,
            },
            "fl_metadata": {
                "contract_version": "1.0",
                "ml_corre_id": "root-process",
                "round_ind": 2,
                "participant_nf_instance_id": branch_id,
                "scope_digest": "a" * 64,
                "input_global_weights_digest": candidate_digest,
                "model_contract_digest": model_digest,
                "preprocessing_contract_digest": preprocessing_digest,
                "base_weights_digest": candidate_digest,
                "weights_digest": candidate_digest,
                "evaluation": {
                    "evaluation_stage": "FINAL_VALIDATION",
                    "evaluation_sample_count": 10,
                    "start_time": start,
                    "end_time": start + timedelta(minutes=1),
                    "base_model_weights_digest": base_digest,
                    "candidate_weights_digest": candidate_digest,
                    "base": {
                        "absolute_error_sum": 10,
                        "absolute_actual_sum": 100,
                    },
                    "candidate": {
                        "absolute_error_sum": 5,
                        "absolute_actual_sum": 100,
                    },
                },
                "subordinate_validation_summaries": [
                    subordinate.model_dump(mode="json")
                ],
            },
        }
    )
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    candidate_artifact = Mock(
        contract=candidate_contract,
        digest=archive_digest,
        path=candidate_path,
        url="http://root.example/candidate/" + archive_digest,
    )
    participant = FLParticipant(
        scope=scope("branch-a", "000001", branch_id),
        candidate=candidate(branch_id, "000001"),
        notification_correlation_id="validation-branch-a",
        expected_scope_digest="a" * 64,
        notification=NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": "validation-branch-a",
                "mlCorreId": "root-process",
                "roundInd": 2,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {
                            "mLModelUrl": "http://branch.example/result"
                        },
                    }
                ],
            }
        ),
    )
    process = FLProcess(
        process_id="root-process",
        intent=None,
        participants=[participant],
        hierarchy_plan_id="00000000-0000-4000-8000-000000000901",
    )
    workspace = Mock()
    workspace.download.return_value = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        workspace,
        Mock(),
        client=Mock(),
    )
    orchestrator._loader = Mock()
    orchestrator._loader.load.side_effect = [
        SimpleNamespace(manifest={}, model=SimpleNamespace(digest=base_digest)),
        SimpleNamespace(manifest={}, model=SimpleNamespace(digest=candidate_digest)),
        SimpleNamespace(
            manifest=local_contract.model_dump(mode="json"),
            model=SimpleNamespace(digest=candidate_digest),
        ),
    ]
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.weights_digest",
        lambda model: model.digest,
    )
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.model_contract_digest",
        lambda _manifest: model_digest,
    )
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.preprocessing_contract_digest",
        lambda _manifest: preprocessing_digest,
    )
    try:
        with pytest.raises(RuntimeError, match="subordinate set"):
            orchestrator._collect_hierarchy_validation(
                process=process,
                candidate=candidate_artifact,
                base_artifact=Mock(),
                validation_round=2,
                expected_candidate_process_id="root-process",
                expected_candidate_round=1,
                expected_subordinates={branch_id: (admitted_leaf,)},
            )
    finally:
        orchestrator.close()


@pytest.mark.parametrize(
    (
        "enforce_gate",
        "candidate_error",
        "expected_state",
        "publishes",
        "stale_base",
        "has_active_scope",
    ),
    [
        (False, 12, FLServerState.COMPLETE, True, False, False),
        (True, 12, FLServerState.VALIDATION_REJECTED, False, False, False),
        (True, 5, FLServerState.COMPLETE, True, False, False),
        (True, 5, FLServerState.CUTOVER_PENDING, True, False, True),
        (True, 5, FLServerState.PUBLISHING, False, True, False),
    ],
)
def test_hierarchy_finalization_applies_leaf_gate_and_reuses_publication_owner(
    tmp_path,
    enforce_gate,
    candidate_error,
    expected_state,
    publishes,
    stale_base,
    has_active_scope,
):
    branch_id = "11111111-1111-4111-8111-111111111111"
    leaf_id = "22222222-2222-4222-8222-222222222222"
    plan_id = "00000000-0000-4000-8000-000000000901"
    family = "ue-communication-default"
    branch_summary = validation_summary(
        branch_id,
        base_error=10,
        candidate_error=candidate_error,
    )
    leaf_summary = validation_summary(
        leaf_id,
        base_error=10,
        candidate_error=candidate_error,
    )
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    retained_candidate = ArtifactMetadata(
        key="d" * 64,
        size_bytes=candidate_path.stat().st_size,
        path=candidate_path,
        url="http://root.example/candidate/" + "d" * 64,
    )
    collection = HierarchyValidationCollection(
        candidate_artifact=retained_candidate,
        validation_summaries=(branch_summary,),
        hierarchy_branches=(
            HierarchyBranchValidation(
                branch_nf_instance_id=branch_id,
                subordinate_validation_summaries=(leaf_summary,),
            ),
        ),
    )
    participant = FLParticipant(
        scope=scope("branch-a", "000001", branch_id),
        candidate=candidate(branch_id, "000001"),
        notification_correlation_id="validation-branch-a",
        training_sample_count=25,
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        state=FLServerState.READY,
        participants=[participant],
        hierarchy_plan_id=plan_id,
        hierarchy_family_key=family,
        hierarchy_active_scopes=(
            (scope("leaf-a", "000002", leaf_id),) if has_active_scope else ()
        ),
        hierarchy_cleanup_complete=True,
    )
    base_artifact = Mock(key="b" * 64)
    current = Mock(
        model_id=1,
        artifact=Mock(key="9" * 64) if stale_base else base_artifact,
    )
    catalog = Mock()
    catalog.current.return_value = current
    catalog.version_key_for_id.return_value = "version-1"
    publication = Mock()
    publication.publish.return_value = Mock(model_id=2, version_key="version-2")
    policy = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(
            final_validation={"enforce_performance_gate": enforce_gate}
        ),
        Mock(),
        policy,
        catalog,
        Mock(),
        Mock(),
        client=Mock(),
        publication=publication,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator.execute_hierarchy_validation = Mock(return_value=collection)
    candidate_artifact = Mock(digest="d" * 64)
    observed = []
    try:
        if stale_base:
            with pytest.raises(RuntimeError, match="catalog base changed"):
                orchestrator.finalize_hierarchy_candidate(
                    process_id=process.process_id,
                    validation_round=2,
                    candidate=candidate_artifact,
                    base_artifact=base_artifact,
                    expected_subordinates={branch_id: (leaf_id,)},
                    state_observer=observed.append,
                )
            publication.publish.assert_not_called()
            assert process.published_model_id is None
            assert observed[-2:] == [
                FLServerState.CANDIDATE_READY,
                FLServerState.PUBLISHING,
            ]
            return
        result = orchestrator.finalize_hierarchy_candidate(
            process_id=process.process_id,
            validation_round=2,
            candidate=candidate_artifact,
            base_artifact=base_artifact,
            expected_subordinates={branch_id: (leaf_id,)},
            state_observer=observed.append,
        )

        assert result.state is expected_state
        assert result.gate_would_accept is (candidate_error < 10)
        if publishes:
            published_candidate = publication.publish.call_args.args[0]
            assert published_candidate.hierarchy_validation.plan_id == plan_id
            assert published_candidate.validation_summaries == (branch_summary,)
            assert published_candidate.participants[0].sample_count == 25
            assert observed[-3:] == [
                FLServerState.CANDIDATE_READY,
                FLServerState.PUBLISHING,
                expected_state,
            ]
            if has_active_scope:
                policy.complete_retrain.assert_not_called()
            else:
                policy.complete_retrain.assert_called_once_with(family)
        else:
            publication.publish.assert_not_called()
            assert observed == [FLServerState.VALIDATION_REJECTED]
    finally:
        orchestrator.close()


@pytest.mark.parametrize("validation", [False, True])
def test_hierarchy_stage_waits_for_every_terminal_outcome_before_failure(
    tmp_path,
    validation,
):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"round-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=participants,
    )
    client = Mock()
    client.patch.return_value = Mock(status_code=204)
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=2),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._validate_hierarchy_candidate = Mock()
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    completed = threading.Event()
    failures = []

    def execute():
        try:
            if validation:
                orchestrator.execute_hierarchy_validation(
                    process_id=process.process_id,
                    validation_round=3,
                    candidate=Mock(url="http://root.example/candidate"),
                    base_artifact=Mock(),
                    expected_candidate_process_id="root-process",
                    expected_candidate_round=2,
                )
            else:
                orchestrator.execute_hierarchy_round(
                    process_id=process.process_id,
                    round_indicator=3,
                    round_input_url="http://root.example/round-input",
                    expected_result_type=RoundLocalResultType.TRAINING,
                )
        except Exception as error:
            failures.append(str(error))
        finally:
            completed.set()

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        dispatch_state = (
            FLServerState.FINAL_VALIDATION_DISPATCH
            if validation
            else FLServerState.ROUND_DISPATCH
        )
        waiting_state = (
            FLServerState.FINAL_VALIDATION_WAITING
            if validation
            else FLServerState.ROUND_WAITING
        )
        deadline = time.monotonic() + 1
        while process.state not in {dispatch_state, waiting_state}:
            if time.monotonic() >= deadline:
                raise AssertionError("hierarchy stage did not begin dispatch")
            time.sleep(0.001)
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participants[0].notification_correlation_id,
                mlCorreId=process.process_id,
                roundInd=3,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )
        assert completed.wait(0.05) is False

        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[1].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "roundInd": 3,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://leaf.example/local-result"
                            },
                        }
                    ],
                }
            )
        )
        assert completed.wait(1) is True
        prefix = (
            "required hierarchy validation participants terminated: "
            if validation
            else "required hierarchy participants terminated: "
        )
        assert failures == [prefix + participant_ids[0]]
        assert process.state is FLServerState.FAILED
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_round_patch_failure_cleans_lower_resources_before_return(tmp_path):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"round-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=participants,
    )
    client = Mock()
    client.patch.side_effect = [Mock(status_code=204), Mock(status_code=500)]
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )
    try:
        with pytest.raises(RuntimeError, match="round patch failed"):
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=3,
                round_input_url="http://branch.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
            )

        assert process.state is FLServerState.FAILED
        assert process.hierarchy_cleanup_complete is True
        assert client.delete.call_args_list == [
            call(participants[0].resource_location),
            call(participants[1].resource_location),
        ]
        assert all(
            participant.notification_correlation_id not in orchestrator._correlations
            for participant in participants
        )
    finally:
        orchestrator.close()


def test_hierarchy_round_deadline_marks_every_missing_participant_and_skips_aggregate(
    tmp_path,
):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"round-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=participants,
    )
    client = Mock()
    client.patch.return_value = Mock(status_code=204)
    client.delete.return_value = Mock(status_code=204)
    workspace = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        Mock(),
        Mock(),
        workspace,
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )
    orchestrator._wait = Mock(
        side_effect=RuntimeError("federated stage deadline expired")
    )
    try:
        with pytest.raises(RuntimeError, match="federated stage deadline expired"):
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=3,
                round_input_url="http://branch.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
            )

        assert process.state is FLServerState.FAILED
        assert [item.round_failure for item in participants] == [
            "round deadline expired",
            "round deadline expired",
        ]
        assert all(item.round_complete for item in participants)
        workspace.download.assert_not_called()
        workspace.publish.assert_not_called()
        assert client.delete.call_args_list == [
            call(participants[0].resource_location),
            call(participants[1].resource_location),
        ]
    finally:
        orchestrator.close()


def test_parent_cancel_during_hierarchy_round_wakes_waiter_and_cleans_resources(
    tmp_path,
):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="round-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=[participant],
    )
    client = Mock()
    client.patch.return_value = Mock(status_code=204)
    client.delete.return_value = Mock(status_code=204)
    workspace = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(
            round_timeout_seconds=30,
            cleanup={"max_attempts": 1},
        ),
        Mock(),
        Mock(),
        Mock(),
        workspace,
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    waiting = threading.Event()
    completed = threading.Event()
    failures = []

    def execute():
        try:
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=0,
                round_input_url="http://branch.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
                state_observer=lambda state: (
                    waiting.set() if state is FLServerState.ROUND_WAITING else None
                ),
            )
        except RuntimeError as error:
            failures.append(str(error))
        finally:
            completed.set()

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert waiting.wait(1) is True
        orchestrator.cancel_hierarchy_preparation(process.process_id, "parent cancelled")
        assert completed.wait(1) is True

        assert failures == ["parent cancelled"]
        assert process.state is FLServerState.FAILED
        assert process.hierarchy_cleanup_complete is True
        workspace.download.assert_not_called()
        workspace.publish.assert_not_called()
        client.delete.assert_called_once_with(participant.resource_location)
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_parent_cancel_during_lower_patch_fanout_fences_remaining_dispatches(
    tmp_path,
):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"round-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=participants,
    )
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )

    def patch_then_cancel(*_args, **_kwargs):
        orchestrator.cancel_hierarchy_preparation(
            process.process_id,
            "parent cancelled during fanout",
        )
        return Mock(status_code=204)

    client.patch.side_effect = patch_then_cancel
    try:
        with pytest.raises(RuntimeError, match="parent cancelled during fanout"):
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=0,
                round_input_url="http://branch.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
            )

        assert client.patch.call_count == 1
        assert process.state is FLServerState.FAILED
        assert process.hierarchy_cleanup_complete is True
    finally:
        orchestrator.close()


def test_hierarchy_wrong_round_records_failure_but_still_collects_other_outcomes(
    tmp_path,
):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    participants = [
        FLParticipant(
            scope=scope(f"scope-{index}", f"00000{index}", nf_id),
            candidate=candidate(nf_id, f"00000{index}"),
            notification_correlation_id=f"round-client-{index}",
            resource_location=f"http://go.example/subscriptions/resource-{index}",
        )
        for index, nf_id in enumerate(participant_ids, start=1)
    ]
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=participants,
    )
    client = Mock()
    client.patch.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(round_timeout_seconds=2),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )
    completed = threading.Event()
    failures = []

    def execute():
        try:
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=3,
                round_input_url="http://root.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
            )
        except Exception as error:
            failures.append(str(error))
        finally:
            completed.set()

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        while process.state is FLServerState.ROUND_DISPATCH:
            time.sleep(0.001)
        with pytest.raises(ValueError, match="ML_MODEL_TRAINING_REQS_NOT_MET"):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participants[0].notification_correlation_id,
                    mlCorreId=process.process_id,
                    roundInd=99,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                )
            )
        assert completed.wait(0.05) is False

        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[1].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "roundInd": 3,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://leaf.example/local-result"
                            },
                        }
                    ],
                }
            )
        )
        assert completed.wait(1) is True
        assert failures == [
            "required hierarchy participants terminated: " + participant_ids[0]
        ]
    finally:
        thread.join(timeout=1)
        orchestrator.close()


@pytest.mark.parametrize(
    "active_state",
    [FLServerState.ROUND_DISPATCH, FLServerState.FINAL_VALIDATION_DISPATCH],
)
def test_hierarchy_callback_during_dispatch_is_idempotent_but_conflict_fails(
    tmp_path,
    active_state,
):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="round-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
        expected_round=3,
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=active_state,
        participants=[participant],
    )
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id

    def notification(url: str) -> NwdafMLModelTrainNotif:
        return NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": participant.notification_correlation_id,
                "mlCorreId": process.process_id,
                "roundInd": 3,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {"mLModelUrl": url},
                    }
                ],
            }
        )

    accepted = notification("http://leaf.example/local-a.tar.gz")
    conflicting = notification("http://leaf.example/local-b.tar.gz")
    try:
        orchestrator.receive_notification(accepted)
        orchestrator.receive_notification(accepted)

        assert participant.round_complete is True
        assert participant.round_failure == ""
        with pytest.raises(ValueError, match="conflicting duplicate round callback"):
            orchestrator.receive_notification(conflicting)
        assert participant.round_failure == "conflicting duplicate round callback"
    finally:
        orchestrator.close()


def test_last_callback_accepted_at_deadline_wins_before_timeout(tmp_path, monkeypatch):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="round-client-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        participants=[participant],
    )
    clock = [100.0]
    monkeypatch.setattr(
        "py_mtlf.core.fl_server.time.monotonic",
        lambda: clock[0],
    )

    def complete_at_deadline(timeout=None):
        assert timeout == 1
        clock[0] = 101.0
        participant.round_complete = True

    process.condition.wait = complete_at_deadline
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    try:
        orchestrator._wait(
            process,
            lambda: participant.round_complete,
            1,
            collect_participant_failures=True,
        )
        assert participant.round_complete is True
        assert participant.round_failure == ""
    finally:
        orchestrator.close()


@pytest.mark.parametrize(
    "evaluating_state",
    [FLServerState.ROUND_EVALUATING, FLServerState.FINAL_VALIDATION_EVALUATING],
)
def test_callback_after_collection_freeze_allows_exact_duplicate_only(
    tmp_path,
    evaluating_state,
):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="round-client-a",
        expected_round=3,
    )
    accepted = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": participant.notification_correlation_id,
            "mlCorreId": "process-1",
            "roundInd": 3,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://leaf.example/local-a.tar.gz"
                    },
                }
            ],
        }
    )
    participant.notification = accepted
    participant.round_complete = True
    participant.accepted_notification_digest = hashlib.sha256(
        json.dumps(
            accepted.model_dump(by_alias=True, exclude_none=True, mode="json"),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=evaluating_state,
        participants=[participant],
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=Mock(),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    conflicting_payload = accepted.model_dump(
        by_alias=True,
        exclude_none=True,
        mode="json",
    )
    conflicting_payload["mLModelInfos"][0]["mLFileAddr"]["mLModelUrl"] = (
        "http://leaf.example/local-b.tar.gz"
    )
    conflicting = NwdafMLModelTrainNotif.model_validate(conflicting_payload)
    try:
        orchestrator.receive_notification(accepted)
        with pytest.raises(ValueError, match="after the active stage"):
            orchestrator.receive_notification(conflicting)

        assert process.state is evaluating_state
        assert participant.notification == accepted
        assert participant.round_failure == ""
    finally:
        process.hierarchy_cleanup_complete = True
        orchestrator.close()


def test_hierarchy_cancellation_wakes_collection_and_late_callback_is_rejected(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="preparation-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
    )
    process.state = FLServerState.PREPARATION_WAITING
    process.participants = [participant]
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(
            preparation_timeout_seconds=30,
            cleanup={"max_attempts": 1},
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    completed = threading.Event()
    failures = []

    def collect():
        try:
            orchestrator.collect_hierarchy_preparation(process.process_id)
        except RuntimeError as error:
            failures.append(str(error))
        finally:
            completed.set()

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        orchestrator.cancel_hierarchy_preparation(process.process_id, "parent cancelled")
        assert completed.wait(1) is True
        assert failures == ["parent cancelled"]

        process.state = FLServerState.READY
        process.failure = ""
        orchestrator._correlations[participant.notification_correlation_id] = process.process_id
        with pytest.raises(ValueError, match="after the active stage"):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participant.notification_correlation_id,
                    mlCorreId=process.process_id,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                )
            )
        assert process.failure == ""
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_preparation_uses_configured_historical_data_window(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    projection = context_client(
        nf_instance_id="33333333-3333-4333-8333-333333333333",
        api_root="http://go-c.example",
        internal_api_root="http://go-c-internal.example",
    )
    client = Mock()
    client.post.return_value = Mock(
        status_code=201,
        headers={"Location": "http://go.example/subscriptions/preparation-a"},
    )
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(preparation_data_window_seconds=3600),
        projection,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="preparation-client-a",
    )
    process = FLProcess(process_id="process-1", intent=Mock())
    try:
        orchestrator._create_preparation(
            process,
            participant,
            "001122",
            "http://server.example/base.tar.gz",
        )

        payload = client.post.call_args.kwargs["json"]
        window = payload["mLModelTrainInfos"][0]["dataAvReq"]["timeWindows"][0]
        start = datetime.fromisoformat(window["startTime"].replace("Z", "+00:00"))
        stop = datetime.fromisoformat(window["stopTime"].replace("Z", "+00:00"))
        assert (stop - start).total_seconds() == 3600
    finally:
        orchestrator.close()


def test_root_aggregation_weights_two_branch_results_by_effective_sample_count(
    tmp_path,
):
    branch_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    leaf_ids = (
        "33333333-3333-4333-8333-333333333333",
        "44444444-4444-4444-8444-444444444444",
    )
    scope_digests = ("a" * 64, "b" * 64)

    def model_with_weight(value: float) -> torch.nn.Module:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        model.eval()
        return model

    base_model = model_with_weight(0)
    branch_models = (model_with_weight(2), model_with_weight(10))
    base_weights_digest = weights_digest(base_model)
    file_digests = {
        "model.py": "1" * 64,
        "model.npy": "2" * 64,
        "scaler.pkl": "3" * 64,
    }
    base_manifest = {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 1},
        "inference": {"seq_length": 1},
        "file_digests": file_digests,
        "fl_metadata": {
            "contract_version": "1.0",
            "ml_corre_id": "process-1",
            "round_ind": 0,
            "model_contract_digest": "0" * 64,
            "preprocessing_contract_digest": "0" * 64,
            "weights_digest": base_weights_digest,
            "client_training": {"epochs": 2},
        },
    }
    base_manifest["fl_metadata"]["model_contract_digest"] = model_contract_digest(
        base_manifest
    )
    base_manifest["fl_metadata"][
        "preprocessing_contract_digest"
    ] = preprocessing_contract_digest(base_manifest)
    base = SimpleNamespace(manifest=base_manifest, model=base_model)

    branch_bundles = []
    sample_counts = (1, 3)
    for index, (branch_id, leaf_id, scope_digest, model, sample_count) in enumerate(
        zip(
            branch_ids,
            leaf_ids,
            scope_digests,
            branch_models,
            sample_counts,
            strict=True,
        ),
        start=1,
    ):
        manifest = {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_LOCAL",
            "result_type": "HIERARCHY_AGGREGATE",
            "file_digests": file_digests,
            "fl_metadata": {
                "contract_version": "1.0",
                "ml_corre_id": "process-1",
                "round_ind": 0,
                "participant_nf_instance_id": branch_id,
                "scope_digest": scope_digest,
                "model_contract_digest": model_contract_digest(base_manifest),
                "preprocessing_contract_digest": preprocessing_contract_digest(
                    base_manifest
                ),
                "input_global_weights_digest": base_weights_digest,
                "base_weights_digest": base_weights_digest,
                "weights_digest": weights_digest(model),
                "training_sample_count": sample_count,
                "lower_round_ind": 9 + index,
                "lower_global_artifact_digest": str(index + 5) * 64,
                "subordinate_participants": [
                    {
                        "participant_nf_instance_id": leaf_id,
                        "training_sample_count": sample_count,
                        "local_artifact_digest": str(index + 7) * 64,
                    }
                ],
            },
        }
        branch_bundles.append(SimpleNamespace(manifest=manifest, model=model))

    participants = []
    for index, (branch_id, scope_digest) in enumerate(
        zip(branch_ids, scope_digests, strict=True),
        start=1,
    ):
        participants.append(
            FLParticipant(
                scope=scope(f"branch-{index}", f"00000{index}", branch_id),
                candidate=candidate(branch_id, f"00000{index}"),
                notification_correlation_id=f"round-branch-{index}",
                expected_scope_digest=scope_digest,
                notification=NwdafMLModelTrainNotif.model_validate(
                    {
                        "notifCorreId": f"round-branch-{index}",
                        "mlCorreId": "process-1",
                        "roundInd": 0,
                        "mLModelInfos": [
                            {
                                "event": "UE_COMMUNICATION",
                                "mLFileAddr": {
                                    "mLModelUrl": (
                                        f"http://branch-{index}.example/aggregate.tar.gz"
                                    )
                                },
                            }
                        ],
                    }
                ),
            )
        )
    process = FLProcess(process_id="process-1", intent=None, participants=participants)
    workspace = Mock()
    workspace.download.side_effect = [
        SimpleNamespace(key="0" * 64),
        SimpleNamespace(key="4" * 64),
        SimpleNamespace(key="5" * 64),
    ]
    published = SimpleNamespace(url="http://root.example/round-global")
    workspace.publish.return_value = published
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        workspace,
        Mock(),
        client=Mock(),
    )
    orchestrator._loader = Mock()
    orchestrator._loader.load.side_effect = [base, *branch_bundles]
    try:
        result = orchestrator._aggregate_round(
            process,
            "http://root.example/round-input",
            0,
            expected_result_type=RoundLocalResultType.HIERARCHY_AGGREGATE,
            expected_subordinates={
                branch_id: (leaf_id,)
                for branch_id, leaf_id in zip(branch_ids, leaf_ids, strict=True)
            },
        )

        assert result is published
        publication = workspace.publish.call_args.kwargs
        assert publication["model"].weight.item() == pytest.approx(8.0)
        assert publication["metadata"]["fl_metadata"]["participants"] == [
            {
                "participant_nf_instance_id": branch_ids[0],
                "training_sample_count": 1,
                "local_artifact_digest": "4" * 64,
            },
            {
                "participant_nf_instance_id": branch_ids[1],
                "training_sample_count": 3,
                "local_artifact_digest": "5" * 64,
            },
        ]
        assert (
            publication["metadata"]["fl_metadata"][
                "aggregated_training_sample_count"
            ]
            == 4
        )
    finally:
        orchestrator.close()


def test_aggregation_rejects_local_artifact_with_different_model_contract(tmp_path):
    participant_id = "11111111-1111-4111-8111-111111111111"
    scope_digest = "a" * 64
    base_weights_digest = hashlib.sha256(b"").hexdigest()
    base_manifest = {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 10},
        "inference": {"seq_length": 30},
        "file_digests": {
            "model.py": "1" * 64,
            "model.npy": "2" * 64,
            "scaler.pkl": "3" * 64,
        },
        "fl_metadata": {
            "contract_version": "1.0",
            "ml_corre_id": "process-1",
            "round_ind": 0,
            "model_contract_digest": "0" * 64,
            "preprocessing_contract_digest": "1" * 64,
            "weights_digest": base_weights_digest,
            "client_training": {"epochs": 1},
        },
    }
    base_manifest["fl_metadata"]["model_contract_digest"] = model_contract_digest(
        base_manifest
    )
    base_manifest["fl_metadata"][
        "preprocessing_contract_digest"
    ] = preprocessing_contract_digest(base_manifest)
    local_manifest = {
        "bundle_schema_version": "1.0",
        "file_digests": {
            "model.py": "1" * 64,
            "model.npy": "4" * 64,
            "scaler.pkl": "3" * 64,
        },
        "artifact_role": "ROUND_LOCAL",
        "result_type": "TRAINING",
        "fl_metadata": {
            "contract_version": "1.0",
            "ml_corre_id": "process-1",
            "model_contract_digest": "f" * 64,
            "preprocessing_contract_digest": preprocessing_contract_digest(base_manifest),
            "base_weights_digest": base_weights_digest,
            "weights_digest": "5" * 64,
            "round_ind": 0,
            "participant_nf_instance_id": participant_id,
            "scope_digest": scope_digest,
            "input_global_weights_digest": base_weights_digest,
            "training_sample_count": 10,
        },
    }
    base = Mock(manifest=base_manifest)
    base.model.state_dict.return_value = {}
    local = Mock(manifest=local_manifest)
    workspace = Mock()
    workspace.download.return_value = Mock(key="6" * 64)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        workspace,
        Mock(),
        client=Mock(),
    )
    orchestrator._loader = Mock()
    orchestrator._loader.load.side_effect = [base, local]
    process = FLProcess(process_id="process-1", intent=Mock())
    participant = FLParticipant(
        scope=scope("scope-a", "000001", participant_id),
        candidate=candidate(participant_id, "000001"),
        notification_correlation_id="round-a",
        expected_scope_digest=scope_digest,
        notification=NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": "round-a",
                "mlCorreId": "process-1",
                "roundInd": 0,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {"mLModelUrl": "http://client.example/local.tar.gz"},
                    }
                ],
            }
        ),
    )
    process.participants = [participant]
    try:
        duplicate = participant.notification.model_copy(deep=True)
        duplicate.ml_model_infos.append(duplicate.ml_model_infos[0].model_copy(deep=True))
        participant.notification = duplicate
        orchestrator._loader.load.side_effect = [base]
        with pytest.raises(RuntimeError, match="notification is missing or invalid"):
            orchestrator._aggregate_round(process, Mock(), 0)

        participant.notification = NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": "round-a",
                "mlCorreId": "process-1",
                "roundInd": 0,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {
                            "mLModelUrl": "http://client.example/local.tar.gz"
                        },
                    }
                ],
            }
        )
        orchestrator._loader.load.side_effect = [base, local]
        with pytest.raises(RuntimeError, match="identity does not match"):
            orchestrator._aggregate_round(process, Mock(), 0)
    finally:
        orchestrator.close()
