import hashlib
import json
import threading
from datetime import datetime
from unittest.mock import Mock

import httpx
import pytest
from nwdaf_context import context_client

from py_mtlf.config import FederatedLearningSettings, FLServerSettings
from py_mtlf.core.accuracy_policy import ScopeReference
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLClientResolver,
    FLParticipant,
    FLProcess,
    FLServerOrchestrator,
    FLServerState,
    _assign,
)
from py_mtlf.core.fl_workspace import preprocessing_contract_digest
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
    orchestrator = FLServerOrchestrator(
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
    orchestrator = FLServerOrchestrator(
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
    orchestrator = FLServerOrchestrator(
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
    orchestrator = FLServerOrchestrator(
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
    orchestrator = FLServerOrchestrator(
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


def test_aggregation_rejects_local_artifact_with_different_model_contract(tmp_path):
    participant_id = "11111111-1111-4111-8111-111111111111"
    scope_digest = "a" * 64
    base_weights_digest = hashlib.sha256(b"").hexdigest()
    base_manifest = {
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
    }
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
    orchestrator = FLServerOrchestrator(
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
        with pytest.raises(RuntimeError, match="identity does not match"):
            orchestrator._aggregate_round(process, Mock(), 0)
    finally:
        orchestrator.close()
