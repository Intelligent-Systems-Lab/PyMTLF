import copy
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
import torch
from conftest import training_scope_descriptor
from nwdaf_context import context_client

from py_mtlf.config import FederatedLearningSettings, FLServerSettings
from py_mtlf.core.accuracy_policy import ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    RoundLocalArtifact,
    RoundLocalResultType,
    ValidationSummary,
    WapeComponents,
)
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_orchestration import (
    FlatExecutionRequest,
    FlatParticipantScope,
    MonitorParticipantSelection,
    StaticParticipantSelection,
    TriggerSource,
)
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLClientResolver,
    FLParticipant,
    FLProcess,
    FLServerAdmissionClosedError,
    FLServerEngine,
    FLServerProcessConflictError,
    FLServerState,
    ProtocolPreparationTarget,
    _assign,
)
from py_mtlf.wire.ml_model_training import (
    FlTopologyNode,
    NwdafMLModelTrainNotif,
)
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


def discovery_profile(target_id: str) -> dict:
    return {
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


def flat_execution(
    family_key: str,
    scopes: tuple[ScopeReference, ...] = (),
) -> FlatExecutionRequest:
    return FlatExecutionRequest(
        model_family_id=family_key,
        trigger_source=TriggerSource.DEGRADATION,
        participant_selection=MonitorParticipantSelection(
            participants=tuple(
                FlatParticipantScope.from_monitor_scope(item) for item in scopes
            )
        ),
        required_cutover_scope_keys=tuple(item.scope_key for item in scopes),
        triggering_scope_key=scopes[0].scope_key if scopes else None,
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
        training_scope=training_scope_descriptor(),
        evaluation_sample_count=10,
        start_time=start,
        end_time=start + timedelta(minutes=1),
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

    with pytest.raises(RuntimeError, match=f"configured participant {owner_id}"):
        _assign(
            (scope("scope-a", "000001", owner_id),),
            (candidate(decoy_id, "000001"),),
        )


def test_protocol_preparation_is_model_free_and_accepts_topology_only_callback(caplog):
    caplog.set_level(logging.INFO, logger="py_mtlf.core.fl_server")
    root_id = "11111111-1111-4111-8111-111111111111"
    branch_id = "22222222-2222-4222-8222-222222222222"
    procedure_id = "99999999-9999-4999-8999-999999999999"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204, request=request)
        payload = json.loads(request.content)
        return httpx.Response(
            201,
            request=request,
            headers={"Location": "http://branch.example/subscriptions/resource-a"},
            json=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = FLExperimentRegistry()
    reservation = registry.reserve_root(procedure_id)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root="/tmp/fl-server-protocol-test"),
        FLServerSettings(),
        context_client(
            nf_instance_id=root_id,
            api_root="http://root.example",
            internal_api_root="http://root-go.example",
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    topology = FlTopologyNode.model_validate(
        {
            "nfInstanceId": branch_id,
            "policy": {
                "selectionMethod": "priority",
                "minAvailableNodes": 1,
                "fractionTrain": 1.0,
                "minTrainNodes": 1,
                "acceptFailures": False,
            },
            "strategy": {
                "method": "fedProx",
                "aggregation": "sampleWeighted",
                "methodParameters": {"proximalMu": 0.01},
            },
            "reportAfter": {"count": 1, "unit": "round"},
            "children": [
                {
                    "nfInstanceId": "33333333-3333-4333-8333-333333333333",
                    "priority": 10,
                }
            ],
        }
    )
    try:
        process = orchestrator.start_protocol_preparation(
            ml_correlation_id=procedure_id,
            reservation_id=reservation.reservation_id,
            ml_event="X_IMAGE_CLASSIFICATION",
            ml_event_filter={},
            model_interoperability="pymtlf-image-classification-mnist",
            targets=(
                ProtocolPreparationTarget(
                    participant_nf_instance_id=branch_id,
                    candidate=candidate(branch_id, "branch"),
                    topology=topology,
                ),
            ),
        )
        create_payload = json.loads(requests[0].content)
        assert create_payload["mlCorreId"] == procedure_id
        assert create_payload["suppFeats"] == "4"
        assert create_payload["mLPreFlag"] is True
        assert "mLModelInfos" not in create_payload
        data_requirement = create_payload["mLModelTrainInfos"][0]["dataAvReq"]
        assert data_requirement["inpEvents"] == [
            {"nwdafEvent": "X_IMAGE_CLASSIFICATION"}
        ]
        assert data_requirement["minNumSamples"] == 1
        assert create_payload["mLModelTrainInfos"][0]["timeAvReq"] == "PT300S"
        assert create_payload["x-flTopology"]["nfInstanceId"] == branch_id
        assert create_payload["mLEventSubscs"] == [
            {
                "mLEvent": "X_IMAGE_CLASSIFICATION",
                "mLEventFilter": {},
                "modelInterInfo": "pymtlf-image-classification-mnist",
                "useCaseCxt": "",
            }
        ]

        participant = process.participants[0]
        assert (
            "FL participant resource created "
            f"process_id={procedure_id} nf={branch_id} "
            f"notif_corre_id={participant.notification_correlation_id} "
            "location=http://branch.example/subscriptions/resource-a"
        ) in caplog.text
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participant.notification_correlation_id,
                    "mlCorreId": procedure_id,
                    "x-flTopologyReport": {
                        "nfInstanceId": branch_id,
                    },
                }
            )
        )
        collected = orchestrator.collect_hierarchy_preparation(process.process_id)

        assert collected.participants[0].notification.fl_topology_report.nf_instance_id == branch_id
        assert collected.timed_out_participant_nf_instance_ids == ()
    finally:
        orchestrator.close()
        registry.reset_generation()
        client.close()


def test_protocol_feature_mismatch_is_a_participant_failure():
    root_id = "11111111-1111-4111-8111-111111111111"
    branch_id = "22222222-2222-4222-8222-222222222222"
    procedure_id = "99999999-9999-4999-8999-999999999999"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204, request=request)
        payload = json.loads(request.content)
        payload.pop("suppFeats", None)
        return httpx.Response(
            201,
            request=request,
            headers={"Location": "http://branch.example/subscriptions/resource-a"},
            json=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = FLExperimentRegistry()
    reservation = registry.reserve_root(procedure_id)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root="/tmp/fl-server-protocol-test"),
        FLServerSettings(cleanup={"max_attempts": 1}),
        context_client(
            nf_instance_id=root_id,
            api_root="http://root.example",
            internal_api_root="http://root-go.example",
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    topology = FlTopologyNode.model_validate(
        {
            "nfInstanceId": branch_id,
            "policy": {
                "selectionMethod": "priority",
                "minAvailableNodes": 1,
                "fractionTrain": 1.0,
                "minTrainNodes": 1,
                "acceptFailures": False,
            },
            "strategy": {
                "method": "fedProx",
                "aggregation": "sampleWeighted",
                "methodParameters": {"proximalMu": 0.01},
            },
            "reportAfter": {"count": 1, "unit": "round"},
        }
    )
    try:
        process = orchestrator.start_protocol_preparation(
            ml_correlation_id=procedure_id,
            reservation_id=reservation.reservation_id,
            ml_event="X_IMAGE_CLASSIFICATION",
            ml_event_filter={},
            model_interoperability="pymtlf-image-classification-mnist",
            targets=(
                ProtocolPreparationTarget(
                    participant_nf_instance_id=branch_id,
                    candidate=candidate(branch_id, "branch"),
                    topology=topology,
                ),
            ),
        )
        collected = orchestrator.collect_hierarchy_preparation(process.process_id)

        assert process.state is FLServerState.PREPARATION_EVALUATING
        assert collected.participants[0].failure == "FEATURE_NOT_SUPPORTED"
        assert [request.method for request in requests] == ["POST", "DELETE"]
    finally:
        orchestrator.close()
        registry.reset_generation()
        client.close()


def test_protocol_process_adds_and_removes_participants_after_admission():
    root_id = "11111111-1111-4111-8111-111111111111"
    branch_a = "22222222-2222-4222-8222-222222222222"
    branch_b = "33333333-3333-4333-8333-333333333333"
    procedure_id = "99999999-9999-4999-8999-999999999999"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204, request=request)
        payload = json.loads(request.content)
        target = request.headers["x-nwdaf-target-nf-instance-id"]
        return httpx.Response(
            201,
            request=request,
            headers={"Location": f"http://branch.example/subscriptions/{target}"},
            json=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = FLExperimentRegistry()
    reservation = registry.reserve_root(procedure_id)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root="/tmp/fl-server-protocol-test"),
        FLServerSettings(cleanup={"max_attempts": 1}),
        context_client(
            nf_instance_id=root_id,
            api_root="http://root.example",
            internal_api_root="http://root-go.example",
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )

    def topology(nf_instance_id: str) -> FlTopologyNode:
        return FlTopologyNode.model_validate(
            {
                "nfInstanceId": nf_instance_id,
                "policy": {
                    "selectionMethod": "priority",
                    "minAvailableNodes": 1,
                    "fractionTrain": 1.0,
                    "minTrainNodes": 1,
                    "acceptFailures": False,
                },
                "strategy": {
                    "method": "fedProx",
                    "aggregation": "sampleWeighted",
                    "methodParameters": {"proximalMu": 0.01},
                },
                "reportAfter": {"count": 1, "unit": "round"},
            }
        )

    def target(nf_instance_id: str) -> ProtocolPreparationTarget:
        return ProtocolPreparationTarget(
            participant_nf_instance_id=nf_instance_id,
            candidate=candidate(nf_instance_id, nf_instance_id[:4]),
            topology=topology(nf_instance_id),
        )

    try:
        process = orchestrator.start_protocol_preparation(
            ml_correlation_id=procedure_id,
            reservation_id=reservation.reservation_id,
            ml_event="X_IMAGE_CLASSIFICATION",
            ml_event_filter={},
            model_interoperability="pymtlf-image-classification-mnist",
            targets=(target(branch_a),),
        )
        first = process.participants[0]
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": first.notification_correlation_id,
                    "mlCorreId": procedure_id,
                    "x-flTopologyReport": {"nfInstanceId": branch_a},
                }
            )
        )
        orchestrator.collect_hierarchy_preparation(process.process_id)
        orchestrator.admit_hierarchy_preparation(process.process_id)

        orchestrator.add_protocol_preparation_targets(
            process_id=process.process_id,
            ml_event="X_IMAGE_CLASSIFICATION",
            ml_event_filter={},
            model_interoperability="pymtlf-image-classification-mnist",
            targets=(target(branch_b),),
        )
        second = next(
            participant
            for participant in process.participants
            if participant.candidate.target.nf_instance_id == branch_b
        )
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": second.notification_correlation_id,
                    "mlCorreId": procedure_id,
                    "x-flTopologyReport": {"nfInstanceId": branch_b},
                }
            )
        )
        collected = orchestrator.collect_hierarchy_preparation(process.process_id)
        orchestrator.admit_hierarchy_preparation(process.process_id)

        assert tuple(
            item.participant_nf_instance_id for item in collected.participants
        ) == (branch_a, branch_b)
        orchestrator.remove_protocol_participant(process.process_id, branch_a)
        assert tuple(
            participant.candidate.target.nf_instance_id
            for participant in process.participants
        ) == (branch_b,)
        assert [request.method for request in requests] == ["POST", "POST", "DELETE"]
    finally:
        orchestrator.close()
        registry.reset_generation()
        client.close()


def test_protocol_replacement_prepares_independently_of_ready_process_cohort(tmp_path):
    root_id = "11111111-1111-4111-8111-111111111111"
    existing_id = "22222222-2222-4222-8222-222222222222"
    replacement_id = "33333333-3333-4333-8333-333333333333"
    procedure_id = "99999999-9999-4999-8999-999999999999"
    request_started = threading.Event()
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204, request=request)
        payload = json.loads(request.content)
        request_started.set()
        return httpx.Response(
            201,
            request=request,
            headers={"Location": "http://replacement.example/subscriptions/new"},
            json=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(preparation_timeout_seconds=2),
        context_client(
            nf_instance_id=root_id,
            api_root="http://root.example",
            internal_api_root="http://root-go.example",
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    existing = FLParticipant(
        scope=scope("existing", "000001", existing_id),
        candidate=candidate(existing_id, "000001"),
        notification_correlation_id="existing-correlation",
        resource_location="http://existing.example/subscriptions/old",
    )
    process = FLProcess(
        process_id=procedure_id,
        intent=None,
        hierarchy_plan_id=procedure_id,
        protocol_hierarchy=True,
        state=FLServerState.READY,
        participants=[existing],
        hierarchy_selected_participant_ids=frozenset({existing_id}),
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[existing.notification_correlation_id] = process.process_id
    topology = FlTopologyNode.model_validate(
        {
            "nfInstanceId": replacement_id,
            "policy": {
                "selectionMethod": "priority",
                "minAvailableNodes": 1,
                "fractionTrain": 1.0,
                "minTrainNodes": 1,
                "acceptFailures": False,
            },
            "strategy": {
                "method": "fedProx",
                "aggregation": "sampleWeighted",
                "methodParameters": {"proximalMu": 0.01},
            },
            "reportAfter": {"count": 1, "unit": "round"},
        }
    )
    result = []
    failures = []

    def prepare():
        try:
            result.append(
                orchestrator.prepare_protocol_replacement_target(
                    process_id=process.process_id,
                    ml_event="X_IMAGE_CLASSIFICATION",
                    ml_event_filter={},
                    model_interoperability="pymtlf-image-classification-mnist",
                    target=ProtocolPreparationTarget(
                        participant_nf_instance_id=replacement_id,
                        candidate=candidate(replacement_id, "000002"),
                        topology=topology,
                    ),
                )
            )
        except Exception as error:
            failures.append(str(error))

    thread = threading.Thread(target=prepare)
    thread.start()
    try:
        assert request_started.wait(1)
        replacement = next(
            participant
            for participant in process.participants
            if participant.candidate.target.nf_instance_id == replacement_id
        )
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": replacement.notification_correlation_id,
                    "mlCorreId": procedure_id,
                    "x-flTopologyReport": {"nfInstanceId": replacement_id},
                }
            )
        )
        thread.join(timeout=1)

        assert thread.is_alive() is False
        assert failures == []
        assert result[0].participant_nf_instance_id == replacement_id
        assert result[0].failure == ""
        assert process.state is FLServerState.READY
        assert process.hierarchy_selected_participant_ids == frozenset({existing_id})
        payload = json.loads(requests[0].content)
        assert payload["mlCorreId"] == procedure_id
        assert payload["notifCorreId"] != existing.notification_correlation_id
    finally:
        thread.join(timeout=1)
        orchestrator.close()
        client.close()


def test_protocol_participant_retirement_fences_local_identity_when_peer_cleanup_fails(
    tmp_path,
):
    participant_id = "22222222-2222-4222-8222-222222222222"
    participant = FLParticipant(
        scope=scope("existing", "000001", participant_id),
        candidate=candidate(participant_id, "000001"),
        notification_correlation_id="failed-branch-correlation",
        resource_location="http://failed.example/subscriptions/old",
    )
    process = FLProcess(
        process_id="99999999-9999-4999-8999-999999999999",
        intent=None,
        hierarchy_plan_id="99999999-9999-4999-8999-999999999999",
        protocol_hierarchy=True,
        state=FLServerState.READY,
        participants=[participant],
    )
    client = Mock()
    client.delete.return_value = Mock(status_code=503, text="unavailable")
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
    orchestrator._correlations[participant.notification_correlation_id] = (
        process.process_id
    )
    try:
        cleanup_failure = orchestrator.remove_protocol_participant(
            process.process_id,
            participant_id,
        )

        assert "cleanup returned 503" in cleanup_failure
        assert process.participants == []
        with pytest.raises(KeyError):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participant.notification_correlation_id,
                    mlCorreId=process.process_id,
                    roundInd=0,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                )
            )
        assert process.state is FLServerState.READY
    finally:
        orchestrator.close()


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


@pytest.mark.parametrize(
    "mismatch",
    [
        "identity",
        "nf_status",
        "service_name",
        "service_status",
        "capability",
        "event",
        "interoperability",
        "tracking_area",
    ],
)
def test_fl_client_discovery_rejects_exact_profile_mismatch(mismatch):
    target_id = "11111111-1111-4111-8111-111111111111"
    profile = discovery_profile(target_id)
    analytics = profile["nwdafInfo"]["mlAnalyticsList"][0]
    service = profile["nfServices"][0]
    if mismatch == "identity":
        profile["nfInstanceId"] = "22222222-2222-4222-8222-222222222222"
    elif mismatch == "nf_status":
        profile["nfStatus"] = "SUSPENDED"
    elif mismatch == "service_name":
        service["serviceName"] = "nnwdaf-analyticsinfo"
    elif mismatch == "service_status":
        service["nfServiceStatus"] = "SUSPENDED"
    elif mismatch == "capability":
        analytics["flCapabilityType"] = "FL_SERVER"
    elif mismatch == "event":
        analytics["mlAnalyticsIds"] = ["DN_PERFORMANCE"]
    elif mismatch == "interoperability":
        analytics["mlModelInterInfo"] = {"vendorList": ["different"]}
    else:
        analytics["trackingAreaList"][0]["tac"] = "000999"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"nfInstances": [profile]},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = FLClientResolver(
        FederatedLearningSettings(),
        context_client(
            nf_instance_id="33333333-3333-4333-8333-333333333333",
            internal_api_root="http://go-internal.example",
        ),
        client,
    )
    try:
        participant_scope = FlatParticipantScope.from_monitor_scope(
            scope("scope-a", "000001", target_id)
        )
        assert resolver.discover(participant_scope, "001122") == ()
    finally:
        client.close()


def test_assignment_rejects_duplicate_exact_training_service_candidates():
    target_id = "11111111-1111-4111-8111-111111111111"
    first = candidate(target_id, "000001")
    second = copy.deepcopy(first)
    second = FLClientCandidate(
        target=second.target.model_copy(
            update={"nf_service_instance_id": "training-duplicate"}
        ),
        tracking_areas=second.tracking_areas,
    )

    with pytest.raises(RuntimeError, match="eligible unique FL Client"):
        _assign((scope("scope-a", "000001", target_id),), (first, second))


def test_flat_discovery_validates_every_participant_before_preparation_dispatch(tmp_path):
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    scopes = (
        scope("scope-a", "000001", first_id),
        scope("scope-b", "000002", second_id),
    )
    catalog = Mock()
    catalog.current.return_value = SimpleNamespace(
        artifact=Mock(key="a" * 64, url="http://root.example/base.tar.gz"),
        descriptor=Mock(model_interoperability="001122"),
    )
    resolver = Mock()
    resolver.discover.side_effect = ((candidate(first_id, "000001"),), ())
    policy = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        Mock(),
        policy,
        catalog,
        Mock(),
        resolver,
        client=Mock(),
    )
    orchestrator._create_preparation = Mock()
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution("ue-communication-default", scopes),
    )
    try:
        orchestrator._run(process)

        assert process.state is FLServerState.FAILED
        orchestrator._create_preparation.assert_not_called()
        policy.complete_retrain.assert_called_once_with("ue-communication-default")
    finally:
        orchestrator.close()


def test_flat_discovery_rejects_identical_duplicate_candidates_before_dispatch(tmp_path):
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    scopes = (
        scope("scope-a", "000001", first_id),
        scope("scope-b", "000002", second_id),
    )
    catalog = Mock()
    catalog.current.return_value = SimpleNamespace(
        artifact=Mock(key="a" * 64, url="http://root.example/base.tar.gz"),
        descriptor=Mock(model_interoperability="001122"),
    )
    resolver = Mock()
    duplicate = candidate(first_id, "000001")
    resolver.discover.side_effect = (
        (duplicate, duplicate),
        (candidate(second_id, "000002"),),
    )
    policy = Mock()
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(preparation_timeout_seconds=1),
        Mock(),
        policy,
        catalog,
        Mock(),
        resolver,
        client=Mock(),
    )
    orchestrator._create_preparation = Mock()
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution("ue-communication-default", scopes),
    )
    try:
        orchestrator._run(process)

        assert process.state is FLServerState.FAILED
        assert "eligible unique FL Client" in process.failure
        orchestrator._create_preparation.assert_not_called()
        policy.complete_retrain.assert_called_once_with("ue-communication-default")
    finally:
        orchestrator.close()


def test_flat_partial_preparation_dispatch_cleans_created_resource(tmp_path):
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    scopes = (
        scope("scope-a", "000001", first_id),
        scope("scope-b", "000002", second_id),
    )
    catalog = Mock()
    catalog.current.return_value = SimpleNamespace(
        artifact=Mock(key="a" * 64, url="http://root.example/base.tar.gz"),
        descriptor=Mock(model_interoperability="001122"),
    )
    resolver = Mock()
    resolver.discover.side_effect = (
        (candidate(first_id, "000001"),),
        (candidate(second_id, "000002"),),
    )
    policy = Mock()
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        policy,
        catalog,
        Mock(),
        resolver,
        client=client,
    )

    def dispatch(_process, participant, _interoperability, _base_url):
        if participant.candidate.target.nf_instance_id == second_id:
            raise RuntimeError("second preparation dispatch failed")
        participant.resource_location = "http://go.example/subscriptions/first"

    orchestrator._create_preparation = Mock(side_effect=dispatch)
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution("ue-communication-default", scopes),
    )
    try:
        orchestrator._run(process)

        assert process.state is FLServerState.FAILED
        client.delete.assert_called_once_with("http://go.example/subscriptions/first")
        policy.complete_retrain.assert_called_once_with("ue-communication-default")
    finally:
        orchestrator.close()


def test_server_process_reserves_shared_slot_until_cleanup_finishes(tmp_path):
    registry = FLExperimentRegistry()
    policy = Mock()
    family_key = "ue-communication-default"
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
        orchestrator.start_flat(flat_execution(family_key))
        assert catalog_entered.wait(timeout=2)
        process = orchestrator.processes()[0]
        active = registry.active()
        assert active is not None
        assert active.server_process_id == process.process_id

        catalog_release.set()
        orchestrator.close()

        assert process.state is FLServerState.FAILED
        assert registry.active() is None
        policy.complete_retrain.assert_called_once_with(family_key)
    finally:
        catalog_release.set()
        if not orchestrator._closing.is_set():
            orchestrator.close()


def test_server_process_rejects_conflict_with_active_client_group(tmp_path):
    registry = FLExperimentRegistry()
    registry.reserve_client("subscription-a", "correlation-a")
    policy = Mock()
    family_key = "ue-communication-default"
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
        with pytest.raises(FLServerProcessConflictError, match="top-level experiment"):
            orchestrator.start_flat(flat_execution(family_key))

        assert orchestrator.processes() == ()
        assert registry.active().upper_client_subscription_ids == frozenset(
            {"subscription-a"}
        )
        policy.complete_retrain.assert_not_called()
    finally:
        orchestrator.close()


def test_flat_executor_admission_failure_is_bounded_and_discards_process(tmp_path):
    registry = FLExperimentRegistry()
    policy = Mock()
    family_key = "ue-communication-default"
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
    orchestrator._executor.submit = Mock(side_effect=RuntimeError("executor details"))

    try:
        with pytest.raises(FLServerAdmissionClosedError, match="executor is unavailable"):
            orchestrator.start_flat(flat_execution(family_key))

        assert orchestrator.processes() == ()
        assert registry.active() is None
        policy.complete_retrain.assert_not_called()
    finally:
        orchestrator.close()


def test_flat_server_publishes_round_input_with_server_owned_epochs(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    second_owner_id = "22222222-2222-4222-8222-222222222222"
    active_scopes = (
        scope("scope-a", "000001", owner_id),
        scope("scope-b", "000002", second_owner_id),
    )
    execution = flat_execution("ue-communication-default", active_scopes)
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
    resolver.discover.side_effect = (
        (candidate(owner_id, "000001"),),
        (candidate(second_owner_id, "000002"),),
    )
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "33333333-3333-4333-8333-333333333333"
    )
    workspace = Mock()
    source = SimpleNamespace(name="flat-base")
    round_inputs = (
        SimpleNamespace(url="http://root.example/round-input/0"),
        SimpleNamespace(url="http://root.example/round-input/1"),
    )
    workspace.publish_round_input.side_effect = round_inputs
    aggregate_paths = (
        tmp_path / "round-global-0.tar.gz",
        tmp_path / "round-global-1.tar.gz",
    )
    for path in aggregate_paths:
        path.write_bytes(b"round-global")
    aggregates = tuple(
        SimpleNamespace(
            digest=str(index + 1) * 64,
            path=path,
            url=f"http://root.example/round-global/{index}",
        )
        for index, path in enumerate(aggregate_paths)
    )
    policy = Mock()
    client = Mock()
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(
            round_count=2,
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
    orchestrator._loader.load.side_effect = (source, source)

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
    orchestrator._aggregate_round = Mock(side_effect=aggregates)
    orchestrator._patch_validation = Mock(side_effect=patch_validation)
    orchestrator._evaluate_final_validation = Mock(side_effect=evaluate_validation)
    process = FLProcess(process_id="process-1", intent=None, execution=execution)
    try:
        orchestrator._run(process)

        publications = workspace.publish_round_input.call_args_list
        assert [item.kwargs["base"] for item in publications] == [source, source]
        assert [item.kwargs["process_id"] for item in publications] == [
            process.process_id,
            process.process_id,
        ]
        assert [item.kwargs["round_indicator"] for item in publications] == [0, 1]
        assert [item.kwargs["epochs"] for item in publications] == [9, 9]
        assert [item.args[3] for item in orchestrator._patch_round.call_args_list] == [
            round_inputs[0].url,
            round_inputs[0].url,
            round_inputs[1].url,
            round_inputs[1].url,
        ]
        assert orchestrator._aggregate_round.call_args_list == [
            call(
                process,
                round_inputs[0].url,
                0,
                round_input_artifact=round_inputs[0],
            ),
            call(
                process,
                round_inputs[1].url,
                1,
                round_input_artifact=round_inputs[1],
            ),
        ]
        workspace.download.assert_not_called()
        assert orchestrator._loader.load.call_args_list[1].args[0].url == aggregates[0].url
        assert orchestrator._loader.load.call_args_list[1].args[0].path == aggregates[0].path
        assert process.current_global_artifact.url == aggregates[1].url
        assert process.current_global_artifact.path == aggregates[1].path
        assert process.candidate_url == aggregates[1].url
        validation_call = orchestrator._evaluate_final_validation.call_args
        assert validation_call.args[2] is process.current_global_artifact
        assert process.state is FLServerState.CANDIDATE_READY
    finally:
        orchestrator.close()


def test_server_aggregation_does_not_download_missing_owned_round_input(tmp_path):
    workspace = Mock()
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
    missing = SimpleNamespace(
        digest="a" * 64,
        path=tmp_path / "missing-round-input.tar.gz",
        url="http://root.example/round-input.tar.gz",
    )
    try:
        with pytest.raises(FileNotFoundError):
            orchestrator._aggregate_round(
                FLProcess(process_id="process-1", intent=Mock()),
                missing.url,
                0,
                round_input_artifact=missing,
            )

        workspace.download.assert_not_called()
    finally:
        orchestrator.close()


def test_server_aggregation_rejects_owned_round_input_url_mismatch_without_download(
    tmp_path,
):
    workspace = Mock()
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
    round_input = SimpleNamespace(
        digest="a" * 64,
        path=tmp_path / "round-input.tar.gz",
        url="http://root.example/round-input.tar.gz",
    )
    try:
        with pytest.raises(RuntimeError, match="input URL does not match artifact"):
            orchestrator._aggregate_round(
                FLProcess(process_id="process-1", intent=Mock()),
                "http://root.example/different-round-input.tar.gz",
                0,
                round_input_artifact=round_input,
            )

        workspace.download.assert_not_called()
    finally:
        orchestrator.close()


def test_final_validation_rejects_owned_candidate_url_mismatch_without_download(
    tmp_path,
):
    workspace = Mock()
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
    process = FLProcess(
        process_id="process-1",
        intent=Mock(),
        candidate_url="http://root.example/candidate.tar.gz",
    )
    candidate_artifact = ArtifactMetadata(
        key="a" * 64,
        size_bytes=1,
        path=tmp_path / "candidate.tar.gz",
        url="http://root.example/different-candidate.tar.gz",
    )
    try:
        with pytest.raises(RuntimeError, match="candidate URL does not match artifact"):
            orchestrator._evaluate_final_validation(
                process,
                Mock(),
                candidate_artifact,
                2,
            )

        workspace.download.assert_not_called()
    finally:
        orchestrator.close()


@pytest.mark.parametrize(
    ("trigger_source", "triggering_scope_key", "candidate_error", "expected_reasons"),
    [
        (TriggerSource.PRIVATE_API, None, 5, ()),
        (
            TriggerSource.DEGRADATION,
            "scope-a",
            10,
            ("triggering_scope_not_improved", "aggregate_not_improved"),
        ),
    ],
)
def test_final_validation_uses_owned_candidate_and_applies_trigger_semantics(
    tmp_path,
    trigger_source,
    triggering_scope_key,
    candidate_error,
    expected_reasons,
):
    participant_id = "11111111-1111-4111-8111-111111111111"
    start = datetime(2026, 8, 20, tzinfo=UTC)
    expected_scope = training_scope_descriptor()
    local_contract = RoundLocalArtifact.model_validate(
        {
            "artifact_role": "ROUND_LOCAL",
            "result_type": "ACCURACY_CHECK",
            "fl_metadata": {
                "ml_corre_id": "process-1",
                "round_ind": 2,
                "participant_nf_instance_id": participant_id,
                "training_scope": expected_scope.model_dump(mode="json"),
                "evaluation": {
                    "evaluation_stage": "FINAL_VALIDATION",
                    "evaluation_sample_count": 10,
                    "start_time": start,
                    "end_time": start + timedelta(minutes=1),
                    "base": {
                        "absolute_error_sum": 10,
                        "absolute_actual_sum": 100,
                    },
                    "candidate": {
                        "absolute_error_sum": candidate_error,
                        "absolute_actual_sum": 100,
                    },
                },
            },
        }
    )
    peer_url = "http://client.example/validation.tar.gz"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", participant_id),
        candidate=candidate(participant_id, "000001"),
        notification_correlation_id="validation-client-a",
        expected_training_scope=expected_scope,
        notification=NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": "validation-client-a",
                "mlCorreId": "process-1",
                "roundInd": 2,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {"mLModelUrl": peer_url},
                    }
                ],
            }
        ),
    )
    candidate_artifact = ArtifactMetadata(
        key="4" * 64,
        size_bytes=1,
        path=tmp_path / "candidate.tar.gz",
        url="http://root.example/candidate.tar.gz",
    )
    base_artifact = Mock()
    peer_artifact = Mock()
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=FlatExecutionRequest(
            model_family_id="ue-communication-default",
            trigger_source=trigger_source,
            participant_selection=StaticParticipantSelection(
                participants=(
                    FlatParticipantScope.from_monitor_scope(participant.scope),
                ),
                topology_version=1,
            ),
            required_cutover_scope_keys=(
                (triggering_scope_key,) if triggering_scope_key is not None else ()
            ),
            triggering_scope_key=triggering_scope_key,
            request_id="00000000-0000-4000-8000-000000000701",
        ),
        participants=[participant],
        candidate_url=candidate_artifact.url,
    )
    workspace = Mock()
    workspace.download.return_value = peer_artifact
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
    contract_fields = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 1},
        "inference": {"seq_length": 1},
    }
    models = [Mock(), Mock(), Mock()]
    for model in models:
        model.state_dict.return_value = {}
    orchestrator._loader.load.side_effect = [
        SimpleNamespace(manifest=contract_fields, model=models[0]),
        SimpleNamespace(manifest=contract_fields, model=models[1]),
        SimpleNamespace(
            manifest={**contract_fields, **local_contract.model_dump(mode="json")},
            model=models[2],
        ),
    ]
    try:
        orchestrator._evaluate_final_validation(
            process,
            base_artifact,
            candidate_artifact,
            2,
        )

        assert orchestrator._loader.load.call_args_list[:2] == [
            call(base_artifact),
            call(candidate_artifact),
        ]
        workspace.download.assert_called_once_with(
            peer_url,
            process.process_id,
            f"validation-{participant_id}",
            owner_plan_id=None,
        )
        assert orchestrator._loader.load.call_args_list[2] == call(peer_artifact)
        assert process.candidate_artifact is candidate_artifact
        assert process.gate_would_accept is (not expected_reasons)
        assert process.gate_rejection_reasons == expected_reasons
        assert len(process.validation_summaries) == 1
        assert process.validation_summaries[0].participant_nf_instance_id == participant_id
    finally:
        orchestrator.close()


def test_cutover_pending_process_releases_slot_only_after_scope_adoption(tmp_path):
    registry = FLExperimentRegistry()
    policy = Mock()
    publication = Mock()
    publication.mark_scope_adopted.return_value = True
    family_key = ("UE_COMMUNICATION", "001122")
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution(family_key),
    )
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
        assert orchestrator.mark_scope_adopted(family_key, 7, "scope-a") is True
        assert process.state is FLServerState.COMPLETE
        assert registry.active() is None
        policy.complete_retrain.assert_called_once_with(family_key)
    finally:
        orchestrator.close()


def test_delay_callback_uses_stage_state_without_body_digest(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="delay-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution("ue-communication-default", (participant.scope,)),
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
            "delayEventNotif": {
                "delayEventInd": True,
                "delayCause": "NEED_MORE_TIME",
                "expCompTime": 30,
            },
        }
    )
    try:
        orchestrator.receive_notification(notification)
        orchestrator.receive_notification(notification)
        process.state = FLServerState.READY
        with pytest.raises(ValueError, match="outside the expected stage"):
            orchestrator.receive_notification(notification)

        assert participant.requested_extension == 30
        assert process.failure == "delay callback arrived outside the expected stage"
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


def test_hierarchy_termination_outside_stage_schedules_standard_unsubscribe(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="superseded-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id="11111111-1111-4111-8111-111111111112",
        state=FLServerState.READY,
        participants=[participant],
    )
    cleanup_delivered = threading.Event()
    allow_cleanup = threading.Event()
    notification_returned = threading.Event()
    notification_errors = []
    client = Mock()

    def delete_resource(location):
        assert location == participant.resource_location
        cleanup_delivered.set()
        assert allow_cleanup.wait(1)
        return Mock(status_code=204)

    client.delete.side_effect = delete_resource
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
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id

    def deliver_termination():
        try:
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participant.notification_correlation_id,
                    mlCorreId=process.process_id,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                )
            )
        except Exception as error:
            notification_errors.append(error)
        finally:
            notification_returned.set()

    notification_thread = threading.Thread(target=deliver_termination)
    try:
        notification_thread.start()
        assert cleanup_delivered.wait(1)
        assert notification_returned.wait(0.2)
        assert notification_errors == []
        allow_cleanup.set()
        deadline = time.monotonic() + 1
        while participant.resource_location:
            if time.monotonic() >= deadline:
                raise AssertionError("terminated participant was not released")
            time.sleep(0.001)
        client.delete.assert_called_once()
    finally:
        allow_cleanup.set()
        notification_thread.join(timeout=1)
        orchestrator.close()


def test_flat_preparation_termination_still_fails_the_process(tmp_path):
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
        execution=flat_execution("ue-communication-default", (participant.scope,)),
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


def test_hierarchy_stage_waits_for_every_terminal_outcome_before_failure(
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
        orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    completed = threading.Event()
    failures = []
    outcomes = []

    def execute():
        try:
            outcomes.append(
                orchestrator.execute_hierarchy_round(
                    process_id=process.process_id,
                    round_indicator=3,
                    round_input_url="http://root.example/round-input",
                    expected_result_type=RoundLocalResultType.TRAINING,
                )
            )
        except Exception as error:
            failures.append(str(error))
        finally:
            completed.set()

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        dispatch_state = FLServerState.ROUND_DISPATCH
        waiting_state = FLServerState.ROUND_WAITING
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
        assert failures == []
        assert len(outcomes) == 1
        assert outcomes[0].accepted is False
        assert outcomes[0].aggregate is None
        assert outcomes[0].successful_participant_nf_instance_ids == (
            participant_ids[1],
        )
        assert outcomes[0].failed_participant_nf_instance_ids == (
            participant_ids[0],
        )
        assert process.state is FLServerState.READY
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_round_freezes_selected_set_and_aggregates_only_successful_results(
    tmp_path,
):
    participant_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
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
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )
    aggregate = Mock(url="http://server.example/aggregate")
    orchestrator._aggregate_round = Mock(return_value=aggregate)
    completed = threading.Event()
    results = []

    def execute():
        results.append(
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=3,
                round_input_url="http://server.example/round-input",
                round_input_artifact=Mock(url="http://server.example/round-input"),
                expected_result_type=RoundLocalResultType.TRAINING,
                selected_participant_nf_instance_ids=participant_ids[:2],
                accept_failures=True,
                minimum_completion_rate=0.5,
            )
        )
        completed.set()

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while process.state is not FLServerState.ROUND_WAITING:
            if time.monotonic() >= deadline:
                raise AssertionError("hierarchy round did not enter waiting state")
            time.sleep(0.001)

        with pytest.raises(ValueError, match="not selected"):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participants[2].notification_correlation_id,
                    mlCorreId=process.process_id,
                    roundInd=3,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                )
            )
        assert participants[2].round_complete is False
        assert participants[2].round_failure == ""

        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[0].notification_correlation_id,
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
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participants[1].notification_correlation_id,
                mlCorreId=process.process_id,
                roundInd=3,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )

        assert completed.wait(1) is True
        assert len(results) == 1
        assert results[0].accepted is True
        assert results[0].aggregate is aggregate
        assert results[0].successful_participant_nf_instance_ids == participant_ids[:1]
        assert results[0].failed_participant_nf_instance_ids == participant_ids[1:2]
        assert client.patch.call_count == 2
        orchestrator._aggregate_round.assert_called_once()
        assert (
            orchestrator._aggregate_round.call_args.kwargs[
                "participant_nf_instance_ids"
            ]
            == participant_ids[:1]
        )
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_round_treats_peer_unavailable_as_typed_participant_failure(
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
    client.patch.side_effect = [Mock(status_code=503), Mock(status_code=204)]
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
    aggregate = Mock(url="http://server.example/aggregate")
    orchestrator._aggregate_round = Mock(return_value=aggregate)
    results = []

    thread = threading.Thread(
        target=lambda: results.append(
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=4,
                round_input_url="http://server.example/round-input",
                round_input_artifact=Mock(url="http://server.example/round-input"),
                expected_result_type=RoundLocalResultType.TRAINING,
                selected_participant_nf_instance_ids=participant_ids,
                accept_failures=True,
                minimum_completion_rate=0.5,
            )
        )
    )
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while process.state is not FLServerState.ROUND_WAITING:
            if time.monotonic() >= deadline:
                raise AssertionError("hierarchy round did not enter waiting state")
            time.sleep(0.001)
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[1].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "roundInd": 4,
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
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert len(results) == 1
        assert results[0].accepted is True
        assert results[0].aggregate is aggregate
        assert results[0].failed_participant_nf_instance_ids == participant_ids[:1]
        assert results[0].successful_participant_nf_instance_ids == participant_ids[1:]
        assert process.state is FLServerState.READY
        orchestrator._aggregate_round.assert_called_once()
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_round_completion_gate_rejects_without_aggregating(tmp_path):
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
        FLServerSettings(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._aggregate_round = Mock()
    for participant in participants:
        orchestrator._correlations[participant.notification_correlation_id] = (
            process.process_id
        )
    failures = []
    results = []

    def execute():
        try:
            results.append(
                orchestrator.execute_hierarchy_round(
                    process_id=process.process_id,
                    round_indicator=1,
                    round_input_url="http://server.example/round-input",
                    round_input_artifact=Mock(url="http://server.example/round-input"),
                    expected_result_type=RoundLocalResultType.TRAINING,
                    selected_participant_nf_instance_ids=participant_ids,
                    accept_failures=True,
                    minimum_completion_rate=1,
                )
            )
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while process.state is not FLServerState.ROUND_WAITING:
            if time.monotonic() >= deadline:
                raise AssertionError("hierarchy round did not enter waiting state")
            time.sleep(0.001)
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[0].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "roundInd": 1,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://leaf.example/local"
                            },
                        }
                    ],
                }
            )
        )
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participants[1].notification_correlation_id,
                mlCorreId=process.process_id,
                roundInd=1,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )

        thread.join(timeout=1)
        assert not thread.is_alive()
        assert failures == []
        assert len(results) == 1
        assert results[0].accepted is False
        assert results[0].aggregate is None
        assert results[0].successful_participant_nf_instance_ids == participant_ids[:1]
        assert results[0].failed_participant_nf_instance_ids == participant_ids[1:]
        assert process.state is FLServerState.READY
        orchestrator._aggregate_round.assert_not_called()
    finally:
        thread.join(timeout=1)
        orchestrator.close()


def test_hierarchy_round_timeout_can_accept_completed_selected_subset(tmp_path):
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
        FLServerSettings(round_timeout_seconds=1),
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
    aggregate = Mock(url="http://server.example/aggregate")
    orchestrator._aggregate_round = Mock(return_value=aggregate)
    results = []

    def execute():
        results.append(
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=2,
                round_input_url="http://server.example/round-input",
                round_input_artifact=Mock(url="http://server.example/round-input"),
                expected_result_type=RoundLocalResultType.TRAINING,
                selected_participant_nf_instance_ids=participant_ids,
                accept_failures=True,
                minimum_completion_rate=0.5,
            )
        )

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while process.state is not FLServerState.ROUND_WAITING:
            if time.monotonic() >= deadline:
                raise AssertionError("hierarchy round did not enter waiting state")
            time.sleep(0.001)
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif.model_validate(
                {
                    "notifCorreId": participants[0].notification_correlation_id,
                    "mlCorreId": process.process_id,
                    "roundInd": 2,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://leaf.example/local"
                            },
                        }
                    ],
                }
            )
        )

        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(results) == 1
        assert results[0].accepted is True
        assert results[0].aggregate is aggregate
        assert results[0].successful_participant_nf_instance_ids == participant_ids[:1]
        assert results[0].failed_participant_nf_instance_ids == participant_ids[1:]
        assert participants[1].round_failure == "round deadline expired"
        assert (
            orchestrator._aggregate_round.call_args.kwargs[
                "participant_nf_instance_ids"
            ]
            == participant_ids[:1]
        )
        prior_failure = participants[1].round_failure
        orchestrator.receive_notification(
            NwdafMLModelTrainNotif(
                notifCorreId=participants[1].notification_correlation_id,
                mlCorreId=process.process_id,
                roundInd=2,
                termTrainReq="NOT_AVAILABLE_ML_TRAIN",
            )
        )
        assert participants[1].round_failure == prior_failure
        assert process.state is FLServerState.READY
    finally:
        thread.join(timeout=2)
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
        outcome = orchestrator.execute_hierarchy_round(
            process_id=process.process_id,
            round_indicator=3,
            round_input_url="http://branch.example/round-input",
            expected_result_type=RoundLocalResultType.TRAINING,
        )

        assert outcome.accepted is False
        assert outcome.aggregate is None
        assert outcome.failed_participant_nf_instance_ids == participant_ids
        assert process.state is FLServerState.READY
        assert [item.round_failure for item in participants] == [
            "round deadline expired",
            "round deadline expired",
        ]
        assert all(item.round_complete for item in participants)
        workspace.download.assert_not_called()
        workspace.publish.assert_not_called()
        client.delete.assert_not_called()
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
        assert orchestrator.processes() == ()
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


def test_go_generation_reset_during_lower_patch_fences_remaining_fanout(tmp_path):
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
    patch_started = threading.Event()
    allow_patch = threading.Event()
    client = Mock()

    def delayed_patch(*_args, **_kwargs):
        patch_started.set()
        assert allow_patch.wait(1)
        return Mock(status_code=204)

    client.patch.side_effect = delayed_patch
    client.delete.return_value = Mock(status_code=404)
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
    failures = []

    def execute_round():
        try:
            orchestrator.execute_hierarchy_round(
                process_id=process.process_id,
                round_indicator=0,
                round_input_url="http://branch.example/round-input",
                expected_result_type=RoundLocalResultType.TRAINING,
            )
        except Exception as error:
            failures.append(str(error))

    thread = threading.Thread(
        target=execute_round,
    )
    thread.start()
    assert patch_started.wait(1)
    orchestrator.abort_generation("containing NWDAF process generation changed")
    allow_patch.set()
    thread.join(timeout=1)

    try:
        assert thread.is_alive() is False
        assert failures == ["containing NWDAF process generation changed"]
        assert client.patch.call_count == 1
        assert orchestrator.processes() == ()
    finally:
        orchestrator.close()


def test_go_generation_reset_discards_server_process_and_callbacks(tmp_path):
    registry = FLExperimentRegistry()
    reservation = registry.reserve_server("process-1")
    policy = Mock()
    participant = FLParticipant(
        scope=scope(
            "scope-a",
            "000001",
            "11111111-1111-4111-8111-111111111111",
        ),
        candidate=candidate(
            "11111111-1111-4111-8111-111111111111",
            "000001",
        ),
        notification_correlation_id="old-callback",
        resource_location="http://go.example/subscriptions/old-resource",
    )
    family_key = ("UE_COMMUNICATION", "001122")
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution(family_key, (participant.scope,)),
        experiment_reservation_id=reservation.reservation_id,
        participants=[participant],
    )
    client = Mock()
    client.delete.return_value = Mock(status_code=404)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        policy,
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id
    old_location = participant.resource_location

    try:
        orchestrator.abort_generation("containing NWDAF process generation changed")

        assert orchestrator.processes() == ()
        with pytest.raises(KeyError):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(notifCorreId="old-callback", termTrainReq="STOP")
            )
        client.delete.assert_called_once_with(old_location)
        policy.complete_retrain.assert_called_once_with(family_key)
    finally:
        registry.reset_generation()
        orchestrator.close()


def test_hierarchy_cleanup_failure_records_error_but_releases_local_owner(tmp_path):
    registry = FLExperimentRegistry()
    plan_id = "11111111-1111-4111-8111-111111111112"
    reservation = registry.reserve_root(plan_id)
    participant = FLParticipant(
        scope=scope(
            "scope-a",
            "000001",
            "22222222-2222-4222-8222-222222222222",
        ),
        candidate=candidate(
            "22222222-2222-4222-8222-222222222222",
            "000001",
        ),
        notification_correlation_id="cleanup-correlation",
        resource_location="http://go.example/subscriptions/unreachable",
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        experiment_reservation_id=reservation.reservation_id,
        hierarchy_plan_id=plan_id,
        participants=[participant],
    )
    registry.attach_server(reservation.reservation_id, plan_id, process.process_id)
    client = Mock()
    client.delete.return_value = Mock(status_code=503)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    orchestrator._processes[process.process_id] = process
    orchestrator._correlations[participant.notification_correlation_id] = process.process_id

    try:
        orchestrator.close_hierarchy_training(process.process_id)

        assert "cleanup returned 503" in process.cleanup_failure
        assert orchestrator.processes() == ()
        assert registry.active().server_process_id is None
        assert client.delete.call_count == 1
        with pytest.raises(KeyError):
            orchestrator.receive_notification(
                NwdafMLModelTrainNotif(
                    notifCorreId=participant.notification_correlation_id,
                    termTrainReq="STOP",
                )
            )
    finally:
        registry.reset_generation()
        orchestrator.close()


def test_go_generation_reset_cleans_preparation_created_during_abort(tmp_path):
    registry = FLExperimentRegistry()
    plan_id = "11111111-1111-4111-8111-111111111112"
    reservation = registry.reserve_root(plan_id)
    request_started = threading.Event()
    allow_response = threading.Event()
    client = Mock()

    def create_after_abort(*_args, **_kwargs):
        request_started.set()
        assert allow_response.wait(1)
        return Mock(
            status_code=201,
            headers={"Location": "http://go.example/subscriptions/late-resource"},
        )

    client.post.side_effect = create_after_abort
    client.delete.return_value = Mock(status_code=204)
    orchestrator = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(cleanup={"max_attempts": 1}),
        context_client(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        client=client,
        experiments=registry,
    )
    failures = []

    def dispatch():
        try:
            orchestrator.start_protocol_preparation(
                ml_correlation_id=plan_id,
                reservation_id=reservation.reservation_id,
                ml_event="X_IMAGE_CLASSIFICATION",
                ml_event_filter={},
                model_interoperability="pymtlf-image-classification-mnist",
                targets=(
                    ProtocolPreparationTarget(
                        participant_nf_instance_id=(
                            "22222222-2222-4222-8222-222222222222"
                        ),
                        candidate=candidate(
                            "22222222-2222-4222-8222-222222222222",
                            "000001",
                        ),
                        topology=FlTopologyNode(
                            nfInstanceId="22222222-2222-4222-8222-222222222222"
                        ),
                    ),
                ),
            )
        except Exception as error:
            failures.append(str(error))

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert request_started.wait(1)
    orchestrator.abort_generation("containing NWDAF process generation changed")
    allow_response.set()
    thread.join(timeout=1)

    try:
        assert thread.is_alive() is False
        assert failures == []
        client.delete.assert_called_once_with(
            "http://go.example/subscriptions/late-resource"
        )
        assert orchestrator.processes() == ()
        assert registry.active().server_process_id is None
    finally:
        registry.reset_generation()
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
        assert completed.wait(1) is True
        assert len(failures) == 1
        assert "ML_MODEL_TRAINING_REQS_NOT_MET" in failures[0]
        assert process.state is FLServerState.FAILED
    finally:
        thread.join(timeout=1)
        orchestrator.close()


@pytest.mark.parametrize(
    "active_state",
    [FLServerState.ROUND_DISPATCH, FLServerState.FINAL_VALIDATION_DISPATCH],
)
def test_hierarchy_callback_during_dispatch_keeps_first_terminal_outcome(
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
        orchestrator.receive_notification(conflicting)
        assert participant.notification == accepted
        assert participant.round_failure == ""
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
@pytest.mark.parametrize("hierarchical", [False, True], ids=("flat", "hierarchical"))
def test_callback_after_collection_freeze_ignores_terminal_retries(
    tmp_path,
    evaluating_state,
    hierarchical,
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
    process = FLProcess(
        process_id="process-1",
        intent=None,
        hierarchy_plan_id=(
            "11111111-1111-4111-8111-111111111112" if hierarchical else ""
        ),
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
        orchestrator.receive_notification(conflicting)

        assert process.state is evaluating_state
        assert participant.notification == accepted
        assert participant.round_failure == ""
    finally:
        if hierarchical:
            process.hierarchy_cleanup_complete = True
        orchestrator.close()


def test_flat_round_termination_freezes_first_terminal_outcome(tmp_path):
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="round-client-a",
        expected_round=3,
    )
    process = FLProcess(
        process_id="process-1",
        intent=None,
        execution=flat_execution("ue-communication-default", (participant.scope,)),
        state=FLServerState.ROUND_WAITING,
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
    termination = NwdafMLModelTrainNotif(
        notifCorreId=participant.notification_correlation_id,
        mlCorreId=process.process_id,
        roundInd=3,
        termTrainReq="NOT_AVAILABLE_ML_TRAIN",
    )
    later_model = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": participant.notification_correlation_id,
            "mlCorreId": process.process_id,
            "roundInd": 3,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://leaf.example/local-result.tar.gz"
                    },
                }
            ],
        }
    )
    try:
        orchestrator.receive_notification(termination)
        failure = process.failure
        orchestrator.receive_notification(later_model)

        assert participant.round_complete is True
        assert participant.notification == termination
        assert process.failure == failure
        assert "NOT_AVAILABLE_ML_TRAIN" in process.failure
    finally:
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
        with pytest.raises(KeyError):
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


def test_preparation_uses_configured_historical_data_window(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="py_mtlf.core.fl_server")
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
        assert (
            "FL participant resource created process_id=process-1 "
            f"nf={owner_id} notif_corre_id=preparation-client-a "
            "location=http://go.example/subscriptions/preparation-a"
        ) in caplog.text
    finally:
        orchestrator.close()


def test_participant_cleanup_logs_exact_deleted_resource(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="py_mtlf.core.fl_server")
    owner_id = "11111111-1111-4111-8111-111111111111"
    participant = FLParticipant(
        scope=scope("scope-a", "000001", owner_id),
        candidate=candidate(owner_id, "000001"),
        notification_correlation_id="cleanup-client-a",
        resource_location="http://go.example/subscriptions/resource-a",
    )
    process = FLProcess(process_id="process-1", intent=Mock())
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
    try:
        assert orchestrator._cleanup_participant(process, participant) == ""
        assert (
            "FL participant resource deleted process_id=process-1 "
            f"nf={owner_id} location={participant.resource_location} status=204"
        ) in caplog.text
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
    training_scopes = (
        training_scope_descriptor("branch-1"),
        training_scope_descriptor("branch-2"),
    )

    def model_with_weight(value: float) -> torch.nn.Module:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        model.eval()
        return model

    base_model = model_with_weight(0)
    branch_models = (model_with_weight(2), model_with_weight(10))
    base_manifest = {
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 1},
        "inference": {"seq_length": 1},
        "fl_metadata": {
            "ml_corre_id": "process-1",
            "round_ind": 0,
            "client_training": {"epochs": 2},
        },
    }
    base = SimpleNamespace(manifest=base_manifest, model=base_model)

    branch_bundles = []
    sample_counts = (1, 3)
    for index, (branch_id, leaf_id, training_scope, model, sample_count) in enumerate(
        zip(
            branch_ids,
            leaf_ids,
            training_scopes,
            branch_models,
            sample_counts,
            strict=True,
        ),
        start=1,
    ):
        manifest = {
            "artifact_role": "ROUND_LOCAL",
            "result_type": "HIERARCHY_AGGREGATE",
            "analytics_event": "UE_COMMUNICATION",
            "model_interoperability": "001122",
            "runtime_compatibility": {"framework": "torch"},
            "model": {"input_size": 1},
            "inference": {"seq_length": 1},
            "fl_metadata": {
                "ml_corre_id": "process-1",
                "round_ind": 0,
                "participant_nf_instance_id": branch_id,
                "training_scope": training_scope.model_dump(mode="json"),
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
    for index, (branch_id, training_scope) in enumerate(
        zip(branch_ids, training_scopes, strict=True),
        start=1,
    ):
        participants.append(
            FLParticipant(
                scope=scope(f"branch-{index}", f"00000{index}", branch_id),
                candidate=candidate(branch_id, f"00000{index}"),
                notification_correlation_id=f"round-branch-{index}",
                expected_training_scope=training_scope,
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
    owned_round_input_path = tmp_path / "owned-round-input.tar.gz"
    owned_round_input_path.write_bytes(b"owned-round-input")
    owned_round_input = SimpleNamespace(
        digest="0" * 64,
        path=owned_round_input_path,
        url="http://root.example/round-input",
    )
    workspace.download.side_effect = [
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
            round_input_artifact=owned_round_input,
            expected_result_type=RoundLocalResultType.HIERARCHY_AGGREGATE,
            expected_subordinates={
                branch_id: (leaf_id,)
                for branch_id, leaf_id in zip(branch_ids, leaf_ids, strict=True)
            },
        )

        assert result is published
        assert [item.args[0] for item in workspace.download.call_args_list] == [
            "http://branch-1.example/aggregate.tar.gz",
            "http://branch-2.example/aggregate.tar.gz",
        ]
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

        workspace.download.reset_mock()
        workspace.download.side_effect = [SimpleNamespace(key="4" * 64)]
        orchestrator._loader.reset_mock()
        orchestrator._loader.load.side_effect = [base, branch_bundles[0]]
        orchestrator._aggregate_round(
            process,
            "http://root.example/round-input",
            0,
            round_input_artifact=owned_round_input,
            expected_result_type=RoundLocalResultType.HIERARCHY_AGGREGATE,
            expected_subordinates={branch_ids[0]: (leaf_ids[0],)},
            participant_nf_instance_ids=(branch_ids[0],),
        )

        partial_publication = workspace.publish.call_args.kwargs
        assert partial_publication["model"].weight.item() == pytest.approx(2.0)
        assert [item.args[0] for item in workspace.download.call_args_list] == [
            "http://branch-1.example/aggregate.tar.gz"
        ]
        assert partial_publication["metadata"]["fl_metadata"][
            "aggregated_training_sample_count"
        ] == 1
    finally:
        orchestrator.close()


def test_aggregation_rejects_local_artifact_with_different_model_contract(tmp_path):
    participant_id = "11111111-1111-4111-8111-111111111111"
    expected_scope = training_scope_descriptor()
    base_manifest = {
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 10},
        "inference": {"seq_length": 30},
        "fl_metadata": {
            "ml_corre_id": "process-1",
            "round_ind": 0,
            "client_training": {"epochs": 1},
        },
    }
    local_manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 11},
        "inference": {"seq_length": 30},
        "artifact_role": "ROUND_LOCAL",
        "result_type": "TRAINING",
        "fl_metadata": {
            "ml_corre_id": "process-1",
            "round_ind": 0,
            "participant_nf_instance_id": participant_id,
            "training_scope": expected_scope.model_dump(mode="json"),
            "training_sample_count": 10,
            "dataset_evidence": {
                "workload_profile": "ue_communication_forecasting",
                "observation_count": 12,
                "training_sample_count": 10,
                "validation_sample_count": 1,
            },
        },
    }
    base = Mock(manifest=base_manifest)
    base.model.state_dict.return_value = {}
    local = Mock(manifest=local_manifest)
    workspace = Mock()
    workspace.download.return_value = Mock(key="6" * 64)
    round_input_path = tmp_path / "round-input.tar.gz"
    round_input_path.write_bytes(b"round-input")
    round_input = SimpleNamespace(
        digest="7" * 64,
        path=round_input_path,
        url="http://root.example/round-input.tar.gz",
    )
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
        expected_training_scope=expected_scope,
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
            orchestrator._aggregate_round(
                process,
                round_input.url,
                0,
                round_input_artifact=round_input,
            )
        workspace.download.assert_not_called()

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
        with pytest.raises(RuntimeError, match="model contract is incompatible"):
            orchestrator._aggregate_round(
                process,
                round_input.url,
                0,
                round_input_artifact=round_input,
            )
        workspace.download.assert_called_once_with(
            "http://client.example/local.tar.gz",
            process.process_id,
            f"round-0-{participant_id}",
            owner_plan_id=None,
        )
    finally:
        orchestrator.close()
