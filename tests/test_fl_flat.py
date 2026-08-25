import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from py_mtlf.config import FederatedLearningSettings, FLServerSettings, OrchestrationSettings
from py_mtlf.core.accuracy_policy import RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_flat import FlatFLCoordinator
from py_mtlf.core.fl_orchestration import (
    ParticipantSource,
    TopLevelCoordinatorUnavailableError,
    TopLevelRequestConflictError,
    TriggerSource,
)
from py_mtlf.core.fl_server import FLClientCandidate, FLProcess, FLServerEngine, FLServerState
from py_mtlf.core.fl_topology import StaticFlatTopologyPlanner
from py_mtlf.wire.private import SelectedTarget

SERVER_ID = "00000000-0000-4000-8000-000000000001"
CLIENT_A_ID = "00000000-0000-4000-8000-000000000301"
CLIENT_B_ID = "00000000-0000-4000-8000-000000000302"
REQUEST_ID = "00000000-0000-4000-8000-000000000701"
OTHER_REQUEST_ID = "00000000-0000-4000-8000-000000000702"
FAMILY_ID = "ue-communication-default"


def static_planner(tmp_path) -> StaticFlatTopologyPlanner:
    path = tmp_path / "flat.yaml"
    path.write_text(
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001102"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return StaticFlatTopologyPlanner.load(path)


def catalog(*, event_filter=None):
    value = Mock()
    value.current.return_value = SimpleNamespace(
        model_id=1,
        descriptor=SimpleNamespace(
            event="UE_COMMUNICATION",
            event_filter=event_filter or {},
            target_ue={"intGroupIds": ["group-a"]},
        ),
    )
    return value


def context():
    value = Mock()
    value.get.return_value = SimpleNamespace(nf_instance_id=SERVER_ID)
    return value


def process_for(execution):
    return FLProcess(
        process_id="process-1",
        intent=None,
        execution=execution,
    )


def test_static_manual_request_uses_typed_participants_without_cutover_scopes(tmp_path):
    server = Mock()
    server.start_flat.side_effect = process_for
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(mode="flat", participant_source="static"),
        server=server,
        policy=Mock(),
        catalog=catalog(),
        nwdaf_context=context(),
        planner=static_planner(tmp_path),
        terminal_status_ttl_seconds=60,
    )

    created = coordinator.submit_manual(
        request_id=REQUEST_ID,
        model_family_id=FAMILY_ID,
    )
    replay = coordinator.submit_manual(
        request_id=REQUEST_ID,
        model_family_id=FAMILY_ID,
    )

    assert replay == created
    assert created.mode == "flat"
    assert created.participant_source == "static"
    assert created.trigger_source == "private_api"
    execution = server.start_flat.call_args.args[0]
    assert execution.trigger_source is TriggerSource.PRIVATE_API
    assert execution.participant_selection.source is ParticipantSource.STATIC
    assert execution.required_cutover_scope_keys == ()
    assert execution.triggering_scope_key is None
    assert tuple(
        item.participant_nf_instance_id
        for item in execution.participant_selection.participants
    ) == (CLIENT_A_ID, CLIENT_B_ID)

    assert execution.participant_selection.participants[0].ml_event_filter[
        "networkArea"
    ] == {
        "tais": [
            {
                "plmnId": {"mcc": "466", "mnc": "92"},
                "tac": "001101",
            }
        ]
    }

    with pytest.raises(TopLevelRequestConflictError, match="different model family"):
        coordinator.submit_manual(
            request_id=REQUEST_ID,
            model_family_id="other-family",
        )
    with pytest.raises(TopLevelRequestConflictError, match="another top-level"):
        coordinator.submit_manual(
            request_id=OTHER_REQUEST_ID,
            model_family_id=FAMILY_ID,
        )

    coordinator.abort_generation("containing NWDAF process generation changed")
    coordinator.abort_generation("duplicate generation reset")
    assert coordinator.get(REQUEST_ID) is None
    restarted = coordinator.submit_manual(
        request_id=OTHER_REQUEST_ID,
        model_family_id=FAMILY_ID,
    )
    assert restarted.request_id == OTHER_REQUEST_ID
    coordinator.close()
    coordinator.close()
    with pytest.raises(TopLevelCoordinatorUnavailableError, match="closing"):
        coordinator.submit_manual(
            request_id=REQUEST_ID,
            model_family_id=FAMILY_ID,
        )


def test_static_selection_rejects_model_family_with_existing_area_source(tmp_path):
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(mode="flat", participant_source="static"),
        server=Mock(),
        policy=Mock(),
        catalog=catalog(event_filter={"networkArea": {"tais": []}}),
        nwdaf_context=context(),
        planner=static_planner(tmp_path),
        terminal_status_ttl_seconds=60,
    )

    with pytest.raises(TopLevelRequestConflictError, match="area source"):
        coordinator.submit_manual(request_id=REQUEST_ID, model_family_id=FAMILY_ID)


def test_monitor_degradation_preserves_cutover_and_trigger_scope_separately():
    scopes = (
        ScopeReference(
            scope_key="scope-a",
            consumer_id=CLIENT_A_ID,
            model_ids=(1,),
            ml_event="UE_COMMUNICATION",
            ml_event_filter={
                "networkArea": {
                    "tais": [
                        {
                            "plmnId": {"mcc": "466", "mnc": "92"},
                            "tac": "001101",
                        }
                    ]
                }
            },
            target_ue=None,
        ),
        ScopeReference(
            scope_key="scope-b",
            consumer_id=CLIENT_B_ID,
            model_ids=(1,),
            ml_event="UE_COMMUNICATION",
            ml_event_filter={
                "networkArea": {
                    "tais": [
                        {
                            "plmnId": {"mcc": "466", "mnc": "92"},
                            "tac": "001102",
                        }
                    ]
                }
            },
            target_ue=None,
        ),
    )
    intent = RetrainIntent(
        family_key=FAMILY_ID,
        triggering_scope_key="scope-a",
        active_scope_keys=("scope-a", "scope-b"),
        triggering_scope=scopes[0],
        active_scopes=scopes,
        created_at=Mock(),
    )
    policy = Mock()
    policy.take_intents.return_value = (intent,)
    server = Mock()
    server.start_flat.side_effect = process_for
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(
            mode="flat",
            participant_source="monitor_scopes",
        ),
        server=server,
        policy=policy,
        catalog=catalog(),
        nwdaf_context=context(),
        planner=None,
        terminal_status_ttl_seconds=60,
    )

    coordinator.accept_policy_intents()

    execution = server.start_flat.call_args.args[0]
    assert execution.trigger_source is TriggerSource.DEGRADATION
    assert execution.required_cutover_scope_keys == ("scope-a", "scope-b")
    assert execution.triggering_scope_key == "scope-a"
    assert tuple(
        item.participant_nf_instance_id
        for item in execution.participant_selection.participants
    ) == (CLIENT_A_ID, CLIENT_B_ID)

    policy.reset_mock()
    coordinator.accept_policy_intents()
    policy.discard_intents.assert_called_once_with()
    policy.take_intents.assert_not_called()


def test_static_degradation_uses_topology_participants_and_monitor_cutover(tmp_path):
    active_scope = ScopeReference(
        scope_key="monitor-scope-a",
        consumer_id="monitor-owner",
        model_ids=(1,),
        ml_event="UE_COMMUNICATION",
        ml_event_filter={},
        target_ue={"intGroupIds": ["group-a"]},
    )
    intent = RetrainIntent(
        family_key=FAMILY_ID,
        triggering_scope_key=active_scope.scope_key,
        active_scope_keys=(active_scope.scope_key,),
        triggering_scope=active_scope,
        active_scopes=(active_scope,),
        created_at=Mock(),
    )
    policy = Mock()
    policy.take_intents.return_value = (intent,)
    server = Mock()
    server.start_flat.side_effect = process_for
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(mode="flat", participant_source="static"),
        server=server,
        policy=policy,
        catalog=catalog(),
        nwdaf_context=context(),
        planner=static_planner(tmp_path),
        terminal_status_ttl_seconds=60,
    )

    coordinator.accept_policy_intents()

    execution = server.start_flat.call_args.args[0]
    assert execution.participant_selection.source is ParticipantSource.STATIC
    assert tuple(
        item.participant_nf_instance_id
        for item in execution.participant_selection.participants
    ) == (CLIENT_A_ID, CLIENT_B_ID)
    assert execution.required_cutover_scope_keys == (active_scope.scope_key,)
    assert execution.triggering_scope_key == active_scope.scope_key


def test_degradation_admission_failure_releases_policy_ownership_once(tmp_path):
    participant = ScopeReference(
        scope_key="scope-a",
        consumer_id=CLIENT_A_ID,
        model_ids=(1,),
        ml_event="UE_COMMUNICATION",
        ml_event_filter={
            "networkArea": {
                "tais": [
                    {
                        "plmnId": {"mcc": "466", "mnc": "92"},
                        "tac": "001101",
                    }
                ]
            }
        },
        target_ue=None,
    )
    intent = RetrainIntent(
        family_key=FAMILY_ID,
        triggering_scope_key=participant.scope_key,
        active_scope_keys=(participant.scope_key,),
        triggering_scope=participant,
        active_scopes=(participant,),
        created_at=Mock(),
    )
    policy = Mock()
    policy.take_intents.return_value = (intent,)
    registry = FLExperimentRegistry()
    registry.reserve_client("upper-subscription", "upper-correlation")
    server = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path),
        FLServerSettings(),
        context(),
        policy,
        catalog(),
        Mock(),
        Mock(),
        client=Mock(),
        experiments=registry,
    )
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(
            mode="flat",
            participant_source="monitor_scopes",
        ),
        server=server,
        policy=policy,
        catalog=catalog(),
        nwdaf_context=context(),
        planner=None,
        terminal_status_ttl_seconds=60,
    )
    try:
        coordinator.accept_policy_intents()

        policy.complete_retrain.assert_called_once_with(FAMILY_ID)
    finally:
        server.close()


def test_terminal_status_is_bounded_and_expires(tmp_path):
    now = [10.0]
    server = Mock()
    process = FLProcess(process_id="process-1", intent=None)
    server.start_flat.return_value = process
    policy = Mock()
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(mode="flat", participant_source="static"),
        server=server,
        policy=policy,
        catalog=catalog(),
        nwdaf_context=context(),
        planner=static_planner(tmp_path),
        terminal_status_ttl_seconds=5,
        clock=lambda: now[0],
    )
    coordinator.submit_manual(request_id=REQUEST_ID, model_family_id=FAMILY_ID)
    process.state = FLServerState.FAILED
    process.failure = "http://secret.example/token=credential"

    failed = coordinator.get(REQUEST_ID)

    assert failed is not None
    assert failed.failure_cause == "TRAINING_FAILED"
    assert failed.failure_detail == "federated training failed"
    coordinator.accept_policy_intents()
    policy.discard_intents.assert_called_once_with()
    policy.take_intents.assert_not_called()
    now[0] += 5
    assert coordinator.get(REQUEST_ID) is None


def test_static_manual_runs_server_publication_and_cleanup_without_cutover(tmp_path):
    base_path = tmp_path / "base.tar.gz"
    base_path.write_bytes(b"base")
    aggregate_path = tmp_path / "aggregate.tar.gz"
    aggregate_path.write_bytes(b"aggregate")
    base_artifact = ArtifactMetadata(
        key="a" * 64,
        size_bytes=base_path.stat().st_size,
        path=base_path,
        url="http://root.example/base.tar.gz",
    )
    current = SimpleNamespace(
        model_id=1,
        artifact=base_artifact,
        descriptor=SimpleNamespace(
            event="UE_COMMUNICATION",
            event_filter={},
            target_ue={"intGroupIds": ["group-a"]},
            model_interoperability="001122",
        ),
    )
    model_catalog = Mock()
    model_catalog.current.return_value = current
    model_catalog.version_key_for_id.return_value = "version-1"
    nwdaf_context = context()
    resolver = Mock()

    def discover(participant, _interoperability):
        tac = participant.ml_event_filter["networkArea"]["tais"][0]["tac"]
        return (
            FLClientCandidate(
                target=SelectedTarget(
                    nfInstanceId=participant.participant_nf_instance_id,
                    nfServiceInstanceId=f"training-{tac}",
                    serviceName="nnwdaf-mlmodeltraining",
                    apiRoot=f"http://{tac}.example",
                    selectionSource="NRF",
                ),
                tracking_areas=(f"466-92-{tac}",),
            ),
        )

    resolver.discover.side_effect = discover
    workspace = Mock()
    workspace.publish_round_input.return_value = SimpleNamespace(
        url="http://root.example/round-input.tar.gz"
    )
    aggregate = SimpleNamespace(
        digest="b" * 64,
        path=aggregate_path,
        url="http://root.example/aggregate.tar.gz",
    )
    publication = Mock()
    publication.publish.return_value = SimpleNamespace(
        model_id=2,
        version_key="version-2",
    )
    policy = Mock()
    client = Mock()
    cleanup_complete = threading.Event()

    def delete_participant(_location):
        if client.delete.call_count == 2:
            cleanup_complete.set()
        return Mock(status_code=204)

    client.delete.side_effect = delete_participant
    registry = FLExperimentRegistry()
    server = FLServerEngine(
        FederatedLearningSettings(workspace_root=tmp_path / "workspaces"),
        FLServerSettings(
            round_count=1,
            cleanup={"max_attempts": 1},
            final_validation={"enforce_performance_gate": True},
        ),
        nwdaf_context,
        policy,
        model_catalog,
        workspace,
        resolver,
        client=client,
        publication=publication,
        experiments=registry,
    )
    server._loader = Mock()
    server._loader.load.return_value = SimpleNamespace(name="base-bundle")

    def prepare(_process, participant, _interoperability, _base_url):
        participant.resource_location = (
            f"http://client.example/subscriptions/{participant.candidate.target.nf_instance_id}"
        )
        participant.preparation_complete = True
        participant.training_sample_count = 10

    def complete_round(_process, participant, *_args):
        participant.notification = Mock()

    def accept_validation(process, _base, candidate_artifact, _round):
        process.candidate_artifact = candidate_artifact
        process.gate_would_accept = True

    server._create_preparation = Mock(side_effect=prepare)
    server._wait = Mock()
    server._patch_round = Mock(side_effect=complete_round)
    server._aggregate_round = Mock(return_value=aggregate)
    server._patch_validation = Mock(side_effect=complete_round)
    server._evaluate_final_validation = Mock(side_effect=accept_validation)
    coordinator = FlatFLCoordinator(
        orchestration=OrchestrationSettings(mode="flat", participant_source="static"),
        server=server,
        policy=policy,
        catalog=model_catalog,
        nwdaf_context=nwdaf_context,
        planner=static_planner(tmp_path),
        terminal_status_ttl_seconds=60,
    )
    try:
        coordinator.submit_manual(request_id=REQUEST_ID, model_family_id=FAMILY_ID)
        assert cleanup_complete.wait(timeout=2)
        snapshot = coordinator.get(REQUEST_ID)

        assert snapshot is not None
        assert snapshot.state == "COMPLETE"
        assert snapshot.completed_rounds == 1
        assert snapshot.candidate_digest == "b" * 64
        published = publication.publish.call_args.args[0]
        assert published.required_scope_keys == ()
        assert [item.participant_nf_instance_id for item in published.participants] == [
            CLIENT_A_ID,
            CLIENT_B_ID,
        ]
        policy.begin_generation.assert_called_once_with(
            FAMILY_ID,
            "version-1",
            "version-2",
            (),
        )
        assert client.delete.call_count == 2
        assert registry.active() is None
    finally:
        server.close()
