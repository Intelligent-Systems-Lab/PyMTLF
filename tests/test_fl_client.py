import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import torch

from py_mtlf.config import (
    DatasetSettings,
    FederatedLearningSettings,
    FLClientSettings,
    NotificationSettings,
)
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.dataset import DatasetCoordinator, DatasetJobState
from py_mtlf.core.fl_artifacts import HierarchyAssignmentArtifact
from py_mtlf.core.fl_branch import (
    BranchPreparationExecution,
    FLBranchPreparationCoordinator,
)
from py_mtlf.core.fl_client import (
    FLClientCapacityError,
    FLClientEngine,
    FLClientResource,
    FLClientState,
    _termination,
)
from py_mtlf.core.fl_experiment import ExperimentRole, FLExperimentRegistry
from py_mtlf.core.fl_hierarchy import HierarchyMessageType, PreparationOutcome
from py_mtlf.core.fl_server import HierarchyValidationCollection
from py_mtlf.core.fl_workspace import (
    ValidatedArchive,
    ValidatedHierarchyArtifact,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.core.trainer import LocalTrainer
from py_mtlf.core.training_data import FEATURE_ORDER
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.models import TrainingDataDescriptor
from py_mtlf.wire.adrf import DataNotification, DataSubscription, NadrfDataStoreRecord
from py_mtlf.wire.ml_model_training import (
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
)


def preparation_payload() -> dict:
    return {
        "mLEventSubscs": [
            {
                "mLEvent": "UE_COMMUNICATION",
                "mLEventFilter": {
                    "networkArea": {
                        "tais": [
                            {
                                "plmnId": {"mcc": "466", "mnc": "92"},
                                "tac": "000001",
                            }
                        ]
                    }
                },
                "modelInterInfo": "001122",
            }
        ],
        "notifUri": "http://go.internal/training/callback",
        "notifCorreId": "prep-client-a",
        "mlCorreId": "fl-process-001",
        "mLPreFlag": True,
        "mLModelInfos": [
            {
                "event": "UE_COMMUNICATION",
                "mLFileAddr": {"mLModelUrl": "http://server.example/base.tar.gz"},
            }
        ],
        "eventReq": {"notifMethod": "ON_EVENT_DETECTION"},
        "tgtRepUe": {"intGroupIds": ["group-G"]},
        "mLModelTrainInfos": [
            {
                "dataAvReq": {
                    "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                    "minNumSamples": 1,
                    "timeWindows": [
                        {
                            "startTime": "2026-07-01T00:00:00Z",
                            "stopTime": "2026-07-27T00:00:00Z",
                        }
                    ],
                },
                "timeAvReq": "PT5M",
            }
        ],
        "mLTrainRepInfo": {"maxResTime": 300},
    }


def fl_settings(tmp_path) -> FederatedLearningSettings:
    return FederatedLearningSettings(workspace_root=tmp_path)


def client_settings() -> FLClientSettings:
    return FLClientSettings(
        training_data={"collection_trigger": "consumer_subscription"},
        model_interoperability_ids=("001122",),
    )


def round_input_bundle(*, epochs: int = 7):
    model = torch.nn.Linear(2, 1)
    manifest = {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2, "output_size": 1},
        "inference": {"feature_order": ["uplink", "downlink"]},
        "file_digests": {
            "model.py": "1" * 64,
            "model.npy": "2" * 64,
            "scaler.pkl": "3" * 64,
        },
    }
    manifest["fl_metadata"] = {
        "contract_version": "1.0",
        "ml_corre_id": "fl-process-001",
        "round_ind": 2,
        "model_contract_digest": model_contract_digest(manifest),
        "preprocessing_contract_digest": preprocessing_contract_digest(manifest),
        "weights_digest": weights_digest(model),
        "client_training": {"epochs": epochs},
    }
    return Mock(manifest=manifest, model=model)


def round_global_bundle():
    model = torch.nn.Linear(2, 1)
    manifest = {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_GLOBAL",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2, "output_size": 1},
        "inference": {"feature_order": ["uplink", "downlink"]},
        "file_digests": {
            "model.py": "1" * 64,
            "model.npy": "2" * 64,
            "scaler.pkl": "3" * 64,
        },
    }
    digest = weights_digest(model)
    manifest["fl_metadata"] = {
        "contract_version": "1.0",
        "ml_corre_id": "fl-process-001",
        "round_ind": 1,
        "model_contract_digest": model_contract_digest(manifest),
        "preprocessing_contract_digest": preprocessing_contract_digest(manifest),
        "base_weights_digest": digest,
        "weights_digest": digest,
        "participants": [
            {
                "participant_nf_instance_id": (
                    "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
                ),
                "training_sample_count": 10,
                "local_artifact_digest": "4" * 64,
            }
        ],
        "aggregated_training_sample_count": 10,
    }
    return Mock(manifest=manifest, model=model)


def hierarchy_assignment(tmp_path, *, branch: bool) -> ValidatedHierarchyArtifact:
    root_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    branch_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    leaf_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    hierarchy_metadata = {
        "contract_version": "1.0",
        "message_type": (
            HierarchyMessageType.BRANCH_ASSIGNMENT
            if branch
            else HierarchyMessageType.LEAF_ASSIGNMENT
        ),
        "plan_id": "11111111-1111-4111-8111-111111111111",
        "publisher_nf_instance_id": root_id if branch else branch_id,
        "intended_recipient_nf_instance_id": branch_id if branch else leaf_id,
        "strategy": {
            "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
            "participant_selection": "all",
            "waiting_policy": "all",
            "aggregation": "sample_weighted",
        },
    }
    if branch:
        hierarchy_metadata.update(
            {
                "assigned_leaf_nf_instance_ids": [leaf_id],
                "admission": {"mode": "complete_required"},
            }
        )
    else:
        hierarchy_metadata["parent_branch_nf_instance_id"] = branch_id
    contract = HierarchyAssignmentArtifact.model_validate(
        {
            "artifact_role": "HIERARCHY_ASSIGNMENT",
            "bundle_schema_version": "1.0",
            "file_digests": {
                "model.py": "1" * 64,
                "model.npy": "2" * 64,
                "scaler.pkl": "3" * 64,
            },
            "hierarchy_metadata": hierarchy_metadata,
        }
    )
    path = tmp_path / ("branch-assignment.tar.gz" if branch else "leaf-assignment.tar.gz")
    path.write_bytes(b"assignment")
    return ValidatedHierarchyArtifact(
        metadata=ArtifactMetadata(
            key="a" * 64,
            size_bytes=path.stat().st_size,
            path=path,
            url="http://parent.example/assignment.tar.gz",
        ),
        manifest={
            "analytics_event": "UE_COMMUNICATION",
            "model_interoperability": "001122",
        },
        contract=contract,
    )


def test_create_admits_before_async_adrf_preparation(tmp_path):
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        Mock(),
    )
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    try:
        resource = service.create(NwdafMLModelTrainSubsc.model_validate(preparation_payload()))
        assert resource.state is FLClientState.PREPARING
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            resource = service.get(resource.subscription_id)
            if resource.dataset_job_id:
                break
            time.sleep(0.01)
        assert resource.dataset_job_id == "dataset-job-1"
        datasets.submit_external.assert_called_once()
    finally:
        service.close()


def test_create_defers_training_scope_resolution_to_preparation_worker(
    tmp_path, monkeypatch
):
    datasets = Mock()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        Mock(),
    )
    submit = Mock()
    monkeypatch.setattr(service, "_submit", submit)
    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(preparation_payload())
        )

        assert resource.state is FLClientState.PREPARING
        datasets.validate_external_scope.assert_not_called()
        submit.assert_called_once()
    finally:
        service.close()


def test_create_reserves_same_correlation_group_and_rejects_another(tmp_path, monkeypatch):
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        FLClientSettings(
            training_data={"collection_trigger": "consumer_subscription"},
            model_interoperability_ids=("001122",),
            max_concurrent_jobs=3,
        ),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
        experiments=registry,
    )
    monkeypatch.setattr(service, "_start_operation", Mock())
    first_payload = preparation_payload()
    second_payload = preparation_payload()
    second_payload["notifCorreId"] = "prep-client-b"
    conflict_payload = preparation_payload()
    conflict_payload["notifCorreId"] = "prep-client-c"
    conflict_payload["mlCorreId"] = "fl-process-002"

    try:
        first = service.create(NwdafMLModelTrainSubsc.model_validate(first_payload))
        second = service.create(NwdafMLModelTrainSubsc.model_validate(second_payload))

        active = registry.active()
        assert active is not None
        assert active.upper_client_subscription_ids == frozenset(
            {first.subscription_id, second.subscription_id}
        )
        with pytest.raises(FLClientCapacityError, match="top-level experiment"):
            service.create(NwdafMLModelTrainSubsc.model_validate(conflict_payload))
    finally:
        service.close()


def test_create_failure_rolls_back_experiment_reservation(tmp_path, monkeypatch):
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
        experiments=registry,
    )
    monkeypatch.setattr(
        service,
        "_start_operation",
        Mock(side_effect=RuntimeError("start failed")),
    )

    try:
        with pytest.raises(RuntimeError, match="start failed"):
            service.create(
                NwdafMLModelTrainSubsc.model_validate(preparation_payload())
            )
        assert registry.active() is None
    finally:
        service.close()


def test_go_generation_reset_discards_idle_prepared_client_resource(
    tmp_path,
    monkeypatch,
):
    registry = FLExperimentRegistry()
    workspace = Mock()
    branch = Mock()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        workspace,
        experiments=registry,
        branch_coordinator=branch,
    )

    def finish_immediately(resource):
        resource.state = FLClientState.READY
        resource.callback_slot_owned = False
        resource.work_slot_owned = False
        service._capacity.release()
        service._outbox_capacity.release()

    monkeypatch.setattr(service, "_start_operation", finish_immediately)
    resource = service.create(
        NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    )
    plan_id = "11111111-1111-4111-8111-111111111111"
    registry.bind_plan(
        resource.experiment_reservation_id,
        plan_id,
        ExperimentRole.LEAF,
    )

    try:
        service.abort_generation("containing NWDAF process generation changed")

        with pytest.raises(KeyError):
            service.get(resource.subscription_id)
        branch.abort_generation.assert_called_once()
        workspace.release_plan.assert_called_once_with(plan_id)
        assert service._closing.is_set() is False
    finally:
        registry.reset_generation()
        service.close()


def test_go_generation_reset_releases_active_client_capacity_for_new_work(
    tmp_path,
    monkeypatch,
):
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
        experiments=registry,
    )
    monkeypatch.setattr(service, "_start_operation", Mock())

    first = service.create(
        NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    )
    service.abort_generation("containing NWDAF process generation changed")
    registry.reset_generation()

    try:
        second_payload = preparation_payload()
        second_payload["notifCorreId"] = "notify-new"
        second = service.create(
            NwdafMLModelTrainSubsc.model_validate(second_payload)
        )

        assert second.subscription_id != first.subscription_id
    finally:
        service.abort_generation("test cleanup")
        registry.reset_generation()
        service.close()


def test_delete_rolls_back_unbound_client_reservation(tmp_path, monkeypatch):
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
        experiments=registry,
    )
    monkeypatch.setattr(service, "_start_operation", Mock())

    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(preparation_payload())
        )
        service.delete(resource.subscription_id)
        assert registry.active() is None
    finally:
        service.close()


def test_delete_cancels_bound_leaf_and_is_idempotent(tmp_path, monkeypatch):
    registry = FLExperimentRegistry()
    workspace = Mock()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        workspace,
        experiments=registry,
    )
    monkeypatch.setattr(service, "_start_operation", Mock())

    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(preparation_payload())
        )
        active = registry.active()
        plan_id = "11111111-1111-4111-8111-111111111111"
        registry.bind_plan(active.reservation_id, plan_id, ExperimentRole.LEAF)

        service.delete(resource.subscription_id)
        service.delete(resource.subscription_id)

        with pytest.raises(KeyError):
            service.get(resource.subscription_id)
        assert registry.active() is None
        assert registry.is_retired(plan_id)
        workspace.release_plan.assert_called_once_with(plan_id)
    finally:
        service.close()


def test_cancelled_client_resource_tombstone_is_pruned_lazily(tmp_path, monkeypatch):
    now = [10.0]
    registry = FLExperimentRegistry()
    settings = FederatedLearningSettings(
        workspace_root=tmp_path,
        lifecycle={"tombstone_ttl_seconds": 5},
    )
    service = FLClientEngine(
        settings,
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
        experiments=registry,
        clock=lambda: now[0],
    )
    monkeypatch.setattr(service, "_start_operation", Mock())
    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(preparation_payload())
        )
        active = registry.active()
        registry.bind_plan(
            active.reservation_id,
            "11111111-1111-4111-8111-111111111111",
            ExperimentRole.LEAF,
        )
        service.delete(resource.subscription_id)
        service.delete(resource.subscription_id)

        now[0] = 16.0

        barrier = threading.Barrier(3)
        outcomes = []

        def delete_expired_tombstone():
            barrier.wait()
            try:
                service.delete(resource.subscription_id)
            except KeyError:
                outcomes.append("not-found")

        threads = [threading.Thread(target=delete_expired_tombstone) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1)

        assert outcomes == ["not-found", "not-found"]
    finally:
        service.close()


def test_delete_still_rejects_in_progress_flat_preparation(tmp_path, monkeypatch):
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    monkeypatch.setattr(service, "_start_operation", Mock())

    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(preparation_payload())
        )
        service._resources[resource.subscription_id].state = FLClientState.PREPARING

        with pytest.raises(RuntimeError, match="ML_TRAINING_NOT_COMPLETE"):
            service.delete(resource.subscription_id)
        assert service.get(resource.subscription_id).state is FLClientState.PREPARING
    finally:
        service.close()


def test_duplicate_notification_correlation_is_rejected(tmp_path):
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        Mock(),
    )
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    try:
        service.create(value)
        try:
            service.create(value)
        except ValueError as error:
            assert "notifCorreId" in str(error)
        else:
            raise AssertionError("duplicate notifCorreId was accepted")
    finally:
        service.close()


@pytest.mark.parametrize(
    ("field", "value", "parameter"),
    [
        ("modelInterInfo", "unsupported", "mLEventSubscs[0].modelInterInfo"),
        ("timeAvReq", "five minutes", "mLModelTrainInfos[0].timeAvReq"),
        ("timeAvReq", "PT0S", "mLModelTrainInfos[0].timeAvReq"),
    ],
)
def test_preparation_rejects_unsupported_contract_requirements(
    tmp_path,
    field,
    value,
    parameter,
):
    payload = preparation_payload()
    if field == "modelInterInfo":
        payload["mLEventSubscs"][0][field] = value
    else:
        payload["mLModelTrainInfos"][0][field] = value
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    try:
        with pytest.raises(RequirementsError) as captured:
            service.create(NwdafMLModelTrainSubsc.model_validate(payload))
        assert [item.parameter for item in captured.value.violations] == [parameter]
    finally:
        service.close()


def test_preparation_uses_trainable_samples_instead_of_raw_records(tmp_path, caplog):
    caplog.set_level("INFO")
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
    )
    service._resources[resource.subscription_id] = resource
    service._dataset_builder = Mock()
    service._dataset_builder.build.return_value.training_scopes = ()
    service._enqueue_delivery = Mock()
    snapshot = Mock()
    snapshot.records = [Mock()] * 100
    job = Mock(
        state=DatasetJobState.READY,
        snapshot=snapshot,
        failure="",
    )
    assert service._capacity.acquire(blocking=False)
    try:
        service._preparation_complete(
            resource.subscription_id,
            resource.revision,
            job,
            {"analytics_event": "UE_COMMUNICATION"},
            "model-contract",
            "preprocessing-contract",
        )

        updated = service.get(resource.subscription_id)
        assert updated.state is FLClientState.FAILED
        assert "minNumSamples" in updated.last_error
        assert "prepared training dataset does not meet minNumSamples" in caplog.text
        service._enqueue_delivery.assert_called_once()
    finally:
        service.close()


def test_preparation_success_returns_validated_input_model_url(tmp_path):
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
    )
    service._resources[resource.subscription_id] = resource
    service._dataset_builder = Mock()
    service._dataset_builder.build.return_value.training_scopes = (
        Mock(training_sample_count=1),
    )
    service._enqueue_delivery = Mock()
    snapshot = Mock(records=[Mock()])
    job = Mock(state=DatasetJobState.READY, snapshot=snapshot, failure="")
    assert service._capacity.acquire(blocking=False)
    try:
        service._preparation_complete(
            resource.subscription_id,
            resource.revision,
            job,
            {"analytics_event": "UE_COMMUNICATION"},
            "model-contract",
            "preprocessing-contract",
        )

        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.round_indicator is None
        assert notification.status_report is None
        assert len(notification.ml_model_infos or ()) == 1
        address = notification.ml_model_infos[0].model_file_address
        assert address is not None
        assert str(address.model_url) == "http://server.example/base.tar.gz"
    finally:
        service.close()


@pytest.mark.parametrize(
    ("descriptor_present", "expected_state"),
    [(True, FLClientState.PREPARED), (False, FLClientState.FAILED)],
    ids=("consumer-collected", "missing-collected-precondition"),
)
def test_flat_preparation_uses_consumer_collected_absolute_window_snapshot(
    tmp_path,
    descriptor_present,
    expected_state,
):
    request = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    policy = Mock()
    resolver = Mock()
    resolver.resolve.return_value = "http://adrf.example"
    datasets = DatasetCoordinator(
        DatasetSettings(),
        Mock(get=Mock(return_value=Mock())),
        policy,
        resolver,
    )
    if descriptor_present:
        descriptor = TrainingDataDescriptor.model_validate(
            {
                "correlationId": "descriptor-static-flat",
                "state": "ACTIVE",
                "storedDataSpec": {
                    "dataSpec": {
                        "smfDataSub": {
                            "supi": "imsi-static-flat",
                            "notifId": "smf-static-flat",
                            "notifUri": "http://anlf.example/callback",
                            "eventSubs": [{"event": "UPF_EVENT"}],
                        }
                    },
                    "timePeriod": {
                        "startTime": "2026-07-01T00:00:00Z",
                        "stopTime": "2026-07-27T00:00:00Z",
                    },
                },
                "mlEventSubscription": {
                    "mLEvent": "UE_COMMUNICATION",
                    "mLEventFilter": request.ml_event_subscriptions[0].ml_event_filter,
                    "tgtUe": {"intGroupIds": ["group-G"]},
                },
                "sourceNfInstanceId": "11111111-1111-4111-8111-111111111111",
                "adrfInstanceId": "22222222-2222-4222-8222-222222222222",
                "retainUntil": "2099-08-04T10:30:00Z",
            }
        )
        datasets.put_training_data_descriptor(descriptor.correlation_id, descriptor)

    observation_start = datetime(2026, 7, 20, tzinfo=UTC)
    notification_items = [
        {
            "eventType": "USER_DATA_USAGE_MEASURES",
            "ueIpv4Addr": "10.0.0.1",
            "timeStamp": (observation_start + timedelta(seconds=index)).isoformat(),
            "userDataUsageMeasurements": [
                {
                    "volumeMeasurement": {
                        "totalVolume": 1,
                        "ulVolume": 2,
                        "dlVolume": 3,
                        "totalNbOfPackets": 4,
                        "ulNbOfPackets": 5,
                        "dlNbOfPackets": 6,
                    },
                    "throughputMeasurement": {
                        "ulThroughput": "1 Mbps",
                        "dlThroughput": "2 Mbps",
                        "ulPacketThroughput": "3 kpps",
                        "dlPacketThroughput": "4 kpps",
                    },
                }
            ],
        }
        for index in range(100)
    ]

    def retrieve_collected(job, _context):
        for resource in job.resources:
            datasets._append_record(
                job,
                resource,
                NadrfDataStoreRecord(
                    dataSub=[DataSubscription(smfDataSub=resource.smf_data_sub)],
                    dataNotif=DataNotification(
                        upfEventNotifs=[
                            {
                                "correlationId": "upf-static-flat",
                                "notificationItems": notification_items,
                            }
                        ]
                    ),
                ),
                "adrf",
                None,
                "collected-record-1",
            )

    datasets._retrieve_adrf = Mock(side_effect=retrieve_collected)
    workspace = Mock()
    artifact_path = tmp_path / "base.tar.gz"
    artifact_path.write_bytes(b"base")
    workspace.download.return_value = ArtifactMetadata(
        key="a" * 64,
        size_bytes=artifact_path.stat().st_size,
        path=artifact_path,
        url="http://server.example/base.tar.gz",
    )
    workspace.inspect_artifact.return_value = None
    base_manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 10, "output_size": 2},
        "inference": {
            "seq_length": 30,
            "out_seq_len": 1,
            "feature_order": list(FEATURE_ORDER),
            "output_fields": ["ul_vol", "dl_vol"],
            "preprocessing": "log1p_standard_scaler",
        },
        "file_digests": {
            "model.py": "1" * 64,
            "model.npy": "2" * 64,
            "scaler.pkl": "3" * 64,
        },
    }
    service = FLClientEngine(
        fl_settings(tmp_path / "workspaces"),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        workspace,
    )
    service._loader = Mock()
    service._loader.load.return_value = Mock(manifest=base_manifest)
    terminal = threading.Event()

    def record_delivery(resource, _notification, final_state):
        resource.state = final_state
        terminal.set()

    service._enqueue_delivery = Mock(side_effect=record_delivery)
    try:
        created = service.create(request)
        assert terminal.wait(timeout=2)
        resource = service.get(created.subscription_id)

        assert resource.state is expected_state
        if descriptor_present:
            assert resource.dataset_snapshot is not None
            assert resource.dataset_snapshot.source == "adrf"
            assert resource.dataset_snapshot.time_window.start_time == datetime(
                2026, 7, 1, tzinfo=UTC
            )
            assert resource.dataset_snapshot.time_window.stop_time == datetime(
                2026, 7, 27, tzinfo=UTC
            )
            assert tuple(record.identity for record in resource.dataset_snapshot.records) == (
                "collected-record-1",
            )
        else:
            assert resource.dataset_snapshot is None
            assert "no usable training-data descriptor" in resource.last_error
            resolver.resolve.assert_not_called()
            datasets._retrieve_adrf.assert_not_called()
    finally:
        service.close()
        datasets.shutdown()


@pytest.mark.parametrize("scope_available", [True, False], ids=("ready", "missing"))
def test_leaf_assignment_binds_plan_before_local_data_preparation(
    tmp_path,
    scope_available,
):
    branch_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    leaf_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    plan_id = "11111111-1111-4111-8111-111111111111"
    assignment_url = "http://branch.example/artifacts/" + "a" * 64
    payload = preparation_payload()
    payload["mLModelInfos"][0]["mLFileAddr"]["mLModelUrl"] = assignment_url
    value = NwdafMLModelTrainSubsc.model_validate(payload)
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
                "message_type": HierarchyMessageType.LEAF_ASSIGNMENT,
                "plan_id": plan_id,
                "publisher_nf_instance_id": branch_id,
                "intended_recipient_nf_instance_id": leaf_id,
                "parent_branch_nf_instance_id": branch_id,
                "strategy": {
                    "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
                    "participant_selection": "all",
                    "waiting_policy": "all",
                    "aggregation": "sample_weighted",
                },
            },
        }
    )
    generic_path = tmp_path / "generic-assignment.tar.gz"
    generic_path.write_bytes(b"validated archive")
    generic = ArtifactMetadata(
        key="a" * 64,
        size_bytes=generic_path.stat().st_size,
        path=generic_path,
        url=assignment_url,
    )
    admitted_path = tmp_path / "admitted-assignment.tar.gz"
    admitted_path.write_bytes(b"admitted archive")
    admitted_metadata = ArtifactMetadata(
        key="a" * 64,
        size_bytes=admitted_path.stat().st_size,
        path=admitted_path,
        url=assignment_url,
    )
    manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    admitted = ValidatedHierarchyArtifact(
        metadata=admitted_metadata,
        manifest=manifest,
        contract=contract,
    )
    workspace = Mock()
    workspace.download.return_value = generic
    workspace.inspect_artifact.return_value = ValidatedArchive(
        manifest=manifest,
        contract=contract,
    )
    workspace.download_assignment.return_value = admitted
    context_client = Mock()
    context_client.get.return_value = NwdafContext(
        nf_instance_id=leaf_id,
        containing_nwdaf_process_instance_id="22222222-2222-4222-8222-222222222222",
        api_root="http://leaf.example",
        internal_api_root="http://leaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    if not scope_available:
        datasets.validate_external_scope.side_effect = RuntimeError(
            "scope has no usable training-data descriptor"
        )
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("resource-1", value.ml_correlation_id or "")
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context_client,
        datasets,
        workspace,
        experiments=registry,
    )
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        experiment_reservation_id=reservation.reservation_id,
    )
    service._resources[resource.subscription_id] = resource
    service._loader = Mock()
    service._enqueue_delivery = Mock()
    service._loader.load.return_value.manifest = manifest
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_preparation(
            resource.subscription_id,
            resource.revision,
            Mock(),
            Mock(),
        )

        active = registry.active()
        updated = service.get(resource.subscription_id)
        assert active is not None
        assert active.plan_id == plan_id
        assert active.assigned_role is ExperimentRole.LEAF
        assert updated.hierarchy_assignment == admitted
        assert updated.preparation_base_artifact == admitted_metadata
        if scope_available:
            datasets.submit_external.assert_called_once()
        else:
            assert updated.state is FLClientState.FAILED
            assert "no usable training-data descriptor" in updated.last_error
            datasets.submit_external.assert_not_called()
            notification = service._enqueue_delivery.call_args.args[1]
            assert notification.termination_request == "NOT_AVAILABLE_ML_TRAIN"
        assert not generic_path.exists()
        workspace.download_assignment.assert_called_once_with(
            assignment_url,
            intended_recipient_nf_instance_id=leaf_id,
        )
    finally:
        service.close()


@pytest.mark.parametrize(
    ("branch_outcome", "expected_termination"),
    [
        (PreparationOutcome.READY, None),
        (PreparationOutcome.FAILED, "NOT_AVAILABLE_ML_TRAIN"),
    ],
)
def test_branch_assignment_binds_plan_and_dispatches_without_local_dataset(
    tmp_path,
    branch_outcome,
    expected_termination,
):
    root_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    branch_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    leaf_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    plan_id = "11111111-1111-4111-8111-111111111111"
    assignment_url = "http://root.example/artifacts/" + "a" * 64
    payload = preparation_payload()
    payload["mLModelInfos"][0]["mLFileAddr"]["mLModelUrl"] = assignment_url
    value = NwdafMLModelTrainSubsc.model_validate(payload)
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
                "message_type": HierarchyMessageType.BRANCH_ASSIGNMENT,
                "plan_id": plan_id,
                "publisher_nf_instance_id": root_id,
                "intended_recipient_nf_instance_id": branch_id,
                "assigned_leaf_nf_instance_ids": [leaf_id],
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
    generic_path = tmp_path / "generic-branch.tar.gz"
    generic_path.write_bytes(b"generic")
    admitted_path = tmp_path / "admitted-branch.tar.gz"
    admitted_path.write_bytes(b"admitted")
    generic = ArtifactMetadata(
        key="a" * 64,
        size_bytes=generic_path.stat().st_size,
        path=generic_path,
        url=assignment_url,
    )
    admitted = ValidatedHierarchyArtifact(
        metadata=ArtifactMetadata(
            key="a" * 64,
            size_bytes=admitted_path.stat().st_size,
            path=admitted_path,
            url=assignment_url,
        ),
        manifest={
            "analytics_event": "UE_COMMUNICATION",
            "model_interoperability": "001122",
        },
        contract=contract,
    )
    workspace = Mock()
    workspace.download.return_value = generic
    workspace.inspect_artifact.return_value = ValidatedArchive(
        manifest=admitted.manifest,
        contract=contract,
    )
    workspace.download_assignment.return_value = admitted
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=branch_id,
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
    datasets = Mock()
    branch = Mock()
    branch.prepare.return_value = Mock(
        execution=Mock(process_id="lower-process"),
        artifact=Mock(url="http://branch.example/preparation-result"),
        outcome=branch_outcome,
    )
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("resource-1", value.ml_correlation_id or "")
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        datasets,
        workspace,
        experiments=registry,
        branch_coordinator=branch,
    )
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        experiment_reservation_id=reservation.reservation_id,
    )
    service._resources[resource.subscription_id] = resource
    service._loader = Mock()
    service._enqueue_delivery = Mock()
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_preparation(resource.subscription_id, resource.revision, Mock(), Mock())

        active = registry.active()
        updated = service.get(resource.subscription_id)
        assert active is not None
        assert active.plan_id == plan_id
        assert active.assigned_role is ExperimentRole.BRANCH
        assert updated.branch_process_id == "lower-process"
        assert updated.hierarchy_assignment == admitted
        assert updated.expected_model_contract_digest == model_contract_digest(
            admitted.manifest
        )
        assert updated.expected_preprocessing_contract_digest == (
            preprocessing_contract_digest(admitted.manifest)
        )
        branch.prepare.assert_called_once_with(
            assignment=admitted,
            representation=value,
            reservation_id=reservation.reservation_id,
        )
        datasets.validate_external_scope.assert_not_called()
        datasets.submit_external.assert_not_called()
        service._loader.load.assert_not_called()
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.termination_request == expected_termination
        assert (
            str(notification.ml_model_infos[0].model_file_address.model_url)
            == "http://branch.example/preparation-result"
        )
    finally:
        service.close()


def test_preparation_rejects_base_bundle_with_different_interoperability(tmp_path):
    datasets = Mock()
    workspace = Mock()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        workspace,
    )
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
    )
    service._resources[resource.subscription_id] = resource
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "different-contract",
    }
    service._enqueue_delivery = Mock()
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_preparation(
            resource.subscription_id,
            resource.revision,
            Mock(),
            Mock(),
        )

        updated = service.get(resource.subscription_id)
        assert updated.state is FLClientState.FAILED
        assert "interoperability" in updated.last_error
        datasets.submit_external.assert_not_called()
    finally:
        service.close()


def test_leaf_scope_failure_is_reported_as_async_preparation_termination(tmp_path):
    datasets = Mock()
    datasets.validate_external_scope.side_effect = RuntimeError(
        "scope has no usable training-data descriptor"
    )
    workspace = Mock()
    workspace.inspect_artifact.return_value = None
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        workspace,
    )
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.PREPARING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
    )
    service._resources[resource.subscription_id] = resource
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    service._enqueue_delivery = Mock()
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_preparation(
            resource.subscription_id,
            resource.revision,
            Mock(),
            Mock(),
        )

        updated = service.get(resource.subscription_id)
        assert updated.state is FLClientState.FAILED
        assert "no usable training-data descriptor" in updated.last_error
        datasets.submit_external.assert_not_called()
        service._enqueue_delivery.assert_called_once()
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.notification_correlation_id == "prep-client-a"
        assert notification.ml_correlation_id == "fl-process-001"
        assert notification.termination_request == "NOT_AVAILABLE_ML_TRAIN"
        assert service._enqueue_delivery.call_args.args[2] is FLClientState.FAILED
    finally:
        service.close()


def test_deadline_extension_patch_does_not_restart_preparation(tmp_path):
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        datasets,
        Mock(),
    )
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    try:
        resource = service.create(NwdafMLModelTrainSubsc.model_validate(preparation_payload()))
        updated = service.patch(
            resource.subscription_id,
            NwdafMLModelTrainSubscPatch.model_validate({"mLTrainRepInfo": {"maxResTime": 600}}),
        )
        assert updated.state is FLClientState.PREPARING
        assert updated.representation.ml_training_report_info.maximum_response_time == 600
        datasets.submit_external.assert_called_once()
    finally:
        service.close()


def test_accuracy_check_patch_enters_validation_without_training(tmp_path):
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 1,
        }
    )
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=NwdafMLModelTrainSubsc.model_validate(payload),
        state=FLClientState.READY,
        scope=TrainingScopeDescriptor.from_training_request(
            NwdafMLModelTrainSubsc.model_validate(payload),
            0,
        ),
        dataset_snapshot=Mock(),
        prepared_training_sample_count=10,
        expected_model_contract_digest="a" * 64,
        expected_preprocessing_contract_digest="b" * 64,
        preparation_base_artifact=Mock(),
    )
    service._resources[resource.subscription_id] = resource
    service._submit = Mock()
    service._trainer = Mock()
    try:
        updated = service.patch(
            resource.subscription_id,
            NwdafMLModelTrainSubscPatch.model_validate(
                {
                    "mLAccChkFlg": True,
                    "skipFlInd": True,
                    "roundInd": 2,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {
                                "mLModelUrl": "http://server.example/final-candidate.tar.gz"
                            },
                        }
                    ],
                }
            ),
        )

        assert updated.state is FLClientState.VALIDATION_RUNNING
        submitted = service._submit.call_args.args
        assert submitted[0].__name__ == "_run_validation"
        service._trainer.train.assert_not_called()
    finally:
        service.close()


@pytest.mark.parametrize("hierarchy_leaf", [False, True])
def test_final_validation_uses_configured_training_device(
    tmp_path,
    monkeypatch,
    hierarchy_leaf,
):
    nwdaf_context = Mock()
    assignment = hierarchy_assignment(tmp_path, branch=False) if hierarchy_leaf else None
    nwdaf_context.get.return_value.nf_instance_id = (
        assignment.contract.hierarchy_metadata.intended_recipient_nf_instance_id
        if assignment is not None
        else "participant-a"
    )
    workspace = Mock()
    workspace.download.return_value = Mock()
    workspace.publish.return_value.url = "http://client.example/validation.tar.gz"
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        nwdaf_context,
        Mock(),
        workspace,
    )
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://server.example/final-candidate.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    snapshot = Mock()
    snapshot.time_window.start_time = datetime(2026, 7, 1, tzinfo=UTC)
    snapshot.time_window.stop_time = datetime(2026, 7, 2, tzinfo=UTC)
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.VALIDATION_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        dataset_snapshot=snapshot,
        prepared_training_sample_count=10,
        expected_model_contract_digest="a" * 64,
        expected_preprocessing_contract_digest="b" * 64,
        preparation_base_artifact=Mock(),
        hierarchy_assignment=assignment,
    )
    service._resources[resource.subscription_id] = resource
    base = Mock(manifest={"bundle": "base"}, model=Mock(), scaler=Mock())
    candidate = round_global_bundle()
    if hierarchy_leaf:
        candidate.manifest["fl_metadata"]["ml_corre_id"] = "root-process"
    candidate.manifest["fl_metadata"]["weights_digest"] = "c" * 64
    candidate.scaler = Mock()
    service._loader = Mock()
    service._loader.load.side_effect = [base, candidate]
    scope = Mock(
        training_sample_count=10,
        validation_sample_count=2,
        validation_targets=np.asarray([2.0, 4.0]),
    )
    dataset = Mock(training_scopes=(scope,), evaluation_scopes=(scope,))
    service._dataset_builder = Mock()
    service._dataset_builder.build.return_value = dataset
    service._trainer = Mock()
    service._enqueue_delivery = Mock()
    predict = Mock(
        side_effect=(np.asarray([1.0, 3.0]), np.asarray([2.0, 3.0]))
    )
    monkeypatch.setattr(LocalTrainer, "_predict", predict)
    monkeypatch.setattr(
        "py_mtlf.core.fl_client.model_contract_digest", lambda _: "a" * 64
    )
    monkeypatch.setattr(
        "py_mtlf.core.fl_client.preprocessing_contract_digest", lambda _: "b" * 64
    )
    monkeypatch.setattr("py_mtlf.core.fl_client.weights_digest", lambda _: "c" * 64)
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_validation(resource.subscription_id, resource.revision)

        assert service.get(resource.subscription_id).state is FLClientState.RESULT_PENDING
        assert predict.call_count == 2
        assert all(call.args[4] == service._device for call in predict.call_args_list)
        service._enqueue_delivery.assert_called_once()
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.ml_correlation_id == value.ml_correlation_id
        assert workspace.publish.call_args.kwargs["model"] is candidate.model
        service._trainer.train.assert_not_called()
    finally:
        service.close()


def test_flat_round_and_final_validation_reuse_the_prepared_snapshot(
    tmp_path,
    monkeypatch,
):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://server.example/round-input.tar.gz"
                    },
                }
            ],
        }
    )
    round_value = NwdafMLModelTrainSubsc.model_validate(payload)
    base = round_input_bundle(epochs=1)
    candidate = round_global_bundle()
    candidate.manifest["fl_metadata"]["round_ind"] = 2
    workspace = Mock()
    workspace.download.side_effect = (Mock(name="round-input"), Mock(name="candidate"))
    workspace.publish.side_effect = (
        Mock(url="http://client.example/local.tar.gz"),
        Mock(url="http://client.example/validation.tar.gz"),
    )
    nwdaf_context = Mock()
    nwdaf_context.get.return_value.nf_instance_id = "participant-a"
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        nwdaf_context,
        Mock(),
        workspace,
    )
    service._loader = Mock()
    service._loader.load.side_effect = (base, base, candidate)
    training_scope = Mock(
        training_sample_count=10,
        validation_sample_count=2,
        validation_targets=np.asarray([2.0, 4.0]),
    )
    dataset = Mock(
        training_scopes=(training_scope,),
        evaluation_scopes=(training_scope,),
    )
    service._dataset_builder = Mock(return_value=dataset)
    service._dataset_builder.build.return_value = dataset
    result_model = torch.nn.Linear(2, 1)
    service._trainer = Mock()
    service._trainer.train.return_value = Mock(
        model=result_model,
        training_sample_count=10,
    )
    service._enqueue_delivery = Mock()
    predict = Mock(
        side_effect=(np.asarray([1.0, 3.0]), np.asarray([2.0, 3.0]))
    )
    monkeypatch.setattr(LocalTrainer, "_predict", predict)
    snapshot = Mock()
    snapshot.time_window.start_time = datetime(2026, 7, 1, tzinfo=UTC)
    snapshot.time_window.stop_time = datetime(2026, 7, 2, tzinfo=UTC)
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=round_value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(round_value, 0),
        dataset_snapshot=snapshot,
        prepared_training_sample_count=10,
        expected_model_contract_digest=model_contract_digest(base.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            base.manifest
        ),
        preparation_base_artifact=Mock(name="preparation-base"),
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_round(resource.subscription_id, resource.revision)

        validation_payload = preparation_payload()
        validation_payload.update(
            {
                "mLPreFlag": False,
                "mLAccChkFlg": True,
                "skipFlInd": True,
                "roundInd": 3,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {
                            "mLModelUrl": "http://server.example/candidate.tar.gz"
                        },
                    }
                ],
            }
        )
        resource.representation = NwdafMLModelTrainSubsc.model_validate(
            validation_payload
        )
        resource.state = FLClientState.VALIDATION_RUNNING
        resource.work_slot_owned = True
        assert service._capacity.acquire(blocking=False)
        service._run_validation(resource.subscription_id, resource.revision)

        assert [
            item.args[0] for item in service._dataset_builder.build.call_args_list
        ] == [snapshot, snapshot]
        assert resource.dataset_snapshot is snapshot
        assert service._trainer.train.call_count == 1
        assert predict.call_count == 2
    finally:
        service.close()


def test_round_termination_preserves_round_identity():
    resource = Mock()
    resource.representation.notification_correlation_id = "round-client-a"
    resource.representation.ml_correlation_id = "fl-process-001"
    resource.representation.round_indicator = 3

    notification = _termination(resource)

    assert notification.round_indicator == 3
    assert notification.termination_request == "NOT_AVAILABLE_ML_TRAIN"


def test_callback_outbox_retries_the_same_notification_until_ack(tmp_path):
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    client = Mock()
    client.post.side_effect = [
        httpx.Response(503),
        httpx.Response(204),
    ]
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(
            max_attempts=1,
            initial_backoff_seconds=0.001,
            max_backoff_seconds=0.001,
        ),
        Mock(),
        datasets,
        Mock(),
        client=client,
    )
    service._loader = Mock()
    service._loader.load.return_value.manifest = {
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
    }
    try:
        resource = service.create(NwdafMLModelTrainSubsc.model_validate(preparation_payload()))
        notification = NwdafMLModelTrainNotif.model_validate(
            {
                "notifCorreId": "prep-client-a",
                "mlCorreId": "fl-process-001",
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {
                            "mLModelUrl": "http://server.example/base.tar.gz"
                        },
                    }
                ],
            }
        )
        service._enqueue_delivery(resource, notification, FLClientState.PREPARED)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if service.get(resource.subscription_id).state is FLClientState.PREPARED:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("callback outbox did not retry through acknowledgement")
        assert client.post.call_count == 2
        assert (
            client.post.call_args_list[0].kwargs["json"]
            == client.post.call_args_list[1].kwargs["json"]
        )
    finally:
        service.close()


def test_duplicate_round_patch_is_idempotent_and_conflict_is_rejected(tmp_path):
    value = NwdafMLModelTrainSubsc.model_validate(
        {
            **preparation_payload(),
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://server.example/global.tar.gz"},
                }
            ],
        }
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.READY,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
    )
    service._resources[resource.subscription_id] = resource
    try:
        same = service.patch(
            resource.subscription_id,
            NwdafMLModelTrainSubscPatch.model_validate(
                {
                    "mLPreFlag": False,
                    "roundInd": 2,
                    "mLModelInfos": [
                        {
                            "event": "UE_COMMUNICATION",
                            "mLFileAddr": {"mLModelUrl": "http://server.example/global.tar.gz"},
                        }
                    ],
                }
            ),
        )
        assert same.revision == 1

        try:
            service.patch(
                resource.subscription_id,
                NwdafMLModelTrainSubscPatch.model_validate(
                    {
                        "roundInd": 2,
                        "mLModelInfos": [
                            {
                                "event": "UE_COMMUNICATION",
                                "mLFileAddr": {
                                    "mLModelUrl": "http://server.example/conflict.tar.gz"
                                },
                            }
                        ],
                    }
                ),
            )
        except RuntimeError as error:
            assert "conflicting" in str(error)
        else:
            raise AssertionError("conflicting duplicate round was accepted")
    finally:
        service.close()


@pytest.mark.parametrize("stale_before_callback", [False, True])
def test_branch_round_delegates_without_local_dataset_or_training(
    tmp_path,
    stale_before_callback,
):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://root.example/round-input.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    base = round_input_bundle(epochs=7)
    assignment = hierarchy_assignment(tmp_path, branch=True)
    workspace = Mock()
    workspace.download.return_value = Mock(key="4" * 64)
    branch = Mock()
    branch.execute_round.return_value = Mock(
        url="http://branch.example/hierarchy-aggregate.tar.gz"
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        workspace,
        branch_coordinator=branch,
    )
    service._loader = Mock()
    service._loader.load.return_value = base
    service._dataset_builder = Mock()
    service._trainer = Mock()
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        expected_model_contract_digest=model_contract_digest(base.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            base.manifest
        ),
        hierarchy_assignment=assignment,
    )
    service._resources[resource.subscription_id] = resource
    if stale_before_callback:
        def retire_resource(**_kwargs):
            service._resources.pop(resource.subscription_id)
            return Mock(url="http://branch.example/hierarchy-aggregate.tar.gz")

        branch.execute_round.side_effect = retire_resource
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_round(resource.subscription_id, resource.revision)

        branch.execute_round.assert_called_once_with(
            assignment=assignment,
            representation=value,
            upper_input=base,
            upper_client_subscription_id=resource.subscription_id,
            upper_resource_revision=resource.revision,
            upper_input_artifact_digest="4" * 64,
            upper_scope_digest=resource.scope.scope_digest,
            callback_margin_seconds=client_settings().callback_deadline_margin_seconds,
        )
        service._dataset_builder.build.assert_not_called()
        service._trainer.train.assert_not_called()
        if stale_before_callback:
            service._enqueue_delivery.assert_not_called()
            return
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.round_indicator == 2
        assert (
            str(notification.ml_model_infos[0].model_file_address.model_url)
            == "http://branch.example/hierarchy-aggregate.tar.gz"
        )
    finally:
        service.close()


@pytest.mark.parametrize("delete_during_validation", [False, True])
def test_branch_validation_delegates_without_local_dataset_or_local_metrics(
    tmp_path,
    delete_during_validation,
):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://root.example/round-global.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    candidate = round_global_bundle()
    base = Mock(manifest=candidate.manifest, model=Mock())
    assignment = hierarchy_assignment(tmp_path, branch=True)
    candidate_artifact = Mock(key="4" * 64)
    workspace = Mock()
    workspace.download.return_value = candidate_artifact
    branch = Mock()
    branch.execute_validation.return_value = Mock(
        url="http://branch.example/validation-result.tar.gz"
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        workspace,
        branch_coordinator=branch,
    )
    service._loader = Mock()
    service._loader.load.side_effect = [base, candidate]
    service._dataset_builder = Mock()
    service._trainer = Mock()
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.VALIDATION_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        expected_model_contract_digest=model_contract_digest(candidate.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            candidate.manifest
        ),
        preparation_base_artifact=Mock(),
        hierarchy_assignment=assignment,
    )
    service._resources[resource.subscription_id] = resource
    dispatched_revision = resource.revision
    if delete_during_validation:
        assert service._outbox_capacity.acquire(blocking=False)

        def delete_parent(**_kwargs):
            service.delete(resource.subscription_id)
            return Mock(url="http://branch.example/validation-result.tar.gz")

        branch.execute_validation.side_effect = delete_parent
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_validation(resource.subscription_id, dispatched_revision)

        branch.execute_validation.assert_called_once_with(
            assignment=assignment,
            representation=value,
            upper_candidate=candidate,
            upper_candidate_artifact=candidate_artifact,
            upper_client_subscription_id=resource.subscription_id,
            upper_resource_revision=dispatched_revision,
            upper_scope_digest=resource.scope.scope_digest,
            callback_margin_seconds=client_settings().callback_deadline_margin_seconds,
        )
        service._dataset_builder.build.assert_not_called()
        service._trainer.train.assert_not_called()
        if delete_during_validation:
            service._enqueue_delivery.assert_not_called()
            assert resource.subscription_id not in service._resources
            return
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.round_indicator == 2
        assert (
            str(notification.ml_model_infos[0].model_file_address.model_url)
            == "http://branch.example/validation-result.tar.gz"
        )
    finally:
        service.close()


def test_parent_delete_cancels_real_branch_validation_and_fences_callback(tmp_path):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://root.example/round-global.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    candidate = round_global_bundle()
    base = Mock(manifest=candidate.manifest, model=Mock())
    assignment = hierarchy_assignment(tmp_path, branch=True)
    metadata = assignment.contract.hierarchy_metadata
    plan_id = metadata.plan_id
    candidate_artifact = Mock(key="4" * 64)
    republished = Mock(
        url="http://branch.example/validation-candidate.tar.gz",
        digest="4" * 64,
    )
    lower_started = threading.Event()
    release_lower = threading.Event()
    server = Mock()

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
    artifacts = Mock()
    artifacts.republish_validation_candidate.return_value = republished
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id=metadata.intended_recipient_nf_instance_id,
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
    branch = FLBranchPreparationCoordinator(
        resolver=Mock(),
        nwdaf_context=context,
        artifact_service=artifacts,
        server=server,
    )
    branch._executions[plan_id] = BranchPreparationExecution(
        plan_id=plan_id,
        parent_assignment=assignment,
        leaf_nodes=(Mock(),),
        leaf_assignments=(Mock(),),
        process_id="lower-process",
    )
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("resource-1", value.ml_correlation_id or "")
    registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.BRANCH)
    workspace = Mock()
    workspace.download.return_value = candidate_artifact
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        workspace,
        experiments=registry,
        branch_coordinator=branch,
    )
    service._loader = Mock()
    service._loader.load.side_effect = [base, candidate]
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.VALIDATION_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        expected_model_contract_digest=model_contract_digest(candidate.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            candidate.manifest
        ),
        preparation_base_artifact=Mock(),
        hierarchy_assignment=assignment,
        experiment_reservation_id=reservation.reservation_id,
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    assert service._outbox_capacity.acquire(blocking=False)
    thread = threading.Thread(
        target=service._run_validation,
        args=(resource.subscription_id, resource.revision),
    )
    thread.start()
    try:
        assert lower_started.wait(1) is True
        service.delete(resource.subscription_id)
        thread.join(timeout=1)

        assert not thread.is_alive()
        assert resource.subscription_id not in service._resources
        assert registry.active() is None
        assert (plan_id, value.ml_correlation_id, value.round_indicator) not in (
            branch._validations
        )
        artifacts.publish_hierarchy_validation_result.assert_not_called()
        service._enqueue_delivery.assert_not_called()
        server.cancel_hierarchy_preparation.assert_called_once_with(
            "lower-process",
            "parent cancelled preparation",
        )
    finally:
        release_lower.set()
        thread.join(timeout=1)
        service.close()
        branch.close()


def test_leaf_round_uses_server_epochs_and_assignment_proximal_mu(tmp_path):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://branch.example/round-input.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    base = round_input_bundle(epochs=7)
    assignment = hierarchy_assignment(tmp_path, branch=False)
    workspace = Mock()
    workspace.download.return_value = Mock()
    workspace.publish.return_value = Mock(url="http://leaf.example/local.tar.gz")
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        workspace,
    )
    service._loader = Mock()
    service._loader.load.return_value = base
    training_scope = Mock(training_sample_count=10)
    dataset = Mock(training_scopes=(training_scope,))
    service._dataset_builder = Mock()
    service._dataset_builder.build.return_value = dataset
    result_model = torch.nn.Linear(2, 1)
    service._trainer = Mock()
    service._trainer.train.return_value = Mock(
        model=result_model,
        training_sample_count=10,
    )
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        dataset_snapshot=Mock(),
        prepared_training_sample_count=10,
        expected_model_contract_digest=model_contract_digest(base.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            base.manifest
        ),
        hierarchy_assignment=assignment,
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_round(resource.subscription_id, resource.revision)

        service._trainer.train.assert_called_once_with(
            base,
            dataset,
            epochs=7,
            proximal_mu=0.01,
        )
        metadata = workspace.publish.call_args.kwargs["metadata"]
        assert metadata["fl_metadata"]["training_sample_count"] == 10
        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.round_indicator == 2
    finally:
        service.close()


def test_go_generation_reset_drops_leaf_result_published_during_abort(tmp_path):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://branch.example/round-input.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    base = round_input_bundle(epochs=1)
    assignment = hierarchy_assignment(tmp_path, branch=False)
    plan_id = assignment.contract.hierarchy_metadata.plan_id
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("resource-1", value.ml_correlation_id)
    registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.LEAF)
    publish_started = threading.Event()
    allow_publish = threading.Event()
    workspace = Mock()
    workspace.download.return_value = Mock()

    def publish_during_abort(**_kwargs):
        publish_started.set()
        assert allow_publish.wait(1)
        return Mock(url="http://leaf.example/local.tar.gz")

    workspace.publish.side_effect = publish_during_abort
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        workspace,
        experiments=registry,
    )
    service._loader = Mock()
    service._loader.load.return_value = base
    service._dataset_builder = Mock(
        build=Mock(return_value=Mock(training_scopes=(Mock(training_sample_count=10),)))
    )
    service._trainer = Mock(
        train=Mock(
            return_value=Mock(
                model=torch.nn.Linear(2, 1),
                training_sample_count=10,
            )
        )
    )
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        dataset_snapshot=Mock(),
        prepared_training_sample_count=10,
        expected_model_contract_digest=model_contract_digest(base.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            base.manifest
        ),
        hierarchy_assignment=assignment,
        experiment_reservation_id=reservation.reservation_id,
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    assert service._outbox_capacity.acquire(blocking=False)
    thread = threading.Thread(
        target=service._run_round,
        args=(resource.subscription_id, resource.revision),
    )
    thread.start()
    assert publish_started.wait(1)

    service.abort_generation("containing NWDAF process generation changed")
    allow_publish.set()
    thread.join(timeout=1)

    try:
        assert thread.is_alive() is False
        service._enqueue_delivery.assert_not_called()
        workspace.release_plan.assert_called_once_with(plan_id)
        assert service._capacity.acquire(blocking=False)
        service._capacity.release()
    finally:
        registry.reset_generation()
        service.close()


def test_flat_client_uses_server_epochs_without_changing_local_objective(tmp_path):
    payload = preparation_payload()
    payload.update(
        {
            "mLPreFlag": False,
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": "http://server.example/round-input.tar.gz"
                    },
                }
            ],
        }
    )
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    base = round_input_bundle(epochs=6)
    workspace = Mock()
    workspace.download.return_value = Mock()
    workspace.publish.return_value = Mock(url="http://client.example/local.tar.gz")
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        workspace,
    )
    service._loader = Mock()
    service._loader.load.return_value = base
    training_scope = Mock(training_sample_count=10)
    dataset = Mock(training_scopes=(training_scope,))
    service._dataset_builder = Mock()
    service._dataset_builder.build.return_value = dataset
    result_model = torch.nn.Linear(2, 1)
    service._trainer = Mock()
    service._trainer.train.return_value = Mock(
        model=result_model,
        training_sample_count=10,
    )
    service._enqueue_delivery = Mock()
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        dataset_snapshot=Mock(),
        prepared_training_sample_count=10,
        expected_model_contract_digest=model_contract_digest(base.manifest),
        expected_preprocessing_contract_digest=preprocessing_contract_digest(
            base.manifest
        ),
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_round(resource.subscription_id, resource.revision)

        service._trainer.train.assert_called_once_with(
            base,
            dataset,
            epochs=6,
            proximal_mu=None,
        )
        assert service.get(resource.subscription_id).state is FLClientState.RESULT_PENDING
    finally:
        service.close()
