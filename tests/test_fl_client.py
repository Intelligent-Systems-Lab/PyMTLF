import time
from datetime import UTC, datetime
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from py_mtlf.config import (
    FederatedLearningSettings,
    FLClientSettings,
    NotificationSettings,
)
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.dataset import DatasetJobState
from py_mtlf.core.fl_artifacts import HierarchyAssignmentArtifact
from py_mtlf.core.fl_client import (
    FLClientCapacityError,
    FLClientEngine,
    FLClientResource,
    FLClientState,
    _termination,
)
from py_mtlf.core.fl_experiment import ExperimentRole, FLExperimentRegistry
from py_mtlf.core.fl_hierarchy import HierarchyMessageType, PreparationOutcome
from py_mtlf.core.fl_workspace import ValidatedArchive, ValidatedHierarchyArtifact
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.core.trainer import LocalTrainer
from py_mtlf.core.training_scope import TrainingScopeDescriptor
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
    return FLClientSettings(model_interoperability_ids=("001122",))


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


def test_create_reserves_same_correlation_group_and_rejects_another(tmp_path, monkeypatch):
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        FLClientSettings(
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


def test_preparation_uses_trainable_samples_instead_of_raw_records(tmp_path):
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


def test_leaf_assignment_binds_plan_before_local_data_preparation(tmp_path):
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
        datasets.submit_external.assert_called_once()
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
        branch.prepare.assert_called_once_with(
            assignment=admitted,
            representation=value,
            reservation_id=reservation.reservation_id,
        )
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


def test_final_validation_uses_configured_training_device(tmp_path, monkeypatch):
    nwdaf_context = Mock()
    nwdaf_context.get.return_value.nf_instance_id = "participant-a"
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
    )
    service._resources[resource.subscription_id] = resource
    base = Mock(manifest={"bundle": "base"}, model=Mock(), scaler=Mock())
    candidate = Mock(manifest={"bundle": "candidate"}, model=Mock(), scaler=Mock())
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


def test_restart_terminal_resource_rejects_future_update(tmp_path):
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    service._resources["resource-1"] = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.FAILED_RESTART,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        restart_terminal=True,
    )
    try:
        try:
            service.patch(
                "resource-1",
                NwdafMLModelTrainSubscPatch.model_validate({"mLTrainRepInfo": {"maxResTime": 600}}),
            )
        except RuntimeError as error:
            assert "NOT_AVAILABLE_FOR_FL_PROCESS_ANYMORE" in str(error)
        else:
            raise AssertionError("restart terminal resource accepted an update")
    finally:
        service.close()
