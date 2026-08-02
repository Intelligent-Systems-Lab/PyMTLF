import time
from unittest.mock import Mock

import httpx
import pytest

from py_mtlf.config import (
    FederatedLearningSettings,
    FLClientSettings,
    NotificationSettings,
)
from py_mtlf.core.dataset import DatasetJobState
from py_mtlf.core.fl_client import (
    FLClientResource,
    FLClientService,
    FLClientState,
    _termination,
)
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
    service = FLClientService(
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
        assert resource.dataset_job_id == "dataset-job-1"
        datasets.submit_external.assert_called_once()
    finally:
        service.close()


def test_duplicate_notification_correlation_is_rejected(tmp_path):
    datasets = Mock()
    datasets.submit_external.return_value = "dataset-job-1"
    service = FLClientService(
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
    service = FLClientService(
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
    service = FLClientService(
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


def test_preparation_rejects_base_bundle_with_different_interoperability(tmp_path):
    datasets = Mock()
    workspace = Mock()
    service = FLClientService(
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
    service = FLClientService(
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
    service = FLClientService(
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
    service = FLClientService(
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
                "statusReport": {"trainInDataInfo": {"samplRatio": 100}},
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
    service = FLClientService(
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
    service = FLClientService(
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
