import json
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
from py_mtlf.core.fl_client import (
    FLClientCapacityError,
    FLClientEngine,
    FLClientResource,
    FLClientState,
    _termination,
)
from py_mtlf.core.fl_experiment import ExperimentRole, FLExperimentRegistry
from py_mtlf.core.fl_workspace import (
    DownloadedArchive,
    ValidatedArchive,
)
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.core.trainer import LocalTrainer
from py_mtlf.core.training_data import FEATURE_ORDER, ScopeTrainingData, TrainingDataset
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.core.workloads import WorkloadProfile
from py_mtlf.models import TrainingDataDescriptor
from py_mtlf.wire.adrf import DataNotification, DataSubscription, NadrfDataStoreRecord
from py_mtlf.wire.ml_model_training import (
    FlTopologyReport,
    InvalidMessageError,
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


def candidate_preparation_payload() -> dict:
    payload = preparation_payload()
    payload.pop("mLModelInfos")
    payload["suppFeats"] = "4"
    payload["x-flTopology"] = {
        "nfInstanceId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "children": [
            {
                "nfInstanceId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "priority": 100,
            }
        ],
        "policy": {
            "selectionMethod": "priority",
            "minAvailableNodes": 1,
            "minTrainNodes": 1,
        },
    }
    return payload


def protocol_leaf_preparation_payload() -> dict:
    payload = candidate_preparation_payload()
    payload["mlCorreId"] = "99999999-9999-4999-8999-999999999999"
    payload["mLEventSubscs"] = [
        {
            "mLEvent": "X_IMAGE_CLASSIFICATION",
            "mLEventFilter": {},
            "modelInterInfo": "pymtlf-image-classification-mnist",
        }
    ]
    payload["x-flTopology"] = {
        "nfInstanceId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "strategy": {
            "method": "fedProx",
            "aggregation": "sampleWeighted",
            "methodParameters": {"proximalMu": 0.01},
        },
        "reportAfter": {"count": 2, "unit": "epoch"},
    }
    return payload


def protocol_branch_preparation_payload() -> dict:
    payload = protocol_leaf_preparation_payload()
    payload["x-flTopology"] = {
        "nfInstanceId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "policy": {
            "allowAdditionalCandidates": False,
            "additionalCandidatePriority": 0,
            "selectionMethod": "priority",
            "minAvailableNodes": 1,
            "fractionTrain": 1.0,
            "minTrainNodes": 1,
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
                "nfInstanceId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "priority": 100,
            }
        ],
    }
    return payload


def round_training_dataset(sample_count: int = 10) -> TrainingDataset:
    observation_count = sample_count + 2
    observations = np.arange(observation_count * len(FEATURE_ORDER), dtype=float).reshape(
        observation_count,
        len(FEATURE_ORDER),
    )
    scope = ScopeTrainingData(
        scope_key="scope-a",
        observation_count=observation_count,
        observation_timestamps=tuple(
            datetime(2026, 8, 26, tzinfo=UTC) + timedelta(seconds=index)
            for index in range(observation_count)
        ),
        observations=observations,
        training_inputs=np.zeros((sample_count, 1, len(FEATURE_ORDER))),
        training_targets=np.zeros((sample_count, 2)),
        validation_inputs=np.zeros((1, 1, len(FEATURE_ORDER))),
        validation_targets=np.zeros((1, 2)),
        training_observations=observations,
        training_eligible=True,
        evaluation_eligible=True,
        exclusion_reason="",
    )
    return TrainingDataset(
        feature_order=FEATURE_ORDER,
        output_fields=("ul_vol", "dl_vol"),
        output_indices=(1, 2),
        seq_length=1,
        out_seq_len=1,
        triggering_scope_key="scope-a",
        scopes=(scope,),
    )


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
        "artifact_role": "ROUND_INPUT",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2, "output_size": 1},
        "inference": {"feature_order": ["uplink", "downlink"]},
    }
    manifest["fl_metadata"] = {
        "ml_corre_id": "fl-process-001",
        "round_ind": 2,
        "client_training": {"epochs": epochs},
    }
    return Mock(manifest=manifest, model=model)


def image_round_input_bundle():
    return Mock(
        workload_profile=WorkloadProfile.IMAGE_CLASSIFICATION,
        manifest={
            "artifact_role": "ROUND_INPUT",
            "analytics_event": "X_IMAGE_CLASSIFICATION",
            "model_interoperability": "pymtlf-image-classification-mnist",
            "workload_profile": "image_classification",
            "dataset": "mnist",
            "runtime_compatibility": {"framework": "torch"},
            "model": {"input_channels": 1, "num_classes": 10},
            "inference": {
                "input_shape": [1, 28, 28],
                "class_count": 10,
                "normalization": "uint8_to_float32_div_255",
            },
            "fl_metadata": {
                "ml_corre_id": "99999999-9999-4999-8999-999999999999",
                "round_ind": 1,
                "client_training": {"epochs": 1},
            },
        },
    )


def round_global_bundle():
    model = torch.nn.Linear(2, 1)
    manifest = {
        "artifact_role": "ROUND_GLOBAL",
        "analytics_event": "UE_COMMUNICATION",
        "model_interoperability": "001122",
        "runtime_compatibility": {"framework": "torch"},
        "model": {"input_size": 2, "output_size": 1},
        "inference": {"feature_order": ["uplink", "downlink"]},
    }
    manifest["fl_metadata"] = {
        "ml_corre_id": "fl-process-001",
        "round_ind": 1,
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


def test_candidate_create_rejects_retained_instruction_before_resource_creation(
    tmp_path,
):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
    )
    try:
        payload = candidate_preparation_payload()
        payload["x-retainedResultReq"] = True
        payload["x-flTopology"]["children"][0]["retainedResultReq"] = True
        with pytest.raises(RequirementsError) as captured:
            service.create(NwdafMLModelTrainSubsc.model_validate(payload))

        assert [item.parameter for item in captured.value.violations] == [
            "x-retainedResultReq",
            "x-flTopology",
        ]
        assert service._resources == {}
        assert service._capacity.acquire(blocking=False)
        assert service._outbox_capacity.acquire(blocking=False)
        service._capacity.release()
        service._outbox_capacity.release()
    finally:
        service.close()


def test_protocol_leaf_preparation_reports_ready_without_model_or_dataset_read(tmp_path):
    delivered = []

    def handler(request: httpx.Request) -> httpx.Response:
        delivered.append(json.loads(request.content))
        return httpx.Response(204, request=request)

    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("X_IMAGE_CLASSIFICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    datasets = Mock()
    workspace = Mock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch = Mock()
    settings = FLClientSettings(
        workload={"profile": "image_classification"},
        training_data={
            "collection_trigger": "local",
            "dataset": "mnist",
            "shard_path": str(tmp_path / "leaf.npz"),
        },
        model_interoperability_ids=("pymtlf-image-classification-mnist",),
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        settings,
        NotificationSettings(),
        context,
        datasets,
        workspace,
        client=client,
        branch_coordinator=branch,
        round_model_distribution=Mock(),
    )
    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(
                protocol_leaf_preparation_payload()
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            resource = service.get(resource.subscription_id)
            if resource.state is FLClientState.PREPARED:
                break
            time.sleep(0.01)

        assert resource.state is FLClientState.PREPARED
        assert resource.representation.supported_features == "4"
        assert resource.protocol_image_dataset.value == "mnist"
        assert delivered == [
            {
                "mlCorreId": "99999999-9999-4999-8999-999999999999",
                "notifCorreId": "prep-client-a",
                "x-flTopologyReport": {
                    "nfInstanceId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "strategy": {
                        "method": "fedProx",
                        "aggregation": "sampleWeighted",
                        "methodParameters": {"proximalMu": 0.01},
                    },
                    "reportAfter": {"count": 2, "unit": "epoch"},
                },
            }
        ]
        workspace.assert_not_called()
        assert not workspace.method_calls
        datasets.assert_not_called()
        assert not datasets.method_calls
        branch.prepare_protocol.assert_not_called()
    finally:
        service.close()
        client.close()


def test_protocol_preparation_with_model_reference_refuses_hierarchy_feature(
    tmp_path,
):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("X_IMAGE_CLASSIFICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    workspace = Mock()
    branch = Mock()
    settings = FLClientSettings(
        workload={"profile": "image_classification"},
        training_data={
            "collection_trigger": "local",
            "dataset": "mnist",
            "shard_path": str(tmp_path / "leaf.npz"),
        },
        model_interoperability_ids=("pymtlf-image-classification-mnist",),
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        settings,
        NotificationSettings(),
        context,
        Mock(),
        workspace,
        branch_coordinator=branch,
        round_model_distribution=Mock(),
    )
    payload = protocol_leaf_preparation_payload()
    payload["mLModelInfos"] = [
        {
            "event": "X_IMAGE_CLASSIFICATION",
            "mLFileAddr": {"mLModelUrl": "http://legacy.example/assignment.tar.gz"},
        }
    ]
    try:
        resource = service.create(NwdafMLModelTrainSubsc.model_validate(payload))

        assert resource.state is FLClientState.READY
        assert resource.representation.supported_features == ""
        assert workspace.method_calls == []
        branch.prepare_protocol.assert_not_called()
    finally:
        service.close()


def test_protocol_intermediate_accepts_image_round_without_local_training_config(
    tmp_path,
):
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    try:
        service._validate_protocol_intermediate_bundle(
            image_round_input_bundle(),
            "X_IMAGE_CLASSIFICATION",
            "pymtlf-image-classification-mnist",
        )
    finally:
        service.close()


@pytest.mark.parametrize(
    ("model_interoperability", "local_dataset"),
    (
        ("pymtlf-image-classification-unknown", "mnist"),
        ("pymtlf-image-classification-mnist", "cifar10"),
    ),
    ids=("unknown-contract", "incompatible-local-dataset"),
)
def test_protocol_leaf_refuses_feature_for_unsupported_image_contract(
    tmp_path,
    model_interoperability,
    local_dataset,
):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id=(
            "11111111-1111-4111-8111-111111111111"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("X_IMAGE_CLASSIFICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    settings = FLClientSettings(
        workload={"profile": "image_classification"},
        training_data={
            "collection_trigger": "local",
            "dataset": local_dataset,
            "shard_path": str(tmp_path / "leaf.npz"),
        },
        model_interoperability_ids=(model_interoperability,),
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        settings,
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
        round_model_distribution=Mock(),
    )
    try:
        payload = protocol_leaf_preparation_payload()
        payload["mLEventSubscs"][0]["modelInterInfo"] = model_interoperability
        resource = service.create(NwdafMLModelTrainSubsc.model_validate(payload))

        assert resource.representation.supported_features == ""
        assert resource.state is FLClientState.READY
        assert resource.protocol_image_dataset is None
    finally:
        service.close()


def test_protocol_leaf_rejects_first_round_bundle_with_incompatible_dataset(
    tmp_path,
):
    settings = FLClientSettings(
        workload={"profile": "image_classification"},
        training_data={
            "collection_trigger": "local",
            "dataset": "mnist",
            "shard_path": str(tmp_path / "leaf.npz"),
        },
        model_interoperability_ids=("pymtlf-image-classification-mnist",),
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        settings,
        NotificationSettings(),
        Mock(),
        Mock(),
        Mock(),
    )
    bundle = image_round_input_bundle()
    bundle.manifest["dataset"] = "cifar10"
    bundle.manifest["model"]["input_channels"] = 3
    bundle.manifest["inference"]["input_shape"] = [3, 32, 32]
    try:
        with pytest.raises(RuntimeError, match="image dataset is incompatible"):
            service._validate_workload_bundle(
                bundle,
                "X_IMAGE_CLASSIFICATION",
                "pymtlf-image-classification-mnist",
            )
    finally:
        service.close()


def test_protocol_topology_patch_reconfigures_branch_without_training_or_model_read(
    tmp_path,
):
    delivered = []

    def handler(request: httpx.Request) -> httpx.Response:
        delivered.append(json.loads(request.content))
        return httpx.Response(204, request=request)

    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id=(
            "11111111-1111-4111-8111-111111111111"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("X_IMAGE_CLASSIFICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    datasets = Mock()
    workspace = Mock()
    branch = Mock()
    branch.prepare_protocol.side_effect = [
        FlTopologyReport(
            nfInstanceId="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        ),
        FlTopologyReport(
            nfInstanceId="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        ),
    ]
    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = FLClientEngine(
        fl_settings(tmp_path),
        FLClientSettings(
            workload={"profile": "image_classification"},
            training_data={
                "collection_trigger": "local",
                "dataset": "mnist",
                "shard_path": str(tmp_path / "branch.npz"),
            },
            model_interoperability_ids=(
                "pymtlf-image-classification-mnist",
            ),
        ),
        NotificationSettings(),
        context,
        datasets,
        workspace,
        client=client,
        branch_coordinator=branch,
        round_model_distribution=Mock(),
    )
    try:
        created = service.create(
            NwdafMLModelTrainSubsc.model_validate(
                protocol_branch_preparation_payload()
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if service.get(created.subscription_id).state is FLClientState.PREPARED:
                break
            time.sleep(0.01)

        topology = protocol_branch_preparation_payload()["x-flTopology"]
        topology["children"][0]["enabled"] = False
        updated = service.patch(
            created.subscription_id,
            NwdafMLModelTrainSubscPatch.model_validate(
                {"x-flTopology": topology}
            ),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            updated = service.get(updated.subscription_id)
            if updated.state is FLClientState.PREPARED:
                break
            time.sleep(0.01)

        assert updated.state is FLClientState.PREPARED
        assert branch.prepare_protocol.call_count == 2
        assert len(delivered) == 2
        datasets.assert_not_called()
        assert not datasets.method_calls
        workspace.assert_not_called()
        assert not workspace.method_calls
    finally:
        service.close()
        client.close()


def test_candidate_create_validates_containing_nwdaf_identity(tmp_path):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        containing_nwdaf_process_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
    )
    try:
        with pytest.raises(InvalidMessageError) as captured:
            service.create(
                NwdafMLModelTrainSubsc.model_validate(candidate_preparation_payload())
            )
        assert captured.value.violations[0].parameter == "x-flTopology.nfInstanceId"
    finally:
        service.close()


def test_candidate_resource_preserves_standard_patch_and_rejects_candidate_mutation(
    tmp_path,
):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id=(
            "11111111-1111-4111-8111-111111111111"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
    )
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
    )
    try:
        created = service.create(
            NwdafMLModelTrainSubsc.model_validate(candidate_preparation_payload())
        )
        updated = service.patch(
            created.subscription_id,
            NwdafMLModelTrainSubscPatch.model_validate(
                {
                    "eventReq": {
                        "notifMethod": "ON_EVENT_DETECTION",
                        "maxReportNbr": 1,
                    }
                }
            ),
        )
        assert updated.state is FLClientState.READY
        assert updated.revision == created.revision + 1
        assert updated.representation.fl_topology is not None
        assert updated.representation.event_request.maximum_report_count == 1

        before_rejected_patch = service.get(created.subscription_id)
        with pytest.raises(RequirementsError) as captured:
            service.patch(
                created.subscription_id,
                NwdafMLModelTrainSubscPatch.model_validate(
                    {"x-flTopology": {"policy": {"minTrainNodes": 1}}}
                ),
            )
        assert captured.value.violations[0].parameter == "suppFeats"
        after_rejected_patch = service.get(created.subscription_id)
        assert after_rejected_patch.revision == before_rejected_patch.revision
        assert (
            after_rejected_patch.representation
            == before_rejected_patch.representation
        )

        replacement_payload = candidate_preparation_payload()
        replacement_payload["x-flTopology"]["children"][0]["priority"] = 80
        with pytest.raises(RequirementsError):
            service.replace(
                created.subscription_id,
                NwdafMLModelTrainSubsc.model_validate(replacement_payload),
            )
        after_rejected_put = service.get(created.subscription_id)
        assert after_rejected_put.revision == before_rejected_patch.revision
        assert after_rejected_put.representation == before_rejected_patch.representation
        assert service._capacity.acquire(blocking=False)
        assert service._outbox_capacity.acquire(blocking=False)
        service._capacity.release()
        service._outbox_capacity.release()
    finally:
        service.close()


def test_candidate_resource_delete_and_generation_reset_remove_contract(tmp_path):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id=(
            "11111111-1111-4111-8111-111111111111"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
    )
    registry = FLExperimentRegistry()
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
        experiments=registry,
    )
    try:
        first = service.create(
            NwdafMLModelTrainSubsc.model_validate(candidate_preparation_payload())
        )
        service.delete(first.subscription_id)
        with pytest.raises(KeyError):
            service.get(first.subscription_id)
        assert registry.active() is None

        second_payload = candidate_preparation_payload()
        second_payload["notifCorreId"] = "prep-client-b"
        second = service.create(
            NwdafMLModelTrainSubsc.model_validate(second_payload)
        )
        service.abort_generation("containing NWDAF process generation changed")
        with pytest.raises(KeyError):
            service.get(second.subscription_id)
        registry.reset_generation()
        assert service._capacity.acquire(blocking=False)
        assert service._outbox_capacity.acquire(blocking=False)
        service._capacity.release()
        service._outbox_capacity.release()
    finally:
        registry.reset_generation()
        service.close()


def test_unnegotiated_candidate_put_is_gated_before_context_lookup(tmp_path):
    context = Mock()
    context.get.side_effect = [
        NwdafContext(
            nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            containing_nwdaf_process_instance_id=(
                "11111111-1111-4111-8111-111111111111"
            ),
            api_root="http://nwdaf.example",
            internal_api_root="http://nwdaf-internal.example",
        ),
        RuntimeError("containing NWDAF context is unavailable"),
    ]
    service = FLClientEngine(
        fl_settings(tmp_path),
        client_settings(),
        NotificationSettings(),
        context,
        Mock(),
        Mock(),
    )
    try:
        resource = service.create(
            NwdafMLModelTrainSubsc.model_validate(candidate_preparation_payload())
        )
        with pytest.raises(RequirementsError):
            service.replace(
                resource.subscription_id,
                NwdafMLModelTrainSubsc.model_validate(
                    candidate_preparation_payload()
                ),
            )
        assert context.get.call_count == 1
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
        )

        updated = service.get(resource.subscription_id)
        assert updated.state is FLClientState.FAILED
        assert "minNumSamples" in updated.last_error
        assert "prepared training dataset does not meet minNumSamples" in caplog.text
        service._enqueue_delivery.assert_called_once()
    finally:
        service.close()


def test_preparation_success_returns_validated_input_model_url(tmp_path, caplog):
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
        )

        notification = service._enqueue_delivery.call_args.args[1]
        assert notification.round_indicator is None
        assert notification.status_report is None
        assert len(notification.ml_model_infos or ()) == 1
        address = notification.ml_model_infos[0].model_file_address
        assert address is not None
        assert str(address.model_url) == "http://server.example/base.tar.gz"
        assert "state=PREPARED records=1 samples=1" in caplog.text
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
    artifact = ArtifactMetadata(
        key="a" * 64,
        size_bytes=artifact_path.stat().st_size,
        path=artifact_path,
        url="http://server.example/base.tar.gz",
    )
    workspace.download_archive.return_value = DownloadedArchive(
        metadata=artifact,
        validated=ValidatedArchive(manifest={}, contract=None),
    )
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


def test_final_validation_uses_configured_training_device(
    tmp_path,
    monkeypatch,
):
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
        preparation_base_artifact=Mock(),
    )
    service._resources[resource.subscription_id] = resource
    candidate = round_global_bundle()
    base = candidate
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
    service._loader.load.side_effect = (base, base, base, candidate)
    dataset = round_training_dataset(10)
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
        side_effect=(np.asarray([[1.0, 3.0]]), np.asarray([[2.0, 3.0]]))
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
    dataset = round_training_dataset()
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
        preparation_base_artifact=Mock(),
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
