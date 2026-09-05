import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from py_mtlf.config import (
    ArtifactSettings,
    FederatedLearningSettings,
    FittingSettings,
    FLClientSettings,
    FLServerSettings,
    NotificationSettings,
)
from py_mtlf.core.accuracy_policy import ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.fl_artifacts import RoundLocalArtifact, validate_fl_artifact_manifest
from py_mtlf.core.fl_client import FLClientEngine, FLClientResource, FLClientState
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLParticipant,
    FLProcess,
    FLServerEngine,
)
from py_mtlf.core.fl_workspace import FLWorkspace
from py_mtlf.core.image_classification import (
    ImageClassificationDataset,
    ImageClassificationEvaluator,
    ImageDatasetError,
    ImageDatasetLoader,
)
from py_mtlf.core.seed_import import build_seed_bundle
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.core.workloads import ImageDatasetName
from py_mtlf.wire.ml_model_training import (
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
)
from py_mtlf.wire.private import SelectedTarget


def write_shard(
    path: Path,
    *,
    channels: int,
    size: int,
    sample_count: int,
    seed: int,
) -> None:
    generator = np.random.default_rng(seed)
    images = generator.integers(
        0,
        256,
        size=(sample_count, channels, size, size),
        dtype=np.uint8,
    )
    labels = np.arange(sample_count, dtype=np.int64) % 10
    np.savez(path, images=images, labels=labels)


def load_seed_bundle(tmp_path: Path, dataset: str, model_id: int = 1) -> LoadedBundle:
    source = Path("seed_models/image_classification") / dataset
    bundle_path = tmp_path / f"{dataset}.tar.gz"
    build_seed_bundle(
        source,
        bundle_path,
        model_id=model_id,
        event=None,
        model_interoperability="image-classification-pytorch",
    )
    repository = ArtifactRepository(tmp_path / f"{dataset}-artifacts", ArtifactSettings())
    repository.open()
    return TrustedBundleLoader().load(repository.publish(bundle_path))


def image_round_request(model_url: str) -> NwdafMLModelTrainSubsc:
    return NwdafMLModelTrainSubsc.model_validate(
        {
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
                    "modelInterInfo": "image-classification-pytorch",
                }
            ],
            "notifUri": "http://go.internal/training/callback",
            "notifCorreId": "image-round-client-a",
            "mlCorreId": "image-process-001",
            "mLPreFlag": False,
            "roundInd": 0,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": model_url},
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
    )


@pytest.mark.parametrize(
    ("dataset", "channels", "size"),
    [("mnist", 1, 28), ("cifar10", 3, 32)],
)
def test_local_image_loader_applies_exact_tensor_contract(
    tmp_path,
    dataset,
    channels,
    size,
):
    path = tmp_path / "train.npz"
    images = np.zeros((2, channels, size, size), dtype=np.uint8)
    images[1].fill(255)
    np.savez(path, images=images, labels=np.asarray([0, 9], dtype=np.int16))

    loaded = ImageDatasetLoader().load(path, dataset)

    assert loaded.inputs.dtype is torch.float32
    assert loaded.targets.dtype is torch.int64
    assert loaded.inputs.shape == (2, channels, size, size)
    assert loaded.inputs[0].max().item() == 0.0
    assert loaded.inputs[1].min().item() == 1.0


def test_local_image_loader_rejects_wrong_shape_and_extra_arrays(tmp_path):
    with pytest.raises(ImageDatasetError, match="not found"):
        ImageDatasetLoader().load(tmp_path / "missing.npz", "mnist")

    wrong_shape = tmp_path / "wrong-shape.npz"
    np.savez(
        wrong_shape,
        images=np.zeros((2, 3, 28, 28), dtype=np.uint8),
        labels=np.asarray([0, 1]),
    )
    with pytest.raises(ImageDatasetError, match="shape"):
        ImageDatasetLoader().load(wrong_shape, "mnist")

    extra = tmp_path / "extra.npz"
    np.savez(
        extra,
        images=np.zeros((2, 1, 28, 28), dtype=np.uint8),
        labels=np.asarray([0, 1]),
        manifest=np.asarray([1]),
    )
    with pytest.raises(ImageDatasetError, match="exactly"):
        ImageDatasetLoader().load(extra, "mnist")


@pytest.mark.parametrize(
    ("dataset", "channels", "size"),
    [("mnist", 1, 28), ("cifar10", 3, 32)],
)
def test_known_image_workloads_execute_real_optimizer_step(
    tmp_path,
    dataset,
    channels,
    size,
):
    base = load_seed_bundle(tmp_path, dataset)
    path = tmp_path / f"{dataset}-train.npz"
    write_shard(path, channels=channels, size=size, sample_count=4, seed=31)
    before = {
        name: value.detach().clone() for name, value in base.model.state_dict().items()
    }

    result = FederatedTrainer(
        FittingSettings(batch_size=2, learning_rate=0.001, random_seed=7)
    ).train(base, ImageDatasetLoader().load(path, dataset), epochs=1)

    assert result.training_sample_count == 4
    assert np.isfinite(result.final_loss)
    assert any(
        not torch.equal(before[name], value)
        for name, value in result.model.state_dict().items()
    )


def test_zero_fedprox_mu_matches_the_unmodified_image_objective(tmp_path):
    base = load_seed_bundle(tmp_path, "mnist")
    path = tmp_path / "train.npz"
    write_shard(path, channels=1, size=28, sample_count=4, seed=32)
    dataset = ImageDatasetLoader().load(path, "mnist")
    trainer = FederatedTrainer(
        FittingSettings(batch_size=2, learning_rate=0.001, random_seed=7)
    )

    without_penalty = trainer.train(base, dataset, epochs=1)
    zero_penalty = trainer.train(base, dataset, epochs=1, proximal_mu=0.0)

    assert zero_penalty.final_loss == pytest.approx(without_penalty.final_loss)
    for name, value in without_penalty.model.state_dict().items():
        assert torch.equal(value, zero_penalty.model.state_dict()[name])


def test_controlled_seed_bundles_share_architecture_without_scaler_or_batchnorm(tmp_path):
    mnist = load_seed_bundle(tmp_path, "mnist", 1)
    cifar = load_seed_bundle(tmp_path, "cifar10", 2)

    assert mnist.scaler is None and mnist.scaler_source == b""
    assert cifar.scaler is None and cifar.scaler_source == b""
    assert mnist.model_source == cifar.model_source
    assert not any(isinstance(item, torch.nn.BatchNorm2d) for item in mnist.model.modules())
    assert mnist.model.features[0].in_channels == 1
    assert cifar.model.features[0].in_channels == 3
    assert mnist.model.classifier.out_features == cifar.model.classifier.out_features == 10
    assert not torch.equal(
        mnist.model.classifier.weight,
        cifar.model.classifier.weight,
    )
    for bundle in (mnist, cifar):
        torch.manual_seed(bundle.manifest["initialization_seed"])
        expected = type(bundle.model)(**bundle.manifest["model"])
        for name, value in bundle.model.state_dict().items():
            assert torch.equal(value, expected.state_dict()[name])


def test_two_client_image_training_aggregation_and_held_out_evaluation(tmp_path):
    base = load_seed_bundle(tmp_path, "mnist")
    first_path = tmp_path / "client-a.npz"
    second_path = tmp_path / "client-b.npz"
    held_out_path = tmp_path / "held-out.npz"
    write_shard(first_path, channels=1, size=28, sample_count=4, seed=11)
    write_shard(second_path, channels=1, size=28, sample_count=6, seed=12)
    write_shard(held_out_path, channels=1, size=28, sample_count=5, seed=13)
    loader = ImageDatasetLoader()
    first_data = loader.load(first_path, "mnist")
    second_data = loader.load(second_path, "mnist")
    trainer = FederatedTrainer(
        FittingSettings(batch_size=2, learning_rate=0.001, random_seed=7)
    )

    first = trainer.train(base, first_data, epochs=1, proximal_mu=0.0)
    second = trainer.train(base, second_data, epochs=1, proximal_mu=0.01)
    fl_settings = FederatedLearningSettings(
        workspace_root=tmp_path / "workspace",
        public_base_url="http://root.example",
    )
    workspace = FLWorkspace(fl_settings, ArtifactSettings())
    workspace.open()
    round_input = workspace.publish_round_input(
        process_id="image-process-001",
        server_nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        round_indicator=0,
        base=base,
        epochs=1,
    )
    participant_ids = (
        "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    )
    value = image_round_request(round_input.url)
    training_scope = TrainingScopeDescriptor.from_training_request(value, 0)
    local_results = []
    for participant_id, result, dataset in zip(
        participant_ids,
        (first, second),
        (ImageDatasetName.MNIST, ImageDatasetName.MNIST),
        strict=True,
    ):
        local_results.append(
            workspace.publish(
                process_id="image-process-001",
                participant_id=participant_id,
                round_indicator=0,
                role="ROUND_LOCAL",
                base=base,
                model=result.model,
                metadata={
                    "artifact_role": "ROUND_LOCAL",
                    "result_type": "TRAINING",
                    "fl_metadata": {
                        "ml_corre_id": "image-process-001",
                        "round_ind": 0,
                        "participant_nf_instance_id": participant_id,
                        "training_scope": training_scope.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "training_sample_count": result.training_sample_count,
                        "dataset_evidence": {
                            "workload_profile": "image_classification",
                            "dataset": dataset.value,
                            "training_sample_count": result.training_sample_count,
                        },
                    },
                },
            )
        )
    local_by_url = {
        item.url: ArtifactMetadata(
            key=item.digest,
            size_bytes=item.path.stat().st_size,
            path=item.path,
            url=item.url,
        )
        for item in local_results
    }
    server_workspace = Mock(wraps=workspace)
    server_workspace.download.side_effect = (
        lambda url, *_args, **_kwargs: local_by_url[url]
    )
    server_context = Mock()
    server_context.get.return_value.nf_instance_id = (
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    )
    participants = []
    for index, (participant_id, local) in enumerate(
        zip(participant_ids, local_results, strict=True),
        start=1,
    ):
        participants.append(
            FLParticipant(
                scope=ScopeReference(
                    scope_key=f"scope-{index}",
                    consumer_id=participant_id,
                    model_ids=(1,),
                    ml_event="UE_COMMUNICATION",
                    ml_event_filter=dict(
                        value.ml_event_subscriptions[0].ml_event_filter
                    ),
                    target_ue={"intGroupIds": ["group-G"]},
                ),
                candidate=FLClientCandidate(
                    target=SelectedTarget(
                        nfInstanceId=participant_id,
                        nfServiceInstanceId=f"training-{index}",
                        serviceName="nnwdaf-mlmodeltraining",
                        apiRoot=f"http://client-{index}.example",
                        selectionSource="STATIC",
                    ),
                    tracking_areas=(f"466-92-00000{index}",),
                ),
                notification_correlation_id=f"image-round-{index}",
                expected_training_scope=training_scope,
                notification=NwdafMLModelTrainNotif.model_validate(
                    {
                        "notifCorreId": f"image-round-{index}",
                        "mlCorreId": "image-process-001",
                        "roundInd": 0,
                        "mLModelInfos": [
                            {
                                "event": "UE_COMMUNICATION",
                                "mLFileAddr": {"mLModelUrl": local.url},
                            }
                        ],
                    }
                ),
            )
        )
    server = FLServerEngine(
        fl_settings,
        FLServerSettings(),
        server_context,
        Mock(),
        Mock(),
        server_workspace,
        Mock(),
        client=Mock(),
    )
    try:
        aggregated = server._aggregate_round(
            FLProcess(
                process_id="image-process-001",
                intent=None,
                participants=participants,
            ),
            round_input.url,
            0,
            round_input_artifact=round_input,
        )
        aggregate = TrustedBundleLoader().load(
            ArtifactMetadata(
                key=aggregated.digest,
                size_bytes=aggregated.path.stat().st_size,
                path=aggregated.path,
                url=aggregated.url,
            )
        ).model
    finally:
        server.close()
        workspace.close()
    evaluation = ImageClassificationEvaluator().evaluate(
        aggregate,
        loader.load(held_out_path, "mnist"),
        run_id="controlled-smoke",
        model_artifact_key="model-key",
        test_dataset_path=held_out_path,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert first.training_sample_count == 4
    assert second.training_sample_count == 6
    assert evaluation.test_sample_count == 5
    assert 0 <= evaluation.accuracy <= 1


def test_fl_client_round_uses_local_image_shard_and_publishes_real_result(tmp_path):
    shard_path = tmp_path / "client-a.npz"
    write_shard(shard_path, channels=1, size=28, sample_count=4, seed=21)
    source = Path("seed_models/image_classification/mnist")
    seed_path = tmp_path / "mnist-seed.tar.gz"
    build_seed_bundle(
        source,
        seed_path,
        model_id=1,
        event=None,
        model_interoperability="image-classification-pytorch",
    )
    repository = ArtifactRepository(tmp_path / "artifacts", ArtifactSettings())
    repository.open()
    preparation_artifact = repository.publish(seed_path)
    base = TrustedBundleLoader().load(preparation_artifact)

    fl_settings = FederatedLearningSettings(
        workspace_root=tmp_path / "workspace",
        public_base_url="http://client.example",
    )
    workspace = FLWorkspace(fl_settings, ArtifactSettings())
    workspace.open()
    round_input = workspace.publish_round_input(
        process_id="image-process-001",
        server_nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        round_indicator=0,
        base=base,
        epochs=1,
    )
    round_metadata = ArtifactMetadata(
        key=round_input.digest,
        size_bytes=round_input.path.stat().st_size,
        path=round_input.path,
        url=round_input.url,
    )
    published_results = []
    transport = Mock(wraps=workspace)
    transport.download.return_value = round_metadata

    def publish_result(**kwargs):
        result = workspace.publish(**kwargs)
        published_results.append(result)
        return result

    transport.publish.side_effect = publish_result
    client_settings = FLClientSettings.model_validate(
        {
            "workload": {"profile": "image_classification"},
            "training_data": {
                "collection_trigger": "local",
                "dataset": "mnist",
                "shard_path": str(shard_path),
            },
            "model_interoperability_ids": ["image-classification-pytorch"],
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "learning_rate": 0.001,
                "random_seed": 7,
            },
        }
    )
    context = Mock()
    context.get.return_value.nf_instance_id = (
        "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )
    service = FLClientEngine(
        fl_settings,
        client_settings,
        NotificationSettings(),
        context,
        Mock(),
        transport,
    )
    service._enqueue_delivery = Mock()
    value = image_round_request(round_input.url)
    resource = FLClientResource(
        subscription_id="resource-1",
        representation=value,
        state=FLClientState.ROUND_RUNNING,
        scope=TrainingScopeDescriptor.from_training_request(value, 0),
        preparation_base_artifact=preparation_artifact,
    )
    service._resources[resource.subscription_id] = resource
    assert service._capacity.acquire(blocking=False)
    try:
        service._run_round(resource.subscription_id, resource.revision)

        assert len(published_results) == 1
        published = published_results[0]
        contract = validate_fl_artifact_manifest(published.manifest)
        assert isinstance(contract, RoundLocalArtifact)
        assert contract.fl_metadata.training_sample_count == 4
        assert contract.fl_metadata.dataset_evidence.workload_profile == (
            "image_classification"
        )
        assert contract.fl_metadata.dataset_evidence.dataset == "mnist"
        trained = TrustedBundleLoader().load(
            ArtifactMetadata(
                key=published.digest,
                size_bytes=published.path.stat().st_size,
                path=published.path,
                url=published.url,
            )
        )
        assert any(
            not torch.equal(trained_value, base_value)
            for trained_value, base_value in zip(
                trained.model.state_dict().values(),
                base.model.state_dict().values(),
                strict=True,
            )
        )
        transport.download.assert_called_once()
        service._enqueue_delivery.assert_called_once()
    finally:
        service.close()
        workspace.close()


def test_image_round_artifact_preserves_profile_without_scaler(tmp_path):
    base = load_seed_bundle(tmp_path, "mnist")
    workspace = FLWorkspace(
        FederatedLearningSettings(
            workspace_root=tmp_path / "workspace",
            public_base_url="http://root.example",
        ),
        ArtifactSettings(),
    )
    workspace.open()
    try:
        published = workspace.publish_round_input(
            process_id="process-1",
            server_nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            round_indicator=0,
            base=base,
            epochs=1,
        )
        loaded = TrustedBundleLoader().load(
            ArtifactMetadata(
                key=published.digest,
                size_bytes=published.path.stat().st_size,
                path=published.path,
                url=published.url,
            )
        )
    finally:
        workspace.close()

    assert loaded.workload_profile.value == "image_classification"
    assert loaded.scaler is None
    assert published.manifest["workload_profile"] == "image_classification"


def test_federated_trainer_rejects_mismatched_image_dataset(tmp_path):
    base = load_seed_bundle(tmp_path, "mnist")
    dataset = ImageClassificationDataset(
        dataset=ImageDatasetName.CIFAR10,
        inputs=torch.zeros((2, 3, 32, 32)),
        targets=torch.zeros(2, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        FederatedTrainer(FittingSettings()).train(base, dataset, epochs=1)


def test_image_evaluator_reports_known_accuracy(tmp_path):
    class AlwaysClassZero(torch.nn.Module):
        def forward(self, value):
            logits = torch.zeros((value.shape[0], 10), dtype=torch.float32)
            logits[:, 0] = 1
            return logits

    dataset = ImageClassificationDataset(
        dataset=ImageDatasetName.MNIST,
        inputs=torch.zeros((4, 1, 28, 28)),
        targets=torch.tensor([0, 1, 0, 2], dtype=torch.int64),
    )

    result = ImageClassificationEvaluator().evaluate(
        AlwaysClassZero(),
        dataset,
        run_id="known-accuracy",
        model_artifact_key="model-key",
        test_dataset_path=tmp_path / "held-out.npz",
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert result.correct_count == 2
    assert result.test_sample_count == 4
    assert result.accuracy == 0.5


def test_offline_evaluator_accepts_a_workspace_artifact_path(tmp_path):
    artifact_path = tmp_path / "final-model.tar.gz"
    test_path = tmp_path / "held-out.npz"
    build_seed_bundle(
        Path("seed_models/image_classification/mnist"),
        artifact_path,
        model_id=1,
        event=None,
        model_interoperability="image-classification-pytorch",
    )
    write_shard(test_path, channels=1, size=28, sample_count=4, seed=33)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/evaluate_image_model.py",
            "--config",
            "config/fl-client-image-classification.yaml",
            "--artifact-path",
            str(artifact_path),
            "--test-data",
            str(test_path),
            "--run-id",
            "offline-evaluator-smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["run_id"] == "offline-evaluator-smoke"
    assert result["dataset"] == "mnist"
    assert result["test_sample_count"] == 4
    assert 0 <= result["accuracy"] <= 1
