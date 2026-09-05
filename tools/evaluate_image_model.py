import argparse
import json
from pathlib import Path

from py_mtlf.config import ImageClassificationWorkloadSettings, load_settings
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository
from py_mtlf.core.image_classification import (
    ImageClassificationEvaluator,
    ImageDatasetLoader,
)
from py_mtlf.core.trainer import TrustedBundleLoader, resolve_device
from py_mtlf.core.workloads import validate_image_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an image-classification artifact on a held-out local dataset"
    )
    parser.add_argument("--config", required=True, type=Path)
    artifact_source = parser.add_mutually_exclusive_group(required=True)
    artifact_source.add_argument("--artifact-key")
    artifact_source.add_argument("--artifact-path", type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    settings = load_settings(args.config)
    client = settings.federated_learning.client
    if client is None or not isinstance(client.workload, ImageClassificationWorkloadSettings):
        raise ValueError("evaluation config must select the image_classification workload")
    if args.artifact_key is not None:
        repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
        artifact = repository.metadata(args.artifact_key)
    else:
        path = args.artifact_path.resolve()
        if not path.is_file():
            raise ValueError("evaluation artifact path was not found")
        artifact = ArtifactMetadata(
            key=path.name.removesuffix(".tar.gz"),
            size_bytes=path.stat().st_size,
            path=path,
            url=path.as_uri(),
        )
    bundle = TrustedBundleLoader().load(artifact)
    contract = validate_image_manifest(bundle.manifest)
    dataset = ImageDatasetLoader().load(args.test_data, contract.name)
    result = ImageClassificationEvaluator().evaluate(
        bundle.model,
        dataset,
        run_id=args.run_id,
        model_artifact_key=artifact.key,
        test_dataset_path=args.test_data,
        device=resolve_device(client.training.device),
        batch_size=client.training.batch_size,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model_artifact_key": result.model_artifact_key,
                "test_dataset_path": result.test_dataset_path,
                "dataset": result.dataset.value,
                "test_sample_count": result.test_sample_count,
                "correct_count": result.correct_count,
                "accuracy": result.accuracy,
                "evaluated_at": result.evaluated_at.isoformat(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
