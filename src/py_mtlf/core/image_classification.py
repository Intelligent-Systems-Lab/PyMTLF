from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from py_mtlf.core.workloads import ImageDatasetName, image_dataset_contract


class ImageDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageClassificationDataset:
    dataset: ImageDatasetName
    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class ImageEvaluationResult:
    run_id: str
    model_artifact_key: str
    test_dataset_path: str
    dataset: ImageDatasetName
    test_sample_count: int
    correct_count: int
    accuracy: float
    evaluated_at: datetime


class ImageDatasetLoader:
    def load(
        self,
        path: str | Path,
        dataset: str | ImageDatasetName,
    ) -> ImageClassificationDataset:
        source = Path(path)
        contract = image_dataset_contract(dataset)
        if not source.is_file():
            raise ImageDatasetError("image dataset shard was not found")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != {"images", "labels"}:
                    raise ImageDatasetError(
                        "image dataset shard must contain exactly images and labels"
                    )
                images = archive["images"]
                labels = archive["labels"]
        except ImageDatasetError:
            raise
        except (OSError, ValueError) as error:
            raise ImageDatasetError("image dataset shard cannot be read") from error
        expected_image_shape = contract.input_shape
        if images.dtype != np.uint8 or images.ndim != 4:
            raise ImageDatasetError("image dataset images must be rank-4 uint8 values")
        if tuple(images.shape[1:]) != expected_image_shape:
            raise ImageDatasetError("image dataset shape does not match configured dataset")
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ImageDatasetError("image dataset labels must be rank-1 integer values")
        if images.shape[0] != labels.shape[0] or images.shape[0] == 0:
            raise ImageDatasetError("image dataset images and labels must have equal non-zero size")
        if np.any(labels < 0) or np.any(labels >= contract.class_count):
            raise ImageDatasetError("image dataset labels are outside the supported class range")
        inputs = torch.from_numpy(images.copy()).to(dtype=torch.float32).div_(255.0)
        targets = torch.from_numpy(labels.astype(np.int64, copy=True))
        if not torch.isfinite(inputs).all():
            raise ImageDatasetError("image dataset tensors are not finite")
        return ImageClassificationDataset(
            dataset=contract.name,
            inputs=inputs,
            targets=targets,
        )


class ImageClassificationEvaluator:
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: ImageClassificationDataset,
        *,
        run_id: str,
        model_artifact_key: str,
        test_dataset_path: str | Path,
        device: torch.device,
        batch_size: int,
    ) -> ImageEvaluationResult:
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        if not model_artifact_key.strip():
            raise ValueError("model_artifact_key must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model.to(device)
        model.eval()
        contract = image_dataset_contract(dataset.dataset)
        correct = 0
        try:
            with torch.no_grad():
                for offset in range(0, dataset.sample_count, batch_size):
                    inputs = dataset.inputs[offset : offset + batch_size].to(device)
                    targets = dataset.targets[offset : offset + batch_size].to(device)
                    logits = model(inputs)
                    if logits.shape != (targets.shape[0], contract.class_count):
                        raise ImageDatasetError(
                            "image classifier output shape is incompatible"
                        )
                    correct += int((logits.argmax(dim=1) == targets).sum().item())
        finally:
            model.to("cpu")
        accuracy = correct / dataset.sample_count
        if not math.isfinite(accuracy):
            raise ImageDatasetError("image classifier accuracy is not finite")
        return ImageEvaluationResult(
            run_id=run_id,
            model_artifact_key=model_artifact_key,
            test_dataset_path=str(Path(test_dataset_path)),
            dataset=dataset.dataset,
            test_sample_count=dataset.sample_count,
            correct_count=correct,
            accuracy=accuracy,
            evaluated_at=datetime.now(UTC),
        )
