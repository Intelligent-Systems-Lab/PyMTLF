from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class WorkloadContractError(ValueError):
    pass


class WorkloadProfile(StrEnum):
    UE_COMMUNICATION_FORECASTING = "ue_communication_forecasting"
    IMAGE_CLASSIFICATION = "image_classification"


class ImageDatasetName(StrEnum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


@dataclass(frozen=True)
class ImageDatasetContract:
    name: ImageDatasetName
    input_channels: int
    height: int
    width: int
    class_count: int = 10

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_channels, self.height, self.width)


IMAGE_DATASET_CONTRACTS = {
    ImageDatasetName.MNIST: ImageDatasetContract(
        name=ImageDatasetName.MNIST,
        input_channels=1,
        height=28,
        width=28,
    ),
    ImageDatasetName.CIFAR10: ImageDatasetContract(
        name=ImageDatasetName.CIFAR10,
        input_channels=3,
        height=32,
        width=32,
    ),
}

TRAFFIC_BUNDLE_FILES = frozenset(
    {"config.json", "model.py", "model.npy", "scaler.pkl"}
)
IMAGE_CLASSIFICATION_BUNDLE_FILES = frozenset(
    {"config.json", "model.py", "model.npy"}
)
IMAGE_NORMALIZATION = "uint8_to_float32_div_255"
IMAGE_CLASSIFICATION_EVENT = "X_IMAGE_CLASSIFICATION"
IMAGE_MODEL_INTEROPERABILITY = {
    ImageDatasetName.MNIST: "pymtlf-image-classification-mnist",
    ImageDatasetName.CIFAR10: "pymtlf-image-classification-cifar10",
}


def workload_profile(manifest: Mapping[str, object]) -> WorkloadProfile:
    value = manifest.get("workload_profile")
    try:
        return WorkloadProfile(value)
    except (TypeError, ValueError) as error:
        raise WorkloadContractError("bundle workload_profile is missing or unsupported") from error


def required_bundle_files(manifest: Mapping[str, object]) -> frozenset[str]:
    profile = workload_profile(manifest)
    if profile is WorkloadProfile.UE_COMMUNICATION_FORECASTING:
        return TRAFFIC_BUNDLE_FILES
    return IMAGE_CLASSIFICATION_BUNDLE_FILES


def image_dataset_contract(value: object) -> ImageDatasetContract:
    try:
        name = ImageDatasetName(value)
    except (TypeError, ValueError) as error:
        raise WorkloadContractError("image dataset is unsupported") from error
    return IMAGE_DATASET_CONTRACTS[name]


def image_training_contract(
    ml_event: str,
    model_interoperability: str,
) -> ImageDatasetContract:
    if ml_event != IMAGE_CLASSIFICATION_EVENT:
        raise WorkloadContractError("image classification event is unsupported")
    for dataset, interoperability_id in IMAGE_MODEL_INTEROPERABILITY.items():
        if model_interoperability == interoperability_id:
            return IMAGE_DATASET_CONTRACTS[dataset]
    raise WorkloadContractError("image model interoperability is unsupported")


def validate_image_manifest(manifest: Mapping[str, object]) -> ImageDatasetContract:
    try:
        contract = image_dataset_contract(manifest["dataset"])
    except KeyError as error:
        raise WorkloadContractError("image bundle dataset is required") from error
    model = manifest.get("model")
    inference = manifest.get("inference")
    if not isinstance(model, Mapping) or not isinstance(inference, Mapping):
        raise WorkloadContractError("image bundle model and inference contracts are required")
    if model.get("input_channels") != contract.input_channels:
        raise WorkloadContractError("image bundle input_channels does not match dataset")
    if model.get("num_classes") != contract.class_count:
        raise WorkloadContractError("image bundle num_classes does not match dataset")
    if inference.get("input_shape") != list(contract.input_shape):
        raise WorkloadContractError("image bundle input_shape does not match dataset")
    if inference.get("class_count") != contract.class_count:
        raise WorkloadContractError("image bundle class_count does not match dataset")
    if inference.get("normalization") != IMAGE_NORMALIZATION:
        raise WorkloadContractError("image bundle normalization is unsupported")
    return contract
