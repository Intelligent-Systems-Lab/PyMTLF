from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from py_mtlf.core.fl_hierarchy import normalize_plan_id
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.models import SHA256_PATTERN, ModelIdentity

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN.pattern)]


class ArtifactContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRole(StrEnum):
    ROUND_INPUT = "ROUND_INPUT"
    ROUND_LOCAL = "ROUND_LOCAL"
    ROUND_GLOBAL = "ROUND_GLOBAL"
    FINAL_MODEL = "FINAL_MODEL"


class RoundLocalResultType(StrEnum):
    TRAINING = "TRAINING"
    ACCURACY_CHECK = "ACCURACY_CHECK"
    HIERARCHY_AGGREGATE = "HIERARCHY_AGGREGATE"


class ClientTrainingDirective(ArtifactContractModel):
    epochs: int = Field(gt=0, strict=True)


class WapeComponents(ArtifactContractModel):
    absolute_error_sum: float = Field(ge=0)
    absolute_actual_sum: float = Field(ge=0)


class ParticipantSample(ArtifactContractModel):
    participant_nf_instance_id: str
    training_sample_count: int = Field(gt=0)

    @field_validator("participant_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))


class RoundParticipant(ParticipantSample):
    local_artifact_digest: Sha256


class ValidationSummary(ArtifactContractModel):
    participant_nf_instance_id: str
    training_scope: TrainingScopeDescriptor
    evaluation_sample_count: int = Field(gt=0)
    start_time: AwareDatetime
    end_time: AwareDatetime
    base: WapeComponents
    candidate: WapeComponents

    @field_validator("participant_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))

    @model_validator(mode="after")
    def validate_window(self) -> ValidationSummary:
        if self.end_time <= self.start_time:
            raise ValueError("validation summary end_time must be after start_time")
        return self


class CommonFLMetadata(ArtifactContractModel):
    ml_corre_id: str = Field(min_length=1)


class RoundInputMetadata(ArtifactContractModel):
    ml_corre_id: str = Field(min_length=1)
    round_ind: int = Field(ge=0)
    client_training: ClientTrainingDirective


class RoundLocalCommonMetadata(CommonFLMetadata):
    round_ind: int = Field(ge=0)
    participant_nf_instance_id: str
    training_scope: TrainingScopeDescriptor

    @field_validator("participant_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))


class TrafficDatasetEvidence(ArtifactContractModel):
    workload_profile: Literal["ue_communication_forecasting"]
    observation_count: int = Field(gt=0)
    training_sample_count: int = Field(gt=0)
    validation_sample_count: int = Field(gt=0)


class ImageDatasetEvidence(ArtifactContractModel):
    workload_profile: Literal["image_classification"]
    dataset: Literal["mnist", "cifar10"]
    training_sample_count: int = Field(gt=0)


DatasetEvidence = Annotated[
    TrafficDatasetEvidence | ImageDatasetEvidence,
    Field(discriminator="workload_profile"),
]


class RoundLocalTrainingMetadata(RoundLocalCommonMetadata):
    training_sample_count: int = Field(gt=0)
    dataset_evidence: DatasetEvidence

    @model_validator(mode="after")
    def validate_dataset_evidence(self) -> RoundLocalTrainingMetadata:
        if self.dataset_evidence.training_sample_count != self.training_sample_count:
            raise ValueError(
                "dataset evidence sample count must match training sample count"
            )
        return self


class RoundLocalHierarchyAggregateMetadata(RoundLocalCommonMetadata):
    training_sample_count: int = Field(gt=0)
    lower_round_ind: int = Field(ge=0)
    lower_global_artifact_digest: Sha256
    subordinate_participants: tuple[RoundParticipant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_subordinates(self) -> RoundLocalHierarchyAggregateMetadata:
        identifiers = [item.participant_nf_instance_id for item in self.subordinate_participants]
        if identifiers != sorted(identifiers):
            raise ValueError("subordinate participants must use canonical NF instance ID ordering")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("subordinate participants must be unique")
        if self.participant_nf_instance_id in identifiers:
            raise ValueError("Branch participant cannot also be a subordinate participant")
        sample_count = sum(item.training_sample_count for item in self.subordinate_participants)
        if self.training_sample_count != sample_count:
            raise ValueError("training sample count must equal subordinate sample counts")
        return self


class AccuracyCheckEvaluation(ArtifactContractModel):
    evaluation_stage: Literal["FINAL_VALIDATION"]
    evaluation_sample_count: int = Field(gt=0)
    start_time: AwareDatetime
    end_time: AwareDatetime
    base: WapeComponents
    candidate: WapeComponents

    @model_validator(mode="after")
    def validate_window(self) -> AccuracyCheckEvaluation:
        if self.end_time <= self.start_time:
            raise ValueError("evaluation end_time must be after start_time")
        return self


class RoundLocalAccuracyCheckMetadata(RoundLocalCommonMetadata):
    evaluation: AccuracyCheckEvaluation
    subordinate_validation_summaries: tuple[ValidationSummary, ...] | None = Field(
        default=None,
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unchanged_candidate(self) -> RoundLocalAccuracyCheckMetadata:
        if self.subordinate_validation_summaries is not None:
            _validate_subordinate_validation_summaries(
                self.participant_nf_instance_id,
                self.evaluation,
                self.subordinate_validation_summaries,
            )
        return self


class RoundGlobalMetadata(CommonFLMetadata):
    round_ind: int = Field(ge=0)
    participants: tuple[RoundParticipant, ...] = Field(min_length=1)
    aggregated_training_sample_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_participants(self) -> RoundGlobalMetadata:
        identifiers = [participant.participant_nf_instance_id for participant in self.participants]
        if identifiers != sorted(identifiers):
            raise ValueError("round participants must use canonical NF instance ID ordering")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("round participants must be unique")
        sample_count = sum(item.training_sample_count for item in self.participants)
        if self.aggregated_training_sample_count != sample_count:
            raise ValueError("aggregated sample count must equal participant sample counts")
        return self


class HierarchyBranchValidation(ArtifactContractModel):
    branch_nf_instance_id: str
    subordinate_validation_summaries: tuple[ValidationSummary, ...] = Field(min_length=1)

    @field_validator("branch_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))

    @model_validator(mode="after")
    def validate_subordinates(self) -> HierarchyBranchValidation:
        identifiers = [
            item.participant_nf_instance_id
            for item in self.subordinate_validation_summaries
        ]
        if identifiers != sorted(identifiers):
            raise ValueError(
                "subordinate validation summaries must use canonical NF instance ID ordering"
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("subordinate validation summaries must be unique")
        if self.branch_nf_instance_id in identifiers:
            raise ValueError("Branch cannot also be a subordinate validation participant")
        return self


class HierarchyValidation(ArtifactContractModel):
    plan_id: str
    branches: tuple[HierarchyBranchValidation, ...] = Field(min_length=1)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return normalize_plan_id(value)

    @model_validator(mode="after")
    def validate_topology(self) -> HierarchyValidation:
        branch_ids = [item.branch_nf_instance_id for item in self.branches]
        if branch_ids != sorted(branch_ids):
            raise ValueError("hierarchy validation branches must use canonical ordering")
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("hierarchy validation branches must be unique")
        leaf_ids = [
            summary.participant_nf_instance_id
            for branch in self.branches
            for summary in branch.subordinate_validation_summaries
        ]
        if len(set(leaf_ids)) != len(leaf_ids):
            raise ValueError("hierarchy validation Leaves must be globally unique")
        if set(branch_ids).intersection(leaf_ids):
            raise ValueError("hierarchy validation Branches and Leaves must be disjoint")
        return self


class FinalModelMetadata(CommonFLMetadata):
    previous_model_unique_id: int | None = Field(default=None, ge=0)
    participants: tuple[ParticipantSample, ...] = Field(min_length=1)
    validation_summary: tuple[ValidationSummary, ...] = Field(min_length=1)
    hierarchy_validation: HierarchyValidation | None = None
    global_gate_accepted: bool
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_final(self) -> FinalModelMetadata:
        identifiers = [participant.participant_nf_instance_id for participant in self.participants]
        if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("final participants must be unique and canonically ordered")
        summary_ids = [item.participant_nf_instance_id for item in self.validation_summary]
        if (
            summary_ids != sorted(summary_ids)
            or set(summary_ids) != set(identifiers)
            or len(summary_ids) != len(set(summary_ids))
        ):
            raise ValueError(
                "validation summary must cover every participant once in canonical order"
            )
        if self.hierarchy_validation is not None:
            branch_ids = tuple(
                item.branch_nf_instance_id
                for item in self.hierarchy_validation.branches
            )
            if branch_ids != tuple(identifiers):
                raise ValueError(
                    "hierarchy validation branches must match final direct participants"
                )
            direct = {
                item.participant_nf_instance_id: item for item in self.validation_summary
            }
            for branch in self.hierarchy_validation.branches:
                _validate_subordinate_validation_summaries(
                    branch.branch_nf_instance_id,
                    direct[branch.branch_nf_instance_id],
                    branch.subordinate_validation_summaries,
                )
        if not self.global_gate_accepted:
            raise ValueError("FINAL_MODEL requires an accepted global gate")
        return self


def _validate_subordinate_validation_summaries(
    branch_nf_instance_id: str,
    aggregate: AccuracyCheckEvaluation | ValidationSummary,
    summaries: tuple[ValidationSummary, ...],
) -> None:
    identifiers = [item.participant_nf_instance_id for item in summaries]
    if identifiers != sorted(identifiers):
        raise ValueError(
            "subordinate validation summaries must use canonical NF instance ID ordering"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("subordinate validation summaries must be unique")
    if branch_nf_instance_id in identifiers:
        raise ValueError("Branch cannot also be a subordinate validation participant")
    if aggregate.evaluation_sample_count != sum(
        item.evaluation_sample_count for item in summaries
    ):
        raise ValueError(
            "aggregate evaluation sample count must equal subordinate validation summaries"
        )
    if aggregate.start_time != min(item.start_time for item in summaries):
        raise ValueError("aggregate start time must cover subordinate validation summaries")
    if aggregate.end_time != max(item.end_time for item in summaries):
        raise ValueError("aggregate end time must cover subordinate validation summaries")
    for label in ("base", "candidate"):
        aggregate_components = getattr(aggregate, label)
        subordinate_error = sum(
            getattr(item, label).absolute_error_sum for item in summaries
        )
        subordinate_actual = sum(
            getattr(item, label).absolute_actual_sum for item in summaries
        )
        if not math.isclose(
            aggregate_components.absolute_error_sum,
            subordinate_error,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            aggregate_components.absolute_actual_sum,
            subordinate_actual,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "aggregate WAPE components must equal subordinate validation summaries"
            )


class RoundInputArtifact(ArtifactContractModel):
    artifact_role: Literal[ArtifactRole.ROUND_INPUT]
    fl_metadata: RoundInputMetadata


class RoundLocalArtifact(ArtifactContractModel):
    artifact_role: Literal[ArtifactRole.ROUND_LOCAL]
    result_type: RoundLocalResultType
    fl_metadata: (
        RoundLocalTrainingMetadata
        | RoundLocalAccuracyCheckMetadata
        | RoundLocalHierarchyAggregateMetadata
    )

    @model_validator(mode="after")
    def validate_result_type(self) -> RoundLocalArtifact:
        expected_type = {
            RoundLocalResultType.TRAINING: RoundLocalTrainingMetadata,
            RoundLocalResultType.ACCURACY_CHECK: RoundLocalAccuracyCheckMetadata,
            RoundLocalResultType.HIERARCHY_AGGREGATE: RoundLocalHierarchyAggregateMetadata,
        }[self.result_type]
        if type(self.fl_metadata) is not expected_type:
            raise ValueError(f"{self.result_type.value} requires {expected_type.__name__}")
        return self


class RoundGlobalArtifact(ArtifactContractModel):
    artifact_role: Literal[ArtifactRole.ROUND_GLOBAL]
    fl_metadata: RoundGlobalMetadata


class FinalModelArtifact(ArtifactContractModel):
    artifact_role: Literal[ArtifactRole.FINAL_MODEL]
    model_identity: ModelIdentity
    fl_metadata: FinalModelMetadata


FLArtifactContract = Annotated[
    RoundInputArtifact
    | RoundLocalArtifact
    | RoundGlobalArtifact
    | FinalModelArtifact,
    Field(discriminator="artifact_role"),
]

_ARTIFACT_ADAPTER = TypeAdapter(FLArtifactContract)


def validate_fl_artifact(value: object) -> FLArtifactContract:
    """Validate a role-aware training artifact projection without publishing it."""

    return _ARTIFACT_ADAPTER.validate_python(value)


def fl_artifact_projection(manifest: Mapping[str, object]) -> dict[str, object]:
    """Extract one strict role-specific projection from a complete model manifest."""

    try:
        role = ArtifactRole(manifest["artifact_role"])
    except (KeyError, ValueError) as error:
        raise ValueError("FL artifact role is missing or unsupported") from error

    base_fields = {"artifact_role"}
    role_fields = {
        ArtifactRole.ROUND_INPUT: {"fl_metadata"},
        ArtifactRole.ROUND_LOCAL: {"result_type", "fl_metadata"},
        ArtifactRole.ROUND_GLOBAL: {"fl_metadata"},
        ArtifactRole.FINAL_MODEL: {"model_identity", "fl_metadata"},
    }[role]
    known_role_fields = {
        "result_type",
        "fl_metadata",
        "model_identity",
    }
    unexpected = sorted((known_role_fields - role_fields).intersection(manifest))
    if unexpected:
        raise ValueError(f"FL artifact role contains incompatible fields: {unexpected}")

    required = base_fields | role_fields
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"FL artifact role is missing required fields: {missing}")
    return {key: manifest[key] for key in required}


def validate_fl_artifact_manifest(manifest: Mapping[str, object]) -> FLArtifactContract:
    """Validate the role-aware projection contained in a complete model manifest."""

    return validate_fl_artifact(fl_artifact_projection(manifest))


def wape(components: WapeComponents) -> float | None:
    if components.absolute_actual_sum == 0:
        return None
    return components.absolute_error_sum / components.absolute_actual_sum


class TensorStateEntry(ArtifactContractModel):
    name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    floating: bool


def validate_tensor_compatibility(
    base: tuple[TensorStateEntry, ...],
    candidate: tuple[TensorStateEntry, ...],
) -> None:
    """Validate the state contract before a later FedAvg implementation uses it."""

    if len({entry.name for entry in base}) != len(base):
        raise ValueError("base tensor names must be unique")
    if len({entry.name for entry in candidate}) != len(candidate):
        raise ValueError("candidate tensor names must be unique")
    if len(base) != len(candidate):
        raise ValueError("candidate tensor state does not match the base model")
    for base_entry, candidate_entry in zip(base, candidate, strict=True):
        if (
            base_entry.name != candidate_entry.name
            or base_entry.shape != candidate_entry.shape
            or base_entry.dtype != candidate_entry.dtype
            or base_entry.floating != candidate_entry.floating
        ):
            raise ValueError("candidate tensor contract does not match the base model")
