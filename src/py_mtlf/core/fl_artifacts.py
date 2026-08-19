from __future__ import annotations

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

from py_mtlf.core.fl_hierarchy import AssignmentMetadata, PreparationResultMetadata
from py_mtlf.models import SHA256_PATTERN, ModelIdentity

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN.pattern)]


class ArtifactContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactRole(StrEnum):
    ROUND_INPUT = "ROUND_INPUT"
    ROUND_LOCAL = "ROUND_LOCAL"
    ROUND_GLOBAL = "ROUND_GLOBAL"
    FINAL_MODEL = "FINAL_MODEL"
    HIERARCHY_ASSIGNMENT = "HIERARCHY_ASSIGNMENT"
    HIERARCHY_PREPARATION_RESULT = "HIERARCHY_PREPARATION_RESULT"


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
    scope_digest: Sha256
    evaluation_sample_count: int = Field(gt=0)
    start_time: AwareDatetime
    end_time: AwareDatetime
    base_model_weights_digest: Sha256
    candidate_weights_digest: Sha256
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
    contract_version: Literal["1.0"]
    ml_corre_id: str = Field(min_length=1)
    model_contract_digest: Sha256
    preprocessing_contract_digest: Sha256
    base_weights_digest: Sha256
    weights_digest: Sha256


class RoundInputMetadata(ArtifactContractModel):
    contract_version: Literal["1.0"]
    ml_corre_id: str = Field(min_length=1)
    round_ind: int = Field(ge=0)
    model_contract_digest: Sha256
    preprocessing_contract_digest: Sha256
    weights_digest: Sha256
    client_training: ClientTrainingDirective


class RoundLocalCommonMetadata(CommonFLMetadata):
    round_ind: int = Field(ge=0)
    participant_nf_instance_id: str
    scope_digest: Sha256
    input_global_weights_digest: Sha256

    @field_validator("participant_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))

    @model_validator(mode="after")
    def validate_base_digest(self) -> RoundLocalCommonMetadata:
        if self.input_global_weights_digest != self.base_weights_digest:
            raise ValueError("input global weights digest must match base weights digest")
        return self


class RoundLocalTrainingMetadata(RoundLocalCommonMetadata):
    training_sample_count: int = Field(gt=0)


class RoundLocalHierarchyAggregateMetadata(RoundLocalTrainingMetadata):
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
    base_model_weights_digest: Sha256
    candidate_weights_digest: Sha256
    base: WapeComponents
    candidate: WapeComponents

    @model_validator(mode="after")
    def validate_window(self) -> AccuracyCheckEvaluation:
        if self.end_time <= self.start_time:
            raise ValueError("evaluation end_time must be after start_time")
        return self


class RoundLocalAccuracyCheckMetadata(RoundLocalCommonMetadata):
    evaluation: AccuracyCheckEvaluation

    @model_validator(mode="after")
    def validate_unchanged_candidate(self) -> RoundLocalAccuracyCheckMetadata:
        if self.weights_digest != self.input_global_weights_digest:
            raise ValueError(
                "accuracy-check output weights digest must match input global weights digest"
            )
        if self.evaluation.candidate_weights_digest != self.weights_digest:
            raise ValueError(
                "accuracy-check candidate weights digest must match artifact weights digest"
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


class FinalModelMetadata(CommonFLMetadata):
    previous_model_unique_id: int | None = Field(default=None, ge=0)
    participants: tuple[ParticipantSample, ...] = Field(min_length=1)
    final_candidate_digest: Sha256
    validation_summary: tuple[ValidationSummary, ...] = Field(min_length=1)
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
        if self.final_candidate_digest != self.weights_digest:
            raise ValueError("final candidate digest must match artifact weights digest")
        for summary in self.validation_summary:
            if summary.base_model_weights_digest != self.base_weights_digest:
                raise ValueError(
                    "validation summary base model digest must match final model base digest"
                )
            if summary.candidate_weights_digest != self.final_candidate_digest:
                raise ValueError(
                    "validation summary candidate digest must match final candidate digest"
                )
        if not self.global_gate_accepted:
            raise ValueError("FINAL_MODEL requires an accepted global gate")
        return self


class ArtifactContractBase(ArtifactContractModel):
    bundle_schema_version: Literal["1.0"]
    file_digests: dict[str, Sha256]

    @field_validator("file_digests")
    @classmethod
    def validate_file_digests(cls, value: dict[str, Sha256]) -> dict[str, Sha256]:
        expected = {"model.py", "model.npy", "scaler.pkl"}
        if set(value) != expected:
            raise ValueError("file_digests must cover the exact model bundle components")
        return value


class RoundInputArtifact(ArtifactContractBase):
    artifact_role: Literal[ArtifactRole.ROUND_INPUT]
    fl_metadata: RoundInputMetadata


class RoundLocalArtifact(ArtifactContractBase):
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


class RoundGlobalArtifact(ArtifactContractBase):
    artifact_role: Literal[ArtifactRole.ROUND_GLOBAL]
    fl_metadata: RoundGlobalMetadata


class FinalModelArtifact(ArtifactContractBase):
    artifact_role: Literal[ArtifactRole.FINAL_MODEL]
    model_identity: ModelIdentity
    fl_metadata: FinalModelMetadata


class HierarchyAssignmentArtifact(ArtifactContractBase):
    artifact_role: Literal[ArtifactRole.HIERARCHY_ASSIGNMENT]
    hierarchy_metadata: AssignmentMetadata


class HierarchyPreparationResultArtifact(ArtifactContractBase):
    artifact_role: Literal[ArtifactRole.HIERARCHY_PREPARATION_RESULT]
    hierarchy_metadata: PreparationResultMetadata


FLArtifactContract = Annotated[
    RoundInputArtifact
    | RoundLocalArtifact
    | RoundGlobalArtifact
    | FinalModelArtifact
    | HierarchyAssignmentArtifact
    | HierarchyPreparationResultArtifact,
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

    base_fields = {"bundle_schema_version", "file_digests", "artifact_role"}
    role_fields = {
        ArtifactRole.ROUND_INPUT: {"fl_metadata"},
        ArtifactRole.ROUND_LOCAL: {"result_type", "fl_metadata"},
        ArtifactRole.ROUND_GLOBAL: {"fl_metadata"},
        ArtifactRole.FINAL_MODEL: {"model_identity", "fl_metadata"},
        ArtifactRole.HIERARCHY_ASSIGNMENT: {"hierarchy_metadata"},
        ArtifactRole.HIERARCHY_PREPARATION_RESULT: {"hierarchy_metadata"},
    }[role]
    known_role_fields = {
        "result_type",
        "fl_metadata",
        "model_identity",
        "hierarchy_metadata",
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
    value_digest: Sha256


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
        if not base_entry.floating and base_entry.value_digest != candidate_entry.value_digest:
            raise ValueError("non-floating tensor state must remain identical to the base model")
