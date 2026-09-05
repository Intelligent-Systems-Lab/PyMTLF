from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class HierarchyContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HierarchyMessageType(StrEnum):
    BRANCH_ASSIGNMENT = "BRANCH_ASSIGNMENT"
    LEAF_ASSIGNMENT = "LEAF_ASSIGNMENT"
    PREPARATION_RESULT = "PREPARATION_RESULT"


class PreparationOutcome(StrEnum):
    READY = "READY"
    FAILED = "FAILED"


class PreparationFailureCause(StrEnum):
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    INVALID_ASSIGNMENT = "INVALID_ASSIGNMENT"
    INVALID_BUNDLE = "INVALID_BUNDLE"
    REQUIREMENTS_NOT_MET = "REQUIREMENTS_NOT_MET"
    NOT_AVAILABLE_ML_TRAIN = "NOT_AVAILABLE_ML_TRAIN"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def normalize_plan_id(value: str) -> str:
    normalized = _normalize_uuid(value, "plan_id")
    if UUID(normalized).version != 4:
        raise ValueError("plan_id must be a UUIDv4")
    return normalized


def normalize_nf_instance_id(value: str) -> str:
    return _normalize_uuid(value, "NF instance ID")


def _normalize_uuid(value: str, name: str) -> str:
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUID") from error


def _normalize_nf_instance_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("NF instance IDs must be a collection")
    try:
        return tuple(normalize_nf_instance_id(item) for item in value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("NF instance IDs must be a collection") from error


def _validate_canonical_ids(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must use canonical NF instance ID ordering")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


class FedProxAlgorithm(HierarchyContractModel):
    name: Literal["fedprox"]
    proximal_mu: float = Field(gt=0)


class FederatedStrategy(HierarchyContractModel):
    algorithm: FedProxAlgorithm
    participant_selection: Literal["all"]
    waiting_policy: Literal["all"]
    aggregation: Literal["sample_weighted"]


class CompleteRequiredAdmission(HierarchyContractModel):
    mode: Literal["complete_required"]


class CommonHierarchyMetadata(HierarchyContractModel):
    message_type: HierarchyMessageType
    plan_id: str
    publisher_nf_instance_id: str
    intended_recipient_nf_instance_id: str

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return normalize_plan_id(value)

    @field_validator("publisher_nf_instance_id", "intended_recipient_nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return normalize_nf_instance_id(value)

    @model_validator(mode="after")
    def validate_peer_identities(self) -> CommonHierarchyMetadata:
        if self.publisher_nf_instance_id == self.intended_recipient_nf_instance_id:
            raise ValueError("hierarchy publisher and intended recipient must differ")
        return self


class BranchAssignmentMetadata(CommonHierarchyMetadata):
    message_type: Literal[HierarchyMessageType.BRANCH_ASSIGNMENT]
    assigned_leaf_nf_instance_ids: tuple[str, ...] = Field(min_length=1)
    admission: CompleteRequiredAdmission
    strategy: FederatedStrategy

    @field_validator("assigned_leaf_nf_instance_ids", mode="before")
    @classmethod
    def normalize_leaf_ids(cls, value: object) -> tuple[str, ...]:
        return _normalize_nf_instance_ids(value)

    @model_validator(mode="after")
    def validate_assignment(self) -> BranchAssignmentMetadata:
        _validate_canonical_ids(
            self.assigned_leaf_nf_instance_ids,
            "assigned Leaf NF instance IDs",
        )
        peer_ids = {
            self.publisher_nf_instance_id,
            self.intended_recipient_nf_instance_id,
        }
        if peer_ids.intersection(self.assigned_leaf_nf_instance_ids):
            raise ValueError("Root, Branch, and assigned Leaf identities must be distinct")
        return self


class LeafAssignmentMetadata(CommonHierarchyMetadata):
    message_type: Literal[HierarchyMessageType.LEAF_ASSIGNMENT]
    parent_branch_nf_instance_id: str
    strategy: FederatedStrategy

    @field_validator("parent_branch_nf_instance_id")
    @classmethod
    def validate_parent_branch_id(cls, value: str) -> str:
        return normalize_nf_instance_id(value)

    @model_validator(mode="after")
    def validate_parent(self) -> LeafAssignmentMetadata:
        if self.parent_branch_nf_instance_id != self.publisher_nf_instance_id:
            raise ValueError("Leaf assignment publisher must be the parent Branch")
        return self


class PreparedClient(HierarchyContractModel):
    nf_instance_id: str

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return normalize_nf_instance_id(value)


class FailedClient(HierarchyContractModel):
    nf_instance_id: str
    cause: PreparationFailureCause

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return normalize_nf_instance_id(value)


class PreparationResultMetadata(CommonHierarchyMetadata):
    message_type: Literal[HierarchyMessageType.PREPARATION_RESULT]
    outcome: PreparationOutcome
    assigned_client_nf_instance_ids: tuple[str, ...] = Field(min_length=1)
    prepared_clients: tuple[PreparedClient, ...]
    failed_clients: tuple[FailedClient, ...]
    timed_out_client_nf_instance_ids: tuple[str, ...]

    @field_validator(
        "assigned_client_nf_instance_ids",
        "timed_out_client_nf_instance_ids",
        mode="before",
    )
    @classmethod
    def normalize_client_ids(cls, value: object) -> tuple[str, ...]:
        return _normalize_nf_instance_ids(value)

    @model_validator(mode="after")
    def validate_partition(self) -> PreparationResultMetadata:
        _validate_canonical_ids(
            self.assigned_client_nf_instance_ids,
            "assigned client NF instance IDs",
        )
        _validate_canonical_ids(
            self.timed_out_client_nf_instance_ids,
            "timed-out client NF instance IDs",
        )
        prepared_ids = tuple(item.nf_instance_id for item in self.prepared_clients)
        failed_ids = tuple(item.nf_instance_id for item in self.failed_clients)
        _validate_canonical_ids(prepared_ids, "prepared client NF instance IDs")
        _validate_canonical_ids(failed_ids, "failed client NF instance IDs")

        assigned = set(self.assigned_client_nf_instance_ids)
        prepared = set(prepared_ids)
        failed = set(failed_ids)
        timed_out = set(self.timed_out_client_nf_instance_ids)
        if prepared.intersection(failed) or prepared.intersection(timed_out) or failed.intersection(
            timed_out
        ):
            raise ValueError("preparation result partitions must be disjoint")
        if prepared | failed | timed_out != assigned:
            raise ValueError("preparation result partitions must exactly cover assigned clients")
        if {self.publisher_nf_instance_id, self.intended_recipient_nf_instance_id}.intersection(
            assigned
        ):
            raise ValueError("Root, Branch, and assigned client identities must be distinct")

        all_prepared = prepared == assigned and not failed and not timed_out
        if self.outcome is PreparationOutcome.READY and not all_prepared:
            raise ValueError("READY requires every assigned client to be prepared")
        if self.outcome is PreparationOutcome.FAILED and all_prepared:
            raise ValueError("FAILED requires at least one failed or timed-out client")
        return self


AssignmentMetadata = Annotated[
    BranchAssignmentMetadata | LeafAssignmentMetadata,
    Field(discriminator="message_type"),
]

HierarchyMetadata = Annotated[
    BranchAssignmentMetadata | LeafAssignmentMetadata | PreparationResultMetadata,
    Field(discriminator="message_type"),
]

_HIERARCHY_METADATA_ADAPTER = TypeAdapter(HierarchyMetadata)


def validate_hierarchy_metadata(value: object) -> HierarchyMetadata:
    return _HIERARCHY_METADATA_ADAPTER.validate_python(value)
