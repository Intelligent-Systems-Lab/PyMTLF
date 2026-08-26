from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator

from py_mtlf.models import DomainModel


class CollectionRequestState(StrEnum):
    PENDING = "PENDING"
    RESOLVING = "RESOLVING"
    SUBSCRIBING = "SUBSCRIBING"
    COLLECTING = "COLLECTING"
    STOPPING = "STOPPING"
    RECOVERING = "RECOVERING"
    RETAINED = "RETAINED"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"


class DescriptorState(StrEnum):
    NONE = "NONE"
    ACTIVE = "ACTIVE"
    RETAINED = "RETAINED"


class TrainingDataCollectionRequest(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    request_id: str = Field(alias="requestId")
    collection_profile_id: str = Field(min_length=1, alias="collectionProfileId")

    @field_validator("request_id")
    @classmethod
    def require_canonical_uuid4(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 4 or value != str(parsed):
            raise ValueError("requestId must be a canonical UUIDv4")
        return value

    @field_validator("collection_profile_id")
    @classmethod
    def normalize_profile_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("collectionProfileId must not be blank")
        return normalized


class TrainingDataCollectionStatus(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    request_id: str = Field(alias="requestId")
    collection_profile_id: str = Field(alias="collectionProfileId")
    collection_trigger: str = Field(default="private_api", alias="collectionTrigger")
    state: CollectionRequestState
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    process_generation: str = Field(alias="processGeneration")
    intended_group_count: int = Field(default=0, ge=0, alias="intendedGroupCount")
    resolved_ue_count: int = Field(default=0, ge=0, alias="resolvedUeCount")
    resolved_smf_target_count: int = Field(
        default=0,
        ge=0,
        alias="resolvedSmfTargetCount",
    )
    active_peer_resource_count: int = Field(
        default=0,
        ge=0,
        alias="activePeerResourceCount",
    )
    pending_cleanup_peer_resource_count: int = Field(
        default=0,
        ge=0,
        alias="pendingCleanupPeerResourceCount",
    )
    storage_transport: str = Field(default="unavailable", alias="storageTransport")
    start_time: datetime | None = Field(default=None, alias="startTime")
    stop_time: datetime | None = Field(default=None, alias="stopTime")
    record_count: int = Field(default=0, ge=0, alias="recordCount")
    observation_count: int = Field(default=0, ge=0, alias="observationCount")
    descriptor_state: DescriptorState = Field(default=DescriptorState.NONE, alias="descriptorState")
    failure_cause: str = Field(default="", alias="failureCause")
    failure_detail: str = Field(default="", alias="failureDetail")
    cleanup_pending: bool = Field(default=False, alias="cleanupPending")
