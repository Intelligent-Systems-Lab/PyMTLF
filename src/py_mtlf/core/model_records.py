from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from py_mtlf.models import SHA256_PATTERN


class DurableRecord(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda name: "".join(
            [name.split("_")[0], *[part.title() for part in name.split("_")[1:]]]
        ),
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class RevisionOrigin(StrEnum):
    SEED = "SEED"
    FEDERATED = "FEDERATED"


class PublicationState(StrEnum):
    RESERVED = "RESERVED"
    FINAL_BUNDLE_READY = "FINAL_BUNDLE_READY"
    STORE_IN_FLIGHT = "STORE_IN_FLIGHT"
    STORE_ACCEPTED = "STORE_ACCEPTED"
    CATALOG_COMMITTED = "CATALOG_COMMITTED"
    FAILED_TERMINAL = "FAILED_TERMINAL"


class AdrfReference(DurableRecord):
    adrf_instance_id: str = Field(min_length=1)
    store_trans_id: str = Field(min_length=1)
    resource_location: str = Field(min_length=1)

    @field_validator("adrf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))


class ParticipantSampleCount(DurableRecord):
    participant_nf_instance_id: str
    sample_count: int = Field(gt=0)

    @field_validator("participant_nf_instance_id")
    @classmethod
    def normalize_nf_instance_id(cls, value: str) -> str:
        return str(UUID(value))


class CatalogValidationSummary(DurableRecord):
    global_gate_accepted: bool | None = None


class CompletedRevision(DurableRecord):
    model_unique_id: int = Field(ge=0)
    previous_model_unique_id: int | None = Field(default=None, ge=0)
    origin: RevisionOrigin
    artifact_key: str = Field(pattern=SHA256_PATTERN.pattern)
    artifact_digest: str = Field(pattern=SHA256_PATTERN.pattern)
    created_at: AwareDatetime
    ml_corre_id: str | None = Field(default=None, min_length=1)
    participants: tuple[ParticipantSampleCount, ...] = ()
    validation_summary: CatalogValidationSummary = Field(
        default_factory=CatalogValidationSummary
    )
    adrf_reference: AdrfReference | None = None

    @model_validator(mode="after")
    def validate_origin(self) -> CompletedRevision:
        if self.artifact_key != self.artifact_digest:
            raise ValueError("artifact key and digest must identify the same content")
        if self.origin is RevisionOrigin.FEDERATED:
            if self.adrf_reference is None or not self.ml_corre_id or not self.participants:
                raise ValueError(
                    "FEDERATED revision requires ADRF reference, process ID, and participants"
                )
        elif self.adrf_reference is not None:
            raise ValueError("SEED revision cannot contain an ADRF reference")
        return self


class ModelCatalogRecord(DurableRecord):
    schema_version: Literal["1.0"]
    latest_model_id: int = Field(ge=0)
    next_model_id: int = Field(ge=0)
    revisions: tuple[CompletedRevision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> ModelCatalogRecord:
        by_id = {revision.model_unique_id: revision for revision in self.revisions}
        if len(by_id) != len(self.revisions):
            raise ValueError("catalog revision IDs must be unique")
        if self.latest_model_id not in by_id:
            raise ValueError("latest model ID must identify a completed revision")
        if self.next_model_id <= max(by_id):
            raise ValueError("next model ID must be greater than every completed revision")

        for revision in self.revisions:
            previous = revision.previous_model_unique_id
            if previous is not None and previous not in by_id:
                raise ValueError("previous model ID must identify a completed revision")

        for model_id in by_id:
            seen: set[int] = set()
            current: int | None = model_id
            while current is not None:
                if current in seen:
                    raise ValueError("catalog revisions cannot form a cycle")
                seen.add(current)
                current = by_id[current].previous_model_unique_id
        lineage: set[int] = set()
        current = self.latest_model_id
        while current is not None:
            lineage.add(current)
            current = by_id[current].previous_model_unique_id
        if lineage != set(by_id):
            raise ValueError("catalog revisions must form one linear lineage ending at latest")
        return self


class PendingPublication(DurableRecord):
    schema_version: Literal["1.0"]
    publication_id: str = Field(min_length=1)
    state: PublicationState
    ml_corre_id: str = Field(min_length=1)
    reserved_model_id: int = Field(ge=0)
    previous_model_id: int | None = Field(default=None, ge=0)
    participants_and_sample_counts: tuple[ParticipantSampleCount, ...] = Field(min_length=1)
    validation_summary: CatalogValidationSummary
    candidate_path: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=SHA256_PATTERN.pattern)
    final_bundle_path: str | None = None
    final_bundle_digest: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    selected_adrf_target: str | None = None
    store_trans_id: str | None = None
    resource_location: str | None = None
    last_error: str | None = None
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_state(self) -> PendingPublication:
        final_required = {
            PublicationState.FINAL_BUNDLE_READY,
            PublicationState.STORE_IN_FLIGHT,
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
        }
        if self.state in final_required and (
            not self.final_bundle_path or not self.final_bundle_digest
        ):
            raise ValueError("publication state requires a durable final bundle")
        store_target_required = {
            PublicationState.STORE_IN_FLIGHT,
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
        }
        if self.state in store_target_required and not self.selected_adrf_target:
            raise ValueError("publication state requires a selected ADRF target")
        accepted = {
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
        }
        if self.state in accepted and (not self.store_trans_id or not self.resource_location):
            raise ValueError("accepted publication requires ADRF record locators")
        if self.state is PublicationState.FAILED_TERMINAL and not self.last_error:
            raise ValueError("terminal failure requires last_error")
        return self


def migrate_seed_catalog(
    *,
    model_unique_id: int,
    artifact_key: str,
    created_at: AwareDatetime,
) -> ModelCatalogRecord:
    """Build the first durable catalog without changing the current runtime catalog."""

    seed = CompletedRevision(
        model_unique_id=model_unique_id,
        origin=RevisionOrigin.SEED,
        artifact_key=artifact_key,
        artifact_digest=artifact_key,
        created_at=created_at,
    )
    return ModelCatalogRecord(
        schema_version="1.0",
        latest_model_id=model_unique_id,
        next_model_id=model_unique_id + 1,
        revisions=(seed,),
    )


def validate_catalog_publications(
    catalog: ModelCatalogRecord,
    publications: tuple[PendingPublication, ...],
) -> None:
    """Validate invariants that cross the catalog and publication journal files."""

    publication_ids = [item.publication_id for item in publications]
    if len(publication_ids) != len(set(publication_ids)):
        raise ValueError("publication IDs must be unique")
    reserved_ids = [item.reserved_model_id for item in publications]
    if len(reserved_ids) != len(set(reserved_ids)):
        raise ValueError("reserved model IDs must not be reused")
    if reserved_ids and catalog.next_model_id <= max(reserved_ids):
        raise ValueError("next model ID must be greater than every reserved model ID")

    completed = {item.model_unique_id for item in catalog.revisions}
    for publication in publications:
        if (
            publication.previous_model_id is not None
            and publication.previous_model_id not in completed
        ):
            raise ValueError("publication previous model ID must be completed")
        committed = publication.state is PublicationState.CATALOG_COMMITTED
        if committed and publication.reserved_model_id not in completed:
            raise ValueError("committed publication model ID must exist in the catalog")
        if not committed and publication.reserved_model_id in completed:
            raise ValueError("uncommitted publication model ID cannot already be completed")
