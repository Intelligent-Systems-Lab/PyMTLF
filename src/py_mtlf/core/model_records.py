from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from py_mtlf.core.fl_artifacts import ValidationSummary
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
    CUTOVER_PENDING = "CUTOVER_PENDING"
    COMPLETE = "COMPLETE"
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
    gate_would_accept: bool | None = None
    gate_rejection_reasons: tuple[str, ...] = ()


class CompletedRevision(DurableRecord):
    model_unique_id: int = Field(ge=0)
    previous_model_unique_id: int | None = Field(default=None, ge=0)
    origin: RevisionOrigin
    artifact_key: str = Field(pattern=SHA256_PATTERN.pattern)
    artifact_digest: str = Field(pattern=SHA256_PATTERN.pattern)
    created_at: AwareDatetime
    generation: int = Field(default=1, gt=0)
    ml_corre_id: str | None = Field(default=None, min_length=1)
    participants: tuple[ParticipantSampleCount, ...] = ()
    validation_summary: CatalogValidationSummary = Field(default_factory=CatalogValidationSummary)
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
    family_id: str = Field(default="", min_length=0)
    expected_generation: int = Field(default=1, gt=0)
    expected_artifact_digest: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN.pattern,
    )
    participants_and_sample_counts: tuple[ParticipantSampleCount, ...] = Field(min_length=1)
    validation_summary: CatalogValidationSummary
    validation_evidence: tuple[ValidationSummary, ...] = Field(min_length=1)
    candidate_path: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=SHA256_PATTERN.pattern)
    final_bundle_path: str | None = None
    final_bundle_digest: str | None = Field(default=None, pattern=SHA256_PATTERN.pattern)
    selected_adrf_target: str | None = None
    selected_adrf_instance_id: str | None = None
    store_trans_id: str | None = None
    resource_location: str | None = None
    required_cutover_scopes: tuple[str, ...] = ()
    adopted_cutover_scopes: tuple[str, ...] = ()
    retry_count: int = Field(default=0, ge=0)
    last_error: str | None = None
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_state(self) -> PendingPublication:
        final_required = {
            PublicationState.FINAL_BUNDLE_READY,
            PublicationState.STORE_IN_FLIGHT,
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
            PublicationState.CUTOVER_PENDING,
            PublicationState.COMPLETE,
        }
        if self.state in final_required and (
            not self.final_bundle_path or not self.final_bundle_digest
        ):
            raise ValueError("publication state requires a durable final bundle")
        store_target_required = {
            PublicationState.STORE_IN_FLIGHT,
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
            PublicationState.CUTOVER_PENDING,
            PublicationState.COMPLETE,
        }
        if self.state in store_target_required and not self.selected_adrf_target:
            raise ValueError("publication state requires a selected ADRF target")
        accepted = {
            PublicationState.STORE_ACCEPTED,
            PublicationState.CATALOG_COMMITTED,
            PublicationState.CUTOVER_PENDING,
            PublicationState.COMPLETE,
        }
        if self.state in accepted and (not self.store_trans_id or not self.resource_location):
            raise ValueError("accepted publication requires ADRF record locators")
        if self.state is PublicationState.FAILED_TERMINAL and not self.last_error:
            raise ValueError("terminal failure requires last_error")
        if not set(self.adopted_cutover_scopes).issubset(self.required_cutover_scopes):
            raise ValueError("adopted cutover scopes must be required")
        if self.state is PublicationState.COMPLETE and set(self.adopted_cutover_scopes) != set(
            self.required_cutover_scopes
        ):
            raise ValueError("complete publication requires every cutover scope")
        return self


class DurableModelState(DurableRecord):
    schema_version: Literal["1.0"]
    provider_namespace: str = Field(min_length=1)
    last_allocated_model_id: int = Field(ge=0)
    families: dict[str, ModelCatalogRecord] = Field(default_factory=dict)
    pending_publications: tuple[PendingPublication, ...] = ()
    tombstoned_model_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_inventory(self) -> DurableModelState:
        if any(not family_id.strip() for family_id in self.families):
            raise ValueError("durable family IDs must not be blank")
        completed = [
            revision.model_unique_id
            for catalog in self.families.values()
            for revision in catalog.revisions
        ]
        if len(completed) != len(set(completed)):
            raise ValueError("completed model IDs must be provider-wide unique")
        pending = [item.reserved_model_id for item in self.pending_publications]
        if len(pending) != len(set(pending)):
            raise ValueError("pending model IDs must be provider-wide unique")
        tombstones = list(self.tombstoned_model_ids)
        if len(tombstones) != len(set(tombstones)):
            raise ValueError("tombstoned model IDs must be unique")
        all_ids = completed + pending + tombstones
        if all_ids and self.last_allocated_model_id < max(all_ids):
            raise ValueError("last allocated model ID is behind durable model state")
        for publication in self.pending_publications:
            catalog = self.families.get(publication.family_id)
            if catalog is None:
                raise ValueError("publication family is not present in durable state")
            committed = publication.state in {
                PublicationState.CATALOG_COMMITTED,
                PublicationState.CUTOVER_PENDING,
                PublicationState.COMPLETE,
            }
            if committed != (publication.reserved_model_id in completed):
                raise ValueError("publication commit state conflicts with completed revisions")
        return self


class DurableModelStateRepository:
    """One atomic snapshot for completed models, allocation and publication recovery."""

    def __init__(self, directory: Path, lock=None) -> None:
        self._directory = Path(directory)
        self._path = self._directory / "model-state.json"
        self._lock = lock or threading.RLock()
        self._state: DurableModelState | None = None

    def open(self, initial: DurableModelState) -> DurableModelState:
        with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    state = DurableModelState.model_validate_json(
                        self._path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError) as error:
                    raise RuntimeError("durable model state cannot be loaded") from error
                if state.provider_namespace != initial.provider_namespace:
                    raise RuntimeError("durable model state provider namespace changed")
                self._state = state
            else:
                self._write(initial)
                self._state = initial
            return self.snapshot()

    def snapshot(self) -> DurableModelState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("durable model state repository is not open")
            return self._state.model_copy(deep=True)

    def update(
        self,
        operation: Callable[[DurableModelState], DurableModelState],
    ) -> DurableModelState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("durable model state repository is not open")
            updated = operation(self._state.model_copy(deep=True))
            updated = DurableModelState.model_validate(
                updated.model_dump(by_alias=True, mode="json")
            )
            self._write(updated)
            self._state = updated
            return updated.model_copy(deep=True)

    def _write(self, state: DurableModelState) -> None:
        payload = json.dumps(
            state.model_dump(by_alias=True, exclude_none=True, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".model-state.",
            dir=self._directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)

    def _fsync_directory(self) -> None:
        descriptor = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


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
        committed = publication.state in {
            PublicationState.CATALOG_COMMITTED,
            PublicationState.CUTOVER_PENDING,
            PublicationState.COMPLETE,
        }
        if committed and publication.reserved_model_id not in completed:
            raise ValueError("committed publication model ID must exist in the catalog")
        if not committed and publication.reserved_model_id in completed:
            raise ValueError("uncommitted publication model ID cannot already be completed")
