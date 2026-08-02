from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from py_mtlf.config import PublicationSettings
from py_mtlf.core.adrf_discovery import AdrfResolver
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository
from py_mtlf.core.fl_artifacts import ValidationSummary
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.model_records import (
    AdrfReference,
    CatalogValidationSummary,
    CompletedRevision,
    DurableModelState,
    DurableModelStateRepository,
    ModelCatalogRecord,
    ParticipantSampleCount,
    PendingPublication,
    PublicationState,
    RevisionOrigin,
)
from py_mtlf.core.seed_catalog import CatalogModel, FamilyKey, ModelCatalog
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.core.trainer import TrustedBundleLoader
from py_mtlf.wire.adrf import (
    AllowedConsumer,
    MLModelAddress,
    MLModelInfo,
    NadrfMLModelStoreRecord,
)

logger = logging.getLogger(__name__)

_STORED = "ML_MODEL_FILE_STORED_IN_ADRF"


class RecoverablePublicationError(RuntimeError):
    """A publication failure that may succeed without changing its identity."""


@dataclass(frozen=True)
class ValidatedCandidate:
    process_id: str
    family_key: FamilyKey
    base_artifact: ArtifactMetadata
    candidate_artifact: ArtifactMetadata
    participants: tuple[ParticipantSampleCount, ...]
    validation_summaries: tuple[ValidationSummary, ...]
    required_scope_keys: tuple[str, ...]
    gate_would_accept: bool
    gate_rejection_reasons: tuple[str, ...]


class PublicationCoordinator:
    """Durably promotes a validated FL candidate through ADRF before cutover."""

    def __init__(
        self,
        settings: PublicationSettings,
        state: DurableModelStateRepository,
        catalog: ModelCatalog,
        artifacts: ArtifactRepository,
        workspace: FLWorkspace,
        adrf_resolver: AdrfResolver,
        projection: SyncProjection,
        client: httpx.Client | None = None,
        on_published: Callable[[PendingPublication, CatalogModel], None] | None = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._catalog = catalog
        self._artifacts = artifacts
        self._workspace = workspace
        self._adrf_resolver = adrf_resolver
        self._projection = projection
        self._on_published = on_published or (lambda _publication, _model: None)
        self._loader = TrustedBundleLoader()
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._condition = threading.Condition(threading.RLock())
        self._advance_lock = threading.RLock()
        self._closing = False
        self._recovery_enabled = False
        self._worker: threading.Thread | None = None
        self._announced: set[str] = set()

    def open(self) -> None:
        self._settings.directory.mkdir(parents=True, exist_ok=True)
        with self._condition:
            if self._worker is not None:
                return
            self._closing = False
            self._worker = threading.Thread(
                target=self._recovery_loop,
                name="model-publication-reconciler",
                daemon=True,
            )
            self._worker.start()

    def resume(self) -> None:
        with self._condition:
            self._recovery_enabled = True
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=self._settings.request_timeout_seconds + 1)
        with self._condition:
            self._worker = None
        if self._owns_client:
            self._client.close()

    def publish(self, candidate: ValidatedCandidate) -> CatalogModel:
        publication = self._reserve(candidate)
        delay = self._settings.retry_interval_seconds
        while True:
            try:
                with self._advance_lock:
                    publication, model = self._advance(publication)
                with self._condition:
                    self._announced.add(publication.publication_id)
                return model
            except RecoverablePublicationError as error:
                publication = self._record_retry(publication.publication_id, error)
                logger.warning(
                    "Federated model publication will retry publication_id=%s retry=%s error=%s",
                    publication.publication_id,
                    publication.retry_count,
                    error,
                )
                with self._condition:
                    if self._closing:
                        raise RuntimeError("publication coordinator is shutting down") from error
                    self._condition.wait(timeout=delay)
                delay = min(delay * 2, self._settings.retry_max_interval_seconds)
            except Exception as error:
                self._mark_terminal(publication.publication_id, error)
                raise

    def mark_scope_adopted(
        self,
        family_key: FamilyKey,
        model_id: int,
        scope_key: str,
    ) -> bool:
        family_id = family_key
        completed = False
        matched = False

        def update(state: DurableModelState) -> DurableModelState:
            nonlocal completed, matched
            publications = []
            for item in state.pending_publications:
                if (
                    item.family_id != family_id
                    or item.reserved_model_id != model_id
                    or item.state
                    not in {PublicationState.CATALOG_COMMITTED, PublicationState.CUTOVER_PENDING}
                    or scope_key not in item.required_cutover_scopes
                ):
                    publications.append(item)
                    continue
                adopted = tuple(sorted(set(item.adopted_cutover_scopes) | {scope_key}))
                matched = True
                completed = set(adopted) == set(item.required_cutover_scopes)
                publications.append(
                    item.model_copy(
                        update={
                            "adopted_cutover_scopes": adopted,
                            "state": (
                                PublicationState.COMPLETE
                                if completed
                                else PublicationState.CUTOVER_PENDING
                            ),
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
            return state.model_copy(update={"pending_publications": tuple(publications)})

        self._state.update(update)
        if matched:
            logger.info(
                "Federated model scope adopted model_id=%s scope=%s complete=%s",
                model_id,
                scope_key,
                completed,
            )
        return completed

    def _reserve(self, candidate: ValidatedCandidate) -> PendingPublication:
        current = self._catalog.current(candidate.family_key)
        if current is None or current.artifact.key != candidate.base_artifact.key:
            raise RuntimeError("catalog base changed before final publication")
        existing = next(
            (
                item
                for item in self._state.snapshot().pending_publications
                if item.ml_corre_id == candidate.process_id
            ),
            None,
        )
        if existing is not None:
            return existing
        now = datetime.now(UTC)
        created: PendingPublication | None = None

        def update(state: DurableModelState) -> DurableModelState:
            nonlocal created
            family = state.families.get(candidate.family_key)
            if family is None or family.latest_model_id != current.model_id:
                raise RuntimeError("durable catalog base changed before version reservation")
            model_id = max(
                int(now.timestamp() * 1000),
                state.last_allocated_model_id + 1,
                family.next_model_id,
            )
            created = PendingPublication(
                schemaVersion="1.0",
                publicationId=str(uuid4()),
                state=PublicationState.RESERVED,
                mlCorreId=candidate.process_id,
                reservedModelId=model_id,
                previousModelId=current.model_id,
                familyId=candidate.family_key,
                expectedGeneration=current.generation,
                expectedArtifactDigest=current.artifact.key,
                participantsAndSampleCounts=candidate.participants,
                validationSummary=CatalogValidationSummary(
                    globalGateAccepted=True,
                    gateWouldAccept=candidate.gate_would_accept,
                    gateRejectionReasons=candidate.gate_rejection_reasons,
                ),
                validationEvidence=candidate.validation_summaries,
                candidatePath=str(candidate.candidate_artifact.path),
                candidateDigest=candidate.candidate_artifact.key,
                requiredCutoverScopes=candidate.required_scope_keys,
                updatedAt=now,
            )
            updated_family = family.model_copy(update={"next_model_id": model_id + 1})
            families = dict(state.families)
            families[candidate.family_key] = updated_family
            return state.model_copy(
                update={
                    "families": families,
                    "last_allocated_model_id": model_id,
                    "pending_publications": (*state.pending_publications, created),
                }
            )

        self._state.update(update)
        if created is None:
            raise RuntimeError("model ID reservation produced no publication")
        self._catalog.restore_reserved_version(candidate.family_key, created.reserved_model_id)
        return created

    def _build_final_bundle(self, publication: PendingPublication) -> PendingPublication:
        base = self._catalog.current(self._catalog.family_key_for_id(publication.family_id))
        if base is None or base.model_id != publication.previous_model_id:
            raise RuntimeError("publication base model is no longer current")
        base_bundle = self._loader.load(base.artifact)
        candidate_metadata = ArtifactMetadata(
            key=publication.candidate_digest,
            size_bytes=Path(publication.candidate_path).stat().st_size,
            path=Path(publication.candidate_path),
            url="",
        )
        candidate_bundle = self._loader.load(candidate_metadata)
        candidate_weights = weights_digest(candidate_bundle.model)
        created_at = datetime.now(UTC)
        final = self._workspace.publish(
            process_id=publication.ml_corre_id,
            participant_id="fl-server",
            round_indicator=0,
            role="FINAL_MODEL",
            base=candidate_bundle,
            model=candidate_bundle.model,
            metadata={
                "artifact_role": "FINAL_MODEL",
                "model_identity": {"model_unique_id": publication.reserved_model_id},
                "model_generation": publication.expected_generation + 1,
                "created_at": created_at.isoformat(),
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": publication.ml_corre_id,
                    "model_contract_digest": model_contract_digest(base_bundle.manifest),
                    "preprocessing_contract_digest": preprocessing_contract_digest(
                        base_bundle.manifest
                    ),
                    "base_weights_digest": weights_digest(base_bundle.model),
                    "weights_digest": candidate_weights,
                    "previous_model_unique_id": publication.previous_model_id,
                    "participants": [
                        {
                            "participant_nf_instance_id": item.participant_nf_instance_id,
                            "training_sample_count": item.sample_count,
                        }
                        for item in publication.participants_and_sample_counts
                    ],
                    "final_candidate_digest": candidate_weights,
                    "validation_summary": [
                        item.model_dump(mode="json") for item in publication.validation_evidence
                    ],
                    "global_gate_accepted": True,
                    "created_at": created_at.isoformat(),
                },
            },
        )
        artifact = self._artifacts.publish(final.path)
        return self._replace_publication(
            publication.model_copy(
                update={
                    "state": PublicationState.FINAL_BUNDLE_READY,
                    "final_bundle_path": str(artifact.path),
                    "final_bundle_digest": artifact.key,
                    "updated_at": datetime.now(UTC),
                }
            )
        )

    def _store_in_adrf(self, publication: PendingPublication) -> PendingPublication:
        if publication.state is PublicationState.FINAL_BUNDLE_READY:
            target = self._adrf_resolver.resolve_model()
            if target is None:
                raise RecoverablePublicationError(
                    "no ADRF with ML model storage capability is available"
                )
            target_api_root = target.api_root
            target_instance_id = target.nf_instance_id
        else:
            target_api_root = publication.selected_adrf_target or ""
            target_instance_id = publication.selected_adrf_instance_id or ""
            if not target_api_root or not target_instance_id:
                raise RuntimeError("in-flight publication lost its selected ADRF identity")
        snapshot = self._projection.snapshot()
        if snapshot is None:
            raise RecoverablePublicationError("containing NWDAF is not synchronized")
        artifact = self._artifacts.metadata(publication.final_bundle_digest or "")
        allowed_consumer_ids = {
            snapshot.containing_nwdaf.nf_instance_id,
            *(
                item.participant_nf_instance_id
                for item in publication.participants_and_sample_counts
            ),
        }
        if publication.state is PublicationState.FINAL_BUNDLE_READY:
            publication = self._replace_publication(
                publication.model_copy(
                    update={
                        "state": PublicationState.STORE_IN_FLIGHT,
                        "selected_adrf_target": target_api_root,
                        "selected_adrf_instance_id": target_instance_id,
                        "updated_at": datetime.now(UTC),
                    }
                )
            )

        record = self._retrieve_by_model_id(publication, target_api_root)
        if record is None:
            request_record = NadrfMLModelStoreRecord(
                nfInstanceId=snapshot.containing_nwdaf.nf_instance_id,
                mlModelInfo=[
                    MLModelInfo(
                        modelUniqueId=publication.reserved_model_id,
                        mlFileAddr=MLModelAddress(mLModelUrl=artifact.url),
                        mlStorageSize=artifact.size_bytes,
                        allowConsumerList=[
                            AllowedConsumer(nfInstanceId=consumer_id)
                            for consumer_id in sorted(allowed_consumer_ids)
                        ],
                    )
                ],
            )
            try:
                response = self._client.post(
                    self._go_base() + "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records",
                    headers={"Target-Api-Root": target_api_root},
                    json=request_record.model_dump(
                        by_alias=True,
                        exclude_none=True,
                        mode="json",
                    ),
                )
            except httpx.HTTPError as error:
                raise RecoverablePublicationError("ADRF ML model store transport failed") from error
            if response.status_code != 201 or not response.headers.get("Location"):
                if response.status_code == 429 or response.status_code >= 500:
                    raise RecoverablePublicationError(
                        f"ADRF ML model store returned status {response.status_code}"
                    )
                raise RuntimeError(f"ADRF ML model store returned status {response.status_code}")
            record = NadrfMLModelStoreRecord.model_validate(response.json())
            resource_location = response.headers["Location"]
        else:
            resource_location = publication.resource_location or ""
            if not resource_location:
                stored_url = record.ml_model_info[0].model_file_address.model_url or ""
                suffix = "/model"
                if not stored_url.endswith(suffix):
                    raise RuntimeError(
                        "recovered ADRF record has no derivable store resource identity"
                    )
                resource_location = stored_url[: -len(suffix)]
        result = record.model_store_result
        if (
            result is None
            or result.model_unique_id != publication.reserved_model_id
            or result.store_result != _STORED
        ):
            raise RuntimeError("ADRF did not confirm successful ML model storage")
        stored_info = record.ml_model_info[0]
        if record.nf_instance_id != snapshot.containing_nwdaf.nf_instance_id:
            raise RuntimeError("ADRF record owner does not match the publishing NWDAF")
        if stored_info.model_unique_id != publication.reserved_model_id:
            raise RuntimeError("ADRF response changed the reserved model identity")
        if stored_info.model_storage_size != artifact.size_bytes:
            raise RuntimeError("ADRF response changed the final model storage size")
        returned_consumers = {
            item.nf_instance_id
            for item in stored_info.allowed_consumers
            if item.nf_instance_id is not None
        }
        if returned_consumers != allowed_consumer_ids:
            raise RuntimeError("ADRF response changed the allowed consumer set")
        store_trans_id = resource_location.rstrip("/").rsplit("/", 1)[-1]
        if not store_trans_id:
            raise RuntimeError("ADRF store response has no transaction identity")
        return self._replace_publication(
            publication.model_copy(
                update={
                    "state": PublicationState.STORE_ACCEPTED,
                    "store_trans_id": store_trans_id,
                    "resource_location": resource_location,
                    "updated_at": datetime.now(UTC),
                }
            )
        )

    def _retrieve_by_model_id(
        self,
        publication: PendingPublication,
        target_api_root: str,
    ) -> NadrfMLModelStoreRecord | None:
        try:
            response = self._client.get(
                self._go_base() + "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records",
                headers={"Target-Api-Root": target_api_root},
                params={"model-unique-ids": str(publication.reserved_model_id)},
            )
        except httpx.HTTPError as error:
            raise RecoverablePublicationError(
                "ADRF ML model recovery lookup transport failed"
            ) from error
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            if response.status_code == 429 or response.status_code >= 500:
                raise RecoverablePublicationError(
                    f"ADRF ML model recovery lookup returned status {response.status_code}"
                )
            raise RuntimeError(
                f"ADRF ML model recovery lookup returned status {response.status_code}"
            )
        return NadrfMLModelStoreRecord.model_validate(response.json())

    def _commit_catalog(self, publication: PendingPublication) -> PendingPublication:
        if not publication.store_trans_id or not publication.resource_location:
            raise RuntimeError("ADRF acceptance record is incomplete")
        family_key = self._catalog.family_key_for_id(publication.family_id)
        current = self._catalog.current(family_key)
        if (
            current is None
            or current.model_id != publication.previous_model_id
            or current.generation != publication.expected_generation
            or current.artifact.key != publication.expected_artifact_digest
        ):
            raise RuntimeError("catalog base changed before durable publication commit")
        artifact = self._artifacts.metadata(publication.final_bundle_digest or "")
        reference = AdrfReference(
            adrfInstanceId=publication.selected_adrf_instance_id,
            storeTransId=publication.store_trans_id,
            resourceLocation=publication.resource_location,
        )
        revision = CompletedRevision(
            modelUniqueId=publication.reserved_model_id,
            previousModelUniqueId=publication.previous_model_id,
            origin=RevisionOrigin.FEDERATED,
            artifactKey=artifact.key,
            artifactDigest=artifact.key,
            createdAt=datetime.now(UTC),
            generation=publication.expected_generation + 1,
            mlCorreId=publication.ml_corre_id,
            participants=publication.participants_and_sample_counts,
            validationSummary=publication.validation_summary,
            adrfReference=reference,
        )

        def update(state: DurableModelState) -> DurableModelState:
            family = state.families[publication.family_id]
            latest = next(
                item for item in family.revisions if item.model_unique_id == family.latest_model_id
            )
            if (
                family.latest_model_id != publication.previous_model_id
                or latest.generation != publication.expected_generation
                or latest.artifact_digest != publication.expected_artifact_digest
            ):
                raise RuntimeError("durable catalog base changed before publication commit")
            updated_family = ModelCatalogRecord(
                schemaVersion="1.0",
                latestModelId=publication.reserved_model_id,
                nextModelId=max(family.next_model_id, publication.reserved_model_id + 1),
                revisions=(*family.revisions, revision),
            )
            families = dict(state.families)
            families[publication.family_id] = updated_family
            publications = tuple(
                item.model_copy(
                    update={
                        "state": PublicationState.CATALOG_COMMITTED,
                        "updated_at": datetime.now(UTC),
                    }
                )
                if item.publication_id == publication.publication_id
                else item
                for item in state.pending_publications
            )
            return state.model_copy(
                update={"families": families, "pending_publications": publications}
            )

        state = self._state.update(update)
        committed = next(
            item
            for item in state.pending_publications
            if item.publication_id == publication.publication_id
        )
        self._catalog.promote(
            family_key,
            expected_generation=publication.expected_generation,
            expected_artifact_key=publication.expected_artifact_digest or "",
            version_key=self._catalog.version_key_for_id(publication.reserved_model_id),
            artifact=artifact,
            adrf_reference=reference,
        )
        return committed

    def _replace_publication(self, publication: PendingPublication) -> PendingPublication:
        def update(state: DurableModelState) -> DurableModelState:
            publications = tuple(
                publication if item.publication_id == publication.publication_id else item
                for item in state.pending_publications
            )
            if publications == state.pending_publications:
                raise RuntimeError("publication journal entry is missing")
            return state.model_copy(update={"pending_publications": publications})

        state = self._state.update(update)
        return next(
            item
            for item in state.pending_publications
            if item.publication_id == publication.publication_id
        )

    def _advance(
        self,
        publication: PendingPublication,
    ) -> tuple[PendingPublication, CatalogModel]:
        if publication.state is PublicationState.RESERVED:
            publication = self._build_final_bundle(publication)
        if publication.state in {
            PublicationState.FINAL_BUNDLE_READY,
            PublicationState.STORE_IN_FLIGHT,
        }:
            publication = self._store_in_adrf(publication)
        if publication.state is PublicationState.STORE_ACCEPTED:
            publication = self._commit_catalog(publication)
        family_key = self._catalog.family_key_for_id(publication.family_id)
        model = self._catalog.current(family_key)
        if model is None or model.model_id != publication.reserved_model_id:
            raise RuntimeError("durable publication did not become the current catalog model")
        if publication.state is PublicationState.CATALOG_COMMITTED:
            publication = self._replace_publication(
                publication.model_copy(
                    update={
                        "state": (
                            PublicationState.CUTOVER_PENDING
                            if publication.required_cutover_scopes
                            else PublicationState.COMPLETE
                        ),
                        "updated_at": datetime.now(UTC),
                    }
                )
            )
        logger.info(
            "Federated model published publication_id=%s model_id=%s state=%s required_scopes=%s",
            publication.publication_id,
            publication.reserved_model_id,
            publication.state,
            len(publication.required_cutover_scopes),
        )
        return publication, model

    def _announce(self, publication: PendingPublication, model: CatalogModel) -> None:
        with self._condition:
            if publication.publication_id in self._announced:
                return
        try:
            self._on_published(publication, model)
        except Exception as error:
            raise RecoverablePublicationError(
                "published model handoff to cutover reconciliation failed"
            ) from error
        with self._condition:
            self._announced.add(publication.publication_id)

    def _record_retry(
        self,
        publication_id: str,
        error: Exception,
    ) -> PendingPublication:
        updated: PendingPublication | None = None

        def update(state: DurableModelState) -> DurableModelState:
            nonlocal updated
            publications = []
            for item in state.pending_publications:
                if item.publication_id != publication_id:
                    publications.append(item)
                    continue
                updated = item.model_copy(
                    update={
                        "retry_count": item.retry_count + 1,
                        "last_error": str(error),
                        "updated_at": datetime.now(UTC),
                    }
                )
                publications.append(updated)
            return state.model_copy(update={"pending_publications": tuple(publications)})

        self._state.update(update)
        if updated is None:
            raise RuntimeError("publication journal entry is missing")
        return updated

    def _mark_terminal(self, publication_id: str, error: Exception) -> None:
        def update(state: DurableModelState) -> DurableModelState:
            publications = []
            tombstones = set(state.tombstoned_model_ids)
            for item in state.pending_publications:
                if item.publication_id != publication_id:
                    publications.append(item)
                    continue
                if item.state in {
                    PublicationState.CATALOG_COMMITTED,
                    PublicationState.CUTOVER_PENDING,
                    PublicationState.COMPLETE,
                }:
                    publications.append(item)
                    continue
                publications.append(
                    item.model_copy(
                        update={
                            "state": PublicationState.FAILED_TERMINAL,
                            "last_error": str(error),
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
                tombstones.add(item.reserved_model_id)
            return state.model_copy(
                update={
                    "pending_publications": tuple(publications),
                    "tombstoned_model_ids": tuple(sorted(tombstones)),
                }
            )

        self._state.update(update)

    def _recovery_loop(self) -> None:
        delay = self._settings.retry_interval_seconds
        while True:
            with self._condition:
                while not self._closing and not self._recovery_enabled:
                    self._condition.wait()
                if self._closing:
                    return
            recoverable = [
                item
                for item in self._state.snapshot().pending_publications
                if item.state
                not in {
                    PublicationState.COMPLETE,
                    PublicationState.FAILED_TERMINAL,
                }
                and item.publication_id not in self._announced
            ]
            if not recoverable:
                with self._condition:
                    self._condition.wait()
                delay = self._settings.retry_interval_seconds
                continue
            made_progress = False
            for publication in recoverable:
                try:
                    with self._advance_lock:
                        publication = next(
                            (
                                item
                                for item in self._state.snapshot().pending_publications
                                if item.publication_id == publication.publication_id
                            ),
                            publication,
                        )
                        if publication.state in {
                            PublicationState.COMPLETE,
                            PublicationState.FAILED_TERMINAL,
                        }:
                            continue
                        with self._condition:
                            if publication.publication_id in self._announced:
                                continue
                        if publication.state not in {
                            PublicationState.CATALOG_COMMITTED,
                            PublicationState.CUTOVER_PENDING,
                        }:
                            self._catalog.restore_reserved_version(
                                self._catalog.family_key_for_id(publication.family_id),
                                publication.reserved_model_id,
                            )
                        publication, model = self._advance(publication)
                    self._announce(publication, model)
                    made_progress = True
                except RecoverablePublicationError as error:
                    self._record_retry(publication.publication_id, error)
                    logger.warning(
                        "Durable publication recovery will retry publication_id=%s error=%s",
                        publication.publication_id,
                        error,
                    )
                except Exception as error:
                    self._mark_terminal(publication.publication_id, error)
                    logger.exception(
                        "Durable ML model publication recovery failed terminally publication_id=%s",
                        publication.publication_id,
                    )
            if made_progress:
                delay = self._settings.retry_interval_seconds
                continue
            with self._condition:
                if self._closing:
                    return
                self._condition.wait(timeout=delay)
            delay = min(delay * 2, self._settings.retry_max_interval_seconds)

    def _go_base(self) -> str:
        snapshot = self._projection.snapshot()
        if snapshot is None:
            raise RuntimeError("containing NWDAF is not synchronized")
        return snapshot.containing_nwdaf.internal_callback_base_uri.rstrip("/")
