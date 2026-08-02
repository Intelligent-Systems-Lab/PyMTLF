import json
import threading
import time
from dataclasses import dataclass

from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository, InvalidArtifactError
from py_mtlf.core.model_records import AdrfReference, DurableModelState, PublicationState
from py_mtlf.wire.ml_model import (
    MLEventNotification,
    MLEventSubscription,
    MLModelAddress,
    MLModelAdrf,
)

FamilyKey = str
ModelVersionKey = int


class StaleCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogModel:
    family_key: FamilyKey
    version_key: ModelVersionKey
    descriptor: SeedModelSettings
    artifact: ArtifactMetadata
    generation: int
    adrf_reference: AdrfReference | None = None

    @property
    def model_id(self) -> int:
        return self.version_key

    def event_notification(self, correlation_id: str) -> MLEventNotification:
        values: dict[str, object] = {
            "event": self.descriptor.event,
            "modelUniqueId": self.model_id,
            "mLEventFilter": self.descriptor.event_filter,
            "tgtUe": self.descriptor.target_ue,
        }
        if self.adrf_reference is None:
            values["mLFileAddr"] = MLModelAddress(mLModelUrl=self.artifact.url)
        else:
            values["mLModelAdrf"] = MLModelAdrf(
                adrfId=self.adrf_reference.adrf_instance_id,
                storTransId=self.adrf_reference.store_trans_id,
            )
        if correlation_id:
            values["notifCorreId"] = correlation_id
        if self.descriptor.use_case_context:
            values["useCaseCxt"] = self.descriptor.use_case_context
        return MLEventNotification.model_validate(values)


@dataclass(frozen=True)
class VersionIndexEntry:
    family_key: FamilyKey
    generation: int
    artifact_key: str
    current: bool


SeedModel = CatalogModel


class ModelCatalog:
    """In-memory family catalog with provider-wide, non-reused version IDs."""

    def __init__(
        self,
        settings: ModelProvisionSettings,
        artifacts: ArtifactRepository,
        lock=None,
    ) -> None:
        self._settings = settings
        self._artifacts = artifacts
        self._lock = lock or threading.RLock()
        self._current_by_family: dict[FamilyKey, CatalogModel] = {}
        self._version_index: dict[ModelVersionKey, VersionIndexEntry] = {}
        self._reserved_ids: set[int] = set()
        self._tombstoned_ids: set[int] = set()
        self._next_model_id = 0

    def open(self) -> None:
        current: dict[FamilyKey, CatalogModel] = {}
        versions: dict[ModelVersionKey, VersionIndexEntry] = {}
        for descriptor in self._settings.seed_models:
            metadata = self._artifacts.metadata(descriptor.artifact_key)
            manifest = self._artifacts.manifest(descriptor.artifact_key)
            family_key = self.family_key_for_id(descriptor.family_id)
            version_key = self.version_key_for_id(descriptor.model_id)
            self._validate_manifest(descriptor, descriptor.model_id, manifest)
            model = CatalogModel(
                family_key=family_key,
                version_key=version_key,
                descriptor=descriptor,
                artifact=metadata,
                generation=1,
            )
            current[family_key] = model
            versions[version_key] = VersionIndexEntry(
                family_key=family_key,
                generation=1,
                artifact_key=metadata.key,
                current=True,
            )
        with self._lock:
            self._current_by_family = current
            self._version_index = versions
            self._reserved_ids.clear()
            self._tombstoned_ids.clear()
            self._next_model_id = (
                max(
                    versions,
                    default=-1,
                )
                + 1
            )

    def restore(self, state: DurableModelState) -> None:
        descriptors = {item.family_id: item for item in self._settings.seed_models}
        if set(state.families) != set(descriptors):
            raise StaleCatalogError("durable model families do not match configured families")
        current: dict[FamilyKey, CatalogModel] = {}
        versions: dict[ModelVersionKey, VersionIndexEntry] = {}
        for family_id, record in state.families.items():
            descriptor = descriptors[family_id]
            family_key = self.family_key_for_id(family_id)
            for revision in record.revisions:
                metadata = self._artifacts.metadata(revision.artifact_key)
                manifest = self._artifacts.manifest(revision.artifact_key)
                self._validate_manifest(descriptor, revision.model_unique_id, manifest)
                version_key = self.version_key_for_id(revision.model_unique_id)
                is_current = revision.model_unique_id == record.latest_model_id
                versions[version_key] = VersionIndexEntry(
                    family_key=family_key,
                    generation=revision.generation,
                    artifact_key=metadata.key,
                    current=is_current,
                )
                if is_current:
                    current[family_key] = CatalogModel(
                        family_key=family_key,
                        version_key=version_key,
                        descriptor=descriptor,
                        artifact=metadata,
                        generation=revision.generation,
                        adrf_reference=revision.adrf_reference,
                    )
        reserved = {
            item.reserved_model_id
            for item in state.pending_publications
            if item.state
            not in {
                PublicationState.CATALOG_COMMITTED,
                PublicationState.CUTOVER_PENDING,
                PublicationState.COMPLETE,
                PublicationState.FAILED_TERMINAL,
            }
        }
        with self._lock:
            self._current_by_family = current
            self._version_index = versions
            self._reserved_ids = reserved
            self._tombstoned_ids = set(state.tombstoned_model_ids)
            self._next_model_id = max(
                state.last_allocated_model_id + 1,
                int(time.time() * 1000),
            )

    def family_key_for_id(self, family_id: str) -> FamilyKey:
        return family_id

    def version_key_for_id(self, model_id: int) -> ModelVersionKey:
        return model_id

    def resolve(self, demand: MLEventSubscription) -> CatalogModel | None:
        with self._lock:
            models = tuple(self._current_by_family.values())
            version_index = dict(self._version_index)
        if demand.model_id is not None:
            entry = version_index.get(self.version_key_for_id(demand.model_id))
            if entry is not None:
                return self.current(entry.family_key)
        for model in models:
            descriptor = model.descriptor
            if descriptor.event != demand.ml_event:
                continue
            if demand.model_interoperability and (
                descriptor.model_interoperability != demand.model_interoperability
            ):
                continue
            if descriptor.event_filter and not self._canonical_equal(
                descriptor.event_filter,
                demand.ml_event_filter,
            ):
                continue
            if descriptor.target_ue is not None and not self._canonical_equal(
                descriptor.target_ue,
                demand.target_ue,
            ):
                continue
            if descriptor.use_case_context and (
                descriptor.use_case_context != demand.use_case_context
            ):
                continue
            return model
        return None

    def resolve_key(self, demand: MLEventSubscription) -> FamilyKey | None:
        model = self.resolve(demand)
        return model.family_key if model is not None else None

    def current(self, family_key: FamilyKey | None) -> CatalogModel | None:
        if family_key is None:
            return None
        with self._lock:
            return self._current_by_family.get(family_key)

    def family_for_version(self, version_key: ModelVersionKey) -> FamilyKey | None:
        with self._lock:
            entry = self._version_index.get(version_key)
            return entry.family_key if entry is not None else None

    def version_entry(self, version_key: ModelVersionKey) -> VersionIndexEntry | None:
        with self._lock:
            return self._version_index.get(version_key)

    def snapshot(self) -> tuple[CatalogModel, ...]:
        with self._lock:
            return tuple(self._current_by_family[key] for key in sorted(self._current_by_family))

    def artifact_manifest(self, artifact_key: str) -> dict[str, object]:
        return self._artifacts.manifest(artifact_key)

    def notifications(
        self,
        family_keys: tuple[FamilyKey | None, ...],
        correlation_id: str,
    ) -> list[MLEventNotification]:
        unique: dict[ModelVersionKey, CatalogModel] = {}
        with self._lock:
            for key in family_keys:
                model = self._current_by_family.get(key) if key is not None else None
                if model is not None:
                    unique.setdefault(model.version_key, model)
        return [model.event_notification(correlation_id) for model in unique.values()]

    def reserve_next_version(self, family_key: FamilyKey) -> ModelVersionKey:
        with self._lock:
            if family_key not in self._current_by_family:
                raise StaleCatalogError("model family no longer exists in current catalog")
            self._next_model_id = max(self._next_model_id, int(time.time() * 1000))
            while (
                self._next_model_id in self._reserved_ids
                or self._next_model_id in self._tombstoned_ids
                or self.version_key_for_id(self._next_model_id) in self._version_index
            ):
                self._next_model_id += 1
            model_id = self._next_model_id
            self._next_model_id += 1
            self._reserved_ids.add(model_id)
            return self.version_key_for_id(model_id)

    def restore_reserved_version(self, family_key: FamilyKey, model_id: int) -> ModelVersionKey:
        with self._lock:
            if family_key not in self._current_by_family:
                raise StaleCatalogError("model family no longer exists in current catalog")
            version_key = self.version_key_for_id(model_id)
            if version_key in self._version_index or model_id in self._tombstoned_ids:
                raise StaleCatalogError("durable reserved model ID is already unavailable")
            self._reserved_ids.add(model_id)
            self._next_model_id = max(self._next_model_id, model_id + 1)
            return version_key

    def observe_external_model_ids(self, model_ids: set[int]) -> None:
        """Reserve restored IDs without guessing which family owned unknown versions."""

        with self._lock:
            known_ids = set(self._version_index)
            self._tombstoned_ids.update(model_ids - known_ids)
            if model_ids:
                self._next_model_id = max(self._next_model_id, max(model_ids) + 1)

    def promote(
        self,
        family_key: FamilyKey,
        *,
        expected_generation: int,
        expected_artifact_key: str,
        version_key: ModelVersionKey,
        artifact: ArtifactMetadata,
        adrf_reference: AdrfReference | None = None,
    ) -> CatalogModel:
        manifest = self._artifacts.manifest(artifact.key)
        with self._lock:
            current = self._current_by_family.get(family_key)
            if current is None:
                raise StaleCatalogError("model family no longer exists in current catalog")
            if (
                current.generation != expected_generation
                or current.artifact.key != expected_artifact_key
            ):
                raise StaleCatalogError("catalog base model changed during training")
            if version_key not in self._reserved_ids:
                raise StaleCatalogError("candidate model version was not reserved by this catalog")
            self._validate_manifest(current.descriptor, version_key, manifest)
            declared_generation = manifest.get("model_generation")
            if declared_generation != expected_generation + 1:
                raise InvalidArtifactError(
                    "candidate manifest generation does not follow current model"
                )
            promoted = CatalogModel(
                family_key=family_key,
                version_key=version_key,
                descriptor=current.descriptor,
                artifact=artifact,
                generation=expected_generation + 1,
                adrf_reference=adrf_reference,
            )
            previous = self._version_index[current.version_key]
            self._version_index[current.version_key] = VersionIndexEntry(
                family_key=previous.family_key,
                generation=previous.generation,
                artifact_key=previous.artifact_key,
                current=False,
            )
            self._version_index[version_key] = VersionIndexEntry(
                family_key=family_key,
                generation=promoted.generation,
                artifact_key=artifact.key,
                current=True,
            )
            self._current_by_family[family_key] = promoted
            self._reserved_ids.discard(version_key)
            return promoted

    def _validate_manifest(
        self,
        descriptor: SeedModelSettings,
        model_id: int,
        manifest: dict[str, object],
    ) -> None:
        identity = manifest.get("model_identity")
        expected_identity = {"model_unique_id": model_id}
        if identity != expected_identity:
            raise InvalidArtifactError("model descriptor identity does not match artifact manifest")
        if manifest.get("analytics_event") != descriptor.event:
            raise InvalidArtifactError("model descriptor event does not match artifact manifest")
        if descriptor.model_interoperability and (
            manifest.get("model_interoperability") != descriptor.model_interoperability
        ):
            raise InvalidArtifactError(
                "model descriptor interoperability does not match artifact manifest"
            )

    @staticmethod
    def _canonical_equal(left: object, right: object) -> bool:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
        )


SeedCatalog = ModelCatalog
