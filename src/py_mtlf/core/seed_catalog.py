import json
import threading
from dataclasses import dataclass

from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository, InvalidArtifactError
from py_mtlf.wire.ml_model import MLEventNotification, MLEventSubscription, MLModelAddress

FamilyKey = tuple[str, str]
ModelVersionKey = tuple[str, int]


class StaleCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogModel:
    family_key: FamilyKey
    version_key: ModelVersionKey
    descriptor: SeedModelSettings
    artifact: ArtifactMetadata
    generation: int

    @property
    def model_id(self) -> int:
        return self.version_key[1]

    def event_notification(self, correlation_id: str) -> MLEventNotification:
        values = {
            "event": self.descriptor.event,
            "mLFileAddr": MLModelAddress(mLModelUrl=self.artifact.url),
            "modelUniqueId": self.model_id,
            "mLEventFilter": self.descriptor.event_filter,
            "tgtUe": self.descriptor.target_ue,
        }
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

    @property
    def provider_namespace(self) -> str:
        return self._settings.provider_namespace

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
            self._next_model_id = max(
                (key[1] for key in versions),
                default=-1,
            ) + 1

    def family_key_for_id(self, family_id: str) -> FamilyKey:
        return self._settings.provider_namespace, family_id

    def version_key_for_id(self, model_id: int) -> ModelVersionKey:
        return self._settings.provider_namespace, model_id

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
            if descriptor.model_interoperability and (
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
            return tuple(
                self._current_by_family[key] for key in sorted(self._current_by_family)
            )

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

    def observe_external_model_ids(self, model_ids: set[int]) -> None:
        """Reserve restored IDs without guessing which family owned unknown versions."""

        with self._lock:
            known_ids = {key[1] for key in self._version_index}
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
            if (
                version_key[0] != self.provider_namespace
                or version_key[1] not in self._reserved_ids
            ):
                raise StaleCatalogError("candidate model version was not reserved by this catalog")
            self._validate_manifest(current.descriptor, version_key[1], manifest)
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
            self._reserved_ids.discard(version_key[1])
            return promoted

    def _validate_manifest(
        self,
        descriptor: SeedModelSettings,
        model_id: int,
        manifest: dict[str, object],
    ) -> None:
        identity = manifest.get("model_identity")
        expected_identity = {
            "provider_id": self._settings.provider_namespace,
            "model_unique_id": model_id,
        }
        if identity != expected_identity:
            raise InvalidArtifactError("model descriptor identity does not match artifact manifest")
        if manifest.get("analytics_event") != descriptor.event:
            raise InvalidArtifactError("model descriptor event does not match artifact manifest")

    @staticmethod
    def _canonical_equal(left: object, right: object) -> bool:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
        )


SeedCatalog = ModelCatalog
