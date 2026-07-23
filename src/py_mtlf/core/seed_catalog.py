import json
from dataclasses import dataclass

from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import ArtifactMetadata, ArtifactRepository, InvalidArtifactError
from py_mtlf.wire.ml_model import MLEventNotification, MLEventSubscription, MLModelAddress


@dataclass(frozen=True)
class SeedModel:
    descriptor: SeedModelSettings
    artifact: ArtifactMetadata

    def event_notification(self, correlation_id: str) -> MLEventNotification:
        values = {
            "event": self.descriptor.event,
            "mLFileAddr": MLModelAddress(mLModelUrl=self.artifact.url),
            "modelUniqueId": self.descriptor.model_id,
            "mLEventFilter": self.descriptor.event_filter,
            "tgtUe": self.descriptor.target_ue,
        }
        if correlation_id:
            values["notifCorreId"] = correlation_id
        if self.descriptor.use_case_context:
            values["useCaseCxt"] = self.descriptor.use_case_context
        return MLEventNotification.model_validate(values)


class SeedCatalog:
    def __init__(
        self,
        settings: ModelProvisionSettings,
        artifacts: ArtifactRepository,
    ) -> None:
        self._settings = settings
        self._artifacts = artifacts
        self._models: tuple[SeedModel, ...] = ()

    @property
    def provider_namespace(self) -> str:
        return self._settings.provider_namespace

    def open(self) -> None:
        models = []
        for descriptor in self._settings.seed_models:
            metadata = self._artifacts.metadata(descriptor.artifact_key)
            manifest = self._artifacts.manifest(descriptor.artifact_key)
            self._validate_manifest(descriptor, manifest)
            models.append(SeedModel(descriptor=descriptor, artifact=metadata))
        self._models = tuple(models)

    def resolve(self, demand: MLEventSubscription) -> SeedModel | None:
        for seed in self._models:
            descriptor = seed.descriptor
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
            return seed
        return None

    def snapshot(self) -> tuple[SeedModel, ...]:
        return self._models

    @staticmethod
    def notifications(
        seeds: tuple[SeedModel | None, ...],
        correlation_id: str,
    ) -> list[MLEventNotification]:
        unique: dict[tuple[int, str], SeedModel] = {}
        for seed in seeds:
            if seed is None:
                continue
            key = (
                seed.descriptor.model_id,
                seed.artifact.key,
            )
            unique.setdefault(key, seed)
        return [
            seed.event_notification(correlation_id)
            for seed in unique.values()
        ]

    def _validate_manifest(
        self,
        descriptor: SeedModelSettings,
        manifest: dict[str, object],
    ) -> None:
        identity = manifest.get("model_identity")
        expected_identity = {
            "provider_id": self._settings.provider_namespace,
            "model_unique_id": descriptor.model_id,
        }
        if identity != expected_identity:
            raise InvalidArtifactError(
                "seed descriptor identity does not match artifact manifest"
            )
        if manifest.get("analytics_event") != descriptor.event:
            raise InvalidArtifactError(
                "seed descriptor event does not match artifact manifest"
            )

    @staticmethod
    def _canonical_equal(left: object, right: object) -> bool:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
        )
