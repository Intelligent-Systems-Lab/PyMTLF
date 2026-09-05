import re
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from py_mtlf.models import SHA256_PATTERN


class FrozenSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerSettings(FrozenSettings):
    binding_host: str = "127.0.0.1"
    port: int = Field(default=9092, ge=1, le=65535)

    @field_validator("binding_host")
    @classmethod
    def host_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("binding_host must not be blank")
        return value


class StorageSettings(FrozenSettings):
    artifact_root: Path = Path("data/artifacts")

    @field_validator("artifact_root", mode="before")
    @classmethod
    def path_must_not_be_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("storage path must not be blank")
        return value


class ModelStateSettings(FrozenSettings):
    directory: Path = Path("data/model-state")

    @field_validator("directory", mode="before")
    @classmethod
    def path_must_not_be_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("model_state.directory must not be blank")
        return value


class PublicationSettings(FrozenSettings):
    directory: Path = Path("data/publications")
    request_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    retry_interval_seconds: float = Field(default=2, gt=0, le=600)
    retry_max_interval_seconds: float = Field(default=30, gt=0, le=3600)
    probe_attempts: int = Field(default=3, gt=0, le=20)

    @field_validator("directory", mode="before")
    @classmethod
    def path_must_not_be_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("publication.directory must not be blank")
        return value

    @model_validator(mode="after")
    def validate_retry(self) -> "PublicationSettings":
        if self.retry_interval_seconds > self.retry_max_interval_seconds:
            raise ValueError("publication initial retry must not exceed maximum")
        return self


class CutoverSettings(FrozenSettings):
    timeout_seconds: int = Field(default=600, gt=0, le=86400)
    retry_interval_seconds: float = Field(default=5, gt=0, le=600)


class RuntimeSettings(FrozenSettings):
    mode: str = "local"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"local", "federated"}:
            raise ValueError("runtime.mode must be 'local' or 'federated'")
        return value


class ContainingNwdafSettings(FrozenSettings):
    internal_api_root: str = "http://127.0.0.1:8091"
    request_timeout_seconds: float = Field(default=30, gt=0, le=3600)

    @field_validator("internal_api_root")
    @classmethod
    def validate_internal_api_root(cls, value: str) -> str:
        return _validate_http_base_url(value, "containing_nwdaf.internal_api_root")


def _validate_http_base_url(value: str, field_name: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain userinfo, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{field_name} must not contain a path")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{field_name} contains an invalid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field_name} port must be between 1 and 65535")
    return value


class ArtifactSettings(FrozenSettings):
    public_base_url: str = "http://127.0.0.1:9092"
    max_compressed_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_extracted_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)
    max_single_file_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    max_entries: int = Field(default=32, gt=0)

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, value: str) -> str:
        return _validate_http_base_url(value, "artifact.public_base_url")

    @model_validator(mode="after")
    def validate_limits(self) -> "ArtifactSettings":
        if self.max_single_file_bytes > self.max_extracted_bytes:
            raise ValueError("max_single_file_bytes must not exceed max_extracted_bytes")
        return self


class ArtifactDownloadSettings(FrozenSettings):
    allowed_origins: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=300, gt=0, le=3600)

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _validate_http_base_url(value, "federated_learning.artifact_download.allowed_origins")
            for value in values
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("artifact download allowed origins must be unique")
        return normalized


class FittingRuntimeSettings(FrozenSettings):
    device: str = "cpu"
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    validation_ratio: float = Field(default=0.10, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value == "cpu":
            return value
        match = re.fullmatch(r"cuda(?::(0|[1-9][0-9]*))?", value)
        if match is None:
            raise ValueError("training.device must be 'cpu', 'cuda', or 'cuda:N'")
        return f"cuda:{match.group(1) or '0'}"


class FittingSettings(FittingRuntimeSettings):
    epochs: int = Field(default=18, gt=0)


class FederatedFittingSettings(FittingRuntimeSettings):
    pass


class ValidationSettings(FrozenSettings):
    enforce_performance_gate: bool = False
    max_scope_wape_regression: float = Field(default=0.02, ge=0)


class LocalTrainingSettings(FittingSettings):
    enabled: bool = True
    max_concurrent_jobs: int = Field(default=1, ge=1, le=32)
    max_queue_size: int = Field(default=16, gt=0)
    validation: ValidationSettings = ValidationSettings()


class FallbackDeadlineSettings(FrozenSettings):
    preparation_timeout_seconds: int = Field(default=300, gt=0, le=86400)
    round_timeout_seconds: int = Field(default=300, gt=0, le=86400)


class ConsumerSubscriptionTrainingDataSettings(FrozenSettings):
    collection_trigger: Literal["consumer_subscription"]


class LocalImageTrainingDataSettings(FrozenSettings):
    collection_trigger: Literal["local"]
    dataset: Literal["mnist", "cifar10"]
    shard_path: Path

    @field_validator("shard_path", mode="before")
    @classmethod
    def validate_shard_path(cls, value: object) -> object:
        if not isinstance(value, (str, Path)):
            raise ValueError("local image shard_path must be a filesystem path")
        if isinstance(value, str) and not value.strip():
            raise ValueError("local image shard_path must not be blank")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("local image shard_path must be resolved from config")
        return path


class PrivateCollectionConsentSettings(FrozenSettings):
    purpose: Literal["model_training"]
    policy: Literal["not_required_by_local_policy"]


class PrivateCollectionTargetSettings(FrozenSettings):
    int_group_ids: tuple[str, ...] = Field(alias="intGroupIds", min_length=1)

    @field_validator("int_group_ids")
    @classmethod
    def validate_group_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("private collection Internal Group IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("private collection Internal Group IDs must be unique")
        return normalized


class PrivateCollectionSnssaiSettings(FrozenSettings):
    sst: int = Field(ge=0, le=255)
    sd: str = ""

    @field_validator("sd")
    @classmethod
    def validate_sd(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized and not re.fullmatch(r"[0-9A-F]{6}", normalized):
            raise ValueError("private collection S-NSSAI sd must be six hexadecimal digits")
        return normalized


class PrivateCollectionPlmnIdSettings(FrozenSettings):
    mcc: str
    mnc: str

    @field_validator("mcc")
    @classmethod
    def validate_mcc(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[0-9]{3}", normalized):
            raise ValueError("private collection MCC must be three digits")
        return normalized

    @field_validator("mnc")
    @classmethod
    def validate_mnc(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[0-9]{2,3}", normalized):
            raise ValueError("private collection MNC must be two or three digits")
        return normalized


class PrivateCollectionTaiSettings(FrozenSettings):
    plmn_id: PrivateCollectionPlmnIdSettings
    tac: str

    @field_validator("tac")
    @classmethod
    def validate_tac(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not re.fullmatch(r"[0-9A-F]{6}", normalized):
            raise ValueError("private collection TAC must be six hexadecimal digits")
        return normalized

    def wire_value(self) -> dict[str, object]:
        return {
            "plmnId": {
                "mcc": self.plmn_id.mcc,
                "mnc": self.plmn_id.mnc,
            },
            "tac": self.tac,
        }


class PrivateCollectionNetworkAreaSettings(FrozenSettings):
    tais: tuple[PrivateCollectionTaiSettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_tais(self) -> "PrivateCollectionNetworkAreaSettings":
        identities = tuple(
            (item.plmn_id.mcc, item.plmn_id.mnc, item.tac) for item in self.tais
        )
        if len(identities) != len(set(identities)):
            raise ValueError("private collection TAIs must be unique")
        return self

    def wire_value(self) -> dict[str, object]:
        return {"tais": [item.wire_value() for item in self.tais]}


def _contains_area_filter(value: object) -> bool:
    forbidden = {"area", "networkarea", "tai", "tais", "trackingarealist"}
    if isinstance(value, dict):
        return any(
            str(key).replace("_", "").lower() in forbidden
            or _contains_area_filter(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_area_filter(item) for item in value)
    return False


class PrivateCollectionProfileSettings(FrozenSettings):
    profile_id: str
    ml_event: Literal["UE_COMMUNICATION"]
    ml_event_filter: dict[str, Any] = Field(default_factory=dict)
    target_ue: PrivateCollectionTargetSettings
    network_area: PrivateCollectionNetworkAreaSettings
    dnns: tuple[str, ...] = ()
    snssais: tuple[PrivateCollectionSnssaiSettings, ...] = ()
    sampling_interval_seconds: int = Field(gt=0, le=86400)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("private collection profile_id must not be blank")
        return normalized

    @field_validator("ml_event_filter")
    @classmethod
    def reject_area_filters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if _contains_area_filter(value):
            raise ValueError("private collection profile must not contain area or TAI filters")
        return value

    @field_validator("dnns")
    @classmethod
    def validate_dnns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip().lower() for value in values)
        label = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
        if any(
            not value
            or len(value) > 253
            or any(not label.fullmatch(part) for part in value.split("."))
            for value in normalized
        ):
            raise ValueError("private collection DNN values must be valid domain names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("private collection DNN values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_unique_snssais(self) -> "PrivateCollectionProfileSettings":
        identities = tuple((item.sst, item.sd) for item in self.snssais)
        if len(identities) != len(set(identities)):
            raise ValueError("private collection S-NSSAI values must be unique")
        return self


class PrivateAPITrainingDataSettings(FrozenSettings):
    collection_trigger: Literal["private_api"]
    callback_base_uri: str
    state_directory: Path
    request_timeout_seconds: float = Field(default=30, gt=0, le=3600)
    retry_initial_backoff_seconds: float = Field(default=1, ge=0, le=600)
    retry_max_backoff_seconds: float = Field(default=30, gt=0, le=3600)
    worker_count: int = Field(default=2, gt=0, le=32)
    queue_capacity: int = Field(default=256, gt=0, le=100000)
    descriptor_retention_seconds: int = Field(default=3600, gt=0, le=604800)
    consent: PrivateCollectionConsentSettings
    collection_profiles: tuple[PrivateCollectionProfileSettings, ...] = Field(min_length=1)

    @field_validator("callback_base_uri")
    @classmethod
    def validate_callback_base_uri(cls, value: str) -> str:
        return _validate_http_base_url(
            value,
            "federated_learning.client.training_data.callback_base_uri",
        )

    @field_validator("state_directory", mode="before")
    @classmethod
    def validate_state_directory(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("private collection state_directory must not be blank")
        path = Path(value)  # type: ignore[arg-type]
        if not path.is_absolute():
            raise ValueError("private collection state_directory must be resolved from config")
        return path

    @model_validator(mode="after")
    def validate_private_collection(self) -> "PrivateAPITrainingDataSettings":
        if self.retry_initial_backoff_seconds > self.retry_max_backoff_seconds:
            raise ValueError("private collection initial retry must not exceed maximum")
        profile_ids = tuple(profile.profile_id for profile in self.collection_profiles)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("private collection profile IDs must be unique")
        return self


FLTrainingDataSettings = Annotated[
    ConsumerSubscriptionTrainingDataSettings
    | PrivateAPITrainingDataSettings
    | LocalImageTrainingDataSettings,
    Field(discriminator="collection_trigger"),
]


class UECommunicationWorkloadSettings(FrozenSettings):
    profile: Literal["ue_communication_forecasting"] = "ue_communication_forecasting"


class ImageClassificationWorkloadSettings(FrozenSettings):
    profile: Literal["image_classification"]


FLWorkloadSettings = Annotated[
    UECommunicationWorkloadSettings | ImageClassificationWorkloadSettings,
    Field(discriminator="profile"),
]


class FLClientSettings(FrozenSettings):
    workload: FLWorkloadSettings = UECommunicationWorkloadSettings()
    training_data: FLTrainingDataSettings | None = None
    callback_deadline_margin_seconds: int = Field(default=5, ge=1, le=300)
    callback_queue_size: int = Field(default=256, gt=0, le=100000)
    max_concurrent_jobs: int = Field(default=2, gt=0, le=32)
    model_interoperability_ids: tuple[str, ...] = ()
    fallback_deadlines: FallbackDeadlineSettings = FallbackDeadlineSettings()
    training: FederatedFittingSettings = FederatedFittingSettings()

    @field_validator("model_interoperability_ids")
    @classmethod
    def validate_model_interoperability_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("model_interoperability_ids must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("model_interoperability_ids must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_capacity_and_deadline_margin(self) -> "FLClientSettings":
        if self.callback_deadline_margin_seconds >= min(
            self.fallback_deadlines.preparation_timeout_seconds,
            self.fallback_deadlines.round_timeout_seconds,
        ):
            raise ValueError("callback deadline margin must be shorter than FL timeouts")
        if self.callback_queue_size < self.max_concurrent_jobs:
            raise ValueError("callback_queue_size must not be smaller than max_concurrent_jobs")
        image_workload = isinstance(self.workload, ImageClassificationWorkloadSettings)
        local_data = isinstance(self.training_data, LocalImageTrainingDataSettings)
        if image_workload and self.training_data is not None and not local_data:
            raise ValueError(
                "image_classification training data must use the local source"
            )
        if not image_workload and (self.training_data is None or local_data):
            raise ValueError(
                "ue_communication_forecasting training_data requires "
                "consumer_subscription or private_api"
            )
        return self


class DelayPolicySettings(FrozenSettings):
    max_extensions: int = Field(default=1, ge=0, le=10)
    max_extension_seconds: int = Field(default=300, ge=0, le=86400)


class CleanupSettings(FrozenSettings):
    max_attempts: int = Field(default=3, gt=0, le=20)
    retry_backoff_seconds: float = Field(default=0.2, ge=0, le=60)


class ClientTrainingSettings(FrozenSettings):
    epochs: int = Field(default=18, gt=0, strict=True)


class FLServerSettings(FrozenSettings):
    callback_uri: str = "http://127.0.0.1:9092/internal/v1/ml-model-training/notifications"
    preparation_timeout_seconds: int = Field(default=300, gt=0, le=86400)
    preparation_data_window_seconds: int = Field(default=3600, gt=0, le=604800)
    round_timeout_seconds: int = Field(default=300, gt=0, le=86400)
    round_count: int = Field(default=2, ge=1, le=100)
    max_active_processes: int = Field(default=1, gt=0, le=32)
    client_training: ClientTrainingSettings = ClientTrainingSettings()
    delay_policy: DelayPolicySettings = DelayPolicySettings()
    cleanup: CleanupSettings = CleanupSettings()
    final_validation: ValidationSettings = ValidationSettings()

    @field_validator("callback_uri")
    @classmethod
    def validate_callback_uri(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "federated_learning.server.callback_uri must be an absolute HTTP(S) URI"
            )
        return value


class FedProxAlgorithmSettings(FrozenSettings):
    name: Literal["fedprox"]
    proximal_mu: float = Field(gt=0)

    @field_validator("proximal_mu")
    @classmethod
    def validate_proximal_mu(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("federated_learning.strategy.algorithm.proximal_mu must be finite")
        return value


class FederatedStrategySettings(FrozenSettings):
    algorithm: FedProxAlgorithmSettings
    participant_selection: Literal["all"]
    waiting_policy: Literal["all"]
    aggregation: Literal["sample_weighted"]


class TopologySettings(FrozenSettings):
    strategy: Literal["static"]
    config_file: Path

    @field_validator("config_file", mode="before")
    @classmethod
    def validate_config_file(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("federated_learning.topology.config_file must not be blank")
        path = Path(value)  # type: ignore[arg-type]
        if not path.is_absolute():
            raise ValueError(
                "federated_learning.topology.config_file must be resolved from the main config"
            )
        return path


class OrchestrationSettings(FrozenSettings):
    mode: Literal["flat", "hierarchical"]
    participant_source: Literal["monitor_scopes", "static"]
    hierarchy_contract: Literal["model_bundle", "protocol"] = "model_bundle"

    @model_validator(mode="after")
    def validate_mode_and_participant_source(self) -> "OrchestrationSettings":
        if self.mode == "hierarchical" and self.participant_source != "static":
            raise ValueError("hierarchical orchestration requires static participants")
        if self.mode == "flat" and "hierarchy_contract" in self.model_fields_set:
            raise ValueError("flat orchestration must not configure hierarchy_contract")
        return self


class DegradationTrainingTriggerSettings(FrozenSettings):
    enabled: bool = False


class PrivateTrainingTriggerSettings(FrozenSettings):
    enabled: bool = False


class TrainingTriggerSettings(FrozenSettings):
    degradation: DegradationTrainingTriggerSettings = DegradationTrainingTriggerSettings()
    private_api: PrivateTrainingTriggerSettings = PrivateTrainingTriggerSettings()


class FLLifecycleSettings(FrozenSettings):
    terminal_status_ttl_seconds: int = Field(default=3600, gt=0)
    tombstone_ttl_seconds: int = Field(default=3600, gt=0)


class FederatedLearningSettings(FrozenSettings):
    workspace_root: Path = Path("data/fl-workspaces")
    workspace_ttl_seconds: int = Field(default=3600, gt=0)
    public_base_url: str = "http://127.0.0.1:9092"
    request_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    artifact_download: ArtifactDownloadSettings = ArtifactDownloadSettings()
    server: FLServerSettings | None = None
    client: FLClientSettings | None = None
    orchestration: OrchestrationSettings | None = None
    strategy: FederatedStrategySettings | None = None
    topology: TopologySettings | None = None
    training_trigger: TrainingTriggerSettings = TrainingTriggerSettings()
    lifecycle: FLLifecycleSettings = FLLifecycleSettings()

    @field_validator("workspace_root", mode="before")
    @classmethod
    def workspace_must_not_be_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("federated_learning.workspace_root must not be blank")
        return value

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, value: str) -> str:
        return _validate_http_base_url(value, "federated_learning.public_base_url")

    @model_validator(mode="after")
    def validate_orchestration_configuration(self) -> "FederatedLearningSettings":
        orchestration = self.orchestration
        degradation_enabled = self.training_trigger.degradation.enabled
        private_api_enabled = self.training_trigger.private_api.enabled
        if orchestration is None:
            if self.server is not None and self.client is None:
                raise ValueError(
                    "server-only federated profile requires autonomous orchestration"
                )
            if self.topology is not None or self.strategy is not None:
                raise ValueError(
                    "topology and strategy require federated_learning.orchestration"
                )
            if degradation_enabled or private_api_enabled:
                raise ValueError(
                    "training triggers require federated_learning.orchestration"
                )
            return self
        if self.server is None:
            raise ValueError("federated_learning.orchestration requires server engine")
        if not degradation_enabled and not private_api_enabled:
            raise ValueError("autonomous orchestration requires at least one training trigger")
        if orchestration.mode == "flat":
            if self.strategy is not None:
                raise ValueError("flat orchestration must not configure hierarchy strategy")
            if orchestration.participant_source == "monitor_scopes":
                if self.topology is not None:
                    raise ValueError("flat monitor_scopes must not configure topology")
                if not degradation_enabled or private_api_enabled:
                    raise ValueError(
                        "flat monitor_scopes requires degradation and forbids private API"
                    )
            elif self.topology is None:
                raise ValueError("flat static orchestration requires topology")
            elif not private_api_enabled:
                raise ValueError("flat static orchestration requires private API trigger")
        else:
            if self.topology is None:
                raise ValueError("hierarchical orchestration requires topology")
            if self.strategy is None:
                raise ValueError("hierarchical orchestration requires strategy")
        return self



class SeedModelSettings(FrozenSettings):
    family_id: str = Field(min_length=1)
    model_id: int = Field(ge=0, le=9223372036854775807)
    artifact_key: str
    event: str = Field(min_length=1)
    event_filter: dict[str, Any] = Field(default_factory=dict)
    target_ue: dict[str, Any] | None = None
    model_interoperability: str = ""
    use_case_context: str = ""

    @field_validator("artifact_key")
    @classmethod
    def validate_artifact_key(cls, value: str) -> str:
        value = value.strip()
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("seed model artifact_key must be lowercase SHA-256 hex")
        return value

    @field_validator("event")
    @classmethod
    def normalize_event(cls, value: str) -> str:
        return value.strip()

    @field_validator("family_id")
    @classmethod
    def normalize_family_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("seed model family_id must not be blank")
        return value


class ModelProvisionSettings(FrozenSettings):
    seed_models: tuple[SeedModelSettings, ...] = ()

    @model_validator(mode="after")
    def validate_unique_models(self) -> "ModelProvisionSettings":
        family_ids = [model.family_id for model in self.seed_models]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("model_provision seed family IDs must be unique")
        model_ids = [model.model_id for model in self.seed_models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model_provision seed model IDs must be unique")
        artifact_keys = [model.artifact_key for model in self.seed_models]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("model_provision seed artifact keys must be unique")
        return self


class NotificationSettings(FrozenSettings):
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.2, ge=0, le=60)
    max_backoff_seconds: float = Field(default=2.0, ge=0, le=300)

    @model_validator(mode="after")
    def validate_backoff(self) -> "NotificationSettings":
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial notification backoff must not exceed maximum backoff")
        return self


class ModelMonitorSettings(FrozenSettings):
    callback_uri: str = "http://127.0.0.1:9092/internal/v1/ml-model-monitor/notifications"
    report_period_seconds: int = Field(default=90, gt=0)
    missed_report_threshold: int = Field(default=2, gt=0, le=20)
    watchdog_grace_seconds: int = Field(default=30, ge=0, le=3600)
    request_timeout_seconds: float = Field(default=30, gt=0, le=300)
    discovery_timeout_seconds: float = Field(default=30, gt=0, le=300)
    retry_interval_seconds: float = Field(default=1, gt=0, le=300)
    retry_max_interval_seconds: float = Field(default=30, gt=0, le=600)

    @field_validator("callback_uri")
    @classmethod
    def validate_callback_uri(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("model_monitor.callback_uri must be an absolute HTTP(S) URI")
        return value

    @model_validator(mode="after")
    def validate_retry(self) -> "ModelMonitorSettings":
        if self.retry_interval_seconds > self.retry_max_interval_seconds:
            raise ValueError("model monitor initial retry must not exceed maximum")
        return self


class AccuracyPolicySettings(FrozenSettings):
    enabled: bool = True
    reference_buffer_size: int = Field(default=12, gt=0)
    min_reference_samples: int = Field(default=5, gt=0)
    min_std: float = Field(default=0.14, gt=0)
    fixed_floor: float = Field(default=0.05, ge=0)
    z_score_threshold: float = Field(default=1.3, gt=0)
    decision_window_size: int = Field(default=5, gt=0)
    required_hits: int = Field(default=3, gt=0)
    scope_state_ttl_seconds: int = Field(default=600, gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> "AccuracyPolicySettings":
        if self.required_hits > self.decision_window_size:
            object.__setattr__(self, "required_hits", self.decision_window_size)
        if self.min_reference_samples > self.reference_buffer_size:
            raise ValueError(
                "accuracy_policy.min_reference_samples must not exceed reference_buffer_size"
            )
        return self


class AdrfSettings(FrozenSettings):
    mode: str = "nrf"
    configured_endpoint: str = ""
    configured_nf_instance_id: str = ""
    discovery_timeout_seconds: float = Field(default=30, gt=0, le=300)
    request_timeout_seconds: float = Field(default=120, gt=0, le=600)
    retry_initial_backoff_seconds: float = Field(default=2, ge=0, le=60)
    retry_max_backoff_seconds: float = Field(default=30, gt=0, le=600)

    @model_validator(mode="after")
    def validate_mode(self) -> "AdrfSettings":
        mode = self.mode.strip().lower()
        object.__setattr__(self, "mode", mode)
        if mode not in {"nrf", "configured"}:
            raise ValueError("adrf.mode must be 'nrf' or 'configured'")
        if mode == "configured":
            object.__setattr__(
                self,
                "configured_endpoint",
                _validate_http_base_url(
                    self.configured_endpoint,
                    "adrf.configured_endpoint",
                ),
            )
            if not self.configured_nf_instance_id.strip():
                raise ValueError("adrf.configured_nf_instance_id is required in configured mode")
        if self.retry_initial_backoff_seconds > self.retry_max_backoff_seconds:
            raise ValueError("ADRF initial retry backoff must not exceed maximum")
        return self


class MongoDatasetSettings(FrozenSettings):
    url: str = "mongodb://127.0.0.1:27017"
    database: str = "free5gc"
    collection: str = "nwdaf_raw_notifications"
    connect_timeout_ms: int = Field(default=5000, gt=0)
    read_timeout_ms: int = Field(default=30000, gt=0)


class DatasetSettings(FrozenSettings):
    retrieval_window_seconds: int = Field(default=1800, gt=0)
    watchdog_timeout_seconds: int = Field(default=120, gt=0)
    fetch_timeout_seconds: float = Field(default=120, gt=0, le=600)
    max_redirects: int = Field(default=3, ge=0, le=10)
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_backoff_seconds: float = Field(default=1, ge=0, le=60)
    retry_max_backoff_seconds: float = Field(default=30, gt=0, le=600)
    max_concurrent_jobs: int = Field(default=2, gt=0, le=32)
    max_records_per_job: int = Field(default=100000, gt=0)
    mongodb: MongoDatasetSettings = MongoDatasetSettings()


class LogSettings(FrozenSettings):
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return value


class Settings(FrozenSettings):
    server: ServerSettings = ServerSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    containing_nwdaf: ContainingNwdafSettings = ContainingNwdafSettings()
    storage: StorageSettings = StorageSettings()
    model_state: ModelStateSettings = ModelStateSettings()
    publication: PublicationSettings = PublicationSettings()
    cutover: CutoverSettings = CutoverSettings()
    artifact: ArtifactSettings = ArtifactSettings()
    federated_learning: FederatedLearningSettings = FederatedLearningSettings()
    model_provision: ModelProvisionSettings = ModelProvisionSettings()
    model_monitor: ModelMonitorSettings = ModelMonitorSettings()
    accuracy_policy: AccuracyPolicySettings = AccuracyPolicySettings()
    adrf: AdrfSettings = AdrfSettings()
    dataset: DatasetSettings = DatasetSettings()
    local_training: LocalTrainingSettings | None = None
    notification: NotificationSettings = NotificationSettings()
    log: LogSettings = LogSettings()

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> "Settings":
        mode = self.runtime.mode
        if mode == "local":
            if (
                self.federated_learning.server is not None
                or self.federated_learning.client is not None
            ):
                raise ValueError("local mode must not configure FL server or client settings")
            if self.local_training is None:
                object.__setattr__(self, "local_training", LocalTrainingSettings())
        elif mode == "federated":
            server = self.federated_learning.server
            client = self.federated_learning.client
            if server is None and client is None:
                raise ValueError(
                    "federated mode requires federated_learning.server or "
                    "federated_learning.client"
                )
            if self.local_training is not None:
                raise ValueError("federated mode must not configure local training settings")
            if server is not None and server.max_active_processes != 1:
                raise ValueError(
                    "federated_learning.server.max_active_processes must be 1"
                )
        workspace_root = self.federated_learning.workspace_root.resolve()
        repository_root = Path(__file__).resolve().parents[2]
        if (
            workspace_root == Path(workspace_root.anchor)
            or Path.cwd().resolve().is_relative_to(workspace_root)
            or repository_root.is_relative_to(workspace_root)
        ):
            raise ValueError("federated_learning.workspace_root is unsafe")
        durable_roots = (
            self.storage.artifact_root.resolve(),
            self.model_state.directory.resolve(),
            self.publication.directory.resolve(),
        )
        if any(_paths_overlap(workspace_root, durable_root) for durable_root in durable_roots):
            raise ValueError(
                "federated_learning.workspace_root must not overlap durable storage"
            )
        return self


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if isinstance(raw, dict):
        federated_learning = raw.get("federated_learning")
        if isinstance(federated_learning, dict):
            topology = federated_learning.get("topology")
            if isinstance(topology, dict):
                topology_path = topology.get("config_file")
                if isinstance(topology_path, str) and topology_path.strip():
                    candidate = Path(topology_path)
                    if not candidate.is_absolute():
                        topology["config_file"] = str(
                            (config_path.parent / candidate).resolve()
                        )
            client = federated_learning.get("client")
            if isinstance(client, dict):
                training_data = client.get("training_data")
                if isinstance(training_data, dict):
                    state_directory = training_data.get("state_directory")
                    if isinstance(state_directory, str) and state_directory.strip():
                        candidate = Path(state_directory)
                        if not candidate.is_absolute():
                            training_data["state_directory"] = str(
                                (config_path.parent / candidate).resolve()
                            )
                    shard_path = training_data.get("shard_path")
                    if isinstance(shard_path, str) and shard_path.strip():
                        candidate = Path(shard_path)
                        if not candidate.is_absolute():
                            training_data["shard_path"] = str(
                                (config_path.parent / candidate).resolve()
                            )
    return Settings.model_validate(raw)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)
