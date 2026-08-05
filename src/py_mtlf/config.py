from pathlib import Path
from typing import Any
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
        if value not in {"local", "fl_server", "fl_client"}:
            raise ValueError("runtime.mode must be 'local', 'fl_server', or 'fl_client'")
        return value


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


class FittingSettings(FrozenSettings):
    device: str = "cpu"
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=18, gt=0)
    validation_ratio: float = Field(default=0.10, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "cpu":
            raise ValueError("training.device must be 'cpu' in the current implementation")
        return value


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


class FLClientSettings(FrozenSettings):
    callback_deadline_margin_seconds: int = Field(default=5, ge=1, le=300)
    callback_queue_size: int = Field(default=256, gt=0, le=100000)
    max_concurrent_jobs: int = Field(default=2, gt=0, le=32)
    model_interoperability_ids: tuple[str, ...] = ()
    fallback_deadlines: FallbackDeadlineSettings = FallbackDeadlineSettings()
    training: FittingSettings = FittingSettings()

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
        return self


class DelayPolicySettings(FrozenSettings):
    max_extensions: int = Field(default=1, ge=0, le=10)
    max_extension_seconds: int = Field(default=300, ge=0, le=86400)


class CleanupSettings(FrozenSettings):
    max_attempts: int = Field(default=3, gt=0, le=20)
    retry_backoff_seconds: float = Field(default=0.2, ge=0, le=60)


class FLServerSettings(FrozenSettings):
    callback_uri: str = "http://127.0.0.1:9092/internal/v1/ml-model-training/notifications"
    preparation_timeout_seconds: int = Field(default=300, gt=0, le=86400)
    preparation_data_window_seconds: int = Field(default=3600, gt=0, le=604800)
    round_timeout_seconds: int = Field(default=300, gt=0, le=86400)
    round_count: int = Field(default=2, ge=1, le=100)
    max_active_processes: int = Field(default=1, gt=0, le=32)
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


class FederatedLearningSettings(FrozenSettings):
    workspace_root: Path = Path("data/fl-workspaces")
    workspace_ttl_seconds: int = Field(default=3600, gt=0)
    public_base_url: str = "http://127.0.0.1:9092"
    request_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    artifact_download: ArtifactDownloadSettings = ArtifactDownloadSettings()
    server: FLServerSettings | None = None
    client: FLClientSettings | None = None

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
        elif mode == "fl_server":
            if self.federated_learning.server is None:
                raise ValueError("fl_server mode requires federated_learning.server")
            if self.federated_learning.client is not None or self.local_training is not None:
                raise ValueError(
                    "fl_server mode must not configure client or local training settings"
                )
        elif mode == "fl_client":
            if self.federated_learning.client is None:
                raise ValueError("fl_client mode requires federated_learning.client")
            if self.federated_learning.server is not None or self.local_training is not None:
                raise ValueError(
                    "fl_client mode must not configure server or local training settings"
                )
        return self


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return Settings.model_validate(raw)
