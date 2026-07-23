from datetime import datetime
from uuid import UUID

from pydantic import AnyHttpUrl, Field, JsonValue, field_validator, model_validator

from py_mtlf.wire.ml_model import StandardModel


class MLModelMonitorRegistration(StandardModel):
    consumer_id: str = Field(default="", alias="consumerId")
    consumer_set_id: str = Field(default="", alias="consumerSetId")
    model_id: int = Field(alias="modelId", ge=0)
    model_accuracy_indication: bool | None = Field(default=None, alias="modelAccuInd")
    ml_event: str = Field(default="", alias="mLEvent")
    ml_event_filter: dict[str, JsonValue] | None = Field(default=None, alias="mLEventFilter")
    target_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtUe")

    @model_validator(mode="after")
    def validate_consumer(self) -> "MLModelMonitorRegistration":
        if bool(self.consumer_id.strip()) == bool(self.consumer_set_id.strip()):
            raise ValueError("exactly one of consumerId or consumerSetId is required")
        if self.consumer_id:
            parsed = UUID(self.consumer_id)
            if parsed.version != 4:
                raise ValueError("consumerId must be a UUIDv4 NF instance ID")
        return self


class MonitorReportingRequirement(StandardModel):
    notification_method: str = Field(default="PERIODIC", alias="notifMethod")
    repetition_period: int = Field(default=90, alias="repPeriod", gt=0)
    immediate_report: bool = Field(default=False, alias="immRep")


class MLModelMonitorSubscription(StandardModel):
    model_ids: list[int] = Field(min_length=1, alias="modelIds")
    notification_uri: AnyHttpUrl = Field(alias="notificationUri")
    notification_id: str = Field(min_length=1, alias="notifCorrId")
    model_metric: str | None = Field(default=None, alias="modelMetric")
    accuracy_threshold: int | None = Field(default=None, alias="accuThreshold", ge=0)
    event_report_request: MonitorReportingRequirement | None = Field(
        default=None,
        alias="eventReportReq",
    )
    immediate_report: "MLModelMonitorNotification | None" = Field(
        default=None,
        alias="immReport",
    )
    ml_event: str = Field(default="", alias="mLEvent")
    ml_event_filter: dict[str, JsonValue] | None = Field(default=None, alias="mLEventFilter")
    target_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtUe")

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("modelIds must be non-negative")
        return values


class MLModelAccuracyInfo(StandardModel):
    model_id: int = Field(alias="modelId", ge=0)
    deviation: float | None = None
    inference_count: int | None = Field(default=None, alias="inferenceNum", ge=0)
    model_metric: str | None = Field(default=None, alias="modelMetric")
    monitor_interval: dict[str, JsonValue] | None = Field(
        default=None,
        alias="monitorInterval",
    )


class MLModelMonitorNotification(StandardModel):
    notification_id: str = Field(min_length=1, alias="notifCorrId")
    model_accuracy_info: list[MLModelAccuracyInfo] = Field(
        default_factory=list,
        alias="modelAccuInfos",
    )
    analytics_feedback: list[dict[str, JsonValue]] = Field(
        default_factory=list,
        alias="anaFeedbacks",
    )
    accuracy_met: bool | None = Field(default=None, alias="accuMeetInd")
    ml_event: str = Field(default="", alias="mLEvent")
    ml_event_filter: dict[str, JsonValue] | None = Field(default=None, alias="mLEventFilter")
    target_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtUe")

    @model_validator(mode="after")
    def validate_report(self) -> "MLModelMonitorNotification":
        if not self.model_accuracy_info and not self.analytics_feedback:
            raise ValueError("modelAccuInfos or anaFeedbacks must be non-empty")
        return self


class MLModelMonitorRegistrationSnapshot(StandardModel):
    registration_id: str = Field(min_length=1, alias="registrationId")
    representation: MLModelMonitorRegistration
    initiator: str


class MLModelMonitorSubscriptionSnapshot(StandardModel):
    subscription_id: str = Field(min_length=1, alias="subscriptionId")
    representation: MLModelMonitorSubscription
    destination: str
    owner_registration_id: str = Field(min_length=1, alias="ownerRegistrationId")


class MonitorInterval(StandardModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")


MLModelMonitorSubscription.model_rebuild()
