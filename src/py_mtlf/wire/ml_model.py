from typing import Any

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, JsonValue, model_validator


class StandardModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ReportingInformation(StandardModel):
    immediate_report: bool = Field(default=False, alias="immRep")


class MLEventSubscription(StandardModel):
    ml_event: str = Field(min_length=1, alias="mLEvent")
    ml_event_filter: dict[str, JsonValue] = Field(alias="mLEventFilter")
    target_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtUe")
    model_interoperability: str = Field(default="", alias="modelInterInfo")
    use_case_context: str = Field(default="", alias="useCaseCxt")
    model_id: int | None = Field(default=None, alias="modelId", ge=0)


class MLModelAddress(StandardModel):
    model_url: AnyHttpUrl | None = Field(default=None, alias="mLModelUrl")
    file_fqdn: str | None = Field(default=None, alias="mlFileFqdn")

    @model_validator(mode="after")
    def validate_address(self) -> "MLModelAddress":
        if (self.model_url is None) == (
            self.file_fqdn is None or not self.file_fqdn.strip()
        ):
            raise ValueError("exactly one of mLModelUrl or mlFileFqdn is required")
        return self


class MLEventNotification(StandardModel):
    event: str = Field(min_length=1)
    notification_correlation_id: str | None = Field(default=None, alias="notifCorreId")
    model_file_address: MLModelAddress | None = Field(default=None, alias="mLFileAddr")
    model_adrf: dict[str, JsonValue] | None = Field(default=None, alias="mLModelAdrf")
    model_unique_id: int | None = Field(default=None, alias="modelUniqueId", ge=0)
    use_case_context: str | None = Field(default=None, alias="useCaseCxt")
    ml_event_filter: dict[str, JsonValue] | None = Field(default=None, alias="mLEventFilter")
    target_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtUe")

    @model_validator(mode="after")
    def validate_delivery(self) -> "MLEventNotification":
        if (self.model_file_address is None) == (self.model_adrf is None):
            raise ValueError("exactly one of mLFileAddr or mLModelAdrf is required")
        return self


class MLModelProvisionSubscription(StandardModel):
    ml_event_subscriptions: list[MLEventSubscription] = Field(
        min_length=1,
        alias="mLEventSubscs",
    )
    notification_uri: AnyHttpUrl = Field(alias="notifUri")
    ml_event_notifications: list[MLEventNotification] | None = Field(
        default=None,
        min_length=1,
        alias="mLEventNotifs",
    )
    notification_correlation_id: str = Field(default="", alias="notifCorreId")
    event_request: ReportingInformation | None = Field(default=None, alias="eventReq")
    failure_event_reports: list[JsonValue] | None = Field(
        default=None,
        min_length=1,
        alias="failEventReports",
    )


class MLModelProvisionNotification(StandardModel):
    event_notifications: list[MLEventNotification] = Field(min_length=1, alias="eventNotifs")
    subscription_id: str = Field(min_length=1, alias="subscriptionId")


class MLModelProvisionSnapshot(StandardModel):
    subscription_id: str = Field(min_length=1, alias="subscriptionId")
    representation: MLModelProvisionSubscription
    initiator: str
    destination: str


class ProblemDetails(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str = ""
    title: str = ""
    status: int
    detail: str = ""
    instance: str = ""
    cause: str = ""
    invalid_params: list[dict[str, Any]] = Field(default_factory=list, alias="invalidParams")
