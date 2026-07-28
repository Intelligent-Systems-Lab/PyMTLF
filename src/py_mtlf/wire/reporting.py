from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StandardModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class MutingExceptionInstructions(StandardModel):
    buffered_notifications: str | None = Field(default=None, alias="bufferedNotifs")
    subscription: str | None = None


class MutingNotificationsSettings(StandardModel):
    maximum_notification_count: int | None = Field(
        default=None,
        alias="maxNoOfNotif",
        ge=0,
    )
    buffered_notification_duration: int | None = Field(
        default=None,
        alias="durationBufferedNotif",
        ge=0,
    )


class ReportingInformation(StandardModel):
    immediate_report: bool | None = Field(default=None, alias="immRep")
    notification_method: str | None = Field(default=None, alias="notifMethod")
    maximum_report_count: int | None = Field(default=None, alias="maxReportNbr", ge=0)
    monitoring_duration: datetime | None = Field(default=None, alias="monDur")
    repetition_period: int | None = Field(default=None, alias="repPeriod", ge=0)
    sampling_ratio: int | None = Field(default=None, alias="sampRatio", ge=0, le=100)
    partition_criteria: list[str] | None = Field(
        default=None,
        alias="partitionCriteria",
        min_length=1,
    )
    group_reporting_time: int | None = Field(default=None, alias="grpRepTime", ge=0)
    notification_flag: str | None = Field(default=None, alias="notifFlag")
    notification_flag_instruction: MutingExceptionInstructions | None = Field(
        default=None,
        alias="notifFlagInstruct",
    )
    muting_setting: MutingNotificationsSettings | None = Field(
        default=None,
        alias="mutingSetting",
    )

    @field_validator("monitoring_duration")
    @classmethod
    def monitoring_duration_requires_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("monDur must include a timezone")
        return value
