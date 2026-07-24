from datetime import datetime

from pydantic import Field, model_validator

from py_mtlf.models import SpecAlignedModel


class TimeWindow(SpecAlignedModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")

    @model_validator(mode="after")
    def validate_order(self) -> "TimeWindow":
        if self.start_time >= self.stop_time:
            raise ValueError("timePeriod startTime must be before stopTime")
        return self


class DataSubscription(SpecAlignedModel):
    smf_data_sub: dict = Field(alias="smfDataSub")


class NadrfDataRetrievalSubscription(SpecAlignedModel):
    notif_corr_id: str = Field(min_length=1, alias="notifCorrId")
    notification_uri: str = Field(min_length=1, alias="notificationURI")
    time_period: TimeWindow = Field(alias="timePeriod")
    data_sub: DataSubscription = Field(alias="dataSub")
    cons_trig_notif: bool = Field(default=True, alias="consTrigNotif")


class FetchInstruction(SpecAlignedModel):
    fetch_uri: str = Field(min_length=1, alias="fetchUri")
    fetch_corr_ids: list[str] = Field(alias="fetchCorrIds")
    expiry: datetime | None = None


class DataNotification(SpecAlignedModel):
    upf_event_notifs: list[dict] = Field(default_factory=list, alias="upfEventNotifs")
    smf_event_notifs: list[dict] = Field(default_factory=list, alias="smfEventNotifs")


class NadrfDataRetrievalNotification(SpecAlignedModel):
    notif_corr_id: str = Field(min_length=1, alias="notifCorrId")
    time_stamp: datetime = Field(alias="timeStamp")
    fetch_instruct: FetchInstruction | None = Field(default=None, alias="fetchInstruct")
    data_notif: DataNotification | None = Field(default=None, alias="dataNotif")
    ana_notifications: list[dict] = Field(default_factory=list, alias="anaNotifications")
    termination_req: bool = Field(default=False, alias="terminationReq")

    @model_validator(mode="after")
    def validate_alternative(self) -> "NadrfDataRetrievalNotification":
        alternatives = sum(
            (
                self.fetch_instruct is not None,
                self.data_notif is not None,
                bool(self.ana_notifications),
            )
        )
        if alternatives != 1:
            raise ValueError("exactly one retrieval notification alternative is required")
        if (
            self.fetch_instruct is not None
            and not self.fetch_instruct.fetch_corr_ids
            and not self.termination_req
        ):
            raise ValueError("empty fetchCorrIds is allowed only for terminal V0 callbacks")
        return self


class NadrfDataStoreRecord(SpecAlignedModel):
    data_sub: list[DataSubscription] = Field(min_length=1, alias="dataSub")
    data_notif: DataNotification = Field(alias="dataNotif")

    @model_validator(mode="after")
    def validate_data(self) -> "NadrfDataStoreRecord":
        if len(self.data_sub) != 1:
            raise ValueError("the current retrieval profile requires one dataSub")
        if bool(self.data_notif.upf_event_notifs) == bool(self.data_notif.smf_event_notifs):
            raise ValueError("exactly one non-empty data notification alternative is required")
        return self
