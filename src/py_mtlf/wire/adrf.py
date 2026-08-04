from datetime import datetime

from pydantic import Field, model_validator

from py_mtlf.models import SpecAlignedModel


class TimeWindow(SpecAlignedModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")

    @model_validator(mode="after")
    def validate_order(self) -> "TimeWindow":
        if self.start_time > self.stop_time:
            raise ValueError("timePeriod startTime must not be after stopTime")
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


class MLModelAddress(SpecAlignedModel):
    model_url: str | None = Field(default=None, alias="mLModelUrl")
    file_fqdn: str | None = Field(default=None, alias="mlFileFqdn")

    @model_validator(mode="after")
    def validate_address(self) -> "MLModelAddress":
        if bool(self.model_url) == bool(self.file_fqdn):
            raise ValueError("exactly one of mLModelUrl or mlFileFqdn is required")
        return self


class AllowedConsumer(SpecAlignedModel):
    nf_instance_id: str | None = Field(default=None, alias="nfInstanceId")
    nf_set_id: str | None = Field(default=None, alias="nfSetId")

    @model_validator(mode="after")
    def validate_owner(self) -> "AllowedConsumer":
        if bool(self.nf_instance_id) == bool(self.nf_set_id):
            raise ValueError("exactly one of nfInstanceId or nfSetId is required")
        return self


class MLModelInfo(SpecAlignedModel):
    model_unique_id: int = Field(ge=0, alias="modelUniqueId")
    model_file_address: MLModelAddress = Field(alias="mlFileAddr")
    model_storage_size: int = Field(ge=0, alias="mlStorageSize")
    allowed_consumers: list[AllowedConsumer] = Field(
        default_factory=list,
        alias="allowConsumerList",
    )


class ModelStoreResult(SpecAlignedModel):
    model_unique_id: int = Field(ge=0, alias="modelUniqueId")
    store_result: str = Field(min_length=1, alias="storeResult")


class NadrfMLModelStoreRecord(SpecAlignedModel):
    nf_instance_id: str | None = Field(default=None, alias="nfInstanceId")
    nf_set_id: str | None = Field(default=None, alias="nfSetId")
    ml_model_info: list[MLModelInfo] = Field(
        default_factory=list,
        min_length=1,
        alias="mlModelInfo",
    )
    model_store_result: ModelStoreResult | None = Field(
        default=None,
        alias="modelStoreResult",
    )
    supported_features: str | None = Field(default=None, alias="suppFeat")

    @model_validator(mode="after")
    def validate_record(self) -> "NadrfMLModelStoreRecord":
        if bool(self.nf_instance_id) == bool(self.nf_set_id):
            raise ValueError("exactly one of nfInstanceId or nfSetId is required")
        if len(self.ml_model_info) != 1:
            raise ValueError("the current profile requires exactly one URL-backed model")
        return self
