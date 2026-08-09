import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentity(DomainModel):
    model_unique_id: int = Field(ge=0, le=9223372036854775807)


class PrivateError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    correlation_id: str = ""


class SpecAlignedModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class DescriptorSmfDataSubscription(SpecAlignedModel):
    supi: str = Field(min_length=1)
    pdu_session_id: int | None = Field(default=None, alias="pduSeId", ge=0, le=255)
    dnn: str = ""
    snssai: dict[str, Any] | None = None
    nf_id: str = Field(default="", alias="nfId")
    notif_id: str = Field(min_length=1, alias="notifId")
    notif_uri: str = Field(min_length=1, alias="notifUri")
    event_subs: list[dict[str, Any]] = Field(min_length=1, alias="eventSubs")


class DescriptorDataSubscription(SpecAlignedModel):
    smf_data_sub: DescriptorSmfDataSubscription = Field(alias="smfDataSub")


class DescriptorTimeWindow(SpecAlignedModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")


class NadrfStoredDataSpec(SpecAlignedModel):
    data_spec: DescriptorDataSubscription = Field(alias="dataSpec")
    time_period: DescriptorTimeWindow = Field(alias="timePeriod")


class DescriptorMLEventSubscription(SpecAlignedModel):
    ml_event: str = Field(min_length=1, alias="mLEvent")
    ml_event_filter: dict[str, Any] | None = Field(default=None, alias="mLEventFilter")
    target_ue: dict[str, Any] | None = Field(default=None, alias="tgtUe")


class TrainingDataDescriptor(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    correlation_id: str = Field(min_length=1, alias="correlationId")
    state: Literal["ACTIVE", "RETAINED"]
    stored_data_spec: NadrfStoredDataSpec = Field(alias="storedDataSpec")
    ml_event_subscription: DescriptorMLEventSubscription = Field(alias="mlEventSubscription")
    source_nf_instance_id: str = Field(min_length=1, alias="sourceNfInstanceId")
    adrf_instance_id: str | None = Field(default=None, alias="adrfInstanceId")
    retain_until: datetime = Field(alias="retainUntil")
