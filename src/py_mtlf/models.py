import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from py_mtlf.wire.ml_model import MLModelProvisionSnapshot
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistrationSnapshot,
    MLModelMonitorSubscriptionSnapshot,
)
from py_mtlf.wire.ml_model_training import MLModelTrainingSubscriptionSnapshot

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


class TargetUeInformation(SpecAlignedModel):
    any_ue: bool = Field(default=False, alias="anyUe")
    supis: list[str] = Field(default_factory=list)
    int_group_ids: list[str] = Field(default_factory=list, alias="intGroupIds")


class NwdafEventSubscription(SpecAlignedModel):
    event: str = Field(min_length=1)
    tgt_ue: TargetUeInformation | None = Field(default=None, alias="tgtUe")


class ReportingInformation(SpecAlignedModel):
    imm_rep: bool = Field(default=False, alias="immRep")
    notif_method: str = Field(default="", alias="notifMethod")
    max_report_nbr: int = Field(default=0, alias="maxReportNbr", ge=0)
    mon_dur: datetime | None = Field(default=None, alias="monDur")
    rep_period: int = Field(default=0, alias="repPeriod", ge=0)


class NnwdafEventsSubscription(SpecAlignedModel):
    event_subscriptions: list[NwdafEventSubscription] = Field(
        min_length=1,
        alias="eventSubscriptions",
    )
    evt_req: ReportingInformation | None = Field(default=None, alias="evtReq")
    notification_uri: str = Field(default="", alias="notificationURI")
    notif_corr_id: str = Field(default="", alias="notifCorrId")


class NwdafIdentity(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    nf_instance_id: str = Field(alias="nfInstanceId")
    api_base_uri: str = Field(alias="apiBaseUri")
    internal_callback_base_uri: str = Field(alias="internalCallbackBaseUri")


class EventsSubscriptionSnapshot(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    subscription_id: str = Field(min_length=1, alias="subscriptionId")
    subscription: NnwdafEventsSubscription
    external_notification_uri: str = Field(alias="externalNotificationUri")


class SmfResourceSnapshot(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    correlation_id: str = Field(min_length=1, alias="correlationId")
    resource_location: str = Field(min_length=1, alias="resourceLocation")
    target_api_root: str = Field(min_length=1, alias="targetApiRoot")
    nwdaf_subscription_ids: list[str] = Field(alias="nwdafSubscriptionIds")
    pending_cleanup: bool = Field(alias="pendingCleanup")
    subscription: dict[str, Any] | None = None


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
    adrf_instance_id: str = Field(min_length=1, alias="adrfInstanceId")
    retain_until: datetime = Field(alias="retainUntil")


class BackendSyncRequest(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    containing_nwdaf: NwdafIdentity = Field(alias="containingNwdaf")
    events_subscriptions: list[EventsSubscriptionSnapshot] = Field(alias="eventsSubscriptions")
    smf_resources: list[SmfResourceSnapshot] = Field(alias="smfResources")
    training_data_descriptors: list[TrainingDataDescriptor] = Field(
        default_factory=list,
        alias="trainingDataDescriptors",
    )
    training_data_source: Literal["adrf", "mongodb", "unavailable"] = Field(
        default="unavailable",
        alias="trainingDataSource",
    )
    ml_model_provision_subscriptions: list[MLModelProvisionSnapshot] = Field(
        default_factory=list,
        alias="mlModelProvisionSubscriptions",
    )
    ml_model_monitor_registrations: list[MLModelMonitorRegistrationSnapshot] = Field(
        default_factory=list,
        alias="mlModelMonitorRegistrations",
    )
    ml_model_monitor_subscriptions: list[MLModelMonitorSubscriptionSnapshot] = Field(
        default_factory=list,
        alias="mlModelMonitorSubscriptions",
    )
    ml_model_training_subscriptions: list[MLModelTrainingSubscriptionSnapshot] = Field(
        default_factory=list,
        alias="mlModelTrainingSubscriptions",
    )


class BackendSyncResponse(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    process_instance_id: str = Field(alias="processInstanceId")
    snapshot_accepted: bool = Field(alias="snapshotAccepted")
