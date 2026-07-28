from dataclasses import dataclass
from datetime import datetime

from pydantic import AnyHttpUrl, Field, JsonValue, model_validator

from py_mtlf.wire.ml_model import MLEventNotification, MLEventSubscription
from py_mtlf.wire.reporting import ReportingInformation, StandardModel


class TimeWindow(StandardModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")

    @model_validator(mode="after")
    def validate_window(self) -> "TimeWindow":
        for name, value in (("startTime", self.start_time), ("stopTime", self.stop_time)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone")
        if self.stop_time < self.start_time:
            raise ValueError("stopTime must not precede startTime")
        return self


class DCCFEvent(StandardModel):
    nwdaf_event: str | None = Field(default=None, alias="nwdafEvent")
    smf_event: str | None = Field(default=None, alias="smfEvent")
    amf_event: str | None = Field(default=None, alias="amfEvent")
    nef_event: str | None = Field(default=None, alias="nefEvent")
    udm_event: str | None = Field(default=None, alias="udmEvent")
    af_event: str | None = Field(default=None, alias="afEvent")
    sac_event: JsonValue | None = Field(default=None, alias="sacEvent")
    nrf_event: str | None = Field(default=None, alias="nrfEvent")
    gmlc_event: str | None = Field(default=None, alias="gmlcEvent")
    upf_event: str | None = Field(default=None, alias="upfEvent")

    @model_validator(mode="after")
    def validate_union(self) -> "DCCFEvent":
        values = (
            self.nwdaf_event,
            self.smf_event,
            self.amf_event,
            self.nef_event,
            self.udm_event,
            self.af_event,
            self.sac_event,
            self.nrf_event,
            self.gmlc_event,
            self.upf_event,
        )
        if sum(value is not None for value in values) != 1:
            raise ValueError("exactly one DCCF event member is required")
        return self


class DataAvReq(StandardModel):
    data_statistical_properties: list[str] | None = Field(
        default=None,
        alias="dataStatProps",
        min_length=1,
    )
    input_events: list[DCCFEvent] = Field(alias="inpEvents", min_length=1)
    minimum_sample_count: int | None = Field(default=None, alias="minNumSamples", ge=0)
    time_windows: list[TimeWindow] | None = Field(
        default=None,
        alias="timeWindows",
        min_length=1,
    )


class MLModelTrainInfo(StandardModel):
    data_availability_requirement: DataAvReq | None = Field(default=None, alias="dataAvReq")
    time_availability_requirement: str | None = Field(default=None, alias="timeAvReq")


class MLTrainReportInfo(StandardModel):
    maximum_response_time: int | None = Field(default=None, alias="maxResTime", ge=0)


class FailureEventInfoForMLModelTrain(StandardModel):
    ml_training_event: str = Field(min_length=1, alias="mLTrainEvent")
    failure_code: str = Field(min_length=1, alias="failureCodeTrain")


class DelayEventNotif(StandardModel):
    delay_event_indicator: bool = Field(alias="delayEventInd")
    delay_cause: str | None = Field(default=None, alias="delayCause")
    expected_completion_time: int | None = Field(default=None, alias="expCompTime", ge=0)


class TrainDataInfo(StandardModel):
    area_information: dict[str, JsonValue] | None = Field(default=None, alias="areaInfo")
    maximum_values: list[str] | None = Field(default=None, alias="maxValues", min_length=1)
    minimum_values: list[str] | None = Field(default=None, alias="minValues", min_length=1)
    sampling_ratio: int | None = Field(default=None, alias="samplRatio", ge=0)


class StatusReportInfo(StandardModel):
    ml_model_accuracy: int | None = Field(default=None, alias="mlModelAcc", ge=0, le=100)
    training_data_info: TrainDataInfo | None = Field(default=None, alias="trainInDataInfo")


class NwdafMLModelTrainNotif(StandardModel):
    delay_event_notification: DelayEventNotif | None = Field(
        default=None,
        alias="delayEventNotif",
    )
    ml_correlation_id: str | None = Field(default=None, alias="mlCorreId")
    ml_model_infos: list[MLEventNotification] | None = Field(
        default=None,
        alias="mLModelInfos",
        min_length=1,
    )
    notification_correlation_id: str = Field(min_length=1, alias="notifCorreId")
    round_indicator: int | None = Field(default=None, alias="roundInd", ge=0)
    status_report: StatusReportInfo | None = Field(default=None, alias="statusReport")
    termination_request: str | None = Field(default=None, alias="termTrainReq")

    @model_validator(mode="after")
    def validate_result_combination(self) -> "NwdafMLModelTrainNotif":
        has_delay = self.delay_event_notification is not None
        has_models = bool(self.ml_model_infos)
        has_termination = bool(self.termination_request)
        if not has_delay and not has_models and not has_termination:
            raise ValueError(
                "at least one of delayEventNotif, mLModelInfos or termTrainReq is required"
            )
        if has_delay and (has_models or has_termination):
            raise ValueError("delayEventNotif cannot coexist with mLModelInfos or termTrainReq")
        return self


class NwdafMLModelTrainSubsc(StandardModel):
    ml_event_subscriptions: list[MLEventSubscription] = Field(
        alias="mLEventSubscs",
        min_length=1,
    )
    notification_uri: AnyHttpUrl = Field(alias="notifUri")
    supported_features: str | None = Field(default=None, alias="suppFeats")
    event_request: ReportingInformation | None = Field(default=None, alias="eventReq")
    failure_event_reports: list[FailureEventInfoForMLModelTrain] | None = Field(
        default=None,
        alias="failEventReports",
        min_length=1,
    )
    ml_correlation_id: str | None = Field(default=None, alias="mlCorreId")
    ml_model_infos: list[MLEventNotification] | None = Field(
        default=None,
        alias="mLModelInfos",
        min_length=1,
    )
    immediate_report: NwdafMLModelTrainNotif | None = Field(default=None, alias="immReport")
    ml_model_training_infos: list[MLModelTrainInfo] | None = Field(
        default=None,
        alias="mLModelTrainInfos",
        min_length=1,
    )
    ml_preparation_flag: bool | None = Field(default=None, alias="mLPreFlag")
    ml_accuracy_check_flag: bool | None = Field(default=None, alias="mLAccChkFlg")
    ml_training_report_info: MLTrainReportInfo | None = Field(
        default=None,
        alias="mLTrainRepInfo",
    )
    notification_correlation_id: str = Field(min_length=1, alias="notifCorreId")
    round_indicator: int | None = Field(default=None, alias="roundInd", ge=0)
    target_reporting_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtRepUe")
    skip_fl_indicator: bool | None = Field(default=None, alias="skipFlInd")


class NwdafMLModelTrainSubscPatch(StandardModel):
    notification_uri: AnyHttpUrl | None = Field(default=None, alias="notifUri")
    event_request: ReportingInformation | None = Field(default=None, alias="eventReq")
    ml_model_infos: list[MLEventNotification] | None = Field(
        default=None,
        alias="mLModelInfos",
        min_length=1,
    )
    ml_model_training_infos: list[MLModelTrainInfo] | None = Field(
        default=None,
        alias="mLModelTrainInfos",
        min_length=1,
    )
    ml_preparation_flag: bool | None = Field(default=None, alias="mLPreFlag")
    ml_accuracy_check_flag: bool | None = Field(default=None, alias="mLAccChkFlg")
    ml_training_report_info: MLTrainReportInfo | None = Field(
        default=None,
        alias="mLTrainRepInfo",
    )
    round_indicator: int | None = Field(default=None, alias="roundInd", ge=0)
    target_reporting_ue: dict[str, JsonValue] | None = Field(default=None, alias="tgtRepUe")
    skip_fl_indicator: bool | None = Field(default=None, alias="skipFlInd")


@dataclass(frozen=True)
class TrainingResourceIdentity:
    subscription_id: str
    ml_correlation_id: str
    notification_correlation_id: str
    expected_round_indicator: int | None = None
    notification_method: str | None = None


@dataclass(frozen=True)
class InvalidParameter:
    parameter: str
    reason: str


class RequirementsError(ValueError):
    cause = "ML_MODEL_TRAINING_REQS_NOT_MET"

    def __init__(self, violations: list[InvalidParameter]) -> None:
        super().__init__(self.cause)
        self.violations = tuple(violations)


def validate_fl_subscription(
    value: NwdafMLModelTrainSubsc,
    existing: TrainingResourceIdentity | None = None,
) -> None:
    violations: list[InvalidParameter] = []
    if not value.ml_correlation_id or not value.ml_correlation_id.strip():
        violations.append(InvalidParameter("mlCorreId", "is required for federated learning"))
    for index, event in enumerate(value.ml_event_subscriptions):
        if not event.model_interoperability.strip():
            violations.append(
                InvalidParameter(
                    f"mLEventSubscs[{index}].modelInterInfo",
                    "is required for federated learning",
                )
            )
    if value.ml_preparation_flag:
        for index, info in enumerate(value.ml_model_training_infos or []):
            if info.data_availability_requirement is None:
                violations.append(
                    InvalidParameter(
                        f"mLModelTrainInfos[{index}].dataAvReq",
                        "is required for training preparation",
                    )
                )
            if (
                info.time_availability_requirement is None
                or not info.time_availability_requirement.strip()
            ):
                violations.append(
                    InvalidParameter(
                        f"mLModelTrainInfos[{index}].timeAvReq",
                        "is required for training preparation",
                    )
                )
    if value.ml_training_report_info is not None and (
        value.event_request is None
        or value.event_request.notification_method != "ON_EVENT_DETECTION"
    ):
        violations.append(
            InvalidParameter(
                "mLTrainRepInfo",
                "requires eventReq.notifMethod ON_EVENT_DETECTION",
            )
        )
    if existing is not None:
        if value.ml_correlation_id != existing.ml_correlation_id:
            violations.append(
                InvalidParameter(
                    "mlCorreId",
                    "must not change for an existing resource",
                )
            )
        if value.notification_correlation_id != existing.notification_correlation_id:
            violations.append(
                InvalidParameter(
                    "notifCorreId",
                    "must match the existing resource",
                )
            )
    if violations:
        raise RequirementsError(violations)


def validate_fl_patch(
    value: NwdafMLModelTrainSubscPatch,
    existing: TrainingResourceIdentity,
) -> None:
    if not existing.ml_correlation_id:
        raise ValueError("existing training resource identity is required")
    effective_method = (
        value.event_request.notification_method
        if value.event_request is not None
        else existing.notification_method
    )
    if value.ml_training_report_info is not None and effective_method != "ON_EVENT_DETECTION":
        raise RequirementsError(
            [
                InvalidParameter(
                    "mLTrainRepInfo",
                    "requires the effective eventReq.notifMethod ON_EVENT_DETECTION",
                )
            ]
        )


def validate_fl_notification(
    value: NwdafMLModelTrainNotif,
    existing: TrainingResourceIdentity,
) -> None:
    violations: list[InvalidParameter] = []
    if value.ml_correlation_id != existing.ml_correlation_id:
        violations.append(
            InvalidParameter(
                "mlCorreId",
                "must match the existing federated learning process",
            )
        )
    if value.notification_correlation_id != existing.notification_correlation_id:
        violations.append(
            InvalidParameter(
                "notifCorreId",
                "must match the existing resource",
            )
        )
    if (
        existing.expected_round_indicator is not None
        and value.round_indicator != existing.expected_round_indicator
    ):
        violations.append(
            InvalidParameter(
                "roundInd",
                "must match the expected training round",
            )
        )
    if violations:
        raise RequirementsError(violations)
