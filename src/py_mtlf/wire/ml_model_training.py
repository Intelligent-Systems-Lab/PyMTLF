from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    field_validator,
    model_validator,
)

from py_mtlf.wire.ml_model import MLEventNotification, MLEventSubscription
from py_mtlf.wire.reporting import ReportingInformation, StandardModel

CANDIDATE_TOPOLOGY_MAX_DEPTH = 16
CANDIDATE_TOPOLOGY_MAX_NODES = 1024
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class StrictCandidateModel(StandardModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


def _reject_explicit_null(value):
    if value is None:
        raise ValueError("null is not allowed")
    return value


class FlPolicy(StrictCandidateModel):
    allow_additional_candidates: bool | None = Field(
        default=None,
        alias="allowAdditionalCandidates",
    )
    additional_candidate_priority: int | None = Field(
        default=None,
        alias="additionalCandidatePriority",
        ge=0,
    )
    selection_method: str | None = Field(default=None, alias="selectionMethod")
    minimum_available_nodes: int | None = Field(
        default=None,
        alias="minAvailableNodes",
        ge=1,
    )
    fraction_train: float | None = Field(
        default=None,
        alias="fractionTrain",
        gt=0,
        le=1,
    )
    minimum_train_nodes: int | None = Field(
        default=None,
        alias="minTrainNodes",
        ge=1,
    )
    accept_failures: bool | None = Field(default=None, alias="acceptFailures")
    minimum_completion_rate: float | None = Field(
        default=None,
        alias="minCompletionRate",
        gt=0,
        le=1,
    )

    _reject_nulls = field_validator(
        "allow_additional_candidates",
        "additional_candidate_priority",
        "selection_method",
        "minimum_available_nodes",
        "fraction_train",
        "minimum_train_nodes",
        "accept_failures",
        "minimum_completion_rate",
        mode="before",
    )(_reject_explicit_null)


class FedProxParameters(StrictCandidateModel):
    proximal_mu: float = Field(alias="proximalMu", ge=0)


class FlStrategy(StrictCandidateModel):
    method: Literal["fedProx"]
    aggregation: str = Field(min_length=1)
    method_parameters: FedProxParameters = Field(alias="methodParameters")


class FlReportAfter(StrictCandidateModel):
    count: int = Field(ge=1)
    unit: str = Field(min_length=1)


class FlTopologyNode(StrictCandidateModel):
    nf_instance_id: str = Field(min_length=1, alias="nfInstanceId")
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=0)
    policy: FlPolicy | None = None
    strategy: FlStrategy | None = None
    report_after: FlReportAfter | None = Field(default=None, alias="reportAfter")
    retained_result_request: StrictBool | None = Field(
        default=None,
        alias="retainedResultReq",
    )
    children: list[FlTopologyNode] | None = Field(default=None, min_length=1)

    _reject_nulls = field_validator(
        "enabled",
        "priority",
        "policy",
        "strategy",
        "report_after",
        "retained_result_request",
        "children",
        mode="before",
    )(_reject_explicit_null)


class FlTopologyReportNode(StrictCandidateModel):
    nf_instance_id: str = Field(min_length=1, alias="nfInstanceId")
    status: str = Field(min_length=1)
    status_timestamp: str = Field(min_length=1, alias="statusTimestamp")
    status_cause: str | None = Field(default=None, alias="statusCause")
    policy: FlPolicy | None = None
    strategy: FlStrategy | None = None
    report_after: FlReportAfter | None = Field(default=None, alias="reportAfter")
    children: list[FlTopologyReportNode] | None = Field(default=None, min_length=1)

    _reject_nulls = field_validator(
        "status_cause",
        "policy",
        "strategy",
        "report_after",
        "children",
        mode="before",
    )(_reject_explicit_null)


class FlTopologyReport(StrictCandidateModel):
    nf_instance_id: str = Field(min_length=1, alias="nfInstanceId")
    policy: FlPolicy | None = None
    strategy: FlStrategy | None = None
    report_after: FlReportAfter | None = Field(default=None, alias="reportAfter")
    children: list[FlTopologyReportNode] | None = Field(default=None, min_length=1)

    _reject_nulls = field_validator(
        "policy",
        "strategy",
        "report_after",
        "children",
        mode="before",
    )(_reject_explicit_null)


class FlPolicyPatch(StrictCandidateModel):
    allow_additional_candidates: bool | None = Field(
        default=None,
        alias="allowAdditionalCandidates",
    )
    additional_candidate_priority: int | None = Field(
        default=None,
        alias="additionalCandidatePriority",
        ge=0,
    )
    selection_method: str | None = Field(default=None, alias="selectionMethod")
    minimum_available_nodes: int | None = Field(
        default=None,
        alias="minAvailableNodes",
        ge=1,
    )
    fraction_train: float | None = Field(
        default=None,
        alias="fractionTrain",
        gt=0,
        le=1,
    )
    minimum_train_nodes: int | None = Field(
        default=None,
        alias="minTrainNodes",
        ge=1,
    )
    accept_failures: bool | None = Field(default=None, alias="acceptFailures")
    minimum_completion_rate: float | None = Field(
        default=None,
        alias="minCompletionRate",
        gt=0,
        le=1,
    )


class FedProxParametersPatch(StrictCandidateModel):
    proximal_mu: float | None = Field(default=None, alias="proximalMu", ge=0)


class FlStrategyPatch(StrictCandidateModel):
    method: Literal["fedProx"] | None = None
    aggregation: str | None = Field(default=None, min_length=1)
    method_parameters: FedProxParametersPatch | None = Field(
        default=None,
        alias="methodParameters",
    )


class FlReportAfterPatch(StrictCandidateModel):
    count: int | None = Field(default=None, ge=1)
    unit: str | None = Field(default=None, min_length=1)


class FlTopologyNodePatch(StrictCandidateModel):
    nf_instance_id: str | None = Field(default=None, min_length=1, alias="nfInstanceId")
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=0)
    policy: FlPolicyPatch | None = None
    strategy: FlStrategyPatch | None = None
    report_after: FlReportAfterPatch | None = Field(default=None, alias="reportAfter")
    retained_result_request: StrictBool | None = Field(
        default=None,
        alias="retainedResultReq",
    )
    children: list[FlTopologyNode] | None = Field(default=None, min_length=1)


class TimeWindow(StandardModel):
    start_time: datetime = Field(alias="startTime")
    stop_time: datetime = Field(alias="stopTime")

    @model_validator(mode="after")
    def validate_window(self) -> TimeWindow:
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
    def validate_union(self) -> DCCFEvent:
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
    fl_topology_report: FlTopologyReport | None = Field(
        default=None,
        alias="flTopologyReport",
    )
    retained_result_status: str | None = Field(
        default=None,
        alias="retainedResultStatus",
    )

    _reject_candidate_nulls = field_validator(
        "fl_topology_report",
        "retained_result_status",
        mode="before",
    )(_reject_explicit_null)

    @model_validator(mode="after")
    def validate_result_combination(self) -> NwdafMLModelTrainNotif:
        has_delay = self.delay_event_notification is not None
        has_models = bool(self.ml_model_infos)
        has_termination = bool(self.termination_request)
        has_candidate = self.fl_topology_report is not None or bool(
            self.retained_result_status
        )

        if not has_delay and not has_models and not has_termination and not has_candidate:
            raise ValueError("at least one detailed notification field is required")
        if has_delay and (has_models or has_termination):
            raise ValueError(
                "delayEventNotif cannot coexist with mLModelInfos or termTrainReq"
            )
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
    fl_topology: FlTopologyNode | None = Field(default=None, alias="flTopology")
    retained_result_request: StrictBool | None = Field(
        default=None,
        alias="retainedResultReq",
    )

    _reject_candidate_nulls = field_validator(
        "fl_topology",
        "retained_result_request",
        mode="before",
    )(_reject_explicit_null)

    @field_validator("supported_features")
    @classmethod
    def supported_features_are_hexadecimal(cls, value: str | None) -> str | None:
        if value is not None and any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError("suppFeats must be a hexadecimal bitmask")
        return value


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
    fl_topology: FlTopologyNodePatch | None = Field(default=None, alias="flTopology")
    retained_result_request: StrictBool | None = Field(
        default=None,
        alias="retainedResultReq",
    )


@dataclass(frozen=True)
class TrainingResourceIdentity:
    subscription_id: str
    ml_correlation_id: str
    notification_correlation_id: str
    expected_round_indicator: int | None = None
    notification_method: str | None = None
    bound_participant_nf_instance_id: str = ""


@dataclass(frozen=True)
class InvalidParameter:
    parameter: str
    reason: str


class RequirementsError(ValueError):
    cause = "ML_MODEL_TRAINING_REQS_NOT_MET"

    def __init__(self, violations: list[InvalidParameter]) -> None:
        super().__init__(self.cause)
        self.violations = tuple(violations)


class InvalidMessageError(ValueError):
    cause = "INVALID_MSG_FORMAT"

    def __init__(self, violations: list[InvalidParameter]) -> None:
        self.violations = tuple(violations)
        detail = "; ".join(
            f"{violation.parameter}: {violation.reason}" for violation in self.violations
        )
        super().__init__(detail or self.cause)


@dataclass(frozen=True)
class CandidateOperationDescriptor:
    top_level_retained_result_request: bool = False
    node_requests: tuple[str, ...] = ()


def has_candidate_subscription_fields(value: NwdafMLModelTrainSubsc) -> bool:
    return value.fl_topology is not None or value.retained_result_request is not None


def has_candidate_patch_fields(value: NwdafMLModelTrainSubscPatch) -> bool:
    return bool(
        {"fl_topology", "retained_result_request"}.intersection(value.model_fields_set)
    )


def has_candidate_notification_fields(value: NwdafMLModelTrainNotif | None) -> bool:
    return value is not None and (
        value.fl_topology_report is not None or bool(value.retained_result_status)
    )


def split_candidate_operations(
    value: NwdafMLModelTrainSubsc,
) -> tuple[NwdafMLModelTrainSubsc, CandidateOperationDescriptor]:
    persistent = value.model_copy(deep=True)
    top_level = persistent.retained_result_request is True
    persistent.retained_result_request = None
    node_requests: list[str] = []

    def strip(node: FlTopologyNode | None) -> None:
        if node is None:
            return
        if node.retained_result_request is True:
            node_requests.append(node.nf_instance_id)
        node.retained_result_request = None
        for child in node.children or ():
            strip(child)

    strip(persistent.fl_topology)
    return persistent, CandidateOperationDescriptor(top_level, tuple(node_requests))


def apply_subscription_patch(
    current: NwdafMLModelTrainSubsc,
    patch: NwdafMLModelTrainSubscPatch,
) -> NwdafMLModelTrainSubsc:
    current_value = current.model_dump(by_alias=True, exclude_none=True, mode="json")
    patch_value = patch.model_dump(
        by_alias=True,
        exclude_unset=True,
        exclude_none=False,
        mode="json",
    )
    effective = _merge_json_value(current_value, patch_value)
    value = NwdafMLModelTrainSubsc.model_validate(effective)
    validate_fl_subscription(value)
    return value


def _merge_json_value(current, patch):
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(current) if isinstance(current, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge_json_value(result.get(key), value)
    return result


def _candidate_invalid(path: str, reason: str) -> InvalidMessageError:
    return InvalidMessageError([InvalidParameter(path, reason)])


def _canonical_nf_instance_id(value: str, path: str) -> str:
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise _candidate_invalid(path, "must be a UUID") from error


def _same_nf_instance_id(first: str, second: str) -> bool:
    try:
        return UUID(first) == UUID(second)
    except (AttributeError, TypeError, ValueError):
        return False


def _validate_policy(
    policy: FlPolicy | None,
    children: list[FlTopologyNode] | None,
    path: str,
    children_path: str,
) -> None:
    if policy is None:
        return
    if (
        policy.minimum_available_nodes is not None
        and policy.minimum_train_nodes is not None
        and policy.minimum_available_nodes < policy.minimum_train_nodes
    ):
        raise _candidate_invalid(path + ".minAvailableNodes", "must be at least minTrainNodes")
    if policy.selection_method == "priority":
        for index, child in enumerate(children or ()):
            if child.enabled is not False and child.priority is None:
                raise _candidate_invalid(
                    f"{children_path}[{index}].priority",
                    "is required for an enabled child when selectionMethod is priority",
                )


def _validate_strategy(strategy: FlStrategy | None, path: str) -> None:
    if strategy is not None and not strategy.aggregation.strip():
        raise _candidate_invalid(path + ".aggregation", "is required")


def _validate_report_after(value: FlReportAfter | None, path: str) -> None:
    if value is not None and not value.unit.strip():
        raise _candidate_invalid(path + ".unit", "is required")


def _validate_rfc3339(value: str, path: str) -> None:
    if _RFC3339_PATTERN.fullmatch(value) is None:
        raise _candidate_invalid(path, "must be an RFC3339 date-time")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _candidate_invalid(path, "must be an RFC3339 date-time") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise _candidate_invalid(path, "must be an RFC3339 date-time")


def _validate_topology(node: FlTopologyNode | None, path: str) -> None:
    if node is None:
        return
    seen: set[str] = set()
    count = 0

    def walk(item: FlTopologyNode, item_path: str, depth: int) -> None:
        nonlocal count
        if depth > CANDIDATE_TOPOLOGY_MAX_DEPTH:
            raise _candidate_invalid(item_path, "exceeds the maximum topology depth")
        count += 1
        if count > CANDIDATE_TOPOLOGY_MAX_NODES:
            raise _candidate_invalid(item_path, "exceeds the maximum topology node count")
        identity = _canonical_nf_instance_id(item.nf_instance_id, item_path + ".nfInstanceId")
        if identity in seen:
            raise _candidate_invalid(
                item_path + ".nfInstanceId",
                "must be unique within the subtree",
            )
        seen.add(identity)
        if item.retained_result_request is True and item.enabled is False:
            raise _candidate_invalid(
                item_path + ".retainedResultReq",
                "cannot be true when the node is disabled",
            )
        _validate_policy(item.policy, item.children, item_path + ".policy", item_path + ".children")
        _validate_strategy(item.strategy, item_path + ".strategy")
        _validate_report_after(item.report_after, item_path + ".reportAfter")
        for index, child in enumerate(item.children or ()):
            walk(child, f"{item_path}.children[{index}]", depth + 1)

    walk(node, path, 1)


def _validate_topology_report(report: FlTopologyReport | None, path: str) -> None:
    if report is None:
        return
    seen = {_canonical_nf_instance_id(report.nf_instance_id, path + ".nfInstanceId")}
    count = 1
    _validate_policy(report.policy, None, path + ".policy", path + ".children")
    _validate_strategy(report.strategy, path + ".strategy")
    _validate_report_after(report.report_after, path + ".reportAfter")

    def walk(item: FlTopologyReportNode, item_path: str, depth: int) -> None:
        nonlocal count
        if depth > CANDIDATE_TOPOLOGY_MAX_DEPTH:
            raise _candidate_invalid(item_path, "exceeds the maximum topology depth")
        count += 1
        if count > CANDIDATE_TOPOLOGY_MAX_NODES:
            raise _candidate_invalid(item_path, "exceeds the maximum topology node count")
        identity = _canonical_nf_instance_id(item.nf_instance_id, item_path + ".nfInstanceId")
        if identity in seen:
            raise _candidate_invalid(
                item_path + ".nfInstanceId",
                "must be unique within the subtree",
            )
        seen.add(identity)
        if not item.status.strip():
            raise _candidate_invalid(item_path + ".status", "is required")
        known_failure = item.status in {"FAILED", "INACTIVE"}
        known_non_failure = item.status in {"UNCONFIRMED", "DEPLOYING", "ACTIVE"}
        _validate_rfc3339(item.status_timestamp, item_path + ".statusTimestamp")
        if known_failure and not (item.status_cause or "").strip():
            raise _candidate_invalid(
                item_path + ".statusCause",
                "is required for FAILED or INACTIVE",
            )
        if known_non_failure and (item.status_cause or "") != "":
            raise _candidate_invalid(
                item_path + ".statusCause",
                "is not allowed for this status",
            )
        _validate_policy(item.policy, None, item_path + ".policy", item_path + ".children")
        _validate_strategy(item.strategy, item_path + ".strategy")
        _validate_report_after(item.report_after, item_path + ".reportAfter")
        for index, child in enumerate(item.children or ()):
            walk(child, f"{item_path}.children[{index}]", depth + 1)

    for index, child in enumerate(report.children or ()):
        walk(child, f"{path}.children[{index}]", 2)


def validate_candidate_subscription_receiver(
    value: NwdafMLModelTrainSubsc,
    expected_nf_instance_id: str,
) -> None:
    if value.fl_topology is None or not expected_nf_instance_id.strip():
        return
    if not _same_nf_instance_id(value.fl_topology.nf_instance_id, expected_nf_instance_id):
        raise _candidate_invalid(
            "flTopology.nfInstanceId",
            "must identify the request receiver",
        )


def validate_candidate_notification_participant(
    value: NwdafMLModelTrainNotif,
    expected_nf_instance_id: str,
) -> None:
    if value.fl_topology_report is None or not expected_nf_instance_id.strip():
        return
    if not _same_nf_instance_id(
        value.fl_topology_report.nf_instance_id,
        expected_nf_instance_id,
    ):
        raise _candidate_invalid(
            "flTopologyReport.nfInstanceId",
            "must identify the bound direct participant",
        )


def _validate_retained_result(value: NwdafMLModelTrainNotif, prefix: str = "") -> None:
    status = value.retained_result_status
    if status is None or status == "":
        return

    def path(field: str) -> str:
        return f"{prefix}.{field}" if prefix else field

    if not status.strip():
        raise _candidate_invalid(path("retainedResultStatus"), "is required")

    if status == "FOUND":
        if value.round_indicator is None:
            raise _candidate_invalid(
                path("roundInd"),
                "is required when retainedResultStatus is FOUND",
            )
        if not value.ml_model_infos:
            raise _candidate_invalid(
                path("mLModelInfos"),
                "is required when retainedResultStatus is FOUND",
            )
    elif status in {"NOT_FOUND", "FAILED"}:
        if value.round_indicator is not None:
            raise _candidate_invalid(
                path("roundInd"),
                "is not allowed when retainedResultStatus is not FOUND",
            )
        if value.ml_model_infos:
            raise _candidate_invalid(
                path("mLModelInfos"),
                "is not allowed when retainedResultStatus is not FOUND",
            )


def validate_fl_subscription(
    value: NwdafMLModelTrainSubsc,
    existing: TrainingResourceIdentity | None = None,
) -> None:
    _validate_topology(value.fl_topology, "flTopology")
    if value.immediate_report is not None:
        _validate_topology_report(
            value.immediate_report.fl_topology_report,
            "immReport.flTopologyReport",
        )
        _validate_retained_result(value.immediate_report, "immReport")
    if (
        value.fl_topology is not None or value.retained_result_request is True
    ) and not (value.ml_correlation_id or "").strip():
        raise _candidate_invalid(
            "mlCorreId",
            "is required for hierarchical FL operations",
        )
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
        if not value.ml_model_training_infos:
            violations.append(
                InvalidParameter(
                    "mLModelTrainInfos",
                    "is required for training preparation",
                )
            )
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
    if has_candidate_patch_fields(value) and not existing.ml_correlation_id.strip():
        raise _candidate_invalid(
            "mlCorreId",
            "the existing resource must identify a hierarchical FL procedure",
        )
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
    _validate_topology_report(value.fl_topology_report, "flTopologyReport")
    _validate_retained_result(value)
    if has_candidate_notification_fields(value) and not (value.ml_correlation_id or "").strip():
        raise _candidate_invalid(
            "mlCorreId",
            "is required for hierarchical FL notifications",
        )
    validate_candidate_notification_participant(
        value,
        existing.bound_participant_nf_instance_id,
    )
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
        and not value.retained_result_status
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
