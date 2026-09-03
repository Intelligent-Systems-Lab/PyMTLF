import json

import pytest
from pydantic import ValidationError

from py_mtlf.wire.ml_model_training import (
    CANDIDATE_TOPOLOGY_MAX_DEPTH,
    CANDIDATE_TOPOLOGY_MAX_NODES,
    CandidateOperationDescriptor,
    InvalidMessageError,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
    TrainingResourceIdentity,
    apply_subscription_patch,
    split_candidate_operations,
    validate_fl_notification,
    validate_fl_patch,
    validate_fl_subscription,
)
from py_mtlf.wire.reporting import ReportingInformation


def preparation_payload() -> dict:
    return {
        "mLEventSubscs": [
            {
                "mLEvent": "UE_COMMUNICATION",
                "mLEventFilter": {
                    "networkArea": {
                        "tais": [
                            {
                                "plmnId": {"mcc": "466", "mnc": "92"},
                                "tac": "000001",
                            }
                        ]
                    }
                },
                "modelInterInfo": "pymtlf-model-bundle-v1",
            }
        ],
        "notifUri": "http://nwdaf-c.example/training/callback",
        "notifCorreId": "prep-client-a",
        "mlCorreId": "fl-process-001",
        "mLPreFlag": True,
        "eventReq": {"immRep": True},
        "tgtRepUe": {"intGroupIds": ["group-G"]},
        "mLModelTrainInfos": [
            {
                "dataAvReq": {
                    "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                    "minNumSamples": 1000,
                    "timeWindows": [
                        {
                            "startTime": "2026-07-01T00:00:00Z",
                            "stopTime": "2026-07-27T00:00:00Z",
                        }
                    ],
                },
                "timeAvReq": "PT10M",
            }
        ],
    }


def candidate_payload() -> dict:
    payload = preparation_payload()
    payload["suppFeats"] = "4"
    payload["x-flTopology"] = {
        "nfInstanceId": "10000000-0000-4000-8000-000000000001",
        "children": [
            {
                "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                "priority": 100,
            }
        ],
        "policy": {
            "allowAdditionalCandidates": True,
            "additionalCandidatePriority": 0,
            "selectionMethod": "priority",
            "minAvailableNodes": 1,
            "fractionTrain": 1,
            "minTrainNodes": 1,
            "acceptFailures": True,
            "minCompletionRate": 0.5,
        },
        "strategy": {
            "method": "fedProx",
            "aggregation": "sampleWeighted",
            "methodParameters": {"proximalMu": 0.01},
        },
        "reportAfter": {"count": 3, "unit": "round"},
    }
    return payload


def test_reporting_information_preserves_complete_release18_shape() -> None:
    payload = {
        "immRep": False,
        "notifMethod": "PERIODIC",
        "maxReportNbr": 0,
        "monDur": "2026-07-28T12:00:00Z",
        "repPeriod": 0,
        "sampRatio": 0,
        "partitionCriteria": ["TAC"],
        "grpRepTime": 0,
        "notifFlag": "ACTIVATE",
        "notifFlagInstruct": {
            "bufferedNotifs": "DROP_OLD",
            "subscription": "CONTINUE_WITHOUT_MUTING",
        },
        "mutingSetting": {
            "maxNoOfNotif": 100,
            "durationBufferedNotif": 60,
        },
        "futureReportingField": {"enabled": True},
    }
    value = ReportingInformation.model_validate(payload)
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["immRep"] is False
    assert encoded["maxReportNbr"] == 0
    assert encoded["notifFlagInstruct"]["bufferedNotifs"] == "DROP_OLD"
    assert encoded["futureReportingField"] == {"enabled": True}


def test_training_preparation_supported_profile() -> None:
    value = NwdafMLModelTrainSubsc.model_validate(preparation_payload())
    validate_fl_subscription(value)
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["mLPreFlag"] is True
    assert encoded["mLModelTrainInfos"][0]["dataAvReq"]["inpEvents"] == [
        {"upfEvent": "USER_DATA_USAGE_TRENDS"}
    ]


def test_training_standard_extensions_are_preserved() -> None:
    payload = preparation_payload()
    payload["futureTrainingField"] = {"enabled": True}
    payload["mLEventSubscs"][0]["futureEventField"] = "future-value"
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["futureTrainingField"] == {"enabled": True}
    assert encoded["mLEventSubscs"][0]["futureEventField"] == "future-value"


def test_training_time_window_requires_timezone_and_order() -> None:
    payload = preparation_payload()
    window = payload["mLModelTrainInfos"][0]["dataAvReq"]["timeWindows"][0]
    window["startTime"] = "2026-07-01T00:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        NwdafMLModelTrainSubsc.model_validate(payload)

    payload = preparation_payload()
    window = payload["mLModelTrainInfos"][0]["dataAvReq"]["timeWindows"][0]
    window["startTime"] = "2026-07-28T00:00:00Z"
    with pytest.raises(ValidationError, match="must not precede"):
        NwdafMLModelTrainSubsc.model_validate(payload)


def test_training_openapi_required_fields() -> None:
    payload = preparation_payload()
    for field in ("mLEventSubscs", "notifUri", "notifCorreId"):
        invalid = dict(payload)
        invalid.pop(field)
        with pytest.raises(ValidationError):
            NwdafMLModelTrainSubsc.model_validate(invalid)


def test_training_fl_conditional_requirements_report_paths() -> None:
    payload = preparation_payload()
    payload.pop("mlCorreId")
    payload["mLEventSubscs"][0].pop("modelInterInfo")
    payload["mLModelTrainInfos"] = [{}]
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    with pytest.raises(RequirementsError) as captured:
        validate_fl_subscription(value)
    assert {item.parameter for item in captured.value.violations} == {
        "mlCorreId",
        "mLEventSubscs[0].modelInterInfo",
        "mLModelTrainInfos[0].dataAvReq",
        "mLModelTrainInfos[0].timeAvReq",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"notifCorreId": "corr", "delayEventNotif": {"delayEventInd": True}},
        {
            "notifCorreId": "corr",
            "delayEventNotif": {"delayEventInd": True},
            "statusReport": {"mlModelAcc": 92},
        },
        {
            "notifCorreId": "corr",
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://client.example/local-model"},
                }
            ],
        },
        {"notifCorreId": "corr", "termTrainReq": "OTHERS"},
        {
            "notifCorreId": "corr",
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://client.example/local-model"},
                }
            ],
            "statusReport": {"trainInDataInfo": {"samplRatio": 100}},
        },
        {
            "notifCorreId": "corr",
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://client.example/local-model"},
                }
            ],
            "termTrainReq": "OTHERS",
        },
    ],
)
def test_training_notification_valid_combinations(payload: dict) -> None:
    NwdafMLModelTrainNotif.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"notifCorreId": "corr"},
        {
            "notifCorreId": "corr",
            "statusReport": {"trainInDataInfo": {"samplRatio": 100}},
        },
        {
            "notifCorreId": "corr",
            "delayEventNotif": {"delayEventInd": True},
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://client.example/local-model"},
                }
            ],
        },
        {
            "notifCorreId": "corr",
            "delayEventNotif": {"delayEventInd": True},
            "termTrainReq": "OTHERS",
        },
        {"notifCorreId": "corr", "delayEventNotif": {}},
        {"notifCorreId": "corr", "mLModelInfos": []},
    ],
)
def test_training_notification_invalid_combinations(payload: dict) -> None:
    with pytest.raises(ValidationError):
        NwdafMLModelTrainNotif.model_validate(payload)


def test_training_notification_process_and_round_identity() -> None:
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "round-client-a",
            "mlCorreId": "fl-process-001",
            "roundInd": 2,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://client.example/round-2"},
                }
            ],
        }
    )
    identity = TrainingResourceIdentity(
        subscription_id="sub-1",
        ml_correlation_id="fl-process-001",
        notification_correlation_id="round-client-a",
        expected_round_indicator=2,
    )
    validate_fl_notification(notification, identity)
    changed = notification.model_copy(update={"round_indicator": 3})
    with pytest.raises(RequirementsError):
        validate_fl_notification(changed, identity)


def test_training_patch_uses_existing_reporting_method() -> None:
    patch = NwdafMLModelTrainSubscPatch.model_validate({"mLTrainRepInfo": {"maxResTime": 30}})
    valid_identity = TrainingResourceIdentity(
        subscription_id="sub-1",
        ml_correlation_id="fl-process-001",
        notification_correlation_id="corr",
        notification_method="ON_EVENT_DETECTION",
    )
    validate_fl_patch(patch, valid_identity)
    invalid_identity = TrainingResourceIdentity(
        subscription_id="sub-1",
        ml_correlation_id="fl-process-001",
        notification_correlation_id="corr",
        notification_method="PERIODIC",
    )
    with pytest.raises(RequirementsError):
        validate_fl_patch(patch, invalid_identity)


def test_training_patch_does_not_define_ml_correlation_id() -> None:
    schema = NwdafMLModelTrainSubscPatch.model_json_schema()
    assert "mlCorreId" not in schema["properties"]

    value = NwdafMLModelTrainSubscPatch.model_validate_json(
        json.dumps({"roundInd": 0, "skipFlInd": False})
    )
    encoded = value.model_dump(by_alias=True, exclude_none=True)
    assert encoded == {"roundInd": 0, "skipFlInd": False}


def test_final_validation_update_preserves_standard_flags_and_candidate() -> None:
    value = NwdafMLModelTrainSubscPatch.model_validate(
        {
            "mLAccChkFlg": True,
            "skipFlInd": True,
            "roundInd": 3,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": (
                            "http://nwdaf-c.example/training/fl-process-001/rounds/2/global.tar.gz"
                        )
                    },
                }
            ],
        }
    )
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["mLAccChkFlg"] is True
    assert encoded["skipFlInd"] is True
    assert encoded["roundInd"] == 3
    assert encoded["mLModelInfos"][0]["mLFileAddr"]["mLModelUrl"].endswith(
        "/rounds/2/global.tar.gz"
    )


def test_accuracy_check_notification_uses_model_info_and_real_accuracy_only() -> None:
    value = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "round-client-a",
            "mlCorreId": "fl-process-001",
            "roundInd": 3,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {
                        "mLModelUrl": (
                            "http://nwdaf-a.example/training/fl-process-001/"
                            "rounds/3/accuracy-check.tar.gz"
                        )
                    },
                }
            ],
            "statusReport": {"mlModelAcc": 92},
        }
    )
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["statusReport"]["mlModelAcc"] == 92

    invalid = encoded | {"statusReport": {"mlModelAcc": 101}}
    with pytest.raises(ValidationError):
        NwdafMLModelTrainNotif.model_validate(invalid)


def test_candidate_subscription_round_trip_and_nested_objects_are_closed() -> None:
    payload = candidate_payload()
    payload["x-retainedResultReq"] = False
    payload["x-flTopology"]["children"][0].update({"enabled": True, "retainedResultReq": False})
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    validate_fl_subscription(value)
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")

    assert encoded["x-flTopology"] == payload["x-flTopology"]
    assert encoded["x-retainedResultReq"] is False

    invalid = candidate_payload()
    invalid["x-flTopology"]["strategy"]["methodParameters"]["unknown"] = True
    with pytest.raises(ValidationError) as captured:
        NwdafMLModelTrainSubsc.model_validate(invalid)
    assert captured.value.errors()[0]["loc"][-1] == "unknown"


def test_candidate_patch_round_trip_preserves_complete_extension_contract() -> None:
    payload = {
        "x-retainedResultReq": True,
        "x-flTopology": {
            "nfInstanceId": "10000000-0000-4000-8000-000000000001",
            "enabled": True,
            "priority": 100,
            "policy": {
                "allowAdditionalCandidates": True,
                "additionalCandidatePriority": 5,
                "selectionMethod": "priority",
                "minAvailableNodes": 2,
                "fractionTrain": 0.5,
                "minTrainNodes": 1,
                "acceptFailures": True,
                "minCompletionRate": 0.5,
            },
            "strategy": {
                "method": "fedProx",
                "aggregation": "sampleWeighted",
                "methodParameters": {"proximalMu": 0.01},
            },
            "reportAfter": {"count": 3, "unit": "round"},
            "retainedResultReq": True,
            "children": [
                {
                    "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                    "enabled": False,
                    "priority": 80,
                }
            ],
        },
    }
    value = NwdafMLModelTrainSubscPatch.model_validate(payload)
    encoded = value.model_dump(
        by_alias=True,
        exclude_unset=True,
        exclude_none=False,
        mode="json",
    )

    assert encoded == payload


def test_candidate_notification_round_trip_preserves_complete_extension_contract() -> None:
    payload = {
        "notifCorreId": "root-branch-a",
        "mlCorreId": "hierarchical-fl-001",
        "x-retainedResultStatus": "NOT_FOUND",
        "x-flTopologyReport": {
            "nfInstanceId": "10000000-0000-4000-8000-000000000001",
            "policy": {
                "allowAdditionalCandidates": True,
                "additionalCandidatePriority": 5,
                "selectionMethod": "priority",
                "minAvailableNodes": 2,
                "fractionTrain": 0.5,
                "minTrainNodes": 1,
                "acceptFailures": True,
                "minCompletionRate": 0.5,
            },
            "strategy": {
                "method": "fedProx",
                "aggregation": "sampleWeighted",
                "methodParameters": {"proximalMu": 0.01},
            },
            "reportAfter": {"count": 3, "unit": "round"},
            "children": [
                {
                    "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                    "status": "ACTIVE",
                    "statusTimestamp": "2026-09-02T06:29:10Z",
                    "policy": {"minAvailableNodes": 1, "minTrainNodes": 1},
                    "strategy": {
                        "method": "fedProx",
                        "aggregation": "sampleWeighted",
                        "methodParameters": {"proximalMu": 0.02},
                    },
                    "reportAfter": {"count": 2, "unit": "round"},
                    "children": [
                        {
                            "nfInstanceId": ("10000000-0000-4000-8000-000000000201"),
                            "status": "FAILED",
                            "statusTimestamp": "2026-09-02T06:30:10Z",
                            "statusCause": "RESOURCE_UNAVAILABLE",
                        }
                    ],
                }
            ],
        },
    }
    value = NwdafMLModelTrainNotif.model_validate(payload)
    validate_fl_notification(
        value,
        TrainingResourceIdentity(
            subscription_id="sub-1",
            ml_correlation_id="hierarchical-fl-001",
            notification_correlation_id="root-branch-a",
            bound_participant_nf_instance_id=("10000000-0000-4000-8000-000000000001"),
        ),
    )
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")

    assert encoded["x-flTopologyReport"] == payload["x-flTopologyReport"]
    assert encoded["x-retainedResultStatus"] == payload["x-retainedResultStatus"]


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("x-flTopology", "children", 0, "enabled"), "yes"),
        (("x-flTopology", "policy", "fractionTrain"), "all"),
        (
            (
                "x-flTopology",
                "strategy",
                "methodParameters",
                "proximalMu",
            ),
            "small",
        ),
    ],
)
def test_candidate_nested_types_are_strict(field_path, invalid_value) -> None:
    payload = candidate_payload()
    target = payload
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = invalid_value

    with pytest.raises(ValidationError) as captured:
        NwdafMLModelTrainSubsc.model_validate(payload)
    assert captured.value.errors()[0]["loc"] == field_path


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (
            lambda payload: payload["x-flTopology"]["children"].append(
                {
                    "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                    "priority": 90,
                }
            ),
            "x-flTopology.children[1].nfInstanceId",
        ),
        (
            lambda payload: payload["x-flTopology"]["children"][0].pop("priority"),
            "x-flTopology.children[0].priority",
        ),
        (
            lambda payload: payload["x-flTopology"]["policy"].update({"minTrainNodes": 2}),
            "x-flTopology.policy.minAvailableNodes",
        ),
        (
            lambda payload: payload["x-flTopology"]["children"][0].update(
                {"enabled": False, "retainedResultReq": True}
            ),
            "x-flTopology.children[0].retainedResultReq",
        ),
    ],
)
def test_candidate_topology_cross_field_validation(mutate, path: str) -> None:
    payload = candidate_payload()
    mutate(payload)
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_subscription(value)
    assert captured.value.violations[0].parameter == path


def test_candidate_topology_safety_bounds() -> None:
    payload = candidate_payload()
    root = payload["x-flTopology"]
    root.pop("children")
    root.pop("policy")
    cursor = root
    for index in range(2, CANDIDATE_TOPOLOGY_MAX_DEPTH + 2):
        child = {"nfInstanceId": f"10000000-0000-4000-8000-{index:012d}"}
        cursor["children"] = [child]
        cursor = child
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    with pytest.raises(InvalidMessageError, match="maximum topology depth"):
        validate_fl_subscription(value)

    payload = candidate_payload()
    payload["x-flTopology"].pop("policy")
    payload["x-flTopology"]["children"] = [
        {"nfInstanceId": f"10000000-0000-4000-8000-{index:012d}"}
        for index in range(2, CANDIDATE_TOPOLOGY_MAX_NODES + 2)
    ]
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    with pytest.raises(InvalidMessageError, match="maximum topology node count"):
        validate_fl_subscription(value)


def test_candidate_full_message_rejects_null_but_patch_can_remove_values() -> None:
    payload = candidate_payload()
    payload["x-flTopology"]["children"][0]["enabled"] = None
    with pytest.raises(ValidationError):
        NwdafMLModelTrainSubsc.model_validate(payload)

    patch = NwdafMLModelTrainSubscPatch.model_validate({"x-flTopology": {"strategy": None}})
    assert "strategy" in patch.fl_topology.model_fields_set
    assert patch.fl_topology.strategy is None


def test_candidate_strategy_method_is_closed() -> None:
    payload = candidate_payload()
    payload["x-flTopology"]["strategy"]["method"] = "fedAvg"
    with pytest.raises(ValidationError) as captured:
        NwdafMLModelTrainSubsc.model_validate(payload)
    assert captured.value.errors()[0]["loc"] == (
        "x-flTopology",
        "strategy",
        "method",
    )


def test_candidate_patch_uses_merge_patch_and_strips_operations() -> None:
    current = NwdafMLModelTrainSubsc.model_validate(candidate_payload())
    patch = NwdafMLModelTrainSubscPatch.model_validate(
        {
            "x-retainedResultReq": True,
            "x-flTopology": {
                "strategy": None,
                "children": [
                    {
                        "nfInstanceId": "10000000-0000-4000-8000-000000000202",
                        "priority": 80,
                        "retainedResultReq": True,
                    }
                ],
            },
        }
    )
    effective = apply_subscription_patch(current, patch)
    persistent, operation = split_candidate_operations(effective)

    assert isinstance(operation, CandidateOperationDescriptor)
    assert operation.top_level_retained_result_request is True
    assert operation.node_requests == ("10000000-0000-4000-8000-000000000202",)
    encoded = persistent.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert "strategy" not in encoded["x-flTopology"]
    assert encoded["x-flTopology"]["children"][0]["nfInstanceId"].endswith("202")
    assert "x-retainedResultReq" not in encoded
    assert "retainedResultReq" not in encoded["x-flTopology"]["children"][0]


def test_candidate_notification_status_and_retained_result_rules() -> None:
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-flTopologyReport": {
                "nfInstanceId": "10000000-0000-4000-8000-000000000001",
                "children": [
                    {
                        "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                        "status": "FAILED",
                        "statusTimestamp": "2026-09-02T06:29:10Z",
                    }
                ],
            },
        }
    )
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            notification,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )
    assert captured.value.violations[0].parameter.endswith("statusCause")

    found = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-retainedResultStatus": "FOUND",
            "roundInd": 5,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://leaf.example/round-5"},
                }
            ],
        }
    )
    validate_fl_notification(
        found,
        TrainingResourceIdentity(
            subscription_id="sub-1",
            ml_correlation_id="hierarchical-fl-001",
            notification_correlation_id="root-branch-a",
            expected_round_indicator=1,
        ),
    )

    not_found_with_model = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-retainedResultStatus": "NOT_FOUND",
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://leaf.example/round-5"},
                }
            ],
        }
    )
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            not_found_with_model,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )
    assert captured.value.violations[0].parameter == "mLModelInfos"


def test_candidate_notification_requires_process_and_bound_participant_identity() -> None:
    without_process = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "x-flTopologyReport": {
                "nfInstanceId": "10000000-0000-4000-8000-000000000001",
            },
        }
    )
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            without_process,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )
    assert captured.value.violations[0].parameter == "mlCorreId"

    wrong_participant = without_process.model_copy(
        update={"ml_correlation_id": "hierarchical-fl-001"}
    )
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            wrong_participant,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
                bound_participant_nf_instance_id=("20000000-0000-4000-8000-000000000001"),
            ),
        )
    assert captured.value.violations[0].parameter == ("x-flTopologyReport.nfInstanceId")


def test_candidate_report_timestamp_requires_strict_rfc3339() -> None:
    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-flTopologyReport": {
                "nfInstanceId": "10000000-0000-4000-8000-000000000001",
                "children": [
                    {
                        "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                        "status": "ACTIVE",
                        "statusTimestamp": "2026-09-02 06:29:10Z",
                    }
                ],
            },
        }
    )
    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            notification,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )
    assert captured.value.violations[0].parameter.endswith("statusTimestamp")


def test_candidate_topology_report_recursive_validation_and_bounds() -> None:
    duplicate = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-flTopologyReport": {
                "nfInstanceId": "10000000-0000-4000-8000-000000000001",
                "children": [
                    {
                        "nfInstanceId": "10000000-0000-4000-8000-000000000001",
                        "status": "ACTIVE",
                        "statusTimestamp": "2026-09-02T06:29:10Z",
                    }
                ],
            },
        }
    )
    with pytest.raises(InvalidMessageError, match="unique within the subtree"):
        validate_fl_notification(
            duplicate,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )

    report = {
        "nfInstanceId": "10000000-0000-4000-8000-000000000001",
        "children": [],
    }
    cursor = report
    for index in range(2, CANDIDATE_TOPOLOGY_MAX_DEPTH + 2):
        child = {
            "nfInstanceId": f"10000000-0000-4000-8000-{index:012d}",
            "status": "ACTIVE",
            "statusTimestamp": "2026-09-02T06:29:10Z",
            "children": [],
        }
        cursor["children"] = [child]
        cursor = child
    cursor.pop("children")
    too_deep = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-flTopologyReport": report,
        }
    )
    with pytest.raises(InvalidMessageError, match="maximum topology depth"):
        validate_fl_notification(
            too_deep,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )

    wide_report = {
        "nfInstanceId": "10000000-0000-4000-8000-000000000001",
        "children": [
            {
                "nfInstanceId": f"10000000-0000-4000-8000-{index:012d}",
                "status": "ACTIVE",
                "statusTimestamp": "2026-09-02T06:29:10Z",
            }
            for index in range(2, CANDIDATE_TOPOLOGY_MAX_NODES + 2)
        ],
    }
    too_wide = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-flTopologyReport": wide_report,
        }
    )
    with pytest.raises(InvalidMessageError, match="maximum topology node count"):
        validate_fl_notification(
            too_wide,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )


def test_candidate_forward_compatible_enums_remain_lossless() -> None:
    payload = candidate_payload()
    payload["x-flTopology"]["policy"]["selectionMethod"] = "vendorSelection"
    payload["x-flTopology"]["strategy"]["aggregation"] = "vendorWeighted"
    payload["x-flTopology"]["reportAfter"]["unit"] = "window"
    value = NwdafMLModelTrainSubsc.model_validate(payload)
    validate_fl_subscription(value)
    encoded = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    assert encoded["x-flTopology"]["policy"]["selectionMethod"] == "vendorSelection"
    assert encoded["x-flTopology"]["strategy"]["aggregation"] == "vendorWeighted"
    assert encoded["x-flTopology"]["reportAfter"]["unit"] == "window"

    notification = NwdafMLModelTrainNotif.model_validate(
        {
            "notifCorreId": "root-branch-a",
            "mlCorreId": "hierarchical-fl-001",
            "x-retainedResultStatus": "VENDOR_OUTCOME",
            "x-flTopologyReport": {
                "nfInstanceId": "10000000-0000-4000-8000-000000000001",
                "children": [
                    {
                        "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                        "status": "VENDOR_PENDING",
                        "statusTimestamp": "2026-09-02T06:29:10Z",
                        "statusCause": "VENDOR_REASON",
                    }
                ],
            },
        }
    )
    validate_fl_notification(
        notification,
        TrainingResourceIdentity(
            subscription_id="sub-1",
            ml_correlation_id="hierarchical-fl-001",
            notification_correlation_id="root-branch-a",
            bound_participant_nf_instance_id=("10000000-0000-4000-8000-000000000001"),
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("aggregation", " ", "x-flTopology.strategy.aggregation"),
        ("reportAfter.unit", " ", "x-flTopology.reportAfter.unit"),
    ],
)
def test_candidate_required_strategy_strings_are_not_blank(
    field: str,
    value: str,
    path: str,
) -> None:
    payload = candidate_payload()
    if field == "aggregation":
        payload["x-flTopology"]["strategy"][field] = value
    else:
        payload["x-flTopology"]["reportAfter"]["unit"] = value
    parsed = NwdafMLModelTrainSubsc.model_validate(payload)

    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_subscription(parsed)
    assert captured.value.violations[0].parameter == path


@pytest.mark.parametrize(
    ("field", "path"),
    [
        ("status", "x-flTopologyReport.children[0].status"),
        ("x-retainedResultStatus", "x-retainedResultStatus"),
    ],
)
def test_candidate_forward_compatible_status_strings_are_not_blank(
    field: str,
    path: str,
) -> None:
    payload = {
        "notifCorreId": "root-branch-a",
        "mlCorreId": "hierarchical-fl-001",
        "x-flTopologyReport": {
            "nfInstanceId": "10000000-0000-4000-8000-000000000001",
            "children": [
                {
                    "nfInstanceId": "10000000-0000-4000-8000-000000000101",
                    "status": "ACTIVE",
                    "statusTimestamp": "2026-09-02T06:29:10Z",
                }
            ],
        },
    }
    if field == "status":
        payload["x-flTopologyReport"]["children"][0][field] = " "
    else:
        payload[field] = " "
    parsed = NwdafMLModelTrainNotif.model_validate(payload)

    with pytest.raises(InvalidMessageError) as captured:
        validate_fl_notification(
            parsed,
            TrainingResourceIdentity(
                subscription_id="sub-1",
                ml_correlation_id="hierarchical-fl-001",
                notification_correlation_id="root-branch-a",
            ),
        )
    assert captured.value.violations[0].parameter == path
