import json

import pytest
from pydantic import ValidationError

from py_mtlf.wire.ml_model_training import (
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
    TrainingResourceIdentity,
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
