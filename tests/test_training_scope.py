from copy import deepcopy

from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainSubsc


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
        "tgtRepUe": {"intGroupIds": ["group-G"]},
        "mLModelTrainInfos": [
            {
                "dataAvReq": {
                    "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
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


def descriptor(payload: dict) -> TrainingScopeDescriptor:
    request = NwdafMLModelTrainSubsc.model_validate(payload)
    return TrainingScopeDescriptor.from_training_request(request, 0)


def test_scope_digest_ignores_object_key_order_and_process_ids() -> None:
    first = preparation_payload()
    second = deepcopy(first)
    first["mLEventSubscs"][0]["mLEventFilter"] = {
        "dnns": ["internet"],
        "appIds": ["app-a"],
        "networkArea": first["mLEventSubscs"][0]["mLEventFilter"]["networkArea"],
    }
    second["mLEventSubscs"][0]["mLEventFilter"] = {
        "networkArea": second["mLEventSubscs"][0]["mLEventFilter"]["networkArea"],
        "appIds": ["app-a"],
        "dnns": ["internet"],
    }
    second["mlCorreId"] = "another-process"
    second["notifCorreId"] = "another-notification"

    first_descriptor = descriptor(first)
    second_descriptor = descriptor(second)
    assert first_descriptor.scope_digest == second_descriptor.scope_digest
    assert first_descriptor.canonical_payload() == second_descriptor.canonical_payload()


def test_scope_digest_preserves_array_order() -> None:
    first = preparation_payload()
    first["mLEventSubscs"][0]["mLEventFilter"]["dnns"] = ["internet", "ims"]
    second = deepcopy(first)
    second["mLEventSubscs"][0]["mLEventFilter"]["dnns"] = ["ims", "internet"]

    assert descriptor(first).scope_digest != descriptor(second).scope_digest


def test_scope_descriptor_preserves_standard_scope_fields() -> None:
    value = descriptor(preparation_payload())
    assert value.event_subscription["mLEvent"] == "UE_COMMUNICATION"
    assert value.event_subscription["modelInterInfo"] == "pymtlf-model-bundle-v1"
    assert value.target_reporting_ue == {"intGroupIds": ["group-G"]}
    assert value.requested_time_windows == [
        {
            "startTime": "2026-07-01T00:00:00Z",
            "stopTime": "2026-07-27T00:00:00Z",
        }
    ]
