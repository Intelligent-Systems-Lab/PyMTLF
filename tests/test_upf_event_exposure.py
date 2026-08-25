import pytest
from pydantic import ValidationError

from py_mtlf.wire.upf_event_exposure import NotificationData


def notification(correlation_id: str) -> NotificationData:
    return NotificationData.model_validate(
        {
            "correlationId": correlation_id,
            "notificationItems": [
                {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "ueIpv4Addr": "10.0.0.1",
                    "timeStamp": "2026-08-26T10:00:01Z",
                    "userDataUsageMeasurements": [
                        {
                            "volumeMeasurement": {
                                "totalVolume": 10,
                                "ulVolume": 4,
                                "dlVolume": 6,
                                "totalNbOfPackets": 3,
                                "ulNbOfPackets": 1,
                                "dlNbOfPackets": 2,
                            },
                            "throughputMeasurement": {
                                "ulThroughput": "1 Mbps",
                                "dlThroughput": "2 Mbps",
                                "ulPacketThroughput": "3 kpps",
                                "dlPacketThroughput": "4 kpps",
                            },
                        }
                    ],
                }
            ],
        }
    )


def test_supported_upf_notification_profile_is_strict():
    value = notification("11111111-1111-4111-8111-111111111111")

    assert value.notification_items[0].event_type == "USER_DATA_USAGE_MEASURES"


@pytest.mark.parametrize(
    "mutation",
    [
        {"correlationId": ""},
        {"notificationItems": []},
        {
            "notificationItems": [
                {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "timeStamp": "2026-08-26T10:00:01Z",
                    "userDataUsageMeasurements": [],
                }
            ]
        },
        {
            "notificationItems": [
                {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "ueIpv4Addr": "10.0.0.1",
                    "timeStamp": "2026-08-26T10:00:01",
                    "userDataUsageMeasurements": [
                        {
                            "volumeMeasurement": {
                                "totalVolume": 10,
                                "ulVolume": 4,
                                "dlVolume": 6,
                                "totalNbOfPackets": 3,
                                "ulNbOfPackets": 1,
                                "dlNbOfPackets": 2,
                            },
                            "throughputMeasurement": {
                                "ulThroughput": "NaN Mbps",
                                "dlThroughput": "2 Mbps",
                                "ulPacketThroughput": "3 kpps",
                                "dlPacketThroughput": "4 kpps",
                            },
                        }
                    ],
                }
            ]
        },
    ],
)
def test_invalid_upf_notification_is_rejected(mutation):
    payload = notification(
        "11111111-1111-4111-8111-111111111111"
    ).model_dump(by_alias=True, mode="json")
    payload.update(mutation)

    with pytest.raises(ValidationError):
        NotificationData.model_validate(payload)
