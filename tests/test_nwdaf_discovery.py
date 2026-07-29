import json

import httpx

from py_mtlf.config import ModelMonitorSettings
from py_mtlf.core.nwdaf_discovery import NwdafMonitorResolver
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.models import BackendSyncRequest

TARGET_ID = "11111111-1111-4111-8111-111111111111"


def projection() -> SyncProjection:
    value = SyncProjection()
    value.replace(
        BackendSyncRequest.model_validate(
            {
                "containingNwdaf": {
                    "nfInstanceId": "33333333-3333-4333-8333-333333333333",
                    "apiBaseUri": "http://go-c.example",
                    "internalCallbackBaseUri": "http://go-c-internal.example",
                },
                "eventsSubscriptions": [],
                "smfResources": [],
            }
        )
    )
    return value


def test_exact_monitor_discovery_selects_requested_nwdaf_and_caches():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        params = request.url.params
        assert params["target-nf-type"] == "NWDAF"
        assert params["requester-nf-type"] == "NWDAF"
        assert params["target-nf-instance-id"] == TARGET_ID
        assert params["service-names"] == "nnwdaf-mlmodelmonitor"
        assert json.loads(params["ml-analytics-info-list"]) == [
            {"mlAnalyticsIds": ["UE_COMMUNICATION"]}
        ]
        return httpx.Response(
            200,
            json={
                "validityPeriod": 60,
                "nfInstances": [
                    {
                        "nfInstanceId": "22222222-2222-4222-8222-222222222222",
                        "nfStatus": "REGISTERED",
                        "nwdafInfo": {
                            "mlAnalyticsList": [
                                {"mlAnalyticsIds": ["UE_COMMUNICATION"]}
                            ]
                        },
                        "nfServices": [
                            {
                                "serviceInstanceId": "wrong-service",
                                "serviceName": "nnwdaf-mlmodelmonitor",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://wrong.example",
                            }
                        ],
                    },
                    {
                        "nfInstanceId": TARGET_ID,
                        "nfStatus": "REGISTERED",
                        "nwdafInfo": {
                            "mlAnalyticsList": [
                                {"mlAnalyticsIds": ["UE_COMMUNICATION"]}
                            ]
                        },
                        "nfServices": [
                            {
                                "serviceInstanceId": "monitor-a",
                                "serviceName": "nnwdaf-mlmodelmonitor",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://nwdaf-a.example",
                            }
                        ],
                    },
                ],
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = NwdafMonitorResolver(ModelMonitorSettings(), projection(), client)

    first = resolver.resolve(TARGET_ID)
    second = resolver.resolve(TARGET_ID)

    assert first == second
    assert first is not None
    assert first.nf_instance_id == TARGET_ID
    assert first.nf_service_instance_id == "monitor-a"
    assert first.api_root == "http://nwdaf-a.example"
    assert first.selection_source == "NRF"
    assert calls == 1
    client.close()


def test_exact_monitor_discovery_keeps_missing_target_pending():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"validityPeriod": 0, "nfInstances": []},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = NwdafMonitorResolver(ModelMonitorSettings(), projection(), client)

    assert resolver.resolve(TARGET_ID) is None
    client.close()
