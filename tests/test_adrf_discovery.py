import httpx
import pytest

from py_mtlf.config import AdrfSettings
from py_mtlf.core.adrf_discovery import AdrfResolver
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.models import BackendSyncRequest


def projection() -> SyncProjection:
    value = SyncProjection()
    value.replace(
        BackendSyncRequest.model_validate(
            {
                "containingNwdaf": {
                    "nfInstanceId": "nwdaf-1",
                    "apiBaseUri": "http://go.example",
                    "internalCallbackBaseUri": "http://go-internal.example",
                },
                "eventsSubscriptions": [],
                "smfResources": [],
            }
        )
    )
    return value


def test_configured_mode_bypasses_go_discovery():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = AdrfResolver(
        AdrfSettings(
            mode="configured",
            configured_endpoint="http://adrf.example:9888",
        ),
        projection(),
        client,
    )

    assert resolver.resolve() == "http://adrf.example:9888"
    assert calls == 0
    client.close()


def test_nrf_mode_selects_deterministically_and_reuses_valid_result():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["target-nf-type"] == "ADRF"
        assert request.url.params["requester-nf-type"] == "NWDAF"
        assert request.url.params["service-names"] == "nadrf-datamanagement"
        return httpx.Response(
            200,
            json={
                "validityPeriod": 60,
                "nfInstances": [
                    {
                        "nfInstanceId": "b",
                        "nfStatus": "REGISTERED",
                        "nfServices": [
                            {
                                "serviceName": "nadrf-datamanagement",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://adrf-b.example:9888",
                            }
                        ],
                    },
                    {
                        "nfInstanceId": "a",
                        "nfStatus": "REGISTERED",
                        "nfServices": [
                            {
                                "serviceName": "nadrf-datamanagement",
                                "nfServiceStatus": "REGISTERED",
                                "scheme": "http",
                                "ipEndPoints": [{"ipv4Address": "192.0.2.10", "port": 9888}],
                            }
                        ],
                    },
                ],
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = AdrfResolver(AdrfSettings(mode="nrf"), projection(), client)

    assert resolver.resolve() == "http://192.0.2.10:9888"
    assert resolver.resolve() == "http://192.0.2.10:9888"
    assert calls == 1
    client.close()


def test_nrf_mode_uses_service_identity_before_api_root():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "validityPeriod": 60,
                "nfInstances": [
                    {
                        "nfInstanceId": "adrf-a",
                        "nfStatus": "REGISTERED",
                        "nfServices": [
                            {
                                "serviceInstanceId": "service-z",
                                "serviceName": "nadrf-datamanagement",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://adrf-a.example:9888",
                            }
                        ],
                        "nfServiceList": {
                            "service-a": {
                                "serviceName": "nadrf-datamanagement",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://adrf-z.example:9888",
                            }
                        },
                    }
                ],
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = AdrfResolver(AdrfSettings(mode="nrf"), projection(), client)

    assert resolver.resolve() == "http://adrf-z.example:9888"
    client.close()


@pytest.mark.parametrize(
    "body",
    [
        {"nfInstances": []},
        {"validityPeriod": -1, "nfInstances": []},
        {"validityPeriod": 10, "nfInstances": None},
    ],
)
def test_nrf_mode_rejects_malformed_search_result(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = AdrfResolver(AdrfSettings(mode="nrf"), projection(), client)

    with pytest.raises(ValueError, match="malformed SearchResult"):
        resolver.resolve()
    client.close()
