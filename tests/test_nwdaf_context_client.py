import httpx
import pytest

from py_mtlf.core.nwdaf_context import (
    CapabilityConsistencyChecker,
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
    NwdafContextClient,
)


def test_context_client_parses_typed_fl_capability_projection():
    payload = _context_payload(
        [
            {
                "mlAnalyticsIds": ["UE_COMMUNICATION"],
                "flCapabilityType": "FL_SERVER_AND_CLIENT",
            }
        ]
    )
    client = _context_client(lambda request: httpx.Response(200, json=payload, request=request))

    context = client.get(refresh=True)

    assert context.ml_analytics_capabilities == (
        MLAnalyticsCapability(
            ml_analytics_ids=("UE_COMMUNICATION",),
            fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
        ),
    )
    assert context.advertised_server is True
    assert context.advertised_client is True


@pytest.mark.parametrize(
    "capabilities",
    [
        {},
        ["not-an-entry"],
        [{"mlAnalyticsIds": [], "flCapabilityType": "FL_SERVER"}],
        [{"mlAnalyticsIds": [1], "flCapabilityType": "FL_SERVER"}],
        [{"mlAnalyticsIds": ["UE_COMMUNICATION"], "flCapabilityType": "UNKNOWN"}],
    ],
)
def test_context_client_rejects_malformed_capability_projection(capabilities):
    payload = _context_payload(capabilities)
    client = _context_client(lambda request: httpx.Response(200, json=payload, request=request))

    with pytest.raises(RuntimeError, match="malformed"):
        client.get(refresh=True)


def test_context_client_refresh_recovers_after_go_becomes_available():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json=_context_payload([]), request=request)

    client = _context_client(handler)

    with pytest.raises(RuntimeError, match="status 503"):
        client.get(refresh=True)
    assert client.get(refresh=True).ml_analytics_capabilities == ()


def test_capability_checker_requires_exact_engine_match_and_recovers():
    matching = NwdafContext(
        nf_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://go.example",
        internal_api_root="http://go-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=FLCapabilityType.SERVER_AND_CLIENT,
            ),
        ),
    )
    client = SequenceContextClient(
        [
            RuntimeError("Go is starting"),
            NwdafContext(
                nf_instance_id=matching.nf_instance_id,
                api_root=matching.api_root,
                internal_api_root=matching.internal_api_root,
                ml_analytics_capabilities=(
                    MLAnalyticsCapability(
                        ml_analytics_ids=("UE_COMMUNICATION",),
                        fl_capability_type=FLCapabilityType.SERVER,
                    ),
                ),
            ),
            matching,
        ]
    )
    checker = CapabilityConsistencyChecker(
        client,
        configured_server=True,
        configured_client=True,
    )

    unavailable = checker.check()
    mismatch = checker.check()
    verified = checker.check()

    assert unavailable.status == "unavailable"
    assert unavailable.advertised_server is None
    assert mismatch.status == "mismatch"
    assert mismatch.advertised_server is True
    assert mismatch.advertised_client is False
    assert verified.status == "verified"
    assert verified.ready is True
    assert client.refresh_values == [True, True, True]


@pytest.mark.parametrize(
    ("configured_server", "configured_client", "capability"),
    [
        (False, False, None),
        (True, False, FLCapabilityType.SERVER),
        (False, True, FLCapabilityType.CLIENT),
        (True, True, FLCapabilityType.SERVER_AND_CLIENT),
    ],
)
def test_capability_checker_accepts_each_exact_engine_profile(
    configured_server,
    configured_client,
    capability,
):
    capabilities = (
        ()
        if capability is None
        else (
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=capability,
            ),
        )
    )
    context = NwdafContext(
        nf_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://go.example",
        internal_api_root="http://go-internal.example",
        ml_analytics_capabilities=capabilities,
    )
    checker = CapabilityConsistencyChecker(
        SequenceContextClient([context]),
        configured_server=configured_server,
        configured_client=configured_client,
    )

    assert checker.check().status == "verified"


class SequenceContextClient:
    def __init__(self, values):
        self.values = list(values)
        self.refresh_values = []

    def get(self, *, refresh=False):
        self.refresh_values.append(refresh)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _context_payload(capabilities):
    return {
        "nfInstanceId": "11111111-1111-4111-8111-111111111111",
        "apiRoot": "http://go.example",
        "internalApiRoot": "http://go-internal.example",
        "mlAnalyticsCapabilities": capabilities,
    }


def _context_client(handler):
    return NwdafContextClient(
        "http://go-internal.example",
        30,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
