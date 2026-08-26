from unittest.mock import Mock

import httpx
import pytest

from py_mtlf.config import MongoDatasetSettings, PrivateCollectionProfileSettings
from py_mtlf.core.collection_relay import (
    CollectionRelayClient,
    CollectionRelayError,
    ServingSmfTarget,
)
from py_mtlf.core.nwdaf_context import NwdafContext
from py_mtlf.wire.private import SelectedTarget


def relay_client(handler):
    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="11111111-1111-4111-8111-111111111111",
        containing_nwdaf_process_instance_id=(
            "22222222-2222-4222-8222-222222222222"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf.internal",
    )
    return CollectionRelayClient(
        context,
        Mock(),
        1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def target() -> ServingSmfTarget:
    return ServingSmfTarget(
        supi="imsi-466920000000001",
        smf_nf_instance_id="33333333-3333-4333-8333-333333333333",
        api_root="http://smf.example",
        pdu_session_id=10,
        dnn="internet",
        snssai={"sst": 1, "sd": "010203"},
    )


def collection_profile() -> PrivateCollectionProfileSettings:
    return PrivateCollectionProfileSettings.model_validate(
        {
            "profile_id": "profile-a",
            "ml_event": "UE_COMMUNICATION",
            "ml_event_filter": {},
            "target_ue": {"intGroupIds": ["group-a.example"]},
            "network_area": {
                "tais": [
                    {
                        "plmn_id": {"mcc": "466", "mnc": "92"},
                        "tac": "001101",
                    }
                ]
            },
            "dnns": ["internet"],
            "snssais": [{"sst": 1, "sd": "010203"}],
            "sampling_interval_seconds": 2,
        }
    )


def test_discovery_derives_registered_service_origin_without_api_prefix():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/v1/nrf/nf-instances"
        return httpx.Response(
            200,
            json={
                "nfInstances": [
                    {
                        "nfInstanceId": "33333333-3333-4333-8333-333333333333",
                        "nfStatus": "REGISTERED",
                        "nfServices": [
                            {
                                "serviceName": "nsmf-event-exposure",
                                "nfServiceStatus": "REGISTERED",
                                "scheme": "http",
                                "ipEndPoints": [
                                    {"ipv4Address": "127.0.0.8", "port": 8080}
                                ],
                            }
                        ],
                    }
                ]
            },
        )

    relay = relay_client(handler)

    assert relay._discover_one("SMF", "nsmf-event-exposure", {}) == (
        "33333333-3333-4333-8333-333333333333",
        "http://127.0.0.8:8080",
    )


def test_exact_discovery_rejects_different_nf_instance():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "nfInstances": [
                    {
                        "nfInstanceId": "99999999-9999-4999-8999-999999999999",
                        "nfStatus": "REGISTERED",
                        "nfServices": [
                            {
                                "serviceName": "nsmf-event-exposure",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://wrong-smf.example",
                            }
                        ],
                    }
                ]
            },
        )

    relay = relay_client(handler)

    with pytest.raises(CollectionRelayError) as captured:
        relay._discover_one(
            "SMF",
            "nsmf-event-exposure",
            {"target-nf-instance-id": "33333333-3333-4333-8333-333333333333"},
        )

    assert captured.value.cause == "TARGET_UNAVAILABLE"


def test_profile_resolution_rejects_wrong_group_without_tai_query():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/v1/nrf/nf-instances":
            assert request.url.params["service-names"] == "nudm-sdm"
            assert not any("tai" in key.lower() for key in request.url.params)
            return httpx.Response(
                200,
                json={
                    "nfInstances": [
                        {
                            "nfInstanceId": "88888888-8888-4888-8888-888888888888",
                            "nfStatus": "REGISTERED",
                            "nfServices": [
                                {
                                    "serviceName": "nudm-sdm",
                                    "nfServiceStatus": "REGISTERED",
                                    "apiPrefix": "http://udm.example",
                                }
                            ],
                        }
                    ]
                },
            )
        assert request.url.path == "/internal/v1/udm-sdm/group-data/group-identifiers"
        assert request.url.params["int-group-id"] == "group-a.example"
        assert not any("tai" in key.lower() for key in request.url.params)
        return httpx.Response(
            200,
            json={
                "intGroupId": "different-group.example",
                "ueIdList": [{"supi": "imsi-1"}],
            },
        )

    relay = relay_client(handler)

    with pytest.raises(CollectionRelayError) as captured:
        relay.resolve_profile(collection_profile())

    assert captured.value.cause == "INVALID_PEER_RESPONSE"


@pytest.mark.parametrize(
    "mutation",
    [
        {"notifId": "different-correlation"},
        {"expiry": "2026-08-26T10:00:00Z"},
        {"subId": "different-subscription"},
        {"notifUri": "http://wrong-callback.example/upf"},
        {"eventSubs": [{"event": "UPF_EVENT", "upfEvents": []}]},
        {
            "eventSubs": [
                {
                    "event": "UPF_EVENT",
                    "networkArea": {
                        "tais": [
                            {
                                "plmnId": {"mcc": "466", "mnc": "92"},
                                "tac": "001102",
                            }
                        ]
                    },
                }
            ]
        },
    ],
)
def test_invalid_accepted_subscription_preserves_provisional_cleanup_identity(
    mutation,
):
    correlation = "44444444-4444-4444-8444-444444444444"
    requested = {
        "notifId": correlation,
        "notifUri": "http://py-mtlf.example/callbacks/upf-event-exposure",
        "eventSubs": [
            {
                "event": "UPF_EVENT",
                "networkArea": {
                    "tais": [
                        {
                            "plmnId": {"mcc": "466", "mnc": "92"},
                            "tac": "001101",
                        }
                    ]
                },
            }
        ],
    }
    representation = {
        **requested,
        "subId": "peer-subscription",
        **mutation,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            headers={
                "Location": (
                    "http://smf.example/nsmf-event-exposure/v1/subscriptions/"
                    "peer-subscription"
                )
            },
            json=representation,
        )

    relay = relay_client(handler)

    with pytest.raises(CollectionRelayError) as captured:
        relay.create_subscription(target(), correlation, requested)

    provisional = captured.value.provisional_subscription
    assert provisional is not None
    assert provisional.subscription_id == "peer-subscription"
    assert provisional.target.api_root == "http://smf.example"


def test_http_transport_failure_is_normalized_as_retryable_relay_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    relay = relay_client(handler)

    with pytest.raises(CollectionRelayError) as captured:
        relay._get("/internal/v1/nrf/nf-instances", "")

    assert captured.value.cause == "PEER_REQUEST_FAILED"
    assert captured.value.retryable is True


def test_adrf_resolution_failure_is_normalized_for_durable_inbox_retry():
    context = Mock()
    resolver = Mock()
    resolver.resolve_data.side_effect = RuntimeError("NRF temporarily unavailable")
    relay = CollectionRelayClient(
        context,
        resolver,
        1,
        client=httpx.Client(transport=httpx.MockTransport(lambda _request: None)),
    )

    with pytest.raises(CollectionRelayError) as captured:
        relay.store_record({}, supi="imsi-1", measurement_time=None)

    assert captured.value.cause == "STORAGE_UNAVAILABLE"
    assert captured.value.retryable is True


def test_adrf_store_requires_created_location_and_rejects_permanent_error():
    resolver = Mock()
    resolver.resolve_data.return_value = SelectedTarget(
        nfInstanceId="77777777-7777-4777-8777-777777777777",
        nfServiceInstanceId="adrf-data",
        serviceName="nadrf-datamanagement",
        apiRoot="http://adrf.example",
        selectionSource="NRF",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/v1/adrf-data-management/data-store-records"
        assert request.headers["Target-Api-Root"] == "http://adrf.example"
        return httpx.Response(400, json={"status": 400, "cause": "INVALID_MSG_FORMAT"})

    relay = relay_client(handler)
    relay._adrf_resolver = resolver

    with pytest.raises(CollectionRelayError) as captured:
        relay.store_record({}, supi="imsi-1", measurement_time=None)

    assert captured.value.cause == "ADRF_RECORD_REJECTED"
    assert captured.value.retryable is False


def test_adrf_store_rejects_malformed_created_location():
    resolver = Mock()
    resolver.resolve_data.return_value = SelectedTarget(
        nfInstanceId="77777777-7777-4777-8777-777777777777",
        nfServiceInstanceId="adrf-data",
        serviceName="nadrf-datamanagement",
        apiRoot="http://adrf.example",
        selectionSource="NRF",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            headers={"Location": "http://user:password@adrf.example/record-a"},
            json={"dataSub": [], "dataNotif": {}},
        )

    relay = relay_client(handler)
    relay._adrf_resolver = resolver

    with pytest.raises(CollectionRelayError) as captured:
        relay.store_record({}, supi="imsi-1", measurement_time=None)

    assert captured.value.cause == "STORAGE_UNAVAILABLE"


def test_transient_adrf_failure_falls_back_to_indexed_mongodb(monkeypatch):
    resolver = Mock()
    resolver.resolve_data.return_value = SelectedTarget(
        nfInstanceId="77777777-7777-4777-8777-777777777777",
        nfServiceInstanceId="adrf-data",
        serviceName="nadrf-datamanagement",
        apiRoot="http://adrf.example",
        selectionSource="NRF",
    )
    collection = Mock()
    database = Mock()
    database.__getitem__ = Mock(return_value=collection)
    mongo = Mock()
    mongo.__getitem__ = Mock(return_value=database)
    monkeypatch.setattr("pymongo.MongoClient", Mock(return_value=mongo))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    context = Mock()
    context.get.return_value = NwdafContext(
        nf_instance_id="11111111-1111-4111-8111-111111111111",
        containing_nwdaf_process_instance_id=(
            "22222222-2222-4222-8222-222222222222"
        ),
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf.internal",
    )
    relay = CollectionRelayClient(
        context,
        resolver,
        1,
        mongo_settings=MongoDatasetSettings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    record = {"dataSub": [{"smfDataSub": {}}], "dataNotif": {"upfEventNotifs": []}}

    receipt = relay.store_record(record, supi="imsi-1", measurement_time=None)

    assert receipt.transport == "mongodb"
    collection.create_index.assert_called_once_with(
        [("supi", 1), ("measurementTime", 1)]
    )
    collection.insert_one.assert_called_once()
    mongo.close.assert_called_once()


def test_transient_adrf_failure_without_mongodb_is_retryable():
    resolver = Mock()
    resolver.resolve_data.return_value = SelectedTarget(
        nfInstanceId="77777777-7777-4777-8777-777777777777",
        nfServiceInstanceId="adrf-data",
        serviceName="nadrf-datamanagement",
        apiRoot="http://adrf.example",
        selectionSource="NRF",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    relay = relay_client(handler)
    relay._adrf_resolver = resolver

    with pytest.raises(CollectionRelayError) as captured:
        relay.store_record({}, supi="imsi-1", measurement_time=None)

    assert captured.value.cause == "STORAGE_UNAVAILABLE"
    assert captured.value.retryable is True
