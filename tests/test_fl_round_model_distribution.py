import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from nwdaf_context import context_client

from py_mtlf.config import AdrfSettings
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_round_model_distribution import (
    RoundModelDistribution,
    RoundModelDistributionError,
)
from py_mtlf.wire.ml_model import MLModelAdrf
from py_mtlf.wire.private import SelectedTarget

ROOT_ID = "11111111-1111-4111-8111-111111111111"
ADRF_ID = "22222222-2222-4222-8222-222222222222"
BRANCH_A = "33333333-3333-4333-8333-333333333333"
BRANCH_B = "44444444-4444-4444-8444-444444444444"
LEAF_A = "55555555-5555-4555-8555-555555555555"
DIGEST = "a" * 64


def _context():
    return context_client(
        nf_instance_id=ROOT_ID,
        api_root="http://root.example",
        internal_api_root="http://root-go.example",
    )


def _target():
    return SelectedTarget(
        nfInstanceId=ADRF_ID,
        nfServiceInstanceId="adrf-model",
        serviceName="nadrf-mlmodelmanagement",
        apiRoot="http://adrf.example",
        selectionSource="NRF",
    )


def _artifact():
    return ArtifactMetadata(
        key=DIGEST,
        size_bytes=4096,
        path=Path("/tmp/model.tar.gz"),
        url=f"http://root.example/internal/v1/artifacts/{DIGEST}",
    )


def _record(model_id, consumers, *, store_id="store-a", owner=ROOT_ID):
    return {
        "nfInstanceId": owner,
        "mlModelInfo": [
            {
                "modelUniqueId": model_id,
                "mlFileAddr": {
                    "mLModelUrl": (
                        "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                        f"mlmodel-store-records/{store_id}/model"
                    )
                },
                "mlStorageSize": 4096,
                "allowConsumerList": [{"nfInstanceId": consumer} for consumer in consumers],
            }
        ],
        "modelStoreResult": {
            "modelUniqueId": model_id,
            "storeResult": "ML_MODEL_FILE_STORED_IN_ADRF",
        },
    }


def _distribution(handler, model_ids):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = Mock(resolve_model=Mock(return_value=_target()))
    values = iter(model_ids)
    distribution = RoundModelDistribution(
        AdrfSettings(),
        resolver,
        _context(),
        client=client,
        model_id_source=lambda: next(values),
    )
    return distribution, resolver, client


def test_round_record_store_update_retrieve_and_cleanup_use_go_proxy():
    requests: list[httpx.Request] = []
    current_consumers = [BRANCH_A, LEAF_A]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_consumers
        requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["nfInstanceId"] == ROOT_ID
            assert body["mlModelInfo"][0]["modelUniqueId"] == 73
            assert body["mlModelInfo"][0]["mlFileAddr"] == {"mLModelUrl": _artifact().url}
            assert body["mlModelInfo"][0]["allowConsumerList"] == [
                {"nfInstanceId": BRANCH_A},
                {"nfInstanceId": LEAF_A},
            ]
            return httpx.Response(
                201,
                request=request,
                headers={
                    "Location": (
                        "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                        "mlmodel-store-records/store-a"
                    )
                },
                json=_record(73, current_consumers),
            )
        if request.method == "PUT":
            body = json.loads(request.content)
            current_consumers = [
                item["nfInstanceId"] for item in body["mlModelInfo"][0]["allowConsumerList"]
            ]
            return httpx.Response(200, request=request, json=_record(73, current_consumers))
        if request.method == "GET":
            return httpx.Response(200, request=request, json=_record(73, current_consumers))
        return httpx.Response(204, request=request)

    distribution, resolver, client = _distribution(handler, [73])
    stored = distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=1,
        artifact=_artifact(),
        allowed_consumer_ids=[LEAF_A, BRANCH_A, LEAF_A],
    )

    assert stored.model_unique_id == 73
    assert stored.store_transaction_id == "store-a"
    assert stored.allowed_consumer_ids == (BRANCH_A, LEAF_A)
    assert stored.wire_reference == MLModelAdrf(adrfId=ADRF_ID, storTransId="store-a")

    updated = distribution.update_consumers(
        "procedure-a",
        1,
        [BRANCH_A, BRANCH_B, LEAF_A],
    )
    assert updated.allowed_consumer_ids == (BRANCH_A, BRANCH_B, LEAF_A)

    retrieved = distribution.retrieve(
        reference=stored.wire_reference,
        model_unique_id=73,
        consumer_nf_instance_id=BRANCH_B,
    )
    assert retrieved.model_url.endswith("/store-a/model")
    assert retrieved.model_size == 4096

    distribution.cleanup("procedure-a", 1)
    assert distribution.snapshot() == ()
    assert [(request.method, request.url.path) for request in requests] == [
        (
            "POST",
            "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records",
        ),
        (
            "PUT",
            "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records/store-a",
        ),
        (
            "GET",
            "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records",
        ),
        (
            "DELETE",
            "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records/store-a",
        ),
    ]
    assert requests[2].url.params["store-trans-id"] == "store-a"
    assert all(request.headers["Target-Api-Root"] == "http://adrf.example" for request in requests)
    assert resolver.resolve_model.call_args_list[-1].args == (ADRF_ID,)
    client.close()


def test_model_id_collision_retries_only_against_active_round_records():
    def handler(request: httpx.Request) -> httpx.Response:
        model_id = json.loads(request.content)["mlModelInfo"][0]["modelUniqueId"]
        store_id = f"store-{model_id}"
        return httpx.Response(
            201,
            request=request,
            headers={
                "Location": (
                    "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                    f"mlmodel-store-records/{store_id}"
                )
            },
            json=_record(model_id, [BRANCH_A], store_id=store_id),
        )

    distribution, _, client = _distribution(handler, [11, 11, 12])
    first = distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=1,
        artifact=_artifact(),
        allowed_consumer_ids=[BRANCH_A],
    )
    second = distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=2,
        artifact=_artifact(),
        allowed_consumer_ids=[BRANCH_A],
    )

    assert first.model_unique_id == 11
    assert second.model_unique_id == 12
    client.close()


@pytest.mark.parametrize(
    ("mutation", "status"),
    [("store", 500), ("update", 500), ("retrieve", 404), ("cleanup", 500)],
)
def test_round_record_transport_failure_does_not_claim_success(mutation, status):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201 if mutation != "store" else status,
                request=request,
                headers={
                    "Location": (
                        "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                        "mlmodel-store-records/store-a"
                    )
                },
                json=_record(9, [BRANCH_A]),
            )
        if request.method == "PUT":
            return httpx.Response(status if mutation == "update" else 204, request=request)
        if request.method == "GET":
            return httpx.Response(
                status if mutation == "retrieve" else 200,
                request=request,
                json=_record(9, [BRANCH_A]),
            )
        return httpx.Response(status if mutation == "cleanup" else 204, request=request)

    distribution, _, client = _distribution(handler, [9])
    if mutation == "store":
        with pytest.raises(RoundModelDistributionError, match="store returned status"):
            distribution.store(
                ml_correlation_id="procedure-a",
                round_indicator=1,
                artifact=_artifact(),
                allowed_consumer_ids=[BRANCH_A],
            )
        assert distribution.snapshot() == ()
        client.close()
        return

    stored = distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=1,
        artifact=_artifact(),
        allowed_consumer_ids=[BRANCH_A],
    )
    if mutation == "update":

        def operation():
            return distribution.update_consumers("procedure-a", 1, [BRANCH_A, BRANCH_B])

    elif mutation == "retrieve":

        def operation():
            return distribution.retrieve(
                reference=stored.wire_reference,
                model_unique_id=stored.model_unique_id,
                consumer_nf_instance_id=BRANCH_A,
            )

    else:

        def operation():
            return distribution.cleanup("procedure-a", 1)

    with pytest.raises(RoundModelDistributionError):
        operation()
    assert distribution.snapshot() == (stored,)
    client.close()


def test_retrieval_requires_exact_adrf_and_allowed_consumer():
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, request=request, json=_record(4, [BRANCH_A]))

    distribution, resolver, client = _distribution(handler, [4])
    resolver.resolve_model.return_value = None
    with pytest.raises(RoundModelDistributionError, match="resolved exactly"):
        distribution.retrieve(
            reference=MLModelAdrf(adrfId=ADRF_ID, storTransId="store-a"),
            model_unique_id=4,
            consumer_nf_instance_id=BRANCH_A,
        )
    assert requests == 0

    resolver.resolve_model.return_value = _target()
    with pytest.raises(RoundModelDistributionError, match="inconsistent record"):
        distribution.retrieve(
            reference=MLModelAdrf(adrfId=ADRF_ID, storTransId="store-a"),
            model_unique_id=4,
            consumer_nf_instance_id=BRANCH_B,
        )
    client.close()


def test_generation_abort_cleans_all_known_round_records():
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            model_id = json.loads(request.content)["mlModelInfo"][0]["modelUniqueId"]
            store_id = f"store-{model_id}"
            return httpx.Response(
                201,
                request=request,
                headers={
                    "Location": (
                        "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                        f"mlmodel-store-records/{store_id}"
                    )
                },
                json=_record(model_id, [BRANCH_A], store_id=store_id),
            )
        deleted.append(request.url.path)
        return httpx.Response(204, request=request)

    distribution, _, client = _distribution(handler, [21, 22])
    distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=1,
        artifact=_artifact(),
        allowed_consumer_ids=[BRANCH_A],
    )
    distribution.store(
        ml_correlation_id="procedure-a",
        round_indicator=2,
        artifact=_artifact(),
        allowed_consumer_ids=[BRANCH_A],
    )

    distribution.abort_generation("containing NWDAF generation changed")

    assert distribution.snapshot() == ()
    assert deleted == [
        "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records/store-21",
        "/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records/store-22",
    ]
    client.close()
