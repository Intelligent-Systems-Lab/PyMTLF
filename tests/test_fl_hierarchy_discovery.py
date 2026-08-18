import json

import httpx
import pytest
from nwdaf_context import context_client

from py_mtlf.config import FederatedLearningSettings
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    HierarchyNodeResolver,
    HierarchyNodeRole,
)

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_ID = "00000000-0000-4000-8000-000000000010"
LEAF_ID = "00000000-0000-4000-8000-000000000101"


def profile(
    nf_instance_id: str,
    *,
    capability: str,
    status: str = "REGISTERED",
    services: list[dict] | None = None,
) -> dict:
    return {
        "nfInstanceId": nf_instance_id,
        "nfStatus": status,
        "nwdafInfo": {
            "mlAnalyticsList": [
                {
                    "mlAnalyticsIds": ["UE_COMMUNICATION"],
                    "flCapabilityType": capability,
                    "mlModelInterInfo": {"vendorList": ["001122"]},
                }
            ]
        },
        "nfServices": services
        if services is not None
        else [
            {
                "serviceInstanceId": f"training-{nf_instance_id}",
                "serviceName": "nnwdaf-mlmodeltraining",
                "nfServiceStatus": "REGISTERED",
                "apiPrefix": f"http://{nf_instance_id}.example",
            }
        ],
    }


def resolver(handler) -> tuple[HierarchyNodeResolver, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    context = context_client(
        nf_instance_id=ROOT_ID,
        api_root="http://root.example",
        internal_api_root="http://go-internal.example",
    )
    return (
        HierarchyNodeResolver(FederatedLearningSettings(), context, client),
        client,
    )


@pytest.mark.parametrize(
    ("role", "target_id", "capability", "query_capability"),
    [
        (HierarchyNodeRole.BRANCH, BRANCH_ID, "FL_SERVER_AND_CLIENT", "FL_SERVER_AND_CLIENT"),
        (HierarchyNodeRole.LEAF, LEAF_ID, "FL_CLIENT", "FL_CLIENT"),
        (HierarchyNodeRole.LEAF, LEAF_ID, "FL_SERVER_AND_CLIENT", "FL_CLIENT"),
    ],
)
def test_exact_hierarchy_discovery_resolves_required_capabilities(
    role,
    target_id,
    capability,
    query_capability,
):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/v1/nrf/nf-instances"
        params = request.url.params
        assert params["target-nf-type"] == "NWDAF"
        assert params["requester-nf-type"] == "NWDAF"
        assert params["target-nf-instance-id"] == target_id
        assert params["service-names"] == "nnwdaf-mlmodeltraining"
        assert json.loads(params["ml-analytics-info-list"]) == [
            {
                "flCapabilityType": query_capability,
                "mlAnalyticsIds": ["UE_COMMUNICATION"],
                "mlModelInterInfo": {"vendorList": ["001122"]},
            }
        ]
        return httpx.Response(
            200,
            json={"validityPeriod": 60, "nfInstances": [profile(target_id, capability=capability)]},
            request=request,
        )

    instance, client = resolver(handler)

    resolved = instance.resolve(
        nf_instance_id=target_id,
        role=role,
        ml_event="UE_COMMUNICATION",
        model_interoperability="001122",
    )

    assert resolved.nf_instance_id == target_id
    assert resolved.role is role
    assert resolved.target.nf_instance_id == target_id
    assert resolved.target.service_name == "nnwdaf-mlmodeltraining"
    assert resolved.target.api_root == f"http://{target_id}.example"
    instance.close()
    client.close()


@pytest.mark.parametrize(
    "profiles",
    [
        [],
        [profile(LEAF_ID, capability="FL_CLIENT")],
        [profile(BRANCH_ID, capability="FL_SERVER_AND_CLIENT", status="SUSPENDED")],
        [profile(BRANCH_ID, capability="FL_CLIENT")],
        [
            profile(
                BRANCH_ID,
                capability="FL_SERVER_AND_CLIENT",
                services=[],
            )
        ],
        [
            profile(BRANCH_ID, capability="FL_SERVER_AND_CLIENT"),
            profile(BRANCH_ID, capability="FL_SERVER_AND_CLIENT"),
        ],
        [
            profile(
                BRANCH_ID,
                capability="FL_SERVER_AND_CLIENT",
                services=[
                    {
                        "serviceInstanceId": "training-a",
                        "serviceName": "nnwdaf-mlmodeltraining",
                        "nfServiceStatus": "REGISTERED",
                        "apiPrefix": "http://branch-a.example",
                    },
                    {
                        "serviceInstanceId": "training-b",
                        "serviceName": "nnwdaf-mlmodeltraining",
                        "nfServiceStatus": "REGISTERED",
                        "apiPrefix": "http://branch-b.example",
                    },
                ],
            )
        ],
    ],
)
def test_hierarchy_discovery_rejects_missing_ambiguous_or_ineligible_target(profiles):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"validityPeriod": 60, "nfInstances": profiles},
            request=request,
        )

    instance, client = resolver(handler)

    with pytest.raises(HierarchyDiscoveryError):
        instance.resolve(
            nf_instance_id=BRANCH_ID,
            role=HierarchyNodeRole.BRANCH,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
        )
    instance.close()
    client.close()


def test_hierarchy_discovery_rejects_malformed_search_result_and_peer_error():
    responses = iter(
        [
            (200, {"validityPeriod": "60", "nfInstances": []}),
            (503, {"status": 503}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        status, payload = next(responses)
        return httpx.Response(status, json=payload, request=request)

    instance, client = resolver(handler)
    for _ in range(2):
        with pytest.raises(HierarchyDiscoveryError):
            instance.resolve(
                nf_instance_id=BRANCH_ID,
                role=HierarchyNodeRole.BRANCH,
                ml_event="UE_COMMUNICATION",
                model_interoperability="001122",
            )
    instance.close()
    client.close()
