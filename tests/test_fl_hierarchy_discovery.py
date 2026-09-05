import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from nwdaf_context import context_client

from py_mtlf.config import FederatedLearningSettings
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    HierarchyNodeResolver,
    HierarchyNodeRole,
)
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainSubsc, RequirementsError

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_ID = "00000000-0000-4000-8000-000000000010"
LEAF_ID = "00000000-0000-4000-8000-000000000101"


def profile(
    nf_instance_id: str,
    *,
    capability: str,
    status: str = "REGISTERED",
    services: list[dict] | None = None,
    tracking_areas: list[dict] | None = None,
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
                    **(
                        {"trackingAreaList": tracking_areas}
                        if tracking_areas is not None
                        else {}
                    ),
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


def resolver(handler, *, clock=None) -> tuple[HierarchyNodeResolver, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    context = context_client(
        nf_instance_id=ROOT_ID,
        api_root="http://root.example",
        internal_api_root="http://go-internal.example",
    )
    return (
        HierarchyNodeResolver(
            FederatedLearningSettings(),
            context,
            client,
            **({"clock": clock} if clock is not None else {}),
        ),
        client,
    )


def training_subscription(tai: dict | None) -> NwdafMLModelTrainSubsc:
    return NwdafMLModelTrainSubsc.model_validate(
        {
            "mLEventSubscs": [
                {
                    "mLEvent": "UE_COMMUNICATION",
                    "mLEventFilter": {
                        "networkArea": {"tais": [] if tai is None else [tai]}
                    },
                    "modelInterInfo": "001122",
                }
            ],
            "notifUri": "http://root.example/callback",
            "notifCorreId": "root-callback",
            "mlCorreId": "fl-process-001",
            "mLPreFlag": True,
            "mLModelInfos": [
                {
                    "event": "UE_COMMUNICATION",
                    "mLFileAddr": {"mLModelUrl": "http://root.example/model.tar.gz"},
                }
            ],
            "eventReq": {"notifMethod": "ON_EVENT_DETECTION"},
            "mLModelTrainInfos": [
                {
                    "dataAvReq": {
                        "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                        "minNumSamples": 1,
                    },
                    "timeAvReq": "PT5M",
                }
            ],
        }
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
    observed_at = datetime(2026, 9, 4, 9, 0, tzinfo=UTC)

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

    instance, client = resolver(handler, clock=lambda: observed_at)

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
    assert resolved.discovery_scope is not None
    assert resolved.discovery_scope.target_nf_instance_id == target_id
    assert resolved.observed_at == observed_at
    assert resolved.valid_until == observed_at + timedelta(seconds=60)
    instance.close()
    client.close()


def test_image_training_contract_maps_to_nrf_vendor_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.url.params["ml-analytics-info-list"])[0]
        assert query["mlAnalyticsIds"] == ["X_IMAGE_CLASSIFICATION"]
        assert query["mlModelInterInfo"] == {"vendorList": ["001122"]}
        value = profile(LEAF_ID, capability="FL_CLIENT")
        value["nwdafInfo"]["mlAnalyticsList"][0]["mlAnalyticsIds"] = [
            "X_IMAGE_CLASSIFICATION"
        ]
        return httpx.Response(
            200,
            json={"validityPeriod": 60, "nfInstances": [value]},
            request=request,
        )

    instance, client = resolver(handler)
    resolved = instance.resolve(
        nf_instance_id=LEAF_ID,
        role=HierarchyNodeRole.LEAF,
        ml_event="X_IMAGE_CLASSIFICATION",
        model_interoperability="pymtlf-image-classification-mnist",
    )

    assert resolved.nf_instance_id == LEAF_ID
    assert resolved.discovery_scope is not None
    assert (
        resolved.discovery_scope.model_interoperability
        == "pymtlf-image-classification-mnist"
    )
    instance.close()
    client.close()


def test_hierarchy_discovery_rejects_unmapped_non_vendor_interoperability() -> None:
    instance, client = resolver(
        lambda request: httpx.Response(200, json={}, request=request)
    )

    with pytest.raises(ValueError, match="VendorId mapping"):
        instance.resolve(
            nf_instance_id=LEAF_ID,
            role=HierarchyNodeRole.LEAF,
            ml_event="X_IMAGE_CLASSIFICATION",
            model_interoperability="unknown-image-contract",
        )
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


def test_list_discovery_uses_bounded_query_and_filters_profiles_deterministically():
    tai = {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "00ab12"}
    other_tai = {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "00FF12"}
    eligible_b = "00000000-0000-4000-8000-000000000202"
    eligible_a = "00000000-0000-4000-8000-000000000201"
    ambiguous = "00000000-0000-4000-8000-000000000203"
    excluded = "00000000-0000-4000-8000-000000000204"
    suspended = "00000000-0000-4000-8000-000000000205"
    wrong_capability = "00000000-0000-4000-8000-000000000206"

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert "target-nf-instance-id" not in params
        assert json.loads(params["ml-analytics-info-list"]) == [
            {
                "flCapabilityType": "FL_CLIENT",
                "mlAnalyticsIds": ["UE_COMMUNICATION"],
                "mlModelInterInfo": {"vendorList": ["001122"]},
                "trackingAreaList": [
                    {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "00AB12"}
                ],
            }
        ]
        return httpx.Response(
            200,
            json={
                "validityPeriod": 60,
                "nfInstances": [
                    profile(eligible_b, capability="FL_CLIENT", tracking_areas=[tai]),
                    profile(eligible_a, capability="FL_CLIENT", tracking_areas=[tai]),
                    profile(eligible_a, capability="FL_CLIENT", tracking_areas=[tai]),
                    profile(ROOT_ID, capability="FL_CLIENT", tracking_areas=[tai]),
                    profile(LEAF_ID, capability="FL_CLIENT", tracking_areas=[other_tai]),
                    profile(excluded, capability="FL_CLIENT", tracking_areas=[tai]),
                    profile(
                        suspended,
                        capability="FL_CLIENT",
                        status="SUSPENDED",
                        tracking_areas=[tai],
                    ),
                    profile(
                        wrong_capability,
                        capability="FL_SERVER",
                        tracking_areas=[tai],
                    ),
                    profile(
                        ambiguous,
                        capability="FL_CLIENT",
                        tracking_areas=[tai],
                        services=[
                            {
                                "serviceInstanceId": "training-a",
                                "serviceName": "nnwdaf-mlmodeltraining",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://a.example",
                            },
                            {
                                "serviceInstanceId": "training-b",
                                "serviceName": "nnwdaf-mlmodeltraining",
                                "nfServiceStatus": "REGISTERED",
                                "apiPrefix": "http://b.example",
                            },
                        ],
                    ),
                ],
            },
            request=request,
        )

    observed_at = datetime(2026, 9, 4, 10, 30, tzinfo=UTC)
    instance, client = resolver(handler, clock=lambda: observed_at)
    snapshot = instance.discover_for_subscription(
        training_subscription(tai),
        role=HierarchyNodeRole.LEAF,
        excluded_nf_instance_ids=(excluded,),
    )

    assert tuple(item.nf_instance_id for item in snapshot.nodes) == (
        eligible_a,
        eligible_b,
    )
    assert snapshot.scope.containing_nf_instance_id == ROOT_ID
    assert snapshot.scope.role is HierarchyNodeRole.LEAF
    assert snapshot.scope.ml_event == "UE_COMMUNICATION"
    assert snapshot.scope.model_interoperability == "001122"
    assert snapshot.scope.tracking_areas == (("001", "01", "00AB12"),)
    assert snapshot.observed_at == observed_at
    assert snapshot.validity_period == 60
    assert snapshot.valid_until == observed_at + timedelta(seconds=60)
    assert snapshot.returned_count == 9
    assert snapshot.complete_nf_instance_count is None
    assert snapshot.is_complete is True
    instance.close()
    client.close()


def test_list_discovery_preserves_partial_search_result_metadata():
    tai = {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "000001"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "validityPeriod": 15,
                "nfInstances": [
                    profile(LEAF_ID, capability="FL_CLIENT", tracking_areas=[tai])
                ],
                "numNfInstComplete": 3,
            },
            request=request,
        )

    observed_at = datetime(2026, 9, 4, 11, 0, tzinfo=UTC)
    instance, client = resolver(handler, clock=lambda: observed_at)

    snapshot = instance.discover(
        role=HierarchyNodeRole.LEAF,
        ml_event="UE_COMMUNICATION",
        model_interoperability="001122",
        tracking_areas=(tai,),
    )

    assert tuple(item.nf_instance_id for item in snapshot.nodes) == (LEAF_ID,)
    assert snapshot.returned_count == 1
    assert snapshot.complete_nf_instance_count == 3
    assert snapshot.is_complete is False
    assert snapshot.valid_until == observed_at + timedelta(seconds=15)
    instance.close()
    client.close()


def test_list_discovery_rejects_unbounded_scope_and_preserves_peer_failure():
    instance, client = resolver(
        lambda request: httpx.Response(503, json={"status": 503}, request=request)
    )

    with pytest.raises(ValueError, match="tracking areas"):
        instance.discover(
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
            tracking_areas=(),
        )
    with pytest.raises(HierarchyDiscoveryError, match="list discovery"):
        instance.discover(
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
            tracking_areas=(
                {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "000001"},
            ),
        )
    instance.close()
    client.close()


def test_subscription_discovery_maps_missing_bounded_scope_to_requirements_error():
    instance, client = resolver(
        lambda request: httpx.Response(200, json={}, request=request)
    )

    with pytest.raises(RequirementsError) as error:
        instance.discover_for_subscription(
            training_subscription(None),
            role=HierarchyNodeRole.LEAF,
        )

    assert error.value.violations[0].parameter.endswith("networkArea.tais")
    instance.close()
    client.close()


def test_list_discovery_rejects_malformed_search_result():
    instance, client = resolver(
        lambda request: httpx.Response(
            200,
            json={"validityPeriod": 60, "nfInstances": {}},
            request=request,
        )
    )

    with pytest.raises(HierarchyDiscoveryError, match="malformed SearchResult"):
        instance.discover(
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
            tracking_areas=(
                {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "000001"},
            ),
        )
    instance.close()
    client.close()


def test_list_discovery_rejects_inconsistent_complete_instance_count():
    tai = {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "000001"}
    instance, client = resolver(
        lambda request: httpx.Response(
            200,
            json={
                "validityPeriod": 60,
                "nfInstances": [
                    profile(LEAF_ID, capability="FL_CLIENT", tracking_areas=[tai])
                ],
                "numNfInstComplete": 0,
            },
            request=request,
        )
    )

    with pytest.raises(HierarchyDiscoveryError, match="malformed SearchResult"):
        instance.discover(
            role=HierarchyNodeRole.LEAF,
            ml_event="UE_COMMUNICATION",
            model_interoperability="001122",
            tracking_areas=(tai,),
        )
    instance.close()
    client.close()
