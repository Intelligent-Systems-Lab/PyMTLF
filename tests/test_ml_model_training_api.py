from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import FederatedLearningSettings, FLClientSettings, Settings
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)


def candidate_payload() -> dict:
    return {
        "mLEventSubscs": [
            {
                "mLEvent": "UE_COMMUNICATION",
                "mLEventFilter": {},
                "modelInterInfo": "001122",
            }
        ],
        "notifUri": "http://server.example/callback",
        "notifCorreId": "correlation-1",
        "suppFeats": "4",
        "mlCorreId": "training-1",
        "mLPreFlag": True,
        "mLModelTrainInfos": [
            {
                "dataAvReq": {"inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}]},
                "timeAvReq": "PT5M",
            }
        ],
        "x-flTopology": {
            "nfInstanceId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "children": [
                {
                    "nfInstanceId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                    "priority": 100,
                }
            ],
            "policy": {
                "selectionMethod": "priority",
                "minAvailableNodes": 1,
                "minTrainNodes": 1,
            },
        },
    }


def candidate_settings(settings, tmp_path) -> Settings:
    payload = settings.model_dump(mode="python")
    payload["runtime"] = {"mode": "federated"}
    payload["local_training"] = None
    payload["federated_learning"] = FederatedLearningSettings(
        workspace_root=tmp_path / "fl-client",
        public_base_url=settings.artifact.public_base_url,
        client=FLClientSettings(
            training_data={"collection_trigger": "consumer_subscription"},
            model_interoperability_ids=("001122",),
        ),
    )
    return Settings.model_validate(payload)


def candidate_context() -> Mock:
    client = Mock()
    client.get.return_value = NwdafContext(
        nf_instance_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        containing_nwdaf_process_instance_id="11111111-1111-4111-8111-111111111111",
        api_root="http://nwdaf.example",
        internal_api_root="http://nwdaf-internal.example",
        ml_analytics_capabilities=(
            MLAnalyticsCapability(
                ml_analytics_ids=("UE_COMMUNICATION",),
                fl_capability_type=FLCapabilityType.CLIENT,
            ),
        ),
    )
    return client


def test_training_create_requires_go_resource_id_and_does_not_replace_existing(
    settings,
    tmp_path,
):
    configured = candidate_settings(settings, tmp_path)
    resource_id = str(uuid4())
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        for value in (None, "not-a-uuid"):
            response = client.post(
                "/internal/v1/ml-model-training/subscriptions",
                headers={"X-NWDAF-Subscription-Id": value} if value else {},
                json=candidate_payload(),
            )
            assert response.status_code == 400

        first = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": resource_id},
            json=candidate_payload(),
        )
        assert first.status_code == 201
        assert first.headers["Location"].endswith("/" + resource_id)

        duplicate_payload = candidate_payload()
        duplicate_payload["notifCorreId"] = "correlation-2"
        duplicate = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": resource_id},
            json=duplicate_payload,
        )
        assert duplicate.status_code == 403
        assert client.app.state.fl_client.get(resource_id).subscription_id == resource_id


def test_training_requirements_failure_identifies_invalid_parameters(
    settings,
    tmp_path,
):
    payload = settings.model_dump(mode="python")
    payload["runtime"] = {"mode": "federated"}
    payload["local_training"] = None
    payload["federated_learning"] = FederatedLearningSettings(
        workspace_root=tmp_path / "fl-client",
        public_base_url=settings.artifact.public_base_url,
        client=FLClientSettings(
            training_data={"collection_trigger": "consumer_subscription"},
            model_interoperability_ids=("001122",),
        ),
    )
    configured = Settings.model_validate(payload)
    with TestClient(create_app(configured)) as client:
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json={
                "mLEventSubscs": [
                    {
                        "mLEvent": "UE_COMMUNICATION",
                        "mLEventFilter": {},
                    }
                ],
                "notifUri": "http://server.example/callback",
                "notifCorreId": "correlation-1",
                "mLPreFlag": True,
            },
        )

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["cause"] == "ML_MODEL_TRAINING_REQS_NOT_MET"
    assert response.json()["invalidParams"] == [
        {
            "param": "mlCorreId",
            "reason": "is required for federated learning",
        },
        {
            "param": "mLEventSubscs[0].modelInterInfo",
            "reason": "is required for federated learning",
        },
        {
            "param": "mLModelTrainInfos",
            "reason": "is required for training preparation",
        },
    ]


def test_training_admission_is_unavailable_without_containing_go_generation(
    settings,
    tmp_path,
):
    payload = settings.model_dump(mode="python")
    payload["runtime"] = {"mode": "federated"}
    payload["local_training"] = None
    payload["federated_learning"] = FederatedLearningSettings(
        workspace_root=tmp_path / "fl-client",
        public_base_url=settings.artifact.public_base_url,
        client=FLClientSettings(
            training_data={"collection_trigger": "consumer_subscription"},
            model_interoperability_ids=("001122",),
        ),
    )
    configured = Settings.model_validate(payload)
    with TestClient(create_app(configured)) as client:
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json={
                "mLEventSubscs": [
                    {
                        "mLEvent": "UE_COMMUNICATION",
                        "mLEventFilter": {},
                        "modelInterInfo": "001122",
                    }
                ],
                "notifUri": "http://server.example/callback",
                "notifCorreId": "correlation-1",
                "mlCorreId": "training-1",
                "mLPreFlag": True,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {"mLModelUrl": "http://server.example/base.tar.gz"},
                    }
                ],
                "mLModelTrainInfos": [
                    {
                        "dataAvReq": {
                            "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                            "minNumSamples": 1,
                            "timeWindows": [
                                {
                                    "startTime": "2026-07-01T00:00:00Z",
                                    "stopTime": "2026-07-02T00:00:00Z",
                                }
                            ],
                        },
                        "timeAvReq": "PT5M",
                    }
                ],
            },
        )

    assert response.status_code == 503
    assert response.json()["cause"] == "UNAVAILABLE_ML_MODEL_TRAINING_FOR_ALLEVENTS"


def test_candidate_create_returns_lossless_persistent_contract_without_feature(
    settings,
    tmp_path,
):
    configured = candidate_settings(settings, tmp_path)
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=candidate_payload(),
        )

        assert response.status_code == 201
        assert "/subscriptions/" in response.headers["location"]
        representation = response.json()
        assert representation["suppFeats"] == ""
        assert representation["x-flTopology"]["nfInstanceId"].startswith("aaaaaaaa")
        assert "x-retainedResultReq" not in representation
        assert "retainedResultReq" not in representation["x-flTopology"]["children"][0]


def test_candidate_retained_result_instruction_is_rejected_atomically(settings, tmp_path):
    configured = candidate_settings(settings, tmp_path)
    payload = candidate_payload()
    payload["x-retainedResultReq"] = True
    payload["x-flTopology"]["children"][0]["retainedResultReq"] = True
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        rejected = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=payload,
        )
        valid = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=candidate_payload(),
        )

    assert rejected.status_code == 403
    assert rejected.json()["cause"] == "ML_MODEL_TRAINING_REQS_NOT_MET"
    assert valid.status_code == 201


def test_candidate_nested_validation_reports_alias_path(settings, tmp_path):
    configured = candidate_settings(settings, tmp_path)
    payload = candidate_payload()
    payload["x-flTopology"]["strategy"] = {
        "method": "fedProx",
        "aggregation": "sampleWeighted",
        "methodParameters": {"proximalMu": 0.01, "unknown": True},
    }
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["cause"] == "INVALID_MSG_FORMAT"
    assert response.json()["invalidParams"] == [
        {
            "param": "x-flTopology.strategy.methodParameters.unknown",
            "reason": "Extra inputs are not permitted",
        }
    ]


def test_candidate_patch_is_rejected_when_feature_was_not_negotiated(settings, tmp_path):
    configured = candidate_settings(settings, tmp_path)
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        created = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=candidate_payload(),
        )
        response = client.patch(
            created.headers["location"],
            json={"x-flTopology": {"policy": {"minTrainNodes": 1}}},
            headers={"Content-Type": "application/merge-patch+json"},
        )

    assert created.status_code == 201
    assert response.status_code == 403
    assert response.json()["cause"] == "ML_MODEL_TRAINING_REQS_NOT_MET"
    assert response.json()["invalidParams"] == [
        {
            "param": "suppFeats",
            "reason": "HierarchicalFLOrch was not negotiated for this resource",
        }
    ]


def test_candidate_put_is_rejected_when_feature_was_not_negotiated(settings, tmp_path):
    configured = candidate_settings(settings, tmp_path)
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        created = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=candidate_payload(),
        )
        replacement = candidate_payload()
        replacement["x-flTopology"]["children"][0]["priority"] = 80
        response = client.put(created.headers["location"], json=replacement)

    assert created.status_code == 201
    assert response.status_code == 403
    assert response.json()["cause"] == "ML_MODEL_TRAINING_REQS_NOT_MET"
    assert response.json()["invalidParams"][0]["param"] == "suppFeats"


def test_candidate_receiver_mismatch_is_structured_bad_request(settings, tmp_path):
    configured = candidate_settings(settings, tmp_path)
    payload = candidate_payload()
    payload["x-flTopology"]["nfInstanceId"] = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    with TestClient(create_app(configured, nwdaf_context_client=candidate_context())) as client:
        response = client.post(
            "/internal/v1/ml-model-training/subscriptions",
            headers={"X-NWDAF-Subscription-Id": str(uuid4())},
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["cause"] == "INVALID_MSG_FORMAT"
    assert response.json()["invalidParams"] == [
        {
            "param": "x-flTopology.nfInstanceId",
            "reason": "must identify the request receiver",
        }
    ]
