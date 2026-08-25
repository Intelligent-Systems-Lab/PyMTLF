from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import FederatedLearningSettings, FLClientSettings, Settings


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
                        "mLFileAddr": {
                            "mLModelUrl": "http://server.example/base.tar.gz"
                        },
                    }
                ],
                "mLModelTrainInfos": [
                    {
                        "dataAvReq": {
                            "inpEvents": [
                                {"upfEvent": "USER_DATA_USAGE_TRENDS"}
                            ],
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
