from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import FederatedLearningSettings, RuntimeSettings


def test_training_requirements_failure_identifies_invalid_parameters(
    settings,
    tmp_path,
):
    configured = settings.model_copy(
        update={
            "runtime": RuntimeSettings(mode="fl_client"),
            "federated_learning": FederatedLearningSettings(
                workspace_root=tmp_path / "fl-client",
                public_base_url=settings.artifact.public_base_url,
                model_interoperability_ids=("001122",),
            ),
        }
    )
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
