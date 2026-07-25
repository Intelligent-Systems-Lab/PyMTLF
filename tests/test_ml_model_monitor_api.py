from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.core.accuracy_policy import AccuracyPolicy
from py_mtlf.core.monitor_reconciler import PreparedMonitorRestore
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)


def registration_body() -> dict[str, object]:
    return {
        "consumerId": "11111111-1111-4111-8111-111111111111",
        "modelId": 1,
        "modelAccuInd": True,
        "mLEvent": "UE_COMMUNICATION",
        "mLEventFilter": {},
        "tgtUe": {"intGroupIds": ["group-a"]},
    }


def monitor_subscription() -> MLModelMonitorSubscription:
    return MLModelMonitorSubscription(
        modelIds=[1],
        notificationUri="http://127.0.0.1:9092/internal/v1/ml-model-monitor/notifications",
        notifCorrId="corr-a",
        modelMetric="ACCURACY",
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": ["group-a"]},
    )


class CatalogStub:
    provider_namespace = "local-mtlf"
    family_key = ("local-mtlf", "ue-communication-default")
    version_key = ("local-mtlf", 1)

    def version_key_for_id(self, model_id):
        return self.provider_namespace, model_id

    def family_for_version(self, version_key):
        return self.family_key if version_key == self.version_key else None

    def current(self, family_key):
        if family_key != self.family_key:
            return None
        return type("Current", (), {"version_key": self.version_key})()


def test_registration_resource_uses_standard_create_and_delete_semantics(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/internal/v1/ml-model-monitor/registrations",
            json=registration_body(),
        )
        deleted = client.delete(created.headers["location"])
        missing = client.delete(created.headers["location"])

    assert created.status_code == 201
    assert created.json() == registration_body()
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/problem+json")


def test_orphan_monitor_notification_does_not_update_accuracy_policy(settings):
    app = create_app(settings)
    orphan = monitor_subscription()
    app.state.monitor_subscriptions.upsert(
        "orphan-subscription",
        "deleted-registration",
        orphan,
    )
    before = app.state.accuracy_policy.snapshot()

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-monitor/notifications",
            json={
                "notifCorrId": orphan.notification_id,
                "modelAccuInfos": [
                    {
                        "modelId": 1,
                        "deviation": 0.9,
                    }
                ],
            },
        )

    assert response.status_code == 404
    assert app.state.accuracy_policy.snapshot() == before


def test_notification_correlates_subscription_and_updates_policy(settings):
    app = create_app(settings)
    app.state.accuracy_policy = AccuracyPolicy(settings.accuracy_policy, CatalogStub())
    registration = app.state.monitor_registrations.create(
        MLModelMonitorRegistration.model_validate(registration_body())
    )
    app.state.monitor_subscriptions.upsert(
        "monitor-a",
        registration.registration_id,
        monitor_subscription(),
    )
    app.state.monitor_reconciler.commit_restore(
        PreparedMonitorRestore(
            subscription_ids={registration.registration_id: "monitor-a"},
            orphan_subscription_ids=frozenset(),
        )
    )

    with TestClient(app) as client:
        sufficient = client.post(
            "/internal/v1/ml-model-monitor/notifications",
            json={
                "notifCorrId": "corr-a",
                "modelAccuInfos": [
                    {
                        "modelId": 1,
                        "deviation": 0.1,
                        "inferenceNum": 2,
                        "modelMetric": "ACCURACY",
                        "monitorInterval": {
                            "startTime": "2026-01-01T00:00:00Z",
                            "stopTime": "2026-01-01T00:01:30Z",
                        },
                    }
                ],
                "mLEvent": "UE_COMMUNICATION",
            },
        )
        insufficient = client.post(
            "/internal/v1/ml-model-monitor/notifications",
            json={
                "notifCorrId": "corr-a",
                "modelAccuInfos": [
                    {
                        "modelId": 1,
                        "inferenceNum": 1,
                        "monitorInterval": {
                            "startTime": "2026-01-01T00:01:30Z",
                            "stopTime": "2026-01-01T00:03:00Z",
                        },
                    }
                ],
                "mLEvent": "UE_COMMUNICATION",
            },
        )
        unknown = client.post(
            "/internal/v1/ml-model-monitor/notifications",
            json={
                "notifCorrId": "unknown",
                "modelAccuInfos": [{"modelId": 1}],
            },
        )

    assert sufficient.status_code == 204
    assert insufficient.status_code == 204
    assert app.state.accuracy_policy.snapshot()["scope_count"] == 1
    assert app.state.accuracy_policy.snapshot()["insufficient_reports"] == 1
    assert unknown.status_code == 404
