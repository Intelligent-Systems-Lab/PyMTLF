from urllib.parse import urlsplit
from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import ArtifactRepository


def _seeded_app(settings, bundle_path):
    repository = ArtifactRepository(
        settings.storage.artifact_root,
        settings.artifact,
    )
    repository.open()
    artifact = repository.publish(bundle_path)
    seeded = settings.model_copy(
        update={
            "model_provision": ModelProvisionSettings(
                provider_namespace="local",
                seed_models=(
                    SeedModelSettings(
                        family_id="ue-communication-default",
                        model_id=1,
                        artifact_key=artifact.key,
                        event="UE_COMMUNICATION",
                    ),
                ),
            )
        }
    )
    return create_app(seeded, artifact_repository=repository), artifact


def _subscription(*, immediate: bool) -> dict[str, object]:
    return {
        "mLEventSubscs": [
            {
                "mLEvent": "UE_COMMUNICATION",
                "mLEventFilter": {},
                "futureEventField": {"release": 18},
            }
        ],
        "notifUri": "http://go.internal/provision-callback",
        "notifCorreId": "corr-1",
        "eventReq": {"immRep": immediate},
        "futureTopLevel": True,
    }


def test_create_returns_backend_owned_location_and_immediate_seed(settings, bundle_path):
    app, artifact = _seeded_app(settings, bundle_path)
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=_subscription(immediate=True),
        )

    assert response.status_code == 201
    resource_id = urlsplit(response.headers["location"]).path.rsplit("/", 1)[-1]
    UUID(resource_id, version=4)
    body = response.json()
    assert body["futureTopLevel"] is True
    assert body["mLEventSubscs"][0]["futureEventField"] == {"release": 18}
    assert body["mLEventNotifs"] == [
        {
            "event": "UE_COMMUNICATION",
            "notifCorreId": "corr-1",
            "mLFileAddr": {"mLModelUrl": artifact.url},
            "modelUniqueId": 1,
            "mLEventFilter": {},
        }
    ]


def test_non_immediate_create_enqueues_notification_and_crud_is_serial(settings, bundle_path):
    app, _ = _seeded_app(settings, bundle_path)
    enqueued = []
    app.state.provision_notifications.enqueue = enqueued.append
    with TestClient(app) as client:
        created = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=_subscription(immediate=False),
        )
        resource_id = created.headers["location"].rsplit("/", 1)[-1]
        replacement = _subscription(immediate=False)
        replacement["notifCorreId"] = "corr-2"
        replaced = client.put(
            f"/internal/v1/ml-model-provision/subscriptions/{resource_id}",
            json=replacement,
        )
        deleted = client.delete(
            f"/internal/v1/ml-model-provision/subscriptions/{resource_id}"
        )
        missing = client.delete(
            f"/internal/v1/ml-model-provision/subscriptions/{resource_id}"
        )

    assert created.status_code == 201
    assert "mLEventNotifs" not in created.json()
    assert replaced.status_code == 200
    assert replaced.json()["notifCorreId"] == "corr-2"
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/problem+json")
    assert len(enqueued) == 2
    assert enqueued[0].revision == 1
    assert enqueued[1].revision == 2


def test_no_matching_seed_never_returns_fake_model_url(settings):
    app = create_app(settings)
    body = _subscription(immediate=True)
    body["mLEventSubscs"][0]["mLEvent"] = "NF_LOAD"
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=body,
        )

    assert response.status_code == 201
    assert "mLEventNotifs" not in response.json()


def test_consumer_supplied_model_notification_is_not_stored_or_echoed(settings):
    app = create_app(settings)
    body = _subscription(immediate=True)
    body["mLEventSubscs"][0]["mLEvent"] = "NF_LOAD"
    body["mLEventNotifs"] = [
        {
            "event": "NF_LOAD",
            "notifCorreId": "corr-1",
            "mLFileAddr": {
                "mLModelUrl": "http://consumer.example/untrusted-model"
            },
            "modelUniqueId": 999,
        }
    ]

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=body,
        )

    assert response.status_code == 201
    assert "mLEventNotifs" not in response.json()
    stored = app.state.provision_store.snapshot()[0]
    assert stored.representation.ml_event_notifications is None


def test_generic_seed_is_reported_once_for_multiple_covered_demands(
    settings,
    bundle_path,
):
    app, _ = _seeded_app(settings, bundle_path)
    body = _subscription(immediate=True)
    body["mLEventSubscs"] = [
        {
            "mLEvent": "UE_COMMUNICATION",
            "mLEventFilter": {},
            "tgtUe": {"supis": ["imsi-1"]},
        },
        {
            "mLEvent": "UE_COMMUNICATION",
            "mLEventFilter": {},
            "tgtUe": {"supis": ["imsi-2"]},
        },
    ]
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=body,
        )

    assert response.status_code == 201
    assert len(response.json()["mLEventNotifs"]) == 1


def test_standard_route_validation_is_problem_details_400(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json={
                "mLEventSubscs": [{"mLEvent": "UE_COMMUNICATION"}],
                "notifUri": "not-a-uri",
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["cause"] == "INVALID_MSG_FORMAT"


def test_provision_notification_never_follows_go_redirect(
    settings,
    bundle_path,
    monkeypatch,
):
    app, _ = _seeded_app(settings, bundle_path)
    enqueued = []
    app.state.provision_notifications.enqueue = enqueued.append
    requested = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, uri, *, json):
            requested.append((uri, json))
            return httpx.Response(
                status_code=307,
                headers={"Location": "http://consumer.example/private"},
            )

    monkeypatch.setattr(
        "py_mtlf.core.notification_delivery.httpx.Client",
        FakeClient,
    )
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/ml-model-provision/subscriptions",
            json=_subscription(immediate=False),
        )
        assert response.status_code == 201
        app.state.provision_notifications._deliver(enqueued[0])

    assert len(requested) == 1
    assert requested[0][0] == "http://go.internal/provision-callback"
