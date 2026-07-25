import threading
import time

import httpx
from conftest import build_bundle

from py_mtlf.config import (
    ModelProvisionSettings,
    NotificationSettings,
    SeedModelSettings,
)
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.notification_delivery import ProvisionNotificationDispatcher
from py_mtlf.core.provision_store import ProvisionResourceStore
from py_mtlf.core.seed_catalog import ModelCatalog
from py_mtlf.wire.ml_model import MLModelProvisionSubscription


def wait_until(predicate, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def seeded_state(settings, bundle_path):
    repository = ArtifactRepository(
        settings.storage.artifact_root,
        settings.artifact,
    )
    repository.open()
    seed = repository.publish(bundle_path)
    catalog = ModelCatalog(
        ModelProvisionSettings(
            provider_namespace="local",
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=1,
                    artifact_key=seed.key,
                    event="UE_COMMUNICATION",
                ),
            ),
        ),
        repository,
    )
    catalog.open()
    store = ProvisionResourceStore(catalog)
    resource = store.create(
        MLModelProvisionSubscription.model_validate(
            {
                "mLEventSubscs": [
                    {
                        "mLEvent": "UE_COMMUNICATION",
                        "mLEventFilter": {},
                    }
                ],
                "notifUri": "http://go.internal/model-update",
                "notifCorreId": "corr-1",
            }
        )
    )
    return repository, seed, catalog, store, resource


def test_retry_coalesces_to_latest_promoted_artifact(
    settings,
    bundle_path,
    tmp_path,
    monkeypatch,
):
    repository, seed, catalog, store, resource = seeded_state(
        settings,
        bundle_path,
    )
    candidate_path = tmp_path / "candidate.tar.gz"
    build_bundle(
        candidate_path,
        mutate_manifest={
            "model_generation": 2,
            "model_identity": {"provider_id": "local", "model_unique_id": 2},
        },
    )
    candidate = repository.publish(candidate_path)
    first_started = threading.Event()
    allow_first = threading.Event()
    requests: list[list[dict]] = []
    delivered = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _uri, *, json):
            requests.append(json)
            if len(requests) == 1:
                first_started.set()
                assert allow_first.wait(timeout=2)
                return httpx.Response(503)
            return httpx.Response(204)

    monkeypatch.setattr(
        "py_mtlf.core.notification_delivery.httpx.Client",
        FakeClient,
    )
    dispatcher = ProvisionNotificationDispatcher(
        NotificationSettings(
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
        store,
        catalog,
        delivered.append,
    )
    dispatcher.open()
    dispatcher.enqueue(resource)
    assert first_started.wait(timeout=2)

    family_key = ("local", "ue-communication-default")
    version_key = catalog.reserve_next_version(family_key)
    promoted = catalog.promote(
        family_key,
        expected_generation=1,
        expected_artifact_key=seed.key,
        version_key=version_key,
        artifact=candidate,
    )
    assert dispatcher.reconcile_family(family_key) == 1
    allow_first.set()
    wait_until(lambda: dispatcher.delivered_version(resource.subscription_id) is not None)
    dispatcher.shutdown()

    assert promoted.generation == 2
    assert len(requests) == 2
    latest = requests[-1][0]["eventNotifs"][0]
    assert latest["modelUniqueId"] == 2
    assert latest["mLFileAddr"]["mLModelUrl"] == candidate.url
    assert delivered == [(family_key,)]


def test_deleted_resource_cancels_retry(
    settings,
    bundle_path,
    monkeypatch,
):
    _repository, _seed, catalog, store, resource = seeded_state(
        settings,
        bundle_path,
    )
    attempted = threading.Event()
    calls = 0

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _uri, *, json):
            nonlocal calls
            assert json
            calls += 1
            attempted.set()
            return httpx.Response(503)

    monkeypatch.setattr(
        "py_mtlf.core.notification_delivery.httpx.Client",
        FakeClient,
    )
    dispatcher = ProvisionNotificationDispatcher(
        NotificationSettings(
            initial_backoff_seconds=1,
            max_backoff_seconds=1,
        ),
        store,
        catalog,
    )
    dispatcher.open()
    dispatcher.enqueue(resource)
    assert attempted.wait(timeout=2)

    assert store.delete(resource.subscription_id)
    dispatcher.cancel(resource.subscription_id)
    dispatcher.shutdown()

    assert calls == 1
    assert dispatcher.delivered_version(resource.subscription_id) is None
