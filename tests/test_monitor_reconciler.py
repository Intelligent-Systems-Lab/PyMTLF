import threading
from unittest.mock import Mock

import httpx

from py_mtlf.core.monitor_reconciler import MonitorSubscriptionReconciler
from py_mtlf.core.monitor_store import (
    MonitorRegistrationStore,
    MonitorSubscriptionProjection,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)
from py_mtlf.wire.private import SelectedTarget


def resolver() -> Mock:
    return Mock(
        resolve=Mock(
            return_value=SelectedTarget(
                nfInstanceId="11111111-1111-4111-8111-111111111111",
                nfServiceInstanceId="monitor-a",
                serviceName="nnwdaf-mlmodelmonitor",
                apiRoot="http://nwdaf-a.example",
                selectionSource="NRF",
            )
        )
    )


def monitor_subscription() -> MLModelMonitorSubscription:
    return MLModelMonitorSubscription(
        modelIds=[1],
        notificationUri="http://127.0.0.1:9092/internal/v1/ml-model-monitor/notifications",
        notifCorrId="orphan-correlation",
        mLEvent="UE_COMMUNICATION",
    )


def test_restore_marks_unowned_subscription_for_retryable_cleanup(settings):
    state_lock = threading.RLock()
    projection = SyncProjection(state_lock)
    registrations = MonitorRegistrationStore(state_lock)
    subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    resource = MonitorSubscriptionProjection(
        subscription_id="orphan-subscription",
        owner_registration_id="deleted-registration",
        representation=monitor_subscription(),
    )
    subscriptions.commit_from_sync({resource.subscription_id: resource})
    reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        projection,
        registrations,
        subscriptions,
        resolver(),
        state_lock,
    )

    prepared = reconciler.prepare_restore((), (resource,))
    reconciler.commit_restore(prepared)

    assert reconciler.snapshot() == {}
    assert reconciler.orphans() == frozenset({"orphan-subscription"})
    assert reconciler._next_action() == ("delete_orphan", "orphan-subscription")

    reconciler._request = Mock(return_value=httpx.Response(status_code=204))
    reconciler._delete_orphan("orphan-subscription")

    assert reconciler.orphans() == frozenset()
    assert subscriptions.snapshot() == ()


def test_restore_does_not_reassign_old_subscription_to_new_same_scope_owner(settings):
    state_lock = threading.RLock()
    projection = SyncProjection(state_lock)
    registrations = MonitorRegistrationStore(state_lock)
    subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    registration = registrations.create(
        MLModelMonitorRegistration(
            consumerId="11111111-1111-4111-8111-111111111111",
            modelId=1,
            mLEvent="UE_COMMUNICATION",
        )
    )
    resource = MonitorSubscriptionProjection(
        subscription_id="old-subscription",
        owner_registration_id="deleted-registration",
        representation=monitor_subscription(),
    )
    reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        projection,
        registrations,
        subscriptions,
        resolver(),
        state_lock,
    )

    prepared = reconciler.prepare_restore((registration,), (resource,))
    reconciler.commit_restore(prepared)

    assert reconciler.snapshot() == {}
    assert reconciler.orphans() == frozenset({"old-subscription"})
    assert reconciler._next_action() == ("delete_orphan", "old-subscription")


def test_create_publishes_owner_identity_and_records_owned_projection(settings):
    state_lock = threading.RLock()
    projection = SyncProjection(state_lock)
    registrations = MonitorRegistrationStore(state_lock)
    subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    registration = registrations.create(
        MLModelMonitorRegistration(
            consumerId="11111111-1111-4111-8111-111111111111",
            modelId=1,
            mLEvent="UE_COMMUNICATION",
        )
    )
    reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        projection,
        registrations,
        subscriptions,
        resolver(),
        state_lock,
    )
    accepted = reconciler._subscription_for(registration)
    reconciler._request = Mock(
        return_value=httpx.Response(
            status_code=201,
            headers={"Location": "http://go.internal/subscriptions/monitor-a"},
            json=accepted.model_dump(by_alias=True, exclude_none=True, mode="json"),
        )
    )

    reconciler._create(registration.registration_id)

    request = reconciler._request.call_args
    assert request.kwargs["headers"] == {
        "X-NWDAF-Monitor-Registration-Id": registration.registration_id,
        "X-NWDAF-Target-Nf-Instance-Id": "11111111-1111-4111-8111-111111111111",
        "X-NWDAF-Target-Nf-Service-Instance-Id": "monitor-a",
        "X-NWDAF-Target-Api-Root": "http://nwdaf-a.example",
        "X-NWDAF-Target-Selection-Source": "NRF",
    }
    restored = subscriptions.snapshot()
    assert len(restored) == 1
    assert restored[0].owner_registration_id == registration.registration_id
    assert reconciler.owns(registration.registration_id, "monitor-a")


def test_newer_model_subscription_retires_old_same_scope_before_deregistration(
    settings,
):
    state_lock = threading.RLock()
    projection = SyncProjection(state_lock)
    registrations = MonitorRegistrationStore(state_lock)
    subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    old = registrations.create(
        MLModelMonitorRegistration(
            consumerId="11111111-1111-4111-8111-111111111111",
            modelId=100,
            mLEvent="UE_COMMUNICATION",
            mLEventFilter={"areaOfInterest": {"tais": [{"plmnId": {"mcc": "001"}}]}},
        )
    )
    reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        projection,
        registrations,
        subscriptions,
        resolver(),
        state_lock,
    )

    def request(method: str, path: str, **_kwargs) -> httpx.Response:
        if method == "DELETE":
            return httpx.Response(status_code=204)
        registration = old if not reconciler.owns(old.registration_id, "monitor-old") else new
        accepted = reconciler._subscription_for(registration)
        suffix = "old" if registration is old else "new"
        return httpx.Response(
            status_code=201,
            headers={"Location": f"http://go.internal/subscriptions/monitor-{suffix}"},
            json=accepted.model_dump(by_alias=True, exclude_none=True, mode="json"),
        )

    reconciler._request = Mock(side_effect=request)
    reconciler._create(old.registration_id)
    new = registrations.create(old.representation.model_copy(update={"model_id": 200}))
    reconciler._create(new.registration_id)

    assert reconciler._next_action() == ("delete", old.registration_id)
    reconciler._delete(old.registration_id)
    assert reconciler._next_action() is None
    assert reconciler.owns(new.registration_id, "monitor-new")
