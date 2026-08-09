import threading
from unittest.mock import Mock

import httpx
from nwdaf_context import context_client

from py_mtlf.core.monitor_reconciler import MonitorSubscriptionReconciler
from py_mtlf.core.monitor_store import (
    MonitorRegistrationStore,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
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


def test_create_publishes_owner_identity_and_records_owned_projection(settings):
    state_lock = threading.RLock()
    projection = context_client()
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
    projection = context_client()
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


def test_watchdog_expires_monitor_without_replaying_failed_cleanup(settings):
    state_lock = threading.RLock()
    registrations = MonitorRegistrationStore(state_lock)
    subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    registration = registrations.create(
        MLModelMonitorRegistration(
            consumerId="11111111-1111-4111-8111-111111111111",
            modelId=1,
            mLEvent="UE_COMMUNICATION",
        )
    )
    timed_out = Mock()
    monitor_settings = settings.model_monitor.model_copy(
        update={
            "report_period_seconds": 1,
            "missed_report_threshold": 1,
            "watchdog_grace_seconds": 0,
        }
    )
    reconciler = MonitorSubscriptionReconciler(
        monitor_settings,
        context_client(),
        registrations,
        subscriptions,
        resolver(),
        state_lock,
        on_subscription_timeout=timed_out,
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
    reconciler._last_reports["monitor-a"] = 0.0

    assert reconciler._next_action() == ("expire", registration.registration_id)
    reconciler._request = Mock(return_value=httpx.Response(status_code=503))
    reconciler._expire(registration.registration_id)

    assert registrations.get(registration.registration_id) is None
    assert subscriptions.find_by_id("monitor-a") is None
    assert not reconciler.owns(registration.registration_id, "monitor-a")
    timed_out.assert_called_once_with(registration.representation)
