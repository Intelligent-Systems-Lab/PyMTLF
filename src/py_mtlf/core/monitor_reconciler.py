import json
import logging
import threading
from collections.abc import Callable
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from py_mtlf.config import ModelMonitorSettings
from py_mtlf.core.monitor_store import (
    MonitorRegistrationResource,
    MonitorRegistrationStore,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.core.nwdaf_discovery import NwdafMonitorResolver
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
    MonitorReportingRequirement,
)
from py_mtlf.wire.private import SelectedTarget, selected_target_headers

logger = logging.getLogger(__name__)


class MonitorSubscriptionReconciler:
    def __init__(
        self,
        settings: ModelMonitorSettings,
        nwdaf_context: NwdafContextClient,
        registrations: MonitorRegistrationStore,
        subscriptions: MonitorSubscriptionProjectionStore,
        resolver: NwdafMonitorResolver,
        state_lock=None,
        on_subscription_created: Callable[[MLModelMonitorRegistration], None] | None = None,
        on_subscription_timeout: Callable[[MLModelMonitorRegistration], None] | None = None,
    ) -> None:
        self._settings = settings
        self._nwdaf_context = nwdaf_context
        self._registrations = registrations
        self._subscriptions = subscriptions
        self._resolver = resolver
        self._on_subscription_created = on_subscription_created or (lambda _registration: None)
        self._on_subscription_timeout = on_subscription_timeout or (lambda _registration: None)
        self._condition = threading.Condition(state_lock or threading.RLock())
        self._subscription_ids: dict[str, str] = {}
        self._correlation_ids: dict[str, str] = {}
        self._selected_targets: dict[str, SelectedTarget] = {}
        self._retired_registration_ids: set[str] = set()
        self._last_reports: dict[str, float] = {}
        self._closing = False
        self._worker: threading.Thread | None = None
        self._session = httpx.Client()

    def open(self) -> None:
        with self._condition:
            if self._worker is not None:
                return
            self._closing = False
            self._worker = threading.Thread(
                target=self._run,
                name="monitor-subscription-reconciler",
                daemon=True,
            )
            self._worker.start()

    def shutdown(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=self._settings.request_timeout_seconds + 1)
        self._session.close()
        with self._condition:
            self._worker = None

    def refresh(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, str]:
        with self._condition:
            return dict(self._subscription_ids)

    def owns(self, registration_id: str, subscription_id: str) -> bool:
        with self._condition:
            return (
                self._subscription_ids.get(registration_id) == subscription_id
            )

    def record_report(self, subscription_id: str) -> None:
        with self._condition:
            if subscription_id in self._last_reports:
                self._last_reports[subscription_id] = monotonic()
                self._condition.notify_all()

    def _run(self) -> None:
        delay = self._settings.retry_interval_seconds
        next_attempt = 0.0
        while True:
            with self._condition:
                while not self._closing:
                    action = self._next_action()
                    now = monotonic()
                    if action is not None and now >= next_attempt:
                        break
                    if action is None:
                        timeout = self._watchdog_wait_seconds(now)
                    else:
                        timeout = max(0.0, next_attempt - now)
                    self._condition.wait(timeout=timeout)
                if self._closing:
                    return
            try:
                kind, registration_id = action
                if kind == "create":
                    self._create(registration_id)
                elif kind == "delete":
                    self._delete(registration_id)
                elif kind == "expire":
                    self._expire(registration_id)
                delay = self._settings.retry_interval_seconds
                next_attempt = 0.0
            except Exception:
                logger.exception("Monitor subscription reconciliation failed: action=%s", action)
                next_attempt = monotonic() + delay
                delay = min(delay * 2, self._settings.retry_max_interval_seconds)

    def _next_action(self) -> tuple[str, str] | None:
        now = monotonic()
        for registration_id, subscription_id in sorted(self._subscription_ids.items()):
            last_report = self._last_reports.get(subscription_id)
            projection = self._subscriptions.find_by_id(subscription_id)
            if (
                last_report is not None
                and projection is not None
                and now >= last_report + self._watchdog_timeout(projection.representation)
            ):
                return "expire", registration_id
        resources = self._registrations.snapshot()
        available = {resource.registration_id for resource in resources}
        self._retired_registration_ids.intersection_update(available)
        for registration_id in sorted(self._retired_registration_ids):
            if registration_id in self._subscription_ids:
                return "delete", registration_id
        desired = available - self._retired_registration_ids
        for registration_id in sorted(desired):
            if registration_id not in self._subscription_ids:
                return "create", registration_id
        for registration_id in sorted(self._subscription_ids):
            if registration_id not in desired:
                return "delete", registration_id
        return None

    def _create(self, registration_id: str) -> None:
        resource = self._registrations.get(registration_id)
        if resource is None:
            return
        registration = resource.representation
        if not registration.consumer_id:
            raise RuntimeError("consumerSetId monitor discovery is not supported")
        target = self._resolver.resolve(registration.consumer_id)
        if target is None:
            raise RuntimeError("monitor consumer NWDAF is not currently discoverable")
        representation = self._subscription_for(resource)
        headers = {
            "X-NWDAF-Monitor-Registration-Id": registration_id,
            **selected_target_headers(target),
        }
        response = self._request(
            "POST",
            "/internal/v1/ml-model-monitor/subscriptions",
            headers=headers,
            json=representation.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            ),
        )
        if response.status_code != 201:
            if response.status_code in {404, 503}:
                self._resolver.invalidate(registration.consumer_id)
            raise RuntimeError(
                f"monitor subscription create returned status {response.status_code}"
            )
        accepted = MLModelMonitorSubscription.model_validate(response.json())
        if not self._same_scope(representation, accepted):
            raise RuntimeError("monitor subscription response changed the requested scope")
        subscription_id = (
            urlsplit(response.headers.get("Location", "")).path.rstrip("/").rsplit("/", 1)[-1]
        )
        if not subscription_id:
            raise RuntimeError("monitor subscription create response has no resource identity")
        with self._condition:
            if self._registrations.get(registration_id) is not None:
                self._subscription_ids[registration_id] = subscription_id
                self._selected_targets[registration_id] = target
                self._subscriptions.upsert(
                    subscription_id,
                    registration_id,
                    accepted,
                    target,
                )
                self._last_reports[subscription_id] = monotonic()
                logger.info(
                    "ML Model Monitor subscription active subscription_id=%s "
                    "registration_id=%s correlation_id=%s",
                    subscription_id,
                    registration_id,
                    accepted.notification_id,
                )
                self._on_subscription_created(registration)
                self._retire_older_scope_registrations(resource)
                self._condition.notify_all()
                return
        self._delete_remote(subscription_id)

    def _delete(self, registration_id: str) -> None:
        with self._condition:
            subscription_id = self._subscription_ids.get(registration_id, "")
        if not subscription_id:
            return
        self._delete_remote(subscription_id)
        self._subscriptions.delete(subscription_id)
        with self._condition:
            self._subscription_ids.pop(registration_id, None)
            self._correlation_ids.pop(registration_id, None)
            self._selected_targets.pop(registration_id, None)
            self._last_reports.pop(subscription_id, None)
        logger.info(
            "ML Model Monitor subscription removed subscription_id=%s registration_id=%s",
            subscription_id,
            registration_id,
        )

    def _expire(self, registration_id: str) -> None:
        with self._condition:
            subscription_id = self._subscription_ids.get(registration_id, "")
        if not subscription_id:
            return
        try:
            self._delete_remote(subscription_id)
        except Exception:
            logger.warning(
                "Monitor subscription watchdog cleanup failed; state will not be replayed "
                "subscription_id=%s registration_id=%s",
                subscription_id,
                registration_id,
                exc_info=True,
            )
        registration = self._registrations.get(registration_id)
        self._subscriptions.delete(subscription_id)
        self._registrations.delete(registration_id)
        with self._condition:
            self._subscription_ids.pop(registration_id, None)
            self._correlation_ids.pop(registration_id, None)
            self._selected_targets.pop(registration_id, None)
            self._last_reports.pop(subscription_id, None)
        if registration is not None:
            self._on_subscription_timeout(registration.representation)
        logger.warning(
            "Monitor subscription expired after missing periodic reports "
            "subscription_id=%s registration_id=%s",
            subscription_id,
            registration_id,
        )

    def _watchdog_wait_seconds(self, now: float) -> float | None:
        deadlines: list[float] = []
        for subscription_id, last_report in self._last_reports.items():
            projection = self._subscriptions.find_by_id(subscription_id)
            if projection is not None:
                deadlines.append(
                    last_report + self._watchdog_timeout(projection.representation)
                )
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - now)

    def _watchdog_timeout(self, subscription: MLModelMonitorSubscription) -> float:
        report_period = self._settings.report_period_seconds
        if (
            subscription.event_report_request is not None
            and subscription.event_report_request.repetition_period is not None
            and subscription.event_report_request.repetition_period > 0
        ):
            report_period = subscription.event_report_request.repetition_period
        return (
            report_period * self._settings.missed_report_threshold
            + self._settings.watchdog_grace_seconds
        )

    def _retire_older_scope_registrations(
        self,
        current: MonitorRegistrationResource,
    ) -> None:
        scope_key = self._registration_scope_key(current.representation)
        current_model_id = current.representation.model_id
        for candidate in self._registrations.snapshot():
            if (
                candidate.registration_id != current.registration_id
                and candidate.representation.model_id < current_model_id
                and self._registration_scope_key(candidate.representation) == scope_key
            ):
                self._retired_registration_ids.add(candidate.registration_id)

    def _delete_remote(self, subscription_id: str) -> None:
        response = self._request(
            "DELETE",
            f"/internal/v1/ml-model-monitor/subscriptions/{subscription_id}",
        )
        if response.status_code not in {204, 404}:
            raise RuntimeError(
                f"monitor subscription delete returned status {response.status_code}"
            )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        base_uri = self._nwdaf_context.get().internal_api_root
        return self._session.request(
            method,
            base_uri + path,
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=False,
            **kwargs,
        )

    def _subscription_for(
        self,
        resource: MonitorRegistrationResource,
    ) -> MLModelMonitorSubscription:
        registration = resource.representation
        with self._condition:
            correlation_id = self._correlation_ids.setdefault(
                resource.registration_id,
                str(uuid4()),
            )
        return MLModelMonitorSubscription(
            modelIds=[registration.model_id],
            notificationUri=self._settings.callback_uri,
            notifCorrId=correlation_id,
            modelMetric="ACCURACY",
            eventReportReq=MonitorReportingRequirement(
                notifMethod="PERIODIC",
                repPeriod=self._settings.report_period_seconds,
            ),
            mLEvent=registration.ml_event,
            mLEventFilter=registration.ml_event_filter,
            tgtUe=registration.target_ue,
        )

    @staticmethod
    def _same_scope(
        left: MLModelMonitorSubscription,
        right: MLModelMonitorSubscription,
    ) -> bool:
        fields = ("model_ids", "ml_event", "ml_event_filter", "target_ue")
        return all(
            json.dumps(
                getattr(left, field),
                sort_keys=True,
                separators=(",", ":"),
            )
            == json.dumps(
                getattr(right, field),
                sort_keys=True,
                separators=(",", ":"),
            )
            for field in fields
        )

    @staticmethod
    def _registration_scope_key(
        registration: MLModelMonitorRegistration,
    ) -> str:
        return json.dumps(
            {
                "consumerId": registration.consumer_id,
                "consumerSetId": registration.consumer_set_id,
                "mLEvent": registration.ml_event,
                "mLEventFilter": registration.ml_event_filter or {},
                "tgtUe": registration.target_ue,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
