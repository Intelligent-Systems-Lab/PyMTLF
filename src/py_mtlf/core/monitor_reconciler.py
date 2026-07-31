import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from py_mtlf.config import ModelMonitorSettings
from py_mtlf.core.monitor_store import (
    MonitorRegistrationResource,
    MonitorRegistrationStore,
    MonitorSubscriptionProjection,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.core.nwdaf_discovery import NwdafMonitorResolver
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
    MonitorReportingRequirement,
)
from py_mtlf.wire.private import SelectedTarget, selected_target_headers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedMonitorRestore:
    subscription_ids: dict[str, str]
    orphan_subscription_ids: frozenset[str]
    correlation_ids: dict[str, str] = field(default_factory=dict)
    selected_targets: dict[str, SelectedTarget] = field(default_factory=dict)
    retired_registration_ids: frozenset[str] = frozenset()


class MonitorSubscriptionReconciler:
    def __init__(
        self,
        settings: ModelMonitorSettings,
        projection: SyncProjection,
        registrations: MonitorRegistrationStore,
        subscriptions: MonitorSubscriptionProjectionStore,
        resolver: NwdafMonitorResolver,
        state_lock=None,
        on_subscription_created: Callable[[MLModelMonitorRegistration], None] | None = None,
    ) -> None:
        self._settings = settings
        self._projection = projection
        self._registrations = registrations
        self._subscriptions = subscriptions
        self._resolver = resolver
        self._on_subscription_created = on_subscription_created or (lambda _registration: None)
        self._condition = threading.Condition(state_lock or threading.RLock())
        self._subscription_ids: dict[str, str] = {}
        self._orphan_subscription_ids: set[str] = set()
        self._correlation_ids: dict[str, str] = {}
        self._selected_targets: dict[str, SelectedTarget] = {}
        self._retired_registration_ids: set[str] = set()
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

    def restore(
        self,
        subscriptions: tuple[MonitorSubscriptionProjection, ...],
    ) -> None:
        prepared = self.prepare_restore(
            self._registrations.snapshot(),
            subscriptions,
        )
        self.commit_restore(prepared)
        self.finalize_restore()

    def prepare_restore(
        self,
        registrations: tuple[MonitorRegistrationResource, ...],
        subscriptions: tuple[MonitorSubscriptionProjection, ...],
    ) -> PreparedMonitorRestore:
        available = list(subscriptions)
        restored: dict[str, str] = {}
        correlation_ids: dict[str, str] = {}
        selected_targets: dict[str, SelectedTarget] = {}
        for registration in registrations:
            for index, candidate in enumerate(available):
                if (
                    candidate.owner_registration_id == registration.registration_id
                    and self._same_scope(
                        self._subscription_for(registration),
                        candidate.representation,
                    )
                ):
                    restored[registration.registration_id] = candidate.subscription_id
                    correlation_ids[registration.registration_id] = (
                        candidate.representation.notification_id
                    )
                    if candidate.selected_target is not None:
                        selected_targets[registration.registration_id] = candidate.selected_target
                    available.pop(index)
                    break
        newest_by_scope: dict[str, MonitorRegistrationResource] = {}
        retired: set[str] = set()
        for registration in registrations:
            scope_key = self._registration_scope_key(registration.representation)
            newest = newest_by_scope.get(scope_key)
            if (
                newest is None
                or registration.representation.model_id > newest.representation.model_id
            ):
                if newest is not None:
                    retired.add(newest.registration_id)
                newest_by_scope[scope_key] = registration
            else:
                retired.add(registration.registration_id)
        return PreparedMonitorRestore(
            subscription_ids=restored,
            orphan_subscription_ids=frozenset(candidate.subscription_id for candidate in available),
            correlation_ids=correlation_ids,
            selected_targets=selected_targets,
            retired_registration_ids=frozenset(retired),
        )

    def commit_restore(self, prepared: PreparedMonitorRestore) -> None:
        with self._condition:
            self._subscription_ids = dict(prepared.subscription_ids)
            self._orphan_subscription_ids = set(prepared.orphan_subscription_ids)
            self._correlation_ids = dict(prepared.correlation_ids)
            self._selected_targets = dict(prepared.selected_targets)
            self._retired_registration_ids = set(prepared.retired_registration_ids)

    def finalize_restore(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, str]:
        with self._condition:
            return dict(self._subscription_ids)

    def orphans(self) -> frozenset[str]:
        with self._condition:
            return frozenset(self._orphan_subscription_ids)

    def owns(self, registration_id: str, subscription_id: str) -> bool:
        with self._condition:
            return (
                self._subscription_ids.get(registration_id) == subscription_id
                and subscription_id not in self._orphan_subscription_ids
            )

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
                    timeout = None if action is None else max(0.0, next_attempt - now)
                    self._condition.wait(timeout=timeout)
                if self._closing:
                    return
            try:
                kind, registration_id = action
                if kind == "create":
                    self._create(registration_id)
                elif kind == "delete":
                    self._delete(registration_id)
                else:
                    self._delete_orphan(registration_id)
                delay = self._settings.retry_interval_seconds
                next_attempt = 0.0
            except Exception:
                logger.exception("Monitor subscription reconciliation failed: action=%s", action)
                next_attempt = monotonic() + delay
                delay = min(delay * 2, self._settings.retry_max_interval_seconds)

    def _next_action(self) -> tuple[str, str] | None:
        if self._orphan_subscription_ids:
            return "delete_orphan", min(self._orphan_subscription_ids)
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
        logger.info(
            "ML Model Monitor subscription removed subscription_id=%s registration_id=%s",
            subscription_id,
            registration_id,
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

    def _delete_orphan(self, subscription_id: str) -> None:
        self._delete_remote(subscription_id)
        self._subscriptions.delete(subscription_id)
        with self._condition:
            self._orphan_subscription_ids.discard(subscription_id)
        logger.info(
            "Orphan ML Model Monitor subscription removed subscription_id=%s",
            subscription_id,
        )

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
        projection = self._projection.snapshot()
        if projection is None:
            raise RuntimeError("containing NWDAF is not synchronized")
        base_uri = projection.containing_nwdaf.internal_callback_base_uri.rstrip("/")
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
