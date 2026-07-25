import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from py_mtlf.config import NotificationSettings
from py_mtlf.core.provision_store import ProvisionResource, ProvisionResourceStore
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog, ModelVersionKey
from py_mtlf.wire.ml_model import MLModelProvisionNotification

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryVersion:
    resource_revision: int
    models: tuple[tuple[FamilyKey, ModelVersionKey, int, str], ...]


@dataclass(frozen=True)
class DesiredDelivery:
    resource: ProvisionResource
    version: DeliveryVersion


class ProvisionNotificationDispatcher:
    def __init__(
        self,
        settings: NotificationSettings,
        store: ProvisionResourceStore,
        catalog: ModelCatalog,
        on_delivered: Callable[[tuple[FamilyKey, ...]], None] | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._catalog = catalog
        self._on_delivered = on_delivered or (lambda _model_keys: None)
        self._condition = threading.Condition(threading.RLock())
        self._executor: ThreadPoolExecutor | None = None
        self._desired: dict[str, DesiredDelivery] = {}
        self._delivered: dict[str, DeliveryVersion] = {}
        self._active: set[str] = set()
        self._closed = True

    def open(self) -> None:
        with self._condition:
            if self._executor is not None:
                return
            self._closed = False
            self._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="model-provision-notify",
            )

    def enqueue(self, resource: ProvisionResource) -> None:
        models = []
        for key in resource.family_keys:
            current = self._catalog.current(key) if key is not None else None
            if key is not None and current is not None:
                models.append(
                    (key, current.version_key, current.generation, current.artifact.key)
                )
        if not models:
            return
        desired = DesiredDelivery(
            resource=resource,
            version=DeliveryVersion(
                resource_revision=resource.revision,
                models=tuple(sorted(set(models))),
            ),
        )
        with self._condition:
            if self._closed or self._executor is None:
                return
            self._desired[resource.subscription_id] = desired
            if resource.subscription_id not in self._active:
                self._active.add(resource.subscription_id)
                self._executor.submit(self._deliver_loop, resource.subscription_id)
            self._condition.notify_all()

    def reconcile_family(self, family_key: FamilyKey) -> int:
        resources = self._store.resources_for_family(family_key)
        for resource in resources:
            self.enqueue(resource)
        return len(resources)

    def delivered_version(self, subscription_id: str) -> DeliveryVersion | None:
        with self._condition:
            return self._delivered.get(subscription_id)

    def cancel(self, subscription_id: str) -> None:
        with self._condition:
            self._desired.pop(subscription_id, None)
            self._delivered.pop(subscription_id, None)
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._condition:
            self._active.clear()

    def _deliver(self, resource: ProvisionResource) -> int:
        """Perform one synchronous delivery attempt for focused contract tests."""
        payload = self._payload(resource)
        if not payload:
            return 0
        with httpx.Client(
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            response = client.post(
                str(resource.representation.notification_uri),
                json=payload,
            )
        return response.status_code

    def _deliver_loop(self, subscription_id: str) -> None:
        attempt = 0
        observed: DeliveryVersion | None = None
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                    desired = self._desired.get(subscription_id)
                    if desired is None:
                        return
                    if desired.version != observed:
                        observed = desired.version
                        attempt = 0
                if not self._store.is_current(
                    subscription_id,
                    desired.resource.revision,
                ):
                    with self._condition:
                        if self._desired.get(subscription_id) == desired:
                            self._desired.pop(subscription_id, None)
                    return

                payload = self._payload(desired.resource)
                if not payload:
                    with self._condition:
                        if self._desired.get(subscription_id) == desired:
                            self._desired.pop(subscription_id, None)
                    return
                retryable = False
                try:
                    with httpx.Client(
                        timeout=self._settings.request_timeout_seconds,
                        follow_redirects=False,
                    ) as client:
                        response = client.post(
                            str(desired.resource.representation.notification_uri),
                            json=payload,
                        )
                    if response.status_code == 204:
                        delivered_keys = tuple(
                            sorted(
                                {
                                    key
                                    for (
                                        key,
                                        _version,
                                        _generation,
                                        _artifact,
                                    ) in desired.version.models
                                }
                            )
                        )
                        with self._condition:
                            if self._desired.get(subscription_id) != desired:
                                continue
                            self._delivered[subscription_id] = desired.version
                            self._desired.pop(subscription_id, None)
                        self._on_delivered(delivered_keys)
                        return
                    retryable = response.status_code >= 500 or response.status_code == 429
                    if not retryable:
                        logger.warning(
                            "Model Provision notification rejected subscription_id=%s status=%s",
                            subscription_id,
                            response.status_code,
                        )
                        with self._condition:
                            if self._desired.get(subscription_id) == desired:
                                self._desired.pop(subscription_id, None)
                        return
                except httpx.HTTPError as error:
                    retryable = True
                    logger.warning(
                        "Model Provision notification transport failure "
                        "subscription_id=%s attempt=%s error=%s",
                        subscription_id,
                        attempt + 1,
                        type(error).__name__,
                    )
                if not retryable:
                    return
                attempt += 1
                backoff = min(
                    self._settings.initial_backoff_seconds * (2 ** max(attempt - 1, 0)),
                    self._settings.max_backoff_seconds,
                )
                with self._condition:
                    if self._closed:
                        return
                    self._condition.wait(timeout=max(backoff, 0.001))
        finally:
            with self._condition:
                self._active.discard(subscription_id)
                if (
                    not self._closed
                    and subscription_id in self._desired
                    and self._executor is not None
                ):
                    self._active.add(subscription_id)
                    self._executor.submit(self._deliver_loop, subscription_id)

    def _payload(self, resource: ProvisionResource) -> list[dict]:
        notifications = self._catalog.notifications(
            resource.family_keys,
            resource.representation.notification_correlation_id,
        )
        if not notifications:
            return []
        return [
            MLModelProvisionNotification(
                eventNotifs=notifications,
                subscriptionId=resource.subscription_id,
            ).model_dump(by_alias=True, exclude_none=True, mode="json")
        ]
