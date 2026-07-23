import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx

from py_mtlf.config import NotificationSettings
from py_mtlf.core.provision_store import ProvisionResource, ProvisionResourceStore
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.wire.ml_model import MLModelProvisionNotification

logger = logging.getLogger(__name__)


class ProvisionNotificationDispatcher:
    def __init__(
        self,
        settings: NotificationSettings,
        store: ProvisionResourceStore,
    ) -> None:
        self._settings = settings
        self._store = store
        self._executor: ThreadPoolExecutor | None = None
        self._closed = threading.Event()
        self._closed.set()

    def open(self) -> None:
        if self._executor is not None:
            return
        self._closed.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="model-provision-notify",
        )

    def enqueue(self, resource: ProvisionResource) -> None:
        executor = self._executor
        if self._closed.is_set() or executor is None or not any(resource.seeds):
            return
        executor.submit(self._deliver, resource)

    def shutdown(self) -> None:
        self._closed.set()
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _deliver(self, resource: ProvisionResource) -> None:
        notifications = SeedCatalog.notifications(
            resource.seeds,
            resource.representation.notification_correlation_id,
        )
        if not notifications:
            return
        payload = [
            MLModelProvisionNotification(
                eventNotifs=notifications,
                subscriptionId=resource.subscription_id,
            ).model_dump(by_alias=True, exclude_none=True, mode="json")
        ]
        callback_uri = str(resource.representation.notification_uri)
        backoff = self._settings.initial_backoff_seconds
        for attempt in range(1, self._settings.max_attempts + 1):
            if self._closed.is_set() or not self._store.is_current(
                resource.subscription_id,
                resource.revision,
            ):
                return
            retryable = False
            try:
                with httpx.Client(
                    timeout=self._settings.request_timeout_seconds,
                    follow_redirects=False,
                ) as client:
                    response = client.post(callback_uri, json=payload)
                if response.status_code == 204:
                    return
                retryable = response.status_code >= 500
                if not retryable:
                    logger.warning(
                        "Model Provision notification rejected subscription_id=%s status=%s",
                        resource.subscription_id,
                        response.status_code,
                    )
                    return
            except httpx.HTTPError as error:
                retryable = True
                logger.warning(
                    "Model Provision notification transport failure "
                    "subscription_id=%s attempt=%s error=%s",
                    resource.subscription_id,
                    attempt,
                    type(error).__name__,
                )
            if not retryable or attempt == self._settings.max_attempts:
                break
            if self._closed.wait(backoff):
                return
            backoff = min(max(backoff * 2, 0.001), self._settings.max_backoff_seconds)
        logger.warning(
            "Model Provision notification delivery exhausted retries subscription_id=%s",
            resource.subscription_id,
        )
