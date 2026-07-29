import threading
from dataclasses import dataclass
from uuid import uuid4

from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorRegistration,
    MLModelMonitorRegistrationSnapshot,
    MLModelMonitorSubscription,
    MLModelMonitorSubscriptionSnapshot,
)
from py_mtlf.wire.private import SelectedTarget


@dataclass(frozen=True)
class MonitorRegistrationResource:
    registration_id: str
    representation: MLModelMonitorRegistration


class MonitorRegistrationStore:
    def __init__(self, lock=None) -> None:
        self._lock = lock or threading.RLock()
        self._resources: dict[str, MonitorRegistrationResource] = {}

    def create(
        self,
        representation: MLModelMonitorRegistration,
    ) -> MonitorRegistrationResource:
        resource = MonitorRegistrationResource(
            registration_id=str(uuid4()),
            representation=representation.model_copy(deep=True),
        )
        with self._lock:
            self._resources[resource.registration_id] = resource
        return self._copy(resource)

    def delete(self, registration_id: str) -> bool:
        with self._lock:
            return self._resources.pop(registration_id, None) is not None

    def get(self, registration_id: str) -> MonitorRegistrationResource | None:
        with self._lock:
            resource = self._resources.get(registration_id)
            return self._copy(resource) if resource is not None else None

    def replace_from_sync(
        self,
        snapshots: list[MLModelMonitorRegistrationSnapshot],
    ) -> tuple[MonitorRegistrationResource, ...]:
        restored = self.prepare_from_sync(snapshots)
        return self.commit_from_sync(restored)

    def prepare_from_sync(
        self,
        snapshots: list[MLModelMonitorRegistrationSnapshot],
    ) -> dict[str, MonitorRegistrationResource]:
        registration_ids = [snapshot.registration_id for snapshot in snapshots]
        if len(registration_ids) != len(set(registration_ids)):
            raise ValueError(
                "mlModelMonitorRegistrations contains duplicate registrationId"
            )
        restored = {
            snapshot.registration_id: MonitorRegistrationResource(
                registration_id=snapshot.registration_id,
                representation=snapshot.representation.model_copy(deep=True),
            )
            for snapshot in snapshots
        }
        return restored

    def commit_from_sync(
        self,
        restored: dict[str, MonitorRegistrationResource],
    ) -> tuple[MonitorRegistrationResource, ...]:
        with self._lock:
            self._resources = dict(restored)
            return tuple(self._copy(resource) for resource in restored.values())

    def snapshot(self) -> tuple[MonitorRegistrationResource, ...]:
        with self._lock:
            return tuple(self._copy(resource) for resource in self._resources.values())

    def find_for_subscription(
        self,
        subscription: MLModelMonitorSubscription,
    ) -> MonitorRegistrationResource | None:
        with self._lock:
            for resource in self._resources.values():
                registration = resource.representation
                if (
                    registration.model_id in subscription.model_ids
                    and registration.ml_event == subscription.ml_event
                    and registration.ml_event_filter == subscription.ml_event_filter
                    and registration.target_ue == subscription.target_ue
                ):
                    return self._copy(resource)
        return None

    @staticmethod
    def _copy(resource: MonitorRegistrationResource) -> MonitorRegistrationResource:
        return MonitorRegistrationResource(
            registration_id=resource.registration_id,
            representation=resource.representation.model_copy(deep=True),
        )


@dataclass(frozen=True)
class MonitorSubscriptionProjection:
    subscription_id: str
    owner_registration_id: str
    representation: MLModelMonitorSubscription
    selected_target: SelectedTarget | None = None


class MonitorSubscriptionProjectionStore:
    def __init__(self, lock=None) -> None:
        self._lock = lock or threading.RLock()
        self._resources: dict[str, MonitorSubscriptionProjection] = {}

    def replace_from_sync(
        self,
        snapshots: list[MLModelMonitorSubscriptionSnapshot],
    ) -> tuple[MonitorSubscriptionProjection, ...]:
        restored = self.prepare_from_sync(snapshots)
        return self.commit_from_sync(restored)

    def prepare_from_sync(
        self,
        snapshots: list[MLModelMonitorSubscriptionSnapshot],
    ) -> dict[str, MonitorSubscriptionProjection]:
        subscription_ids = [snapshot.subscription_id for snapshot in snapshots]
        if len(subscription_ids) != len(set(subscription_ids)):
            raise ValueError(
                "mlModelMonitorSubscriptions contains duplicate subscriptionId"
            )
        restored = {
            snapshot.subscription_id: MonitorSubscriptionProjection(
                subscription_id=snapshot.subscription_id,
                owner_registration_id=snapshot.owner_registration_id,
                representation=snapshot.representation.model_copy(deep=True),
                selected_target=snapshot.selected_target,
            )
            for snapshot in snapshots
        }
        return restored

    def commit_from_sync(
        self,
        restored: dict[str, MonitorSubscriptionProjection],
    ) -> tuple[MonitorSubscriptionProjection, ...]:
        with self._lock:
            self._resources = dict(restored)
            return tuple(self._copy(resource) for resource in restored.values())

    def upsert(
        self,
        subscription_id: str,
        owner_registration_id: str,
        representation: MLModelMonitorSubscription,
        selected_target: SelectedTarget | None = None,
    ) -> None:
        with self._lock:
            self._resources[subscription_id] = MonitorSubscriptionProjection(
                subscription_id=subscription_id,
                owner_registration_id=owner_registration_id,
                representation=representation.model_copy(deep=True),
                selected_target=selected_target,
            )

    def delete(self, subscription_id: str) -> None:
        with self._lock:
            self._resources.pop(subscription_id, None)

    def snapshot(self) -> tuple[MonitorSubscriptionProjection, ...]:
        with self._lock:
            return tuple(self._copy(resource) for resource in self._resources.values())

    def find_by_correlation(
        self,
        notification_id: str,
    ) -> MonitorSubscriptionProjection | None:
        with self._lock:
            for resource in self._resources.values():
                if resource.representation.notification_id == notification_id:
                    return self._copy(resource)
        return None

    @staticmethod
    def _copy(resource: MonitorSubscriptionProjection) -> MonitorSubscriptionProjection:
        return MonitorSubscriptionProjection(
            subscription_id=resource.subscription_id,
            owner_registration_id=resource.owner_registration_id,
            representation=resource.representation.model_copy(deep=True),
            selected_target=resource.selected_target,
        )
