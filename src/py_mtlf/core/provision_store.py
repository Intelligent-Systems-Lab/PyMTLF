import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4

from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog
from py_mtlf.wire.ml_model import (
    MLModelProvisionSnapshot,
    MLModelProvisionSubscription,
)


@dataclass(frozen=True)
class ProvisionResource:
    subscription_id: str
    representation: MLModelProvisionSubscription
    family_keys: tuple[FamilyKey | None, ...]
    revision: int


class ProvisionResourceStore:
    def __init__(self, catalog: ModelCatalog, lock=None) -> None:
        self._catalog = catalog
        self._lock = lock or threading.RLock()
        self._resources: dict[str, ProvisionResource] = {}

    def create(self, representation: MLModelProvisionSubscription) -> ProvisionResource:
        resource = self._prepare(str(uuid4()), representation, revision=1)
        with self._lock:
            self._resources[resource.subscription_id] = resource
        return self._copy(resource)

    def replace(
        self,
        subscription_id: str,
        representation: MLModelProvisionSubscription,
    ) -> ProvisionResource | None:
        with self._lock:
            current = self._resources.get(subscription_id)
            if current is None:
                return None
            prepared = self._prepare(
                subscription_id,
                representation,
                revision=current.revision + 1,
            )
            self._resources[subscription_id] = prepared
            return self._copy(prepared)

    def delete(self, subscription_id: str) -> bool:
        with self._lock:
            return self._resources.pop(subscription_id, None) is not None

    def get(self, subscription_id: str) -> ProvisionResource | None:
        with self._lock:
            resource = self._resources.get(subscription_id)
            return self._copy(resource) if resource is not None else None

    def is_current(self, subscription_id: str, revision: int) -> bool:
        with self._lock:
            resource = self._resources.get(subscription_id)
            return resource is not None and resource.revision == revision

    def replace_from_sync(
        self,
        snapshots: list[MLModelProvisionSnapshot],
    ) -> tuple[ProvisionResource, ...]:
        restored = self.prepare_from_sync(snapshots)
        return self.commit_from_sync(restored)

    def prepare_from_sync(
        self,
        snapshots: list[MLModelProvisionSnapshot],
    ) -> dict[str, ProvisionResource]:
        subscription_ids = [snapshot.subscription_id for snapshot in snapshots]
        if len(subscription_ids) != len(set(subscription_ids)):
            raise ValueError("mlModelProvisionSubscriptions contains duplicate subscriptionId")
        restored: dict[str, ProvisionResource] = {}
        for snapshot in snapshots:
            restored[snapshot.subscription_id] = self._prepare(
                snapshot.subscription_id,
                snapshot.representation,
                revision=1,
            )
        return restored

    def commit_from_sync(
        self,
        restored: dict[str, ProvisionResource],
    ) -> tuple[ProvisionResource, ...]:
        with self._lock:
            reconciled: dict[str, ProvisionResource] = {}
            for subscription_id, resource in restored.items():
                current = self._resources.get(subscription_id)
                if (
                    current is not None
                    and current.representation == resource.representation
                    and current.family_keys == resource.family_keys
                ):
                    revision = current.revision
                elif current is not None:
                    revision = current.revision + 1
                else:
                    revision = 1
                reconciled[subscription_id] = ProvisionResource(
                    subscription_id=resource.subscription_id,
                    representation=resource.representation,
                    family_keys=resource.family_keys,
                    revision=revision,
                )
            self._resources = reconciled
            return tuple(self._copy(resource) for resource in reconciled.values())

    def snapshot(self) -> tuple[ProvisionResource, ...]:
        with self._lock:
            return tuple(self._copy(resource) for resource in self._resources.values())

    def resources_for_family(self, family_key: FamilyKey) -> tuple[ProvisionResource, ...]:
        with self._lock:
            return tuple(
                self._copy(resource)
                for resource in self._resources.values()
                if family_key in resource.family_keys
            )

    @contextmanager
    def hold_resources_for_family(
        self,
        family_key: FamilyKey,
    ) -> Iterator[tuple[ProvisionResource, ...]]:
        """Keep the current model demand stable across promotion and enqueue."""

        with self._lock:
            yield tuple(
                self._copy(resource)
                for resource in self._resources.values()
                if family_key in resource.family_keys
            )

    def _prepare(
        self,
        subscription_id: str,
        representation: MLModelProvisionSubscription,
        *,
        revision: int,
    ) -> ProvisionResource:
        canonical = representation.model_copy(
            update={"ml_event_notifications": None},
            deep=True,
        )
        family_keys = tuple(
            self._catalog.resolve_key(demand) for demand in canonical.ml_event_subscriptions
        )
        return ProvisionResource(
            subscription_id=subscription_id,
            representation=canonical,
            family_keys=family_keys,
            revision=revision,
        )

    @staticmethod
    def _copy(resource: ProvisionResource) -> ProvisionResource:
        return ProvisionResource(
            subscription_id=resource.subscription_id,
            representation=resource.representation.model_copy(deep=True),
            family_keys=resource.family_keys,
            revision=resource.revision,
        )
