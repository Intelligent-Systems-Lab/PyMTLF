import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4

from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog
from py_mtlf.wire.ml_model import MLModelProvisionSubscription


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
