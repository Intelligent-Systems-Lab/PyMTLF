import threading

from py_mtlf.models import BackendSyncRequest


class SyncProjection:
    def __init__(self, lock=None) -> None:
        self._lock = lock or threading.RLock()
        self._snapshot: BackendSyncRequest | None = None

    def prepare(self, snapshot: BackendSyncRequest) -> BackendSyncRequest:
        identity_lists = (
            (
                "eventsSubscriptions.subscriptionId",
                [item.subscription_id for item in snapshot.events_subscriptions],
            ),
            (
                "smfResources.correlationId",
                [item.correlation_id for item in snapshot.smf_resources],
            ),
            (
                "trainingDataDescriptors.correlationId",
                [item.correlation_id for item in snapshot.training_data_descriptors],
            ),
            (
                "mlModelProvisionSubscriptions.subscriptionId",
                [item.subscription_id for item in snapshot.ml_model_provision_subscriptions],
            ),
            (
                "mlModelMonitorRegistrations.registrationId",
                [item.registration_id for item in snapshot.ml_model_monitor_registrations],
            ),
            (
                "mlModelMonitorSubscriptions.subscriptionId",
                [item.subscription_id for item in snapshot.ml_model_monitor_subscriptions],
            ),
            (
                "mlModelTrainingSubscriptions.subscriptionId",
                [item.subscription_id for item in snapshot.ml_model_training_subscriptions],
            ),
        )
        for name, identities in identity_lists:
            if len(identities) != len(set(identities)):
                raise ValueError(f"{name} contains duplicate values")
        return snapshot.model_copy(deep=True)

    def replace(self, snapshot: BackendSyncRequest) -> None:
        self.commit(self.prepare(snapshot))

    def commit(self, snapshot: BackendSyncRequest) -> None:
        with self._lock:
            self._snapshot = snapshot.model_copy(deep=True)

    def snapshot(self) -> BackendSyncRequest | None:
        with self._lock:
            return self._snapshot.model_copy(deep=True) if self._snapshot is not None else None
