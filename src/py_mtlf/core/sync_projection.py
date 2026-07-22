import threading

from py_mtlf.models import BackendSyncRequest


class SyncProjection:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: BackendSyncRequest | None = None

    def replace(self, snapshot: BackendSyncRequest) -> None:
        with self._lock:
            self._snapshot = snapshot.model_copy(deep=True)

    def snapshot(self) -> BackendSyncRequest | None:
        with self._lock:
            return self._snapshot.model_copy(deep=True) if self._snapshot is not None else None
