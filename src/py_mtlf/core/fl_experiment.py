import threading
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4


class ExperimentRole(StrEnum):
    ROOT = "ROOT"
    BRANCH = "BRANCH"
    LEAF = "LEAF"


class ExperimentLifecycle(StrEnum):
    PROVISIONAL = "PROVISIONAL"
    ACTIVE = "ACTIVE"
    TERMINAL = "TERMINAL"
    CLEANING = "CLEANING"


class ExperimentRegistryError(RuntimeError):
    pass


class ExperimentConflictError(ExperimentRegistryError):
    pass


class ExperimentStateError(ExperimentRegistryError):
    pass


@dataclass(frozen=True)
class ExperimentSnapshot:
    reservation_id: str
    plan_id: str | None
    assigned_role: ExperimentRole | None
    lifecycle: ExperimentLifecycle
    upper_client_ml_correlation_id: str | None
    upper_client_subscription_ids: frozenset[str]
    server_process_id: str | None
    terminal_outcome: str | None
    cleanup_pending: bool


@dataclass
class _ExperimentRecord:
    reservation_id: str
    lifecycle: ExperimentLifecycle
    plan_id: str | None = None
    assigned_role: ExperimentRole | None = None
    upper_client_ml_correlation_id: str | None = None
    upper_client_subscription_ids: set[str] = field(default_factory=set)
    server_process_id: str | None = None
    terminal_outcome: str | None = None
    cleanup_pending: bool = False


class FLExperimentRegistry:
    """Arbitrate one process-local top-level experiment.

    Registry methods perform no callbacks, I/O, or blocking waits. Callers must
    acquire a registry reservation before entering an engine-owned state lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: _ExperimentRecord | None = None
        self._retired_plan_ids: set[str] = set()
        self._shutting_down = False

    def reserve_client(
        self,
        subscription_id: str,
        ml_correlation_id: str,
    ) -> ExperimentSnapshot:
        subscription_id = _required_identity(subscription_id, "subscription_id")
        ml_correlation_id = _required_identity(ml_correlation_id, "ml_correlation_id")
        with self._lock:
            self._ensure_admission_open()
            if self._active is None:
                self._active = _ExperimentRecord(
                    reservation_id=str(uuid4()),
                    lifecycle=ExperimentLifecycle.PROVISIONAL,
                    upper_client_ml_correlation_id=ml_correlation_id,
                    upper_client_subscription_ids={subscription_id},
                )
                return self._snapshot(self._active)

            record = self._active
            if record.lifecycle not in {
                ExperimentLifecycle.PROVISIONAL,
                ExperimentLifecycle.ACTIVE,
            }:
                raise ExperimentConflictError("the active experiment is terminating")
            if record.upper_client_ml_correlation_id != ml_correlation_id:
                raise ExperimentConflictError("another top-level experiment is active")
            if subscription_id in record.upper_client_subscription_ids:
                raise ExperimentStateError("client subscription is already reserved")
            record.upper_client_subscription_ids.add(subscription_id)
            return self._snapshot(record)

    def rollback_client(
        self,
        reservation_id: str,
        subscription_id: str,
    ) -> ExperimentSnapshot | None:
        with self._lock:
            record = self._required(reservation_id)
            if (
                record.lifecycle is not ExperimentLifecycle.PROVISIONAL
                or record.plan_id is not None
            ):
                raise ExperimentStateError("only an unbound provisional client group can roll back")
            if subscription_id not in record.upper_client_subscription_ids:
                raise ExperimentStateError("client subscription is not reserved")
            record.upper_client_subscription_ids.remove(subscription_id)
            if not record.upper_client_subscription_ids:
                self._active = None
                return None
            return self._snapshot(record)

    def remove_client(
        self,
        reservation_id: str,
        subscription_id: str,
    ) -> ExperimentSnapshot | None:
        subscription_id = _required_identity(subscription_id, "subscription_id")
        with self._lock:
            record = self._required(reservation_id)
            if subscription_id not in record.upper_client_subscription_ids:
                raise ExperimentStateError("client subscription is not reserved")
            if (
                record.lifecycle is ExperimentLifecycle.PROVISIONAL
                and record.plan_id is None
            ):
                record.upper_client_subscription_ids.remove(subscription_id)
                if not record.upper_client_subscription_ids:
                    self._active = None
                    return None
                return self._snapshot(record)
            if record.lifecycle is not ExperimentLifecycle.CLEANING:
                raise ExperimentStateError(
                    "bound client removal requires experiment cleanup"
                )
            record.upper_client_subscription_ids.remove(subscription_id)
            return self._snapshot(record)

    def bind_plan(
        self,
        reservation_id: str,
        plan_id: str,
        role: ExperimentRole,
    ) -> ExperimentSnapshot:
        plan_id = _uuid4_identity(plan_id, "plan_id")
        if not isinstance(role, ExperimentRole):
            raise ValueError("role must be an ExperimentRole")
        if role not in {ExperimentRole.BRANCH, ExperimentRole.LEAF}:
            raise ExperimentStateError("an upper client group can only bind BRANCH or LEAF")
        with self._lock:
            self._ensure_admission_open()
            record = self._required(reservation_id)
            if plan_id in self._retired_plan_ids:
                raise ExperimentConflictError("plan_id is retired in this process")
            if (
                record.lifecycle is not ExperimentLifecycle.PROVISIONAL
                or record.plan_id is not None
            ):
                raise ExperimentStateError("experiment plan is already bound")
            if not record.upper_client_subscription_ids:
                raise ExperimentStateError("client plan binding requires a reserved client group")
            record.plan_id = plan_id
            record.assigned_role = role
            record.lifecycle = ExperimentLifecycle.ACTIVE
            return self._snapshot(record)

    def reserve_root(self, plan_id: str) -> ExperimentSnapshot:
        plan_id = _uuid4_identity(plan_id, "plan_id")
        with self._lock:
            self._ensure_admission_open()
            self._ensure_slot_available()
            if plan_id in self._retired_plan_ids:
                raise ExperimentConflictError("plan_id is retired in this process")
            self._active = _ExperimentRecord(
                reservation_id=str(uuid4()),
                lifecycle=ExperimentLifecycle.ACTIVE,
                plan_id=plan_id,
                assigned_role=ExperimentRole.ROOT,
            )
            return self._snapshot(self._active)

    def reserve_server(self, process_id: str) -> ExperimentSnapshot:
        process_id = _required_identity(process_id, "process_id")
        with self._lock:
            self._ensure_admission_open()
            self._ensure_slot_available()
            self._active = _ExperimentRecord(
                reservation_id=str(uuid4()),
                lifecycle=ExperimentLifecycle.ACTIVE,
                server_process_id=process_id,
            )
            return self._snapshot(self._active)

    def attach_server(
        self,
        reservation_id: str,
        plan_id: str,
        process_id: str,
    ) -> ExperimentSnapshot:
        plan_id = _uuid4_identity(plan_id, "plan_id")
        process_id = _required_identity(process_id, "process_id")
        with self._lock:
            self._ensure_admission_open()
            record = self._required(reservation_id)
            if record.lifecycle is not ExperimentLifecycle.ACTIVE:
                raise ExperimentStateError("server attachment requires an active experiment")
            if record.plan_id != plan_id:
                raise ExperimentStateError("server plan_id does not match the active experiment")
            if record.assigned_role not in {ExperimentRole.ROOT, ExperimentRole.BRANCH}:
                raise ExperimentStateError("the assigned role cannot own an FL Server process")
            if record.server_process_id is not None:
                raise ExperimentStateError("an FL Server process is already attached")
            record.server_process_id = process_id
            return self._snapshot(record)

    def detach_server(
        self,
        reservation_id: str,
        process_id: str,
    ) -> ExperimentSnapshot:
        process_id = _required_identity(process_id, "process_id")
        with self._lock:
            record = self._required(reservation_id)
            if record.server_process_id != process_id:
                raise ExperimentStateError("server process is not attached")
            record.server_process_id = None
            return self._snapshot(record)

    def mark_terminal(
        self,
        reservation_id: str,
        outcome: str,
    ) -> ExperimentSnapshot:
        outcome = _required_identity(outcome, "outcome")
        with self._lock:
            record = self._required(reservation_id)
            if record.lifecycle not in {
                ExperimentLifecycle.PROVISIONAL,
                ExperimentLifecycle.ACTIVE,
            }:
                raise ExperimentStateError("experiment is already terminating")
            record.lifecycle = ExperimentLifecycle.TERMINAL
            record.terminal_outcome = outcome
            record.cleanup_pending = True
            return self._snapshot(record)

    def begin_cleanup(self, reservation_id: str) -> ExperimentSnapshot:
        with self._lock:
            record = self._required(reservation_id)
            if record.lifecycle is not ExperimentLifecycle.TERMINAL:
                raise ExperimentStateError("cleanup requires a terminal experiment")
            record.lifecycle = ExperimentLifecycle.CLEANING
            return self._snapshot(record)

    def release(self, reservation_id: str) -> None:
        with self._lock:
            record = self._required(reservation_id)
            if record.lifecycle is not ExperimentLifecycle.CLEANING:
                raise ExperimentStateError("release requires completed cleanup")
            if record.plan_id is not None:
                self._retired_plan_ids.add(record.plan_id)
            self._active = None

    def active(self) -> ExperimentSnapshot | None:
        with self._lock:
            return self._snapshot(self._active) if self._active is not None else None

    def for_client_subscription(self, subscription_id: str) -> ExperimentSnapshot | None:
        with self._lock:
            if self._active is None:
                return None
            if subscription_id not in self._active.upper_client_subscription_ids:
                return None
            return self._snapshot(self._active)

    def for_server_process(self, process_id: str) -> ExperimentSnapshot | None:
        with self._lock:
            if self._active is None or self._active.server_process_id != process_id:
                return None
            return self._snapshot(self._active)

    def is_retired(self, plan_id: str) -> bool:
        with self._lock:
            return plan_id in self._retired_plan_ids

    def retired_plan_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._retired_plan_ids)

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    def _ensure_admission_open(self) -> None:
        if self._shutting_down:
            raise ExperimentStateError("experiment registry is shutting down")

    def _ensure_slot_available(self) -> None:
        if self._active is not None:
            raise ExperimentConflictError("another top-level experiment is active")

    def _required(self, reservation_id: str) -> _ExperimentRecord:
        if self._active is None or self._active.reservation_id != reservation_id:
            raise ExperimentStateError("experiment reservation was not found")
        return self._active

    @staticmethod
    def _snapshot(record: _ExperimentRecord) -> ExperimentSnapshot:
        return ExperimentSnapshot(
            reservation_id=record.reservation_id,
            plan_id=record.plan_id,
            assigned_role=record.assigned_role,
            lifecycle=record.lifecycle,
            upper_client_ml_correlation_id=record.upper_client_ml_correlation_id,
            upper_client_subscription_ids=frozenset(record.upper_client_subscription_ids),
            server_process_id=record.server_process_id,
            terminal_outcome=record.terminal_outcome,
            cleanup_pending=record.cleanup_pending,
        )


def _required_identity(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if normalized != value:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _uuid4_identity(value: str, field_name: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUIDv4")
    return value
