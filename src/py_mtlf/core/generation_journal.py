import hashlib
import json
import sqlite3
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from py_mtlf.models import ApplyEvidence, ApplyStatus, ArtifactRecord, ModelIdentity

SCHEMA_VERSION = 1


class JournalError(RuntimeError):
    pass


class JournalConflictError(JournalError):
    pass


class JournalNotFoundError(JournalError):
    pass


class ProvisionState(StrEnum):
    PENDING_DELIVERY = "PENDING_DELIVERY"
    PENDING_APPLY = "PENDING_APPLY"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    STALE = "STALE"
    NO_MATCH = "NO_MATCH"
    CONFLICT = "CONFLICT"


TERMINAL_STATES = {
    ProvisionState.APPLIED,
    ProvisionState.FAILED,
    ProvisionState.STALE,
    ProvisionState.NO_MATCH,
    ProvisionState.CONFLICT,
}


@dataclass(frozen=True)
class ProvisionEvent:
    event_id: str
    training_job_id: str
    model_identity: ModelIdentity
    base_generation: int
    target_generation: int
    artifact: ArtifactRecord
    analytics_event: str
    state: ProvisionState
    apply_status: str | None
    apply_code: str
    apply_detail: str
    apply_active_generation: int | None
    apply_active_artifact_digest: str | None
    affected_runtime_count: int | None
    apply_result_digest: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    payload_digest: str


class GenerationJournal:
    def __init__(self, database_path: Path, busy_timeout_seconds: float = 5):
        self._database_path = Path(database_path)
        self._busy_timeout_ms = int(busy_timeout_seconds * 1000)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def open(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self._database_path,
                timeout=self._busy_timeout_ms / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
                connection.execute("PRAGMA journal_mode = WAL")
                self._connection = connection
                self._migrate()
                self._validate_invariants()
            except BaseException:
                with suppress(Exception):
                    connection.close()
                self._connection = None
                raise

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def probe(self) -> None:
        with self._lock:
            connection = self._require_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("SELECT version FROM schema_version")
                connection.execute("ROLLBACK")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def allocate(
        self,
        *,
        event_id: str,
        training_job_id: str,
        model_identity: ModelIdentity,
        artifact: ArtifactRecord,
        analytics_event: str,
    ) -> ProvisionEvent:
        now = self._now()
        canonical_payload = {
            "event_id": event_id,
            "training_job_id": training_job_id,
            "model_identity": model_identity.model_dump(mode="json"),
            "artifact": artifact.model_dump(mode="json"),
            "analytics_event": analytics_event,
        }
        payload_digest = self._payload_digest(canonical_payload)
        with self._lock:
            connection = self._require_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM provision_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if existing is not None:
                    if existing["payload_digest"] != payload_digest:
                        raise JournalConflictError("event_id is already bound to another payload")
                    connection.execute("COMMIT")
                    return self._row_to_event(existing)

                state = connection.execute(
                    """
                    SELECT * FROM model_generation_state
                    WHERE provider_id = ? AND model_unique_id = ?
                    """,
                    (model_identity.provider_id, model_identity.model_unique_id),
                ).fetchone()
                last_allocated = int(state["last_allocated_generation"]) if state else 0
                active_generation = int(state["active_generation"]) if state else 0
                target_generation = last_allocated + 1
                connection.execute(
                    """
                    INSERT INTO model_generation_state (
                        provider_id, model_unique_id, last_allocated_generation,
                        active_generation, active_artifact_digest, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider_id, model_unique_id) DO UPDATE SET
                        last_allocated_generation = excluded.last_allocated_generation,
                        updated_at = excluded.updated_at
                    """,
                    (
                        model_identity.provider_id,
                        model_identity.model_unique_id,
                        target_generation,
                        active_generation,
                        state["active_artifact_digest"] if state else None,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO provision_events (
                        event_id, provider_id, model_unique_id, training_job_id,
                        base_generation, target_generation, artifact_key, artifact_url,
                        artifact_digest, artifact_size, analytics_event, state,
                        apply_status, apply_code, apply_detail, apply_active_generation,
                        apply_active_artifact_digest, affected_runtime_count,
                        apply_result_digest, created_at, updated_at, completed_at,
                        payload_digest
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, '', '',
                        NULL, NULL, NULL, NULL, ?, ?, NULL, ?
                    )
                    """,
                    (
                        event_id,
                        model_identity.provider_id,
                        model_identity.model_unique_id,
                        training_job_id,
                        active_generation,
                        target_generation,
                        artifact.digest,
                        artifact.url,
                        artifact.digest,
                        artifact.size_bytes,
                        analytics_event,
                        ProvisionState.PENDING_DELIVERY,
                        now,
                        now,
                        payload_digest,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_event(event_id)

    def mark_delivery_accepted(self, event_id: str) -> ProvisionEvent:
        with self._lock:
            connection = self._require_connection()
            now = self._now()
            cursor = connection.execute(
                """
                UPDATE provision_events SET state = ?, updated_at = ?
                WHERE event_id = ? AND state = ?
                """,
                (
                    ProvisionState.PENDING_APPLY,
                    now,
                    event_id,
                    ProvisionState.PENDING_DELIVERY,
                ),
            )
            if cursor.rowcount == 0:
                event = self.get_event(event_id)
                if event.state != ProvisionState.PENDING_APPLY:
                    raise JournalConflictError("event is not pending delivery")
        return self.get_event(event_id)

    def record_apply_result(self, result: ApplyEvidence) -> ProvisionEvent:
        state = ProvisionState(result.status.value)
        result_digest = self._apply_result_digest(result)
        with self._lock:
            connection = self._require_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM provision_events WHERE event_id = ?", (result.event_id,)
                ).fetchone()
                if row is None:
                    raise JournalNotFoundError("provision event was not found")
                event = self._row_to_event(row)
                self._validate_result(event, result)
                if event.state in TERMINAL_STATES:
                    if event.apply_result_digest == result_digest:
                        connection.execute("COMMIT")
                        return event
                    raise JournalConflictError("event already has different terminal evidence")
                now = self._format_datetime(result.completed_at)
                connection.execute(
                    """
                    UPDATE provision_events
                    SET state = ?, apply_status = ?, apply_code = ?, apply_detail = ?,
                        apply_active_generation = ?, apply_active_artifact_digest = ?,
                        affected_runtime_count = ?, apply_result_digest = ?,
                        updated_at = ?, completed_at = ?
                    WHERE event_id = ?
                    """,
                    (
                        state,
                        result.status,
                        result.failure_code,
                        result.failure_detail,
                        result.active_generation,
                        result.active_artifact_digest,
                        result.affected_runtime_count,
                        result_digest,
                        now,
                        now,
                        result.event_id,
                    ),
                )
                if result.status == ApplyStatus.APPLIED:
                    connection.execute(
                        """
                        UPDATE model_generation_state
                        SET active_generation = ?, active_artifact_digest = ?, updated_at = ?
                        WHERE provider_id = ? AND model_unique_id = ?
                        """,
                        (
                            event.target_generation,
                            event.artifact.digest,
                            now,
                            event.model_identity.provider_id,
                            event.model_identity.model_unique_id,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_event(result.event_id)

    def record_reconciliation_conflict(self, event_id: str, detail: str) -> ProvisionEvent:
        with self._lock:
            connection = self._require_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM provision_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if row is None:
                    raise JournalNotFoundError("provision event was not found")
                event = self._row_to_event(row)
                if event.state == ProvisionState.CONFLICT:
                    if (
                        event.apply_code == "RECONCILIATION_CONFLICT"
                        and event.apply_detail == detail
                    ):
                        connection.execute("COMMIT")
                        return event
                    raise JournalConflictError("event already has different conflict evidence")
                if event.state in TERMINAL_STATES:
                    raise JournalConflictError("event already has terminal evidence")
                now = self._now()
                connection.execute(
                    """
                    UPDATE provision_events
                    SET state = ?, apply_status = ?, apply_code = ?, apply_detail = ?,
                        apply_active_generation = NULL,
                        apply_active_artifact_digest = NULL,
                        affected_runtime_count = 0, apply_result_digest = NULL,
                        updated_at = ?, completed_at = ?
                    WHERE event_id = ?
                    """,
                    (
                        ProvisionState.CONFLICT,
                        ApplyStatus.CONFLICT,
                        "RECONCILIATION_CONFLICT",
                        detail,
                        now,
                        now,
                        event_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_event(event_id)

    def get_event(self, event_id: str) -> ProvisionEvent:
        with self._lock:
            row = self._require_connection().execute(
                "SELECT * FROM provision_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise JournalNotFoundError("provision event was not found")
        return self._row_to_event(row)

    def list_pending(self) -> list[ProvisionEvent]:
        with self._lock:
            rows = self._require_connection().execute(
                """
                SELECT * FROM provision_events
                WHERE state IN (?, ?) ORDER BY created_at, event_id
                """,
                (ProvisionState.PENDING_DELIVERY, ProvisionState.PENDING_APPLY),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def generation_state(self, identity: ModelIdentity) -> tuple[int, int, str | None]:
        with self._lock:
            row = self._require_connection().execute(
                """
                SELECT last_allocated_generation, active_generation, active_artifact_digest
                FROM model_generation_state
                WHERE provider_id = ? AND model_unique_id = ?
                """,
                (identity.provider_id, identity.model_unique_id),
            ).fetchone()
        if row is None:
            return 0, 0, None
        return (
            int(row["last_allocated_generation"]),
            int(row["active_generation"]),
            row["active_artifact_digest"],
        )

    def record_training_terminal(
        self,
        training_job_id: str,
        identity: ModelIdentity,
        status: str,
        reason: str,
        artifact_digest: str | None,
    ) -> None:
        now = self._now()
        with self._lock:
            connection = self._require_connection()
            try:
                connection.execute(
                    """
                    INSERT INTO training_job_terminal_state (
                        training_job_id, provider_id, model_unique_id, terminal_status,
                        terminal_reason, candidate_artifact_digest, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        training_job_id,
                        identity.provider_id,
                        identity.model_unique_id,
                        status,
                        reason,
                        artifact_digest,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    "SELECT * FROM training_job_terminal_state WHERE training_job_id = ?",
                    (training_job_id,),
                ).fetchone()
                same = (
                    existing is not None
                    and existing["provider_id"] == identity.provider_id
                    and existing["model_unique_id"] == identity.model_unique_id
                    and existing["terminal_status"] == status
                    and existing["terminal_reason"] == reason
                    and existing["candidate_artifact_digest"] == artifact_digest
                )
                if not same:
                    raise JournalConflictError(
                        "training job already has different terminal evidence"
                    ) from exc

    def _migrate(self) -> None:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            row = connection.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_version(version) VALUES (0)")
                current_version = 0
            else:
                current_version = int(row["version"])
            if current_version > SCHEMA_VERSION:
                raise JournalError("database schema is newer than this service")
            if current_version < 1:
                statements = (
                    """
                    CREATE TABLE model_generation_state (
                        provider_id TEXT NOT NULL,
                        model_unique_id INTEGER NOT NULL,
                        last_allocated_generation INTEGER NOT NULL
                          CHECK(last_allocated_generation >= 0),
                        active_generation INTEGER NOT NULL CHECK(active_generation >= 0),
                        active_artifact_digest TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(provider_id, model_unique_id),
                        CHECK(active_generation <= last_allocated_generation)
                    )
                    """,
                    """
                    CREATE TABLE provision_events (
                        event_id TEXT PRIMARY KEY,
                        provider_id TEXT NOT NULL,
                        model_unique_id INTEGER NOT NULL,
                        training_job_id TEXT NOT NULL,
                        base_generation INTEGER NOT NULL,
                        target_generation INTEGER NOT NULL,
                        artifact_key TEXT NOT NULL,
                        artifact_url TEXT NOT NULL,
                        artifact_digest TEXT NOT NULL,
                        artifact_size INTEGER NOT NULL,
                        analytics_event TEXT NOT NULL,
                        state TEXT NOT NULL,
                        apply_status TEXT,
                        apply_code TEXT NOT NULL,
                        apply_detail TEXT NOT NULL,
                        apply_active_generation INTEGER CHECK(apply_active_generation >= 0),
                        apply_active_artifact_digest TEXT,
                        affected_runtime_count INTEGER CHECK(affected_runtime_count >= 0),
                        apply_result_digest TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        payload_digest TEXT NOT NULL,
                        UNIQUE(provider_id, model_unique_id, target_generation),
                        FOREIGN KEY(provider_id, model_unique_id)
                          REFERENCES model_generation_state(provider_id, model_unique_id)
                    )
                    """,
                    """
                    CREATE TABLE training_job_terminal_state (
                        training_job_id TEXT PRIMARY KEY,
                        provider_id TEXT NOT NULL,
                        model_unique_id INTEGER NOT NULL,
                        terminal_status TEXT NOT NULL,
                        terminal_reason TEXT NOT NULL,
                        candidate_artifact_digest TEXT,
                        completed_at TEXT NOT NULL
                    )
                    """,
                )
                for statement in statements:
                    connection.execute(statement)
                connection.execute("UPDATE schema_version SET version = 1")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _validate_invariants(self) -> None:
        connection = self._require_connection()
        version_rows = connection.execute("SELECT version FROM schema_version").fetchall()
        if len(version_rows) != 1 or int(version_rows[0]["version"]) != SCHEMA_VERSION:
            raise JournalError("generation journal schema version invariant violation")
        provision_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(provision_events)")
        }
        required_columns = {
            "apply_active_generation",
            "apply_active_artifact_digest",
            "affected_runtime_count",
            "apply_result_digest",
        }
        if not required_columns.issubset(provision_columns):
            raise JournalError("generation journal schema is incomplete")
        row = connection.execute(
            """
            SELECT COUNT(*) AS count FROM model_generation_state
            WHERE active_generation > last_allocated_generation
            """
        ).fetchone()
        if row["count"]:
            raise JournalError("generation journal invariant violation")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise JournalError("generation journal is not open")
        return self._connection

    @staticmethod
    def _validate_result(event: ProvisionEvent, result: ApplyEvidence) -> None:
        if result.model_identity != event.model_identity:
            raise JournalConflictError("apply result model identity does not match event")
        if result.target_generation != event.target_generation:
            raise JournalConflictError("apply result target generation does not match event")
        if result.status == ApplyStatus.APPLIED:
            if result.active_generation != event.target_generation:
                raise JournalConflictError("APPLIED result must confirm target generation")
            if result.active_artifact_digest != event.artifact.digest:
                raise JournalConflictError("APPLIED result must confirm target artifact digest")

    @classmethod
    def _apply_result_digest(cls, result: ApplyEvidence) -> str:
        payload = result.model_dump(mode="json")
        payload["completed_at"] = cls._format_datetime(result.completed_at)
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ProvisionEvent:
        return ProvisionEvent(
            event_id=row["event_id"],
            training_job_id=row["training_job_id"],
            model_identity=ModelIdentity(
                provider_id=row["provider_id"], model_unique_id=row["model_unique_id"]
            ),
            base_generation=row["base_generation"],
            target_generation=row["target_generation"],
            artifact=ArtifactRecord(
                url=row["artifact_url"],
                digest=row["artifact_digest"],
                size_bytes=row["artifact_size"],
            ),
            analytics_event=row["analytics_event"],
            state=ProvisionState(row["state"]),
            apply_status=row["apply_status"],
            apply_code=row["apply_code"],
            apply_detail=row["apply_detail"],
            apply_active_generation=row["apply_active_generation"],
            apply_active_artifact_digest=row["apply_active_artifact_digest"],
            affected_runtime_count=row["affected_runtime_count"],
            apply_result_digest=row["apply_result_digest"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            payload_digest=row["payload_digest"],
        )

    @staticmethod
    def _payload_digest(payload: dict[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @classmethod
    def _now(cls) -> str:
        return cls._format_datetime(datetime.now(UTC))
