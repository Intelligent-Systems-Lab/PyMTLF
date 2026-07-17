import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from py_mtlf.core.generation_journal import (
    GenerationJournal,
    JournalConflictError,
    JournalError,
    ProvisionState,
)
from py_mtlf.models import (
    ApplyEvidence,
    ApplyStatus,
    ArtifactRecord,
    ModelIdentity,
)


def artifact(digest: str = "a" * 64) -> ArtifactRecord:
    return ArtifactRecord(
        url=f"http://127.0.0.1:9092/internal/v1/artifacts/{digest}",
        digest=digest,
        size_bytes=123,
    )


def allocate(
    journal: GenerationJournal,
    event_id: str,
    *,
    identity: ModelIdentity | None = None,
    descriptor: ArtifactRecord | None = None,
):
    return journal.allocate(
        event_id=event_id,
        training_job_id=f"job-{event_id}",
        model_identity=identity or ModelIdentity(provider_id="local", model_unique_id=1),
        artifact=descriptor or artifact(),
        analytics_event="UE_COMMUNICATION",
    )


@pytest.fixture
def journal(tmp_path):
    value = GenerationJournal(tmp_path / "state.sqlite3")
    value.open()
    yield value
    value.close()


def test_first_allocation_uses_base_zero_target_one(journal):
    event = allocate(journal, "event-1")

    assert event.base_generation == 0
    assert event.target_generation == 1
    assert event.state == ProvisionState.PENDING_DELIVERY
    assert journal.generation_state(event.model_identity) == (1, 0, None)


def test_allocations_are_monotonic_and_failed_generation_is_not_reused(journal):
    first = allocate(journal, "event-1")
    journal.record_apply_result(
        ApplyEvidence(
            event_id=first.event_id,
            model_identity=first.model_identity,
            target_generation=first.target_generation,
            status=ApplyStatus.FAILED,
            affected_runtime_count=0,
            failure_code="LOAD_FAILED",
            completed_at=datetime.now(UTC),
        )
    )

    second = allocate(journal, "event-2")

    assert second.base_generation == 0
    assert second.target_generation == 2


def test_only_applied_updates_confirmed_active_generation(journal):
    first = allocate(journal, "event-1")
    journal.record_apply_result(
        ApplyEvidence(
            event_id=first.event_id,
            model_identity=first.model_identity,
            target_generation=first.target_generation,
            status=ApplyStatus.APPLIED,
            active_generation=first.target_generation,
            active_artifact_digest=first.artifact.digest,
            affected_runtime_count=2,
            completed_at=datetime.now(UTC),
        )
    )

    assert journal.generation_state(first.model_identity) == (1, 1, first.artifact.digest)
    second = allocate(journal, "rollback", descriptor=artifact("b" * 64))
    assert (second.base_generation, second.target_generation) == (1, 2)


def test_duplicate_event_is_idempotent_but_changed_payload_conflicts(journal):
    first = allocate(journal, "event-1")
    duplicate = allocate(journal, "event-1")

    assert duplicate == first
    with pytest.raises(JournalConflictError):
        allocate(journal, "event-1", descriptor=artifact("b" * 64))
    assert allocate(journal, "event-2").target_generation == 2


def test_duplicate_terminal_result_is_idempotent_but_changed_result_conflicts(journal):
    event = allocate(journal, "event-1")
    result = ApplyEvidence(
        event_id=event.event_id,
        model_identity=event.model_identity,
        target_generation=event.target_generation,
        status=ApplyStatus.FAILED,
        affected_runtime_count=0,
        failure_code="LOAD_FAILED",
        completed_at=datetime.now(UTC),
    )

    first = journal.record_apply_result(result)
    second = journal.record_apply_result(result)

    assert first.state == ProvisionState.FAILED
    assert second.state == ProvisionState.FAILED
    assert first.affected_runtime_count == 0
    assert first.apply_result_digest is not None
    with pytest.raises(JournalConflictError):
        journal.record_apply_result(result.model_copy(update={"failure_code": "OTHER"}))


@pytest.mark.parametrize(
    "update",
    [
        {"affected_runtime_count": 3},
        {"active_generation": 1},
        {"active_artifact_digest": "b" * 64},
        {"completed_at": datetime(2026, 7, 17, tzinfo=UTC)},
    ],
)
def test_changed_terminal_evidence_conflicts(journal, update):
    event = allocate(journal, "event-1")
    result = ApplyEvidence(
        event_id=event.event_id,
        model_identity=event.model_identity,
        target_generation=event.target_generation,
        status=ApplyStatus.FAILED,
        affected_runtime_count=0,
        failure_code="LOAD_FAILED",
        completed_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    journal.record_apply_result(result)

    with pytest.raises(JournalConflictError, match="different terminal evidence"):
        journal.record_apply_result(result.model_copy(update=update))


def test_concurrent_allocations_serialize_per_identity(journal):
    with ThreadPoolExecutor(max_workers=8) as executor:
        events = list(executor.map(lambda index: allocate(journal, f"event-{index}"), range(20)))

    assert sorted(event.target_generation for event in events) == list(range(1, 21))


def test_different_model_identities_allocate_independently(journal):
    first = allocate(
        journal,
        "event-a",
        identity=ModelIdentity(provider_id="provider-a", model_unique_id=1),
    )
    second = allocate(
        journal,
        "event-b",
        identity=ModelIdentity(provider_id="provider-b", model_unique_id=1),
    )

    assert first.target_generation == 1
    assert second.target_generation == 1


def test_rollback_reuses_old_digest_with_a_new_generation(journal):
    first = allocate(journal, "event-1", descriptor=artifact("a" * 64))
    journal.record_apply_result(
        ApplyEvidence(
            event_id=first.event_id,
            model_identity=first.model_identity,
            target_generation=first.target_generation,
            status=ApplyStatus.APPLIED,
            active_generation=first.target_generation,
            active_artifact_digest=first.artifact.digest,
            affected_runtime_count=1,
            completed_at=datetime.now(UTC),
        )
    )
    failed = allocate(journal, "event-2", descriptor=artifact("b" * 64))
    journal.record_apply_result(
        ApplyEvidence(
            event_id=failed.event_id,
            model_identity=failed.model_identity,
            target_generation=failed.target_generation,
            status=ApplyStatus.FAILED,
            affected_runtime_count=0,
            completed_at=datetime.now(UTC),
        )
    )

    rollback = allocate(journal, "event-3", descriptor=artifact("a" * 64))

    assert rollback.base_generation == 1
    assert rollback.target_generation == 3
    assert rollback.artifact.digest == first.artifact.digest


def test_reopen_preserves_generation_and_pending_event(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = GenerationJournal(path)
    first.open()
    event = allocate(first, "event-1")
    first.close()

    reopened = GenerationJournal(path)
    reopened.open()
    try:
        assert reopened.get_event(event.event_id) == event
        assert reopened.list_pending() == [event]
        assert reopened.generation_state(event.model_identity) == (1, 0, None)
    finally:
        reopened.close()


def test_reopen_preserves_complete_terminal_evidence(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = GenerationJournal(path)
    first.open()
    event = allocate(first, "event-1")
    result = ApplyEvidence(
        event_id=event.event_id,
        model_identity=event.model_identity,
        target_generation=event.target_generation,
        status=ApplyStatus.FAILED,
        active_generation=0,
        affected_runtime_count=2,
        failure_code="LOAD_FAILED",
        failure_detail="candidate rejected",
        completed_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    terminal = first.record_apply_result(result)
    first.close()

    reopened = GenerationJournal(path)
    reopened.open()
    try:
        persisted = reopened.get_event(event.event_id)
        assert persisted == terminal
        assert persisted.apply_active_generation == 0
        assert persisted.affected_runtime_count == 2
        assert persisted.apply_result_digest is not None
    finally:
        reopened.close()


def test_training_terminal_state_conflict_is_rejected(journal):
    identity = ModelIdentity(provider_id="local", model_unique_id=1)
    journal.record_training_terminal("job-1", identity, "FAILED", "reason", None)
    journal.record_training_terminal("job-1", identity, "FAILED", "reason", None)

    with pytest.raises(JournalConflictError):
        journal.record_training_terminal("job-1", identity, "SUCCEEDED", "", "a" * 64)


def test_training_terminal_state_survives_reopen(tmp_path):
    path = tmp_path / "state.sqlite3"
    identity = ModelIdentity(provider_id="local", model_unique_id=1)
    first = GenerationJournal(path)
    first.open()
    first.record_training_terminal("job-1", identity, "FAILED", "reason", None)
    first.close()

    reopened = GenerationJournal(path)
    reopened.open()
    try:
        reopened.record_training_terminal("job-1", identity, "FAILED", "reason", None)
        with pytest.raises(JournalConflictError):
            reopened.record_training_terminal("job-1", identity, "SUCCEEDED", "", "a" * 64)
    finally:
        reopened.close()


def test_newer_database_schema_fails_closed(tmp_path):
    path = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_version(version) VALUES (2)")
    connection.commit()
    connection.close()

    journal = GenerationJournal(path)
    with pytest.raises(JournalError, match="newer"):
        journal.open()
    with pytest.raises(JournalError, match="not open"):
        journal.probe()
    with pytest.raises(JournalError, match="newer"):
        journal.open()


def test_incomplete_schema_fails_closed(tmp_path):
    path = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_version(version) VALUES (1)")
    connection.execute("CREATE TABLE provision_events (event_id TEXT PRIMARY KEY)")
    connection.execute(
        """
        CREATE TABLE model_generation_state (
            provider_id TEXT NOT NULL,
            model_unique_id INTEGER NOT NULL,
            last_allocated_generation INTEGER NOT NULL,
            active_generation INTEGER NOT NULL,
            PRIMARY KEY(provider_id, model_unique_id)
        )
        """
    )
    connection.commit()
    connection.close()

    journal = GenerationJournal(path)
    with pytest.raises(JournalError, match="incomplete"):
        journal.open()
    with pytest.raises(JournalError, match="not open"):
        journal.probe()
