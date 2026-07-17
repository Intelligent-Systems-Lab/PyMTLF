import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import ReconciliationSettings
from py_mtlf.core.generation_journal import GenerationJournal, ProvisionState
from py_mtlf.core.reconciliation import ReconciliationEngine, ReconciliationOutcome
from py_mtlf.models import ArtifactRecord, ModelIdentity, ObservedModelState, ObservedModelStatus


class FakeStateQuery:
    def __init__(self, state: ObservedModelState | Exception):
        self.state = state

    async def get_active_model_state(self, _event):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


def pending_event(journal):
    return journal.allocate(
        event_id="event-1",
        training_job_id="job-1",
        model_identity=ModelIdentity(provider_id="local", model_unique_id=1),
        artifact=ArtifactRecord(
            url="http://127.0.0.1:9092/internal/v1/artifacts/" + "a" * 64,
            digest="a" * 64,
            size_bytes=123,
        ),
        analytics_event="UE_COMMUNICATION",
    )


def state(event, status=ObservedModelStatus.CONSISTENT, generation=None, digest=None):
    return ObservedModelState(
        model_identity=event.model_identity,
        status=status,
        active_generation=generation,
        active_artifact_digest=digest,
        matching_runtime_count=1,
        observed_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_target_generation_and_digest_reconciles_applied(tmp_path):
    journal = GenerationJournal(tmp_path / "state.sqlite3")
    journal.open()
    event = pending_event(journal)
    engine = ReconciliationEngine(
        journal,
        FakeStateQuery(
            state(
                event,
                generation=event.target_generation,
                digest=event.artifact.digest,
            )
        ),
    )

    result = await engine.reconcile(event)

    assert result.outcome == ReconciliationOutcome.APPLIED
    assert journal.get_event(event.event_id).state == ProvisionState.APPLIED
    journal.close()


@pytest.mark.asyncio
async def test_base_generation_remains_pending(tmp_path):
    journal = GenerationJournal(tmp_path / "state.sqlite3")
    journal.open()
    event = pending_event(journal)
    result = await ReconciliationEngine(
        journal, FakeStateQuery(state(event, generation=event.base_generation, digest=None))
    ).reconcile(event)

    assert result.outcome == ReconciliationOutcome.PENDING
    assert journal.get_event(event.event_id).state == ProvisionState.PENDING_DELIVERY
    journal.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("active_status", "generation", "digest", "expected"),
    [
        (ObservedModelStatus.DIVERGED, None, None, ReconciliationOutcome.DIVERGED),
        (ObservedModelStatus.UNAVAILABLE, None, None, ReconciliationOutcome.UNAVAILABLE),
        (ObservedModelStatus.NO_MATCH, None, None, ReconciliationOutcome.PENDING),
        (ObservedModelStatus.CONSISTENT, 2, "b" * 64, ReconciliationOutcome.CONFLICT),
        (ObservedModelStatus.CONSISTENT, 3, "a" * 64, ReconciliationOutcome.CONFLICT),
    ],
)
async def test_reconciliation_outcomes(
    tmp_path, active_status, generation, digest, expected
):
    journal = GenerationJournal(tmp_path / "state.sqlite3")
    journal.open()
    event = pending_event(journal)
    result = await ReconciliationEngine(
        journal, FakeStateQuery(state(event, active_status, generation, digest))
    ).reconcile(event)

    assert result.outcome == expected
    if expected == ReconciliationOutcome.CONFLICT:
        conflict = journal.get_event(event.event_id)
        assert conflict.state == ProvisionState.CONFLICT
        assert conflict.apply_code == "RECONCILIATION_CONFLICT"
    journal.close()


def test_unresolved_pending_event_blocks_readiness(settings):
    journal = GenerationJournal(settings.storage.database_path)
    journal.open()
    pending_event(journal)
    journal.close()

    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["reconciliation"] == "unresolved"


def test_background_reconciliation_transitions_readiness(settings):
    journal = GenerationJournal(settings.storage.database_path)
    journal.open()
    event = pending_event(journal)
    journal.close()
    engine = ReconciliationEngine(
        journal,
        FakeStateQuery(
            state(
                event,
                generation=event.target_generation,
                digest=event.artifact.digest,
            )
        ),
    )

    with TestClient(create_app(settings, journal=journal, reconciliation_engine=engine)) as client:
        response = client.get("/health/ready")
        for _ in range(20):
            if response.status_code == 200:
                break
            time.sleep(0.01)
            response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["reconciliation"] == "ready"


def test_background_reconciliation_task_is_cancelled_on_shutdown(settings):
    fast_settings = settings.model_copy(
        update={
            "reconciliation": ReconciliationSettings(
                retry_interval_seconds=0.01,
                shutdown_timeout_seconds=0.1,
            )
        }
    )
    journal = GenerationJournal(fast_settings.storage.database_path)
    journal.open()
    event = pending_event(journal)
    journal.close()
    engine = ReconciliationEngine(
        journal,
        FakeStateQuery(state(event, generation=event.base_generation, digest=None)),
    )

    started = time.monotonic()
    with TestClient(
        create_app(fast_settings, journal=journal, reconciliation_engine=engine)
    ) as client:
        assert client.get("/health/ready").status_code == 503

    assert time.monotonic() - started < 1


def test_unexpected_reconciliation_error_becomes_unresolved(settings):
    journal = GenerationJournal(settings.storage.database_path)
    journal.open()
    pending_event(journal)
    journal.close()
    engine = ReconciliationEngine(journal, FakeStateQuery(RuntimeError("unexpected")))

    with TestClient(create_app(settings, journal=journal, reconciliation_engine=engine)) as client:
        response = client.get("/health/ready")
        for _ in range(20):
            if response.json()["reconciliation"] == "unresolved":
                break
            time.sleep(0.01)
            response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["reconciliation"] == "unresolved"
