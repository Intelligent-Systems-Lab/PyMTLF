from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from py_mtlf.core.generation_journal import GenerationJournal, ProvisionEvent
from py_mtlf.models import (
    ApplyEvidence,
    ApplyStatus,
    ObservedModelState,
    ObservedModelStatus,
)


class ActiveStateQuery(Protocol):
    async def get_active_model_state(self, event: ProvisionEvent) -> ObservedModelState: ...


class ReconciliationOutcome(StrEnum):
    APPLIED = "APPLIED"
    PENDING = "PENDING"
    CONFLICT = "CONFLICT"
    DIVERGED = "DIVERGED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ReconciliationResult:
    event_id: str
    outcome: ReconciliationOutcome
    detail: str = ""


class ActiveStateUnavailableError(RuntimeError):
    pass


class ReconciliationEngine:
    def __init__(self, journal: GenerationJournal, state_query: ActiveStateQuery | None):
        self._journal = journal
        self._state_query = state_query

    async def reconcile(self, event: ProvisionEvent) -> ReconciliationResult:
        if self._state_query is None:
            return ReconciliationResult(event.event_id, ReconciliationOutcome.UNAVAILABLE)
        try:
            state = await self._state_query.get_active_model_state(event)
        except (OSError, TimeoutError, ActiveStateUnavailableError) as exc:
            return ReconciliationResult(
                event.event_id, ReconciliationOutcome.UNAVAILABLE, type(exc).__name__
            )
        if state.status == ObservedModelStatus.UNAVAILABLE:
            return ReconciliationResult(event.event_id, ReconciliationOutcome.UNAVAILABLE)
        if state.model_identity != event.model_identity:
            return self._conflict(event, "active state model identity does not match pending event")
        if state.status == ObservedModelStatus.DIVERGED:
            return ReconciliationResult(
                event.event_id, ReconciliationOutcome.DIVERGED, state.divergence_summary
            )
        if state.status == ObservedModelStatus.NO_MATCH:
            return ReconciliationResult(event.event_id, ReconciliationOutcome.PENDING)
        if state.active_generation == event.target_generation:
            if state.active_artifact_digest == event.artifact.digest:
                self._journal.record_apply_result(
                    ApplyEvidence(
                        event_id=event.event_id,
                        model_identity=event.model_identity,
                        target_generation=event.target_generation,
                        status=ApplyStatus.APPLIED,
                        active_generation=event.target_generation,
                        active_artifact_digest=event.artifact.digest,
                        affected_runtime_count=state.matching_runtime_count,
                        completed_at=datetime.now(UTC),
                    )
                )
                return ReconciliationResult(event.event_id, ReconciliationOutcome.APPLIED)
            return self._conflict(
                event, "target generation is active with a different artifact digest"
            )
        if state.active_generation == event.base_generation:
            return ReconciliationResult(event.event_id, ReconciliationOutcome.PENDING)
        if (
            state.active_generation is not None
            and state.active_generation > event.target_generation
        ):
            return self._conflict(event, "active generation is ahead of pending target")
        return self._conflict(
            event, "active state does not match pending event base or target"
        )

    def _conflict(self, event: ProvisionEvent, detail: str) -> ReconciliationResult:
        self._journal.record_reconciliation_conflict(event.event_id, detail)
        return ReconciliationResult(event.event_id, ReconciliationOutcome.CONFLICT, detail)
