from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from py_mtlf.core.accuracy_policy import ScopeReference
from py_mtlf.core.seed_catalog import FamilyKey


class TriggerSource(StrEnum):
    DEGRADATION = "degradation"
    PRIVATE_API = "private_api"


class ParticipantSource(StrEnum):
    MONITOR_SCOPES = "monitor_scopes"
    STATIC = "static"


@dataclass(frozen=True)
class FlatParticipantScope:
    scope_key: str
    participant_nf_instance_id: str
    model_ids: tuple[int, ...]
    ml_event: str
    ml_event_filter: dict
    target_ue: dict | None

    @classmethod
    def from_monitor_scope(cls, scope: ScopeReference) -> FlatParticipantScope:
        return cls(
            scope_key=scope.scope_key,
            participant_nf_instance_id=scope.consumer_id,
            model_ids=scope.model_ids,
            ml_event=scope.ml_event,
            ml_event_filter=dict(scope.ml_event_filter),
            target_ue=dict(scope.target_ue) if scope.target_ue is not None else None,
        )


@dataclass(frozen=True)
class MonitorParticipantSelection:
    participants: tuple[FlatParticipantScope, ...]
    source: ParticipantSource = ParticipantSource.MONITOR_SCOPES


@dataclass(frozen=True)
class StaticParticipantSelection:
    participants: tuple[FlatParticipantScope, ...]
    topology_version: int
    source: ParticipantSource = ParticipantSource.STATIC


FlatParticipantSelection = MonitorParticipantSelection | StaticParticipantSelection


@dataclass(frozen=True)
class FlatExecutionRequest:
    model_family_id: FamilyKey
    trigger_source: TriggerSource
    participant_selection: FlatParticipantSelection
    required_cutover_scope_keys: tuple[str, ...]
    triggering_scope_key: str | None
    request_id: str | None = None


class TopLevelCoordinatorError(RuntimeError):
    pass


class TopLevelRequestConflictError(TopLevelCoordinatorError):
    pass


class TopLevelModelFamilyNotFoundError(TopLevelCoordinatorError):
    pass


class TopLevelCoordinatorUnavailableError(TopLevelCoordinatorError):
    pass


class TopLevelRequestSnapshot(Protocol):
    request_id: str
    model_family_id: FamilyKey
    state: str
    mode: str
    participant_source: str
    trigger_source: str
    current_round: int | None
    completed_rounds: int
    candidate_digest: str
    failure_cause: str
    failure_detail: str


class TopLevelCoordinator(Protocol):
    def submit_manual(
        self,
        *,
        request_id: str,
        model_family_id: FamilyKey,
    ) -> TopLevelRequestSnapshot: ...

    def accept_policy_intents(self) -> None: ...

    def get(self, request_id: str) -> TopLevelRequestSnapshot | None: ...

    def close(self) -> None: ...

    def abort_generation(self, reason: str) -> None: ...
