from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

from py_mtlf.config import OrchestrationSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent
from py_mtlf.core.fl_orchestration import (
    FlatExecutionRequest,
    FlatParticipantScope,
    MonitorParticipantSelection,
    ParticipantSource,
    StaticParticipantSelection,
    TopLevelCoordinatorUnavailableError,
    TopLevelModelFamilyNotFoundError,
    TopLevelRequestConflictError,
    TriggerSource,
)
from py_mtlf.core.fl_server import (
    FLProcess,
    FLServerAdmissionClosedError,
    FLServerEngine,
    FLServerProcessConflictError,
    FLServerState,
)
from py_mtlf.core.fl_topology import StaticFlatTopologyPlanner
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog

logger = logging.getLogger(__name__)


_TERMINAL_STATES = {
    FLServerState.CANDIDATE_READY,
    FLServerState.VALIDATION_REJECTED,
    FLServerState.COMPLETE,
    FLServerState.FAILED,
}


@dataclass(frozen=True)
class FlatRequestSnapshot:
    request_id: str
    model_family_id: FamilyKey
    state: str
    mode: str
    participant_source: str
    trigger_source: str
    current_round: int | None = None
    completed_rounds: int = 0
    candidate_digest: str = ""
    failure_cause: str = ""
    failure_detail: str = ""


@dataclass
class _FlatRequestRecord:
    request_id: str
    model_family_id: FamilyKey
    trigger_source: TriggerSource
    participant_source: ParticipantSource
    process: FLProcess
    generation: int
    terminal_at: float | None = None


class FlatFLCoordinator:
    def __init__(
        self,
        *,
        orchestration: OrchestrationSettings,
        server: FLServerEngine,
        policy: AccuracyPolicy,
        catalog: ModelCatalog,
        nwdaf_context: NwdafContextClient,
        planner: StaticFlatTopologyPlanner | None,
        terminal_status_ttl_seconds: int,
        clock=time.monotonic,
    ) -> None:
        if orchestration.mode != "flat":
            raise ValueError("Flat coordinator requires flat orchestration")
        if terminal_status_ttl_seconds <= 0:
            raise ValueError("terminal_status_ttl_seconds must be positive")
        if orchestration.participant_source == "static" and planner is None:
            raise ValueError("static flat orchestration requires a topology planner")
        if orchestration.participant_source == "monitor_scopes" and planner is not None:
            raise ValueError("monitor-scoped flat orchestration must not have a topology planner")
        self._orchestration = orchestration
        self._server = server
        self._policy = policy
        self._catalog = catalog
        self._nwdaf_context = nwdaf_context
        self._planner = planner
        self._terminal_status_ttl_seconds = terminal_status_ttl_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, _FlatRequestRecord] = {}
        self._active_request_id: str | None = None
        self._generation = 0
        self._failure_latched = False
        self._closing = False

    def submit_manual(
        self,
        *,
        request_id: str,
        model_family_id: FamilyKey,
    ) -> FlatRequestSnapshot:
        normalized_request_id = _uuid4_identity(request_id)
        model_family_id = _required_identity(model_family_id, "model_family_id")
        if self._orchestration.participant_source != "static":
            raise TopLevelRequestConflictError(
                "manual flat training requires static participant selection"
            )
        with self._lock:
            self._refresh_locked()
            existing = self._records.get(normalized_request_id)
            if existing is not None:
                if existing.model_family_id != model_family_id:
                    raise TopLevelRequestConflictError(
                        "request_id is already bound to a different model family"
                    )
                return self._snapshot(existing)
            if self._closing:
                raise TopLevelCoordinatorUnavailableError("Flat coordinator is closing")
            if self._active_request_id is not None:
                raise TopLevelRequestConflictError(
                    "another top-level training request is active"
                )
            current = self._catalog.current(model_family_id)
            if current is None:
                raise TopLevelModelFamilyNotFoundError(
                    f"model family {model_family_id} was not found"
                )
            selection = self._static_selection(model_family_id)
            execution = FlatExecutionRequest(
                model_family_id=model_family_id,
                trigger_source=TriggerSource.PRIVATE_API,
                participant_selection=selection,
                required_cutover_scope_keys=(),
                triggering_scope_key=None,
                request_id=normalized_request_id,
            )
            record = self._start_locked(normalized_request_id, execution)
            self._failure_latched = False
            return self._snapshot(record)

    def accept_policy_intents(self) -> None:
        with self._lock:
            self._refresh_locked()
            if self._closing:
                return
            if self._active_request_id is not None:
                self._policy.discard_intents()
                logger.info("Discarded degradation intent while flat training is active")
                return
            if self._failure_latched:
                self._policy.discard_intents()
                logger.warning("Ignored degradation intent due to flat failure latch")
                return
            intents = self._policy.take_intents()
            for index, intent in enumerate(intents):
                if index > 0:
                    self._policy.complete_retrain(intent.family_key)
                    continue
                try:
                    execution = self._execution_from_intent(intent)
                    self._start_locked(str(uuid4()), execution)
                except Exception:
                    self._policy.complete_retrain(intent.family_key)
                    self._failure_latched = True
                    logger.exception(
                        "Failed to accept flat degradation intent family=%s",
                        intent.family_key,
                    )

    def get(self, request_id: str) -> FlatRequestSnapshot | None:
        try:
            normalized = _uuid4_identity(request_id)
        except ValueError:
            return None
        with self._lock:
            self._refresh_locked()
            record = self._records.get(normalized)
            return self._snapshot(record) if record is not None else None

    def close(self) -> None:
        with self._lock:
            self._closing = True

    def abort_generation(self, reason: str) -> None:
        del reason
        with self._lock:
            self._generation += 1
            self._records.clear()
            self._active_request_id = None
            self._failure_latched = False

    def _start_locked(
        self,
        request_id: str,
        execution: FlatExecutionRequest,
    ) -> _FlatRequestRecord:
        try:
            process = self._server.start_flat(execution)
        except FLServerProcessConflictError as error:
            raise TopLevelRequestConflictError(str(error)) from error
        except FLServerAdmissionClosedError as error:
            raise TopLevelCoordinatorUnavailableError(str(error)) from error
        record = _FlatRequestRecord(
            request_id=request_id,
            model_family_id=execution.model_family_id,
            trigger_source=execution.trigger_source,
            participant_source=execution.participant_selection.source,
            process=process,
            generation=self._generation,
        )
        self._records[request_id] = record
        self._active_request_id = request_id
        return record

    def _execution_from_intent(self, intent: RetrainIntent) -> FlatExecutionRequest:
        if self._catalog.current(intent.family_key) is None:
            raise TopLevelModelFamilyNotFoundError(
                f"model family {intent.family_key} was not found"
            )
        if self._orchestration.participant_source == "monitor_scopes":
            selection = MonitorParticipantSelection(
                participants=tuple(
                    FlatParticipantScope.from_monitor_scope(scope)
                    for scope in intent.active_scopes
                )
            )
        else:
            selection = self._static_selection(intent.family_key)
        return FlatExecutionRequest(
            model_family_id=intent.family_key,
            trigger_source=TriggerSource.DEGRADATION,
            participant_selection=selection,
            required_cutover_scope_keys=intent.active_scope_keys,
            triggering_scope_key=intent.triggering_scope_key,
        )

    def _static_selection(self, model_family_id: FamilyKey) -> StaticParticipantSelection:
        if self._planner is None:
            raise TopLevelRequestConflictError("static flat topology is unavailable")
        current = self._catalog.current(model_family_id)
        if current is None:
            raise TopLevelModelFamilyNotFoundError(
                f"model family {model_family_id} was not found"
            )
        event_filter = copy.deepcopy(current.descriptor.event_filter)
        if "networkArea" in event_filter or "aoi" in event_filter:
            raise TopLevelRequestConflictError(
                "model family filter already defines an area source"
            )
        assignment = self._planner.build(
            server_nf_instance_id=self._nwdaf_context.get().nf_instance_id
        )
        participants = []
        for client in assignment.clients:
            participant_filter = copy.deepcopy(event_filter)
            participant_filter["networkArea"] = {
                "tais": [tracking_area.wire_value() for tracking_area in client.tracking_areas]
            }
            scope_payload = {
                "family": model_family_id,
                "participant": client.nf_instance_id,
                "event": current.descriptor.event,
                "filter": participant_filter,
                "target": current.descriptor.target_ue,
                "topology": assignment.topology_digest,
            }
            scope_key = "static:" + hashlib.sha256(
                json.dumps(
                    scope_payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            participants.append(
                FlatParticipantScope(
                    scope_key=scope_key,
                    participant_nf_instance_id=client.nf_instance_id,
                    model_ids=(current.model_id,),
                    ml_event=current.descriptor.event,
                    ml_event_filter=participant_filter,
                    target_ue=(
                        copy.deepcopy(current.descriptor.target_ue)
                        if current.descriptor.target_ue is not None
                        else None
                    ),
                )
            )
        return StaticParticipantSelection(
            participants=tuple(participants),
            topology_digest=assignment.topology_digest,
        )

    def _refresh_locked(self) -> None:
        now = self._clock()
        for record in self._records.values():
            if record.generation != self._generation:
                continue
            if record.process.state in _TERMINAL_STATES and record.terminal_at is None:
                record.terminal_at = now
                if self._active_request_id == record.request_id:
                    self._active_request_id = None
                if record.process.state is FLServerState.FAILED:
                    self._failure_latched = True
        expired = [
            request_id
            for request_id, record in self._records.items()
            if record.terminal_at is not None
            and now - record.terminal_at >= self._terminal_status_ttl_seconds
        ]
        for request_id in expired:
            self._records.pop(request_id, None)

    @staticmethod
    def _snapshot(record: _FlatRequestRecord) -> FlatRequestSnapshot:
        process = record.process
        return FlatRequestSnapshot(
            request_id=record.request_id,
            model_family_id=record.model_family_id,
            state=process.state.value,
            mode="flat",
            participant_source=record.participant_source.value,
            trigger_source=record.trigger_source.value,
            current_round=process.current_round,
            completed_rounds=process.completed_rounds,
            candidate_digest=(
                process.candidate_artifact.key
                if process.candidate_artifact is not None
                else ""
            ),
            failure_cause="TRAINING_FAILED" if process.state is FLServerState.FAILED else "",
            failure_detail=(
                "federated training failed" if process.state is FLServerState.FAILED else ""
            ),
        )


def _uuid4_identity(value: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("request_id must be a canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("request_id must be a canonical UUIDv4")
    return value


def _required_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical non-empty value")
    return value
