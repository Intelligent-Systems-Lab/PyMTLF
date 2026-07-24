import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean, pstdev

from py_mtlf.config import AccuracyPolicySettings
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorNotification,
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)

ModelKey = tuple[str, int]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ScopePolicyState:
    reference: deque[float]
    hits: deque[bool]
    last_update: datetime
    report_count: int = 0


@dataclass(frozen=True)
class RetrainIntent:
    model_key: ModelKey
    triggering_scope_key: str
    active_scope_keys: tuple[str, ...]
    triggering_scope: "ScopeReference"
    active_scopes: tuple["ScopeReference", ...]
    created_at: datetime


@dataclass(frozen=True)
class ScopeReference:
    scope_key: str
    consumer_id: str
    model_ids: tuple[int, ...]
    ml_event: str
    ml_event_filter: dict
    target_ue: dict | None


@dataclass(frozen=True)
class PolicyDecision:
    evaluated: bool
    baseline_ready: bool = False
    signal: bool = False
    hit: bool = False
    hit_count: int = 0
    triggered: bool = False
    z_score: float = 0


class AccuracyPolicy:
    def __init__(
        self,
        settings: AccuracyPolicySettings,
        provider_namespace: str,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        self._provider_namespace = provider_namespace
        self._clock = clock
        self._lock = threading.RLock()
        self._scopes: dict[tuple[ModelKey, str], ScopePolicyState] = {}
        self._scope_references: dict[tuple[ModelKey, str], ScopeReference] = {}
        self._in_flight: set[ModelKey] = set()
        self._generation: dict[ModelKey, int] = {}
        self._intents: list[RetrainIntent] = []
        self._liveness_reports = 0

    def observe(
        self,
        subscription: MLModelMonitorSubscription,
        notification: MLModelMonitorNotification,
        registration: MLModelMonitorRegistration | None = None,
    ) -> list[PolicyDecision]:
        now = self._clock()
        decisions: list[PolicyDecision] = []
        with self._lock:
            self._gc(now)
            for info in notification.model_accuracy_info:
                model_key = (self._provider_namespace, info.model_id)
                scope_key = self.scope_key(subscription, registration)
                self._scope_references[(model_key, scope_key)] = ScopeReference(
                    scope_key=scope_key,
                    consumer_id=registration.consumer_id if registration else "",
                    model_ids=tuple(subscription.model_ids),
                    ml_event=subscription.ml_event,
                    ml_event_filter=dict(subscription.ml_event_filter or {}),
                    target_ue=(
                        dict(subscription.target_ue) if subscription.target_ue is not None else None
                    ),
                )
                if info.deviation is None:
                    self._liveness_reports += 1
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                if not self._settings.enabled or model_key in self._in_flight:
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                decisions.append(
                    self._observe_one(
                        model_key,
                        scope_key,
                        float(info.deviation),
                        now,
                    )
                )
        return decisions

    def _observe_one(
        self,
        model_key: ModelKey,
        scope_key: str,
        deviation: float,
        now: datetime,
    ) -> PolicyDecision:
        state = self._scopes.get((model_key, scope_key))
        if state is None:
            state = ScopePolicyState(
                reference=deque(maxlen=self._settings.reference_buffer_size),
                hits=deque(maxlen=self._settings.decision_window_size),
                last_update=now,
            )
            self._scopes[(model_key, scope_key)] = state
        state.last_update = now
        state.report_count += 1

        baseline_ready = len(state.reference) >= self._settings.min_reference_samples
        mean = fmean(state.reference) if state.reference else 0.0
        std = pstdev(state.reference) if state.reference else 0.0
        z_score = (deviation - mean) / max(std, self._settings.min_std) if state.reference else 0.0
        signal = bool(state.reference) and z_score > self._settings.z_score_threshold
        eligible = deviation > self._settings.fixed_floor
        hit = baseline_ready and eligible and signal
        if baseline_ready:
            state.hits.append(hit)
        else:
            state.hits.clear()
        if len(state.reference) < self._settings.min_reference_samples or not signal:
            state.reference.append(deviation)

        hit_count = sum(state.hits)
        triggered = (
            baseline_ready
            and hit_count >= self._settings.required_hits
            and model_key not in self._in_flight
        )
        if triggered:
            active_scope_keys = tuple(
                sorted(
                    key for (candidate_model, key) in self._scopes if candidate_model == model_key
                )
            )
            for (candidate_model, _key), candidate in self._scopes.items():
                if candidate_model == model_key:
                    candidate.hits.clear()
            self._in_flight.add(model_key)
            self._intents.append(
                RetrainIntent(
                    model_key=model_key,
                    triggering_scope_key=scope_key,
                    active_scope_keys=active_scope_keys,
                    triggering_scope=self._scope_references[(model_key, scope_key)],
                    active_scopes=tuple(
                        self._scope_references[(model_key, key)] for key in active_scope_keys
                    ),
                    created_at=now,
                )
            )
        return PolicyDecision(
            evaluated=True,
            baseline_ready=baseline_ready,
            signal=signal,
            hit=hit,
            hit_count=hit_count,
            triggered=triggered,
            z_score=z_score,
        )

    def complete_retrain(self, model_key: ModelKey) -> None:
        with self._lock:
            self._in_flight.discard(model_key)

    def advance_generation(self, model_key: ModelKey, generation: int) -> None:
        with self._lock:
            if generation <= self._generation.get(model_key, 0):
                return
            self._generation[model_key] = generation
            self._in_flight.discard(model_key)
            for key in [key for key in self._scopes if key[0] == model_key]:
                self._scopes.pop(key, None)
                self._scope_references.pop(key, None)

    def intents(self) -> tuple[RetrainIntent, ...]:
        with self._lock:
            return tuple(self._intents)

    def take_intents(self) -> tuple[RetrainIntent, ...]:
        with self._lock:
            intents = tuple(self._intents)
            self._intents.clear()
            return intents

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "scope_count": len(self._scopes),
                "in_flight": tuple(sorted(self._in_flight)),
                "intent_count": len(self._intents),
                "liveness_reports": self._liveness_reports,
            }

    def _gc(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._settings.scope_state_ttl_seconds)
        for key in [
            key
            for key, state in self._scopes.items()
            if state.last_update < cutoff and key[0] not in self._in_flight
        ]:
            self._scopes.pop(key, None)
            self._scope_references.pop(key, None)

    @staticmethod
    def scope_key(
        subscription: MLModelMonitorSubscription,
        registration: MLModelMonitorRegistration | None = None,
    ) -> str:
        value = {
            "consumerId": registration.consumer_id if registration is not None else "",
            "modelIds": subscription.model_ids,
            "mLEvent": subscription.ml_event,
            "mLEventFilter": subscription.ml_event_filter or {},
            "tgtUe": subscription.target_ue,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
