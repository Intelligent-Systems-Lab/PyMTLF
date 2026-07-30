import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean, pstdev

from py_mtlf.config import AccuracyPolicySettings
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog, ModelVersionKey
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorNotification,
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)


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
    family_key: FamilyKey
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


@dataclass
class FamilyAdoptionState:
    previous_version: ModelVersionKey
    current_version: ModelVersionKey
    expected_scope_keys: set[str]
    adopted_scope_keys: set[str]


class AccuracyPolicy:
    def __init__(
        self,
        settings: AccuracyPolicySettings,
        catalog: ModelCatalog,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._clock = clock
        self._lock = threading.RLock()
        self._scopes: dict[tuple[FamilyKey, str], ScopePolicyState] = {}
        self._scope_references: dict[tuple[FamilyKey, str], ScopeReference] = {}
        self._active_versions: dict[tuple[FamilyKey, str], ModelVersionKey] = {}
        self._in_flight: set[FamilyKey] = set()
        self._adoptions: dict[FamilyKey, FamilyAdoptionState] = {}
        self._intents: list[RetrainIntent] = []
        self._insufficient_reports = 0
        self._rejected_version_reports = 0

    def record_registration(self, registration: MLModelMonitorRegistration) -> bool:
        version_key = self._catalog.version_key_for_id(registration.model_id)
        family_key = self._catalog.family_for_version(version_key)
        current = self._catalog.current(family_key)
        if family_key is None or current is None or current.version_key != version_key:
            return False
        scope_key = self.registration_scope_key(registration)
        with self._lock:
            previous = self._active_versions.get((family_key, scope_key))
            self._active_versions[(family_key, scope_key)] = version_key
            self._scope_references[(family_key, scope_key)] = ScopeReference(
                scope_key=scope_key,
                consumer_id=registration.consumer_id,
                model_ids=(registration.model_id,),
                ml_event=registration.ml_event,
                ml_event_filter=dict(registration.ml_event_filter or {}),
                target_ue=(
                    dict(registration.target_ue) if registration.target_ue is not None else None
                ),
            )
            if previous != version_key:
                self._scopes.pop((family_key, scope_key), None)
            adoption = self._adoptions.get(family_key)
            if adoption is not None and adoption.current_version == version_key:
                adoption.adopted_scope_keys.add(scope_key)
        return True

    def remove_registration(self, registration: MLModelMonitorRegistration) -> None:
        version_key = self._catalog.version_key_for_id(registration.model_id)
        family_key = self._catalog.family_for_version(version_key)
        if family_key is None:
            return
        scope_key = self.registration_scope_key(registration)
        with self._lock:
            key = (family_key, scope_key)
            if self._active_versions.get(key) != version_key:
                return
            self._active_versions.pop(key, None)
            self._scope_references.pop(key, None)
            self._scopes.pop(key, None)
            adoption = self._adoptions.get(family_key)
            if adoption is not None:
                adoption.expected_scope_keys.discard(scope_key)
                adoption.adopted_scope_keys.discard(scope_key)

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
                if (
                    registration is None
                    or registration.model_id != info.model_id
                    or info.model_id not in subscription.model_ids
                ):
                    self._rejected_version_reports += 1
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                version_key = self._catalog.version_key_for_id(info.model_id)
                family_key = self._catalog.family_for_version(version_key)
                current = self._catalog.current(family_key)
                scope_key = self.scope_key(subscription, registration)
                if family_key is None or current is None or current.version_key != version_key:
                    self._rejected_version_reports += 1
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                self.record_registration(registration)
                if self._active_versions.get((family_key, scope_key)) != version_key:
                    self._rejected_version_reports += 1
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                if info.deviation is None:
                    self._insufficient_reports += 1
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                if not self._settings.enabled or family_key in self._in_flight:
                    decisions.append(PolicyDecision(evaluated=False))
                    continue
                decisions.append(
                    self._observe_one(
                        family_key,
                        scope_key,
                        float(info.deviation),
                        now,
                    )
                )
        return decisions

    def _observe_one(
        self,
        family_key: FamilyKey,
        scope_key: str,
        deviation: float,
        now: datetime,
    ) -> PolicyDecision:
        state = self._scopes.get((family_key, scope_key))
        if state is None:
            state = ScopePolicyState(
                reference=deque(maxlen=self._settings.reference_buffer_size),
                hits=deque(maxlen=self._settings.decision_window_size),
                last_update=now,
            )
            self._scopes[(family_key, scope_key)] = state
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
            and family_key not in self._in_flight
        )
        if triggered:
            active_scope_keys = tuple(
                sorted(
                    key
                    for candidate_family, key in self._active_versions
                    if candidate_family == family_key
                )
            )
            for (candidate_family, _key), candidate in self._scopes.items():
                if candidate_family == family_key:
                    candidate.hits.clear()
            self._in_flight.add(family_key)
            self._intents.append(
                RetrainIntent(
                    family_key=family_key,
                    triggering_scope_key=scope_key,
                    active_scope_keys=active_scope_keys,
                    triggering_scope=self._scope_references[(family_key, scope_key)],
                    active_scopes=tuple(
                        self._scope_references[(family_key, key)]
                        for key in active_scope_keys
                        if (family_key, key) in self._scope_references
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

    def complete_retrain(self, family_key: FamilyKey) -> None:
        with self._lock:
            self._in_flight.discard(family_key)

    def begin_generation(
        self,
        family_key: FamilyKey,
        previous_version: ModelVersionKey,
        current_version: ModelVersionKey,
        expected_scope_keys: tuple[str, ...],
    ) -> None:
        with self._lock:
            self._adoptions[family_key] = FamilyAdoptionState(
                previous_version=previous_version,
                current_version=current_version,
                expected_scope_keys=set(expected_scope_keys),
                adopted_scope_keys=set(),
            )

    def active_scope_keys(self, family_key: FamilyKey) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    scope_key
                    for candidate, scope_key in self._active_versions
                    if candidate == family_key
                )
            )

    def scope_reference(
        self,
        family_key: FamilyKey,
        scope_key: str,
    ) -> ScopeReference | None:
        with self._lock:
            return self._scope_references.get((family_key, scope_key))

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
                "insufficient_reports": self._insufficient_reports,
                "rejected_version_reports": self._rejected_version_reports,
                "adoption_count": len(self._adoptions),
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
            self._active_versions.pop(key, None)

    @staticmethod
    def registration_scope_key(registration: MLModelMonitorRegistration) -> str:
        value = {
            "consumerId": registration.consumer_id,
            "mLEvent": registration.ml_event,
            "mLEventFilter": registration.ml_event_filter or {},
            "tgtUe": registration.target_ue,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def scope_key(
        cls,
        subscription: MLModelMonitorSubscription,
        registration: MLModelMonitorRegistration | None = None,
    ) -> str:
        if registration is not None:
            return cls.registration_scope_key(registration)
        value = {
            "consumerId": "",
            "mLEvent": subscription.ml_event,
            "mLEventFilter": subscription.ml_event_filter or {},
            "tgtUe": subscription.target_ue,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
