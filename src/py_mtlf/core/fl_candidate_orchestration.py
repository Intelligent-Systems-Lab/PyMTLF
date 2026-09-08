from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from py_mtlf.core.fl_hierarchy import normalize_nf_instance_id
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryScope,
    HierarchyDiscoverySnapshot,
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.wire.ml_model_training import (
    FlPolicy,
    FlReportAfter,
    FlStrategy,
    FlTopologyNode,
    FlTopologyReport,
    FlTopologyReportNode,
    InvalidParameter,
    RequirementsError,
)


class LocalExecutionRole(StrEnum):
    LEAF = "LEAF"
    INTERMEDIATE = "INTERMEDIATE"


class CandidateProvenance(StrEnum):
    UPSTREAM_ASSIGNED = "UPSTREAM_ASSIGNED"
    LOCALLY_DISCOVERED = "LOCALLY_DISCOVERED"


class CandidateRelationshipStatus(StrEnum):
    UNCONFIRMED = "UNCONFIRMED"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    INACTIVE = "INACTIVE"


class CandidateStatusCause(StrEnum):
    RESPONSE_TIMEOUT = "RESPONSE_TIMEOUT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    FEATURE_NOT_SUPPORTED = "FEATURE_NOT_SUPPORTED"
    REQUIREMENTS_NOT_MET = "REQUIREMENTS_NOT_MET"
    REMOVED_BY_POLICY = "REMOVED_BY_POLICY"
    OTHER = "OTHER"


@dataclass(frozen=True)
class EffectivePolicy:
    allow_additional_candidates: bool
    additional_candidate_priority: int
    selection_method: str
    minimum_available_nodes: int
    fraction_train: float
    minimum_train_nodes: int
    accept_failures: bool
    minimum_completion_rate: float

    def as_wire(self) -> FlPolicy:
        return FlPolicy(
            allowAdditionalCandidates=self.allow_additional_candidates,
            additionalCandidatePriority=self.additional_candidate_priority,
            selectionMethod=self.selection_method,
            minAvailableNodes=self.minimum_available_nodes,
            fractionTrain=self.fraction_train,
            minTrainNodes=self.minimum_train_nodes,
            acceptFailures=self.accept_failures,
            minCompletionRate=self.minimum_completion_rate,
        )


@dataclass(frozen=True)
class EffectiveStrategy:
    method: str
    aggregation: str
    proximal_mu: float

    def as_wire(self) -> FlStrategy:
        return FlStrategy(
            method=self.method,
            aggregation=self.aggregation,
            methodParameters={"proximalMu": self.proximal_mu},
        )


@dataclass(frozen=True)
class EffectiveReportAfter:
    count: int
    unit: str

    def as_wire(self) -> FlReportAfter:
        return FlReportAfter(count=self.count, unit=self.unit)


@dataclass(frozen=True)
class ClientLocalWork:
    epochs: int
    proximal_mu: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.epochs, int)
            or isinstance(self.epochs, bool)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(self.proximal_mu) or self.proximal_mu < 0:
            raise ValueError("proximal_mu must be finite and non-negative")


@dataclass(frozen=True)
class IntermediateLocalWork:
    lower_round_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lower_round_count, int)
            or isinstance(self.lower_round_count, bool)
            or self.lower_round_count <= 0
        ):
            raise ValueError("lower_round_count must be a positive integer")


@dataclass(frozen=True)
class LocalContractDefaults:
    policy: FlPolicy | None = None
    strategy: FlStrategy | None = None
    report_after: FlReportAfter | None = None


@dataclass(frozen=True)
class EffectiveNodeContract:
    nf_instance_id: str
    policy: EffectivePolicy
    strategy: EffectiveStrategy
    report_after: EffectiveReportAfter
    explicit_children: tuple[FlTopologyNode, ...]


@dataclass
class CandidateRecord:
    nf_instance_id: str
    provenance: set[CandidateProvenance] = field(default_factory=set)
    upstream_instruction: FlTopologyNode | None = None
    resolved: ResolvedHierarchyNode | None = None
    enabled: bool = True
    priority: int = 0
    status: CandidateRelationshipStatus = CandidateRelationshipStatus.UNCONFIRMED
    status_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status_cause: str | None = None
    resource_location: str = ""
    child_report: FlTopologyReport | None = None
    discovery_scope: HierarchyDiscoveryScope | None = None
    last_seen_at: datetime | None = None
    discovery_revision: int | None = None
    revision: int = 1


@dataclass(frozen=True)
class EstablishmentIntent:
    nf_instance_id: str
    revision: int
    role: HierarchyNodeRole
    target: ResolvedHierarchyNode | None
    instruction: FlTopologyNode | None


@dataclass(frozen=True)
class DeleteIntent:
    nf_instance_id: str
    revision: int
    resource_location: str


@dataclass(frozen=True)
class CandidateReconciliation:
    establishment_intents: tuple[EstablishmentIntent, ...] = ()
    delete_intents: tuple[DeleteIntent, ...] = ()


@dataclass(frozen=True)
class DiscoveryRefreshIntent:
    scope: HierarchyDiscoveryScope
    revision: int


@dataclass(frozen=True)
class RoundSelection:
    participant_nf_instance_ids: tuple[str, ...]
    pool_revision: int


@dataclass(frozen=True)
class CompletionDecision:
    accepted: bool
    successful_nf_instance_ids: tuple[str, ...]
    failed_nf_instance_ids: tuple[str, ...]


def client_local_work(contract: EffectiveNodeContract) -> ClientLocalWork:
    if contract.report_after.unit != "epoch":
        raise ValueError("client local work requires reportAfter unit epoch")
    return ClientLocalWork(
        epochs=contract.report_after.count,
        proximal_mu=contract.strategy.proximal_mu,
    )


def intermediate_local_work(contract: EffectiveNodeContract) -> IntermediateLocalWork:
    if contract.report_after.unit != "round":
        raise ValueError("intermediate local work requires reportAfter unit round")
    return IntermediateLocalWork(lower_round_count=contract.report_after.count)


def resolve_effective_node_contract(
    node: FlTopologyNode,
    *,
    role: LocalExecutionRole,
    defaults: LocalContractDefaults,
) -> EffectiveNodeContract:
    violations: list[InvalidParameter] = []
    wire_policy = node.policy
    default_policy = defaults.policy

    def policy_value(name: str):
        value = getattr(wire_policy, name) if wire_policy is not None else None
        if value is None and default_policy is not None:
            value = getattr(default_policy, name)
        return value

    selection_method = policy_value("selection_method")
    minimum_available = policy_value("minimum_available_nodes")
    fraction_train = policy_value("fraction_train")
    minimum_train = policy_value("minimum_train_nodes")
    accept_failures = policy_value("accept_failures")
    minimum_completion = policy_value("minimum_completion_rate")
    required = (
        ("policy.selectionMethod", selection_method),
        ("policy.minAvailableNodes", minimum_available),
        ("policy.fractionTrain", fraction_train),
        ("policy.minTrainNodes", minimum_train),
        ("policy.acceptFailures", accept_failures),
    )
    violations.extend(
        InvalidParameter(path, "has no protocol value or local default")
        for path, value in required
        if value is None
    )
    if selection_method not in {None, "priority", "random"}:
        violations.append(
            InvalidParameter("policy.selectionMethod", "has no supported local executor")
        )
    if (
        minimum_available is not None
        and minimum_train is not None
        and minimum_available < minimum_train
    ):
        violations.append(
            InvalidParameter("policy.minAvailableNodes", "must be >= minTrainNodes")
        )
    if accept_failures is True and minimum_completion is None:
        violations.append(
            InvalidParameter(
                "policy.minCompletionRate", "is required when acceptFailures is true"
            )
        )
    if minimum_completion is None:
        minimum_completion = 1.0

    strategy = node.strategy or defaults.strategy
    if strategy is None:
        violations.append(InvalidParameter("strategy", "has no protocol value or local default"))
    else:
        if strategy.method != "fedProx":
            violations.append(InvalidParameter("strategy.method", "has no supported executor"))
        if strategy.aggregation != "sampleWeighted":
            violations.append(
                InvalidParameter("strategy.aggregation", "has no supported executor")
            )

    report_after = node.report_after or defaults.report_after
    if report_after is None:
        violations.append(
            InvalidParameter("reportAfter", "has no protocol value or local default")
        )
    else:
        expected_unit = "epoch" if role is LocalExecutionRole.LEAF else "round"
        if report_after.unit != expected_unit:
            violations.append(
                InvalidParameter(
                    "reportAfter.unit",
                    f"must be {expected_unit} for the receiving local role",
                )
            )

    if violations:
        raise RequirementsError(violations)
    assert selection_method is not None
    assert minimum_available is not None
    assert fraction_train is not None
    assert minimum_train is not None
    assert accept_failures is not None
    assert strategy is not None
    assert report_after is not None
    return EffectiveNodeContract(
        nf_instance_id=normalize_nf_instance_id(node.nf_instance_id),
        policy=EffectivePolicy(
            allow_additional_candidates=(
                wire_policy.allow_additional_candidates is True
                if wire_policy is not None
                else False
            ),
            additional_candidate_priority=(
                wire_policy.additional_candidate_priority
                if wire_policy is not None
                and wire_policy.additional_candidate_priority is not None
                else 0
            ),
            selection_method=selection_method,
            minimum_available_nodes=minimum_available,
            fraction_train=fraction_train,
            minimum_train_nodes=minimum_train,
            accept_failures=accept_failures,
            minimum_completion_rate=minimum_completion,
        ),
        strategy=EffectiveStrategy(
            method=strategy.method,
            aggregation=strategy.aggregation,
            proximal_mu=strategy.method_parameters.proximal_mu,
        ),
        report_after=EffectiveReportAfter(
            count=report_after.count,
            unit=report_after.unit,
        ),
        explicit_children=tuple(
            child.model_copy(deep=True) for child in node.children or ()
        ),
    )


class CandidatePool:
    def __init__(
        self,
        contract: EffectiveNodeContract,
        *,
        random_source: random.Random | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._contract = contract
        self._random = random_source or random.Random()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._records: dict[str, CandidateRecord] = {}
        self._revision = 0
        self._discovery_refresh_revision = 0
        self._pending_discovery_refresh: DiscoveryRefreshIntent | None = None
        self._discovery_snapshot: HierarchyDiscoverySnapshot | None = None
        self._applied_discovery_revision: int | None = None
        self.reconcile(contract.explicit_children)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def contract(self) -> EffectiveNodeContract:
        with self._lock:
            return self._contract

    def records(self) -> tuple[CandidateRecord, ...]:
        with self._lock:
            return tuple(self._copy_record(self._records[key]) for key in sorted(self._records))

    def reconcile(self, explicit_children: Iterable[FlTopologyNode]) -> CandidateReconciliation:
        children = tuple(explicit_children)
        with self._lock:
            return self._reconcile_locked(children)

    def reconfigure(self, contract: EffectiveNodeContract) -> CandidateReconciliation:
        if normalize_nf_instance_id(contract.nf_instance_id) != self._contract.nf_instance_id:
            raise ValueError("candidate pool contract identity cannot change")
        with self._lock:
            authority_changed = (
                contract.policy.allow_additional_candidates
                != self._contract.policy.allow_additional_candidates
            )
            self._contract = contract
            if authority_changed:
                self._discovery_refresh_revision += 1
                self._pending_discovery_refresh = None
                self._discovery_snapshot = None
                self._applied_discovery_revision = None
            return self._reconcile_locked(contract.explicit_children)

    def request_discovery_refresh(
        self,
        scope: HierarchyDiscoveryScope,
    ) -> DiscoveryRefreshIntent | None:
        if not self._contract.policy.allow_additional_candidates:
            raise RequirementsError(
                [InvalidParameter("policy.allowAdditionalCandidates", "is false")]
            )
        if (
            scope.containing_nf_instance_id != self._contract.nf_instance_id
            or scope.role is not HierarchyNodeRole.LEAF
            or scope.target_nf_instance_id is not None
            or not scope.tracking_areas
        ):
            raise ValueError("discovery scope does not match the local candidate pool")
        with self._lock:
            if self._snapshot_is_fresh_locked(scope):
                return None
            if (
                self._pending_discovery_refresh is not None
                and self._pending_discovery_refresh.scope == scope
            ):
                return self._pending_discovery_refresh
            self._discovery_refresh_revision += 1
            intent = DiscoveryRefreshIntent(
                scope=scope,
                revision=self._discovery_refresh_revision,
            )
            self._pending_discovery_refresh = intent
            return intent

    def complete_discovery_refresh(
        self,
        intent: DiscoveryRefreshIntent,
        snapshot: HierarchyDiscoverySnapshot,
    ) -> bool:
        with self._lock:
            if (
                self._pending_discovery_refresh != intent
                or intent.revision != self._discovery_refresh_revision
                or snapshot.scope != intent.scope
            ):
                return False
            self._pending_discovery_refresh = None
            self._apply_discovery_snapshot_locked(snapshot, intent.revision)
            return True

    def fail_discovery_refresh(self, intent: DiscoveryRefreshIntent) -> bool:
        with self._lock:
            if (
                self._pending_discovery_refresh != intent
                or intent.revision != self._discovery_refresh_revision
            ):
                return False
            self._pending_discovery_refresh = None
            return True

    def discovery_snapshot(self) -> HierarchyDiscoverySnapshot | None:
        with self._lock:
            return self._discovery_snapshot

    def set_resolved(self, node: ResolvedHierarchyNode) -> None:
        nf_id = normalize_nf_instance_id(node.nf_instance_id)
        with self._lock:
            record = self._required_locked(nf_id)
            if node.role is not _required_child_role(record.upstream_instruction):
                raise ValueError("resolved candidate role does not match child instruction")
            if record.resolved == node:
                return
            record.resolved = node
            record.revision += 1
            self._revision += 1

    def resolve_establishment_intent(
        self,
        intent: EstablishmentIntent,
        node: ResolvedHierarchyNode,
    ) -> EstablishmentIntent:
        nf_id = normalize_nf_instance_id(node.nf_instance_id)
        with self._lock:
            record = self._required_locked(nf_id)
            if (
                intent.nf_instance_id != nf_id
                or record.revision != intent.revision
                or record.status is not CandidateRelationshipStatus.DEPLOYING
            ):
                raise RuntimeError("candidate establishment intent became stale")
            if node.role is not intent.role:
                raise ValueError("resolved candidate role does not match establishment intent")
            record.resolved = node
            return EstablishmentIntent(
                nf_instance_id=intent.nf_instance_id,
                revision=intent.revision,
                role=intent.role,
                target=node,
                instruction=(
                    intent.instruction.model_copy(deep=True)
                    if intent.instruction is not None
                    else None
                ),
            )

    def fail_relationship(
        self,
        nf_instance_id: str,
        *,
        cause: str | CandidateStatusCause,
    ) -> DeleteIntent | None:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        cause_value = cause.value if isinstance(cause, CandidateStatusCause) else cause
        with self._lock:
            record = self._required_locked(nf_id)
            resource_location = record.resource_location
            self._transition_locked(
                record,
                CandidateRelationshipStatus.FAILED,
                cause_value,
            )
            record.revision += 1
            self._revision += 1
            if not resource_location:
                return None
            return DeleteIntent(
                nf_instance_id=nf_id,
                revision=record.revision,
                resource_location=resource_location,
            )

    def next_establishment_intents(self, capacity: int) -> tuple[EstablishmentIntent, ...]:
        if capacity <= 0:
            return ()
        with self._lock:
            eligible = [
                record
                for record in self._records.values()
                if record.enabled
                and record.status is CandidateRelationshipStatus.UNCONFIRMED
                and self._candidate_can_establish_locked(record)
            ]
            ordered = self._ordered(eligible)
            intents = []
            for record in ordered[:capacity]:
                self._transition_locked(record, CandidateRelationshipStatus.DEPLOYING)
                record.revision += 1
                intents.append(
                    EstablishmentIntent(
                        nf_instance_id=record.nf_instance_id,
                        revision=record.revision,
                        role=_required_child_role(record.upstream_instruction),
                        target=self._fresh_resolved_target_locked(record),
                        instruction=(
                            record.upstream_instruction.model_copy(deep=True)
                            if record.upstream_instruction is not None
                            else None
                        ),
                    )
                )
            if intents:
                self._revision += 1
            return tuple(intents)

    def complete_establishment(
        self,
        nf_instance_id: str,
        revision: int,
        *,
        resource_location: str = "",
        failure_cause: str | CandidateStatusCause | None = None,
    ) -> bool:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        with self._lock:
            record = self._required_locked(nf_id)
            if (
                record.revision != revision
                or record.status is not CandidateRelationshipStatus.DEPLOYING
            ):
                return False
            if failure_cause is None:
                record.resource_location = resource_location
                self._transition_locked(record, CandidateRelationshipStatus.ACTIVE)
            else:
                self._transition_locked(
                    record,
                    CandidateRelationshipStatus.FAILED,
                    str(failure_cause),
                )
            record.revision += 1
            self._revision += 1
            return True

    def complete_delete(
        self,
        nf_instance_id: str,
        revision: int,
        *,
        cause: str | CandidateStatusCause = CandidateStatusCause.REMOVED_BY_POLICY,
    ) -> bool:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        with self._lock:
            record = self._required_locked(nf_id)
            if (
                record.revision != revision
                or record.status is not CandidateRelationshipStatus.INACTIVE
            ):
                return False
            cause_value = cause.value if isinstance(cause, CandidateStatusCause) else cause
            self._transition_locked(record, CandidateRelationshipStatus.INACTIVE, cause_value)
            if cause_value not in {
                CandidateStatusCause.RESPONSE_TIMEOUT.value,
                CandidateStatusCause.COMMUNICATION_FAILURE.value,
            }:
                record.resource_location = ""
            record.revision += 1
            self._revision += 1
            return True

    def deactivate(
        self,
        nf_instance_id: str,
        *,
        cause: str | CandidateStatusCause = CandidateStatusCause.OTHER,
    ) -> DeleteIntent | None:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        cause_value = cause.value if isinstance(cause, CandidateStatusCause) else cause
        with self._lock:
            record = self._required_locked(nf_id)
            record.enabled = False
            intent = self._deactivate_locked(record, cause=cause_value)
            if intent is not None:
                self._revision += 1
            return intent

    def pending_delete_intents(self) -> tuple[DeleteIntent, ...]:
        with self._lock:
            return tuple(
                DeleteIntent(
                    nf_instance_id=record.nf_instance_id,
                    revision=record.revision,
                    resource_location=record.resource_location,
                )
                for record in sorted(
                    self._records.values(),
                    key=lambda value: value.nf_instance_id,
                )
                if record.status is CandidateRelationshipStatus.INACTIVE
                and bool(record.resource_location)
            )

    def ready(self) -> bool:
        with self._lock:
            active = self._active_locked()
            return (
                len(active) >= self._contract.policy.minimum_available_nodes
                and len(active) >= self._contract.policy.minimum_train_nodes
            )

    def select_round(self) -> RoundSelection:
        with self._lock:
            active = self._active_locked()
            if (
                len(active) < self._contract.policy.minimum_available_nodes
                or len(active) < self._contract.policy.minimum_train_nodes
            ):
                raise RuntimeError("candidate pool is not ready for a training round")
            count = min(
                len(active),
                max(
                    math.floor(len(active) * self._contract.policy.fraction_train),
                    self._contract.policy.minimum_train_nodes,
                ),
            )
            selected = self._ordered(active)[:count]
            return RoundSelection(
                participant_nf_instance_ids=tuple(
                    sorted(record.nf_instance_id for record in selected)
                ),
                pool_revision=self._revision,
            )

    def evaluate_completion(
        self,
        selection: RoundSelection,
        successful_nf_instance_ids: Iterable[str],
    ) -> CompletionDecision:
        selected = set(selection.participant_nf_instance_ids)
        successful = {
            normalized
            for value in successful_nf_instance_ids
            if (normalized := normalize_nf_instance_id(value)) in selected
        }
        with self._lock:
            failed = selected - successful
            if not selected:
                accepted = False
            elif not self._contract.policy.accept_failures:
                accepted = not failed
            else:
                accepted = len(successful) / len(selected) >= (
                    self._contract.policy.minimum_completion_rate
                )
            return CompletionDecision(
                accepted=accepted,
                successful_nf_instance_ids=tuple(sorted(successful)),
                failed_nf_instance_ids=tuple(sorted(failed)),
            )

    def attach_child_report(
        self,
        nf_instance_id: str,
        report: FlTopologyReport,
    ) -> None:
        nf_id = normalize_nf_instance_id(nf_instance_id)
        if normalize_nf_instance_id(report.nf_instance_id) != nf_id:
            raise ValueError("child report identity does not match candidate")
        with self._lock:
            record = self._required_locked(nf_id)
            record.child_report = report.model_copy(deep=True)
            record.revision += 1
            self._revision += 1

    def snapshot(self) -> FlTopologyReport:
        with self._lock:
            children = []
            for nf_id in sorted(self._records):
                record = self._records[nf_id]
                nested = (
                    tuple(record.child_report.children or ())
                    if record.child_report is not None
                    else ()
                )
                value = {
                    "nfInstanceId": nf_id,
                    "status": record.status.value,
                    "statusTimestamp": _timestamp(record.status_timestamp),
                }
                if record.status_cause is not None:
                    value["statusCause"] = record.status_cause
                if record.child_report is not None:
                    if record.child_report.policy is not None:
                        value["policy"] = record.child_report.policy
                    if record.child_report.strategy is not None:
                        value["strategy"] = record.child_report.strategy
                    if record.child_report.report_after is not None:
                        value["reportAfter"] = record.child_report.report_after
                if nested:
                    value["children"] = list(nested)
                children.append(FlTopologyReportNode.model_validate(value))
            return FlTopologyReport(
                nfInstanceId=self._contract.nf_instance_id,
                policy=self._contract.policy.as_wire(),
                strategy=self._contract.strategy.as_wire(),
                reportAfter=self._contract.report_after.as_wire(),
                children=children or None,
            )

    def _reconcile_locked(
        self,
        children: tuple[FlTopologyNode, ...],
    ) -> CandidateReconciliation:
        current_ids = {normalize_nf_instance_id(child.nf_instance_id) for child in children}
        deletes: list[DeleteIntent] = []
        for record in self._records.values():
            if CandidateProvenance.UPSTREAM_ASSIGNED not in record.provenance:
                continue
            if record.nf_instance_id in current_ids:
                continue
            record.provenance.discard(CandidateProvenance.UPSTREAM_ASSIGNED)
            record.upstream_instruction = None
            delete = None
            if (
                CandidateProvenance.LOCALLY_DISCOVERED in record.provenance
                and self._contract.policy.allow_additional_candidates
            ):
                record.enabled = True
                record.priority = self._contract.policy.additional_candidate_priority
            else:
                record.enabled = False
                delete = self._deactivate_locked(record)
                if delete is not None:
                    deletes.append(delete)
            if delete is None:
                record.revision += 1
        for child in children:
            nf_id = normalize_nf_instance_id(child.nf_instance_id)
            record = self._records.get(nf_id)
            if record is None:
                record = CandidateRecord(nf_instance_id=nf_id, status_timestamp=self._clock())
                self._records[nf_id] = record
            instruction = child.model_copy(deep=True)
            instruction_changed = record.upstream_instruction != instruction
            provenance_changed = CandidateProvenance.UPSTREAM_ASSIGNED not in record.provenance
            record.provenance.add(CandidateProvenance.UPSTREAM_ASSIGNED)
            record.upstream_instruction = instruction
            was_enabled = record.enabled
            record.enabled = child.enabled is not False
            record.priority = child.priority or 0
            delete = None
            if not record.enabled:
                delete = self._deactivate_locked(record)
                if delete is not None:
                    deletes.append(delete)
            elif not was_enabled and record.status is CandidateRelationshipStatus.INACTIVE:
                self._transition_locked(record, CandidateRelationshipStatus.UNCONFIRMED)
            if delete is None and (instruction_changed or provenance_changed):
                record.revision += 1
        if not self._contract.policy.allow_additional_candidates:
            for record in self._records.values():
                if record.provenance != {CandidateProvenance.LOCALLY_DISCOVERED}:
                    continue
                record.enabled = False
                delete = self._deactivate_locked(record)
                if delete is not None:
                    deletes.append(delete)
        self._revision += 1
        return CandidateReconciliation(delete_intents=tuple(deletes))

    def _apply_discovery_snapshot_locked(
        self,
        snapshot: HierarchyDiscoverySnapshot,
        discovery_revision: int,
    ) -> None:
        seen: set[str] = set()
        for node in sorted(
            snapshot.nodes,
            key=lambda value: (
                value.nf_instance_id,
                value.target.nf_service_instance_id,
            ),
        ):
            if node.role is not HierarchyNodeRole.LEAF:
                continue
            nf_id = normalize_nf_instance_id(node.nf_instance_id)
            if nf_id == self._contract.nf_instance_id:
                continue
            record = self._records.get(nf_id)
            if (
                record is not None
                and record.upstream_instruction is not None
                and record.upstream_instruction.enabled is False
            ):
                continue
            if (
                record is not None
                and record.upstream_instruction is not None
                and _required_child_role(record.upstream_instruction)
                is HierarchyNodeRole.BRANCH
            ):
                continue
            seen.add(nf_id)
            if record is None:
                record = CandidateRecord(
                    nf_instance_id=nf_id,
                    status_timestamp=self._clock(),
                )
                self._records[nf_id] = record
            relationship_changed = (
                CandidateProvenance.LOCALLY_DISCOVERED not in record.provenance
                or not _same_resolved_target(record.resolved, node)
            )
            record.provenance.add(CandidateProvenance.LOCALLY_DISCOVERED)
            record.resolved = node
            record.discovery_scope = snapshot.scope
            record.last_seen_at = snapshot.observed_at
            record.discovery_revision = discovery_revision
            if CandidateProvenance.UPSTREAM_ASSIGNED not in record.provenance:
                if (
                    not record.enabled
                    or record.priority
                    != self._contract.policy.additional_candidate_priority
                ):
                    relationship_changed = True
                record.enabled = True
                record.priority = self._contract.policy.additional_candidate_priority
            if relationship_changed:
                record.revision += 1

        if snapshot.is_complete:
            for nf_id, record in tuple(self._records.items()):
                if (
                    CandidateProvenance.LOCALLY_DISCOVERED not in record.provenance
                    or record.status is not CandidateRelationshipStatus.UNCONFIRMED
                    or nf_id in seen
                ):
                    continue
                record.provenance.discard(CandidateProvenance.LOCALLY_DISCOVERED)
                record.discovery_scope = None
                record.last_seen_at = None
                record.discovery_revision = None
                if (
                    record.resolved is not None
                    and record.resolved.discovery_scope is not None
                ):
                    record.resolved = None
                if not record.provenance:
                    del self._records[nf_id]
                else:
                    record.revision += 1

        self._discovery_snapshot = snapshot
        self._applied_discovery_revision = discovery_revision
        self._revision += 1

    def _snapshot_is_fresh_locked(self, scope: HierarchyDiscoveryScope) -> bool:
        snapshot = self._discovery_snapshot
        return (
            snapshot is not None
            and snapshot.scope == scope
            and self._clock() < snapshot.valid_until
        )

    def _candidate_can_establish_locked(self, record: CandidateRecord) -> bool:
        if CandidateProvenance.UPSTREAM_ASSIGNED in record.provenance:
            return True
        snapshot = self._discovery_snapshot
        return (
            CandidateProvenance.LOCALLY_DISCOVERED in record.provenance
            and snapshot is not None
            and record.discovery_scope == snapshot.scope
            and record.discovery_revision == self._applied_discovery_revision
            and self._clock() < snapshot.valid_until
        )

    def _fresh_resolved_target_locked(
        self,
        record: CandidateRecord,
    ) -> ResolvedHierarchyNode | None:
        resolved = record.resolved
        if (
            resolved is not None
            and resolved.valid_until is not None
            and self._clock() >= resolved.valid_until
        ):
            return None
        return resolved

    def _active_locked(self) -> list[CandidateRecord]:
        return [
            record
            for record in self._records.values()
            if record.enabled and record.status is CandidateRelationshipStatus.ACTIVE
        ]

    def _ordered(self, records: list[CandidateRecord]) -> list[CandidateRecord]:
        values = list(records)
        if self._contract.policy.selection_method == "priority":
            values.sort(
                key=lambda record: (-record.priority, record.nf_instance_id)
            )
        else:
            self._random.shuffle(values)
        return values

    def _deactivate_locked(
        self,
        record: CandidateRecord,
        *,
        cause: str = CandidateStatusCause.REMOVED_BY_POLICY.value,
    ) -> DeleteIntent | None:
        if record.status not in {
            CandidateRelationshipStatus.DEPLOYING,
            CandidateRelationshipStatus.ACTIVE,
        }:
            return None
        self._transition_locked(
            record,
            CandidateRelationshipStatus.INACTIVE,
            cause,
        )
        record.revision += 1
        return DeleteIntent(
            nf_instance_id=record.nf_instance_id,
            revision=record.revision,
            resource_location=record.resource_location,
        )

    def _transition_locked(
        self,
        record: CandidateRecord,
        status: CandidateRelationshipStatus,
        cause: str | None = None,
    ) -> None:
        if record.status is status and record.status_cause == cause:
            return
        record.status = status
        record.status_cause = cause
        record.status_timestamp = self._clock()

    def _required_locked(self, nf_instance_id: str) -> CandidateRecord:
        try:
            return self._records[nf_instance_id]
        except KeyError as error:
            raise KeyError(nf_instance_id) from error

    @staticmethod
    def _copy_record(record: CandidateRecord) -> CandidateRecord:
        return CandidateRecord(
            nf_instance_id=record.nf_instance_id,
            provenance=set(record.provenance),
            upstream_instruction=(
                record.upstream_instruction.model_copy(deep=True)
                if record.upstream_instruction is not None
                else None
            ),
            resolved=record.resolved,
            enabled=record.enabled,
            priority=record.priority,
            status=record.status,
            status_timestamp=record.status_timestamp,
            status_cause=record.status_cause,
            resource_location=record.resource_location,
            child_report=(
                record.child_report.model_copy(deep=True)
                if record.child_report is not None
                else None
            ),
            discovery_scope=record.discovery_scope,
            last_seen_at=record.last_seen_at,
            discovery_revision=record.discovery_revision,
            revision=record.revision,
        )


def _timestamp(value: datetime) -> str:
    normalized = value.astimezone(UTC).isoformat(timespec="milliseconds")
    return normalized.replace("+00:00", "Z")


def _required_child_role(
    instruction: FlTopologyNode | None,
) -> HierarchyNodeRole:
    if instruction is not None and (instruction.children or instruction.policy is not None):
        return HierarchyNodeRole.BRANCH
    return HierarchyNodeRole.LEAF


def _same_resolved_target(
    current: ResolvedHierarchyNode | None,
    updated: ResolvedHierarchyNode,
) -> bool:
    return (
        current is not None
        and current.nf_instance_id == updated.nf_instance_id
        and current.role is updated.role
        and current.target == updated.target
    )
