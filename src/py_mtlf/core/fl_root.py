from __future__ import annotations

import logging
import math
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4

from py_mtlf.config import FLServerSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import RoundLocalResultType
from py_mtlf.core.fl_candidate_orchestration import (
    CandidatePool,
    CandidateStatusCause,
    LocalContractDefaults,
    LocalExecutionRole,
    resolve_effective_node_contract,
)
from py_mtlf.core.fl_experiment import (
    ExperimentAdmissionClosedError,
    ExperimentConflictError,
    ExperimentLifecycle,
    FLExperimentRegistry,
)
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    HierarchyNodeResolver,
    HierarchyNodeRole,
)
from py_mtlf.core.fl_orchestration import (
    TopLevelCoordinatorError,
    TopLevelCoordinatorUnavailableError,
    TopLevelModelFamilyNotFoundError,
    TopLevelRequestConflictError,
)
from py_mtlf.core.fl_round_model_distribution import RoundModelDistribution
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLProcess,
    FLServerEngine,
    FLServerState,
    HierarchyParticipantPreparationOutcome,
    HierarchyPreparationCollection,
    ProtocolPreparationTarget,
)
from py_mtlf.core.fl_topology import TopologyBranchGroupAssignment, TopologyPlanner
from py_mtlf.core.fl_workspace import FLWorkspace
from py_mtlf.core.nwdaf_context import FLCapabilityType, NwdafContextClient
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog
from py_mtlf.core.trainer import TrustedBundleLoader
from py_mtlf.wire.ml_model import MLEventNotification
from py_mtlf.wire.ml_model_training import (
    FlPolicy,
    FlReportAfter,
    FlStrategy,
    FlTopologyNode,
    FlTopologyReport,
)

logger = logging.getLogger(__name__)


class RootRequestState(StrEnum):
    ACCEPTED = "ACCEPTED"
    VALIDATING = "VALIDATING"
    DISPATCHING = "DISPATCHING"
    PREPARATION_WAITING = "PREPARATION_WAITING"
    PREPARATION_EVALUATING = "PREPARATION_EVALUATING"
    ADMITTED = "ADMITTED"
    ROUND_DISPATCH = "ROUND_DISPATCH"
    ROUND_WAITING = "ROUND_WAITING"
    AGGREGATING = "AGGREGATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RootFailureCause(StrEnum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    PREPARATION_DISPATCH_FAILED = "PREPARATION_DISPATCH_FAILED"
    PREPARATION_FAILED = "PREPARATION_FAILED"
    PREPARATION_TIMEOUT = "PREPARATION_TIMEOUT"
    RESULT_VALIDATION_FAILED = "RESULT_VALIDATION_FAILED"
    ADMISSION_REJECTED = "ADMISSION_REJECTED"
    ROUND_FAILED = "ROUND_FAILED"
    SHUTDOWN = "SHUTDOWN"


class RootCoordinatorError(TopLevelCoordinatorError):
    pass


class RootRequestConflictError(TopLevelRequestConflictError, RootCoordinatorError):
    pass


class RootModelFamilyNotFoundError(TopLevelModelFamilyNotFoundError, RootCoordinatorError):
    pass


class RootCoordinatorUnavailableError(TopLevelCoordinatorUnavailableError, RootCoordinatorError):
    pass


class RootPreparationError(RuntimeError):
    def __init__(self, cause: RootFailureCause, detail: str) -> None:
        super().__init__(detail)
        self.cause = cause


@dataclass(frozen=True)
class AdmittedBranchSnapshot:
    branch_nf_instance_id: str
    prepared_leaf_nf_instance_ids: tuple[str, ...]
    upper_resource_location: str


@dataclass(frozen=True)
class RootAdmissionSnapshot:
    plan_id: str
    branches: tuple[AdmittedBranchSnapshot, ...]


@dataclass(frozen=True)
class RootBranchGroupSnapshot:
    group_index: int
    state: str
    active_branch_nf_instance_id: str = ""
    candidate_nf_instance_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RootRequestSnapshot:
    request_id: str
    plan_id: str
    model_family_id: FamilyKey
    state: RootRequestState
    mode: str = "hierarchical"
    participant_source: str = "static"
    trigger_source: str = ""
    failure_cause: str = ""
    failure_detail: str = ""
    admission: RootAdmissionSnapshot | None = None
    branch_groups: tuple[RootBranchGroupSnapshot, ...] = ()
    current_round: int | None = None
    completed_rounds: int = 0
    candidate_url: str = ""
    candidate_digest: str = ""


@dataclass(frozen=True)
class RootInitiation:
    request_id: str
    plan_id: str
    model_family_id: FamilyKey
    source: str
    active_scopes: tuple[ScopeReference, ...] = ()


class RootBranchGroupState(StrEnum):
    ACTIVE = "ACTIVE"
    BRANCH_REPLACING = "BRANCH_REPLACING"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class _RootBranchGroup:
    assignment: TopologyBranchGroupAssignment
    candidate_pool: CandidatePool
    active_branch_nf_instance_id: str = ""
    active_report: FlTopologyReport | None = None
    upper_resource_location: str = ""
    state: RootBranchGroupState = RootBranchGroupState.UNAVAILABLE
    replacement_future: Future | None = None
    replacement_error: Exception | None = None


@dataclass
class _RootRequestRecord:
    initiation: RootInitiation
    reservation_id: str
    state: RootRequestState = RootRequestState.ACCEPTED
    failure_cause: str = ""
    failure_detail: str = ""
    server_process_id: str = ""
    admission: RootAdmissionSnapshot | None = None
    branch_groups: list[_RootBranchGroup] = field(default_factory=list)
    current_round: int | None = None
    completed_rounds: int = 0
    candidate_url: str = ""
    candidate_digest: str = ""
    terminal_at: float | None = None
    terminal_deadline: float | None = None
    workspace_retained: bool = False
    generation: int = 0
    future: Future | None = field(default=None, repr=False)


class FLRootCoordinator:
    def __init__(
        self,
        *,
        server_settings: FLServerSettings,
        planner: TopologyPlanner,
        resolver: HierarchyNodeResolver,
        nwdaf_context: NwdafContextClient,
        catalog: ModelCatalog,
        artifact_service: HierarchyArtifactService,
        workspace: FLWorkspace,
        server: FLServerEngine,
        policy: AccuracyPolicy,
        experiments: FLExperimentRegistry,
        loader: TrustedBundleLoader | None = None,
        round_model_distribution: RoundModelDistribution | None = None,
        terminal_status_ttl_seconds: int = 3600,
        clock=time.monotonic,
        random_source: random.Random | None = None,
    ) -> None:
        if terminal_status_ttl_seconds <= 0:
            raise ValueError("terminal_status_ttl_seconds must be positive")
        if round_model_distribution is None:
            raise ValueError("protocol hierarchy requires round model distribution")
        self._server_settings = server_settings
        self._planner = planner
        self._resolver = resolver
        self._nwdaf_context = nwdaf_context
        self._catalog = catalog
        self._artifact_service = artifact_service
        self._workspace = workspace
        self._server = server
        self._policy = policy
        self._experiments = experiments
        self._loader = loader or TrustedBundleLoader()
        self._round_model_distribution = round_model_distribution
        self._terminal_status_ttl_seconds = terminal_status_ttl_seconds
        self._clock = clock
        self._random = random_source or random.Random()
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[str, _RootRequestRecord] = {}
        self._active_request_id: str | None = None
        self._generation = 0
        self._failure_latched = False
        self._closing = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fl-root")
        self._replacement_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fl-root-replacement",
        )

    def submit_manual(
        self,
        *,
        request_id: str,
        model_family_id: FamilyKey,
    ) -> RootRequestSnapshot:
        return self._submit(
            request_id=_uuid4_identity(request_id, "request_id"),
            model_family_id=_required_identity(model_family_id, "model_family_id"),
            source="PRIVATE_API",
            active_scopes=(),
            manual=True,
        )

    def accept_policy_intents(self) -> None:
        with self._condition:
            if self._closing:
                logger.info("Ignored degradation intent dispatch while Root coordinator is closing")
                return
            if self._active_request_id is not None:
                logger.info(
                    "Deferred degradation intent dispatch while Root request %s is active",
                    self._active_request_id,
                )
                return
            if self._failure_latched:
                logger.warning("Ignored degradation intent dispatch due to terminal failure latch")
                return
        intents = self._policy.take_intents()
        for index, intent in enumerate(intents):
            if index > 0:
                self._policy.complete_retrain(intent.family_key)
                logger.warning(
                    "Skipped concurrent degradation intent family=%s due to single-active policy",
                    intent.family_key,
                )
                continue
            try:
                self._submit_policy_intent(intent)
            except RootCoordinatorError as error:
                logger.error("Failed to accept degradation intent: %s", error)

    def get(self, request_id: str) -> RootRequestSnapshot | None:
        try:
            normalized = _uuid4_identity(request_id, "request_id")
        except ValueError:
            return None
        with self._condition:
            self._prune_terminal_records_locked()
            record = self._records.get(normalized)
            return self._snapshot(record) if record is not None else None

    def requests(self) -> tuple[RootRequestSnapshot, ...]:
        with self._condition:
            self._prune_terminal_records_locked()
            return tuple(self._snapshot(self._records[key]) for key in sorted(self._records))

    def wait_for_state(
        self,
        request_id: str,
        states: set[RootRequestState],
        *,
        timeout: float,
    ) -> RootRequestSnapshot:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._prune_terminal_records_locked()
                record = self._records.get(request_id)
                if record is None:
                    raise KeyError(request_id)
                if record.state in states:
                    return self._snapshot(record)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Root request {request_id} did not reach {states}")
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            active = (
                self._records.get(self._active_request_id)
                if self._active_request_id is not None
                else None
            )
        if active is not None and active.server_process_id:
            try:
                self._server.cancel_hierarchy_preparation(
                    active.server_process_id,
                    "Root coordinator is closing",
                )
            except (KeyError, RuntimeError):
                logger.exception(
                    "Failed to wake hierarchy Server process %s during shutdown",
                    active.server_process_id,
                )
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._replacement_executor.shutdown(wait=True, cancel_futures=False)
        if active is not None:
            self._cleanup_attempt(
                active,
                cancel_server=False,
                reason="Root coordinator is closing",
            )
            with self._condition:
                active.state = RootRequestState.FAILED
                active.failure_cause = RootFailureCause.SHUTDOWN.value
                active.failure_detail = "Root coordinator is closing"
                self._retain_terminal_record_locked(active)
                self._active_request_id = None
                self._condition.notify_all()
        with self._condition:
            retained = tuple(
                record for record in self._records.values() if record.workspace_retained
            )
            for record in retained:
                record.workspace_retained = False
        for record in retained:
            self._release_retained_workspace(record)
        self._resolver.close()

    def abort_generation(self, reason: str) -> None:
        """Discard the old containing-Go Root request without closing this coordinator."""
        with self._condition:
            self._generation += 1
            active = (
                self._records.get(self._active_request_id)
                if self._active_request_id is not None
                else None
            )
            self._active_request_id = None
            retained = tuple(
                record for record in self._records.values() if record.workspace_retained
            )
            for record in retained:
                record.workspace_retained = False
            self._records.clear()
            self._failure_latched = False
            self._condition.notify_all()
        for record in retained:
            self._release_retained_workspace(record)
        if active is None:
            return
        if active.server_process_id:
            try:
                self._server.cancel_hierarchy_preparation(
                    active.server_process_id,
                    reason,
                )
            except RuntimeError:
                logger.exception(
                    "Failed to cancel Root Server process during generation reset process_id=%s",
                    active.server_process_id,
                )
        self._cleanup_attempt(
            active,
            cancel_server=False,
            reason=reason,
        )
        active.state = RootRequestState.FAILED
        active.failure_cause = RootFailureCause.SHUTDOWN.value
        active.failure_detail = "containing NWDAF process generation changed"

    def _submit_policy_intent(self, intent: RetrainIntent) -> RootRequestSnapshot:
        return self._submit(
            request_id=str(uuid4()),
            model_family_id=intent.family_key,
            source="DEGRADATION",
            active_scopes=intent.active_scopes,
            manual=False,
        )

    def _submit(
        self,
        *,
        request_id: str,
        model_family_id: FamilyKey,
        source: str,
        active_scopes: tuple[ScopeReference, ...],
        manual: bool,
    ) -> RootRequestSnapshot:
        with self._condition:
            self._prune_terminal_records_locked()
            existing = self._records.get(request_id)
            if existing is not None:
                if existing.initiation.model_family_id != model_family_id:
                    raise RootRequestConflictError(
                        "request_id is already bound to a different model family"
                    )
                return self._snapshot(existing)
            if self._closing:
                raise RootCoordinatorUnavailableError("Root coordinator is closing")
            if not manual and self._failure_latched:
                raise RootRequestConflictError("automatic training is latched after failure")
            if self._active_request_id is not None:
                raise RootRequestConflictError("another top-level training request is active")
            if self._catalog.current(model_family_id) is None:
                raise RootModelFamilyNotFoundError(
                    f"model family {model_family_id} was not found"
                )

            plan_id = str(uuid4())
            try:
                reservation = self._experiments.reserve_root(plan_id)
            except ExperimentAdmissionClosedError as error:
                raise RootCoordinatorUnavailableError(str(error)) from error
            except ExperimentConflictError as error:
                raise RootRequestConflictError(str(error)) from error
            initiation = RootInitiation(
                request_id=request_id,
                plan_id=plan_id,
                model_family_id=model_family_id,
                source=source,
                active_scopes=active_scopes,
            )
            record = _RootRequestRecord(
                initiation=initiation,
                reservation_id=reservation.reservation_id,
                generation=self._generation,
            )
            self._records[request_id] = record
            self._active_request_id = request_id
            if manual:
                self._failure_latched = False
            try:
                record.future = self._executor.submit(self._run, record)
            except Exception as error:
                self._active_request_id = None
                self._cleanup_attempt(record, cancel_server=False, reason=str(error))
                record.state = RootRequestState.FAILED
                record.failure_cause = RootFailureCause.VALIDATION_FAILED.value
                record.failure_detail = _public_failure_detail(
                    RootFailureCause.VALIDATION_FAILED
                )
                self._failure_latched = True
                self._retain_terminal_record_locked(record)
                raise RootCoordinatorUnavailableError(
                    "Root request executor is unavailable"
                ) from error
            logger.info(
                "Accepted hierarchy Root request request_id=%s plan_id=%s source=%s family=%s",
                request_id,
                plan_id,
                source,
                model_family_id,
            )
            return self._snapshot(record)

    def _run(self, record: _RootRequestRecord) -> None:
        cause = RootFailureCause.VALIDATION_FAILED
        try:
            self._set_state(record, RootRequestState.VALIDATING)
            current = self._catalog.current(record.initiation.model_family_id)
            if current is None:
                raise RuntimeError("FL base model is no longer current")
            descriptor = current.descriptor
            if descriptor.event != "X_IMAGE_CLASSIFICATION":
                raise RuntimeError(
                    "protocol hierarchical FL requires X_IMAGE_CLASSIFICATION"
                )
            if not descriptor.model_interoperability:
                raise RuntimeError("FL base model has no model interoperability identifier")
            context = self._nwdaf_context.get()
            if not any(
                descriptor.event in capability.ml_analytics_ids
                and capability.fl_capability_type
                in {FLCapabilityType.SERVER, FLCapabilityType.SERVER_AND_CLIENT}
                for capability in context.ml_analytics_capabilities
            ):
                raise RuntimeError(
                    "containing NWDAF does not advertise the required FL Server capability"
                )
            topology = self._planner.build(root_nf_instance_id=context.nf_instance_id)
            base = self._loader.load(current.artifact)

            cause = RootFailureCause.DISCOVERY_FAILED
            branch_groups = [
                self._root_branch_group(topology, group)
                for group in topology.branch_groups
            ]
            with self._condition:
                record.branch_groups = branch_groups
                self._condition.notify_all()
            resolved_branches = []
            for group in branch_groups:
                resolved_branches.append(
                    (
                        group,
                        *self._resolve_next_branch_target(
                            group,
                            descriptor.event,
                            descriptor.model_interoperability,
                        ),
                    )
                )

            latest = self._catalog.current(record.initiation.model_family_id)
            if latest is None or latest.artifact.key != current.artifact.key:
                cause = RootFailureCause.VALIDATION_FAILED
                raise RuntimeError("FL base model changed during hierarchy validation")

            self._run_protocol_hierarchy(
                record=record,
                current=current,
                base=base,
                descriptor=descriptor,
                context=context,
                topology=topology,
                resolved_branches=tuple(resolved_branches),
            )
        except Exception as error:
            if isinstance(error, HierarchyDiscoveryError):
                cause = RootFailureCause.DISCOVERY_FAILED
            elif isinstance(error, RootCoordinatorUnavailableError):
                cause = RootFailureCause.SHUTDOWN
            elif isinstance(error, RootPreparationError):
                cause = error.cause
            else:
                cause = _protocol_failure_cause(record.state)
            logger.exception(
                "Hierarchy Root request failed request_id=%s plan_id=%s",
                record.initiation.request_id,
                record.initiation.plan_id,
            )
            self._cleanup_attempt(record, cancel_server=True, reason=str(error))
            with self._condition:
                if record.generation != self._generation:
                    self._condition.notify_all()
                    return
                if record.terminal_deadline is not None:
                    self._condition.notify_all()
                    return
                record.state = RootRequestState.FAILED
                record.failure_cause = cause.value
                record.failure_detail = _public_failure_detail(cause)
                if self._active_request_id == record.initiation.request_id:
                    self._active_request_id = None
                self._failure_latched = True
                self._retain_terminal_record_locked(record)
                self._condition.notify_all()

    def _run_protocol_hierarchy(
        self,
        *,
        record: _RootRequestRecord,
        current,
        base,
        descriptor,
        context,
        topology,
        resolved_branches,
    ) -> FLProcess:
        distribution = self._round_model_distribution
        if distribution is None:
            raise RuntimeError("protocol hierarchy has no round model distribution owner")
        self._set_state(record, RootRequestState.DISPATCHING)
        targets = tuple(target for _group, _intent, target in resolved_branches)
        process = self._server.start_protocol_preparation(
            ml_correlation_id=record.initiation.plan_id,
            reservation_id=record.reservation_id,
            ml_event=descriptor.event,
            ml_event_filter=descriptor.event_filter,
            model_interoperability=descriptor.model_interoperability,
            targets=targets,
        )
        with self._condition:
            if self._active_request_id != record.initiation.request_id:
                raise RuntimeError("Root request became stale during protocol dispatch")
            record.server_process_id = process.process_id
            record.state = RootRequestState.PREPARATION_WAITING
            self._condition.notify_all()
        collection = self._server.collect_hierarchy_preparation(process.process_id)
        self._set_state(record, RootRequestState.PREPARATION_EVALUATING)
        self._validate_preparation_collection(record, process, collection)
        for group, intent, target in resolved_branches:
            outcome = self._preparation_outcome(collection, target.participant_nf_instance_id)
            while not self._apply_group_preparation(group, intent, outcome):
                if outcome.resource_location:
                    self._server.remove_protocol_participant(
                        process.process_id,
                        outcome.participant_nf_instance_id,
                    )
                intent, target = self._resolve_next_branch_target(
                    group,
                    descriptor.event,
                    descriptor.model_interoperability,
                )
                self._server.add_protocol_preparation_targets(
                    process_id=process.process_id,
                    ml_event=descriptor.event,
                    ml_event_filter=descriptor.event_filter,
                    model_interoperability=descriptor.model_interoperability,
                    targets=(target,),
                )
                collection = self._server.collect_hierarchy_preparation(process.process_id)
                self._validate_preparation_collection(record, process, collection)
                outcome = self._preparation_outcome(
                    collection,
                    target.participant_nf_instance_id,
                )
        self._server.admit_hierarchy_preparation(process.process_id)
        with self._condition:
            if self._active_request_id != record.initiation.request_id:
                raise RuntimeError("Root request became stale during protocol admission")
            self._refresh_admission_locked(record)
            record.state = RootRequestState.ADMITTED
            self._condition.notify_all()

        source = base
        aggregate = None
        round_indicator = 0
        while record.completed_rounds < self._server_settings.round_count:
            self._wait_for_root_readiness(record, topology)
            active_groups = self._active_branch_groups(record)
            expected_subordinates = {
                group.active_branch_nf_instance_id: _active_report_children(
                    group.active_report
                )
                for group in active_groups
            }
            branch_priorities = {
                group.active_branch_nf_instance_id: _branch_priority(
                    group,
                    group.active_branch_nf_instance_id,
                )
                for group in active_groups
            }
            active_branch_ids = tuple(sorted(expected_subordinates))
            selected_count = min(
                len(active_branch_ids),
                max(
                    math.floor(
                        len(active_branch_ids) * topology.policy.fraction_train
                    ),
                    topology.policy.minimum_train_nodes,
                ),
            )
            ordered = list(active_branch_ids)
            if topology.policy.selection_method == "priority":
                ordered.sort(key=lambda value: (-branch_priorities[value], value))
            else:
                self._random.shuffle(ordered)
            selected_branch_ids = tuple(sorted(ordered[:selected_count]))
            allowed_consumers = tuple(
                sorted(
                    {
                        participant_id
                        for group in active_groups
                        if group.active_branch_nf_instance_id in selected_branch_ids
                        for participant_id in (
                            group.active_branch_nf_instance_id,
                            *_active_report_descendants(group.active_report),
                        )
                    }
                )
            )
            self._set_round_state(record, RootRequestState.ROUND_DISPATCH, round_indicator)
            round_input = self._artifact_service.publish_round_input(
                plan_id=record.initiation.plan_id,
                base=source,
                process_id=process.process_id,
                server_nf_instance_id=context.nf_instance_id,
                round_indicator=round_indicator,
                epochs=self._server_settings.client_training.epochs,
            )
            round_metadata = ArtifactMetadata(
                key=round_input.digest,
                size_bytes=round_input.path.stat().st_size,
                path=round_input.path,
                url=round_input.url,
            )
            stored = distribution.store(
                ml_correlation_id=record.initiation.plan_id,
                round_indicator=round_indicator,
                artifact=round_metadata,
                allowed_consumer_ids=allowed_consumers,
            )
            try:
                round_model = MLEventNotification(
                    event=descriptor.event,
                    modelUniqueId=stored.model_unique_id,
                    mLModelAdrf=stored.wire_reference,
                )
                outcome = self._server.execute_hierarchy_round(
                    process_id=process.process_id,
                    round_indicator=round_indicator,
                    round_input_url=round_input.url,
                    round_input_artifact=round_input,
                    round_input_model=round_model,
                    input_round_indicator=round_indicator,
                    expected_result_type=RoundLocalResultType.HIERARCHY_AGGREGATE,
                    expected_subordinates=expected_subordinates,
                    selected_participant_nf_instance_ids=selected_branch_ids,
                    accept_failures=topology.policy.accept_failures,
                    minimum_completion_rate=topology.policy.minimum_completion_rate,
                    state_observer=lambda state, current_round=round_indicator: (
                        self._observe_server_round_state(record, current_round, state)
                    ),
                )
            finally:
                distribution.cleanup(record.initiation.plan_id, round_indicator)
            self._ensure_active_generation(record)
            if len(outcome.failed_participant_nf_instance_ids) > 1:
                raise RuntimeError("multiple direct Branch failures are not recoverable")
            completed_after_attempt = record.completed_rounds + int(outcome.accepted)
            for failed_branch_id in outcome.failed_participant_nf_instance_ids:
                self._retire_failed_branch(
                    record=record,
                    process=process,
                    descriptor=descriptor,
                    failed_branch_nf_instance_id=failed_branch_id,
                    replace=(completed_after_attempt < self._server_settings.round_count),
                )
            if outcome.accepted:
                if outcome.aggregate is None:
                    raise RuntimeError("accepted Root round has no aggregate")
                aggregate = outcome.aggregate
                source = self._loader.load(
                    ArtifactMetadata(
                        key=aggregate.digest,
                        size_bytes=aggregate.path.stat().st_size,
                        path=aggregate.path,
                        url=aggregate.url,
                    )
                )
            with self._condition:
                if (
                    self._closing
                    or record.generation != self._generation
                    or self._active_request_id != record.initiation.request_id
                ):
                    raise RootCoordinatorUnavailableError("Root coordinator is closing")
                if outcome.accepted:
                    record.completed_rounds += 1
                    record.candidate_url = aggregate.url
                    record.candidate_digest = aggregate.digest
                self._refresh_admission_locked(record)
                self._condition.notify_all()
            logger.info(
                "Root hierarchy round outcome plan_id=%s round_ind=%s accepted=%s "
                "selected=%s successful=%s failed=%s completed_rounds=%s",
                record.initiation.plan_id,
                round_indicator,
                outcome.accepted,
                ",".join(outcome.selected_participant_nf_instance_ids),
                ",".join(outcome.successful_participant_nf_instance_ids),
                ",".join(outcome.failed_participant_nf_instance_ids),
                record.completed_rounds,
            )
            round_indicator += 1

        if aggregate is None:
            raise RuntimeError("protocol hierarchy completed without an aggregate")
        self._ensure_active_generation(record)
        self._complete_protocol_training(record, process.process_id)
        return process

    def _protocol_branch_node(self, group, branch) -> FlTopologyNode:
        strategy = FlStrategy.model_validate(
            group.strategy.model_dump(by_alias=True, mode="json")
        )
        policy = FlPolicy.model_validate(
            group.policy.model_dump(by_alias=True, mode="json")
        )
        children = [
            FlTopologyNode(
                nfInstanceId=leaf.nf_instance_id,
                enabled=leaf.enabled,
                priority=leaf.priority,
                strategy=strategy,
                reportAfter=(
                    FlReportAfter.model_validate(
                        leaf.report_after.model_dump(by_alias=True, mode="json")
                    )
                    if leaf.report_after is not None
                    else None
                ),
            )
            for leaf in group.leaves
        ]
        return FlTopologyNode(
            nfInstanceId=branch.nf_instance_id,
            enabled=branch.enabled,
            priority=branch.priority,
            policy=policy,
            strategy=strategy,
            reportAfter=(
                FlReportAfter.model_validate(
                    branch.report_after.model_dump(by_alias=True, mode="json")
                )
                if branch.report_after is not None
                else None
            ),
            children=children,
        )

    def _root_branch_group(self, topology, group) -> _RootBranchGroup:
        root_node = FlTopologyNode(
            nfInstanceId=topology.root_nf_instance_id,
            policy=FlPolicy.model_validate(
                topology.policy.model_dump(by_alias=True, mode="json")
            ),
            strategy=FlStrategy.model_validate(
                topology.strategy.model_dump(by_alias=True, mode="json")
            ),
            reportAfter=FlReportAfter(count=1, unit="round"),
            children=[
                self._protocol_branch_node(group, branch)
                for branch in group.branches
            ],
        )
        contract = resolve_effective_node_contract(
            root_node,
            role=LocalExecutionRole.INTERMEDIATE,
            defaults=LocalContractDefaults(),
        )
        return _RootBranchGroup(
            assignment=group,
            candidate_pool=CandidatePool(contract, random_source=self._random),
        )

    def _resolve_next_branch_target(
        self,
        group: _RootBranchGroup,
        ml_event: str,
        model_interoperability: str,
    ) -> tuple[object, ProtocolPreparationTarget]:
        while True:
            intents = group.candidate_pool.next_establishment_intents(1)
            if not intents:
                raise RootPreparationError(
                    RootFailureCause.PREPARATION_FAILED,
                    "Branch candidate pool is exhausted",
                )
            intent = intents[0]
            try:
                resolved = self._resolver.resolve(
                    nf_instance_id=intent.nf_instance_id,
                    role=HierarchyNodeRole.BRANCH,
                    ml_event=ml_event,
                    model_interoperability=model_interoperability,
                )
            except HierarchyDiscoveryError:
                group.candidate_pool.complete_establishment(
                    intent.nf_instance_id,
                    intent.revision,
                    failure_cause=CandidateStatusCause.COMMUNICATION_FAILURE,
                )
                continue
            resolved_intent = group.candidate_pool.resolve_establishment_intent(
                intent,
                resolved,
            )
            if resolved_intent.instruction is None:
                raise RuntimeError("Root Branch candidate has no topology instruction")
            return (
                resolved_intent,
                ProtocolPreparationTarget(
                    participant_nf_instance_id=resolved_intent.nf_instance_id,
                    candidate=FLClientCandidate(
                        target=resolved.target,
                        tracking_areas=(),
                    ),
                    topology=resolved_intent.instruction,
                ),
            )

    @staticmethod
    def _preparation_outcome(
        collection: HierarchyPreparationCollection,
        participant_nf_instance_id: str,
    ) -> HierarchyParticipantPreparationOutcome:
        for outcome in collection.participants:
            if outcome.participant_nf_instance_id == participant_nf_instance_id:
                if (
                    participant_nf_instance_id
                    in collection.timed_out_participant_nf_instance_ids
                    and not outcome.failure
                ):
                    return HierarchyParticipantPreparationOutcome(
                        participant_nf_instance_id=outcome.participant_nf_instance_id,
                        resource_location=outcome.resource_location,
                        notification=outcome.notification,
                        failure=CandidateStatusCause.RESPONSE_TIMEOUT.value,
                        delay_extensions=outcome.delay_extensions,
                        granted_extension_seconds=outcome.granted_extension_seconds,
                    )
                return outcome
        raise RootPreparationError(
            RootFailureCause.RESULT_VALIDATION_FAILED,
            "protocol preparation omitted the attempted Branch",
        )

    def _apply_group_preparation(
        self,
        group: _RootBranchGroup,
        intent,
        outcome: HierarchyParticipantPreparationOutcome,
    ) -> bool:
        if outcome.participant_nf_instance_id != intent.nf_instance_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "Branch preparation outcome identity does not match the attempt",
            )
        report = (
            outcome.notification.fl_topology_report
            if outcome.notification is not None
            else None
        )
        if outcome.failure or report is None:
            if not group.candidate_pool.complete_establishment(
                intent.nf_instance_id,
                intent.revision,
                resource_location=outcome.resource_location,
                failure_cause=outcome.failure or CandidateStatusCause.REQUIREMENTS_NOT_MET,
            ):
                raise RuntimeError("Branch preparation completion became stale")
            return False
        self._validate_group_report(group, intent.nf_instance_id, report)
        if not group.candidate_pool.complete_establishment(
            intent.nf_instance_id,
            intent.revision,
            resource_location=outcome.resource_location,
        ):
            raise RuntimeError("Branch preparation completion became stale")
        group.candidate_pool.attach_child_report(intent.nf_instance_id, report)
        with self._condition:
            group.active_branch_nf_instance_id = intent.nf_instance_id
            group.active_report = report.model_copy(deep=True)
            group.upper_resource_location = outcome.resource_location
            group.state = RootBranchGroupState.ACTIVE
            group.replacement_error = None
            self._condition.notify_all()
        return True

    @staticmethod
    def _validate_group_report(
        group: _RootBranchGroup,
        branch_nf_instance_id: str,
        report: FlTopologyReport,
    ) -> None:
        if report.nf_instance_id != branch_nf_instance_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "Branch topology report identity does not match",
            )
        active_children = _active_report_children(report)
        configured = {
            leaf.nf_instance_id for leaf in group.assignment.leaves if leaf.enabled
        }
        if (
            not group.assignment.policy.allow_additional_candidates
            and not set(active_children).issubset(configured)
        ):
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "Branch topology report contains an unassigned active child",
            )
        policy = group.assignment.policy
        if (
            len(active_children) < policy.minimum_available_nodes
            or len(active_children) < policy.minimum_train_nodes
        ):
            raise RootPreparationError(
                RootFailureCause.ADMISSION_REJECTED,
                "Branch topology report does not satisfy its configured policy",
            )

    def _active_branch_groups(
        self,
        record: _RootRequestRecord,
    ) -> tuple[_RootBranchGroup, ...]:
        with self._condition:
            return tuple(
                group
                for group in record.branch_groups
                if group.state is RootBranchGroupState.ACTIVE
                and group.active_branch_nf_instance_id
                and group.active_report is not None
            )

    def _wait_for_root_readiness(self, record: _RootRequestRecord, topology) -> None:
        with self._condition:
            while True:
                self._ensure_active_generation(record)
                errors = [
                    group.replacement_error
                    for group in record.branch_groups
                    if group.replacement_error is not None
                ]
                if errors:
                    raise errors[0]
                active_count = sum(
                    group.state is RootBranchGroupState.ACTIVE
                    for group in record.branch_groups
                )
                if (
                    active_count >= topology.policy.minimum_available_nodes
                    and active_count >= topology.policy.minimum_train_nodes
                ):
                    return
                pending = any(
                    group.replacement_future is not None
                    and not group.replacement_future.done()
                    for group in record.branch_groups
                )
                if not pending:
                    raise RuntimeError("Root Branch pool cannot satisfy its readiness policy")
                self._condition.wait(timeout=0.1)

    def _retire_failed_branch(
        self,
        *,
        record: _RootRequestRecord,
        process: FLProcess,
        descriptor,
        failed_branch_nf_instance_id: str,
        replace: bool,
    ) -> None:
        group = next(
            (
                item
                for item in record.branch_groups
                if item.active_branch_nf_instance_id == failed_branch_nf_instance_id
            ),
            None,
        )
        if group is None:
            raise RuntimeError("failed Branch does not own an active group")
        cleanup_failure = self._server.remove_protocol_participant(
            process.process_id,
            failed_branch_nf_instance_id,
        )
        if cleanup_failure:
            logger.warning(
                "Failed Branch remote cleanup remains pending nf=%s error=%s",
                failed_branch_nf_instance_id,
                cleanup_failure,
            )
        group.candidate_pool.fail_relationship(
            failed_branch_nf_instance_id,
            cause=CandidateStatusCause.COMMUNICATION_FAILURE,
        )
        with self._condition:
            group.active_branch_nf_instance_id = ""
            group.active_report = None
            group.upper_resource_location = ""
            group.state = (
                RootBranchGroupState.BRANCH_REPLACING
                if replace
                else RootBranchGroupState.UNAVAILABLE
            )
            self._refresh_admission_locked(record)
            self._condition.notify_all()
        if not replace:
            return
        logger.info(
            "Root Branch replacement started plan_id=%s failed_nf=%s",
            record.initiation.plan_id,
            failed_branch_nf_instance_id,
        )
        try:
            future = self._replacement_executor.submit(
                self._replace_branch_group,
                record,
                process.process_id,
                group,
                descriptor.event,
                descriptor.event_filter,
                descriptor.model_interoperability,
            )
        except RuntimeError as error:
            with self._condition:
                group.replacement_error = error
                group.state = RootBranchGroupState.UNAVAILABLE
                self._condition.notify_all()
            raise
        with self._condition:
            group.replacement_future = future
            self._condition.notify_all()

    def _replace_branch_group(
        self,
        record: _RootRequestRecord,
        process_id: str,
        group: _RootBranchGroup,
        ml_event: str,
        ml_event_filter: dict,
        model_interoperability: str,
    ) -> None:
        try:
            while True:
                self._ensure_active_generation(record)
                try:
                    intent, target = self._resolve_next_branch_target(
                        group,
                        ml_event,
                        model_interoperability,
                    )
                except RootPreparationError as error:
                    if error.cause is RootFailureCause.PREPARATION_FAILED:
                        with self._condition:
                            group.state = RootBranchGroupState.UNAVAILABLE
                            self._condition.notify_all()
                        return
                    raise
                outcome = self._server.prepare_protocol_replacement_target(
                    process_id=process_id,
                    ml_event=ml_event,
                    ml_event_filter=ml_event_filter,
                    model_interoperability=model_interoperability,
                    target=target,
                )
                try:
                    self._ensure_active_generation(record)
                except RootCoordinatorError:
                    if outcome.resource_location:
                        try:
                            self._server.remove_protocol_participant(
                                process_id,
                                outcome.participant_nf_instance_id,
                            )
                        except (KeyError, RuntimeError):
                            logger.info(
                                "Replacement resource was already removed during Root reset "
                                "nf=%s",
                                outcome.participant_nf_instance_id,
                            )
                    raise
                if self._apply_group_preparation(group, intent, outcome):
                    with self._condition:
                        self._refresh_admission_locked(record)
                        self._condition.notify_all()
                    logger.info(
                        "Root Branch replacement ready plan_id=%s replacement_nf=%s",
                        record.initiation.plan_id,
                        outcome.participant_nf_instance_id,
                    )
                    return
                if outcome.resource_location:
                    self._server.remove_protocol_participant(
                        process_id,
                        outcome.participant_nf_instance_id,
                    )
        except Exception as error:
            with self._condition:
                group.replacement_error = error
                group.state = RootBranchGroupState.UNAVAILABLE
                self._condition.notify_all()

    def _refresh_admission_locked(self, record: _RootRequestRecord) -> None:
        record.admission = RootAdmissionSnapshot(
            plan_id=record.initiation.plan_id,
            branches=tuple(
                AdmittedBranchSnapshot(
                    branch_nf_instance_id=group.active_branch_nf_instance_id,
                    prepared_leaf_nf_instance_ids=_active_report_children(
                        group.active_report
                    ),
                    upper_resource_location=group.upper_resource_location,
                )
                for group in record.branch_groups
                if group.state is RootBranchGroupState.ACTIVE
                and group.active_report is not None
            ),
        )

    def _validate_preparation_collection(
        self,
        record: _RootRequestRecord,
        process: FLProcess,
        collection: HierarchyPreparationCollection,
    ) -> None:
        if collection.plan_id != record.initiation.plan_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "protocol preparation collection mlCorreId does not match",
            )
        if collection.process_id != process.process_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "protocol preparation collection process does not match",
            )

    def _set_state(self, record: _RootRequestRecord, state: RootRequestState) -> None:
        with self._condition:
            if self._closing:
                raise RootCoordinatorUnavailableError("Root coordinator is closing")
            if self._active_request_id != record.initiation.request_id:
                raise RootRequestConflictError("Root request is stale")
            record.state = state
            self._condition.notify_all()

    def _ensure_active_generation(self, record: _RootRequestRecord) -> None:
        with self._condition:
            if self._closing:
                raise RootCoordinatorUnavailableError("Root coordinator is closing")
            if (
                record.generation != self._generation
                or self._active_request_id != record.initiation.request_id
            ):
                raise RootRequestConflictError("Root request is stale")

    def _set_round_state(
        self,
        record: _RootRequestRecord,
        state: RootRequestState,
        round_indicator: int,
    ) -> None:
        with self._condition:
            if self._closing:
                raise RootCoordinatorUnavailableError("Root coordinator is closing")
            if self._active_request_id != record.initiation.request_id:
                raise RootRequestConflictError("Root request is stale")
            record.current_round = round_indicator
            record.state = state
            self._condition.notify_all()

    def _observe_server_round_state(
        self,
        record: _RootRequestRecord,
        round_indicator: int,
        state: FLServerState,
    ) -> None:
        projected = {
            FLServerState.ROUND_DISPATCH: RootRequestState.ROUND_DISPATCH,
            FLServerState.ROUND_WAITING: RootRequestState.ROUND_WAITING,
            FLServerState.ROUND_EVALUATING: RootRequestState.ROUND_WAITING,
            FLServerState.AGGREGATING: RootRequestState.AGGREGATING,
        }.get(state)
        if projected is not None:
            self._set_round_state(record, projected, round_indicator)

    def _complete_protocol_training(
        self,
        record: _RootRequestRecord,
        process_id: str,
    ) -> None:
        self._server.close_hierarchy_training(process_id)
        self._release_reservation(record, RootRequestState.COMPLETE.value)
        with self._condition:
            if (
                self._closing
                or record.generation != self._generation
                or self._active_request_id != record.initiation.request_id
            ):
                raise RootCoordinatorUnavailableError("Root coordinator is closing")
            record.state = RootRequestState.COMPLETE
            record.workspace_retained = True
            self._active_request_id = None
            self._retain_terminal_record_locked(record)
            self._condition.notify_all()

    def _cleanup_attempt(
        self,
        record: _RootRequestRecord,
        *,
        cancel_server: bool,
        reason: str,
    ) -> None:
        active = self._experiments.active()
        process_id = record.server_process_id
        if not process_id and active is not None and active.reservation_id == record.reservation_id:
            process_id = active.server_process_id or ""
        if cancel_server and process_id:
            try:
                self._server.cancel_hierarchy_preparation(process_id, reason)
            except (KeyError, RuntimeError):
                logger.exception("Failed to clean hierarchy Server process %s", process_id)
        try:
            self._workspace.release_plan(record.initiation.plan_id)
        except RuntimeError:
            logger.exception(
                "Failed to release hierarchy workspace plan_id=%s",
                record.initiation.plan_id,
            )
        self._release_reservation(record, RootRequestState.FAILED.value)

    def _release_reservation(self, record: _RootRequestRecord, outcome: str) -> None:
        active = self._experiments.active()
        if active is None or active.reservation_id != record.reservation_id:
            return
        try:
            if active.lifecycle in {ExperimentLifecycle.PROVISIONAL, ExperimentLifecycle.ACTIVE}:
                self._experiments.mark_terminal(record.reservation_id, outcome)
                active = self._experiments.active()
            if active is not None and active.lifecycle is ExperimentLifecycle.TERMINAL:
                self._experiments.begin_cleanup(record.reservation_id)
                active = self._experiments.active()
            if active is not None and active.lifecycle is ExperimentLifecycle.CLEANING:
                self._experiments.release(record.reservation_id)
        except RuntimeError:
            logger.exception(
                "Failed to release hierarchy registry reservation request_id=%s",
                record.initiation.request_id,
            )

    def _retain_terminal_record_locked(self, record: _RootRequestRecord) -> None:
        if not record.workspace_retained:
            record.candidate_url = ""
        if record.terminal_deadline is None:
            record.terminal_at = self._clock()
            record.terminal_deadline = (
                record.terminal_at + self._terminal_status_ttl_seconds
            )

    def _prune_terminal_records_locked(self) -> None:
        now = self._clock()
        expired = tuple(
            record
            for record in self._records.values()
            if record.terminal_deadline is not None and record.terminal_deadline <= now
        )
        for record in expired:
            if record.workspace_retained:
                record.workspace_retained = False
                self._release_retained_workspace(record)
        self._records = {
            request_id: record
            for request_id, record in self._records.items()
            if record.terminal_deadline is None or record.terminal_deadline > now
        }

    def _release_retained_workspace(self, record: _RootRequestRecord) -> None:
        try:
            self._workspace.release_plan(record.initiation.plan_id)
        except RuntimeError:
            logger.exception(
                "Failed to release retained protocol hierarchy workspace plan_id=%s",
                record.initiation.plan_id,
            )

    @staticmethod
    def _snapshot(record: _RootRequestRecord) -> RootRequestSnapshot:
        return RootRequestSnapshot(
            request_id=record.initiation.request_id,
            plan_id=record.initiation.plan_id,
            model_family_id=record.initiation.model_family_id,
            state=record.state,
            trigger_source=record.initiation.source.lower(),
            failure_cause=record.failure_cause,
            failure_detail=record.failure_detail,
            admission=record.admission,
            branch_groups=tuple(
                RootBranchGroupSnapshot(
                    group_index=index,
                    state=group.state.value,
                    active_branch_nf_instance_id=(
                        group.active_branch_nf_instance_id
                    ),
                    candidate_nf_instance_ids=tuple(
                        candidate.nf_instance_id
                        for candidate in group.assignment.branches
                    ),
                )
                for index, group in enumerate(record.branch_groups)
            ),
            current_round=record.current_round,
            completed_rounds=record.completed_rounds,
            candidate_url=record.candidate_url,
            candidate_digest=record.candidate_digest,
        )


def _required_identity(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be a non-empty canonical value")
    return value


def _uuid4_identity(value: str, name: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv4")
    return value


def _active_report_children(report: FlTopologyReport | None) -> tuple[str, ...]:
    if report is None:
        return ()
    return tuple(
        sorted(
            child.nf_instance_id
            for child in report.children or ()
            if child.status == "ACTIVE"
        )
    )


def _active_report_descendants(report: FlTopologyReport | None) -> tuple[str, ...]:
    if report is None:
        return ()
    active = []
    pending = list(report.children or ())
    while pending:
        node = pending.pop()
        if node.status == "ACTIVE":
            active.append(node.nf_instance_id)
            pending.extend(node.children or ())
    return tuple(sorted(active))


def _branch_priority(group: _RootBranchGroup, nf_instance_id: str) -> int:
    for candidate in group.assignment.branches:
        if candidate.nf_instance_id == nf_instance_id:
            return candidate.priority
    raise KeyError(nf_instance_id)


def _protocol_failure_cause(state: RootRequestState) -> RootFailureCause:
    if state in {
        RootRequestState.DISPATCHING,
        RootRequestState.PREPARATION_WAITING,
    }:
        return RootFailureCause.PREPARATION_DISPATCH_FAILED
    if state is RootRequestState.PREPARATION_EVALUATING:
        return RootFailureCause.RESULT_VALIDATION_FAILED
    return RootFailureCause.ROUND_FAILED


def _public_failure_detail(cause: RootFailureCause) -> str:
    return {
        RootFailureCause.VALIDATION_FAILED: "Root hierarchy validation failed",
        RootFailureCause.DISCOVERY_FAILED: "configured hierarchy node discovery failed",
        RootFailureCause.PREPARATION_DISPATCH_FAILED: "upper-tier preparation dispatch failed",
        RootFailureCause.PREPARATION_FAILED: "one or more Branch preparations failed",
        RootFailureCause.PREPARATION_TIMEOUT: "one or more Branch preparations timed out",
        RootFailureCause.RESULT_VALIDATION_FAILED: "Branch preparation result validation failed",
        RootFailureCause.ADMISSION_REJECTED: "complete-required hierarchy admission was rejected",
        RootFailureCause.ROUND_FAILED: "hierarchical training round failed",
        RootFailureCause.SHUTDOWN: "Root coordinator is shutting down",
    }[cause]
