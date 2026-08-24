from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4

from py_mtlf.config import FederatedStrategySettings, FLServerSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    HierarchyPreparationResultArtifact,
    RoundLocalResultType,
)
from py_mtlf.core.fl_experiment import (
    ExperimentAdmissionClosedError,
    ExperimentConflictError,
    ExperimentLifecycle,
    FLExperimentRegistry,
)
from py_mtlf.core.fl_hierarchy import (
    FederatedStrategy,
    FedProxAlgorithm,
    HierarchyMessageType,
    PreparationOutcome,
    PreparationResultMetadata,
)
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryError,
    HierarchyNodeResolver,
    HierarchyNodeRole,
)
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLProcess,
    FLServerEngine,
    FLServerState,
    HierarchyPreparationCollection,
    HierarchyPreparationTarget,
)
from py_mtlf.core.fl_topology import TopologyPlanner
from py_mtlf.core.fl_workspace import FLWorkspace
from py_mtlf.core.nwdaf_context import FLCapabilityType, NwdafContextClient
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog
from py_mtlf.core.trainer import TrustedBundleLoader

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
    FINAL_VALIDATION_DISPATCH = "FINAL_VALIDATION_DISPATCH"
    FINAL_VALIDATION_WAITING = "FINAL_VALIDATION_WAITING"
    FINAL_VALIDATION_EVALUATING = "FINAL_VALIDATION_EVALUATING"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    CANDIDATE_READY = "CANDIDATE_READY"
    PUBLISHING = "PUBLISHING"
    CUTOVER_PENDING = "CUTOVER_PENDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RootFailureCause(StrEnum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    ASSIGNMENT_PUBLICATION_FAILED = "ASSIGNMENT_PUBLICATION_FAILED"
    PREPARATION_DISPATCH_FAILED = "PREPARATION_DISPATCH_FAILED"
    PREPARATION_FAILED = "PREPARATION_FAILED"
    PREPARATION_TIMEOUT = "PREPARATION_TIMEOUT"
    RESULT_VALIDATION_FAILED = "RESULT_VALIDATION_FAILED"
    ADMISSION_REJECTED = "ADMISSION_REJECTED"
    ROUND_FAILED = "ROUND_FAILED"
    FINAL_VALIDATION_FAILED = "FINAL_VALIDATION_FAILED"
    PUBLICATION_FAILED = "PUBLICATION_FAILED"
    SHUTDOWN = "SHUTDOWN"


class RootCoordinatorError(RuntimeError):
    pass


class RootRequestConflictError(RootCoordinatorError):
    pass


class RootModelFamilyNotFoundError(RootCoordinatorError):
    pass


class RootCoordinatorUnavailableError(RootCoordinatorError):
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
    result_digest: str


@dataclass(frozen=True)
class RootAdmissionSnapshot:
    plan_id: str
    branches: tuple[AdmittedBranchSnapshot, ...]


@dataclass(frozen=True)
class RootRequestSnapshot:
    request_id: str
    plan_id: str
    model_family_id: FamilyKey
    state: RootRequestState
    failure_cause: str = ""
    failure_detail: str = ""
    admission: RootAdmissionSnapshot | None = None
    current_round: int | None = None
    completed_rounds: int = 0
    candidate_url: str = ""
    candidate_digest: str = ""
    published_model_id: int | None = None


@dataclass(frozen=True)
class RootInitiation:
    request_id: str
    plan_id: str
    model_family_id: FamilyKey
    source: str
    active_scopes: tuple[ScopeReference, ...] = ()


@dataclass
class _RootRequestRecord:
    initiation: RootInitiation
    reservation_id: str
    state: RootRequestState = RootRequestState.ACCEPTED
    failure_cause: str = ""
    failure_detail: str = ""
    server_process_id: str = ""
    admission: RootAdmissionSnapshot | None = None
    current_round: int | None = None
    completed_rounds: int = 0
    candidate_url: str = ""
    candidate_digest: str = ""
    published_model_id: int | None = None
    terminal_at: float | None = None
    terminal_deadline: float | None = None
    generation: int = 0
    future: Future | None = field(default=None, repr=False)


class FLRootCoordinator:
    def __init__(
        self,
        *,
        strategy: FederatedStrategySettings,
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
        terminal_status_ttl_seconds: int = 3600,
        clock=time.monotonic,
    ) -> None:
        if terminal_status_ttl_seconds <= 0:
            raise ValueError("terminal_status_ttl_seconds must be positive")
        self._strategy = FederatedStrategy(
            algorithm=FedProxAlgorithm(
                name=strategy.algorithm.name,
                proximal_mu=strategy.algorithm.proximal_mu,
            ),
            participant_selection=strategy.participant_selection,
            waiting_policy=strategy.waiting_policy,
            aggregation=strategy.aggregation,
        )
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
        self._terminal_status_ttl_seconds = terminal_status_ttl_seconds
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[str, _RootRequestRecord] = {}
        self._active_request_id: str | None = None
        self._generation = 0
        self._failure_latched = False
        self._closing = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fl-root")

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
            self._records.clear()
            self._failure_latched = False
            self._condition.notify_all()
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
        process: FLProcess | None = None
        try:
            self._set_state(record, RootRequestState.VALIDATING)
            current = self._catalog.current(record.initiation.model_family_id)
            if current is None:
                raise RuntimeError("FL base model is no longer current")
            descriptor = current.descriptor
            if descriptor.event != "UE_COMMUNICATION":
                raise RuntimeError("hierarchical FL V1 only supports UE_COMMUNICATION")
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
            resolved_branches = []
            for branch in topology.branches:
                branch_node = self._resolver.resolve(
                    nf_instance_id=branch.nf_instance_id,
                    role=HierarchyNodeRole.BRANCH,
                    ml_event=descriptor.event,
                    model_interoperability=descriptor.model_interoperability,
                )
                for leaf_id in branch.leaf_nf_instance_ids:
                    self._resolver.resolve(
                        nf_instance_id=leaf_id,
                        role=HierarchyNodeRole.LEAF,
                        ml_event=descriptor.event,
                        model_interoperability=descriptor.model_interoperability,
                    )
                resolved_branches.append((branch, branch_node))

            latest = self._catalog.current(record.initiation.model_family_id)
            if latest is None or latest.artifact.key != current.artifact.key:
                cause = RootFailureCause.VALIDATION_FAILED
                raise RuntimeError("FL base model changed during hierarchy validation")

            self._set_state(record, RootRequestState.DISPATCHING)
            cause = RootFailureCause.ASSIGNMENT_PUBLICATION_FAILED
            targets = []
            branch_assignments = {}
            for branch, resolved in resolved_branches:
                self._ensure_active_generation(record)
                artifact = self._artifact_service.publish_branch_assignment(
                    base=base,
                    plan_id=record.initiation.plan_id,
                    publisher_nf_instance_id=context.nf_instance_id,
                    branch_nf_instance_id=branch.nf_instance_id,
                    assigned_leaf_nf_instance_ids=branch.leaf_nf_instance_ids,
                    strategy=self._strategy,
                )
                branch_assignments[branch.nf_instance_id] = artifact
                targets.append(
                    HierarchyPreparationTarget(
                        participant_nf_instance_id=branch.nf_instance_id,
                        candidate=FLClientCandidate(
                            target=resolved.target,
                            tracking_areas=(),
                        ),
                        assignment_url=artifact.url,
                    )
                )

            latest = self._catalog.current(record.initiation.model_family_id)
            if latest is None or latest.artifact.key != current.artifact.key:
                cause = RootFailureCause.VALIDATION_FAILED
                raise RuntimeError("FL base model changed during assignment publication")

            cause = RootFailureCause.PREPARATION_DISPATCH_FAILED
            self._ensure_active_generation(record)
            process = self._server.start_hierarchy_preparation(
                plan_id=record.initiation.plan_id,
                reservation_id=record.reservation_id,
                family_key=record.initiation.model_family_id,
                model_id=current.model_id,
                ml_event=descriptor.event,
                ml_event_filter=descriptor.event_filter,
                target_ue=descriptor.target_ue,
                model_interoperability=descriptor.model_interoperability,
                targets=tuple(targets),
                active_scopes=record.initiation.active_scopes,
            )
            with self._condition:
                if self._active_request_id != record.initiation.request_id:
                    raise RuntimeError("Root request became stale during dispatch")
                record.server_process_id = process.process_id
                record.state = RootRequestState.PREPARATION_WAITING
                self._condition.notify_all()
            self._ensure_active_generation(record)
            cause = RootFailureCause.PREPARATION_FAILED
            collection = self._server.collect_hierarchy_preparation(process.process_id)
            self._set_state(record, RootRequestState.PREPARATION_EVALUATING)
            cause = RootFailureCause.RESULT_VALIDATION_FAILED
            admission = self._evaluate_preparation(
                record=record,
                process=process,
                collection=collection,
                topology=topology,
                assignments=branch_assignments,
                root_nf_instance_id=context.nf_instance_id,
                ml_event=descriptor.event,
                base_artifact_key=current.artifact.key,
            )
            self._server.admit_hierarchy_preparation(process.process_id)
            with self._condition:
                if self._active_request_id != record.initiation.request_id:
                    raise RuntimeError("Root request became stale during admission")
                record.admission = admission
                record.state = RootRequestState.ADMITTED
                self._condition.notify_all()

            cause = RootFailureCause.ROUND_FAILED
            source = base
            expected_subordinates = {
                item.branch_nf_instance_id: item.prepared_leaf_nf_instance_ids
                for item in admission.branches
            }
            for round_indicator in range(self._server_settings.round_count):
                self._set_round_state(
                    record,
                    RootRequestState.ROUND_DISPATCH,
                    round_indicator,
                )
                round_input = self._artifact_service.publish_round_input(
                    plan_id=record.initiation.plan_id,
                    base=source,
                    process_id=process.process_id,
                    server_nf_instance_id=context.nf_instance_id,
                    round_indicator=round_indicator,
                    epochs=self._server_settings.client_training.epochs,
                )
                aggregate = self._server.execute_hierarchy_round(
                    process_id=process.process_id,
                    round_indicator=round_indicator,
                    round_input_url=round_input.url,
                    round_input_artifact=round_input,
                    expected_result_type=RoundLocalResultType.HIERARCHY_AGGREGATE,
                    expected_subordinates=expected_subordinates,
                    state_observer=lambda state, current_round=round_indicator: (
                        self._observe_server_round_state(
                            record,
                            current_round,
                            state,
                        )
                    ),
                )
                self._ensure_active_generation(record)
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
                        raise RootCoordinatorUnavailableError(
                            "Root coordinator is closing"
                        )
                    record.completed_rounds = round_indicator + 1
                    record.candidate_url = aggregate.url
                    record.candidate_digest = aggregate.digest
                    self._condition.notify_all()
            cause = RootFailureCause.FINAL_VALIDATION_FAILED
            self._ensure_active_generation(record)
            result = self._server.finalize_hierarchy_candidate(
                process_id=process.process_id,
                validation_round=self._server_settings.round_count,
                candidate=aggregate,
                base_artifact=current.artifact,
                expected_subordinates=expected_subordinates,
                state_observer=lambda state: self._observe_server_finalization_state(
                    record,
                    state,
                ),
            )
            with self._condition:
                if record.generation == self._generation:
                    record.published_model_id = getattr(
                        result,
                        "published_model_id",
                        None,
                    )
            self._close_training_outcome(record, result.state)
        except Exception as error:
            if isinstance(error, HierarchyDiscoveryError):
                cause = RootFailureCause.DISCOVERY_FAILED
            elif isinstance(error, RootCoordinatorUnavailableError):
                cause = RootFailureCause.SHUTDOWN
            elif isinstance(error, RootPreparationError):
                cause = error.cause
            elif (
                process is not None
                and getattr(process, "state", None) is FLServerState.PUBLISHING
            ):
                cause = RootFailureCause.PUBLICATION_FAILED
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

    def _evaluate_preparation(
        self,
        *,
        record: _RootRequestRecord,
        process: FLProcess,
        collection: HierarchyPreparationCollection,
        topology,
        assignments: dict[str, object],
        root_nf_instance_id: str,
        ml_event: str,
        base_artifact_key: str,
    ) -> RootAdmissionSnapshot:
        if collection.plan_id != record.initiation.plan_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "upper preparation collection plan does not match",
            )
        if collection.process_id != process.process_id:
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "upper preparation collection process does not match",
            )
        if collection.timed_out_participant_nf_instance_ids:
            raise RootPreparationError(
                RootFailureCause.PREPARATION_TIMEOUT,
                "one or more Branch callbacks timed out",
            )
        expected = {
            branch.nf_instance_id: branch.leaf_nf_instance_ids
            for branch in topology.branches
        }
        outcomes = {
            item.participant_nf_instance_id: item for item in collection.participants
        }
        if set(outcomes) != set(expected):
            raise RootPreparationError(
                RootFailureCause.RESULT_VALIDATION_FAILED,
                "upper preparation collection does not cover configured Branches",
            )
        admitted = []
        rejected = False
        for branch_id in sorted(expected):
            outcome = outcomes[branch_id]
            notification = outcome.notification
            if outcome.failure or notification is None:
                raise RootPreparationError(
                    RootFailureCause.PREPARATION_FAILED,
                    "Branch preparation did not return a valid outcome",
                )
            if len(notification.ml_model_infos or ()) != 1:
                raise RootPreparationError(
                    RootFailureCause.RESULT_VALIDATION_FAILED,
                    "Branch preparation result URL is missing",
                )
            model_info = notification.ml_model_infos[0]
            if model_info.event != ml_event or model_info.model_file_address is None:
                raise RootPreparationError(
                    RootFailureCause.RESULT_VALIDATION_FAILED,
                    "Branch preparation result event or address is invalid",
                )
            validated = self._workspace.download_hierarchy(
                str(model_info.model_file_address.model_url),
                expected_role=ArtifactRole.HIERARCHY_PREPARATION_RESULT,
                expected_message_type=HierarchyMessageType.PREPARATION_RESULT,
                expected_publisher_nf_instance_id=branch_id,
                intended_recipient_nf_instance_id=root_nf_instance_id,
                expected_plan_id=record.initiation.plan_id,
            )
            contract = validated.contract
            metadata = contract.hierarchy_metadata
            if not isinstance(contract, HierarchyPreparationResultArtifact) or not isinstance(
                metadata,
                PreparationResultMetadata,
            ):
                raise RootPreparationError(
                    RootFailureCause.RESULT_VALIDATION_FAILED,
                    "Branch result artifact contract is invalid",
                )
            assignment = assignments[branch_id]
            if contract.file_digests != assignment.contract.file_digests:
                raise RootPreparationError(
                    RootFailureCause.RESULT_VALIDATION_FAILED,
                    "Branch result changed the assigned model bundle",
                )
            if metadata.assigned_client_nf_instance_ids != expected[branch_id]:
                raise RootPreparationError(
                    RootFailureCause.RESULT_VALIDATION_FAILED,
                    "Branch result assigned Leaf set does not match topology",
                )
            prepared_ids = tuple(item.nf_instance_id for item in metadata.prepared_clients)
            if (
                metadata.outcome is not PreparationOutcome.READY
                or prepared_ids != expected[branch_id]
                or metadata.failed_clients
                or metadata.timed_out_client_nf_instance_ids
                or notification.termination_request is not None
            ):
                rejected = True
            admitted.append(
                AdmittedBranchSnapshot(
                    branch_nf_instance_id=branch_id,
                    prepared_leaf_nf_instance_ids=prepared_ids,
                    upper_resource_location=outcome.resource_location,
                    result_digest=validated.metadata.key,
                )
            )
        latest = self._catalog.current(record.initiation.model_family_id)
        active = self._experiments.active()
        if latest is None or latest.artifact.key != base_artifact_key:
            raise RootPreparationError(
                RootFailureCause.ADMISSION_REJECTED,
                "Root base model changed during preparation",
            )
        if (
            active is None
            or active.reservation_id != record.reservation_id
            or active.plan_id != record.initiation.plan_id
            or active.server_process_id != process.process_id
        ):
            raise RootPreparationError(
                RootFailureCause.ADMISSION_REJECTED,
                "active hierarchy experiment changed during preparation",
            )
        if rejected:
            raise RootPreparationError(
                RootFailureCause.ADMISSION_REJECTED,
                "complete-required hierarchy admission was rejected",
            )
        return RootAdmissionSnapshot(
            plan_id=record.initiation.plan_id,
            branches=tuple(admitted),
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

    def _observe_server_finalization_state(
        self,
        record: _RootRequestRecord,
        state: FLServerState,
    ) -> None:
        projected = {
            FLServerState.FINAL_VALIDATION_DISPATCH: (
                RootRequestState.FINAL_VALIDATION_DISPATCH
            ),
            FLServerState.FINAL_VALIDATION_WAITING: (
                RootRequestState.FINAL_VALIDATION_WAITING
            ),
            FLServerState.FINAL_VALIDATION_EVALUATING: (
                RootRequestState.FINAL_VALIDATION_EVALUATING
            ),
            FLServerState.CANDIDATE_READY: RootRequestState.CANDIDATE_READY,
            FLServerState.PUBLISHING: RootRequestState.PUBLISHING,
        }.get(state)
        if projected is not None:
            with self._condition:
                if self._closing:
                    raise RootCoordinatorUnavailableError("Root coordinator is closing")
                if self._active_request_id != record.initiation.request_id:
                    raise RootRequestConflictError("Root request is stale")
                record.state = projected
                self._condition.notify_all()
            return
        if state is FLServerState.COMPLETE:
            with self._condition:
                adoption_completed = record.state is RootRequestState.CUTOVER_PENDING
            if adoption_completed:
                self._close_training_outcome(record, state)

    def _close_training_outcome(
        self,
        record: _RootRequestRecord,
        state: FLServerState,
    ) -> None:
        retain_top_level = state is FLServerState.CUTOVER_PENDING
        process_id = record.server_process_id
        if process_id:
            try:
                self._server.close_hierarchy_training(
                    process_id,
                    retain_for_adoption=retain_top_level,
                )
            except (KeyError, RuntimeError):
                logger.exception("Failed to close hierarchy Server process %s", process_id)
        try:
            self._workspace.release_plan(record.initiation.plan_id)
        except RuntimeError:
            logger.exception(
                "Failed to release hierarchy workspace plan_id=%s",
                record.initiation.plan_id,
            )
        record.candidate_url = ""
        if retain_top_level:
            with self._condition:
                record.state = RootRequestState.CUTOVER_PENDING
                self._condition.notify_all()
            return
        self._release_reservation(record, state.value)
        with self._condition:
            record.state = RootRequestState(state.value)
            if self._active_request_id == record.initiation.request_id:
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
        record.candidate_url = ""
        if record.terminal_deadline is None:
            record.terminal_at = self._clock()
            record.terminal_deadline = (
                record.terminal_at + self._terminal_status_ttl_seconds
            )

    def _prune_terminal_records_locked(self) -> None:
        now = self._clock()
        self._records = {
            request_id: record
            for request_id, record in self._records.items()
            if record.terminal_deadline is None or record.terminal_deadline > now
        }

    @staticmethod
    def _snapshot(record: _RootRequestRecord) -> RootRequestSnapshot:
        return RootRequestSnapshot(
            request_id=record.initiation.request_id,
            plan_id=record.initiation.plan_id,
            model_family_id=record.initiation.model_family_id,
            state=record.state,
            failure_cause=record.failure_cause,
            failure_detail=record.failure_detail,
            admission=record.admission,
            current_round=record.current_round,
            completed_rounds=record.completed_rounds,
            candidate_url=record.candidate_url,
            candidate_digest=record.candidate_digest,
            published_model_id=record.published_model_id,
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


def _public_failure_detail(cause: RootFailureCause) -> str:
    return {
        RootFailureCause.VALIDATION_FAILED: "Root hierarchy validation failed",
        RootFailureCause.DISCOVERY_FAILED: "configured hierarchy node discovery failed",
        RootFailureCause.ASSIGNMENT_PUBLICATION_FAILED: "hierarchy assignment publication failed",
        RootFailureCause.PREPARATION_DISPATCH_FAILED: "upper-tier preparation dispatch failed",
        RootFailureCause.PREPARATION_FAILED: "one or more Branch preparations failed",
        RootFailureCause.PREPARATION_TIMEOUT: "one or more Branch preparations timed out",
        RootFailureCause.RESULT_VALIDATION_FAILED: "Branch preparation result validation failed",
        RootFailureCause.ADMISSION_REJECTED: "complete-required hierarchy admission was rejected",
        RootFailureCause.ROUND_FAILED: "hierarchical training round failed",
        RootFailureCause.FINAL_VALIDATION_FAILED: (
            "hierarchical final validation failed"
        ),
        RootFailureCause.PUBLICATION_FAILED: "hierarchical model publication failed",
        RootFailureCause.SHUTDOWN: "Root coordinator is shutting down",
    }[cause]
