from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from enum import StrEnum

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    HierarchyAssignmentArtifact,
    RoundGlobalArtifact,
    RoundInputArtifact,
    RoundLocalResultType,
    validate_fl_artifact_manifest,
)
from py_mtlf.core.fl_candidate_orchestration import (
    CandidatePool,
    CandidateStatusCause,
    IntermediateLocalWork,
    LocalContractDefaults,
    LocalExecutionRole,
    resolve_effective_node_contract,
)
from py_mtlf.core.fl_hierarchy import (
    BranchAssignmentMetadata,
    FailedClient,
    PreparationFailureCause,
    PreparationOutcome,
    PreparedClient,
)
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyNodeResolver,
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLServerEngine,
    HierarchyPreparationCollection,
    HierarchyPreparationTarget,
    HierarchyValidationCollection,
    ProtocolPreparationTarget,
)
from py_mtlf.core.fl_workspace import FLWorkspaceArtifact, ValidatedHierarchyArtifact
from py_mtlf.core.nwdaf_context import FLCapabilityType, NwdafContextClient
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.ml_model import MLEventNotification
from py_mtlf.wire.ml_model_training import (
    FlReportAfter,
    FlTopologyNode,
    FlTopologyReport,
    NwdafMLModelTrainSubsc,
)


@dataclass(frozen=True)
class BranchPreparationExecution:
    plan_id: str
    parent_assignment: ValidatedHierarchyArtifact
    leaf_nodes: tuple[ResolvedHierarchyNode, ...]
    leaf_assignments: tuple[FLWorkspaceArtifact, ...]
    process_id: str


@dataclass(frozen=True)
class BranchPreparationResult:
    artifact: FLWorkspaceArtifact
    outcome: PreparationOutcome
    execution: BranchPreparationExecution | None


@dataclass
class ProtocolBranchExecution:
    ml_correlation_id: str
    process_id: str
    candidate_pool: CandidatePool


class BranchRoundState(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BranchRoundExecution:
    plan_id: str
    upper_client_subscription_id: str
    upper_resource_revision: int
    upper_ml_corre_id: str
    upper_round_indicator: int
    upper_input_artifact_digest: str
    upper_training_scope: TrainingScopeDescriptor
    lower_server_process_id: str
    lower_ml_corre_id: str
    lower_round_indicator: int
    lower_input_artifact_digest: str = ""
    state: BranchRoundState = BranchRoundState.RUNNING
    upper_result: FLWorkspaceArtifact | None = None
    failure: str = ""


@dataclass(frozen=True)
class BranchValidationExecution:
    plan_id: str
    upper_client_subscription_id: str
    upper_resource_revision: int
    upper_ml_corre_id: str
    upper_round_indicator: int
    upper_candidate_artifact_digest: str
    upper_training_scope: TrainingScopeDescriptor
    lower_server_process_id: str
    lower_validation_round: int
    republished_candidate: FLWorkspaceArtifact | None = None
    state: BranchRoundState = BranchRoundState.RUNNING
    upper_result: FLWorkspaceArtifact | None = None
    failure: str = ""


class BranchPreDispatchError(RuntimeError):
    def __init__(
        self,
        leaf_nf_instance_id: str | None,
        cause: PreparationFailureCause,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.leaf_nf_instance_id = leaf_nf_instance_id
        self.cause = cause


class BranchPreparationCancelled(RuntimeError):
    pass


class FLBranchPreparationCoordinator:
    """Expand one validated Branch assignment into lower-tier preparation."""

    def __init__(
        self,
        *,
        resolver: HierarchyNodeResolver,
        nwdaf_context: NwdafContextClient,
        artifact_service: HierarchyArtifactService,
        server: FLServerEngine,
        tombstone_ttl_seconds: int = 3600,
        clock=time.monotonic,
    ) -> None:
        if tombstone_ttl_seconds <= 0:
            raise ValueError("tombstone_ttl_seconds must be positive")
        self._resolver = resolver
        self._nwdaf_context = nwdaf_context
        self._artifact_service = artifact_service
        self._server = server
        self._loader = TrustedBundleLoader()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._executions: dict[str, BranchPreparationExecution] = {}
        self._protocol_executions: dict[str, ProtocolBranchExecution] = {}
        self._rounds: dict[tuple[str, str, int], BranchRoundExecution] = {}
        self._validations: dict[
            tuple[str, str, int], BranchValidationExecution
        ] = {}
        self._next_lower_round: dict[str, int] = {}
        self._cancelled_plan_ids: dict[str, float] = {}
        self._tombstone_ttl_seconds = tombstone_ttl_seconds
        self._clock = clock
        self._closing = False

    def prepare_protocol(
        self,
        *,
        representation: NwdafMLModelTrainSubsc,
        reservation_id: str,
    ) -> FlTopologyReport:
        node = representation.fl_topology
        if node is None or not (representation.ml_correlation_id or "").strip():
            raise ValueError("protocol Branch preparation requires topology and mlCorreId")
        event = representation.ml_event_subscriptions[0]
        context = self._nwdaf_context.get()
        if context.nf_instance_id != node.nf_instance_id:
            raise RuntimeError("protocol Branch topology recipient changed")
        contract = resolve_effective_node_contract(
            node,
            role=LocalExecutionRole.INTERMEDIATE,
            defaults=LocalContractDefaults(),
        )
        with self._lock:
            execution = self._protocol_executions.get(
                representation.ml_correlation_id
            )
        if execution is None:
            pool = CandidatePool(contract)
            process_id = ""
            reconciliation = None
        else:
            pool = execution.candidate_pool
            process_id = execution.process_id
            reconciliation = pool.reconfigure(contract)
        if reconciliation is not None:
            for intent in reconciliation.delete_intents:
                try:
                    self._server.remove_protocol_participant(
                        process_id,
                        intent.nf_instance_id,
                    )
                except (KeyError, RuntimeError):
                    pool.complete_delete(
                        intent.nf_instance_id,
                        intent.revision,
                        cause=CandidateStatusCause.COMMUNICATION_FAILURE,
                    )
                else:
                    pool.complete_delete(intent.nf_instance_id, intent.revision)
        records = {record.nf_instance_id: record for record in pool.records()}
        for child in contract.explicit_children:
            record = records[child.nf_instance_id]
            if child.enabled is False or record.resolved is not None:
                continue
            role = (
                HierarchyNodeRole.BRANCH
                if child.children
                or (
                    child.policy is not None
                    and child.policy.allow_additional_candidates is True
                )
                else HierarchyNodeRole.LEAF
            )
            resolved = self._resolver.resolve(
                nf_instance_id=child.nf_instance_id,
                role=role,
                ml_event=event.ml_event,
                model_interoperability=event.model_interoperability,
            )
            pool.set_resolved(resolved)
        if contract.policy.allow_additional_candidates:
            scope = self._resolver.discovery_scope_for_subscription(
                representation,
                role=HierarchyNodeRole.LEAF,
            )
            refresh = pool.request_discovery_refresh(scope)
            if refresh is not None:
                try:
                    snapshot = self._resolver.discover_for_subscription(
                        representation,
                        role=HierarchyNodeRole.LEAF,
                        excluded_nf_instance_ids=(context.nf_instance_id,),
                    )
                except Exception:
                    pool.fail_discovery_refresh(refresh)
                    raise
                if not pool.complete_discovery_refresh(refresh, snapshot):
                    raise RuntimeError("protocol Branch discovery result became stale")

        established = False
        while not pool.ready():
            intents = pool.next_establishment_intents(1)
            if not intents:
                raise RuntimeError("protocol Branch candidate pool did not reach readiness")
            intent = intents[0]
            target = self._protocol_target(intent, contract)
            if not process_id:
                process = self._server.start_protocol_preparation(
                    ml_correlation_id=representation.ml_correlation_id,
                    reservation_id=reservation_id,
                    ml_event=event.ml_event,
                    ml_event_filter=dict(event.ml_event_filter),
                    model_interoperability=event.model_interoperability,
                    targets=(target,),
                )
                process_id = process.process_id
            else:
                self._server.add_protocol_preparation_targets(
                    process_id=process_id,
                    ml_event=event.ml_event,
                    ml_event_filter=dict(event.ml_event_filter),
                    model_interoperability=event.model_interoperability,
                    targets=(target,),
                )
            collection = self._server.collect_hierarchy_preparation(process_id)
            outcomes = {
                item.participant_nf_instance_id: item
                for item in collection.participants
            }
            outcome = outcomes[intent.nf_instance_id]
            report = (
                outcome.notification.fl_topology_report
                if outcome.notification is not None
                else None
            )
            failure = outcome.failure
            if report is None and not failure:
                failure = CandidateStatusCause.REQUIREMENTS_NOT_MET.value
            applied = pool.complete_establishment(
                intent.nf_instance_id,
                intent.revision,
                resource_location=outcome.resource_location,
                failure_cause=failure or None,
            )
            if not applied:
                raise RuntimeError("protocol Branch candidate completion became stale")
            if report is not None and not failure:
                pool.attach_child_report(intent.nf_instance_id, report)
            established = True
        if not pool.ready():
            raise RuntimeError("protocol Branch candidate pool did not reach readiness")
        if not process_id:
            raise RuntimeError("protocol Branch has no downstream Server process")
        if established:
            self._server.admit_hierarchy_preparation(process_id)
        if execution is None:
            execution = ProtocolBranchExecution(
                ml_correlation_id=representation.ml_correlation_id,
                process_id=process_id,
                candidate_pool=pool,
            )
            with self._lock:
                if self._closing:
                    self._server.cancel_hierarchy_preparation(
                        process_id,
                        "Branch preparation coordinator is closing",
                    )
                    raise RuntimeError("Branch preparation coordinator is closing")
                self._protocol_executions[
                    representation.ml_correlation_id
                ] = execution
        return pool.snapshot()

    def _protocol_target(self, intent, contract) -> ProtocolPreparationTarget:
        if intent.target is None:
            raise RuntimeError("protocol Branch candidate target is unresolved")
        instruction = intent.instruction
        if instruction is None:
            instruction = FlTopologyNode(
                nfInstanceId=intent.nf_instance_id,
                enabled=True,
                priority=contract.policy.additional_candidate_priority,
            )
        updates = {}
        if instruction.strategy is None:
            updates["strategy"] = contract.strategy.as_wire()
        if instruction.report_after is None:
            updates["report_after"] = FlReportAfter(
                count=(
                    1
                    if intent.role is HierarchyNodeRole.BRANCH
                    else self._server.default_protocol_client_epochs
                ),
                unit=(
                    "round"
                    if intent.role is HierarchyNodeRole.BRANCH
                    else "epoch"
                ),
            )
        if updates:
            instruction = instruction.model_copy(update=updates, deep=True)
        return ProtocolPreparationTarget(
            participant_nf_instance_id=intent.nf_instance_id,
            candidate=FLClientCandidate(
                target=intent.target.target,
                tracking_areas=(),
            ),
            topology=instruction,
        )

    def dispatch(
        self,
        *,
        assignment: ValidatedHierarchyArtifact,
        representation: NwdafMLModelTrainSubsc,
        reservation_id: str,
    ) -> BranchPreparationExecution:
        contract = assignment.contract
        metadata = contract.hierarchy_metadata
        if not isinstance(contract, HierarchyAssignmentArtifact) or not isinstance(
            metadata,
            BranchAssignmentMetadata,
        ):
            raise ValueError("Branch coordinator requires a BRANCH_ASSIGNMENT")
        event = representation.ml_event_subscriptions[0]
        context = self._nwdaf_context.get()
        if context.nf_instance_id != metadata.intended_recipient_nf_instance_id:
            raise RuntimeError("Branch assignment recipient no longer matches local NWDAF")
        if not any(
            event.ml_event in capability.ml_analytics_ids
            and capability.fl_capability_type is FLCapabilityType.SERVER_AND_CLIENT
            for capability in context.ml_analytics_capabilities
        ):
            raise RuntimeError(
                "containing NWDAF does not advertise the required Branch capability"
            )
        with self._lock:
            self._prune_cancelled_locked()
            if self._closing:
                raise RuntimeError("Branch preparation coordinator is closing")
            if metadata.plan_id in self._cancelled_plan_ids:
                raise BranchPreparationCancelled("Branch preparation was cancelled")
            if metadata.plan_id in self._executions:
                return self._executions[metadata.plan_id]

        resolved = []
        for leaf_id in metadata.assigned_leaf_nf_instance_ids:
            try:
                resolved.append(
                    self._resolver.resolve(
                        nf_instance_id=leaf_id,
                        role=HierarchyNodeRole.LEAF,
                        ml_event=event.ml_event,
                        model_interoperability=event.model_interoperability,
                    )
                )
            except Exception as error:
                raise BranchPreDispatchError(
                    leaf_id,
                    PreparationFailureCause.DISCOVERY_FAILED,
                    "assigned Leaf discovery failed",
                ) from error
        leaf_nodes = tuple(resolved)
        self._ensure_dispatch_active(metadata.plan_id)
        published_assignments = []
        for node in leaf_nodes:
            try:
                published_assignments.append(
                    self._artifact_service.republish_leaf_assignment(
                        parent=assignment,
                        containing_branch_nf_instance_id=context.nf_instance_id,
                        leaf_nf_instance_id=node.nf_instance_id,
                    )
                )
            except Exception as error:
                raise BranchPreDispatchError(
                    node.nf_instance_id,
                    PreparationFailureCause.INVALID_BUNDLE,
                    "Leaf assignment publication failed",
                ) from error
        leaf_assignments = tuple(published_assignments)
        self._ensure_dispatch_active(metadata.plan_id)
        target_ue = representation.target_reporting_ue or event.target_ue
        process = self._server.start_hierarchy_preparation(
            plan_id=metadata.plan_id,
            reservation_id=reservation_id,
            family_key=None,
            model_id=None,
            ml_event=event.ml_event,
            ml_event_filter=dict(event.ml_event_filter),
            target_ue=dict(target_ue) if target_ue is not None else None,
            model_interoperability=event.model_interoperability,
            targets=tuple(
                HierarchyPreparationTarget(
                    participant_nf_instance_id=node.nf_instance_id,
                    candidate=FLClientCandidate(
                        target=node.target,
                        tracking_areas=(),
                    ),
                    assignment_url=published.url,
                )
                for node, published in zip(
                    leaf_nodes,
                    leaf_assignments,
                    strict=True,
                )
            ),
            active_scopes=(),
        )
        execution = BranchPreparationExecution(
            plan_id=metadata.plan_id,
            parent_assignment=assignment,
            leaf_nodes=leaf_nodes,
            leaf_assignments=leaf_assignments,
            process_id=process.process_id,
        )
        with self._lock:
            self._prune_cancelled_locked()
            if self._closing or metadata.plan_id in self._cancelled_plan_ids:
                self._server.cancel_hierarchy_preparation(
                    process.process_id,
                    "Branch preparation coordinator is closing",
                )
                raise RuntimeError("Branch preparation coordinator is closing")
            self._executions[metadata.plan_id] = execution
        return execution

    def prepare(
        self,
        *,
        assignment: ValidatedHierarchyArtifact,
        representation: NwdafMLModelTrainSubsc,
        reservation_id: str,
    ) -> BranchPreparationResult:
        metadata = assignment.contract.hierarchy_metadata
        if not isinstance(metadata, BranchAssignmentMetadata):
            raise ValueError("Branch coordinator requires a BRANCH_ASSIGNMENT")
        try:
            execution = self.dispatch(
                assignment=assignment,
                representation=representation,
                reservation_id=reservation_id,
            )
        except BranchPreparationCancelled:
            raise
        except BranchPreDispatchError as error:
            failed = tuple(
                FailedClient(
                    nf_instance_id=leaf_id,
                    cause=(
                        error.cause
                        if leaf_id == error.leaf_nf_instance_id
                        else PreparationFailureCause.NOT_AVAILABLE_ML_TRAIN
                    ),
                )
                for leaf_id in metadata.assigned_leaf_nf_instance_ids
            )
            artifact = self._artifact_service.publish_preparation_result(
                parent=assignment,
                containing_branch_nf_instance_id=(
                    metadata.intended_recipient_nf_instance_id
                ),
                outcome=PreparationOutcome.FAILED,
                prepared_clients=(),
                failed_clients=failed,
                timed_out_client_nf_instance_ids=(),
            )
            return BranchPreparationResult(
                artifact=artifact,
                outcome=PreparationOutcome.FAILED,
                execution=None,
            )
        except Exception as error:
            with self._lock:
                cancelled = (
                    self._closing or metadata.plan_id in self._cancelled_plan_ids
                )
            if cancelled:
                raise BranchPreparationCancelled(
                    "Branch preparation was cancelled"
                ) from error
            failed = tuple(
                FailedClient(
                    nf_instance_id=leaf_id,
                    cause=PreparationFailureCause.INTERNAL_ERROR,
                )
                for leaf_id in metadata.assigned_leaf_nf_instance_ids
            )
            artifact = self._artifact_service.publish_preparation_result(
                parent=assignment,
                containing_branch_nf_instance_id=(
                    metadata.intended_recipient_nf_instance_id
                ),
                outcome=PreparationOutcome.FAILED,
                prepared_clients=(),
                failed_clients=failed,
                timed_out_client_nf_instance_ids=(),
            )
            return BranchPreparationResult(
                artifact=artifact,
                outcome=PreparationOutcome.FAILED,
                execution=None,
            )

        collection = self._server.collect_hierarchy_preparation(execution.process_id)
        prepared, failed, timed_out = self._classify(execution, collection)
        outcome = (
            PreparationOutcome.READY
            if len(prepared) == len(metadata.assigned_leaf_nf_instance_ids)
            else PreparationOutcome.FAILED
        )
        artifact = self._artifact_service.publish_preparation_result(
            parent=assignment,
            containing_branch_nf_instance_id=metadata.intended_recipient_nf_instance_id,
            outcome=outcome,
            prepared_clients=prepared,
            failed_clients=failed,
            timed_out_client_nf_instance_ids=timed_out,
        )
        if outcome is PreparationOutcome.READY:
            self._server.admit_hierarchy_preparation(execution.process_id)
        return BranchPreparationResult(
            artifact=artifact,
            outcome=outcome,
            execution=execution,
        )

    def execute_protocol_round(
        self,
        *,
        representation: NwdafMLModelTrainSubsc,
        upper_input: LoadedBundle,
        upper_input_artifact: ArtifactMetadata,
        upper_model: MLEventNotification,
        upper_client_subscription_id: str,
        upper_resource_revision: int,
        upper_training_scope: TrainingScopeDescriptor,
        callback_margin_seconds: int,
        local_work: IntermediateLocalWork,
    ) -> FLWorkspaceArtifact:
        ml_correlation_id = representation.ml_correlation_id or ""
        upper_round = representation.round_indicator
        if not ml_correlation_id or upper_round is None:
            raise ValueError("protocol Branch round requires procedure and round identity")
        if not upper_client_subscription_id or upper_resource_revision <= 0:
            raise ValueError("protocol Branch round requires upper resource identity")
        report = representation.ml_training_report_info
        parent_budget = report.maximum_response_time if report is not None else None
        if parent_budget is None or parent_budget <= callback_margin_seconds:
            raise RuntimeError("protocol Branch budget cannot contain lower execution")
        lower_round_count = local_work.lower_round_count
        lower_budget = parent_budget - callback_margin_seconds
        if lower_budget < lower_round_count:
            raise RuntimeError("protocol Branch budget cannot contain all lower rounds")
        per_round_timeout = lower_budget // lower_round_count
        input_contract = validate_fl_artifact_manifest(upper_input.manifest)
        if not isinstance(input_contract, RoundInputArtifact):
            raise RuntimeError("protocol Branch input is not a ROUND_INPUT artifact")
        if input_contract.fl_metadata.ml_corre_id != ml_correlation_id:
            raise RuntimeError("protocol Branch input belongs to another procedure")
        context = self._nwdaf_context.get()
        with self._condition:
            execution = self._protocol_executions.get(ml_correlation_id)
            if execution is None:
                raise RuntimeError("protocol Branch lower process is unavailable")
            lower_round = self._next_lower_round.get(ml_correlation_id, 0)
            self._next_lower_round[ml_correlation_id] = lower_round + lower_round_count
        current_input = upper_input
        current_artifact = FLWorkspaceArtifact(
            process_id=ml_correlation_id,
            participant_id="",
            round_indicator=input_contract.fl_metadata.round_ind,
            role="ROUND_INPUT",
            digest=upper_input_artifact.key,
            path=upper_input_artifact.path,
            url=upper_input_artifact.url,
            manifest=dict(upper_input.manifest),
            contract=input_contract,
        )
        lower_global = None
        for offset in range(lower_round_count):
            current_lower_round = lower_round + offset
            selection = execution.candidate_pool.select_round()
            if offset == 0:
                dispatch_model = upper_model.model_copy(deep=True)
            else:
                current_artifact = self._artifact_service.publish_round_input(
                    plan_id=ml_correlation_id,
                    base=current_input,
                    process_id=ml_correlation_id,
                    server_nf_instance_id=context.nf_instance_id,
                    round_indicator=current_lower_round,
                    epochs=_round_epochs(upper_input),
                )
                dispatch_model = None
            lower_global = self._server.execute_hierarchy_round(
                process_id=execution.process_id,
                round_indicator=current_lower_round,
                round_input_url=current_artifact.url,
                round_input_artifact=current_artifact,
                round_input_model=dispatch_model,
                input_round_indicator=current_artifact.contract.fl_metadata.round_ind,
                expected_result_type=RoundLocalResultType.TRAINING,
                selected_participant_nf_instance_ids=(
                    selection.participant_nf_instance_ids
                ),
                accept_failures=execution.candidate_pool.contract.policy.accept_failures,
                minimum_completion_rate=(
                    execution.candidate_pool.contract.policy.minimum_completion_rate
                ),
                timeout_seconds=per_round_timeout,
            )
            if not isinstance(lower_global.contract, RoundGlobalArtifact):
                raise RuntimeError("protocol Branch lower result is not ROUND_GLOBAL")
            current_input = self._loader.load(
                ArtifactMetadata(
                    key=lower_global.digest,
                    size_bytes=lower_global.path.stat().st_size,
                    path=lower_global.path,
                    url=lower_global.url,
                )
            )
            current_artifact = lower_global
        assert lower_global is not None
        return self._artifact_service.publish_hierarchy_aggregate(
            upper_input=upper_input,
            lower_global=lower_global,
            plan_id=ml_correlation_id,
            upper_process_id=ml_correlation_id,
            branch_nf_instance_id=context.nf_instance_id,
            upper_round_indicator=upper_round,
            upper_training_scope=upper_training_scope,
        )

    def execute_round(
        self,
        *,
        assignment: ValidatedHierarchyArtifact,
        representation: NwdafMLModelTrainSubsc,
        upper_input: LoadedBundle,
        upper_client_subscription_id: str,
        upper_resource_revision: int,
        upper_input_artifact_digest: str,
        upper_training_scope: TrainingScopeDescriptor,
        callback_margin_seconds: int,
        local_work: IntermediateLocalWork | None = None,
    ) -> FLWorkspaceArtifact:
        metadata = assignment.contract.hierarchy_metadata
        if not isinstance(metadata, BranchAssignmentMetadata):
            raise ValueError("Branch round requires a BRANCH_ASSIGNMENT")
        upper_process_id = representation.ml_correlation_id or ""
        upper_round = representation.round_indicator
        if not upper_process_id or upper_round is None:
            raise ValueError("Branch round requires upper process and round identities")
        if not upper_client_subscription_id or upper_resource_revision <= 0:
            raise ValueError("Branch round requires upper resource identity")
        if len(upper_input_artifact_digest) != 64:
            raise ValueError("Branch round requires the upper input artifact digest")
        key = (metadata.plan_id, upper_process_id, upper_round)
        report = representation.ml_training_report_info
        parent_budget = report.maximum_response_time if report is not None else None
        if parent_budget is None or parent_budget <= callback_margin_seconds:
            raise RuntimeError("Branch upper round budget cannot contain lower execution")
        context = self._nwdaf_context.get()
        if context.nf_instance_id != metadata.intended_recipient_nf_instance_id:
            raise RuntimeError("Branch assignment recipient no longer matches local NWDAF")
        epochs = _round_epochs(upper_input)
        lower_round_count = local_work.lower_round_count if local_work is not None else 1
        lower_budget = parent_budget - callback_margin_seconds
        if lower_budget < lower_round_count:
            raise RuntimeError("Branch upper round budget cannot contain all lower rounds")
        per_round_timeout = lower_budget // lower_round_count
        conflicting = False
        lower_round = -1
        with self._condition:
            self._ensure_dispatch_active(metadata.plan_id)
            execution = self._executions.get(metadata.plan_id)
            if execution is None:
                raise RuntimeError("Branch lower process is unavailable")
            while True:
                existing = self._rounds.get(key)
                if existing is None:
                    lower_round = self._next_lower_round.get(metadata.plan_id, 0)
                    self._next_lower_round[metadata.plan_id] = (
                        lower_round + lower_round_count
                    )
                    self._rounds[key] = BranchRoundExecution(
                        plan_id=metadata.plan_id,
                        upper_client_subscription_id=upper_client_subscription_id,
                        upper_resource_revision=upper_resource_revision,
                        upper_ml_corre_id=upper_process_id,
                        upper_round_indicator=upper_round,
                        upper_input_artifact_digest=upper_input_artifact_digest,
                        upper_training_scope=upper_training_scope,
                        lower_server_process_id=execution.process_id,
                        lower_ml_corre_id=execution.process_id,
                        lower_round_indicator=lower_round,
                    )
                    break
                if not _same_round_command(
                    existing,
                    upper_client_subscription_id=upper_client_subscription_id,
                    upper_resource_revision=upper_resource_revision,
                    upper_input_artifact_digest=upper_input_artifact_digest,
                    upper_training_scope=upper_training_scope,
                ):
                    conflicting = True
                    break
                if existing.state is BranchRoundState.COMPLETE:
                    if existing.upper_result is None:
                        raise RuntimeError("completed Branch round has no upper result")
                    return existing.upper_result
                if existing.state is BranchRoundState.FAILED:
                    raise RuntimeError(existing.failure or "Branch round execution failed")
                self._condition.wait()
                self._ensure_dispatch_active(metadata.plan_id)
        if conflicting:
            failure = "conflicting duplicate Branch upper round command"
            self.cancel(metadata.plan_id, failure)
            raise RuntimeError(failure)

        try:
            current_input = upper_input
            lower_global = None
            for offset in range(lower_round_count):
                current_lower_round = lower_round + offset
                with self._condition:
                    self._ensure_dispatch_active(metadata.plan_id)
                    current = self._rounds.get(key)
                    if current is None:
                        raise RuntimeError("Branch round mapping disappeared")
                    lower_input = self._artifact_service.publish_round_input(
                        plan_id=execution.plan_id,
                        base=current_input,
                        process_id=execution.process_id,
                        server_nf_instance_id=context.nf_instance_id,
                        round_indicator=current_lower_round,
                        epochs=epochs,
                    )
                    self._rounds[key] = replace(
                        current,
                        lower_round_indicator=current_lower_round,
                        lower_input_artifact_digest=lower_input.digest,
                    )
                lower_global = self._server.execute_hierarchy_round(
                    process_id=execution.process_id,
                    round_indicator=current_lower_round,
                    round_input_url=lower_input.url,
                    round_input_artifact=lower_input,
                    expected_result_type=RoundLocalResultType.TRAINING,
                    timeout_seconds=per_round_timeout,
                )
                if (
                    not isinstance(lower_global.contract, RoundGlobalArtifact)
                    or lower_global.contract.fl_metadata.ml_corre_id
                    != execution.process_id
                    or lower_global.contract.fl_metadata.round_ind
                    != current_lower_round
                ):
                    raise RuntimeError(
                        "Branch lower result does not match the mapped lower round"
                    )
                if offset + 1 < lower_round_count:
                    current_input = self._loader.load(
                        ArtifactMetadata(
                            key=lower_global.digest,
                            size_bytes=lower_global.path.stat().st_size,
                            path=lower_global.path,
                            url=lower_global.url,
                        )
                    )
            assert lower_global is not None
            with self._condition:
                self._ensure_dispatch_active(metadata.plan_id)
                current = self._rounds.get(key)
                if current is None:
                    raise RuntimeError("Branch round mapping disappeared")
                upper_result = self._artifact_service.publish_hierarchy_aggregate(
                    upper_input=upper_input,
                    lower_global=lower_global,
                    plan_id=execution.plan_id,
                    upper_process_id=upper_process_id,
                    branch_nf_instance_id=context.nf_instance_id,
                    upper_round_indicator=upper_round,
                    upper_training_scope=upper_training_scope,
                )
                self._rounds[key] = replace(
                    current,
                    state=BranchRoundState.COMPLETE,
                    upper_result=upper_result,
                )
                self._condition.notify_all()
            return upper_result
        except Exception as error:
            with self._condition:
                current = self._rounds.get(key)
                if current is not None:
                    self._rounds[key] = replace(
                        current,
                        state=BranchRoundState.FAILED,
                        failure=str(error),
                    )
                self._condition.notify_all()
            raise

    def execute_validation(
        self,
        *,
        assignment: ValidatedHierarchyArtifact,
        representation: NwdafMLModelTrainSubsc,
        upper_candidate: LoadedBundle,
        upper_candidate_artifact: ArtifactMetadata,
        upper_client_subscription_id: str,
        upper_resource_revision: int,
        upper_training_scope: TrainingScopeDescriptor,
        callback_margin_seconds: int,
    ) -> FLWorkspaceArtifact:
        metadata = assignment.contract.hierarchy_metadata
        if not isinstance(metadata, BranchAssignmentMetadata):
            raise ValueError("Branch validation requires a BRANCH_ASSIGNMENT")
        upper_process_id = representation.ml_correlation_id or ""
        upper_round = representation.round_indicator
        if not upper_process_id or upper_round is None or upper_round <= 0:
            raise ValueError("Branch validation requires upper process and round identities")
        if not upper_client_subscription_id or upper_resource_revision <= 0:
            raise ValueError("Branch validation requires upper resource identity")
        if len(upper_candidate_artifact.key) != 64:
            raise ValueError("Branch validation requires the candidate archive digest")
        report = representation.ml_training_report_info
        parent_budget = report.maximum_response_time if report is not None else None
        if parent_budget is None or parent_budget <= callback_margin_seconds:
            raise RuntimeError("Branch upper validation budget cannot contain lower execution")
        context = self._nwdaf_context.get()
        if context.nf_instance_id != metadata.intended_recipient_nf_instance_id:
            raise RuntimeError("Branch assignment recipient no longer matches local NWDAF")
        key = (metadata.plan_id, upper_process_id, upper_round)
        conflicting = False
        lower_round = -1
        with self._condition:
            self._ensure_dispatch_active(metadata.plan_id)
            preparation = self._executions.get(metadata.plan_id)
            if preparation is None:
                raise RuntimeError("Branch lower process is unavailable")
            while True:
                existing = self._validations.get(key)
                if existing is None:
                    lower_round = self._next_lower_round.get(metadata.plan_id, 0)
                    self._next_lower_round[metadata.plan_id] = lower_round + 1
                    self._validations[key] = BranchValidationExecution(
                        plan_id=metadata.plan_id,
                        upper_client_subscription_id=upper_client_subscription_id,
                        upper_resource_revision=upper_resource_revision,
                        upper_ml_corre_id=upper_process_id,
                        upper_round_indicator=upper_round,
                        upper_candidate_artifact_digest=upper_candidate_artifact.key,
                        upper_training_scope=upper_training_scope,
                        lower_server_process_id=preparation.process_id,
                        lower_validation_round=lower_round,
                    )
                    break
                if not _same_validation_command(
                    existing,
                    upper_client_subscription_id=upper_client_subscription_id,
                    upper_resource_revision=upper_resource_revision,
                    upper_candidate_artifact_digest=upper_candidate_artifact.key,
                    upper_training_scope=upper_training_scope,
                ):
                    conflicting = True
                    break
                if existing.state is BranchRoundState.COMPLETE:
                    if existing.upper_result is None:
                        raise RuntimeError("completed Branch validation has no upper result")
                    return existing.upper_result
                if existing.state is BranchRoundState.FAILED:
                    raise RuntimeError(
                        existing.failure or "Branch validation execution failed"
                    )
                self._condition.wait()
                self._ensure_dispatch_active(metadata.plan_id)
        if conflicting:
            failure = "conflicting duplicate Branch upper validation command"
            self.cancel(metadata.plan_id, failure)
            raise RuntimeError(failure)

        try:
            republished = self._artifact_service.republish_validation_candidate(
                source=upper_candidate_artifact,
                plan_id=metadata.plan_id,
                containing_branch_nf_instance_id=context.nf_instance_id,
                validation_round_indicator=lower_round,
            )
            with self._condition:
                self._ensure_dispatch_active(metadata.plan_id)
                current = self._validations.get(key)
                if current is None:
                    raise RuntimeError("Branch validation mapping disappeared")
                self._validations[key] = replace(
                    current,
                    republished_candidate=republished,
                )
            collection: HierarchyValidationCollection = (
                self._server.execute_hierarchy_validation(
                    process_id=preparation.process_id,
                    validation_round=lower_round,
                    candidate=republished,
                    base_artifact=assignment.metadata,
                    expected_candidate_process_id=upper_process_id,
                    expected_candidate_round=upper_round - 1,
                    timeout_seconds=parent_budget - callback_margin_seconds,
                )
            )
            with self._condition:
                self._ensure_dispatch_active(metadata.plan_id)
                current = self._validations.get(key)
                if current is None:
                    raise RuntimeError("Branch validation mapping disappeared")
                upper_result = self._artifact_service.publish_hierarchy_validation_result(
                    upper_candidate=upper_candidate,
                    plan_id=metadata.plan_id,
                    upper_process_id=upper_process_id,
                    branch_nf_instance_id=context.nf_instance_id,
                    upper_round_indicator=upper_round,
                    upper_training_scope=upper_training_scope,
                    subordinate_summaries=collection.validation_summaries,
                )
                self._validations[key] = replace(
                    current,
                    state=BranchRoundState.COMPLETE,
                    upper_result=upper_result,
                )
                self._condition.notify_all()
            return upper_result
        except Exception as error:
            with self._condition:
                current = self._validations.get(key)
                if current is not None:
                    self._validations[key] = replace(
                        current,
                        state=BranchRoundState.FAILED,
                        failure=str(error),
                    )
                self._condition.notify_all()
            raise

    @staticmethod
    def _classify(
        execution: BranchPreparationExecution,
        collection: HierarchyPreparationCollection,
    ) -> tuple[tuple[PreparedClient, ...], tuple[FailedClient, ...], tuple[str, ...]]:
        expected_urls = {
            node.nf_instance_id: artifact.url
            for node, artifact in zip(
                execution.leaf_nodes,
                execution.leaf_assignments,
                strict=True,
            )
        }
        timed_out = set(collection.timed_out_participant_nf_instance_ids)
        expected_event = execution.parent_assignment.manifest.get("analytics_event")
        prepared = []
        failed = []
        for participant in collection.participants:
            participant_id = participant.participant_nf_instance_id
            if participant_id in timed_out:
                continue
            notification = participant.notification
            if participant.failure:
                cause = PreparationFailureCause.INVALID_ASSIGNMENT
            elif notification is not None and notification.termination_request:
                cause = PreparationFailureCause.NOT_AVAILABLE_ML_TRAIN
            elif (
                notification is not None
                and len(notification.ml_model_infos or ()) == 1
                and notification.ml_model_infos[0].event == expected_event
                and participant.assignment_url == expected_urls.get(participant_id)
            ):
                prepared.append(PreparedClient(nf_instance_id=participant_id))
                continue
            else:
                cause = PreparationFailureCause.INVALID_ASSIGNMENT
            failed.append(FailedClient(nf_instance_id=participant_id, cause=cause))
        return (
            tuple(sorted(prepared, key=lambda item: item.nf_instance_id)),
            tuple(sorted(failed, key=lambda item: item.nf_instance_id)),
            tuple(sorted(timed_out)),
        )

    def _ensure_dispatch_active(self, plan_id: str) -> None:
        with self._lock:
            self._prune_cancelled_locked()
            if self._closing:
                raise BranchPreparationCancelled(
                    "Branch preparation coordinator is closing"
                )
            if plan_id in self._cancelled_plan_ids:
                raise BranchPreparationCancelled("Branch preparation was cancelled")

    def cancel(self, plan_id: str, reason: str) -> None:
        with self._condition:
            self._prune_cancelled_locked()
            self._cancelled_plan_ids[plan_id] = (
                self._clock() + self._tombstone_ttl_seconds
            )
            execution = self._executions.pop(plan_id, None)
            protocol_execution = self._protocol_executions.pop(plan_id, None)
            self._rounds = {
                key: value for key, value in self._rounds.items() if key[0] != plan_id
            }
            self._validations = {
                key: value
                for key, value in self._validations.items()
                if key[0] != plan_id
            }
            self._next_lower_round.pop(plan_id, None)
            self._condition.notify_all()
        if execution is not None:
            self._server.cancel_hierarchy_preparation(execution.process_id, reason)
        if protocol_execution is not None:
            self._server.cancel_hierarchy_preparation(
                protocol_execution.process_id,
                reason,
            )

    def close(self) -> None:
        with self._condition:
            self._closing = True
            executions = tuple(self._executions.values())
            protocol_executions = tuple(self._protocol_executions.values())
            deadline = self._clock() + self._tombstone_ttl_seconds
            self._cancelled_plan_ids.update(
                {plan_id: deadline for plan_id in self._executions}
            )
            self._executions.clear()
            self._protocol_executions.clear()
            self._rounds.clear()
            self._validations.clear()
            self._next_lower_round.clear()
            self._condition.notify_all()
        for execution in executions:
            self._server.cancel_hierarchy_preparation(
                execution.process_id,
                "Branch preparation coordinator is closing",
            )
        for execution in protocol_executions:
            self._server.cancel_hierarchy_preparation(
                execution.process_id,
                "Branch preparation coordinator is closing",
            )
        self._resolver.close()

    def abort_generation(self, reason: str) -> None:
        """Discard paired upper/lower state without closing the coordinator."""
        with self._condition:
            executions = tuple(self._executions.values())
            protocol_executions = tuple(self._protocol_executions.values())
            self._executions.clear()
            self._protocol_executions.clear()
            self._rounds.clear()
            self._validations.clear()
            self._next_lower_round.clear()
            self._cancelled_plan_ids.clear()
            self._condition.notify_all()
        for execution in executions:
            try:
                self._server.cancel_hierarchy_preparation(execution.process_id, reason)
            except RuntimeError:
                # Continue clearing every local mapping; the app-level Server abort
                # retries process cleanup after the Branch ownership pass.
                continue
        for execution in protocol_executions:
            try:
                self._server.cancel_hierarchy_preparation(execution.process_id, reason)
            except RuntimeError:
                continue

    def _prune_cancelled_locked(self) -> None:
        now = self._clock()
        self._cancelled_plan_ids = {
            plan_id: deadline
            for plan_id, deadline in self._cancelled_plan_ids.items()
            if deadline > now
        }


def _round_epochs(bundle: LoadedBundle) -> int:
    metadata = bundle.manifest.get("fl_metadata")
    directive = metadata.get("client_training") if isinstance(metadata, dict) else None
    epochs = directive.get("epochs") if isinstance(directive, dict) else None
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("ROUND_INPUT client training epochs are invalid")
    return epochs


def _same_round_command(
    execution: BranchRoundExecution,
    *,
    upper_client_subscription_id: str,
    upper_resource_revision: int,
    upper_input_artifact_digest: str,
    upper_training_scope: TrainingScopeDescriptor,
) -> bool:
    return (
        execution.upper_client_subscription_id == upper_client_subscription_id
        and execution.upper_resource_revision == upper_resource_revision
        and execution.upper_input_artifact_digest == upper_input_artifact_digest
        and execution.upper_training_scope == upper_training_scope
    )


def _same_validation_command(
    execution: BranchValidationExecution,
    *,
    upper_client_subscription_id: str,
    upper_resource_revision: int,
    upper_candidate_artifact_digest: str,
    upper_training_scope: TrainingScopeDescriptor,
) -> bool:
    return (
        execution.upper_client_subscription_id == upper_client_subscription_id
        and execution.upper_resource_revision == upper_resource_revision
        and execution.upper_candidate_artifact_digest
        == upper_candidate_artifact_digest
        and execution.upper_training_scope == upper_training_scope
    )
