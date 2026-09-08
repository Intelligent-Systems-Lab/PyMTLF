from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.experiment_recording import ExperimentRecorder
from py_mtlf.core.fl_artifacts import (
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
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyNodeResolver,
    HierarchyNodeRole,
)
from py_mtlf.core.fl_server import (
    FLClientCandidate,
    FLServerEngine,
    ProtocolPreparationTarget,
)
from py_mtlf.core.fl_workspace import FLWorkspaceArtifact
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.ml_model import MLEventNotification
from py_mtlf.wire.ml_model_training import (
    FlReportAfter,
    FlTopologyNode,
    FlTopologyReport,
    NwdafMLModelTrainSubsc,
)

logger = logging.getLogger(__name__)


@dataclass
class ProtocolBranchExecution:
    ml_correlation_id: str
    process_id: str
    candidate_pool: CandidatePool


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
        experiment_recorder: ExperimentRecorder | None = None,
        tombstone_ttl_seconds: int = 3600,
        clock=time.monotonic,
    ) -> None:
        if tombstone_ttl_seconds <= 0:
            raise ValueError("tombstone_ttl_seconds must be positive")
        self._resolver = resolver
        self._nwdaf_context = nwdaf_context
        self._artifact_service = artifact_service
        self._server = server
        self._experiment_recorder = experiment_recorder
        self._loader = TrustedBundleLoader()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._protocol_executions: dict[str, ProtocolBranchExecution] = {}
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
                    failure = self._server.remove_protocol_participant(
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
                    if failure:
                        logger.warning(
                            "Protocol participant cleanup remains pending nf=%s error=%s",
                            intent.nf_instance_id,
                            failure,
                        )
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
            outcome = self._server.execute_hierarchy_round(
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
            if not outcome.accepted or outcome.aggregate is None:
                raise RuntimeError("protocol Branch lower round did not reach completion policy")
            lower_global = outcome.aggregate
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
            if (
                self._experiment_recorder is not None
                and self._experiment_recorder.validation_enabled
            ):
                self._experiment_recorder.record_model_evaluation(
                    ml_correlation_id=ml_correlation_id,
                    evaluation_stage="BRANCH_DOMAIN",
                    round_indicator=current_lower_round,
                    model=current_input.model,
                    manifest=current_input.manifest,
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
            protocol_execution = self._protocol_executions.pop(plan_id, None)
            self._next_lower_round.pop(plan_id, None)
            self._condition.notify_all()
        if protocol_execution is not None:
            self._server.cancel_hierarchy_preparation(
                protocol_execution.process_id,
                reason,
            )

    def close(self) -> None:
        with self._condition:
            self._closing = True
            protocol_executions = tuple(self._protocol_executions.values())
            deadline = self._clock() + self._tombstone_ttl_seconds
            self._cancelled_plan_ids.update(
                {plan_id: deadline for plan_id in self._protocol_executions}
            )
            self._protocol_executions.clear()
            self._next_lower_round.clear()
            self._condition.notify_all()
        for execution in protocol_executions:
            self._server.cancel_hierarchy_preparation(
                execution.process_id,
                "Branch preparation coordinator is closing",
            )
        self._resolver.close()

    def abort_generation(self, reason: str) -> None:
        """Discard paired upper/lower state without closing the coordinator."""
        with self._condition:
            protocol_executions = tuple(self._protocol_executions.values())
            self._protocol_executions.clear()
            self._next_lower_round.clear()
            self._cancelled_plan_ids.clear()
            self._condition.notify_all()
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
