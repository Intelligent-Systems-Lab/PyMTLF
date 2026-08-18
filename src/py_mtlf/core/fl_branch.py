from __future__ import annotations

import threading
from dataclasses import dataclass

from py_mtlf.core.fl_artifacts import HierarchyAssignmentArtifact
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
)
from py_mtlf.core.fl_workspace import FLWorkspaceArtifact, ValidatedHierarchyArtifact
from py_mtlf.core.nwdaf_context import FLCapabilityType, NwdafContextClient
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainSubsc


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
    ) -> None:
        self._resolver = resolver
        self._nwdaf_context = nwdaf_context
        self._artifact_service = artifact_service
        self._server = server
        self._lock = threading.RLock()
        self._executions: dict[str, BranchPreparationExecution] = {}
        self._cancelled_plan_ids: set[str] = set()
        self._closing = False

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
        context = self._nwdaf_context.get(refresh=True)
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
        return BranchPreparationResult(
            artifact=artifact,
            outcome=outcome,
            execution=execution,
        )

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
            if self._closing:
                raise BranchPreparationCancelled(
                    "Branch preparation coordinator is closing"
                )
            if plan_id in self._cancelled_plan_ids:
                raise BranchPreparationCancelled("Branch preparation was cancelled")

    def cancel(self, plan_id: str, reason: str) -> None:
        with self._lock:
            self._cancelled_plan_ids.add(plan_id)
            execution = self._executions.pop(plan_id, None)
        if execution is not None:
            self._server.cancel_hierarchy_preparation(execution.process_id, reason)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            executions = tuple(self._executions.values())
            self._cancelled_plan_ids.update(self._executions)
            self._executions.clear()
        for execution in executions:
            self._server.cancel_hierarchy_preparation(
                execution.process_id,
                "Branch preparation coordinator is closing",
            )
        self._resolver.close()
