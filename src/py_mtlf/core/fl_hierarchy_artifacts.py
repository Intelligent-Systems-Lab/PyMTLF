from __future__ import annotations

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    HierarchyAssignmentArtifact,
    RoundGlobalArtifact,
    ValidationSummary,
)
from py_mtlf.core.fl_hierarchy import (
    BranchAssignmentMetadata,
    CompleteRequiredAdmission,
    FailedClient,
    FederatedStrategy,
    HierarchyMessageType,
    LeafAssignmentMetadata,
    PreparationOutcome,
    PreparationResultMetadata,
    PreparedClient,
    normalize_nf_instance_id,
)
from py_mtlf.core.fl_workspace import (
    FLArtifactContractError,
    FLWorkspace,
    FLWorkspaceArtifact,
    ValidatedHierarchyArtifact,
    validate_model_compatibility,
)
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor

type HierarchyArtifactSource = FLWorkspaceArtifact | ValidatedHierarchyArtifact


class HierarchyArtifactOperationError(ValueError):
    pass


class HierarchyArtifactService:
    def __init__(
        self,
        workspace: FLWorkspace,
        loader: TrustedBundleLoader | None = None,
    ) -> None:
        self._workspace = workspace
        self._loader = loader or TrustedBundleLoader()

    def publish_branch_assignment(
        self,
        *,
        base: LoadedBundle,
        plan_id: str,
        publisher_nf_instance_id: str,
        branch_nf_instance_id: str,
        assigned_leaf_nf_instance_ids: tuple[str, ...],
        strategy: FederatedStrategy,
    ) -> FLWorkspaceArtifact:
        metadata = BranchAssignmentMetadata(
            message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
            plan_id=plan_id,
            publisher_nf_instance_id=publisher_nf_instance_id,
            intended_recipient_nf_instance_id=branch_nf_instance_id,
            assigned_leaf_nf_instance_ids=assigned_leaf_nf_instance_ids,
            admission=CompleteRequiredAdmission(mode="complete_required"),
            strategy=strategy,
        )
        return self._publish(base, metadata, ArtifactRole.HIERARCHY_ASSIGNMENT)

    def publish_round_input(
        self,
        *,
        plan_id: str,
        base: LoadedBundle,
        process_id: str,
        server_nf_instance_id: str,
        round_indicator: int,
        epochs: int,
    ) -> FLWorkspaceArtifact:
        return self._workspace.publish_round_input(
            process_id=process_id,
            server_nf_instance_id=server_nf_instance_id,
            round_indicator=round_indicator,
            base=base,
            epochs=epochs,
            owner_plan_id=plan_id,
        )

    def publish_hierarchy_aggregate(
        self,
        *,
        upper_input: LoadedBundle,
        lower_global: FLWorkspaceArtifact,
        plan_id: str,
        upper_process_id: str,
        branch_nf_instance_id: str,
        upper_round_indicator: int,
        upper_training_scope: TrainingScopeDescriptor,
    ) -> FLWorkspaceArtifact:
        if not isinstance(lower_global.contract, RoundGlobalArtifact):
            raise HierarchyArtifactOperationError(
                "lower result is not a ROUND_GLOBAL artifact"
            )
        lower_bundle = self._loader.load(
            ArtifactMetadata(
                key=lower_global.digest,
                size_bytes=lower_global.path.stat().st_size,
                path=lower_global.path,
                url=lower_global.url,
            )
        )

        lower_metadata = lower_global.contract.fl_metadata
        try:
            validate_model_compatibility(upper_input, lower_bundle)
        except FLArtifactContractError as error:
            raise HierarchyArtifactOperationError(
                "lower ROUND_GLOBAL does not match the upper round input"
            ) from error
        return self._workspace.publish(
            process_id=upper_process_id,
            participant_id=branch_nf_instance_id,
            round_indicator=upper_round_indicator,
            role="ROUND_LOCAL",
            base=upper_input,
            model=lower_bundle.model,
            owner_plan_id=plan_id,
            metadata={
                "artifact_role": "ROUND_LOCAL",
                "result_type": "HIERARCHY_AGGREGATE",
                "fl_metadata": {
                    "ml_corre_id": upper_process_id,
                    "round_ind": upper_round_indicator,
                    "participant_nf_instance_id": branch_nf_instance_id,
                    "training_scope": upper_training_scope.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "training_sample_count": lower_metadata.aggregated_training_sample_count,
                    "lower_round_ind": lower_metadata.round_ind,
                    "lower_global_artifact_digest": lower_global.digest,
                    "subordinate_participants": [
                        item.model_dump(mode="python")
                        for item in lower_metadata.participants
                    ],
                },
            },
        )

    def republish_validation_candidate(
        self,
        *,
        source: ArtifactMetadata,
        plan_id: str,
        containing_branch_nf_instance_id: str,
        validation_round_indicator: int,
    ) -> FLWorkspaceArtifact:
        return self._workspace.republish_validation_candidate(
            source=source,
            plan_id=plan_id,
            participant_id=containing_branch_nf_instance_id,
            round_indicator=validation_round_indicator,
        )

    def publish_hierarchy_validation_result(
        self,
        *,
        upper_candidate: LoadedBundle,
        plan_id: str,
        upper_process_id: str,
        branch_nf_instance_id: str,
        upper_round_indicator: int,
        upper_training_scope: TrainingScopeDescriptor,
        subordinate_summaries: tuple[ValidationSummary, ...],
    ) -> FLWorkspaceArtifact:
        if not subordinate_summaries:
            raise HierarchyArtifactOperationError(
                "hierarchy validation result requires subordinate evidence"
            )
        return self._workspace.publish(
            process_id=upper_process_id,
            participant_id=branch_nf_instance_id,
            round_indicator=upper_round_indicator,
            role="ROUND_LOCAL",
            base=upper_candidate,
            model=upper_candidate.model,
            owner_plan_id=plan_id,
            metadata={
                "artifact_role": "ROUND_LOCAL",
                "result_type": "ACCURACY_CHECK",
                "fl_metadata": {
                    "ml_corre_id": upper_process_id,
                    "round_ind": upper_round_indicator,
                    "participant_nf_instance_id": branch_nf_instance_id,
                    "training_scope": upper_training_scope.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "evaluation": {
                        "evaluation_stage": "FINAL_VALIDATION",
                        "evaluation_sample_count": sum(
                            item.evaluation_sample_count
                            for item in subordinate_summaries
                        ),
                        "start_time": min(
                            item.start_time for item in subordinate_summaries
                        ).isoformat(),
                        "end_time": max(
                            item.end_time for item in subordinate_summaries
                        ).isoformat(),
                        "base": {
                            "absolute_error_sum": sum(
                                item.base.absolute_error_sum
                                for item in subordinate_summaries
                            ),
                            "absolute_actual_sum": sum(
                                item.base.absolute_actual_sum
                                for item in subordinate_summaries
                            ),
                        },
                        "candidate": {
                            "absolute_error_sum": sum(
                                item.candidate.absolute_error_sum
                                for item in subordinate_summaries
                            ),
                            "absolute_actual_sum": sum(
                                item.candidate.absolute_actual_sum
                                for item in subordinate_summaries
                            ),
                        },
                    },
                    "subordinate_validation_summaries": [
                        item.model_dump(mode="json")
                        for item in subordinate_summaries
                    ],
                },
            },
        )

    def republish_leaf_assignment(
        self,
        *,
        parent: HierarchyArtifactSource,
        containing_branch_nf_instance_id: str,
        leaf_nf_instance_id: str,
    ) -> FLWorkspaceArtifact:
        assignment = _branch_assignment(parent)
        branch_id = normalize_nf_instance_id(containing_branch_nf_instance_id)
        leaf_id = normalize_nf_instance_id(leaf_nf_instance_id)
        if assignment.intended_recipient_nf_instance_id != branch_id:
            raise HierarchyArtifactOperationError(
                "containing Branch does not match assignment recipient"
            )
        if leaf_id not in assignment.assigned_leaf_nf_instance_ids:
            raise HierarchyArtifactOperationError(
                "Leaf is not present in the parent Branch assignment"
            )

        base = self._loader.load(_artifact_metadata(parent))
        metadata = LeafAssignmentMetadata(
            message_type=HierarchyMessageType.LEAF_ASSIGNMENT,
            plan_id=assignment.plan_id,
            publisher_nf_instance_id=branch_id,
            intended_recipient_nf_instance_id=leaf_id,
            parent_branch_nf_instance_id=branch_id,
            strategy=assignment.strategy,
        )
        return self._publish(base, metadata, ArtifactRole.HIERARCHY_ASSIGNMENT)

    def publish_preparation_result(
        self,
        *,
        parent: HierarchyArtifactSource,
        containing_branch_nf_instance_id: str,
        outcome: PreparationOutcome,
        prepared_clients: tuple[PreparedClient, ...],
        failed_clients: tuple[FailedClient, ...],
        timed_out_client_nf_instance_ids: tuple[str, ...],
    ) -> FLWorkspaceArtifact:
        assignment = _branch_assignment(parent)
        branch_id = normalize_nf_instance_id(containing_branch_nf_instance_id)
        if assignment.intended_recipient_nf_instance_id != branch_id:
            raise HierarchyArtifactOperationError(
                "containing Branch does not match assignment recipient"
            )

        base = self._loader.load(_artifact_metadata(parent))
        metadata = PreparationResultMetadata(
            message_type=HierarchyMessageType.PREPARATION_RESULT,
            plan_id=assignment.plan_id,
            publisher_nf_instance_id=branch_id,
            intended_recipient_nf_instance_id=assignment.publisher_nf_instance_id,
            outcome=outcome,
            assigned_client_nf_instance_ids=assignment.assigned_leaf_nf_instance_ids,
            prepared_clients=prepared_clients,
            failed_clients=failed_clients,
            timed_out_client_nf_instance_ids=timed_out_client_nf_instance_ids,
        )
        return self._publish(
            base,
            metadata,
            ArtifactRole.HIERARCHY_PREPARATION_RESULT,
        )

    def _publish(
        self,
        base: LoadedBundle,
        metadata: BranchAssignmentMetadata
        | LeafAssignmentMetadata
        | PreparationResultMetadata,
        role: ArtifactRole,
    ) -> FLWorkspaceArtifact:
        return self._workspace.publish(
            process_id=metadata.plan_id,
            participant_id=metadata.intended_recipient_nf_instance_id,
            round_indicator=0,
            role=role.value,
            base=base,
            model=base.model,
            owner_plan_id=metadata.plan_id,
            metadata={
                "artifact_role": role.value,
                "hierarchy_metadata": metadata.model_dump(mode="json"),
            },
        )


def _branch_assignment(source: HierarchyArtifactSource) -> BranchAssignmentMetadata:
    if not isinstance(source, (FLWorkspaceArtifact, ValidatedHierarchyArtifact)):
        raise HierarchyArtifactOperationError(
            "parent artifact must be a validated hierarchy artifact"
        )
    contract = source.contract
    if not isinstance(contract, HierarchyAssignmentArtifact) or not isinstance(
        contract.hierarchy_metadata,
        BranchAssignmentMetadata,
    ):
        raise HierarchyArtifactOperationError(
            "parent artifact must be a BRANCH_ASSIGNMENT"
        )
    return contract.hierarchy_metadata


def _artifact_metadata(source: HierarchyArtifactSource) -> ArtifactMetadata:
    if isinstance(source, ValidatedHierarchyArtifact):
        return source.metadata
    return ArtifactMetadata(
        key=source.digest,
        size_bytes=source.path.stat().st_size,
        path=source.path,
        url=source.url,
    )
