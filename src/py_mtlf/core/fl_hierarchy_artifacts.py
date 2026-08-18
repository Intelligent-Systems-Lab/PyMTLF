from __future__ import annotations

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import ArtifactRole, HierarchyAssignmentArtifact
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
    FLWorkspace,
    FLWorkspaceArtifact,
    ValidatedHierarchyArtifact,
)
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader

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
            contract_version="1.0",
            message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
            plan_id=plan_id,
            publisher_nf_instance_id=publisher_nf_instance_id,
            intended_recipient_nf_instance_id=branch_nf_instance_id,
            assigned_leaf_nf_instance_ids=assigned_leaf_nf_instance_ids,
            admission=CompleteRequiredAdmission(mode="complete_required"),
            strategy=strategy,
        )
        return self._publish(base, metadata, ArtifactRole.HIERARCHY_ASSIGNMENT)

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
            contract_version="1.0",
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
            contract_version="1.0",
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
