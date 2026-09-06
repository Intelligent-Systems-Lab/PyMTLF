from __future__ import annotations

from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import RoundGlobalArtifact
from py_mtlf.core.fl_workspace import (
    FLArtifactContractError,
    FLWorkspace,
    FLWorkspaceArtifact,
    validate_model_compatibility,
)
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor


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
