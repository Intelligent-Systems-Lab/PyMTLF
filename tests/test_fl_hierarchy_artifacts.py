import hashlib
import io
import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import joblib
import numpy as np
import pytest
import torch
from conftest import build_bundle
from sklearn.preprocessing import StandardScaler

from py_mtlf.config import (
    ArtifactDownloadSettings,
    ArtifactSettings,
    FederatedLearningSettings,
    FLClientSettings,
    NotificationSettings,
)
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    HierarchyAssignmentArtifact,
    HierarchyPreparationResultArtifact,
    RoundGlobalArtifact,
    RoundInputArtifact,
    RoundLocalArtifact,
    ValidationSummary,
    WapeComponents,
)
from py_mtlf.core.fl_client import FLClientEngine, FLClientResource, FLClientState
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy import (
    FailedClient,
    FederatedStrategy,
    HierarchyMessageType,
    PreparationOutcome,
    PreparedClient,
)
from py_mtlf.core.fl_hierarchy_artifacts import (
    HierarchyArtifactOperationError,
    HierarchyArtifactService,
)
from py_mtlf.core.fl_workspace import (
    FLArtifactContractError,
    FLArtifactIdentityError,
    FLArtifactIntegrityError,
    FLArtifactUnavailableError,
    FLWorkspace,
    FLWorkspaceError,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.nwdaf_context import (
    FLCapabilityType,
    MLAnalyticsCapability,
    NwdafContext,
)
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.ml_model_training import NwdafMLModelTrainSubsc

ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEAF_A = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEAF_B = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PLAN = "11111111-1111-4111-8111-111111111111"
PLAN_B = "22222222-2222-4222-8222-222222222222"
BRANCH_B = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"

MODEL_SOURCE = b"""
import torch

class Model(torch.nn.Module):
    def __init__(self, input_size=2, output_size=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, value):
        return self.linear(value)
"""


class RuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, value):
        return self.linear(value)


def strategy() -> FederatedStrategy:
    return FederatedStrategy(
        algorithm={"name": "fedprox", "proximal_mu": 0.01},
        participant_selection="all",
        waiting_policy="all",
        aggregation="sample_weighted",
    )


def base_bundle() -> LoadedBundle:
    torch.manual_seed(7)
    model = RuntimeModel()
    scaler = StandardScaler().fit(np.arange(20).reshape(10, 2))
    scaler_stream = io.BytesIO()
    joblib.dump(scaler, scaler_stream)
    return LoadedBundle(
        manifest={
            "bundle_schema_version": "1.0",
            "model_identity": {"model_unique_id": 1},
            "analytics_event": "UE_COMMUNICATION",
            "model_interoperability": "001122",
            "runtime_compatibility": {"python": ">=3.12", "framework": "torch"},
            "MODEL_SCRIPT": "model.py",
            "MODEL_PATH": "model.npy",
            "SCALER_PATH": "scaler.pkl",
            "model": {"input_size": 2, "output_size": 1},
            "inference": {
                "feature_order": ["uplink", "downlink"],
                "output_fields": ["prediction"],
            },
        },
        model=model,
        scaler=scaler,
        model_source=MODEL_SOURCE,
        scaler_source=scaler_stream.getvalue(),
    )


def workspace(
    root,
    public_base_url: str,
    *,
    client: httpx.Client | None = None,
    allowed_origins: tuple[str, ...] = (),
) -> FLWorkspace:
    value = FLWorkspace(
        FederatedLearningSettings(
            workspace_root=root,
            public_base_url=public_base_url,
            artifact_download=ArtifactDownloadSettings(allowed_origins=allowed_origins),
        ),
        ArtifactSettings(),
        client,
    )
    value.open()
    return value


def publish_root_assignment(root_workspace: FLWorkspace, plan_id: str = PLAN):
    return HierarchyArtifactService(root_workspace).publish_branch_assignment(
        base=base_bundle(),
        plan_id=plan_id,
        publisher_nf_instance_id=ROOT,
        branch_nf_instance_id=BRANCH,
        assigned_leaf_nf_instance_ids=(LEAF_A, LEAF_B),
        strategy=strategy(),
    )


def metadata(artifact) -> ArtifactMetadata:
    return ArtifactMetadata(
        key=artifact.digest,
        size_bytes=artifact.path.stat().st_size,
        path=artifact.path,
        url=artifact.url,
    )


def test_round_input_and_hierarchy_aggregate_preserve_training_contract(tmp_path) -> None:
    branch_workspace = workspace(tmp_path / "branch", "http://branch.example")
    try:
        service = HierarchyArtifactService(branch_workspace)
        upper = base_bundle()
        lower_input = service.publish_round_input(
            plan_id=PLAN,
            base=upper,
            process_id="lower-process",
            server_nf_instance_id=BRANCH,
            round_indicator=4,
            epochs=7,
        )
        assert isinstance(lower_input.contract, RoundInputArtifact)
        assert lower_input.contract.fl_metadata.client_training.epochs == 7

        base_digest = weights_digest(upper.model)
        lower_global = branch_workspace.publish(
            process_id="lower-process",
            participant_id=BRANCH,
            round_indicator=4,
            role="ROUND_GLOBAL",
            base=upper,
            model=upper.model,
            metadata={
                "artifact_role": "ROUND_GLOBAL",
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": "lower-process",
                    "round_ind": 4,
                    "model_contract_digest": model_contract_digest(upper.manifest),
                    "preprocessing_contract_digest": preprocessing_contract_digest(
                        upper.manifest
                    ),
                    "base_weights_digest": base_digest,
                    "weights_digest": base_digest,
                    "participants": [
                        {
                            "participant_nf_instance_id": LEAF_A,
                            "training_sample_count": 12,
                            "local_artifact_digest": "1" * 64,
                        },
                        {
                            "participant_nf_instance_id": LEAF_B,
                            "training_sample_count": 8,
                            "local_artifact_digest": "2" * 64,
                        },
                    ],
                    "aggregated_training_sample_count": 20,
                },
            },
        )
        upper_result = service.publish_hierarchy_aggregate(
            upper_input=upper,
            lower_global=lower_global,
            plan_id=PLAN,
            upper_process_id="upper-process",
            branch_nf_instance_id=BRANCH,
            upper_round_indicator=2,
            upper_scope_digest="3" * 64,
        )
        assert isinstance(upper_result.contract, RoundLocalArtifact)
        assert upper_result.contract.result_type == "HIERARCHY_AGGREGATE"
        assert upper_result.contract.fl_metadata.training_sample_count == 20
        assert tuple(
            item.participant_nf_instance_id
            for item in upper_result.contract.fl_metadata.subordinate_participants
        ) == (LEAF_A, LEAF_B)

        with torch.no_grad():
            upper.model.linear.weight.add_(1.0)
        with pytest.raises(
            HierarchyArtifactOperationError,
            match="does not match the upper round input",
        ):
            service.publish_hierarchy_aggregate(
                upper_input=upper,
                lower_global=lower_global,
                plan_id=PLAN,
                upper_process_id="upper-process",
                branch_nf_instance_id=BRANCH,
                upper_round_indicator=3,
                upper_scope_digest="3" * 64,
            )
    finally:
        branch_workspace.close()


def test_branch_republishes_validation_candidate_byte_identically_under_plan_owner(
    tmp_path,
) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    base = base_bundle()
    base_digest = weights_digest(base.model)
    candidate = root_workspace.publish(
        process_id="root-process",
        participant_id=ROOT,
        round_indicator=1,
        role="ROUND_GLOBAL",
        base=base,
        model=base.model,
        metadata={
            "artifact_role": "ROUND_GLOBAL",
            "fl_metadata": {
                "contract_version": "1.0",
                "ml_corre_id": "root-process",
                "round_ind": 1,
                "model_contract_digest": model_contract_digest(base.manifest),
                "preprocessing_contract_digest": preprocessing_contract_digest(
                    base.manifest
                ),
                "base_weights_digest": base_digest,
                "weights_digest": base_digest,
                "participants": [
                    {
                        "participant_nf_instance_id": BRANCH,
                        "training_sample_count": 20,
                        "local_artifact_digest": "1" * 64,
                    }
                ],
                "aggregated_training_sample_count": 20,
            },
        },
    )

    def serve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Artifact-SHA256": candidate.digest},
            content=candidate.path.read_bytes(),
            request=request,
        )

    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=httpx.Client(transport=httpx.MockTransport(serve)),
        allowed_origins=("http://root.example",),
    )
    try:
        downloaded = branch_workspace.download(
            candidate.url,
            "root-process",
            "validation-candidate",
        )
        republished = HierarchyArtifactService(
            branch_workspace
        ).republish_validation_candidate(
            source=downloaded,
            plan_id=PLAN,
            containing_branch_nf_instance_id=BRANCH,
            validation_round_indicator=2,
        )

        assert isinstance(republished.contract, RoundGlobalArtifact)
        assert republished.digest == candidate.digest
        assert republished.path.read_bytes() == candidate.path.read_bytes()
        assert republished.manifest == candidate.manifest
        assert republished.url != candidate.url
        assert republished.process_id == PLAN
        loaded_candidate = TrustedBundleLoader().load(metadata(republished))
        start = datetime(2026, 8, 20, tzinfo=UTC)
        summaries = tuple(
            ValidationSummary(
                participant_nf_instance_id=leaf_id,
                scope_digest=str(index) * 64,
                evaluation_sample_count=10,
                start_time=start,
                end_time=start + timedelta(minutes=1),
                base_model_weights_digest=base_digest,
                candidate_weights_digest=base_digest,
                base=WapeComponents(
                    absolute_error_sum=10,
                    absolute_actual_sum=100,
                ),
                candidate=WapeComponents(
                    absolute_error_sum=5,
                    absolute_actual_sum=100,
                ),
            )
            for index, leaf_id in enumerate((LEAF_A, LEAF_B), start=1)
        )
        result = HierarchyArtifactService(
            branch_workspace
        ).publish_hierarchy_validation_result(
            upper_candidate=loaded_candidate,
            plan_id=PLAN,
            upper_process_id="root-process",
            branch_nf_instance_id=BRANCH,
            upper_round_indicator=2,
            upper_scope_digest="3" * 64,
            subordinate_summaries=summaries,
        )
        assert isinstance(result.contract, RoundLocalArtifact)
        assert result.contract.result_type == "ACCURACY_CHECK"
        assert (
            result.contract.fl_metadata.subordinate_validation_summaries
            == summaries
        )
        assert result.contract.fl_metadata.evaluation.evaluation_sample_count == 20
        branch_workspace.release_plan(PLAN)
        assert branch_workspace.resolve(PLAN, BRANCH, 2, "ROUND_GLOBAL", candidate.digest) is None
    finally:
        branch_workspace.close()
        root_workspace.close()


def test_publish_republish_and_result_round_trip_preserves_model_contract(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    branch_workspace = workspace(tmp_path / "branch", "http://branch.example")
    try:
        root_assignment = publish_root_assignment(root_workspace)
        assert isinstance(root_assignment.contract, HierarchyAssignmentArtifact)
        assert root_assignment.url.startswith("http://root.example/")

        branch_service = HierarchyArtifactService(branch_workspace)
        leaf_assignment = branch_service.republish_leaf_assignment(
            parent=root_assignment,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_A,
        )
        assert leaf_assignment.url.startswith("http://branch.example/")
        assert leaf_assignment.url != root_assignment.url
        assert (
            leaf_assignment.contract.hierarchy_metadata.intended_recipient_nf_instance_id
            == LEAF_A
        )
        assert leaf_assignment.contract.file_digests == root_assignment.contract.file_digests
        assert "assigned_leaf_nf_instance_ids" not in leaf_assignment.manifest["hierarchy_metadata"]

        result = branch_service.publish_preparation_result(
            parent=root_assignment,
            containing_branch_nf_instance_id=BRANCH,
            outcome=PreparationOutcome.FAILED,
            prepared_clients=(PreparedClient(nf_instance_id=LEAF_A),),
            failed_clients=(
                FailedClient(nf_instance_id=LEAF_B, cause="REQUIREMENTS_NOT_MET"),
            ),
            timed_out_client_nf_instance_ids=(),
        )
        assert isinstance(result.contract, HierarchyPreparationResultArtifact)
        assert result.contract.hierarchy_metadata.outcome is PreparationOutcome.FAILED
        assert result.contract.file_digests == root_assignment.contract.file_digests

        loader = TrustedBundleLoader()
        root_model = loader.load(metadata(root_assignment))
        leaf_model = loader.load(metadata(leaf_assignment))
        result_model = loader.load(metadata(result))
        assert weights_digest(root_model.model) == weights_digest(leaf_model.model)
        assert weights_digest(root_model.model) == weights_digest(result_model.model)
        assert root_model.manifest["model"] == leaf_model.manifest["model"]
        assert root_model.manifest["inference"] == result_model.manifest["inference"]
    finally:
        root_workspace.close()
        branch_workspace.close()


def test_hierarchy_publication_is_recipient_specific_and_idempotent(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    branch_workspace = workspace(tmp_path / "branch", "http://branch.example")
    try:
        first = publish_root_assignment(root_workspace)
        repeated = publish_root_assignment(root_workspace)
        second_branch = HierarchyArtifactService(root_workspace).publish_branch_assignment(
            base=base_bundle(),
            plan_id=PLAN,
            publisher_nf_instance_id=ROOT,
            branch_nf_instance_id=BRANCH_B,
            assigned_leaf_nf_instance_ids=(LEAF_A, LEAF_B),
            strategy=strategy(),
        )

        assert repeated.digest == first.digest
        assert repeated.path == first.path
        assert repeated.path.read_bytes() == first.path.read_bytes()
        assert second_branch.digest != first.digest
        assert second_branch.url != first.url
        assert (
            second_branch.contract.hierarchy_metadata.intended_recipient_nf_instance_id
            == BRANCH_B
        )

        branch_service = HierarchyArtifactService(branch_workspace)
        leaf_a = branch_service.republish_leaf_assignment(
            parent=first,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_A,
        )
        repeated_leaf_a = branch_service.republish_leaf_assignment(
            parent=first,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_A,
        )
        leaf_b = branch_service.republish_leaf_assignment(
            parent=first,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_B,
        )

        assert repeated_leaf_a.digest == leaf_a.digest
        assert repeated_leaf_a.path == leaf_a.path
        assert leaf_b.digest != leaf_a.digest
        assert (
            leaf_b.contract.hierarchy_metadata.intended_recipient_nf_instance_id
            == LEAF_B
        )

        first.path.write_bytes(b"corrupted existing publication")
        with pytest.raises(FLArtifactIntegrityError, match="existing.*digest"):
            publish_root_assignment(root_workspace)
    finally:
        root_workspace.close()
        branch_workspace.close()


def test_hierarchy_publication_validates_bundle_before_workspace_write(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        invalid_base = replace(base_bundle(), scaler_source=b"")
        with pytest.raises(RuntimeError, match="scaler source"):
            HierarchyArtifactService(root_workspace).publish_branch_assignment(
                base=invalid_base,
                plan_id=PLAN,
                publisher_nf_instance_id=ROOT,
                branch_nf_instance_id=BRANCH,
                assigned_leaf_nf_instance_ids=(LEAF_A, LEAF_B),
                strategy=strategy(),
            )

        assert not (tmp_path / "root" / PLAN).exists()
    finally:
        root_workspace.close()


def test_republish_rejects_leaf_outside_parent_assignment(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    branch_workspace = workspace(tmp_path / "branch", "http://branch.example")
    try:
        root_assignment = publish_root_assignment(root_workspace)
        with pytest.raises(ValueError, match="not present"):
            HierarchyArtifactService(branch_workspace).republish_leaf_assignment(
                parent=root_assignment,
                containing_branch_nf_instance_id=BRANCH,
                leaf_nf_instance_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            )
        with pytest.raises(HierarchyArtifactOperationError, match="validated"):
            HierarchyArtifactService(branch_workspace).republish_leaf_assignment(
                parent=root_assignment.url,  # type: ignore[arg-type]
                containing_branch_nf_instance_id=BRANCH,
                leaf_nf_instance_id=LEAF_A,
            )
    finally:
        root_workspace.close()
        branch_workspace.close()


def test_hierarchy_download_binds_url_header_body_and_identity(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"X-Artifact-SHA256": root_assignment.digest},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("HTTP://ROOT.EXAMPLE",),
    )
    try:
        downloaded = branch_workspace.download_hierarchy(
            root_assignment.url,
            expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
            expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
            expected_publisher_nf_instance_id=ROOT,
            intended_recipient_nf_instance_id=BRANCH,
        )
        assert downloaded.metadata.path.parent == tmp_path / "branch" / PLAN / "downloads"
        assert downloaded.metadata.path.read_bytes() == content
        assert not tuple((tmp_path / "branch" / ".staging").iterdir())

        leaf_assignment = HierarchyArtifactService(branch_workspace).republish_leaf_assignment(
            parent=downloaded,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_A,
        )
        assert leaf_assignment.url.startswith("http://branch.example/")
        assert (
            leaf_assignment.contract.hierarchy_metadata.intended_recipient_nf_instance_id
            == LEAF_A
        )
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


def test_assignment_ingress_discovers_typed_message_without_expected_publisher(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"X-Artifact-SHA256": root_assignment.digest},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        downloaded = branch_workspace.download_assignment(
            root_assignment.url,
            intended_recipient_nf_instance_id=BRANCH,
        )

        assert isinstance(downloaded.contract, HierarchyAssignmentArtifact)
        assert downloaded.contract.hierarchy_metadata.message_type is (
            HierarchyMessageType.BRANCH_ASSIGNMENT
        )
        assert downloaded.contract.hierarchy_metadata.publisher_nf_instance_id == ROOT
        assert downloaded.contract.hierarchy_metadata.plan_id == PLAN
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


def test_branch_and_leaf_assignment_admission_each_fetches_once_and_adopts_same_bytes(
    tmp_path,
) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    root_content = root_assignment.path.read_bytes()
    root_requests = 0

    def root_handler(request: httpx.Request) -> httpx.Response:
        nonlocal root_requests
        root_requests += 1
        return httpx.Response(
            200,
            content=root_content,
            headers={"X-Artifact-SHA256": root_assignment.digest},
            request=request,
        )

    branch_client = httpx.Client(transport=httpx.MockTransport(root_handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=branch_client,
        allowed_origins=("http://root.example",),
    )
    leaf_workspace = None
    leaf_client = None
    try:
        fetched_branch = branch_workspace.download_archive(
            root_assignment.url,
            "upper-process",
            "preparation-base",
        )
        admitted_branch = branch_workspace.admit_assignment(
            fetched_branch,
            intended_recipient_nf_instance_id=BRANCH,
        )
        assert root_requests == 1
        assert admitted_branch.metadata.path.parent == tmp_path / "branch" / PLAN / "downloads"
        assert admitted_branch.metadata.path.read_bytes() == root_content
        assert not (tmp_path / "branch" / "upper-process").exists()

        leaf_assignment = HierarchyArtifactService(
            branch_workspace
        ).republish_leaf_assignment(
            parent=admitted_branch,
            containing_branch_nf_instance_id=BRANCH,
            leaf_nf_instance_id=LEAF_A,
        )
        leaf_content = leaf_assignment.path.read_bytes()
        leaf_requests = 0

        def leaf_handler(request: httpx.Request) -> httpx.Response:
            nonlocal leaf_requests
            leaf_requests += 1
            return httpx.Response(
                200,
                content=leaf_content,
                headers={"X-Artifact-SHA256": leaf_assignment.digest},
                request=request,
            )

        leaf_client = httpx.Client(transport=httpx.MockTransport(leaf_handler))
        leaf_workspace = workspace(
            tmp_path / "leaf",
            "http://leaf.example",
            client=leaf_client,
            allowed_origins=("http://branch.example",),
        )
        fetched_leaf = leaf_workspace.download_archive(
            leaf_assignment.url,
            "lower-process",
            "preparation-base",
        )
        admitted_leaf = leaf_workspace.admit_assignment(
            fetched_leaf,
            intended_recipient_nf_instance_id=LEAF_A,
        )

        assert leaf_requests == 1
        assert admitted_leaf.metadata.path.parent == tmp_path / "leaf" / PLAN / "downloads"
        assert admitted_leaf.metadata.path.read_bytes() == leaf_content
        assert not (tmp_path / "leaf" / "lower-process").exists()
    finally:
        root_workspace.close()
        branch_workspace.close()
        branch_client.close()
        if leaf_workspace is not None:
            leaf_workspace.close()
        if leaf_client is not None:
            leaf_client.close()


def test_branch_and_leaf_client_preparation_each_performs_one_assignment_get(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "publisher-root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    root_content = root_assignment.path.read_bytes()

    def root_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=root_content,
            headers={"X-Artifact-SHA256": root_assignment.digest},
            request=request,
        )

    publisher_client = httpx.Client(transport=httpx.MockTransport(root_handler))
    branch_publisher = workspace(
        tmp_path / "publisher-branch",
        "http://branch.example",
        client=publisher_client,
        allowed_origins=("http://root.example",),
    )
    downloaded = branch_publisher.download_archive(
        root_assignment.url,
        "publisher-process",
        "preparation-base",
    )
    admitted = branch_publisher.admit_assignment(
        downloaded,
        intended_recipient_nf_instance_id=BRANCH,
    )
    leaf_assignment = HierarchyArtifactService(branch_publisher).republish_leaf_assignment(
        parent=admitted,
        containing_branch_nf_instance_id=BRANCH,
        leaf_nf_instance_id=LEAF_A,
    )

    cases = (
        (
            "branch",
            root_assignment,
            ROOT,
            BRANCH,
            FLCapabilityType.SERVER_AND_CLIENT,
        ),
        ("leaf", leaf_assignment, BRANCH, LEAF_A, FLCapabilityType.CLIENT),
    )
    try:
        for name, source, publisher_id, recipient_id, capability in cases:
            content = source.path.read_bytes()
            request_count = 0

            def handler(
                request: httpx.Request,
                response_content=content,
                response_digest=source.digest,
            ) -> httpx.Response:
                nonlocal request_count
                request_count += 1
                return httpx.Response(
                    200,
                    content=response_content,
                    headers={"X-Artifact-SHA256": response_digest},
                    request=request,
                )

            receiver_client = httpx.Client(transport=httpx.MockTransport(handler))
            allowed_origin = (
                "http://root.example" if name == "branch" else "http://branch.example"
            )
            settings = FederatedLearningSettings(
                workspace_root=tmp_path / f"receiver-{name}",
                public_base_url=f"http://{name}-receiver.example",
                artifact_download=ArtifactDownloadSettings(
                    allowed_origins=(allowed_origin,)
                ),
            )
            receiver_workspace = FLWorkspace(
                settings,
                ArtifactSettings(),
                receiver_client,
            )
            receiver_workspace.open()
            context = Mock()
            context.get.return_value = NwdafContext(
                nf_instance_id=recipient_id,
                containing_nwdaf_process_instance_id=(
                    "22222222-2222-4222-8222-222222222222"
                ),
                api_root=f"http://{name}-receiver.example",
                internal_api_root=f"http://{name}-receiver-internal.example",
                ml_analytics_capabilities=(
                    MLAnalyticsCapability(
                        ml_analytics_ids=("UE_COMMUNICATION",),
                        fl_capability_type=capability,
                    ),
                ),
            )
            payload = {
                "mLEventSubscs": [
                    {
                        "mLEvent": "UE_COMMUNICATION",
                        "mLEventFilter": {"networkArea": {"tais": []}},
                        "modelInterInfo": "001122",
                    }
                ],
                "notifUri": "http://go.internal/training/callback",
                "notifCorreId": f"{name}-callback",
                "mlCorreId": "fl-process-001",
                "mLPreFlag": True,
                "mLModelInfos": [
                    {
                        "event": "UE_COMMUNICATION",
                        "mLFileAddr": {"mLModelUrl": source.url},
                    }
                ],
                "eventReq": {"notifMethod": "ON_EVENT_DETECTION"},
                "mLModelTrainInfos": [
                    {
                        "dataAvReq": {
                            "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                            "minNumSamples": 1,
                            "timeWindows": [
                                {
                                    "startTime": "2026-07-01T00:00:00Z",
                                    "stopTime": "2026-07-02T00:00:00Z",
                                }
                            ],
                        },
                        "timeAvReq": "PT5M",
                    }
                ],
                "mLTrainRepInfo": {"maxResTime": 300},
            }
            value = NwdafMLModelTrainSubsc.model_validate(payload)
            datasets = Mock()
            datasets.submit_external.return_value = "dataset-job-1"
            branch = Mock()
            branch.prepare.return_value = Mock(
                execution=Mock(process_id="lower-process"),
                artifact=Mock(url="http://branch.example/preparation-result"),
                outcome=PreparationOutcome.READY,
            )
            registry = FLExperimentRegistry()
            reservation = registry.reserve_client("resource-1", "fl-process-001")
            service = FLClientEngine(
                settings,
                FLClientSettings(
                    training_data={"collection_trigger": "consumer_subscription"},
                    model_interoperability_ids=("001122",),
                ),
                NotificationSettings(),
                context,
                datasets,
                receiver_workspace,
                experiments=registry,
                branch_coordinator=branch if name == "branch" else None,
            )
            service._enqueue_delivery = Mock()
            resource = FLClientResource(
                subscription_id="resource-1",
                representation=value,
                state=FLClientState.PREPARING,
                scope=TrainingScopeDescriptor.from_training_request(value, 0),
                experiment_reservation_id=reservation.reservation_id,
            )
            service._resources[resource.subscription_id] = resource
            assert service._capacity.acquire(blocking=False)
            try:
                service._run_preparation(
                    resource.subscription_id,
                    resource.revision,
                    Mock(),
                    Mock(),
                )

                assert request_count == 1
                accepted = service.get(resource.subscription_id).hierarchy_assignment
                assert accepted is not None
                assert accepted.metadata.path.read_bytes() == content
                assert accepted.contract.hierarchy_metadata.publisher_nf_instance_id == (
                    publisher_id
                )
            finally:
                service.close()
                receiver_workspace.close()
                receiver_client.close()
    finally:
        root_workspace.close()
        branch_publisher.close()
        publisher_client.close()


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing-header", FLArtifactIntegrityError),
        ("duplicate-header", FLArtifactIntegrityError),
        ("publisher", FLArtifactIdentityError),
        ("recipient", FLArtifactIdentityError),
        ("plan", FLArtifactIdentityError),
    ],
)
def test_single_fetch_assignment_admission_preserves_strict_identity_checks(
    tmp_path,
    failure,
    expected_error,
) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        headers: list[tuple[str, str]] = []
        if failure != "missing-header":
            headers.append(("X-Artifact-SHA256", root_assignment.digest))
        if failure == "duplicate-header":
            headers.append(("X-Artifact-SHA256", root_assignment.digest))
        return httpx.Response(200, content=content, headers=headers, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        downloaded = branch_workspace.download_archive(
            root_assignment.url,
            "upper-process",
            "preparation-base",
        )
        with pytest.raises(expected_error):
            branch_workspace.admit_assignment(
                downloaded,
                intended_recipient_nf_instance_id=(
                    ROOT if failure == "recipient" else BRANCH
                ),
                expected_plan_id=PLAN_B if failure == "plan" else PLAN,
                expected_publisher_nf_instance_id=(
                    BRANCH if failure == "publisher" else ROOT
                ),
            )

        assert not (tmp_path / "branch" / "upper-process").exists()
        assert not (tmp_path / "branch" / PLAN / "downloads").exists()
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


def test_single_fetch_assignment_rejects_valid_archive_with_wrong_body_digest(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    requested = publish_root_assignment(root_workspace)
    substituted = publish_root_assignment(root_workspace, PLAN_B)
    substituted_content = substituted.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=substituted_content,
            headers={"X-Artifact-SHA256": requested.digest},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        downloaded = branch_workspace.download_archive(
            requested.url,
            "upper-process",
            "preparation-base",
        )
        with pytest.raises(FLArtifactIntegrityError, match="archive digest"):
            branch_workspace.admit_assignment(
                downloaded,
                intended_recipient_nf_instance_id=BRANCH,
            )

        assert not (tmp_path / "branch" / "upper-process").exists()
        assert not (tmp_path / "branch" / PLAN_B / "downloads").exists()
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


@pytest.mark.parametrize("failure", ["header", "body", "publisher", "recipient", "plan"])
def test_hierarchy_download_rejects_integrity_and_identity_mismatch(tmp_path, failure: str) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        body = content + b"tampered" if failure == "body" else content
        header = "f" * 64 if failure == "header" else root_assignment.digest
        return httpx.Response(
            200,
            content=body,
            headers={"X-Artifact-SHA256": header},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        expected_error = (
            FLArtifactIntegrityError
            if failure in {"header", "body"}
            else FLArtifactIdentityError
        )
        with pytest.raises(expected_error):
            branch_workspace.download_hierarchy(
                root_assignment.url,
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=BRANCH if failure == "publisher" else ROOT,
                intended_recipient_nf_instance_id=ROOT if failure == "recipient" else BRANCH,
                expected_plan_id=PLAN_B if failure == "plan" else PLAN,
            )
        assert not tuple((tmp_path / "branch" / ".staging").iterdir())
        assert not (tmp_path / "branch" / PLAN / "downloads").exists()
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


@pytest.mark.parametrize("header_mode", ["missing", "duplicate"])
def test_hierarchy_download_requires_one_digest_header(tmp_path, header_mode: str) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        headers: list[tuple[str, str]] = []
        if header_mode == "duplicate":
            headers = [
                ("X-Artifact-SHA256", root_assignment.digest),
                ("X-Artifact-SHA256", root_assignment.digest),
            ]
        return httpx.Response(200, content=content, headers=headers, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        with pytest.raises(FLArtifactIntegrityError, match="header"):
            branch_workspace.download_hierarchy(
                root_assignment.url,
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


@pytest.mark.parametrize(
    "failure",
    ["role", "message", "origin", "not_found", "redirect"],
)
def test_hierarchy_download_rejects_contract_origin_and_transport_mismatch(
    tmp_path,
    failure: str,
) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_assignment = publish_root_assignment(root_workspace)
    content = root_assignment.path.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if failure in {"not_found", "redirect"}:
            return httpx.Response(404 if failure == "not_found" else 302, request=request)
        return httpx.Response(
            200,
            content=content,
            headers={"X-Artifact-SHA256": root_assignment.digest},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    allowed_origin = "http://different.example" if failure == "origin" else "http://root.example"
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=(allowed_origin,),
    )
    expected_error = {
        "role": FLArtifactContractError,
        "message": FLArtifactContractError,
        "origin": FLArtifactIdentityError,
        "not_found": FLArtifactUnavailableError,
        "redirect": FLArtifactUnavailableError,
    }[failure]
    try:
        with pytest.raises(expected_error):
            branch_workspace.download_hierarchy(
                root_assignment.url,
                expected_role=(
                    ArtifactRole.HIERARCHY_PREPARATION_RESULT
                    if failure == "role"
                    else ArtifactRole.HIERARCHY_ASSIGNMENT
                ),
                expected_message_type=(
                    HierarchyMessageType.LEAF_ASSIGNMENT
                    if failure == "message"
                    else HierarchyMessageType.BRANCH_ASSIGNMENT
                ),
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
    finally:
        root_workspace.close()
        branch_workspace.close()
        client.close()


@pytest.mark.parametrize(
    "url",
    [
        f"http://root.example/internal/v1/fl-artifacts/path/{'A' * 64}",
        f"http://root.example:invalid/internal/v1/fl-artifacts/path/{'a' * 64}",
        f"http://user@root.example/internal/v1/fl-artifacts/path/{'a' * 64}",
        f"http://root.example/internal/v1/fl-artifacts/path/{'a' * 64}?version=1",
        f"http://root.example/internal/v1/fl-artifacts/path/{'a' * 64}#fragment",
    ],
)
def test_hierarchy_download_rejects_unsupported_url_before_request(tmp_path, url: str) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected HTTP request: {request.url}")
        )
    )
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        with pytest.raises(FLArtifactIntegrityError, match="digest|unsupported|port"):
            branch_workspace.download_hierarchy(
                url,
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
    finally:
        branch_workspace.close()
        client.close()


def test_hierarchy_download_rejects_empty_expected_plan_before_request(tmp_path) -> None:
    digest = "a" * 64
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected HTTP request: {request.url}")
        )
    )
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        with pytest.raises(ValueError, match="UUID"):
            branch_workspace.download_hierarchy(
                f"http://root.example/internal/v1/fl-artifacts/path/{digest}",
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
                expected_plan_id="",
            )
    finally:
        branch_workspace.close()
        client.close()


@pytest.mark.parametrize(
    ("bundle_kind", "expected_error"),
    [
        ("invalid_archive", FLArtifactIntegrityError),
        ("invalid_contract", FLArtifactContractError),
    ],
)
def test_hierarchy_download_uses_bounded_archive_error_categories(
    tmp_path,
    bundle_kind: str,
    expected_error: type[Exception],
) -> None:
    if bundle_kind == "invalid_archive":
        content = b"not a gzip tar archive"
    else:
        content = build_bundle(
            tmp_path / "invalid-contract.tar.gz",
            mutate_manifest={"artifact_role": "HIERARCHY_ASSIGNMENT"},
        )
    digest = hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"X-Artifact-SHA256": digest},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        with pytest.raises(expected_error):
            branch_workspace.download_hierarchy(
                f"http://root.example/internal/v1/fl-artifacts/path/{digest}",
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
        assert not tuple((tmp_path / "branch" / ".staging").iterdir())
    finally:
        branch_workspace.close()
        client.close()


def test_hierarchy_download_maps_transport_failure_and_removes_staging(tmp_path) -> None:
    digest = "a" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("peer unavailable", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    try:
        with pytest.raises(FLArtifactUnavailableError, match="transport"):
            branch_workspace.download_hierarchy(
                f"http://root.example/internal/v1/fl-artifacts/path/{digest}",
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
        assert not tuple((tmp_path / "branch" / ".staging").iterdir())
    finally:
        branch_workspace.close()
        client.close()


def test_hierarchy_download_maps_staging_io_failure(tmp_path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected HTTP request: {request.url}")
        )
    )
    branch_workspace = workspace(
        tmp_path / "branch",
        "http://branch.example",
        client=client,
        allowed_origins=("http://root.example",),
    )
    (tmp_path / "branch" / ".staging").write_text("not a directory")
    try:
        with pytest.raises(FLWorkspaceError, match="staging"):
            branch_workspace.download_hierarchy(
                f"http://root.example/internal/v1/fl-artifacts/path/{'a' * 64}",
                expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
                expected_message_type=HierarchyMessageType.BRANCH_ASSIGNMENT,
                expected_publisher_nf_instance_id=ROOT,
                intended_recipient_nf_instance_id=BRANCH,
            )
    finally:
        branch_workspace.close()
        client.close()


def test_release_plan_is_exact_and_idempotent(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        first = publish_root_assignment(root_workspace, PLAN)
        second = publish_root_assignment(root_workspace, PLAN_B)
        root_workspace.release_plan(PLAN)
        root_workspace.release_plan(PLAN)

        assert not first.path.exists()
        assert second.path.exists()
        with pytest.raises(ValueError, match="UUID"):
            root_workspace.release_plan("not-a-plan")
        assert second.path.exists()
    finally:
        root_workspace.close()


def test_release_plan_removes_all_explicitly_owned_process_directories(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        owned = root_workspace.publish(
            process_id="root-process",
            participant_id=ROOT,
            round_indicator=0,
            role="ROUND_INPUT",
            base=base_bundle(),
            model=base_bundle().model,
            metadata={
                "artifact_role": "ROUND_INPUT",
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": "root-process",
                    "round_ind": 0,
                    "model_contract_digest": "1" * 64,
                    "preprocessing_contract_digest": "2" * 64,
                    "weights_digest": "3" * 64,
                    "client_training": {"epochs": 1},
                },
            },
            owner_plan_id=PLAN,
        )
        unowned_flat = root_workspace.publish(
            process_id="flat-process",
            participant_id=ROOT,
            round_indicator=0,
            role="ROUND_INPUT",
            base=base_bundle(),
            model=base_bundle().model,
            metadata={
                "artifact_role": "ROUND_INPUT",
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": "flat-process",
                    "round_ind": 0,
                    "model_contract_digest": "1" * 64,
                    "preprocessing_contract_digest": "2" * 64,
                    "weights_digest": "3" * 64,
                    "client_training": {"epochs": 1},
                },
            },
        )
        sibling = publish_root_assignment(root_workspace, PLAN_B)

        root_workspace.release_plan(PLAN)

        assert not owned.path.exists()
        assert sibling.path.exists()
        assert unowned_flat.path.exists()
    finally:
        root_workspace.close()


def test_late_worker_cannot_republish_after_plan_release(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    root_workspace.release_plan(PLAN)
    try:
        with pytest.raises(FLWorkspaceError, match="already released"):
            root_workspace.publish(
                process_id="late-process",
                participant_id=ROOT,
                round_indicator=0,
                role="ROUND_INPUT",
                base=base_bundle(),
                model=base_bundle().model,
                metadata={
                    "artifact_role": "ROUND_INPUT",
                    "fl_metadata": {
                        "contract_version": "1.0",
                        "ml_corre_id": "late-process",
                        "round_ind": 0,
                        "model_contract_digest": "1" * 64,
                        "preprocessing_contract_digest": "2" * 64,
                        "weights_digest": "3" * 64,
                        "client_training": {"epochs": 1},
                    },
                },
                owner_plan_id=PLAN,
            )

        assert not (tmp_path / "root" / "late-process").exists()
    finally:
        root_workspace.close()


def test_open_clears_previous_process_workspace_before_admission(tmp_path) -> None:
    root = tmp_path / "root"
    stale = root / "old-process" / "downloads" / "artifact.tar.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    root_workspace = FLWorkspace(
        FederatedLearningSettings(
            workspace_root=root,
            public_base_url="http://root.example",
        ),
        ArtifactSettings(),
    )
    try:
        root_workspace.open()

        assert root.exists()
        assert tuple(root.iterdir()) == ()
    finally:
        root_workspace.close()


def test_open_fails_instead_of_admitting_with_partially_cleared_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    stale = root / "old-process" / "artifact.tar.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    root_workspace = FLWorkspace(
        FederatedLearningSettings(
            workspace_root=root,
            public_base_url="http://root.example",
        ),
        ArtifactSettings(),
    )
    monkeypatch.setattr(
        root_workspace,
        "_delete_direct_child",
        Mock(side_effect=OSError("busy")),
    )

    try:
        with pytest.raises(FLWorkspaceError, match="startup cleanup failed"):
            root_workspace.open()

        assert stale.exists()
    finally:
        root_workspace.close()


def test_cleanup_expired_removes_stale_staging_but_keeps_active_plan(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        active = publish_root_assignment(root_workspace, PLAN)
        staging = tmp_path / "root" / ".staging"
        staging.mkdir()
        staged_file = staging / "abandoned.tar.gz"
        staged_file.write_bytes(b"partial")
        old = time.time() - 7200
        os.utime(active.path.parents[3], (old, old))
        os.utime(staged_file, (old, old))

        root_workspace.cleanup_expired()

        assert active.path.exists()
        assert not staged_file.exists()
    finally:
        root_workspace.close()


def test_failed_plan_deletion_is_retried_by_later_workspace_operation(
    tmp_path,
    monkeypatch,
) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    artifact = publish_root_assignment(root_workspace, PLAN)
    original_delete = root_workspace._delete_direct_child
    failed_once = False

    def fail_once(path):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("busy")
        original_delete(path)

    monkeypatch.setattr(root_workspace, "_delete_direct_child", fail_once)
    try:
        with pytest.raises(FLWorkspaceError, match="release failed"):
            root_workspace.release_plan(PLAN)
        assert artifact.path.exists()
        with root_workspace._lock:
            for path in root_workspace._cleanup_failures:
                root_workspace._cleanup_failures[path] = (
                    time.time() - root_workspace._settings.workspace_ttl_seconds - 1
                )

        root_workspace.cleanup_expired()

        assert not artifact.path.exists()
        assert root_workspace._cleanup_failures == {}
    finally:
        root_workspace.close()


def test_open_reader_finishes_after_plan_release_and_new_resolve_is_not_found(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        artifact = publish_root_assignment(root_workspace, PLAN)
        reader = root_workspace.open_artifact(
            artifact.process_id,
            artifact.participant_id,
            artifact.round_indicator,
            artifact.role,
            artifact.digest,
        )
        assert reader is not None

        root_workspace.release_plan(PLAN)

        assert artifact.path.exists()
        assert (
            root_workspace.resolve(
                artifact.process_id,
                artifact.participant_id,
                artifact.round_indicator,
                artifact.role,
                artifact.digest,
            )
            is None
        )
        assert b"".join(reader.iter_bytes())
        assert not artifact.path.exists()
    finally:
        root_workspace.close()
