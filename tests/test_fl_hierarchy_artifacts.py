import io
import os
import time

import httpx
import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from py_mtlf.config import (
    ArtifactDownloadSettings,
    ArtifactSettings,
    FederatedLearningSettings,
)
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    HierarchyAssignmentArtifact,
    HierarchyPreparationResultArtifact,
)
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
    weights_digest,
)
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader

ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEAF_A = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEAF_B = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PLAN = "11111111-1111-4111-8111-111111111111"
PLAN_B = "22222222-2222-4222-8222-222222222222"

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
        allowed_origins=("http://root.example",),
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
        with pytest.raises(FLArtifactIntegrityError, match="digest|unsupported"):
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


def test_cleanup_expired_removes_stale_staging_and_plan_but_keeps_current_plan(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        stale = publish_root_assignment(root_workspace, PLAN)
        current = publish_root_assignment(root_workspace, PLAN_B)
        staging = tmp_path / "root" / ".staging"
        staging.mkdir()
        staged_file = staging / "abandoned.tar.gz"
        staged_file.write_bytes(b"partial")
        old = time.time() - 7200
        os.utime(stale.path.parents[3], (old, old))
        os.utime(staged_file, (old, old))

        root_workspace.cleanup_expired()

        assert not stale.path.exists()
        assert not staged_file.exists()
        assert current.path.exists()
    finally:
        root_workspace.close()
