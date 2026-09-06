import io
import os
import time
from unittest.mock import Mock

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
    RoundInputArtifact,
    RoundLocalArtifact,
)
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    FLWorkspaceError,
)
from py_mtlf.core.trainer import LoadedBundle
from py_mtlf.core.training_scope import TrainingScopeDescriptor

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


def base_bundle() -> LoadedBundle:
    torch.manual_seed(7)
    model = RuntimeModel()
    scaler = StandardScaler().fit(np.arange(20).reshape(10, 2))
    scaler_stream = io.BytesIO()
    joblib.dump(scaler, scaler_stream)
    return LoadedBundle(
        manifest={
            "workload_profile": "ue_communication_forecasting",
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


def publish_owned_round_input(root_workspace: FLWorkspace, plan_id: str = PLAN):
    return HierarchyArtifactService(root_workspace).publish_round_input(
        base=base_bundle(),
        plan_id=plan_id,
        process_id=plan_id,
        server_nf_instance_id=ROOT,
        round_indicator=0,
        epochs=1,
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
                    "ml_corre_id": "lower-process",
                    "round_ind": 4,
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
            upper_training_scope=TrainingScopeDescriptor(
                eventSubscription={"mLEvent": "UE_COMMUNICATION"}
            ),
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
        service.publish_hierarchy_aggregate(
            upper_input=upper,
            lower_global=lower_global,
            plan_id=PLAN,
            upper_process_id="upper-process",
            branch_nf_instance_id=BRANCH,
            upper_round_indicator=3,
            upper_training_scope=TrainingScopeDescriptor(
                eventSubscription={"mLEvent": "UE_COMMUNICATION"}
            ),
        )
    finally:
        branch_workspace.close()


def test_release_plan_is_exact_and_idempotent(tmp_path) -> None:
    root_workspace = workspace(tmp_path / "root", "http://root.example")
    try:
        first = publish_owned_round_input(root_workspace, PLAN)
        second = publish_owned_round_input(root_workspace, PLAN_B)
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
                    "ml_corre_id": "root-process",
                    "round_ind": 0,
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
                    "ml_corre_id": "flat-process",
                    "round_ind": 0,
                    "client_training": {"epochs": 1},
                },
            },
        )
        sibling = publish_owned_round_input(root_workspace, PLAN_B)

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
                        "ml_corre_id": "late-process",
                        "round_ind": 0,
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
        active = publish_owned_round_input(root_workspace, PLAN)
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
    artifact = publish_owned_round_input(root_workspace, PLAN)
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
        artifact = publish_owned_round_input(root_workspace, PLAN)
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
