from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from py_mtlf.core.fl_artifacts import (
    FinalModelArtifact,
    HierarchyAssignmentArtifact,
    HierarchyPreparationResultArtifact,
    RoundInputArtifact,
    RoundLocalArtifact,
    TensorStateEntry,
    WapeComponents,
    validate_fl_artifact,
    validate_fl_artifact_manifest,
    validate_tensor_compatibility,
    wape,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
CLIENT_A = "00000000-0000-4000-8000-000000000001"
CLIENT_B = "00000000-0000-4000-8000-000000000002"
ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
PLAN = "11111111-1111-4111-8111-111111111111"


def training_scope(name: str = "scope-a") -> dict[str, object]:
    return {
        "event_subscription": {"mLEvent": "UE_COMMUNICATION", "scopeName": name},
        "target_reporting_ue": {"intGroupIds": ["group-G"]},
        "requested_time_windows": [],
    }


def common_metadata() -> dict[str, object]:
    return {"ml_corre_id": "fl-process-001"}


def dataset_evidence(training_sample_count: int = 120) -> dict[str, object]:
    return {
        "observation_count": 400,
        "training_sample_count": training_sample_count,
        "validation_sample_count": 34,
    }


def hierarchy_strategy() -> dict[str, object]:
    return {
        "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
        "participant_selection": "all",
        "waiting_policy": "all",
        "aggregation": "sample_weighted",
    }


def validation_summary(
    participant_id: str,
    *,
    sample_count: int,
    base_error: float,
    candidate_error: float,
    actual: float,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    return {
        "participant_nf_instance_id": participant_id,
        "training_scope": training_scope(participant_id),
        "evaluation_sample_count": sample_count,
        "start_time": start,
        "end_time": end,
        "base": {"absolute_error_sum": base_error, "absolute_actual_sum": actual},
        "candidate": {
            "absolute_error_sum": candidate_error,
            "absolute_actual_sum": actual,
        },
    }


def round_input() -> dict[str, object]:
    return {
        "artifact_role": "ROUND_INPUT",
        "fl_metadata": {
            "ml_corre_id": "fl-process-001",
            "round_ind": 0,
            "client_training": {"epochs": 3},
        },
    }


def test_round_input_requires_server_controlled_positive_epochs() -> None:
    artifact = validate_fl_artifact(round_input())
    assert isinstance(artifact, RoundInputArtifact)
    assert artifact.fl_metadata.client_training.epochs == 3

    for invalid in (0, -1, 1.5, "3", None):
        value = round_input()
        value["fl_metadata"]["client_training"]["epochs"] = invalid
        with pytest.raises(ValidationError):
            validate_fl_artifact(value)


def test_hierarchy_aggregate_requires_canonical_subordinates_and_exact_sum() -> None:
    metadata = {
        **common_metadata(),
        "round_ind": 2,
        "participant_nf_instance_id": BRANCH,
        "training_scope": training_scope(),
        "training_sample_count": 200,
        "lower_round_ind": 7,
        "lower_global_artifact_digest": DIGEST_A,
        "subordinate_participants": [
            {
                "participant_nf_instance_id": CLIENT_A,
                "training_sample_count": 120,
                "local_artifact_digest": DIGEST_A,
            },
            {
                "participant_nf_instance_id": CLIENT_B,
                "training_sample_count": 80,
                "local_artifact_digest": DIGEST_B,
            },
        ],
    }
    value = {
        "artifact_role": "ROUND_LOCAL",
        "result_type": "HIERARCHY_AGGREGATE",
        "fl_metadata": metadata,
    }
    artifact = validate_fl_artifact(value)
    assert isinstance(artifact, RoundLocalArtifact)

    metadata["training_sample_count"] = 199
    with pytest.raises(ValidationError, match="subordinate sample counts"):
        validate_fl_artifact(value)
    metadata["training_sample_count"] = 200
    metadata["subordinate_participants"].reverse()
    with pytest.raises(ValidationError, match="canonical"):
        validate_fl_artifact(value)


def test_round_local_training_requires_explicit_scope_and_sample_counts() -> None:
    metadata = {
        **common_metadata(),
        "round_ind": 0,
        "participant_nf_instance_id": CLIENT_A,
        "training_scope": training_scope(),
        "training_sample_count": 120,
        "dataset_evidence": dataset_evidence(),
    }
    value = {
        "artifact_role": "ROUND_LOCAL",
        "result_type": "TRAINING",
        "fl_metadata": metadata,
    }
    assert validate_fl_artifact(value).fl_metadata.training_sample_count == 120

    metadata["dataset_evidence"] = dataset_evidence(119)
    with pytest.raises(ValidationError, match="dataset evidence sample count"):
        validate_fl_artifact(value)


def test_round_global_requires_canonical_participants_and_exact_sample_sum() -> None:
    metadata = {
        **common_metadata(),
        "round_ind": 0,
        "participants": [
            {
                "participant_nf_instance_id": CLIENT_A,
                "training_sample_count": 120,
                "local_artifact_digest": DIGEST_A,
            },
            {
                "participant_nf_instance_id": CLIENT_B,
                "training_sample_count": 80,
                "local_artifact_digest": DIGEST_B,
            },
        ],
        "aggregated_training_sample_count": 200,
    }
    value = {
        "artifact_role": "ROUND_GLOBAL",
        "fl_metadata": metadata,
    }
    assert validate_fl_artifact(value).fl_metadata.aggregated_training_sample_count == 200
    metadata["aggregated_training_sample_count"] = 199
    with pytest.raises(ValidationError, match="aggregated sample count"):
        validate_fl_artifact(value)


def test_round_local_accuracy_check_preserves_wape_components() -> None:
    now = datetime.now(UTC)
    metadata = {
        **common_metadata(),
        "round_ind": 1,
        "participant_nf_instance_id": CLIENT_A,
        "training_scope": training_scope(),
        "evaluation": {
            "evaluation_stage": "FINAL_VALIDATION",
            "evaluation_sample_count": 40,
            "start_time": now,
            "end_time": now + timedelta(minutes=1),
            "base": {"absolute_error_sum": 10, "absolute_actual_sum": 100},
            "candidate": {"absolute_error_sum": 5, "absolute_actual_sum": 100},
        },
    }
    artifact = validate_fl_artifact(
        {
            "artifact_role": "ROUND_LOCAL",
            "result_type": "ACCURACY_CHECK",
            "fl_metadata": metadata,
        }
    )
    assert wape(artifact.fl_metadata.evaluation.candidate) == 0.05
    assert wape(WapeComponents(absolute_error_sum=1, absolute_actual_sum=0)) is None


def test_hierarchy_accuracy_check_requires_exact_canonical_subordinate_evidence() -> None:
    start = datetime.now(UTC)
    middle = start + timedelta(minutes=1)
    end = middle + timedelta(minutes=1)
    subordinates = [
        validation_summary(
            CLIENT_A,
            sample_count=40,
            base_error=10,
            candidate_error=5,
            actual=100,
            start=start,
            end=middle,
        ),
        validation_summary(
            CLIENT_B,
            sample_count=30,
            base_error=8,
            candidate_error=4,
            actual=90,
            start=middle,
            end=end,
        ),
    ]
    metadata = {
        **common_metadata(),
        "round_ind": 2,
        "participant_nf_instance_id": BRANCH,
        "training_scope": training_scope(),
        "evaluation": {
            "evaluation_stage": "FINAL_VALIDATION",
            "evaluation_sample_count": 70,
            "start_time": start,
            "end_time": end,
            "base": {"absolute_error_sum": 18, "absolute_actual_sum": 190},
            "candidate": {"absolute_error_sum": 9, "absolute_actual_sum": 190},
        },
        "subordinate_validation_summaries": subordinates,
    }
    value = {
        "artifact_role": "ROUND_LOCAL",
        "result_type": "ACCURACY_CHECK",
        "fl_metadata": metadata,
    }
    assert isinstance(validate_fl_artifact(value), RoundLocalArtifact)
    metadata["evaluation"]["candidate"]["absolute_error_sum"] = 10
    with pytest.raises(ValidationError, match="subordinate validation summaries"):
        validate_fl_artifact(value)
    metadata["evaluation"]["candidate"]["absolute_error_sum"] = 9
    metadata["subordinate_validation_summaries"].reverse()
    with pytest.raises(ValidationError, match="canonical"):
        validate_fl_artifact(value)


def test_round_local_result_type_rejects_wrong_metadata_shape() -> None:
    value = {
        "artifact_role": "ROUND_LOCAL",
        "result_type": "ACCURACY_CHECK",
        "fl_metadata": {
            **common_metadata(),
            "round_ind": 0,
            "participant_nf_instance_id": CLIENT_A,
            "training_scope": training_scope(),
            "training_sample_count": 120,
            "dataset_evidence": dataset_evidence(),
        },
    }
    with pytest.raises(ValidationError):
        validate_fl_artifact(value)


def test_final_model_requires_identity_complete_evidence_and_accepted_gate() -> None:
    now = datetime.now(UTC)
    summaries = [
        validation_summary(
            CLIENT_A,
            sample_count=40,
            base_error=10,
            candidate_error=5,
            actual=100,
            start=now,
            end=now + timedelta(minutes=1),
        ),
        validation_summary(
            CLIENT_B,
            sample_count=30,
            base_error=8,
            candidate_error=4,
            actual=90,
            start=now,
            end=now + timedelta(minutes=1),
        ),
    ]
    metadata = {
        **common_metadata(),
        "previous_model_unique_id": 4,
        "participants": [
            {"participant_nf_instance_id": CLIENT_A, "training_sample_count": 120},
            {"participant_nf_instance_id": CLIENT_B, "training_sample_count": 80},
        ],
        "validation_summary": summaries,
        "global_gate_accepted": True,
        "created_at": now,
    }
    value = {
        "artifact_role": "FINAL_MODEL",
        "model_identity": {"model_unique_id": 5},
        "fl_metadata": metadata,
    }
    assert isinstance(validate_fl_artifact(value), FinalModelArtifact)
    metadata["global_gate_accepted"] = False
    with pytest.raises(ValidationError, match="accepted global gate"):
        validate_fl_artifact(value)


def test_final_model_hierarchy_provenance_matches_direct_branch_evidence() -> None:
    start = datetime.now(UTC)
    middle = start + timedelta(minutes=1)
    end = middle + timedelta(minutes=1)
    subordinates = [
        validation_summary(
            CLIENT_A,
            sample_count=40,
            base_error=10,
            candidate_error=5,
            actual=100,
            start=start,
            end=middle,
        ),
        validation_summary(
            CLIENT_B,
            sample_count=30,
            base_error=8,
            candidate_error=4,
            actual=90,
            start=middle,
            end=end,
        ),
    ]
    branch_summary = validation_summary(
        BRANCH,
        sample_count=70,
        base_error=18,
        candidate_error=9,
        actual=190,
        start=start,
        end=end,
    )
    metadata = {
        **common_metadata(),
        "previous_model_unique_id": 4,
        "participants": [
            {"participant_nf_instance_id": BRANCH, "training_sample_count": 200}
        ],
        "validation_summary": [branch_summary],
        "hierarchy_validation": {
            "plan_id": PLAN,
            "branches": [
                {
                    "branch_nf_instance_id": BRANCH,
                    "subordinate_validation_summaries": subordinates,
                }
            ],
        },
        "global_gate_accepted": True,
        "created_at": datetime.now(UTC),
    }
    value = {
        "artifact_role": "FINAL_MODEL",
        "model_identity": {"model_unique_id": 5},
        "fl_metadata": metadata,
    }
    artifact = validate_fl_artifact(value)
    assert isinstance(artifact, FinalModelArtifact)

    metadata["hierarchy_validation"]["branches"][0]["branch_nf_instance_id"] = ROOT
    with pytest.raises(ValidationError, match="direct participants"):
        validate_fl_artifact(value)


def test_hierarchy_assignment_and_result_use_nested_message_discriminator() -> None:
    branch = validate_fl_artifact(
        {
            "artifact_role": "HIERARCHY_ASSIGNMENT",
            "hierarchy_metadata": {
                "message_type": "BRANCH_ASSIGNMENT",
                "plan_id": PLAN,
                "publisher_nf_instance_id": ROOT,
                "intended_recipient_nf_instance_id": BRANCH,
                "assigned_leaf_nf_instance_ids": [CLIENT_A, CLIENT_B],
                "admission": {"mode": "complete_required"},
                "strategy": hierarchy_strategy(),
            },
        }
    )
    assert isinstance(branch, HierarchyAssignmentArtifact)

    leaf = validate_fl_artifact(
        {
            "artifact_role": "HIERARCHY_ASSIGNMENT",
            "hierarchy_metadata": {
                "message_type": "LEAF_ASSIGNMENT",
                "plan_id": PLAN,
                "publisher_nf_instance_id": BRANCH,
                "intended_recipient_nf_instance_id": CLIENT_A,
                "parent_branch_nf_instance_id": BRANCH,
                "strategy": hierarchy_strategy(),
            },
        }
    )
    assert isinstance(leaf, HierarchyAssignmentArtifact)

    result = validate_fl_artifact(
        {
            "artifact_role": "HIERARCHY_PREPARATION_RESULT",
            "hierarchy_metadata": {
                "message_type": "PREPARATION_RESULT",
                "plan_id": PLAN,
                "publisher_nf_instance_id": BRANCH,
                "intended_recipient_nf_instance_id": ROOT,
                "outcome": "READY",
                "assigned_client_nf_instance_ids": [CLIENT_A, CLIENT_B],
                "prepared_clients": [
                    {"nf_instance_id": CLIENT_A},
                    {"nf_instance_id": CLIENT_B},
                ],
                "failed_clients": [],
                "timed_out_client_nf_instance_ids": [],
            },
        }
    )
    assert isinstance(result, HierarchyPreparationResultArtifact)

    with pytest.raises(ValidationError):
        validate_fl_artifact(
            {
                "artifact_role": "HIERARCHY_PREPARATION_RESULT",
                "hierarchy_metadata": leaf.hierarchy_metadata.model_dump(mode="json"),
            }
        )


def test_complete_manifest_projection_rejects_incompatible_role_fields() -> None:
    manifest = {
        "artifact_role": "HIERARCHY_ASSIGNMENT",
        "fl_metadata": {},
        "hierarchy_metadata": {
            "message_type": "LEAF_ASSIGNMENT",
            "plan_id": PLAN,
            "publisher_nf_instance_id": BRANCH,
            "intended_recipient_nf_instance_id": CLIENT_A,
            "parent_branch_nf_instance_id": BRANCH,
            "strategy": hierarchy_strategy(),
        },
    }
    with pytest.raises(ValueError, match="incompatible fields"):
        validate_fl_artifact_manifest(manifest)


def test_tensor_compatibility_accepts_values_but_rejects_contract_changes() -> None:
    base = (
        TensorStateEntry(name="weight", shape=(2, 2), dtype="float32", floating=True),
        TensorStateEntry(
            name="num_batches_tracked", shape=(), dtype="int64", floating=False
        ),
    )
    validate_tensor_compatibility(base, base)
    changed = (base[0].model_copy(update={"shape": (3, 2)}), base[1])
    with pytest.raises(ValueError, match="tensor contract"):
        validate_tensor_compatibility(base, changed)
