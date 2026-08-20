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
DIGEST_C = "c" * 64
CLIENT_A = "00000000-0000-4000-8000-000000000001"
CLIENT_B = "00000000-0000-4000-8000-000000000002"
ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
PLAN = "11111111-1111-4111-8111-111111111111"
FILE_DIGESTS = {
    "model.py": DIGEST_A,
    "model.npy": DIGEST_C,
    "scaler.pkl": DIGEST_B,
}


def common_metadata() -> dict[str, object]:
    return {
        "contract_version": "1.0",
        "ml_corre_id": "fl-process-001",
        "model_contract_digest": DIGEST_A,
        "preprocessing_contract_digest": DIGEST_B,
        "base_weights_digest": DIGEST_A,
        "weights_digest": DIGEST_C,
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
        "scope_digest": DIGEST_B,
        "evaluation_sample_count": sample_count,
        "start_time": start,
        "end_time": end,
        "base_model_weights_digest": DIGEST_A,
        "candidate_weights_digest": DIGEST_C,
        "base": {
            "absolute_error_sum": base_error,
            "absolute_actual_sum": actual,
        },
        "candidate": {
            "absolute_error_sum": candidate_error,
            "absolute_actual_sum": actual,
        },
    }


def round_input() -> dict[str, object]:
    return {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_INPUT",
        "file_digests": FILE_DIGESTS,
        "fl_metadata": {
            "contract_version": "1.0",
            "ml_corre_id": "fl-process-001",
            "round_ind": 0,
            "model_contract_digest": DIGEST_A,
            "preprocessing_contract_digest": DIGEST_B,
            "weights_digest": DIGEST_C,
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

    missing = round_input()
    del missing["fl_metadata"]["client_training"]
    with pytest.raises(ValidationError):
        validate_fl_artifact(missing)


def test_hierarchy_aggregate_requires_canonical_subordinates_and_exact_sum() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "round_ind": 2,
            "participant_nf_instance_id": BRANCH,
            "scope_digest": DIGEST_B,
            "input_global_weights_digest": DIGEST_A,
            "training_sample_count": 200,
            "lower_round_ind": 7,
            "lower_global_artifact_digest": DIGEST_C,
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
    )
    artifact = validate_fl_artifact(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_LOCAL",
            "result_type": "HIERARCHY_AGGREGATE",
            "fl_metadata": metadata,
            "file_digests": FILE_DIGESTS,
        }
    )
    assert isinstance(artifact, RoundLocalArtifact)
    assert artifact.fl_metadata.training_sample_count == 200

    metadata["training_sample_count"] = 199
    with pytest.raises(ValidationError, match="subordinate sample counts"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_LOCAL",
                "result_type": "HIERARCHY_AGGREGATE",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )

    metadata["training_sample_count"] = 200
    metadata["subordinate_participants"].reverse()
    with pytest.raises(ValidationError, match="canonical"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_LOCAL",
                "result_type": "HIERARCHY_AGGREGATE",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


def test_round_local_contract_has_no_formal_model_identity() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "round_ind": 0,
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_B,
            "training_sample_count": 120,
            "input_global_weights_digest": DIGEST_A,
        }
    )
    artifact = validate_fl_artifact(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_LOCAL",
            "result_type": "TRAINING",
            "fl_metadata": metadata,
            "file_digests": FILE_DIGESTS,
        }
    )
    assert artifact.fl_metadata.training_sample_count == 120
    assert "model_identity" not in artifact.model_dump()


def test_round_global_requires_canonical_participants_and_exact_sample_sum() -> None:
    metadata = common_metadata()
    metadata.update(
        {
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
    )
    assert (
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_GLOBAL",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        ).fl_metadata.aggregated_training_sample_count
        == 200
    )
    metadata["aggregated_training_sample_count"] = 199
    with pytest.raises(ValidationError, match="aggregated sample count"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_GLOBAL",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


def test_round_local_accuracy_check_preserves_wape_components() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "base_weights_digest": DIGEST_C,
            "weights_digest": DIGEST_C,
            "round_ind": 1,
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_B,
            "input_global_weights_digest": DIGEST_C,
            "evaluation": {
                "evaluation_stage": "FINAL_VALIDATION",
                "evaluation_sample_count": 40,
                "start_time": datetime.now(UTC),
                "end_time": datetime.now(UTC) + timedelta(minutes=1),
                "base_model_weights_digest": DIGEST_A,
                "candidate_weights_digest": DIGEST_C,
                "base": {"absolute_error_sum": 10, "absolute_actual_sum": 100},
                "candidate": {"absolute_error_sum": 5, "absolute_actual_sum": 100},
            },
        }
    )
    artifact = validate_fl_artifact(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "ROUND_LOCAL",
            "result_type": "ACCURACY_CHECK",
            "fl_metadata": metadata,
            "file_digests": FILE_DIGESTS,
        }
    )
    assert wape(artifact.fl_metadata.evaluation.candidate) == 0.05
    assert wape(WapeComponents(absolute_error_sum=1, absolute_actual_sum=0)) is None


def test_round_local_accuracy_check_rejects_modified_candidate_weights() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "base_weights_digest": DIGEST_C,
            "weights_digest": DIGEST_B,
            "round_ind": 1,
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_B,
            "input_global_weights_digest": DIGEST_C,
            "evaluation": {
                "evaluation_stage": "FINAL_VALIDATION",
                "evaluation_sample_count": 40,
                "start_time": datetime.now(UTC),
                "end_time": datetime.now(UTC) + timedelta(minutes=1),
                "base_model_weights_digest": DIGEST_A,
                "candidate_weights_digest": DIGEST_C,
                "base": {"absolute_error_sum": 10, "absolute_actual_sum": 100},
                "candidate": {"absolute_error_sum": 5, "absolute_actual_sum": 100},
            },
        }
    )
    with pytest.raises(ValidationError, match="output weights digest"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_LOCAL",
                "result_type": "ACCURACY_CHECK",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


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
    metadata = common_metadata()
    metadata.update(
        {
            "base_weights_digest": DIGEST_C,
            "weights_digest": DIGEST_C,
            "round_ind": 2,
            "participant_nf_instance_id": BRANCH,
            "scope_digest": DIGEST_B,
            "input_global_weights_digest": DIGEST_C,
            "evaluation": {
                "evaluation_stage": "FINAL_VALIDATION",
                "evaluation_sample_count": 70,
                "start_time": start,
                "end_time": end,
                "base_model_weights_digest": DIGEST_A,
                "candidate_weights_digest": DIGEST_C,
                "base": {"absolute_error_sum": 18, "absolute_actual_sum": 190},
                "candidate": {"absolute_error_sum": 9, "absolute_actual_sum": 190},
            },
            "subordinate_validation_summaries": subordinates,
        }
    )
    value = {
        "bundle_schema_version": "1.0",
        "artifact_role": "ROUND_LOCAL",
        "result_type": "ACCURACY_CHECK",
        "fl_metadata": metadata,
        "file_digests": FILE_DIGESTS,
    }

    artifact = validate_fl_artifact(value)
    assert isinstance(artifact, RoundLocalArtifact)
    assert tuple(
        item.participant_nf_instance_id
        for item in artifact.fl_metadata.subordinate_validation_summaries
    ) == (CLIENT_A, CLIENT_B)

    metadata["evaluation"]["candidate"]["absolute_error_sum"] = 10
    with pytest.raises(ValidationError, match="subordinate validation summaries"):
        validate_fl_artifact(value)

    metadata["evaluation"]["candidate"]["absolute_error_sum"] = 9
    metadata["subordinate_validation_summaries"].reverse()
    with pytest.raises(ValidationError, match="canonical"):
        validate_fl_artifact(value)


def test_round_local_result_type_rejects_wrong_metadata_shape() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "round_ind": 0,
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_B,
            "training_sample_count": 120,
            "input_global_weights_digest": DIGEST_A,
        }
    )
    with pytest.raises(ValidationError):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "ROUND_LOCAL",
                "result_type": "ACCURACY_CHECK",
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


def test_final_model_requires_identity_and_accepted_gate() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "previous_model_unique_id": 4,
            "participants": [
                {"participant_nf_instance_id": CLIENT_A, "training_sample_count": 120},
                {"participant_nf_instance_id": CLIENT_B, "training_sample_count": 80},
            ],
            "final_candidate_digest": DIGEST_C,
            "validation_summary": [
                {
                    "participant_nf_instance_id": CLIENT_A,
                    "scope_digest": DIGEST_A,
                    "evaluation_sample_count": 40,
                    "start_time": datetime.now(UTC),
                    "end_time": datetime.now(UTC) + timedelta(minutes=1),
                    "base_model_weights_digest": DIGEST_A,
                    "candidate_weights_digest": DIGEST_C,
                    "base": {"absolute_error_sum": 10, "absolute_actual_sum": 100},
                    "candidate": {"absolute_error_sum": 5, "absolute_actual_sum": 100},
                },
                {
                    "participant_nf_instance_id": CLIENT_B,
                    "scope_digest": DIGEST_B,
                    "evaluation_sample_count": 30,
                    "start_time": datetime.now(UTC),
                    "end_time": datetime.now(UTC) + timedelta(minutes=1),
                    "base_model_weights_digest": DIGEST_A,
                    "candidate_weights_digest": DIGEST_C,
                    "base": {"absolute_error_sum": 8, "absolute_actual_sum": 90},
                    "candidate": {"absolute_error_sum": 4, "absolute_actual_sum": 90},
                },
            ],
            "global_gate_accepted": True,
            "created_at": datetime.now(UTC),
        }
    )
    artifact = validate_fl_artifact(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "FINAL_MODEL",
            "model_identity": {"model_unique_id": 5},
            "fl_metadata": metadata,
            "file_digests": FILE_DIGESTS,
        }
    )
    assert isinstance(artifact, FinalModelArtifact)
    assert artifact.model_identity.model_unique_id == 5

    metadata["validation_summary"][0]["candidate_weights_digest"] = DIGEST_B
    with pytest.raises(ValidationError, match="candidate digest"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "FINAL_MODEL",
                "model_identity": {"model_unique_id": 5},
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )
    metadata["validation_summary"][0]["candidate_weights_digest"] = DIGEST_C

    metadata["global_gate_accepted"] = False
    with pytest.raises(ValidationError, match="accepted global gate"):
        validate_fl_artifact(
            {
                "bundle_schema_version": "1.0",
                "artifact_role": "FINAL_MODEL",
                "model_identity": {"model_unique_id": 5},
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


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
    metadata = common_metadata()
    metadata.update(
        {
            "previous_model_unique_id": 4,
            "participants": [
                {"participant_nf_instance_id": BRANCH, "training_sample_count": 200}
            ],
            "final_candidate_digest": DIGEST_C,
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
    )
    value = {
        "bundle_schema_version": "1.0",
        "artifact_role": "FINAL_MODEL",
        "model_identity": {"model_unique_id": 5},
        "fl_metadata": metadata,
        "file_digests": FILE_DIGESTS,
    }

    artifact = validate_fl_artifact(value)
    assert isinstance(artifact, FinalModelArtifact)
    assert artifact.fl_metadata.hierarchy_validation.plan_id == PLAN

    metadata["hierarchy_validation"]["branches"][0][
        "branch_nf_instance_id"
    ] = ROOT
    with pytest.raises(ValidationError, match="direct participants"):
        validate_fl_artifact(value)


def test_unknown_artifact_schema_and_incomplete_digest_inventory_fail() -> None:
    metadata = common_metadata()
    metadata.update(
        {
            "round_ind": 0,
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_B,
            "training_sample_count": 120,
            "input_global_weights_digest": DIGEST_A,
        }
    )
    with pytest.raises(ValidationError):
        validate_fl_artifact(
            {
                "bundle_schema_version": "2.0",
                "artifact_role": "ROUND_LOCAL",
                "result_type": "TRAINING",
                "fl_metadata": metadata,
                "file_digests": {"model.npy": DIGEST_C},
            }
        )


def test_hierarchy_assignment_artifact_uses_nested_message_discriminator() -> None:
    base = {
        "bundle_schema_version": "1.0",
        "artifact_role": "HIERARCHY_ASSIGNMENT",
        "file_digests": FILE_DIGESTS,
    }
    branch = validate_fl_artifact(
        {
            **base,
            "hierarchy_metadata": {
                "contract_version": "1.0",
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
    assert branch.hierarchy_metadata.message_type == "BRANCH_ASSIGNMENT"

    leaf = validate_fl_artifact(
        {
            **base,
            "hierarchy_metadata": {
                "contract_version": "1.0",
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
    assert leaf.hierarchy_metadata.message_type == "LEAF_ASSIGNMENT"


def test_hierarchy_result_role_requires_result_metadata() -> None:
    result = validate_fl_artifact(
        {
            "bundle_schema_version": "1.0",
            "artifact_role": "HIERARCHY_PREPARATION_RESULT",
            "file_digests": FILE_DIGESTS,
            "hierarchy_metadata": {
                "contract_version": "1.0",
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
                "bundle_schema_version": "1.0",
                "artifact_role": "HIERARCHY_PREPARATION_RESULT",
                "file_digests": FILE_DIGESTS,
                "hierarchy_metadata": {
                    "contract_version": "1.0",
                    "message_type": "LEAF_ASSIGNMENT",
                    "plan_id": PLAN,
                    "publisher_nf_instance_id": BRANCH,
                    "intended_recipient_nf_instance_id": CLIENT_A,
                    "parent_branch_nf_instance_id": BRANCH,
                    "strategy": hierarchy_strategy(),
                },
            }
        )


def test_complete_manifest_projection_rejects_incompatible_role_fields() -> None:
    manifest = {
        "bundle_schema_version": "1.0",
        "artifact_role": "HIERARCHY_ASSIGNMENT",
        "file_digests": FILE_DIGESTS,
        "fl_metadata": {},
        "hierarchy_metadata": {
            "contract_version": "1.0",
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


def test_tensor_compatibility_allows_float_updates_but_rejects_non_float_changes() -> None:
    base = (
        TensorStateEntry(
            name="weight",
            shape=(2, 2),
            dtype="float32",
            floating=True,
            value_digest=DIGEST_A,
        ),
        TensorStateEntry(
            name="num_batches_tracked",
            shape=(),
            dtype="int64",
            floating=False,
            value_digest=DIGEST_B,
        ),
    )
    candidate = (
        base[0].model_copy(update={"value_digest": DIGEST_C}),
        base[1],
    )
    validate_tensor_compatibility(base, candidate)
    changed_non_float = (
        candidate[0],
        base[1].model_copy(update={"value_digest": DIGEST_C}),
    )
    with pytest.raises(ValueError, match="non-floating tensor state"):
        validate_tensor_compatibility(base, changed_non_float)
