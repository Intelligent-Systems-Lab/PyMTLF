from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from py_mtlf.core.fl_artifacts import (
    FinalModelArtifact,
    TensorStateEntry,
    WapeComponents,
    validate_fl_artifact,
    validate_tensor_compatibility,
    wape,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
CLIENT_A = "00000000-0000-4000-8000-000000000001"
CLIENT_B = "00000000-0000-4000-8000-000000000002"
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
            "model_identity": {"provider_id": "nwdaf-c", "model_unique_id": 5},
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
                "model_identity": {"provider_id": "nwdaf-c", "model_unique_id": 5},
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
                "model_identity": {"provider_id": "nwdaf-c", "model_unique_id": 5},
                "fl_metadata": metadata,
                "file_digests": FILE_DIGESTS,
            }
        )


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
