import math

import pytest
from pydantic import ValidationError

from py_mtlf.core.fl_hierarchy import (
    BranchAssignmentMetadata,
    FailedClient,
    FederatedStrategy,
    LeafAssignmentMetadata,
    PreparationOutcome,
    PreparationResultMetadata,
    PreparedClient,
    validate_hierarchy_metadata,
)

ROOT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRANCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEAF_A = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEAF_B = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PLAN = "11111111-1111-4111-8111-111111111111"


def strategy() -> dict[str, object]:
    return {
        "algorithm": {"name": "fedprox", "proximal_mu": 0.01},
        "participant_selection": "all",
        "waiting_policy": "all",
        "aggregation": "sample_weighted",
    }


def branch_assignment() -> dict[str, object]:
    return {
        "contract_version": "1.0",
        "message_type": "BRANCH_ASSIGNMENT",
        "plan_id": PLAN,
        "publisher_nf_instance_id": ROOT,
        "intended_recipient_nf_instance_id": BRANCH,
        "assigned_leaf_nf_instance_ids": [LEAF_A, LEAF_B],
        "admission": {"mode": "complete_required"},
        "strategy": strategy(),
    }


def test_fedprox_strategy_accepts_only_first_version_values() -> None:
    value = FederatedStrategy.model_validate(strategy())
    assert value.algorithm.name == "fedprox"
    assert value.algorithm.proximal_mu == 0.01

    for algorithm in ("fedavg", "custom"):
        invalid = strategy()
        invalid["algorithm"] = {"name": algorithm, "proximal_mu": 0.01}
        with pytest.raises(ValidationError):
            FederatedStrategy.model_validate(invalid)

    for proximal_mu in (0, -0.1, math.nan, math.inf, -math.inf):
        invalid = strategy()
        invalid["algorithm"] = {"name": "fedprox", "proximal_mu": proximal_mu}
        with pytest.raises(ValidationError):
            FederatedStrategy.model_validate(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("participant_selection", "fixed_count"),
        ("waiting_policy", "minimum_results"),
        ("aggregation", "uniform"),
    ],
)
def test_strategy_rejects_unimplemented_options(field: str, value: str) -> None:
    invalid = strategy()
    invalid[field] = value
    with pytest.raises(ValidationError):
        FederatedStrategy.model_validate(invalid)

    with pytest.raises(ValidationError):
        FederatedStrategy.model_validate({**strategy(), "parameters": {}})


def test_branch_assignment_normalizes_identities_and_preserves_canonical_order() -> None:
    value = branch_assignment()
    value["publisher_nf_instance_id"] = ROOT.upper()
    assignment = BranchAssignmentMetadata.model_validate(value)

    assert assignment.publisher_nf_instance_id == ROOT
    assert assignment.assigned_leaf_nf_instance_ids == (LEAF_A, LEAF_B)
    assert assignment.model_dump(mode="json")["strategy"] == strategy()


def test_branch_assignment_rejects_invalid_plan_and_leaf_partition() -> None:
    invalid = branch_assignment()
    invalid["plan_id"] = "11111111-1111-1111-8111-111111111111"
    with pytest.raises(ValidationError, match="UUIDv4"):
        BranchAssignmentMetadata.model_validate(invalid)

    invalid = branch_assignment()
    invalid["assigned_leaf_nf_instance_ids"] = [LEAF_B, LEAF_A]
    with pytest.raises(ValidationError, match="canonical"):
        BranchAssignmentMetadata.model_validate(invalid)

    invalid = branch_assignment()
    invalid["assigned_leaf_nf_instance_ids"] = [LEAF_A, LEAF_A]
    with pytest.raises(ValidationError, match="unique"):
        BranchAssignmentMetadata.model_validate(invalid)

    invalid = branch_assignment()
    invalid["assigned_leaf_nf_instance_ids"] = [BRANCH]
    with pytest.raises(ValidationError, match="distinct"):
        BranchAssignmentMetadata.model_validate(invalid)


def test_leaf_assignment_requires_branch_publisher_and_forbids_unknown_fields() -> None:
    value = {
        "contract_version": "1.0",
        "message_type": "LEAF_ASSIGNMENT",
        "plan_id": PLAN,
        "publisher_nf_instance_id": BRANCH,
        "intended_recipient_nf_instance_id": LEAF_A,
        "parent_branch_nf_instance_id": BRANCH,
        "strategy": strategy(),
    }
    assignment = LeafAssignmentMetadata.model_validate(value)
    assert assignment.parent_branch_nf_instance_id == BRANCH

    with pytest.raises(ValidationError, match="parent Branch"):
        LeafAssignmentMetadata.model_validate(
            {**value, "parent_branch_nf_instance_id": ROOT}
        )
    with pytest.raises(ValidationError):
        LeafAssignmentMetadata.model_validate({**value, "assigned_leaves": []})


def test_preparation_result_accepts_ready_and_failed_partitions() -> None:
    common = {
        "contract_version": "1.0",
        "message_type": "PREPARATION_RESULT",
        "plan_id": PLAN,
        "publisher_nf_instance_id": BRANCH,
        "intended_recipient_nf_instance_id": ROOT,
        "assigned_client_nf_instance_ids": [LEAF_A, LEAF_B],
    }
    ready = PreparationResultMetadata.model_validate(
        {
            **common,
            "outcome": "READY",
            "prepared_clients": [
                {"nf_instance_id": LEAF_A},
                {"nf_instance_id": LEAF_B},
            ],
            "failed_clients": [],
            "timed_out_client_nf_instance_ids": [],
        }
    )
    assert ready.outcome is PreparationOutcome.READY

    failed = PreparationResultMetadata.model_validate(
        {
            **common,
            "outcome": "FAILED",
            "prepared_clients": [{"nf_instance_id": LEAF_A}],
            "failed_clients": [
                {"nf_instance_id": LEAF_B, "cause": "REQUIREMENTS_NOT_MET"}
            ],
            "timed_out_client_nf_instance_ids": [],
        }
    )
    assert failed.failed_clients == (
        FailedClient(nf_instance_id=LEAF_B, cause="REQUIREMENTS_NOT_MET"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {
            "prepared_clients": [{"nf_instance_id": LEAF_A}],
            "failed_clients": [],
            "timed_out_client_nf_instance_ids": [],
        },
        {
            "prepared_clients": [{"nf_instance_id": LEAF_A}],
            "failed_clients": [
                {"nf_instance_id": LEAF_A, "cause": "INTERNAL_ERROR"}
            ],
            "timed_out_client_nf_instance_ids": [LEAF_B],
        },
        {
            "prepared_clients": [
                {"nf_instance_id": LEAF_A},
                {"nf_instance_id": LEAF_B},
            ],
            "failed_clients": [],
            "timed_out_client_nf_instance_ids": [],
        },
    ],
)
def test_preparation_result_rejects_invalid_failed_partitions(
    changes: dict[str, object],
) -> None:
    value = {
        "contract_version": "1.0",
        "message_type": "PREPARATION_RESULT",
        "plan_id": PLAN,
        "publisher_nf_instance_id": BRANCH,
        "intended_recipient_nf_instance_id": ROOT,
        "outcome": "FAILED",
        "assigned_client_nf_instance_ids": [LEAF_A, LEAF_B],
        **changes,
    }
    with pytest.raises(ValidationError):
        PreparationResultMetadata.model_validate(value)


def test_preparation_result_rejects_unknown_cause_and_noncanonical_lists() -> None:
    value = {
        "contract_version": "1.0",
        "message_type": "PREPARATION_RESULT",
        "plan_id": PLAN,
        "publisher_nf_instance_id": BRANCH,
        "intended_recipient_nf_instance_id": ROOT,
        "outcome": "FAILED",
        "assigned_client_nf_instance_ids": [LEAF_A, LEAF_B],
        "prepared_clients": [PreparedClient(nf_instance_id=LEAF_A)],
        "failed_clients": [{"nf_instance_id": LEAF_B, "cause": "RAW_EXCEPTION"}],
        "timed_out_client_nf_instance_ids": [],
    }
    with pytest.raises(ValidationError):
        PreparationResultMetadata.model_validate(value)

    value["assigned_client_nf_instance_ids"] = [LEAF_B, LEAF_A]
    value["failed_clients"] = [
        {"nf_instance_id": LEAF_B, "cause": "INTERNAL_ERROR"}
    ]
    with pytest.raises(ValidationError, match="canonical"):
        PreparationResultMetadata.model_validate(value)


def test_hierarchy_metadata_union_is_discriminated_and_fail_closed() -> None:
    assert isinstance(validate_hierarchy_metadata(branch_assignment()), BranchAssignmentMetadata)
    with pytest.raises(ValidationError):
        validate_hierarchy_metadata({**branch_assignment(), "message_type": "UNKNOWN"})
