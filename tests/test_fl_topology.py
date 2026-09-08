from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.core.fl_topology import (
    StaticFlatTopologyPlanner,
    StaticTopologyPlanner,
    TopologyConfigurationError,
)

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_A_ID = "00000000-0000-4000-8000-000000000010"
BRANCH_B_ID = "00000000-0000-4000-8000-000000000020"
BRANCH_C_ID = "00000000-0000-4000-8000-000000000030"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
LEAF_C_ID = "00000000-0000-4000-8000-000000000201"
CLIENT_A_ID = "00000000-0000-4000-8000-000000000301"
CLIENT_B_ID = "00000000-0000-4000-8000-000000000302"


def write_topology(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_static_topology_planner_canonicalizes_and_freezes_snapshot(tmp_path):
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
admission:
  mode: complete_required
policy:
  allow_additional_candidates: false
  additional_candidate_priority: 0
  selection_method: priority
  min_available_nodes: 1
  fraction_train: 1.0
  min_train_nodes: 1
  accept_failures: true
  min_completion_rate: 0.5
strategy:
  method: fedProx
  aggregation: sampleWeighted
  method_parameters:
    proximal_mu: 0.01
branch_groups:
  - branches:
      - nf_instance_id: {BRANCH_C_ID}
        enabled: false
        priority: 50
      - nf_instance_id: {BRANCH_B_ID}
        priority: 100
        report_after: {{count: 2, unit: round}}
    policy:
      allow_additional_candidates: false
      additional_candidate_priority: 0
      selection_method: priority
      min_available_nodes: 1
      fraction_train: 1.0
      min_train_nodes: 1
      accept_failures: false
      min_completion_rate: 1.0
    strategy:
      method: fedProx
      aggregation: sampleWeighted
      method_parameters: {{proximal_mu: 0.02}}
    leaves:
      - nf_instance_id: {LEAF_C_ID}
        priority: 5
        report_after: {{count: 3, unit: epoch}}
  - branches:
      - nf_instance_id: {BRANCH_A_ID}
        priority: 100
        report_after: {{count: 1, unit: round}}
    policy:
      allow_additional_candidates: false
      additional_candidate_priority: 0
      selection_method: priority
      min_available_nodes: 2
      fraction_train: 1.0
      min_train_nodes: 2
      accept_failures: false
      min_completion_rate: 1.0
    strategy:
      method: fedProx
      aggregation: sampleWeighted
      method_parameters: {{proximal_mu: 0.03}}
    leaves:
      - nf_instance_id: {LEAF_B_ID}
        priority: 10
        report_after: {{count: 4, unit: epoch}}
      - nf_instance_id: {LEAF_A_ID}
        priority: 20
        report_after: {{count: 5, unit: epoch}}
""",
    )
    planner = StaticTopologyPlanner.load(path)

    assignment = planner.build(root_nf_instance_id=ROOT_ID)

    assert assignment.root_nf_instance_id == ROOT_ID
    assert assignment.policy.minimum_available_nodes == 1
    assert assignment.strategy.method_parameters.proximal_mu == 0.01
    assert tuple(
        candidate.nf_instance_id
        for group in assignment.branch_groups
        for candidate in group.branches
    ) == (
        BRANCH_A_ID,
        BRANCH_B_ID,
        BRANCH_C_ID,
    )
    assert tuple(
        leaf.nf_instance_id for leaf in assignment.branch_groups[0].leaves
    ) == (LEAF_A_ID, LEAF_B_ID)
    assert assignment.branch_groups[0].leaves[0].report_after.count == 5
    assert assignment.branch_groups[1].branches[1].enabled is False
    assert assignment.branch_groups[1].strategy.method_parameters.proximal_mu == 0.02
    assert assignment.admission_mode == "complete_required"

    write_topology(path, "unexpected: true")
    assert planner.build(root_nf_instance_id=ROOT_ID) == assignment


@pytest.mark.parametrize(
    "body",
    [
        "version: 1",
        "admission: {mode: partial_allowed}",
        "admission: {mode: complete_required}\nbranch_groups: []",
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: priority, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, report_after: {{count: 1, unit: epoch}}}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {LEAF_A_ID}, report_after: {{count: 1, unit: epoch}}}}
""",
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: priority, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, priority: 100, report_after: {{count: 1, unit: round}}}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {LEAF_A_ID}, report_after: {{count: 1, unit: round}}}}
""",
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: priority, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, enabled: true}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {LEAF_A_ID}, report_after: {{count: 1, unit: epoch}}}}
""",
    ],
)
def test_static_topology_rejects_invalid_contract(tmp_path, body):
    path = tmp_path / "topology.yaml"
    write_topology(path, body)

    with pytest.raises((ValidationError, TopologyConfigurationError)):
        StaticTopologyPlanner.load(path)


@pytest.mark.parametrize("missing_priority_role", ["branch", "leaf"])
def test_priority_selection_requires_explicit_enabled_candidate_priority(
    tmp_path,
    missing_priority_role,
):
    branch_priority = "" if missing_priority_role == "branch" else "priority: 100, "
    leaf_priority = "" if missing_priority_role == "leaf" else "priority: 100, "
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: priority, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, {branch_priority}report_after: {{count: 1, unit: round}}}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {LEAF_A_ID}, {leaf_priority}report_after: {{count: 1, unit: epoch}}}}
""",
    )

    with pytest.raises(ValidationError, match="priority"):
        StaticTopologyPlanner.load(path)


def test_random_selection_allows_omitted_candidate_priority(tmp_path):
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: random, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, report_after: {{count: 1, unit: round}}}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {LEAF_A_ID}, report_after: {{count: 1, unit: epoch}}}}
""",
    )

    assignment = StaticTopologyPlanner.load(path).build(root_nf_instance_id=ROOT_ID)

    assert assignment.branch_groups[0].branches[0].priority == 0
    assert assignment.branch_groups[0].leaves[0].priority == 0


def test_static_topology_rejects_containing_root_collision(tmp_path):
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
admission: {{mode: complete_required}}
policy: &policy
  {{allow_additional_candidates: false, additional_candidate_priority: 0,
   selection_method: priority, min_available_nodes: 1, fraction_train: 1.0,
   min_train_nodes: 1, accept_failures: false, min_completion_rate: 1.0}}
strategy: &strategy
  {{method: fedProx, aggregation: sampleWeighted,
   method_parameters: {{proximal_mu: 0.01}}}}
branch_groups:
  - branches:
      - {{nf_instance_id: {BRANCH_A_ID}, priority: 100, report_after: {{count: 1, unit: round}}}}
    policy: *policy
    strategy: *strategy
    leaves:
      - {{nf_instance_id: {ROOT_ID}, priority: 100, report_after: {{count: 1, unit: epoch}}}}
""",
    )
    planner = StaticTopologyPlanner.load(path)

    with pytest.raises(TopologyConfigurationError, match="containing Root"):
        planner.build(root_nf_instance_id=ROOT_ID)


def test_static_topology_reports_missing_and_non_mapping_files(tmp_path):
    with pytest.raises(TopologyConfigurationError, match="read static topology"):
        StaticTopologyPlanner.load(tmp_path / "missing.yaml")

    path = tmp_path / "topology.yaml"
    write_topology(path, "- version: 1")
    with pytest.raises(TopologyConfigurationError, match="YAML mapping"):
        StaticTopologyPlanner.load(path)


def test_static_flat_topology_canonicalizes_clients_tais_and_version(tmp_path):
    path = tmp_path / "flat.yaml"
    write_topology(
        path,
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001102"
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001100"
""",
    )
    planner = StaticFlatTopologyPlanner.load(path)

    assignment = planner.build(server_nf_instance_id=ROOT_ID)

    assert tuple(client.nf_instance_id for client in assignment.clients) == (
        CLIENT_A_ID,
        CLIENT_B_ID,
    )
    assert tuple(tai.tac for tai in assignment.clients[0].tracking_areas) == (
        "001100",
        "001101",
    )
    assert assignment.topology_version == 1

    reordered = tmp_path / "flat-reordered.yaml"
    write_topology(
        reordered,
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001100"
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001102"
""",
    )
    assert (
        StaticFlatTopologyPlanner.load(reordered)
        .build(server_nf_instance_id=ROOT_ID)
        .topology_version
        == assignment.topology_version
    )


@pytest.mark.parametrize(
    "body",
    [
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
""",
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope: &scope
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
  - nf_instance_id: {CLIENT_A_ID}
    scope: *scope
""",
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - &tai {{plmn_id: {{mcc: "466", mnc: "92"}}, tac: "001101"}}
        - *tai
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001102"
""",
        f"""
version: 1
clients:
  - nf_instance_id: {CLIENT_A_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "46", mnc: "92"}}
          tac: "001101"
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "xyz"
""",
    ],
)
def test_static_flat_topology_rejects_invalid_contract(tmp_path, body):
    path = tmp_path / "flat.yaml"
    write_topology(path, body)

    with pytest.raises((ValidationError, TopologyConfigurationError)):
        StaticFlatTopologyPlanner.load(path)


def test_static_flat_topology_rejects_containing_server_collision(tmp_path):
    path = tmp_path / "flat.yaml"
    write_topology(
        path,
        f"""
version: 1
clients:
  - nf_instance_id: {ROOT_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001101"
  - nf_instance_id: {CLIENT_B_ID}
    scope:
      tracking_areas:
        - plmn_id: {{mcc: "466", mnc: "92"}}
          tac: "001102"
""",
    )

    with pytest.raises(TopologyConfigurationError, match="containing Server"):
        StaticFlatTopologyPlanner.load(path).build(server_nf_instance_id=ROOT_ID)
