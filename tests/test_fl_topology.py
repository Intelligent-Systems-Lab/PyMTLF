from pathlib import Path

import pytest
from pydantic import ValidationError

from py_mtlf.core.fl_topology import StaticTopologyPlanner, TopologyConfigurationError

ROOT_ID = "00000000-0000-4000-8000-000000000001"
BRANCH_A_ID = "00000000-0000-4000-8000-000000000010"
BRANCH_B_ID = "00000000-0000-4000-8000-000000000020"
LEAF_A_ID = "00000000-0000-4000-8000-000000000101"
LEAF_B_ID = "00000000-0000-4000-8000-000000000102"
LEAF_C_ID = "00000000-0000-4000-8000-000000000201"


def write_topology(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_static_topology_planner_canonicalizes_and_freezes_snapshot(tmp_path):
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
version: 1
admission:
  mode: complete_required
branches:
  - nf_instance_id: {BRANCH_B_ID}
    leaves:
      - nf_instance_id: {LEAF_C_ID}
  - nf_instance_id: {BRANCH_A_ID}
    leaves:
      - nf_instance_id: {LEAF_B_ID}
      - nf_instance_id: {LEAF_A_ID}
""",
    )
    planner = StaticTopologyPlanner.load(path)

    assignment = planner.build(root_nf_instance_id=ROOT_ID)

    assert assignment.root_nf_instance_id == ROOT_ID
    assert tuple(branch.nf_instance_id for branch in assignment.branches) == (
        BRANCH_A_ID,
        BRANCH_B_ID,
    )
    assert assignment.branches[0].leaf_nf_instance_ids == (LEAF_A_ID, LEAF_B_ID)
    assert assignment.branches[1].leaf_nf_instance_ids == (LEAF_C_ID,)
    assert assignment.admission_mode == "complete_required"

    write_topology(path, "version: 2")
    assert planner.build(root_nf_instance_id=ROOT_ID) == assignment


@pytest.mark.parametrize(
    "body",
    [
        """
version: 2
admission: {mode: complete_required}
branches: []
""",
        """
version: 1
admission: {mode: partial_allowed}
branches: []
""",
        """
version: 1
admission: {mode: complete_required}
branches: []
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: []
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {LEAF_B_ID}}}]
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
  - nf_instance_id: {BRANCH_B_ID}
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {BRANCH_B_ID}}}]
  - nf_instance_id: {BRANCH_B_ID}
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: not-a-uuid
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
""",
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
""",
        f"""
version: 1
admission: {{mode: complete_required}}
unexpected: true
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {LEAF_A_ID}}}]
""",
    ],
)
def test_static_topology_rejects_invalid_contract(tmp_path, body):
    path = tmp_path / "topology.yaml"
    write_topology(path, body)

    with pytest.raises((ValidationError, TopologyConfigurationError)):
        StaticTopologyPlanner.load(path)


def test_static_topology_rejects_containing_root_collision(tmp_path):
    path = tmp_path / "topology.yaml"
    write_topology(
        path,
        f"""
version: 1
admission: {{mode: complete_required}}
branches:
  - nf_instance_id: {BRANCH_A_ID}
    leaves: [{{nf_instance_id: {ROOT_ID}}}]
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
