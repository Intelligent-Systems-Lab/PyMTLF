from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TopologyConfigurationError(ValueError):
    pass


class TopologyContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _normalize_uuid4(value: str, name: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a UUIDv4")
    return str(parsed)


class StaticTopologyLeaf(TopologyContractModel):
    nf_instance_id: str

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return _normalize_uuid4(value, "Leaf NF instance ID")


class StaticTopologyBranch(TopologyContractModel):
    nf_instance_id: str
    leaves: tuple[StaticTopologyLeaf, ...] = Field(min_length=1)

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return _normalize_uuid4(value, "Branch NF instance ID")

    @model_validator(mode="after")
    def validate_unique_leaves(self) -> StaticTopologyBranch:
        leaf_ids = tuple(leaf.nf_instance_id for leaf in self.leaves)
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("Leaf NF instance IDs must be unique")
        if self.nf_instance_id in leaf_ids:
            raise ValueError("Branch and Leaf NF instance IDs must be distinct")
        return self


class CompleteRequiredTopologyAdmission(TopologyContractModel):
    mode: Literal["complete_required"]


class StaticTopologyFile(TopologyContractModel):
    version: Literal[1]
    admission: CompleteRequiredTopologyAdmission
    branches: tuple[StaticTopologyBranch, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tree_identities(self) -> StaticTopologyFile:
        branch_ids = tuple(branch.nf_instance_id for branch in self.branches)
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Branch NF instance IDs must be unique")
        leaf_ids = tuple(
            leaf.nf_instance_id for branch in self.branches for leaf in branch.leaves
        )
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("Leaf NF instance IDs must be globally unique")
        if set(branch_ids).intersection(leaf_ids):
            raise ValueError("Branch and Leaf NF instance IDs must be distinct")
        return self


class TopologyBranchAssignment(TopologyContractModel):
    nf_instance_id: str
    leaf_nf_instance_ids: tuple[str, ...] = Field(min_length=1)


class TopologyAssignment(TopologyContractModel):
    root_nf_instance_id: str
    admission_mode: Literal["complete_required"]
    branches: tuple[TopologyBranchAssignment, ...] = Field(min_length=1)


class TopologyPlanner(Protocol):
    def build(self, *, root_nf_instance_id: str) -> TopologyAssignment: ...


class StaticTopologyPlanner:
    def __init__(self, topology: StaticTopologyFile) -> None:
        self._topology = topology

    @classmethod
    def load(cls, path: str | Path) -> StaticTopologyPlanner:
        topology_path = Path(path)
        try:
            with topology_path.open(encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            raise TopologyConfigurationError(
                f"failed to read static topology: {topology_path}"
            ) from error
        if not isinstance(raw, dict):
            raise TopologyConfigurationError("static topology must be a YAML mapping")
        return cls(StaticTopologyFile.model_validate(raw))

    def build(self, *, root_nf_instance_id: str) -> TopologyAssignment:
        root_id = _normalize_uuid4(root_nf_instance_id, "containing Root NF instance ID")
        assigned_ids = {
            branch.nf_instance_id for branch in self._topology.branches
        } | {
            leaf.nf_instance_id
            for branch in self._topology.branches
            for leaf in branch.leaves
        }
        if root_id in assigned_ids:
            raise TopologyConfigurationError(
                "containing Root NF instance ID must not appear in the topology assignment"
            )
        branches = tuple(
            TopologyBranchAssignment(
                nf_instance_id=branch.nf_instance_id,
                leaf_nf_instance_ids=tuple(
                    sorted(leaf.nf_instance_id for leaf in branch.leaves)
                ),
            )
            for branch in sorted(
                self._topology.branches,
                key=lambda item: item.nf_instance_id,
            )
        )
        return TopologyAssignment(
            root_nf_instance_id=root_id,
            admission_mode=self._topology.admission.mode,
            branches=branches,
        )
