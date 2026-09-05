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


class StaticFlatPlmnId(TopologyContractModel):
    mcc: str = Field(pattern=r"^[0-9]{3}$")
    mnc: str = Field(pattern=r"^[0-9]{2,3}$")


class StaticFlatTrackingArea(TopologyContractModel):
    plmn_id: StaticFlatPlmnId
    tac: str = Field(pattern=r"^[0-9A-Fa-f]{6}$")

    @field_validator("tac")
    @classmethod
    def canonicalize_tac(cls, value: str) -> str:
        return value.upper()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.plmn_id.mcc, self.plmn_id.mnc, self.tac)

    def wire_value(self) -> dict[str, object]:
        return {
            "plmnId": {"mcc": self.plmn_id.mcc, "mnc": self.plmn_id.mnc},
            "tac": self.tac,
        }


class StaticFlatClientScope(TopologyContractModel):
    tracking_areas: tuple[StaticFlatTrackingArea, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_tracking_areas(self) -> StaticFlatClientScope:
        keys = tuple(item.key for item in self.tracking_areas)
        if len(keys) != len(set(keys)):
            raise ValueError("Client tracking areas must be unique")
        return self


class StaticFlatClient(TopologyContractModel):
    nf_instance_id: str
    scope: StaticFlatClientScope

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return _normalize_uuid4(value, "Client NF instance ID")


class StaticFlatTopologyFile(TopologyContractModel):
    version: Literal[1]
    clients: tuple[StaticFlatClient, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_clients(self) -> StaticFlatTopologyFile:
        client_ids = tuple(client.nf_instance_id for client in self.clients)
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("Client NF instance IDs must be unique")
        return self


class TopologyBranchAssignment(TopologyContractModel):
    nf_instance_id: str
    leaf_nf_instance_ids: tuple[str, ...] = Field(min_length=1)


class TopologyAssignment(TopologyContractModel):
    root_nf_instance_id: str
    admission_mode: Literal["complete_required"]
    branches: tuple[TopologyBranchAssignment, ...] = Field(min_length=1)


class StaticFlatClientAssignment(TopologyContractModel):
    nf_instance_id: str
    tracking_areas: tuple[StaticFlatTrackingArea, ...] = Field(min_length=1)


class StaticFlatTopologyAssignment(TopologyContractModel):
    server_nf_instance_id: str
    topology_version: int = Field(ge=1)
    clients: tuple[StaticFlatClientAssignment, ...] = Field(min_length=2)


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


class StaticFlatTopologyPlanner:
    def __init__(self, topology: StaticFlatTopologyFile) -> None:
        self._topology = topology

    @classmethod
    def load(cls, path: str | Path) -> StaticFlatTopologyPlanner:
        topology_path = Path(path)
        try:
            with topology_path.open(encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            raise TopologyConfigurationError(
                f"failed to read static flat topology: {topology_path}"
            ) from error
        if not isinstance(raw, dict):
            raise TopologyConfigurationError("static flat topology must be a YAML mapping")
        return cls(StaticFlatTopologyFile.model_validate(raw))

    def build(self, *, server_nf_instance_id: str) -> StaticFlatTopologyAssignment:
        server_id = _normalize_uuid4(
            server_nf_instance_id,
            "containing Server NF instance ID",
        )
        if any(client.nf_instance_id == server_id for client in self._topology.clients):
            raise TopologyConfigurationError(
                "containing Server NF instance ID must not appear in the flat topology"
            )
        return StaticFlatTopologyAssignment(
            server_nf_instance_id=server_id,
            topology_version=self._topology.version,
            clients=tuple(
                StaticFlatClientAssignment(
                    nf_instance_id=client.nf_instance_id,
                    tracking_areas=tuple(
                        sorted(client.scope.tracking_areas, key=lambda item: item.key)
                    ),
                )
                for client in sorted(
                    self._topology.clients,
                    key=lambda item: item.nf_instance_id,
                )
            ),
        )
