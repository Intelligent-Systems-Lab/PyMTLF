from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from py_mtlf.wire.ml_model_training import FlPolicy, FlReportAfter, FlStrategy


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


class StaticTopologyPolicy(FlPolicy):
    minimum_available_nodes: int | None = Field(
        default=None,
        validation_alias="min_available_nodes",
        serialization_alias="minAvailableNodes",
        ge=1,
    )
    minimum_train_nodes: int | None = Field(
        default=None,
        validation_alias="min_train_nodes",
        serialization_alias="minTrainNodes",
        ge=1,
    )
    minimum_completion_rate: float | None = Field(
        default=None,
        validation_alias="min_completion_rate",
        serialization_alias="minCompletionRate",
        gt=0,
        le=1,
    )
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_complete_policy(self) -> StaticTopologyPolicy:
        required = {
            "allow_additional_candidates": self.allow_additional_candidates,
            "additional_candidate_priority": self.additional_candidate_priority,
            "selection_method": self.selection_method,
            "minimum_available_nodes": self.minimum_available_nodes,
            "fraction_train": self.fraction_train,
            "minimum_train_nodes": self.minimum_train_nodes,
            "accept_failures": self.accept_failures,
            "minimum_completion_rate": self.minimum_completion_rate,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "static topology policy requires: " + ", ".join(missing)
            )
        if self.selection_method not in {"priority", "random"}:
            raise ValueError("selection_method must be priority or random")
        if self.minimum_available_nodes < self.minimum_train_nodes:
            raise ValueError("min_available_nodes must be >= min_train_nodes")
        return self


class StaticTopologyStrategy(FlStrategy):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_supported_strategy(self) -> StaticTopologyStrategy:
        if self.aggregation != "sampleWeighted":
            raise ValueError("aggregation must be sampleWeighted")
        return self


class StaticTopologyReportAfter(FlReportAfter):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class StaticTopologyLeaf(TopologyContractModel):
    nf_instance_id: str
    enabled: StrictBool = True
    priority: int = Field(default=0, ge=0, strict=True)
    report_after: StaticTopologyReportAfter | None = None

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return _normalize_uuid4(value, "Leaf NF instance ID")

    @model_validator(mode="after")
    def validate_leaf_instruction(self) -> StaticTopologyLeaf:
        if self.enabled and self.report_after is None:
            raise ValueError("enabled Leaf requires report_after")
        if self.report_after is not None and self.report_after.unit != "epoch":
            raise ValueError("Leaf report_after unit must be epoch")
        return self


class StaticTopologyBranch(TopologyContractModel):
    nf_instance_id: str
    enabled: StrictBool = True
    priority: int = Field(default=0, ge=0, strict=True)
    report_after: StaticTopologyReportAfter | None = None

    @field_validator("nf_instance_id")
    @classmethod
    def validate_nf_instance_id(cls, value: str) -> str:
        return _normalize_uuid4(value, "Branch NF instance ID")

    @model_validator(mode="after")
    def validate_branch_instruction(self) -> StaticTopologyBranch:
        if self.enabled and self.report_after is None:
            raise ValueError("enabled Branch requires report_after")
        if self.report_after is not None and self.report_after.unit != "round":
            raise ValueError("Branch report_after unit must be round")
        return self


class StaticTopologyBranchGroup(TopologyContractModel):
    branches: tuple[StaticTopologyBranch, ...] = Field(min_length=1)
    policy: StaticTopologyPolicy
    strategy: StaticTopologyStrategy
    leaves: tuple[StaticTopologyLeaf, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> StaticTopologyBranchGroup:
        branch_ids = tuple(branch.nf_instance_id for branch in self.branches)
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Branch candidate NF instance IDs must be unique")
        leaf_ids = tuple(leaf.nf_instance_id for leaf in self.leaves)
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("Leaf NF instance IDs must be unique within a branch group")
        if set(branch_ids).intersection(leaf_ids):
            raise ValueError("Branch and Leaf NF instance IDs must be distinct")
        if not any(branch.enabled for branch in self.branches):
            raise ValueError("branch group requires an enabled Branch candidate")
        if self.policy.selection_method == "priority" and any(
            leaf.enabled and "priority" not in leaf.model_fields_set
            for leaf in self.leaves
        ):
            raise ValueError(
                "enabled Leaf candidates require priority for priority selection"
            )
        enabled_leaves = sum(leaf.enabled for leaf in self.leaves)
        if not self.policy.allow_additional_candidates and (
            self.policy.minimum_available_nodes > enabled_leaves
            or self.policy.minimum_train_nodes > enabled_leaves
        ):
            raise ValueError("branch group policy exceeds enabled explicit Leaves")
        return self


class StaticTopologyFile(TopologyContractModel):
    on_branch_failure: Literal["replace_branch", "reparent_leaves_to_root"]
    policy: StaticTopologyPolicy
    strategy: StaticTopologyStrategy
    branch_groups: tuple[StaticTopologyBranchGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tree_identities(self) -> StaticTopologyFile:
        branch_ids = tuple(
            branch.nf_instance_id
            for group in self.branch_groups
            for branch in group.branches
        )
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Branch NF instance IDs must be globally unique")
        leaf_ids = tuple(
            leaf.nf_instance_id
            for group in self.branch_groups
            for leaf in group.leaves
        )
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("Leaf NF instance IDs must be globally unique")
        if set(branch_ids).intersection(leaf_ids):
            raise ValueError("Branch and Leaf NF instance IDs must be distinct")
        if self.policy.allow_additional_candidates:
            raise ValueError("Root policy does not support additional Branch discovery")
        if self.policy.selection_method == "priority" and any(
            branch.enabled and "priority" not in branch.model_fields_set
            for group in self.branch_groups
            for branch in group.branches
        ):
            raise ValueError(
                "enabled Branch candidates require priority for priority selection"
            )
        enabled_groups = sum(
            any(branch.enabled for branch in group.branches)
            for group in self.branch_groups
        )
        if (
            self.policy.minimum_available_nodes > enabled_groups
            or self.policy.minimum_train_nodes > enabled_groups
        ):
            raise ValueError("Root policy exceeds configured Branch groups")
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


class TopologyBranchGroupAssignment(TopologyContractModel):
    branches: tuple[StaticTopologyBranch, ...] = Field(min_length=1)
    policy: StaticTopologyPolicy
    strategy: StaticTopologyStrategy
    leaves: tuple[StaticTopologyLeaf, ...] = Field(min_length=1)


class TopologyAssignment(TopologyContractModel):
    root_nf_instance_id: str
    on_branch_failure: Literal["replace_branch", "reparent_leaves_to_root"]
    policy: StaticTopologyPolicy
    strategy: StaticTopologyStrategy
    branch_groups: tuple[TopologyBranchGroupAssignment, ...] = Field(min_length=1)


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
            branch.nf_instance_id
            for group in self._topology.branch_groups
            for branch in group.branches
        } | {
            leaf.nf_instance_id
            for group in self._topology.branch_groups
            for leaf in group.leaves
        }
        if root_id in assigned_ids:
            raise TopologyConfigurationError(
                "containing Root NF instance ID must not appear in the topology assignment"
            )
        branch_groups = tuple(
            TopologyBranchGroupAssignment(
                branches=tuple(
                    branch.model_copy(deep=True)
                    for branch in sorted(
                        group.branches,
                        key=lambda item: item.nf_instance_id,
                    )
                ),
                policy=group.policy.model_copy(deep=True),
                strategy=group.strategy.model_copy(deep=True),
                leaves=tuple(
                    leaf.model_copy(deep=True)
                    for leaf in sorted(
                        group.leaves,
                        key=lambda item: item.nf_instance_id,
                    )
                ),
            )
            for group in sorted(
                self._topology.branch_groups,
                key=lambda item: tuple(
                    sorted(branch.nf_instance_id for branch in item.branches)
                ),
            )
        )
        return TopologyAssignment(
            root_nf_instance_id=root_id,
            on_branch_failure=self._topology.on_branch_failure,
            policy=self._topology.policy.model_copy(deep=True),
            strategy=self._topology.strategy.model_copy(deep=True),
            branch_groups=branch_groups,
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
