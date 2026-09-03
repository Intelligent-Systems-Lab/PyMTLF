from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from py_mtlf.core.fl_candidate_orchestration import (
    CandidatePool,
    CandidateProvenance,
    CandidateRelationshipStatus,
    CandidateStatusCause,
    LocalContractDefaults,
    LocalExecutionRole,
    client_local_work,
    intermediate_local_work,
    resolve_effective_node_contract,
)
from py_mtlf.core.fl_hierarchy_discovery import (
    HierarchyDiscoveryScope,
    HierarchyDiscoverySnapshot,
    HierarchyNodeRole,
    ResolvedHierarchyNode,
)
from py_mtlf.wire.ml_model_training import (
    FlPolicy,
    FlReportAfter,
    FlStrategy,
    FlTopologyNode,
    FlTopologyReport,
    RequirementsError,
)
from py_mtlf.wire.private import SelectedTarget

ROOT = "00000000-0000-4000-8000-000000000001"
CHILD_A = "00000000-0000-4000-8000-000000000010"
CHILD_B = "00000000-0000-4000-8000-000000000011"
CHILD_C = "00000000-0000-4000-8000-000000000012"
CHILD_D = "00000000-0000-4000-8000-000000000013"
CHILD_E = "00000000-0000-4000-8000-000000000014"
CHILD_F = "00000000-0000-4000-8000-000000000015"


def policy(**updates) -> FlPolicy:
    values = {
        "allowAdditionalCandidates": False,
        "additionalCandidatePriority": 0,
        "selectionMethod": "priority",
        "minAvailableNodes": 2,
        "fractionTrain": 0.5,
        "minTrainNodes": 1,
        "acceptFailures": True,
        "minCompletionRate": 0.5,
    }
    values.update(updates)
    return FlPolicy.model_validate(values)


def strategy() -> FlStrategy:
    return FlStrategy.model_validate(
        {
            "method": "fedProx",
            "aggregation": "sampleWeighted",
            "methodParameters": {"proximalMu": 0.0},
        }
    )


def node(*, children=None, node_policy=None, report_count=2) -> FlTopologyNode:
    return FlTopologyNode.model_validate(
        {
            "nfInstanceId": ROOT,
            "policy": (node_policy or policy()).model_dump(
                by_alias=True, exclude_none=True
            ),
            "strategy": strategy().model_dump(by_alias=True),
            "reportAfter": {"count": report_count, "unit": "round"},
            **({"children": children} if children is not None else {}),
        }
    )


def child(identity: str, priority: int, *, enabled: bool = True) -> dict:
    return {"nfInstanceId": identity, "enabled": enabled, "priority": priority}


def resolved(
    identity: str,
    role: HierarchyNodeRole = HierarchyNodeRole.LEAF,
) -> ResolvedHierarchyNode:
    return ResolvedHierarchyNode(
        nf_instance_id=identity,
        role=role,
        target=SelectedTarget(
            nfInstanceId=identity,
            nfServiceInstanceId=f"training-{identity}",
            serviceName="nnwdaf-mlmodeltraining",
            apiRoot=f"http://{identity}.example",
            selectionSource="NRF",
        ),
    )


def discovery_scope(
    *,
    tracking_area: tuple[str, str, str] = ("001", "01", "000001"),
) -> HierarchyDiscoveryScope:
    return HierarchyDiscoveryScope(
        containing_nf_instance_id=ROOT,
        role=HierarchyNodeRole.LEAF,
        ml_event="UE_COMMUNICATION",
        model_interoperability="001122",
        tracking_areas=(tracking_area,),
    )


def discovery_snapshot(
    identities: tuple[str, ...],
    *,
    observed_at: datetime,
    validity_period: int = 60,
    complete_nf_instance_count: int | None = None,
    scope: HierarchyDiscoveryScope | None = None,
) -> HierarchyDiscoverySnapshot:
    return HierarchyDiscoverySnapshot(
        scope=scope or discovery_scope(),
        nodes=tuple(resolved(identity) for identity in identities),
        observed_at=observed_at,
        validity_period=validity_period,
        returned_count=len(identities),
        complete_nf_instance_count=complete_nf_instance_count,
    )


def apply_discovery(
    pool: CandidatePool,
    nodes: tuple[ResolvedHierarchyNode, ...],
    *,
    observed_at: datetime | None = None,
    scope: HierarchyDiscoveryScope | None = None,
) -> None:
    observed = observed_at or datetime.now(UTC)
    selected_scope = scope or discovery_scope()
    refresh = pool.request_discovery_refresh(selected_scope)
    assert refresh is not None
    assert pool.complete_discovery_refresh(
        refresh,
        HierarchyDiscoverySnapshot(
            scope=selected_scope,
            nodes=nodes,
            observed_at=observed,
            validity_period=3600,
            returned_count=len(nodes),
            complete_nf_instance_count=None,
        ),
    )


def contract(value: FlTopologyNode):
    return resolve_effective_node_contract(
        value,
        role=LocalExecutionRole.INTERMEDIATE,
        defaults=LocalContractDefaults(),
    )


def activate(pool: CandidatePool, identities: tuple[str, ...]) -> None:
    intents = pool.next_establishment_intents(len(identities))
    assert {intent.nf_instance_id for intent in intents} == set(identities)
    for intent in intents:
        assert pool.complete_establishment(
            intent.nf_instance_id,
            intent.revision,
            resource_location=f"http://child.example/{intent.nf_instance_id}",
        )


def test_effective_contract_applies_protocol_and_local_defaults_without_expanding_authority():
    value = FlTopologyNode.model_validate(
        {
            "nfInstanceId": ROOT,
            "policy": {"minAvailableNodes": 2},
            "strategy": strategy().model_dump(by_alias=True),
        }
    )
    effective = resolve_effective_node_contract(
        value,
        role=LocalExecutionRole.INTERMEDIATE,
        defaults=LocalContractDefaults(
            policy=policy(
                allowAdditionalCandidates=True,
                additionalCandidatePriority=9,
            ),
            report_after=FlReportAfter(count=3, unit="round"),
        ),
    )

    assert effective.policy.allow_additional_candidates is False
    assert effective.policy.additional_candidate_priority == 0
    assert effective.policy.minimum_available_nodes == 2
    assert effective.policy.minimum_train_nodes == 1
    assert effective.report_after.count == 3
    assert effective.strategy.proximal_mu == 0
    assert intermediate_local_work(effective).lower_round_count == 3


def test_effective_contract_detaches_explicit_children_from_mutable_wire_value():
    value = node(children=[child(CHILD_A, 3)])
    effective = contract(value)

    value.children[0].priority = 9

    assert effective.explicit_children[0].priority == 3


def test_leaf_effective_contract_maps_report_after_and_strategy_to_local_work():
    value = FlTopologyNode.model_validate(
        {
            "nfInstanceId": ROOT,
            "policy": policy().model_dump(by_alias=True),
            "strategy": strategy().model_dump(by_alias=True),
            "reportAfter": {"count": 4, "unit": "epoch"},
        }
    )
    effective = resolve_effective_node_contract(
        value,
        role=LocalExecutionRole.LEAF,
        defaults=LocalContractDefaults(),
    )

    work = client_local_work(effective)
    assert work.epochs == 4
    assert work.proximal_mu == 0


@pytest.mark.parametrize(
    ("role", "unit"),
    [
        (LocalExecutionRole.LEAF, "round"),
        (LocalExecutionRole.INTERMEDIATE, "epoch"),
    ],
)
def test_effective_contract_rejects_report_after_unit_for_wrong_local_role(role, unit):
    value = FlTopologyNode.model_validate(
        {
            "nfInstanceId": ROOT,
            "policy": policy().model_dump(by_alias=True),
            "strategy": strategy().model_dump(by_alias=True),
            "reportAfter": {"count": 1, "unit": unit},
        }
    )

    with pytest.raises(RequirementsError) as error:
        resolve_effective_node_contract(
            value,
            role=role,
            defaults=LocalContractDefaults(),
        )

    assert error.value.violations[0].parameter == "reportAfter.unit"


def test_effective_contract_rejects_unknown_executor_and_missing_required_default():
    value = FlTopologyNode.model_validate(
        {
            "nfInstanceId": ROOT,
            "policy": {
                "selectionMethod": "custom",
                "minAvailableNodes": 1,
                "fractionTrain": 1,
                "minTrainNodes": 1,
                "acceptFailures": False,
            },
        }
    )

    with pytest.raises(RequirementsError) as error:
        resolve_effective_node_contract(
            value,
            role=LocalExecutionRole.INTERMEDIATE,
            defaults=LocalContractDefaults(),
        )

    paths = {item.parameter for item in error.value.violations}
    assert paths == {"policy.selectionMethod", "strategy", "reportAfter"}


def test_hybrid_pool_preserves_both_provenances_and_explicit_priority():
    effective = contract(
        node(
            node_policy=policy(
                allowAdditionalCandidates=True,
                additionalCandidatePriority=3,
            ),
            children=[child(CHILD_A, 9), child(CHILD_B, 4)],
        )
    )
    pool = CandidatePool(effective, random_source=random.Random(3))
    apply_discovery(pool, (resolved(CHILD_A), resolved(CHILD_C)))

    records = {record.nf_instance_id: record for record in pool.records()}
    assert records[CHILD_A].provenance == {
        CandidateProvenance.UPSTREAM_ASSIGNED,
        CandidateProvenance.LOCALLY_DISCOVERED,
    }
    assert records[CHILD_A].priority == 9
    assert records[CHILD_C].provenance == {CandidateProvenance.LOCALLY_DISCOVERED}
    assert records[CHILD_C].priority == 3


def test_priority_establishment_is_bounded_and_stale_completion_is_fenced():
    times = iter(
        datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=index)
        for index in range(20)
    )
    pool = CandidatePool(
        contract(node(children=[child(CHILD_A, 1), child(CHILD_B, 7)])),
        random_source=random.Random(7),
        clock=lambda: next(times),
    )

    first = pool.next_establishment_intents(1)
    assert first[0].nf_instance_id == CHILD_B
    assert pool.complete_establishment(CHILD_B, first[0].revision - 1) is False
    assert pool.complete_establishment(
        CHILD_B,
        first[0].revision,
        resource_location="http://child-b.example/subscriptions/1",
    )
    record = {item.nf_instance_id: item for item in pool.records()}[CHILD_B]
    assert record.status is CandidateRelationshipStatus.ACTIVE
    assert record.status_cause is None


def test_only_active_candidates_are_round_participants():
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(
                    minAvailableNodes=1,
                    minTrainNodes=1,
                    fractionTrain=1,
                ),
                children=[child(CHILD_A, 2), child(CHILD_B, 1)],
            )
        )
    )
    assert pool.ready() is False

    establishment = pool.next_establishment_intents(1)[0]
    assert pool.complete_establishment(
        establishment.nf_instance_id,
        establishment.revision,
    )

    selection = pool.select_round()
    assert selection.participant_nf_instance_ids == (establishment.nf_instance_id,)
    assert any(
        item.status is CandidateRelationshipStatus.UNCONFIRMED
        for item in pool.records()
    )


def test_explicit_subtree_requires_dual_role_candidate_resolution():
    subtree = child(CHILD_A, 4)
    subtree["children"] = [child(CHILD_B, 2)]
    pool = CandidatePool(contract(node(children=[subtree])))
    intent = pool.next_establishment_intents(1)[0]

    assert intent.role is HierarchyNodeRole.BRANCH
    with pytest.raises(ValueError, match="role"):
        pool.set_resolved(resolved(CHILD_A))
    pool.set_resolved(resolved(CHILD_A, HierarchyNodeRole.BRANCH))


def test_explicit_candidate_does_not_reuse_expired_exact_discovery_target():
    now = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
    current = [now]
    pool = CandidatePool(
        contract(node(children=[child(CHILD_A, 4)])),
        clock=lambda: current[0],
    )
    exact_scope = HierarchyDiscoveryScope(
        containing_nf_instance_id=ROOT,
        role=HierarchyNodeRole.LEAF,
        ml_event="UE_COMMUNICATION",
        model_interoperability="001122",
        tracking_areas=(),
        target_nf_instance_id=CHILD_A,
    )
    candidate = resolved(CHILD_A)
    pool.set_resolved(
        ResolvedHierarchyNode(
            nf_instance_id=candidate.nf_instance_id,
            role=candidate.role,
            target=candidate.target,
            discovery_scope=exact_scope,
            observed_at=now,
            valid_until=now + timedelta(seconds=60),
        )
    )

    current[0] += timedelta(seconds=61)
    intent = pool.next_establishment_intents(1)[0]

    assert intent.nf_instance_id == CHILD_A
    assert intent.target is None


def test_delegated_discovery_only_adds_direct_fl_clients():
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(allowAdditionalCandidates=True),
            )
        )
    )

    apply_discovery(
        pool,
        (
            resolved(CHILD_A, HierarchyNodeRole.BRANCH),
            resolved(CHILD_B),
        ),
    )

    assert [record.nf_instance_id for record in pool.records()] == [CHILD_B]


def test_repeated_discovery_does_not_stale_inflight_establishment():
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(
                    allowAdditionalCandidates=True,
                    minAvailableNodes=1,
                )
            )
        )
    )
    candidate = resolved(CHILD_A)
    apply_discovery(pool, (candidate,))
    intent = pool.next_establishment_intents(1)[0]

    assert pool.request_discovery_refresh(discovery_scope()) is None

    assert pool.complete_establishment(CHILD_A, intent.revision) is True
    assert pool.records()[0].status is CandidateRelationshipStatus.ACTIVE


def test_readiness_selection_formula_and_completion_boundaries():
    effective = contract(
        node(
            node_policy=policy(
                minAvailableNodes=3,
                minTrainNodes=2,
                fractionTrain=0.5,
                minCompletionRate=0.5,
            ),
            children=[
                child(CHILD_A, 4),
                child(CHILD_B, 3),
                child(CHILD_C, 2),
                child(CHILD_D, 1),
            ],
        )
    )
    pool = CandidatePool(effective, random_source=random.Random(1))
    intents = pool.next_establishment_intents(4)
    for intent in intents[:2]:
        pool.complete_establishment(intent.nf_instance_id, intent.revision)
    assert pool.ready() is False
    with pytest.raises(RuntimeError, match="not ready"):
        pool.select_round()
    pool.complete_establishment(intents[2].nf_instance_id, intents[2].revision)
    pool.complete_establishment(intents[3].nf_instance_id, intents[3].revision)

    selection = pool.select_round()
    assert selection.participant_nf_instance_ids == (CHILD_A, CHILD_B)
    rejected = pool.evaluate_completion(selection, ())
    boundary = pool.evaluate_completion(selection, (CHILD_A,))
    assert rejected.accepted is False
    assert boundary.accepted is True
    assert boundary.failed_nf_instance_ids == (CHILD_B,)


@pytest.mark.parametrize(
    ("fraction_train", "minimum_train", "expected_count"),
    [
        (0.25, 1, 1),
        (0.25, 2, 2),
        (1.0, 1, 4),
    ],
)
def test_round_selection_count_uses_fraction_floor_minimum_and_active_cap(
    fraction_train,
    minimum_train,
    expected_count,
):
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(
                    minAvailableNodes=minimum_train,
                    minTrainNodes=minimum_train,
                    fractionTrain=fraction_train,
                ),
                children=[
                    child(CHILD_A, 4),
                    child(CHILD_B, 3),
                    child(CHILD_C, 2),
                    child(CHILD_D, 1),
                ],
            )
        ),
        random_source=random.Random(1),
    )
    activate(pool, (CHILD_A, CHILD_B, CHILD_C, CHILD_D))

    assert len(pool.select_round().participant_nf_instance_ids) == expected_count


@pytest.mark.parametrize(
    ("accept_failures", "minimum_rate", "successful", "accepted"),
    [
        (False, 0.5, (), False),
        (False, 0.5, (CHILD_A, CHILD_B), True),
        (True, 0.5, (), False),
        (True, 0.5, (CHILD_A,), True),
        (True, 0.75, (CHILD_A,), False),
    ],
)
def test_completion_policy_covers_zero_boundary_and_all_success(
    accept_failures,
    minimum_rate,
    successful,
    accepted,
):
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(
                    minAvailableNodes=2,
                    minTrainNodes=2,
                    fractionTrain=1,
                    acceptFailures=accept_failures,
                    minCompletionRate=minimum_rate,
                ),
                children=[child(CHILD_A, 2), child(CHILD_B, 1)],
            )
        )
    )
    activate(pool, (CHILD_A, CHILD_B))
    selection = pool.select_round()

    assert pool.evaluate_completion(selection, successful).accepted is accepted


def test_seeded_random_selection_is_reproducible():
    effective = contract(
        node(
            node_policy=policy(
                selectionMethod="random",
                minAvailableNodes=4,
                minTrainNodes=2,
                fractionTrain=0.5,
            ),
            children=[
                child(CHILD_A, 4),
                child(CHILD_B, 3),
                child(CHILD_C, 2),
                child(CHILD_D, 1),
            ],
        )
    )
    pools = [CandidatePool(effective, random_source=random.Random(17)) for _ in range(2)]
    for pool in pools:
        activate(pool, (CHILD_A, CHILD_B, CHILD_C, CHILD_D))

    assert pools[0].select_round().participant_nf_instance_ids == (
        pools[1].select_round().participant_nf_instance_ids
    )


def test_disabled_explicit_candidate_blocks_discovery_and_emits_delete_intent():
    effective = contract(
        node(
            node_policy=policy(allowAdditionalCandidates=True),
            children=[child(CHILD_A, 2)],
        )
    )
    pool = CandidatePool(effective)
    intent = pool.next_establishment_intents(1)[0]
    pool.complete_establishment(
        CHILD_A,
        intent.revision,
        resource_location="http://child-a.example/subscriptions/1",
    )

    outcome = pool.reconcile(
        (FlTopologyNode.model_validate(child(CHILD_A, 2, enabled=False)),)
    )
    apply_discovery(pool, (resolved(CHILD_A),))
    record = pool.records()[0]

    assert len(outcome.delete_intents) == 1
    assert outcome.delete_intents[0].resource_location.endswith("/1")
    assert record.status is CandidateRelationshipStatus.INACTIVE
    assert record.status_cause == CandidateStatusCause.REMOVED_BY_POLICY.value
    assert pool.next_establishment_intents(1) == ()

    assert pool.complete_delete(CHILD_A, outcome.delete_intents[0].revision)
    pool.reconcile((FlTopologyNode.model_validate(child(CHILD_A, 2)),))
    assert pool.records()[0].status is CandidateRelationshipStatus.UNCONFIRMED
    assert pool.next_establishment_intents(1)[0].nf_instance_id == CHILD_A


def test_upstream_omission_preserves_local_provenance_and_uses_local_priority():
    effective = contract(
        node(
            node_policy=policy(
                allowAdditionalCandidates=True,
                additionalCandidatePriority=6,
            ),
            children=[child(CHILD_A, 9)],
        )
    )
    pool = CandidatePool(effective)
    apply_discovery(pool, (resolved(CHILD_A),))
    activate(pool, (CHILD_A,))
    active = pool.records()[0]

    outcome = pool.reconcile(())
    record = pool.records()[0]

    assert outcome.delete_intents == ()
    assert record.provenance == {CandidateProvenance.LOCALLY_DISCOVERED}
    assert record.enabled is True
    assert record.priority == 6
    assert record.status is CandidateRelationshipStatus.ACTIVE
    assert record.status_timestamp == active.status_timestamp


def test_additional_authority_revocation_keeps_rediscovered_inactive_candidate_inactive():
    enabled_contract = contract(
        node(
            node_policy=policy(
                allowAdditionalCandidates=True,
                additionalCandidatePriority=6,
                minAvailableNodes=1,
            )
        )
    )
    pool = CandidatePool(enabled_contract)
    apply_discovery(pool, (resolved(CHILD_A),))
    activate(pool, (CHILD_A,))

    disabled_contract = contract(
        node(
            node_policy=policy(
                allowAdditionalCandidates=False,
                minAvailableNodes=1,
            )
        )
    )
    outcome = pool.reconfigure(disabled_contract)
    assert len(outcome.delete_intents) == 1
    delete = outcome.delete_intents[0]
    assert pool.complete_delete(CHILD_A, delete.revision - 1) is False
    assert pool.complete_delete(CHILD_A, delete.revision) is True
    assert pool.records()[0].status is CandidateRelationshipStatus.INACTIVE

    pool.reconfigure(enabled_contract)
    apply_discovery(pool, (resolved(CHILD_A),))
    record = pool.records()[0]
    assert record.enabled is True
    assert record.status is CandidateRelationshipStatus.INACTIVE
    assert pool.next_establishment_intents(1) == ()


def test_delete_failure_preserves_retry_intent_until_cleanup_succeeds():
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(minAvailableNodes=1),
                children=[child(CHILD_A, 2)],
            )
        )
    )
    activate(pool, (CHILD_A,))

    first = pool.deactivate(
        CHILD_A,
        cause=CandidateStatusCause.OTHER,
    )
    assert first is not None
    assert pool.complete_delete(
        CHILD_A,
        first.revision,
        cause=CandidateStatusCause.COMMUNICATION_FAILURE,
    )
    retry = pool.pending_delete_intents()
    assert len(retry) == 1
    assert retry[0].resource_location.endswith(CHILD_A)
    assert retry[0].revision > first.revision
    assert pool.complete_delete(CHILD_A, first.revision) is False
    assert pool.complete_delete(CHILD_A, retry[0].revision) is True
    assert pool.pending_delete_intents() == ()


def test_idempotent_reconcile_does_not_change_relationship_timestamp():
    times = iter(
        datetime(2026, 9, 4, tzinfo=UTC) + timedelta(seconds=index)
        for index in range(20)
    )
    child_value = FlTopologyNode.model_validate(child(CHILD_A, 2))
    pool = CandidatePool(
        contract(node(children=[child(CHILD_A, 2)])),
        clock=lambda: next(times),
    )
    before = pool.records()[0]

    pool.reconcile((child_value,))
    after = pool.records()[0]

    assert after.status_timestamp == before.status_timestamp
    assert after.revision == before.revision


def test_snapshot_is_stable_and_preserves_unknown_descendant_values():
    pool = CandidatePool(contract(node(children=[child(CHILD_B, 2), child(CHILD_A, 3)])))
    intent = pool.next_establishment_intents(1)[0]
    pool.complete_establishment(intent.nf_instance_id, intent.revision)
    pool.attach_child_report(
        CHILD_A,
        FlTopologyReport.model_validate(
            {
                "nfInstanceId": CHILD_A,
                "children": [
                    {
                        "nfInstanceId": CHILD_D,
                        "status": "VENDOR_DRAINING",
                        "statusTimestamp": "2026-09-04T09:00:00Z",
                        "statusCause": "VENDOR_MAINTENANCE",
                    }
                ],
            }
        ),
    )

    first = pool.snapshot().model_dump(by_alias=True, exclude_none=True, mode="json")
    second = pool.snapshot().model_dump(by_alias=True, exclude_none=True, mode="json")

    assert first == second
    assert [item["nfInstanceId"] for item in first["children"]] == [CHILD_A, CHILD_B]
    assert first["children"][0]["children"][0]["status"] == "VENDOR_DRAINING"
    assert first["children"][0]["children"][0]["statusCause"] == (
        "VENDOR_MAINTENANCE"
    )


def test_discovery_freshness_gates_local_candidate_establishment():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    current = [now]
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(
                    allowAdditionalCandidates=True,
                    minAvailableNodes=1,
                )
            )
        ),
        clock=lambda: current[0],
    )
    scope = discovery_scope()
    refresh = pool.request_discovery_refresh(scope)
    assert refresh is not None
    assert pool.complete_discovery_refresh(
        refresh,
        discovery_snapshot((CHILD_A, CHILD_B), observed_at=now),
    )

    intent = pool.next_establishment_intents(1)
    assert len(intent) == 1
    remaining = {
        item.nf_instance_id
        for item in pool.records()
        if item.status is CandidateRelationshipStatus.UNCONFIRMED
    }
    assert len(remaining) == 1
    current[0] = now + timedelta(seconds=61)
    assert pool.next_establishment_intents(1) == ()
    assert pool.request_discovery_refresh(scope) is not None


def test_scope_change_requires_refresh_even_while_previous_snapshot_is_fresh():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    pool = CandidatePool(
        contract(node(node_policy=policy(allowAdditionalCandidates=True))),
        clock=lambda: now,
    )
    first_scope = discovery_scope()
    refresh = pool.request_discovery_refresh(first_scope)
    assert refresh is not None
    assert pool.complete_discovery_refresh(
        refresh,
        discovery_snapshot((CHILD_A,), observed_at=now, scope=first_scope),
    )

    assert pool.request_discovery_refresh(first_scope) is None
    assert pool.request_discovery_refresh(
        discovery_scope(tracking_area=("001", "01", "000002"))
    ) is not None


def test_complete_refresh_prunes_only_absent_local_unconfirmed_candidates():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    current = [now]
    pool = CandidatePool(
        contract(
            node(
                node_policy=policy(allowAdditionalCandidates=True),
                children=[child(CHILD_D, 9, enabled=False)],
            )
        ),
        random_source=random.Random(11),
        clock=lambda: current[0],
    )
    scope = discovery_scope()
    first = pool.request_discovery_refresh(scope)
    assert first is not None
    assert pool.complete_discovery_refresh(
        first,
        discovery_snapshot(
            (CHILD_A, CHILD_B, CHILD_C, CHILD_E, CHILD_F),
            observed_at=now,
            scope=scope,
        ),
    )
    intents = pool.next_establishment_intents(4)
    active_id = intents[0].nf_instance_id
    failed_id = intents[1].nf_instance_id
    inactive_id = intents[2].nf_instance_id
    deploying_id = intents[3].nf_instance_id
    unconfirmed_id = next(
        item.nf_instance_id
        for item in pool.records()
        if item.status is CandidateRelationshipStatus.UNCONFIRMED
        and item.nf_instance_id != CHILD_D
    )
    assert pool.complete_establishment(active_id, intents[0].revision)
    assert pool.complete_establishment(
        failed_id,
        intents[1].revision,
        failure_cause=CandidateStatusCause.COMMUNICATION_FAILURE,
    )
    assert pool.deactivate(inactive_id) is not None

    second = pool.request_discovery_refresh(scope)
    assert second is None
    current[0] = now + timedelta(seconds=61)
    second = pool.request_discovery_refresh(scope)
    assert second is not None
    assert pool.complete_discovery_refresh(
        second,
        discovery_snapshot((CHILD_D,), observed_at=current[0], scope=scope),
    )

    records = {item.nf_instance_id: item for item in pool.records()}
    assert unconfirmed_id not in records
    assert records[active_id].status is CandidateRelationshipStatus.ACTIVE
    assert records[failed_id].status is CandidateRelationshipStatus.FAILED
    assert records[inactive_id].status is CandidateRelationshipStatus.INACTIVE
    assert records[deploying_id].status is CandidateRelationshipStatus.DEPLOYING
    assert records[CHILD_D].provenance == {CandidateProvenance.UPSTREAM_ASSIGNED}


def test_partial_refresh_does_not_prune_absent_candidate():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    current = [now]
    pool = CandidatePool(
        contract(node(node_policy=policy(allowAdditionalCandidates=True))),
        clock=lambda: current[0],
    )
    scope = discovery_scope()
    first = pool.request_discovery_refresh(scope)
    assert first is not None
    assert pool.complete_discovery_refresh(
        first,
        discovery_snapshot((CHILD_A, CHILD_B), observed_at=now, scope=scope),
    )

    current[0] += timedelta(seconds=61)
    second = pool.request_discovery_refresh(scope)
    assert second is not None
    assert pool.complete_discovery_refresh(
        second,
        discovery_snapshot(
            (CHILD_A,),
            observed_at=current[0],
            complete_nf_instance_count=2,
            scope=scope,
        ),
    )

    assert {item.nf_instance_id for item in pool.records()} == {CHILD_A, CHILD_B}
    assert tuple(
        item.nf_instance_id for item in pool.next_establishment_intents(2)
    ) == (CHILD_A,)


def test_failed_and_late_refresh_do_not_replace_current_snapshot_or_pool():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    current = [now]
    pool = CandidatePool(
        contract(node(node_policy=policy(allowAdditionalCandidates=True))),
        clock=lambda: current[0],
    )
    scope = discovery_scope()
    first = pool.request_discovery_refresh(scope)
    assert first is not None
    assert pool.complete_discovery_refresh(
        first,
        discovery_snapshot((CHILD_A,), observed_at=now, scope=scope),
    )
    baseline = pool.discovery_snapshot()

    current[0] += timedelta(seconds=61)
    failed = pool.request_discovery_refresh(scope)
    assert failed is not None
    assert pool.fail_discovery_refresh(failed)
    assert pool.discovery_snapshot() == baseline
    assert {item.nf_instance_id for item in pool.records()} == {CHILD_A}

    older = pool.request_discovery_refresh(scope)
    assert older is not None
    newer_scope = discovery_scope(tracking_area=("001", "01", "000002"))
    newer = pool.request_discovery_refresh(newer_scope)
    assert newer is not None
    assert not pool.complete_discovery_refresh(
        older,
        discovery_snapshot((CHILD_B,), observed_at=current[0], scope=scope),
    )
    assert pool.discovery_snapshot() == baseline
