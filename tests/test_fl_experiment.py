import threading
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from py_mtlf.core.fl_experiment import (
    ExperimentConflictError,
    ExperimentLifecycle,
    ExperimentRole,
    ExperimentStateError,
    FLExperimentRegistry,
)


def test_client_reservations_join_only_the_same_upper_process_group():
    registry = FLExperimentRegistry()

    first = registry.reserve_client("subscription-a", "correlation-a")
    joined = registry.reserve_client("subscription-b", "correlation-a")

    assert joined.reservation_id == first.reservation_id
    assert joined.lifecycle is ExperimentLifecycle.PROVISIONAL
    assert joined.upper_client_subscription_ids == frozenset(
        {"subscription-a", "subscription-b"}
    )
    with pytest.raises(ExperimentConflictError):
        registry.reserve_client("subscription-c", "correlation-b")


def test_client_reservation_rollback_releases_only_an_empty_provisional_group():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")
    registry.reserve_client("subscription-b", "correlation-a")

    remaining = registry.rollback_client(reservation.reservation_id, "subscription-a")
    released = registry.rollback_client(reservation.reservation_id, "subscription-b")

    assert remaining is not None
    assert remaining.upper_client_subscription_ids == frozenset({"subscription-b"})
    assert released is None
    assert registry.active() is None


def test_client_group_binds_one_valid_plan_and_role():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")
    plan_id = str(uuid4())

    bound = registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.BRANCH)

    assert bound.plan_id == plan_id
    assert bound.assigned_role is ExperimentRole.BRANCH
    assert bound.lifecycle is ExperimentLifecycle.ACTIVE
    with pytest.raises(ExperimentStateError):
        registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.BRANCH)
    with pytest.raises(ExperimentStateError):
        registry.bind_plan(reservation.reservation_id, str(uuid4()), ExperimentRole.LEAF)


def test_plan_bind_rejects_untyped_role_values():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")

    with pytest.raises(ValueError, match="ExperimentRole"):
        registry.bind_plan(reservation.reservation_id, str(uuid4()), "BRANCH")


@pytest.mark.parametrize("plan_id", ["", "not-a-uuid", "11111111-1111-1111-8111-111111111111"])
def test_plan_bind_requires_uuid_v4(plan_id):
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")

    with pytest.raises(ValueError, match="UUIDv4"):
        registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.LEAF)


def test_branch_attaches_one_lower_server_process_to_the_same_plan():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")
    plan_id = str(uuid4())
    registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.BRANCH)

    attached = registry.attach_server(reservation.reservation_id, plan_id, "server-a")

    assert attached.server_process_id == "server-a"
    assert registry.for_server_process("server-a") == attached
    with pytest.raises(ExperimentStateError):
        registry.attach_server(reservation.reservation_id, plan_id, "server-b")
    with pytest.raises(ExperimentStateError):
        registry.attach_server(reservation.reservation_id, str(uuid4()), "server-b")

    detached = registry.detach_server(reservation.reservation_id, "server-a")
    assert detached.server_process_id is None
    with pytest.raises(ExperimentStateError):
        registry.detach_server(reservation.reservation_id, "server-a")


def test_leaf_rejects_server_attachment_and_root_accepts_it():
    leaf_registry = FLExperimentRegistry()
    leaf = leaf_registry.reserve_client("subscription-a", "correlation-a")
    leaf_plan = str(uuid4())
    leaf_registry.bind_plan(leaf.reservation_id, leaf_plan, ExperimentRole.LEAF)

    with pytest.raises(ExperimentStateError):
        leaf_registry.attach_server(leaf.reservation_id, leaf_plan, "server-a")

    root_registry = FLExperimentRegistry()
    root_plan = str(uuid4())
    root = root_registry.reserve_root(root_plan)
    attached = root_registry.attach_server(root.reservation_id, root_plan, "server-root")

    assert attached.assigned_role is ExperimentRole.ROOT
    assert attached.server_process_id == "server-root"


def test_nonhierarchical_server_uses_the_same_active_slot_without_a_fake_plan():
    registry = FLExperimentRegistry()

    reserved = registry.reserve_server("server-a")

    assert reserved.plan_id is None
    assert reserved.assigned_role is None
    assert reserved.server_process_id == "server-a"
    with pytest.raises(ExperimentConflictError):
        registry.reserve_server("server-b")
    with pytest.raises(ExperimentConflictError):
        registry.reserve_client("subscription-a", "correlation-a")


def test_terminal_cleanup_keeps_slot_occupied_then_retires_plan_on_release():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")
    plan_id = str(uuid4())
    registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.LEAF)

    terminal = registry.mark_terminal(reservation.reservation_id, "FAILED")
    assert terminal.lifecycle is ExperimentLifecycle.TERMINAL
    assert terminal.cleanup_pending is True
    with pytest.raises(ExperimentConflictError):
        registry.reserve_client("subscription-b", "correlation-b")

    cleaning = registry.begin_cleanup(reservation.reservation_id)
    assert cleaning.lifecycle is ExperimentLifecycle.CLEANING
    registry.release(reservation.reservation_id)

    assert registry.active() is None
    assert registry.is_retired(plan_id) is True
    with pytest.raises(ExperimentConflictError):
        registry.reserve_root(plan_id)
    assert registry.reserve_client("subscription-b", "correlation-b") is not None


def test_new_registry_does_not_restore_active_or_retired_state():
    first = FLExperimentRegistry()
    root = first.reserve_root(str(uuid4()))
    first.mark_terminal(root.reservation_id, "COMPLETE")
    first.begin_cleanup(root.reservation_id)
    first.release(root.reservation_id)

    replacement = FLExperimentRegistry()

    assert replacement.active() is None
    assert replacement.retired_plan_ids() == frozenset()


def test_lookup_miss_does_not_create_a_record_and_snapshots_are_immutable():
    registry = FLExperimentRegistry()

    assert registry.for_client_subscription("missing") is None
    assert registry.for_server_process("missing") is None
    assert registry.active() is None

    snapshot = registry.reserve_client("subscription-a", "correlation-a")
    with pytest.raises(FrozenInstanceError):
        snapshot.plan_id = str(uuid4())


def test_shutdown_blocks_new_admission_but_allows_existing_cleanup():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_server("server-a")

    registry.shutdown()

    with pytest.raises(ExperimentStateError, match="shutting down"):
        registry.reserve_client("subscription-a", "correlation-a")
    registry.mark_terminal(reservation.reservation_id, "FAILED")
    registry.begin_cleanup(reservation.reservation_id)
    registry.release(reservation.reservation_id)
    assert registry.active() is None


def test_shutdown_blocks_plan_binding_and_server_attachment():
    binding_registry = FLExperimentRegistry()
    provisional = binding_registry.reserve_client("subscription-a", "correlation-a")
    binding_registry.shutdown()

    with pytest.raises(ExperimentStateError, match="shutting down"):
        binding_registry.bind_plan(
            provisional.reservation_id,
            str(uuid4()),
            ExperimentRole.BRANCH,
        )

    attachment_registry = FLExperimentRegistry()
    plan_id = str(uuid4())
    branch = attachment_registry.reserve_client("subscription-a", "correlation-a")
    attachment_registry.bind_plan(branch.reservation_id, plan_id, ExperimentRole.BRANCH)
    attachment_registry.shutdown()

    with pytest.raises(ExperimentStateError, match="shutting down"):
        attachment_registry.attach_server(branch.reservation_id, plan_id, "server-a")


def test_client_removal_requires_provisional_rollback_or_active_cleanup():
    registry = FLExperimentRegistry()
    reservation = registry.reserve_client("subscription-a", "correlation-a")
    plan_id = str(uuid4())
    registry.bind_plan(reservation.reservation_id, plan_id, ExperimentRole.LEAF)

    with pytest.raises(ExperimentStateError, match="cleanup"):
        registry.remove_client(reservation.reservation_id, "subscription-a")

    registry.mark_terminal(reservation.reservation_id, "COMPLETE")
    registry.begin_cleanup(reservation.reservation_id)
    cleaning = registry.remove_client(reservation.reservation_id, "subscription-a")

    assert cleaning is not None
    assert cleaning.lifecycle is ExperimentLifecycle.CLEANING
    assert cleaning.upper_client_subscription_ids == frozenset()
    registry.release(reservation.reservation_id)
    assert registry.active() is None


def test_concurrent_admission_allows_only_one_top_level_experiment():
    registry = FLExperimentRegistry()
    barrier = threading.Barrier(3)
    outcomes = []
    lock = threading.Lock()

    def reserve(index):
        barrier.wait()
        try:
            registry.reserve_client(f"subscription-{index}", f"correlation-{index}")
            outcome = "reserved"
        except ExperimentConflictError:
            outcome = "conflict"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["conflict", "reserved"]
