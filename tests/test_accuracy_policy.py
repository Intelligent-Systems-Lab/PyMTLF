from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from py_mtlf.config import AccuracyPolicySettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy
from py_mtlf.wire.ml_model_monitor import (
    MLModelMonitorNotification,
    MLModelMonitorRegistration,
    MLModelMonitorSubscription,
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class CatalogStub:
    def __init__(self) -> None:
        self.family_key = "ue-communication-default"
        self.current_version = 1

    def version_key_for_id(self, model_id):
        return model_id

    def family_for_version(self, version_key):
        return self.family_key if version_key in {1, 2} else None

    def current(self, family_key):
        if family_key != self.family_key:
            return None
        return type("Current", (), {"version_key": self.current_version})()


def subscription(correlation: str, group: str, model_id: int = 1) -> MLModelMonitorSubscription:
    return MLModelMonitorSubscription(
        modelIds=[model_id],
        notificationUri="http://go.example/notify",
        notifCorrId=correlation,
        modelMetric="ACCURACY",
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def registration(group: str, model_id: int = 1) -> MLModelMonitorRegistration:
    return MLModelMonitorRegistration(
        consumerId="11111111-1111-4111-8111-111111111111",
        modelId=model_id,
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def notification(
    correlation: str,
    deviation: float | None,
    model_id: int = 1,
) -> MLModelMonitorNotification:
    info = {"modelId": model_id, "inferenceNum": 2}
    if deviation is not None:
        info["deviation"] = deviation
        info["modelMetric"] = "ACCURACY"
    return MLModelMonitorNotification(
        notifCorrId=correlation,
        modelAccuInfos=[info],
        mLEvent="UE_COMMUNICATION",
    )


def settings(**overrides) -> AccuracyPolicySettings:
    values = dict(
        reference_buffer_size=5,
        min_reference_samples=3,
        min_std=0.1,
        fixed_floor=0.05,
        z_score_threshold=1,
        decision_window_size=3,
        required_hits=2,
    )
    values.update(overrides)
    return AccuracyPolicySettings(**values)


def seed(policy, sub, reg) -> None:
    for value in (0.1, 0.1, 0.1):
        policy.observe(sub, notification(sub.notification_id, value), reg)


def test_degradation_only_policy_uses_reference_and_n_in_m_window():
    policy = AccuracyPolicy(settings(), CatalogStub())
    sub = subscription("corr-a", "group-a")
    reg = registration("group-a")
    seed(policy, sub, reg)

    first = policy.observe(sub, notification("corr-a", 0.5), reg)[0]
    miss = policy.observe(sub, notification("corr-a", 0.1), reg)[0]
    second = policy.observe(sub, notification("corr-a", 0.5), reg)[0]

    assert first.hit and not first.triggered
    assert not miss.hit
    assert second.triggered
    assert len(policy.intents()) == 1
    assert policy.intents()[0].triggering_scope_key == policy.scope_key(sub, reg)


def test_missing_deviation_only_updates_liveness():
    policy = AccuracyPolicy(settings(), CatalogStub())
    sub = subscription("corr-a", "group-a")

    decision = policy.observe(sub, notification("corr-a", None), registration("group-a"))[0]

    assert not decision.evaluated
    assert policy.snapshot()["scope_count"] == 0
    assert policy.snapshot()["insufficient_reports"] == 1


def test_any_scope_triggers_one_model_intent_and_resets_all_windows():
    policy = AccuracyPolicy(settings(required_hits=1), CatalogStub())
    sub_a = subscription("corr-a", "group-a")
    sub_b = subscription("corr-b", "group-b")
    reg_a = registration("group-a")
    reg_b = registration("group-b")
    seed(policy, sub_a, reg_a)
    seed(policy, sub_b, reg_b)

    decision = policy.observe(sub_a, notification("corr-a", 0.5), reg_a)[0]
    skipped = policy.observe(sub_b, notification("corr-b", 0.5), reg_b)[0]

    assert decision.triggered
    assert not skipped.evaluated
    intent = policy.intents()[0]
    assert set(intent.active_scope_keys) == {
        policy.scope_key(sub_a, reg_a),
        policy.scope_key(sub_b, reg_b),
    }


def test_simultaneous_same_model_reports_claim_one_in_flight_intent():
    policy = AccuracyPolicy(settings(required_hits=1), CatalogStub())
    sub = subscription("corr-a", "group-a")
    reg = registration("group-a")
    seed(policy, sub, reg)

    with ThreadPoolExecutor(max_workers=4) as executor:
        decisions = list(
            executor.map(
                lambda _index: policy.observe(
                    sub,
                    notification("corr-a", 0.5),
                    reg,
                )[0],
                range(4),
            )
        )

    assert sum(decision.triggered for decision in decisions) == 1
    assert len(policy.intents()) == 1


def test_discard_intents_atomically_releases_each_family_in_flight_marker():
    catalog = CatalogStub()
    policy = AccuracyPolicy(settings(required_hits=1), catalog)
    sub = subscription("corr-a", "group-a")
    reg = registration("group-a")
    seed(policy, sub, reg)
    policy.observe(sub, notification("corr-a", 0.5), reg)

    assert policy.snapshot()["in_flight"] == (catalog.family_key,)

    discarded = policy.discard_intents()

    assert tuple(intent.family_key for intent in discarded) == (catalog.family_key,)
    assert policy.intents() == ()
    assert policy.snapshot()["in_flight"] == ()


def test_scope_ttl_uses_injected_clock():
    clock = Clock()
    policy = AccuracyPolicy(
        settings(scope_state_ttl_seconds=10),
        CatalogStub(),
        clock=clock,
    )
    sub_a = subscription("corr-a", "group-a")
    policy.observe(sub_a, notification("corr-a", 0.1), registration("group-a"))
    clock.now += timedelta(seconds=11)
    sub_b = subscription("corr-b", "group-b")
    policy.observe(sub_b, notification("corr-b", 0.1), registration("group-b"))

    assert policy.snapshot()["scope_count"] == 1


def test_new_generation_uses_registration_as_adoption_evidence():
    catalog = CatalogStub()
    policy = AccuracyPolicy(settings(required_hits=1), catalog)
    sub = subscription("corr-a", "group-a", model_id=1)
    reg = registration("group-a", model_id=1)
    seed(policy, sub, reg)
    scope_key = policy.scope_key(sub, reg)
    family_key = catalog.family_key
    catalog.current_version = 2
    policy.begin_generation(
        family_key,
        1,
        2,
        (scope_key,),
    )
    old_report = policy.observe(
        sub,
        notification("corr-a", 0.5),
        reg,
    )[0]
    new_sub = subscription("corr-b", "group-a", model_id=2)
    new_reg = registration("group-a", model_id=2)
    policy.record_registration(new_reg)
    first_new = policy.observe(
        new_sub,
        notification("corr-b", 0.1, model_id=2),
        new_reg,
    )[0]

    assert not old_report.evaluated
    assert first_new.evaluated
    assert not first_new.baseline_ready
    assert policy.snapshot()["adoption_count"] == 1


def test_restored_generation_keeps_retrain_in_flight_until_cutover_completes():
    catalog = CatalogStub()
    policy = AccuracyPolicy(settings(), catalog)
    family_key = catalog.family_key

    policy.restore_generation(
        family_key,
        1,
        2,
        ("scope-a",),
    )

    assert family_key in policy.snapshot()["in_flight"]
    policy.complete_retrain(family_key)
    assert family_key not in policy.snapshot()["in_flight"]


def test_deleting_retired_registration_does_not_remove_adopted_scope():
    catalog = CatalogStub()
    policy = AccuracyPolicy(settings(), catalog)
    old = registration("group-a", model_id=1)
    policy.record_registration(old)
    catalog.current_version = 2
    new = registration("group-a", model_id=2)
    policy.record_registration(new)

    policy.remove_registration(old)

    assert policy.active_scope_keys(catalog.family_key) == (policy.registration_scope_key(new),)

    policy.remove_registration(new)
    assert policy.active_scope_keys(catalog.family_key) == ()
