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


def subscription(correlation: str, group: str) -> MLModelMonitorSubscription:
    return MLModelMonitorSubscription(
        modelIds=[1],
        notificationUri="http://go.example/notify",
        notifCorrId=correlation,
        modelMetric="ACCURACY",
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def registration(group: str) -> MLModelMonitorRegistration:
    return MLModelMonitorRegistration(
        consumerId="11111111-1111-4111-8111-111111111111",
        modelId=1,
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        tgtUe={"intGroupIds": [group]},
    )


def notification(correlation: str, deviation: float | None) -> MLModelMonitorNotification:
    info = {"modelId": 1, "inferenceNum": 2}
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
    policy = AccuracyPolicy(settings(), "local-mtlf")
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
    policy = AccuracyPolicy(settings(), "local-mtlf")
    sub = subscription("corr-a", "group-a")

    decision = policy.observe(sub, notification("corr-a", None), registration("group-a"))[0]

    assert not decision.evaluated
    assert policy.snapshot()["scope_count"] == 0
    assert policy.snapshot()["liveness_reports"] == 1


def test_any_scope_triggers_one_model_intent_and_resets_all_windows():
    policy = AccuracyPolicy(settings(required_hits=1), "local-mtlf")
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
    policy = AccuracyPolicy(settings(required_hits=1), "local-mtlf")
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


def test_scope_ttl_uses_injected_clock():
    clock = Clock()
    policy = AccuracyPolicy(
        settings(scope_state_ttl_seconds=10),
        "local-mtlf",
        clock=clock,
    )
    sub_a = subscription("corr-a", "group-a")
    policy.observe(sub_a, notification("corr-a", 0.1), registration("group-a"))
    clock.now += timedelta(seconds=11)
    sub_b = subscription("corr-b", "group-b")
    policy.observe(sub_b, notification("corr-b", 0.1), registration("group-b"))

    assert policy.snapshot()["scope_count"] == 1
