import json
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from nwdaf_context import context_client, verified_capability_checker

from py_mtlf.app import create_app
from py_mtlf.config import PrivateAPITrainingDataSettings
from py_mtlf.core.collection_relay import (
    AcceptedSubscription,
    CollectionRelayError,
    ServingSmfTarget,
    StorageReceipt,
)
from py_mtlf.core.training_data_collection import (
    CollectionManagerError,
    TrainingDataCollectionManager,
)
from py_mtlf.wire.training_data_collection import (
    CollectionRequestState,
    TrainingDataCollectionRequest,
)
from py_mtlf.wire.upf_event_exposure import NotificationData


class RelayStub:
    def __init__(self) -> None:
        self.created = threading.Event()
        self.store_entered = threading.Event()
        self.allow_store = threading.Event()
        self.deleted: list[str] = []
        self.create_calls = 0
        self.created_payloads: list[dict] = []
        self.store_calls = 0
        self.store_failures = 0
        self.delete_failures = 0
        self.provisional_create_failure = False
        self.closed = False

    def resolve_profile(self, profile):
        del profile
        return (
            ServingSmfTarget(
                supi="imsi-466920000000001",
                smf_nf_instance_id="11111111-1111-4111-8111-111111111111",
                api_root="http://smf.example",
                pdu_session_id=10,
                dnn="internet",
                snssai={"sst": 1, "sd": "010203"},
            ),
        )

    def create_subscription(self, target, correlation_id, payload):
        self.create_calls += 1
        self.created_payloads.append(payload)
        self.created.set()
        accepted = AcceptedSubscription(
            subscription_id=f"subscription-{self.create_calls}",
            location=(
                "http://smf.example/nsmf-event-exposure/v1/subscriptions/"
                f"subscription-{self.create_calls}"
            ),
            target=target,
            correlation_id=correlation_id,
            representation=payload,
        )
        if self.provisional_create_failure:
            raise CollectionRelayError(
                "UNSUPPORTED_FINITE_LEASE",
                "finite SMF subscription expiry is not supported",
                retryable=False,
                provisional_subscription=accepted,
            )
        return accepted

    def delete_subscription(self, subscription):
        self.deleted.append(subscription.subscription_id)
        if self.delete_failures:
            self.delete_failures -= 1
            return False
        return True

    def store_record(self, record, *, supi, measurement_time):
        del record, supi, measurement_time
        self.store_calls += 1
        self.store_entered.set()
        assert self.allow_store.wait(timeout=2)
        if self.store_failures:
            self.store_failures -= 1
            raise CollectionRelayError(
                "STORAGE_UNAVAILABLE",
                "temporary storage failure",
            )
        return StorageReceipt(
            "adrf",
            "22222222-2222-4222-8222-222222222222",
        )

    def close(self):
        self.closed = True


class BlockingResolutionRelay(RelayStub):
    def __init__(self, expected_resolutions: int) -> None:
        super().__init__()
        self.expected_resolutions = expected_resolutions
        self.resolve_count = 0
        self.all_resolving = threading.Event()
        self.allow_resolution = threading.Event()
        self._resolve_lock = threading.Lock()

    def resolve_profile(self, profile):
        with self._resolve_lock:
            self.resolve_count += 1
            if self.resolve_count == self.expected_resolutions:
                self.all_resolving.set()
        assert self.allow_resolution.wait(timeout=2)
        return super().resolve_profile(profile)


def private_settings(
    path: Path,
    *,
    minimum: int = 1,
    worker_count: int = 1,
) -> PrivateAPITrainingDataSettings:
    return PrivateAPITrainingDataSettings.model_validate(
        {
            "collection_trigger": "private_api",
            "callback_base_uri": "http://py-mtlf.example",
            "state_directory": path,
            "request_timeout_seconds": 1,
            "retry_initial_backoff_seconds": 0,
            "retry_max_backoff_seconds": 1,
            "worker_count": worker_count,
            "queue_capacity": 1,
            "descriptor_retention_seconds": 60,
            "consent": {
                "purpose": "model_training",
                "policy": "not_required_by_local_policy",
            },
            "collection_profiles": [
                {
                    "profile_id": "profile-a",
                    "ml_event": "UE_COMMUNICATION",
                    "ml_event_filter": {},
                    "target_ue": {"intGroupIds": ["group-a.example"]},
                    "dnns": ["internet"],
                    "snssais": [{"sst": 1, "sd": "010203"}],
                    "sampling_interval_seconds": 2,
                    "minimum_observation_count": minimum,
                }
            ],
        }
    )


def request() -> TrainingDataCollectionRequest:
    return TrainingDataCollectionRequest(
        requestId=str(uuid4()),
        collectionProfileId="profile-a",
    )


def notification(
    correlation_id: str,
    *,
    timestamp: str = "2026-08-26T10:00:01Z",
) -> NotificationData:
    return NotificationData.model_validate(
        {
            "correlationId": correlation_id,
            "notificationItems": [
                {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "ueIpv4Addr": "10.0.0.1",
                    "timeStamp": timestamp,
                    "startTime": "2026-08-26T10:00:00Z",
                    "userDataUsageMeasurements": [
                        {
                            "volumeMeasurement": {
                                "totalVolume": 10,
                                "ulVolume": 4,
                                "dlVolume": 6,
                                "totalNbOfPackets": 3,
                                "ulNbOfPackets": 1,
                                "dlNbOfPackets": 2,
                            },
                            "throughputMeasurement": {
                                "ulThroughput": "1 Mbps",
                                "dlThroughput": "2 Mbps",
                                "ulPacketThroughput": "3 kpps",
                                "dlPacketThroughput": "4 kpps",
                            },
                        }
                    ],
                }
            ],
        }
    )


def manager(tmp_path, context_client):
    relay = RelayStub()
    datasets = Mock()
    owner = TrainingDataCollectionManager(
        private_settings(tmp_path / "collection-state"),
        context_client,
        datasets,
        relay,
    )
    owner.open()
    return owner, relay, datasets


def private_app_settings(settings, tmp_path):
    payload = settings.model_dump(mode="python")
    payload["runtime"] = {"mode": "federated"}
    payload["local_training"] = None
    payload["federated_learning"] = {
        "workspace_root": tmp_path / "fl-workspaces",
        "public_base_url": settings.artifact.public_base_url,
        "client": {
            "training_data": private_settings(tmp_path / "collection-state").model_dump(
                by_alias=True,
                mode="python",
            ),
            "model_interoperability_ids": ["001122"],
        },
    }
    return settings.__class__.model_validate(payload)


def wait_for_state(owner, request_id, expected):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = owner.get(request_id)
        if snapshot.state is expected:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"collection request did not reach {expected}")


def test_create_is_idempotent_and_request_id_conflict_is_rejected(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    value = request()

    first = owner.create(value)
    replay = owner.create(value)
    assert first.request_id == replay.request_id
    assert relay.created.wait(timeout=2)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)

    with pytest.raises(CollectionManagerError) as conflict:
        owner.create(
            TrainingDataCollectionRequest(
                requestId=value.request_id,
                collectionProfileId="different-profile",
            )
        )
    assert conflict.value.status_code == 409
    owner.shutdown()


def test_subscription_payload_uses_exact_profile_and_fixed_measurements(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)

    payload = relay.created_payloads[0]
    assert payload["supi"] == "imsi-466920000000001"
    assert payload["pduSeId"] == 10
    assert payload["nfId"] == "11111111-1111-4111-8111-111111111111"
    assert payload["notifUri"] == (
        "http://py-mtlf.example/callbacks/upf-event-exposure"
    )
    event = payload["eventSubs"][0]
    assert event["event"] == "UPF_EVENT"
    assert event["upfEvents"][0]["measurementTypes"] == [
        "VOLUME_MEASUREMENT",
        "THROUGHPUT_MEASUREMENT",
    ]
    assert "tai" not in str(payload).lower()
    owner.shutdown()


def test_callback_is_durable_before_ack_and_storage_publishes_private_descriptor(
    tmp_path,
):
    owner, relay, datasets = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    assert relay.created.wait(timeout=2)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))

    assert owner.accept_callback(notification(correlation_id)) is True
    assert relay.store_entered.wait(timeout=2)
    assert list((tmp_path / "collection-state" / "inbox").glob("*.json"))
    relay.allow_store.set()
    ready = wait_for_state(owner, value.request_id, CollectionRequestState.READY)

    assert ready.storage_transport == "adrf"
    assert ready.record_count == 1
    assert ready.observation_count == 1
    datasets.put_private_training_data_descriptor.assert_called()
    descriptor = datasets.put_private_training_data_descriptor.call_args.args[2]
    assert descriptor.state == "ACTIVE"
    assert descriptor.source_nf_instance_id == "11111111-1111-4111-8111-111111111111"
    assert descriptor.ml_event_subscription.target_ue == {
        "intGroupIds": ["group-a.example"]
    }
    assert descriptor.stored_data_spec.time_period.start_time.isoformat() == (
        "2026-08-26T10:00:00+00:00"
    )
    assert not list((tmp_path / "collection-state" / "inbox").glob("*.json"))
    assert owner.accept_callback(notification(correlation_id)) is False
    assert relay.store_calls == 1
    owner.shutdown()


def test_delete_retires_correlation_and_retains_stored_descriptor(
    tmp_path,
):
    owner, relay, _ = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    assert relay.created.wait(timeout=2)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))
    owner.accept_callback(notification(correlation_id))
    assert relay.store_entered.wait(timeout=2)
    relay.allow_store.set()
    wait_for_state(owner, value.request_id, CollectionRequestState.READY)

    owner.delete(value.request_id)
    retained = wait_for_state(owner, value.request_id, CollectionRequestState.RETAINED)

    assert retained.descriptor_state == "RETAINED"
    assert retained.active_peer_resource_count == 0
    assert retained.pending_cleanup_peer_resource_count == 0
    assert relay.deleted == ["subscription-1"]
    with pytest.raises(CollectionManagerError) as late:
        owner.accept_callback(notification(correlation_id))
    assert late.value.status_code == 404
    owner.shutdown()


def test_transient_peer_delete_retries_with_bounded_backoff(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    relay.delete_failures = 2
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)

    owner.delete(value.request_id)
    terminal = wait_for_state(
        owner,
        value.request_id,
        CollectionRequestState.TERMINATED,
    )

    assert relay.deleted == ["subscription-1"] * 3
    assert terminal.active_peer_resource_count == 0
    assert terminal.cleanup_pending is False
    owner.shutdown()


def test_pending_cleanup_blocks_duplicate_profile_request(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    relay.delete_failures = 3
    first = request()
    owner.create(first)
    wait_for_state(owner, first.request_id, CollectionRequestState.COLLECTING)
    owner.delete(first.request_id)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        pending = owner.get(first.request_id)
        if pending.cleanup_pending:
            break
        time.sleep(0.005)
    else:
        raise AssertionError("collection cleanup did not become pending")

    with pytest.raises(CollectionManagerError) as conflict:
        owner.create(request())

    assert conflict.value.status_code == 409
    assert relay.create_calls == 1
    owner.shutdown()


def test_pending_cleanup_blocks_different_profile_with_same_collection_key(tmp_path):
    settings = private_settings(tmp_path / "collection-state")
    second_profile = settings.collection_profiles[0].model_copy(
        update={
            "profile_id": "profile-b",
            "minimum_observation_count": 2,
        }
    )
    settings = settings.model_copy(
        update={"collection_profiles": (*settings.collection_profiles, second_profile)}
    )
    relay = RelayStub()
    relay.delete_failures = 3
    owner = TrainingDataCollectionManager(
        settings,
        context_client(),
        Mock(),
        relay,
    )
    owner.open()
    first = request()
    owner.create(first)
    wait_for_state(owner, first.request_id, CollectionRequestState.COLLECTING)
    owner.delete(first.request_id)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not owner.get(first.request_id).cleanup_pending:
        time.sleep(0.005)

    with pytest.raises(CollectionManagerError) as conflict:
        owner.create(
            TrainingDataCollectionRequest(
                requestId=str(uuid4()),
                collectionProfileId="profile-b",
            )
        )

    assert conflict.value.status_code == 409
    assert relay.create_calls == 1
    owner.shutdown()


def test_provisional_subscription_is_owned_until_cleanup(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    relay.provisional_create_failure = True
    value = request()

    owner.create(value)
    failed = wait_for_state(owner, value.request_id, CollectionRequestState.FAILED)

    assert failed.failure_cause == "UNSUPPORTED_FINITE_LEASE"
    assert failed.pending_cleanup_peer_resource_count == 0
    assert relay.deleted == ["subscription-1"]
    owner.shutdown()


def test_transient_storage_failure_retries_from_durable_inbox(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    relay.store_failures = 1
    relay.allow_store.set()
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))

    owner.accept_callback(notification(correlation_id))
    ready = wait_for_state(owner, value.request_id, CollectionRequestState.READY)

    assert ready.record_count == 1
    assert relay.store_calls == 2
    assert not list((tmp_path / "collection-state" / "inbox").glob("*.json"))
    owner.shutdown()


def test_recovery_does_not_restore_inbox_already_committed_in_ledger(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    relay.allow_store.set()
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))
    accepted = notification(correlation_id)
    owner.accept_callback(accepted)
    wait_for_state(owner, value.request_id, CollectionRequestState.READY)
    digest = next(iter(owner._stored_inbox_digests))
    inbox = tmp_path / "collection-state" / "inbox" / f"{digest}.json"
    owner._atomic_json_write(
        inbox,
        {
            "schemaVersion": owner._LEDGER_VERSION,
            "requestIds": [value.request_id],
            "correlationId": correlation_id,
            "notification": accepted.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            ),
        },
        mode=0o600,
    )

    owner._store_inbox(inbox)

    assert relay.store_calls == 1
    assert not inbox.exists()
    owner.shutdown()


def test_callback_queue_full_is_rejected_without_durable_acceptance(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))

    assert owner.accept_callback(notification(correlation_id)) is True
    assert relay.store_entered.wait(timeout=2)
    with pytest.raises(CollectionManagerError) as rejected:
        owner.accept_callback(
            notification(correlation_id, timestamp="2026-08-26T10:00:02Z")
        )

    assert rejected.value.status_code == 503
    assert rejected.value.cause == "QUEUE_FULL"
    assert len(list((tmp_path / "collection-state" / "inbox").glob("*.json"))) == 1
    relay.allow_store.set()
    owner.shutdown()


def test_generation_reset_drains_admitted_callback_before_peer_cleanup(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(owner._requests[value.request_id].resources))
    owner.accept_callback(notification(correlation_id))
    assert relay.store_entered.wait(timeout=2)

    reset = threading.Thread(target=owner.abort_generation, args=("generation changed",))
    reset.start()
    time.sleep(0.02)

    assert reset.is_alive()
    assert relay.deleted == []
    assert owner.admission_ready is False

    relay.allow_store.set()
    reset.join(timeout=2)
    assert not reset.is_alive()
    assert relay.deleted == ["subscription-1"]
    assert owner.get(value.request_id).state is CollectionRequestState.RETAINED
    assert owner.admission_ready is True
    owner.shutdown()


def test_matching_requests_share_peer_resource_until_last_reference(tmp_path):
    owner, relay, _ = manager(tmp_path, context_client())
    first = request()
    second = request()
    owner.create(first)
    wait_for_state(owner, first.request_id, CollectionRequestState.COLLECTING)
    owner.create(second)
    wait_for_state(owner, second.request_id, CollectionRequestState.COLLECTING)

    assert relay.create_calls == 1
    owner.delete(first.request_id)
    wait_for_state(owner, first.request_id, CollectionRequestState.TERMINATED)
    assert relay.deleted == []

    owner.delete(second.request_id)
    wait_for_state(owner, second.request_id, CollectionRequestState.TERMINATED)
    assert relay.deleted == ["subscription-1"]
    owner.shutdown()


def test_concurrent_matching_requests_create_one_peer_resource(tmp_path):
    relay = BlockingResolutionRelay(expected_resolutions=2)
    owner = TrainingDataCollectionManager(
        private_settings(
            tmp_path / "collection-state",
            worker_count=2,
        ),
        context_client(),
        Mock(),
        relay,
    )
    owner.open()
    first = request()
    second = request()

    owner.create(first)
    owner.create(second)
    assert relay.all_resolving.wait(timeout=2)
    relay.allow_resolution.set()
    wait_for_state(owner, first.request_id, CollectionRequestState.COLLECTING)
    wait_for_state(owner, second.request_id, CollectionRequestState.COLLECTING)

    assert relay.create_calls == 1
    first_resource = next(iter(owner._requests[first.request_id].resources.values()))
    second_resource = next(iter(owner._requests[second.request_id].resources.values()))
    assert first_resource is second_resource
    assert first_resource.references == {first.request_id, second.request_id}
    owner.shutdown()


def test_delete_supersedes_resolution_before_peer_create(tmp_path):
    relay = BlockingResolutionRelay(expected_resolutions=1)
    owner = TrainingDataCollectionManager(
        private_settings(tmp_path / "collection-state"),
        context_client(),
        Mock(),
        relay,
    )
    owner.open()
    value = request()

    owner.create(value)
    assert relay.all_resolving.wait(timeout=2)
    owner.delete(value.request_id)
    relay.allow_resolution.set()
    terminal = wait_for_state(
        owner,
        value.request_id,
        CollectionRequestState.TERMINATED,
    )

    assert terminal.active_peer_resource_count == 0
    assert relay.create_calls == 0
    assert relay.deleted == []
    owner.shutdown()


def test_restart_performs_cleanup_only_recovery_without_auto_resume(tmp_path):
    original, original_relay, _ = manager(tmp_path / "original", context_client())
    value = request()
    original.create(value)
    wait_for_state(original, value.request_id, CollectionRequestState.COLLECTING)
    recovery_state = tmp_path / "recovery" / "collection-state"
    recovery_state.parent.mkdir()
    shutil.copytree(tmp_path / "original" / "collection-state", recovery_state)
    original.shutdown()
    assert original_relay.create_calls == 1

    recovery_relay = RelayStub()
    recovered = TrainingDataCollectionManager(
        private_settings(recovery_state),
        context_client(),
        Mock(),
        recovery_relay,
    )
    recovered.open()

    snapshot = recovered.get(value.request_id)
    assert snapshot.state is CollectionRequestState.FAILED
    assert snapshot.failure_cause == "INTERRUPTED"
    assert recovery_relay.create_calls == 0
    assert recovery_relay.deleted == ["subscription-1"]

    next_request = request()
    recovered.create(next_request)
    wait_for_state(recovered, next_request.request_id, CollectionRequestState.COLLECTING)
    assert recovery_relay.create_calls == 1
    recovered.shutdown()


def test_restart_does_not_restore_released_shared_resource_reference(tmp_path):
    original, original_relay, _ = manager(tmp_path / "original", context_client())
    released = request()
    active = request()
    original.create(released)
    wait_for_state(original, released.request_id, CollectionRequestState.COLLECTING)
    original.create(active)
    wait_for_state(original, active.request_id, CollectionRequestState.COLLECTING)
    original.delete(released.request_id)
    wait_for_state(original, released.request_id, CollectionRequestState.TERMINATED)
    assert original_relay.deleted == []

    recovery_state = tmp_path / "recovery" / "collection-state"
    recovery_state.parent.mkdir()
    shutil.copytree(tmp_path / "original" / "collection-state", recovery_state)
    original.shutdown()

    recovery_relay = RelayStub()
    recovered = TrainingDataCollectionManager(
        private_settings(recovery_state),
        context_client(),
        Mock(),
        recovery_relay,
    )
    recovered.open()

    assert recovery_relay.deleted == ["subscription-1"]
    assert recovered.get(active.request_id).state is CollectionRequestState.FAILED
    assert recovered.get(released.request_id).state is CollectionRequestState.TERMINATED
    recovered.shutdown()


def test_restart_restores_retained_descriptor_without_extending_ttl(tmp_path):
    original, original_relay, _ = manager(tmp_path / "original", context_client())
    value = request()
    original.create(value)
    wait_for_state(original, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(original._requests[value.request_id].resources))
    original.accept_callback(notification(correlation_id))
    assert original_relay.store_entered.wait(timeout=2)
    original_relay.allow_store.set()
    wait_for_state(original, value.request_id, CollectionRequestState.READY)
    original.delete(value.request_id)
    retained = wait_for_state(
        original,
        value.request_id,
        CollectionRequestState.RETAINED,
    )
    expected_expiry = original._requests[value.request_id].retain_until
    assert expected_expiry is not None

    recovery_state = tmp_path / "recovery" / "collection-state"
    recovery_state.parent.mkdir()
    shutil.copytree(tmp_path / "original" / "collection-state", recovery_state)
    original.shutdown()

    datasets = Mock()
    recovered = TrainingDataCollectionManager(
        private_settings(recovery_state),
        context_client(),
        datasets,
        RelayStub(),
    )
    recovered.open()

    datasets.put_private_training_data_descriptor.assert_called_once()
    descriptor = datasets.put_private_training_data_descriptor.call_args.args[2]
    assert descriptor.state == "RETAINED"
    assert retained.state is CollectionRequestState.RETAINED
    assert descriptor.retain_until == expected_expiry
    recovered.shutdown()


def test_restart_drains_durable_inbox_before_peer_cleanup(tmp_path):
    original, original_relay, _ = manager(tmp_path / "original", context_client())
    value = request()
    original.create(value)
    wait_for_state(original, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(original._requests[value.request_id].resources))
    original.accept_callback(notification(correlation_id))
    assert original_relay.store_entered.wait(timeout=2)
    recovery_state = tmp_path / "recovery" / "collection-state"
    recovery_state.parent.mkdir()
    shutil.copytree(tmp_path / "original" / "collection-state", recovery_state)
    original_relay.allow_store.set()
    original.shutdown()

    recovery_relay = RelayStub()
    recovery_relay.allow_store.set()
    datasets = Mock()
    recovered = TrainingDataCollectionManager(
        private_settings(recovery_state),
        context_client(),
        datasets,
        recovery_relay,
    )
    recovered.open()

    snapshot = recovered.get(value.request_id)
    assert snapshot.state is CollectionRequestState.RETAINED
    assert snapshot.record_count == 1
    assert recovery_relay.store_calls == 1
    assert recovery_relay.deleted == ["subscription-1"]
    datasets.put_private_training_data_descriptor.assert_called()
    retained_descriptor = datasets.put_private_training_data_descriptor.call_args.args[2]
    assert retained_descriptor.state == "RETAINED"
    assert retained_descriptor.adrf_instance_id == "22222222-2222-4222-8222-222222222222"
    assert not list((recovery_state / "inbox").glob("*.json"))
    recovered.shutdown()


def test_restart_keeps_admission_closed_when_durable_inbox_cannot_drain(tmp_path):
    original, original_relay, _ = manager(tmp_path / "original", context_client())
    value = request()
    original.create(value)
    wait_for_state(original, value.request_id, CollectionRequestState.COLLECTING)
    correlation_id = next(iter(original._requests[value.request_id].resources))
    original.accept_callback(notification(correlation_id))
    assert original_relay.store_entered.wait(timeout=2)
    recovery_state = tmp_path / "recovery" / "collection-state"
    recovery_state.parent.mkdir()
    shutil.copytree(tmp_path / "original" / "collection-state", recovery_state)
    original_relay.allow_store.set()
    original.shutdown()

    recovery_relay = RelayStub()
    recovery_relay.allow_store.set()
    recovery_relay.store_failures = 3
    recovered = TrainingDataCollectionManager(
        private_settings(recovery_state),
        context_client(),
        Mock(),
        recovery_relay,
    )

    with pytest.raises(RuntimeError, match="inbox recovery remains pending"):
        recovered.open()

    assert recovered.admission_ready is False
    assert recovery_relay.deleted == []
    assert list((recovery_state / "inbox").glob("*.json"))
    recovered.shutdown()


def test_corrupt_ledger_fails_closed_without_discarding_file(tmp_path):
    state = tmp_path / "collection-state"
    state.mkdir()
    ledger = state / "ledger.json"
    ledger.write_text('{"schemaVersion":999,"requests":[]}', encoding="utf-8")
    owner = TrainingDataCollectionManager(
        private_settings(state),
        context_client(),
        Mock(),
        RelayStub(),
    )

    with pytest.raises(RuntimeError, match="corrupt"):
        owner.open()

    assert ledger.exists()
    owner.shutdown()


def test_ledger_persists_and_verifies_request_profile_digests_and_retry_attempt(
    tmp_path,
):
    owner, _, _ = manager(tmp_path, context_client())
    value = request()
    owner.create(value)
    wait_for_state(owner, value.request_id, CollectionRequestState.COLLECTING)
    ledger = tmp_path / "collection-state" / "ledger.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    stored = next(
        item for item in payload["requests"] if item["request"]["requestId"] == value.request_id
    )

    assert len(stored["requestDigest"]) == 64
    assert len(stored["profileDigest"]) == 64
    assert stored["retryAttempt"] == 0
    assert stored["resources"][0]["referenced"] is True
    owner.shutdown()

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["requests"][0]["profile"]["minimum_observation_count"] = 999
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    recovered = TrainingDataCollectionManager(
        private_settings(tmp_path / "collection-state"),
        context_client(),
        Mock(),
        RelayStub(),
    )

    with pytest.raises(RuntimeError, match="corrupt"):
        recovered.open()

    recovered.shutdown()


def test_private_collection_routes_expose_bounded_status_and_problem_details(
    settings,
    tmp_path,
):
    relay = RelayStub()
    app = create_app(
        private_app_settings(settings, tmp_path),
        capability_checker=verified_capability_checker(client=True),
        nwdaf_context_client=context_client(),
        collection_relay_client=relay,
    )
    value = request()

    with TestClient(app) as client:
        created = client.post(
            "/internal/v1/training-data-collections",
            json=value.model_dump(by_alias=True, mode="json"),
        )
        replay = client.post(
            "/internal/v1/training-data-collections",
            json=value.model_dump(by_alias=True, mode="json"),
        )
        unknown = client.get(
            "/internal/v1/training-data-collections/11111111-1111-4111-8111-111111111111"
        )
        invalid = client.post(
            "/internal/v1/training-data-collections",
            json={
                **value.model_dump(by_alias=True, mode="json"),
                "targetUe": {"intGroupIds": ["not-allowed"]},
            },
        )
        oversized = client.post(
            "/internal/v1/training-data-collections",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(65 * 1024),
            },
        )
        unsupported_media = client.post(
            "/internal/v1/training-data-collections",
            content=b"{}",
            headers={"Content-Type": "text/plain"},
        )
        oversized_callback = client.post(
            "/callbacks/upf-event-exposure",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(4 * 1024 * 1024 + 1),
            },
        )
        unknown_profile = client.post(
            "/internal/v1/training-data-collections",
            json={
                "requestId": "55555555-5555-4555-8555-555555555555",
                "collectionProfileId": "unknown-profile",
            },
        )
        deleted = client.delete(
            f"/internal/v1/training-data-collections/{value.request_id}"
        )
        deleted_replay = client.delete(
            f"/internal/v1/training-data-collections/{value.request_id}"
        )
        unknown_delete = client.delete(
            "/internal/v1/training-data-collections/"
            "66666666-6666-4666-8666-666666666666"
        )

    assert created.status_code == 202
    assert created.headers["Location"].endswith(value.request_id)
    assert replay.status_code == 202
    status_body = created.json()
    assert status_body["collectionTrigger"] == "private_api"
    assert "resolvedSmfTargetCount" in status_body
    assert not {
        "supis",
        "smfApiRoot",
        "subscriptionId",
        "correlationId",
        "stateDirectory",
    }.intersection(status_body)
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/problem+json")
    assert invalid.status_code == 400
    assert invalid.json()["cause"] == "INVALID_MSG_FORMAT"
    assert oversized.status_code == 413
    assert oversized.json()["cause"] == "REQUEST_TOO_LARGE"
    assert unsupported_media.status_code == 415
    assert unsupported_media.json()["cause"] == "UNSUPPORTED_MEDIA_TYPE"
    assert oversized_callback.status_code == 413
    assert oversized_callback.json()["cause"] == "REQUEST_TOO_LARGE"
    assert unknown_profile.status_code == 404
    assert unknown_profile.json()["cause"] == "PROFILE_NOT_FOUND"
    assert deleted.status_code == 202
    assert deleted_replay.status_code == 202
    assert unknown_delete.status_code == 404


def test_private_collection_routes_are_disabled_without_private_client(settings):
    app = create_app(settings)

    with TestClient(app) as client:
        management = client.post(
            "/internal/v1/training-data-collections",
            json={
                "requestId": "55555555-5555-4555-8555-555555555555",
                "collectionProfileId": "profile-a",
            },
        )
        callback = client.post(
            "/callbacks/upf-event-exposure",
            json={"correlationId": "unknown", "notificationItems": []},
        )

    assert management.status_code == 404
    assert callback.status_code == 404


def test_collection_manager_lifecycle_wraps_dataset_and_containing_context(
    settings,
    tmp_path,
):
    relay = RelayStub()
    containing = context_client()
    app = create_app(
        private_app_settings(settings, tmp_path),
        capability_checker=verified_capability_checker(client=True),
        nwdaf_context_client=containing,
        collection_relay_client=relay,
    )
    order = []
    context_open = containing.open
    context_close = containing.close
    collection_open = app.state.training_data_collection_manager.open
    collection_shutdown = app.state.training_data_collection_manager.shutdown
    dataset_shutdown = app.state.dataset_coordinator.shutdown

    def record_context_open():
        order.append("context-open")
        context_open()

    def record_collection_open():
        order.append("collection-open")
        collection_open()

    def record_collection_shutdown():
        order.append("collection-shutdown")
        collection_shutdown()

    def record_dataset_shutdown():
        order.append("dataset-shutdown")
        dataset_shutdown()

    def record_context_close():
        order.append("context-close")
        context_close()

    containing.open = record_context_open
    containing.close = record_context_close
    app.state.training_data_collection_manager.open = record_collection_open
    app.state.training_data_collection_manager.shutdown = record_collection_shutdown
    app.state.dataset_coordinator.shutdown = record_dataset_shutdown

    with TestClient(app):
        pass

    assert order.index("context-open") < order.index("collection-open")
    assert order.index("collection-shutdown") < order.index("dataset-shutdown")
    assert order.index("dataset-shutdown") < order.index("context-close")


def test_app_generation_reset_fences_collection_before_other_fl_owners(
    settings,
    tmp_path,
):
    app = create_app(
        private_app_settings(settings, tmp_path),
        capability_checker=verified_capability_checker(client=True),
        nwdaf_context_client=context_client(),
        collection_relay_client=RelayStub(),
    )
    order = []
    app.state.training_data_collection_manager.abort_generation = (
        lambda _reason: order.append("collection")
    )
    app.state.publication.abort_generation = lambda: order.append("publication")
    app.state.fl_client.abort_generation = lambda _reason: order.append("client")
    app.state.accuracy_policy.abort_generation = lambda: order.append("accuracy")
    app.state.fl_experiments.reset_generation = lambda: order.append("experiments")
    app.state.fl_workspace.reset_generation = lambda: order.append("workspace")

    app.state.generation_monitor._reset_callback("generation changed")

    assert order == [
        "collection",
        "publication",
        "client",
        "accuracy",
        "experiments",
        "workspace",
    ]
