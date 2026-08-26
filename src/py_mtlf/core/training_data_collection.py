import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from py_mtlf.config import PrivateAPITrainingDataSettings, PrivateCollectionProfileSettings
from py_mtlf.core.collection_relay import (
    AcceptedSubscription,
    CollectionRelayClient,
    CollectionRelayError,
    ServingSmfTarget,
)
from py_mtlf.core.dataset import DatasetCoordinator
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.models import TrainingDataDescriptor
from py_mtlf.wire.training_data_collection import (
    CollectionRequestState,
    DescriptorState,
    TrainingDataCollectionRequest,
    TrainingDataCollectionStatus,
)
from py_mtlf.wire.upf_event_exposure import NotificationData


class CollectionManagerError(RuntimeError):
    def __init__(self, status_code: int, cause: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.cause = cause
        self.detail = detail


@dataclass
class CollectionResource:
    key: str
    subscription: AcceptedSubscription
    references: set[str] = field(default_factory=set)
    cleanup_pending: bool = False


@dataclass
class CollectionRequestRecord:
    request: TrainingDataCollectionRequest
    profile: PrivateCollectionProfileSettings
    request_digest: str
    profile_digest: str
    state: CollectionRequestState
    created_at: datetime
    updated_at: datetime
    process_generation: str
    resources: dict[str, CollectionResource] = field(default_factory=dict)
    resolved_ue_count: int = 0
    storage_transport: str = "unavailable"
    stored_start: datetime | None = None
    stored_stop: datetime | None = None
    record_count: int = 0
    observation_count: int = 0
    adrf_instance_id: str = ""
    observed_correlations: set[str] = field(default_factory=set)
    descriptor_state: DescriptorState = DescriptorState.NONE
    failure_cause: str = ""
    failure_detail: str = ""
    retain_until: datetime | None = None
    retry_attempt: int = 0

    @property
    def collection_group_id(self) -> str:
        return f"{self.request.request_id}/{self.request.collection_profile_id}"


class TrainingDataCollectionManager:
    _LEDGER_VERSION = 2
    _ACTIVE_STATES = {
        CollectionRequestState.PENDING,
        CollectionRequestState.RESOLVING,
        CollectionRequestState.SUBSCRIBING,
        CollectionRequestState.COLLECTING,
        CollectionRequestState.STOPPING,
        CollectionRequestState.RECOVERING,
    }

    def __init__(
        self,
        settings: PrivateAPITrainingDataSettings,
        context: NwdafContextClient,
        datasets: DatasetCoordinator,
        relay: CollectionRelayClient,
        *,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._context = context
        self._datasets = datasets
        self._relay = relay
        self._clock = clock
        self._profiles = {item.profile_id: item for item in settings.collection_profiles}
        self._state_directory = settings.state_directory
        self._inbox_directory = self._state_directory / "inbox"
        self._ledger_path = self._state_directory / "ledger.json"
        self._lock = threading.RLock()
        self._peer_lock = threading.Lock()
        self._requests: dict[str, CollectionRequestRecord] = {}
        self._resources: dict[str, CollectionResource] = {}
        self._correlations: dict[str, set[str]] = {}
        self._stored_inbox_digests: set[str] = set()
        self._closing = threading.Event()
        self._admission_ready = threading.Event()
        self._work = ThreadPoolExecutor(
            max_workers=settings.worker_count,
            thread_name_prefix="training-data-collection",
        )
        self._storage = ThreadPoolExecutor(
            max_workers=settings.worker_count,
            thread_name_prefix="training-data-storage",
        )
        self._futures: set[Future] = set()
        self._storage_futures: set[Future] = set()
        self._storage_slots = threading.BoundedSemaphore(settings.queue_capacity)

    @property
    def admission_ready(self) -> bool:
        return self._admission_ready.is_set() and not self._closing.is_set()

    def open(self) -> None:
        self._prepare_directories()
        self._load_ledger()
        self._restore_retained_descriptors()
        self._recover_inbox()
        self._recover_active_requests()
        self._admission_ready.set()

    def create(self, request: TrainingDataCollectionRequest) -> TrainingDataCollectionStatus:
        if not self.admission_ready:
            raise CollectionManagerError(
                503,
                "SERVICE_UNAVAILABLE",
                "collection manager is not ready",
            )
        now = self._clock().astimezone(UTC)
        with self._lock:
            self._expire_locked(now)
            existing = self._requests.get(request.request_id)
            if existing is not None:
                if existing.request != request:
                    raise CollectionManagerError(
                        409,
                        "REQUEST_CONFLICT",
                        "requestId is already bound to a different collection request",
                    )
                return self._status(existing)
            profile = self._profiles.get(request.collection_profile_id)
            if profile is None:
                raise CollectionManagerError(
                    404,
                    "PROFILE_NOT_FOUND",
                    "collection profile was not found",
                )
            profile_digest = self._profile_digest(profile)
            if any(
                resource.cleanup_pending
                and self._collection_key(profile, resource.subscription.target)
                == resource.key
                for resource in self._resources.values()
            ):
                raise CollectionManagerError(
                    409,
                    "CLEANUP_PENDING",
                    "collection profile still owns a peer resource pending cleanup",
                )
            generation = self._generation()
            record = CollectionRequestRecord(
                request=request,
                profile=profile,
                request_digest=self._request_digest(request),
                profile_digest=profile_digest,
                state=CollectionRequestState.PENDING,
                created_at=now,
                updated_at=now,
                process_generation=generation,
            )
            self._requests[request.request_id] = record
            self._persist_locked()
            self._submit(self._provision, request.request_id)
            return self._status(record)

    def get(self, request_id: str) -> TrainingDataCollectionStatus:
        with self._lock:
            self._expire_locked(self._clock().astimezone(UTC))
            record = self._requests.get(request_id)
            if record is None:
                raise CollectionManagerError(
                    404,
                    "REQUEST_NOT_FOUND",
                    "collection request was not found",
                )
            return self._status(record)

    def delete(self, request_id: str) -> TrainingDataCollectionStatus:
        with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise CollectionManagerError(
                    404,
                    "REQUEST_NOT_FOUND",
                    "collection request was not found",
                )
            if record.state not in {
                CollectionRequestState.RETAINED,
                CollectionRequestState.TERMINATED,
                CollectionRequestState.FAILED,
            }:
                record.state = CollectionRequestState.STOPPING
                record.updated_at = self._clock().astimezone(UTC)
                self._persist_locked()
                self._submit(self._cleanup_request, request_id, False)
            return self._status(record)

    def accept_callback(self, notification: NotificationData) -> bool:
        if not self.admission_ready:
            raise CollectionManagerError(503, "SERVICE_UNAVAILABLE", "callback admission is fenced")
        raw = notification.model_dump(by_alias=True, exclude_none=True, mode="json")
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        inbox_path = self._inbox_directory / f"{digest}.json"
        with self._lock:
            request_ids = self._correlations.get(notification.correlation_id, set())
            active_request_ids = sorted(
                request_id
                for request_id in request_ids
                if self._requests.get(request_id) is not None
                and self._requests[request_id].state
                is CollectionRequestState.COLLECTING
            )
            if not active_request_ids:
                raise CollectionManagerError(
                    404,
                    "SUBSCRIPTION_NOT_FOUND",
                    "correlationId does not identify an active collection resource",
                )
            if digest in self._stored_inbox_digests or inbox_path.exists():
                return False
            if not self._storage_slots.acquire(blocking=False):
                raise CollectionManagerError(503, "QUEUE_FULL", "callback storage queue is full")
            envelope = {
                "schemaVersion": self._LEDGER_VERSION,
                "requestIds": active_request_ids,
                "correlationId": notification.correlation_id,
                "notification": raw,
            }
            try:
                self._atomic_json_write(inbox_path, envelope, mode=0o600)
            except Exception:
                self._storage_slots.release()
                raise
            future = self._storage.submit(self._store_inbox, inbox_path)
            self._track_future(
                future,
                release_storage_slot=True,
                storage_work=True,
            )
            return True

    def abort_generation(self, reason: str) -> None:
        self._admission_ready.clear()
        with self._lock:
            active = [
                item.request.request_id
                for item in self._requests.values()
                if item.state in self._ACTIVE_STATES
            ]
            for request_id in active:
                record = self._requests[request_id]
                record.state = CollectionRequestState.RECOVERING
                record.failure_cause = "CONTAINING_NWDAF_RESET"
                record.failure_detail = reason[:256]
                record.updated_at = self._clock().astimezone(UTC)
            self._persist_locked()
            storage_futures = tuple(self._storage_futures)
        for future in storage_futures:
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(
                    "durable callback drain failed during generation reset"
                ) from error
        if any(self._inbox_directory.glob("*.json")):
            raise RuntimeError(
                "durable callback drain remains pending during generation reset"
            )
        for request_id in active:
            self._cleanup_request(request_id, True)
        if not self._closing.is_set():
            self._admission_ready.set()

    def shutdown(self) -> None:
        if self._closing.is_set():
            return
        self._admission_ready.clear()
        self._closing.set()
        with self._lock:
            active = [
                item.request.request_id
                for item in self._requests.values()
                if item.state in self._ACTIVE_STATES
            ]
        self._work.shutdown(wait=True, cancel_futures=False)
        self._storage.shutdown(wait=True, cancel_futures=False)
        for request_id in active:
            self._cleanup_request(request_id, True)
        with self._lock:
            self._persist_locked()
        self._relay.close()

    def _provision(self, request_id: str) -> None:
        with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.state is not CollectionRequestState.PENDING:
                return
            self._transition_locked(record, CollectionRequestState.RESOLVING)
        try:
            targets = self._relay.resolve_profile(record.profile)
            if not targets:
                raise CollectionRelayError(
                    "TARGET_UNAVAILABLE",
                    "collection resolved no serving SMF targets",
                )
            with self._lock:
                current = self._requests.get(request_id)
                if current is None or current.state is not CollectionRequestState.RESOLVING:
                    return
                current.resolved_ue_count = len({target.supi for target in targets})
                self._transition_locked(current, CollectionRequestState.SUBSCRIBING)
            for target in targets:
                self._acquire_resource(request_id, record, target)
            with self._lock:
                current = self._requests.get(request_id)
                if current is not None and current.state is CollectionRequestState.SUBSCRIBING:
                    self._transition_locked(current, CollectionRequestState.COLLECTING)
        except CollectionRelayError as error:
            self._fail_and_cleanup(request_id, error.cause, error.detail)
        except Exception:
            self._fail_and_cleanup(
                request_id,
                "COLLECTION_FAILED",
                "collection provisioning failed",
            )

    def _store_inbox(self, inbox_path: Path, attempt: int = 0) -> None:
        try:
            with self._lock:
                already_stored = inbox_path.stem in self._stored_inbox_digests
            if already_stored:
                inbox_path.unlink(missing_ok=True)
                return
            envelope = self._read_json(inbox_path)
            if envelope.get("schemaVersion") != self._LEDGER_VERSION:
                raise RuntimeError("unsupported durable inbox version")
            request_ids = tuple(str(item) for item in envelope["requestIds"])
            correlation_id = str(envelope["correlationId"])
            notification = NotificationData.model_validate(envelope["notification"])
            with self._lock:
                resource = next(
                    (
                        item
                        for item in self._resources.values()
                        if item.subscription.correlation_id == correlation_id
                    ),
                    None,
                )
                if resource is None:
                    raise RuntimeError("durable inbox owner is unavailable")
                subscription = resource.subscription
            raw_notification = notification.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            )
            record_body = {
                "dataSub": [{"smfDataSub": subscription.representation}],
                "dataNotif": {"upfEventNotifs": [raw_notification]},
            }
            measurement_start = min(
                item.start_time or item.time_stamp for item in notification.notification_items
            ).astimezone(UTC)
            measurement_stop = max(
                item.time_stamp for item in notification.notification_items
            ).astimezone(UTC)
            receipt = self._relay.store_record(
                record_body,
                supi=subscription.target.supi,
                measurement_time=measurement_start,
            )
            with self._lock:
                for request_id in request_ids:
                    current = self._requests.get(request_id)
                    if current is None or request_id not in resource.references:
                        continue
                    current.storage_transport = receipt.transport
                    current.adrf_instance_id = receipt.adrf_instance_id
                    current.stored_start = (
                        measurement_start
                        if current.stored_start is None
                        else min(current.stored_start, measurement_start)
                    )
                    current.stored_stop = (
                        measurement_stop
                        if current.stored_stop is None
                        else max(current.stored_stop, measurement_stop)
                    )
                    current.record_count += 1
                    current.observation_count += len(notification.notification_items)
                    current.observed_correlations.add(correlation_id)
                    current.retry_attempt = 0
                    current.updated_at = self._clock().astimezone(UTC)
                    self._publish_descriptor_locked(
                        current,
                        subscription,
                        receipt.adrf_instance_id,
                    )
                self._stored_inbox_digests.add(inbox_path.stem)
                self._persist_locked()
            inbox_path.unlink(missing_ok=True)
        except CollectionRelayError as error:
            with self._lock:
                request_ids = self._read_json(inbox_path).get("requestIds", [])
                for request_id in request_ids:
                    record = self._requests.get(str(request_id))
                    if record is not None:
                        record.retry_attempt = min(attempt + 1, 3)
                self._persist_locked()
            if error.retryable and attempt < 2 and not self._closing.is_set():
                delay = min(
                    self._settings.retry_initial_backoff_seconds * (2**attempt),
                    self._settings.retry_max_backoff_seconds,
                )
                if not self._closing.wait(delay):
                    self._store_inbox(inbox_path, attempt + 1)
                return
            if not error.retryable:
                with self._lock:
                    request_ids = self._read_json(inbox_path).get("requestIds", [])
                    for request_id in request_ids:
                        record = self._requests.get(str(request_id))
                        if record is not None:
                            record.failure_cause = error.cause
                            record.failure_detail = error.detail[:256]
                            record.updated_at = self._clock().astimezone(UTC)
                    self._persist_locked()
                inbox_path.unlink(missing_ok=True)
            else:
                with self._lock:
                    request_ids = self._read_json(inbox_path).get("requestIds", [])
                    for request_id in request_ids:
                        record = self._requests.get(str(request_id))
                        if record is not None:
                            record.failure_cause = "STORAGE_RETRY_PENDING"
                            record.failure_detail = (
                                "durable inbox storage retry remains pending"
                            )
                            record.updated_at = self._clock().astimezone(UTC)
                    self._persist_locked()

    def _publish_descriptor_locked(
        self,
        record: CollectionRequestRecord,
        subscription: AcceptedSubscription,
        adrf_instance_id: str,
        *,
        retain_until: datetime | None = None,
    ) -> None:
        if retain_until is None:
            retain_until = self._clock().astimezone(UTC) + timedelta(
                seconds=self._settings.descriptor_retention_seconds
            )
        event_filter = deepcopy(record.profile.ml_event_filter)
        event_filter["networkArea"] = record.profile.network_area.wire_value()
        descriptor = TrainingDataDescriptor.model_validate(
            {
                "correlationId": subscription.correlation_id,
                "state": (
                    "RETAINED"
                    if record.state
                    in {
                        CollectionRequestState.STOPPING,
                        CollectionRequestState.RETAINED,
                    }
                    else "ACTIVE"
                ),
                "storedDataSpec": {
                    "dataSpec": {"smfDataSub": subscription.representation},
                    "timePeriod": {
                        "startTime": record.stored_start,
                        "stopTime": record.stored_stop,
                    },
                },
                "mlEventSubscription": {
                    "mLEvent": record.profile.ml_event,
                    "mLEventFilter": event_filter,
                    "tgtUe": {
                        "intGroupIds": list(record.profile.target_ue.int_group_ids)
                    },
                },
                "sourceNfInstanceId": subscription.target.smf_nf_instance_id,
                "adrfInstanceId": adrf_instance_id or None,
                "retainUntil": retain_until,
            }
        )
        self._datasets.put_private_training_data_descriptor(
            descriptor.correlation_id,
            record.collection_group_id,
            descriptor,
        )
        record.descriptor_state = DescriptorState.ACTIVE
        record.retain_until = retain_until

    def _restore_retained_descriptors(self) -> None:
        with self._lock:
            self._expire_locked(self._clock().astimezone(UTC))
            for record in self._requests.values():
                if record.state is not CollectionRequestState.RETAINED:
                    continue
                if (
                    record.stored_start is None
                    or record.stored_stop is None
                    or record.retain_until is None
                ):
                    raise RuntimeError(
                        "retained collection ledger is missing descriptor state"
                    )
                for correlation_id, resource in record.resources.items():
                    if correlation_id not in record.observed_correlations:
                        continue
                    self._publish_descriptor_locked(
                        record,
                        resource.subscription,
                        record.adrf_instance_id,
                        retain_until=record.retain_until,
                    )
                record.descriptor_state = DescriptorState.RETAINED

    def _cleanup_request(self, request_id: str, interrupted: bool) -> None:
        with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                return
            resources = tuple(record.resources.values())
            if record.state not in {
                CollectionRequestState.STOPPING,
                CollectionRequestState.RECOVERING,
            }:
                record.state = (
                    CollectionRequestState.RECOVERING
                    if interrupted
                    else CollectionRequestState.STOPPING
                )
        pending = False
        for resource in resources:
            with self._peer_lock:
                with self._lock:
                    resource.references.discard(request_id)
                    correlation_refs = self._correlations.get(
                        resource.subscription.correlation_id,
                        set(),
                    )
                    correlation_refs.discard(request_id)
                    if correlation_refs:
                        self._correlations[resource.subscription.correlation_id] = (
                            correlation_refs
                        )
                    else:
                        self._correlations.pop(resource.subscription.correlation_id, None)
                    last_reference = not resource.references
                if not last_reference:
                    continue
                cleaned = False
                for attempt in range(3):
                    try:
                        cleaned = self._relay.delete_subscription(resource.subscription)
                    except CollectionRelayError as error:
                        with self._lock:
                            record.failure_cause = error.cause
                            record.failure_detail = error.detail[:256]
                        if not error.retryable:
                            break
                    if cleaned:
                        with self._lock:
                            record.retry_attempt = 0
                        break
                    with self._lock:
                        record.retry_attempt = attempt + 1
                        self._persist_locked()
                    if attempt < 2:
                        delay = min(
                            self._settings.retry_initial_backoff_seconds * (2**attempt),
                            self._settings.retry_max_backoff_seconds,
                        )
                        self._closing.wait(delay)
                resource.cleanup_pending = not cleaned
                pending = pending or not cleaned
                if cleaned:
                    with self._lock:
                        self._resources.pop(resource.key, None)
        with self._lock:
            record.updated_at = self._clock().astimezone(UTC)
            if pending:
                record.failure_cause = record.failure_cause or "CLEANUP_PENDING"
                record.failure_detail = record.failure_detail or "peer cleanup has not converged"
            elif record.record_count:
                record.state = CollectionRequestState.RETAINED
                record.descriptor_state = DescriptorState.RETAINED
                for resource in resources:
                    if (
                        resource.subscription.correlation_id
                        not in record.observed_correlations
                    ):
                        continue
                    if record.stored_start is not None and record.stored_stop is not None:
                        self._publish_descriptor_locked(
                            record,
                            resource.subscription,
                            record.adrf_instance_id,
                        )
                record.descriptor_state = DescriptorState.RETAINED
            else:
                record.state = (
                    CollectionRequestState.FAILED
                    if interrupted or record.failure_cause
                    else CollectionRequestState.TERMINATED
                )
                if interrupted and not record.failure_cause:
                    record.failure_cause = "INTERRUPTED"
                    record.failure_detail = (
                        "collection was stopped during cleanup-only restart recovery"
                    )
            self._persist_locked()

    def _fail_and_cleanup(self, request_id: str, cause: str, detail: str) -> None:
        with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                return
            record.state = CollectionRequestState.STOPPING
            record.failure_cause = cause
            record.failure_detail = detail[:256]
            self._persist_locked()
        self._cleanup_request(request_id, False)

    def _recover_active_requests(self) -> None:
        with self._lock:
            active = [
                record.request.request_id
                for record in self._requests.values()
                if record.state in self._ACTIVE_STATES
            ]
            for request_id in active:
                self._requests[request_id].state = CollectionRequestState.RECOVERING
            if active:
                self._persist_locked()
        for request_id in active:
            self._cleanup_request(request_id, True)

    def _recover_inbox(self) -> None:
        for path in sorted(self._inbox_directory.glob("*.json")):
            envelope = self._read_json(path)
            with self._lock:
                attempt = max(
                    (
                        self._requests[str(request_id)].retry_attempt
                        for request_id in envelope.get("requestIds", [])
                        if str(request_id) in self._requests
                    ),
                    default=0,
                )
            self._store_inbox(path, attempt)
        if any(self._inbox_directory.glob("*.json")):
            raise RuntimeError("durable inbox recovery remains pending")

    def _acquire_resource(
        self,
        request_id: str,
        record: CollectionRequestRecord,
        target: ServingSmfTarget,
    ) -> None:
        key = self._collection_key(record.profile, target)
        with self._peer_lock:
            with self._lock:
                existing = self._resources.get(key)
                if existing is not None and not existing.cleanup_pending:
                    existing.references.add(request_id)
                    record.resources[existing.subscription.correlation_id] = existing
                    self._correlations.setdefault(
                        existing.subscription.correlation_id,
                        set(),
                    ).add(request_id)
                    record.updated_at = self._clock().astimezone(UTC)
                    self._persist_locked()
                    return
            correlation_id = str(uuid4())
            payload = self._subscription_payload(record, target, correlation_id)
            try:
                accepted = self._relay.create_subscription(target, correlation_id, payload)
            except CollectionRelayError as error:
                provisional = error.provisional_subscription
                if provisional is not None:
                    resource = CollectionResource(
                        key,
                        provisional,
                        {request_id},
                        cleanup_pending=True,
                    )
                    with self._lock:
                        current = self._requests.get(request_id)
                        if current is not None:
                            self._resources[key] = resource
                            current.resources[correlation_id] = resource
                            self._correlations[correlation_id] = {request_id}
                            current.updated_at = self._clock().astimezone(UTC)
                            self._persist_locked()
                raise
            resource = CollectionResource(key, accepted, {request_id})
            with self._lock:
                current = self._requests.get(request_id)
                if current is None or current.state is not CollectionRequestState.SUBSCRIBING:
                    self._relay.delete_subscription(accepted)
                    return
                self._resources[key] = resource
                current.resources[correlation_id] = resource
                self._correlations[correlation_id] = {request_id}
                current.updated_at = self._clock().astimezone(UTC)
                self._persist_locked()

    @staticmethod
    def _collection_key(
        profile: PrivateCollectionProfileSettings,
        target: ServingSmfTarget,
    ) -> str:
        value = {
            "supi": target.supi,
            "smfNfInstanceId": target.smf_nf_instance_id,
            "apiRoot": target.api_root,
            "pduSessionId": target.pdu_session_id,
            "dnn": target.dnn,
            "snssai": target.snssai,
            "samplingIntervalSeconds": profile.sampling_interval_seconds,
            "requiredMeasurements": [
                "VOLUME_MEASUREMENT",
                "THROUGHPUT_MEASUREMENT",
            ],
            "mlEventFilter": profile.ml_event_filter,
            "networkArea": profile.network_area.wire_value(),
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _transition_locked(
        self,
        record: CollectionRequestRecord,
        state: CollectionRequestState,
    ) -> None:
        record.state = state
        record.updated_at = self._clock().astimezone(UTC)
        self._persist_locked()

    def _status(self, record: CollectionRequestRecord) -> TrainingDataCollectionStatus:
        pending = sum(item.cleanup_pending for item in record.resources.values())
        return TrainingDataCollectionStatus(
            requestId=record.request.request_id,
            collectionProfileId=record.request.collection_profile_id,
            state=record.state,
            createdAt=record.created_at,
            updatedAt=record.updated_at,
            processGeneration=record.process_generation,
            intendedGroupCount=len(record.profile.target_ue.int_group_ids),
            resolvedUeCount=record.resolved_ue_count,
            resolvedSmfTargetCount=len(record.resources),
            activePeerResourceCount=sum(
                record.request.request_id in item.references and not item.cleanup_pending
                for item in record.resources.values()
            ),
            pendingCleanupPeerResourceCount=pending,
            storageTransport=record.storage_transport,
            startTime=record.stored_start,
            stopTime=record.stored_stop,
            recordCount=record.record_count,
            observationCount=record.observation_count,
            descriptorState=record.descriptor_state,
            failureCause=record.failure_cause,
            failureDetail=record.failure_detail,
            cleanupPending=bool(pending),
        )

    def _subscription_payload(
        self,
        record: CollectionRequestRecord,
        target: ServingSmfTarget,
        correlation_id: str,
    ) -> dict:
        context = self._context.get()
        return {
            "supi": target.supi,
            "pduSeId": target.pdu_session_id,
            "dnn": target.dnn,
            "snssai": target.snssai,
            "nfId": context.nf_instance_id,
            "notifId": correlation_id,
            "notifUri": self._settings.callback_base_uri + "/callbacks/upf-event-exposure",
            "eventSubs": [
                {
                    "event": "UPF_EVENT",
                    "networkArea": record.profile.network_area.wire_value(),
                    "upfEvents": [
                        {
                            "type": "USER_DATA_USAGE_MEASURES",
                            "measurementTypes": [
                                "VOLUME_MEASUREMENT",
                                "THROUGHPUT_MEASUREMENT",
                            ],
                            "granularityOfMeasurement": "PER_SESSION",
                        }
                    ],
                }
            ],
            "notifMethod": "PERIODIC",
            "repPeriod": record.profile.sampling_interval_seconds,
        }

    def _generation(self) -> str:
        try:
            return self._context.get().containing_nwdaf_process_instance_id
        except RuntimeError as error:
            raise CollectionManagerError(
                503,
                "CONTAINING_NWDAF_UNAVAILABLE",
                "containing NWDAF generation is unavailable",
            ) from error

    def _profile_digest(self, profile: PrivateCollectionProfileSettings) -> str:
        payload = profile.model_dump(by_alias=True, exclude_none=True, mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _request_digest(request: TrainingDataCollectionRequest) -> str:
        payload = request.model_dump(by_alias=True, exclude_none=True, mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _submit(self, function, *args) -> None:
        try:
            future = self._work.submit(function, *args)
        except RuntimeError as error:
            raise CollectionManagerError(
                503,
                "SERVICE_UNAVAILABLE",
                "collection manager is closing",
            ) from error
        self._track_future(future)

    def _track_future(
        self,
        future: Future,
        *,
        release_storage_slot: bool = False,
        storage_work: bool = False,
    ) -> None:
        with self._lock:
            self._futures.add(future)
            if storage_work:
                self._storage_futures.add(future)

        def complete(done: Future) -> None:
            if release_storage_slot:
                self._storage_slots.release()
            with self._lock:
                self._futures.discard(done)
                self._storage_futures.discard(done)

        future.add_done_callback(complete)

    def _expire_locked(self, now: datetime) -> None:
        expired = [
            request_id
            for request_id, record in self._requests.items()
            if record.state is CollectionRequestState.RETAINED
            and record.retain_until is not None
            and record.retain_until <= now
        ]
        for request_id in expired:
            record = self._requests.pop(request_id)
            for resource in record.resources.values():
                self._datasets.delete_private_training_data_descriptor(
                    resource.subscription.correlation_id,
                    record.collection_group_id,
                )
        if expired:
            self._persist_locked()

    def _prepare_directories(self) -> None:
        self._state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._inbox_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._state_directory, 0o700)
        os.chmod(self._inbox_directory, 0o700)

    def _persist_locked(self) -> None:
        payload = {
            "schemaVersion": self._LEDGER_VERSION,
            "storedInboxDigests": sorted(self._stored_inbox_digests),
            "requests": [self._serialize(record) for record in self._requests.values()],
        }
        self._atomic_json_write(self._ledger_path, payload, mode=0o600)

    def _load_ledger(self) -> None:
        if not self._ledger_path.exists():
            return
        payload = self._read_json(self._ledger_path)
        if payload.get("schemaVersion") != self._LEDGER_VERSION or not isinstance(
            payload.get("requests"), list
        ):
            raise RuntimeError("unsupported or corrupt training-data collection ledger")
        try:
            stored_digests = payload.get("storedInboxDigests", [])
            if not isinstance(stored_digests, list) or any(
                not isinstance(item, str) or len(item) != 64 for item in stored_digests
            ):
                raise ValueError("stored inbox digests are malformed")
            self._stored_inbox_digests = set(stored_digests)
            for value in payload["requests"]:
                record = self._deserialize(value)
                self._requests[record.request.request_id] = record
                for correlation_id, resource in record.resources.items():
                    if record.request.request_id in resource.references:
                        self._correlations.setdefault(correlation_id, set()).add(
                            record.request.request_id
                        )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("unsupported or corrupt training-data collection ledger") from error

    def _serialize(self, record: CollectionRequestRecord) -> dict:
        return {
            "request": record.request.model_dump(by_alias=True, mode="json"),
            "profile": record.profile.model_dump(by_alias=True, mode="json"),
            "requestDigest": record.request_digest,
            "profileDigest": record.profile_digest,
            "state": record.state,
            "createdAt": record.created_at.isoformat(),
            "updatedAt": record.updated_at.isoformat(),
            "processGeneration": record.process_generation,
            "resolvedUeCount": record.resolved_ue_count,
            "storageTransport": record.storage_transport,
            "storedStart": record.stored_start.isoformat() if record.stored_start else None,
            "storedStop": record.stored_stop.isoformat() if record.stored_stop else None,
            "recordCount": record.record_count,
            "observationCount": record.observation_count,
            "adrfInstanceId": record.adrf_instance_id,
            "observedCorrelations": sorted(record.observed_correlations),
            "descriptorState": record.descriptor_state,
            "failureCause": record.failure_cause,
            "failureDetail": record.failure_detail,
            "retainUntil": record.retain_until.isoformat() if record.retain_until else None,
            "retryAttempt": record.retry_attempt,
            "resources": [
                {
                    "resourceKey": item.key,
                    "correlationId": item.subscription.correlation_id,
                    "subscriptionId": item.subscription.subscription_id,
                    "location": item.subscription.location,
                    "target": item.subscription.target.__dict__,
                    "representation": item.subscription.representation,
                    "cleanupPending": item.cleanup_pending,
                    "referenced": record.request.request_id in item.references,
                }
                for item in record.resources.values()
            ],
        }

    def _deserialize(self, value: dict) -> CollectionRequestRecord:
        request = TrainingDataCollectionRequest.model_validate(value["request"])
        profile = PrivateCollectionProfileSettings.model_validate(value["profile"])
        request_digest = str(value["requestDigest"])
        profile_digest = str(value["profileDigest"])
        request_mismatch = request_digest != self._request_digest(request)
        profile_mismatch = profile_digest != self._profile_digest(profile)
        if request_mismatch or profile_mismatch:
            raise ValueError("collection ledger digest mismatch")
        retry_attempt = int(value.get("retryAttempt", 0))
        if retry_attempt < 0 or retry_attempt > 3:
            raise ValueError("collection ledger retry attempt is invalid")
        state = CollectionRequestState(value["state"])
        observed_correlations = value.get("observedCorrelations", [])
        if not isinstance(observed_correlations, list) or any(
            not isinstance(item, str) or not item for item in observed_correlations
        ):
            raise ValueError("collection ledger observed correlations are malformed")
        resources = {}
        for item in value.get("resources", []):
            target = ServingSmfTarget(**item["target"])
            subscription = AcceptedSubscription(
                item["subscriptionId"],
                item["location"],
                target,
                item["correlationId"],
                item["representation"],
            )
            key = str(item["resourceKey"])
            referenced = item["referenced"]
            if not isinstance(referenced, bool):
                raise ValueError("collection ledger resource reference is invalid")
            resource = self._resources.get(key)
            if resource is None:
                resource = CollectionResource(
                    key,
                    subscription,
                    set(),
                    bool(item.get("cleanupPending")),
                )
                self._resources[key] = resource
            if referenced:
                resource.references.add(request.request_id)
            resources[subscription.correlation_id] = resource
        if not set(observed_correlations).issubset(resources):
            raise ValueError(
                "collection ledger observed correlations do not identify resources"
            )
        if int(value.get("recordCount", 0)) and not observed_correlations:
            raise ValueError("collection ledger stored records have no observed resource")
        return CollectionRequestRecord(
            request=request,
            profile=profile,
            request_digest=request_digest,
            profile_digest=profile_digest,
            state=state,
            created_at=datetime.fromisoformat(value["createdAt"]),
            updated_at=datetime.fromisoformat(value["updatedAt"]),
            process_generation=str(value["processGeneration"]),
            resources=resources,
            resolved_ue_count=int(value.get("resolvedUeCount", 0)),
            storage_transport=str(value.get("storageTransport", "unavailable")),
            stored_start=(
                datetime.fromisoformat(value["storedStart"]) if value.get("storedStart") else None
            ),
            stored_stop=(
                datetime.fromisoformat(value["storedStop"]) if value.get("storedStop") else None
            ),
            record_count=int(value.get("recordCount", 0)),
            observation_count=int(value.get("observationCount", 0)),
            adrf_instance_id=str(value.get("adrfInstanceId", "")),
            observed_correlations=set(observed_correlations),
            descriptor_state=DescriptorState(value.get("descriptorState", "NONE")),
            failure_cause=str(value.get("failureCause", "")),
            failure_detail=str(value.get("failureDetail", "")),
            retain_until=(
                datetime.fromisoformat(value["retainUntil"]) if value.get("retainUntil") else None
            ),
            retry_attempt=retry_attempt,
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("durable collection state must be a JSON object")
        return value

    @staticmethod
    def _atomic_json_write(path: Path, value: dict, *, mode: int) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
