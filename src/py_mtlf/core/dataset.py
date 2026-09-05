import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from urllib.parse import urlencode, urljoin, urlsplit
from uuid import uuid4

import httpx

from py_mtlf.config import DatasetSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent, ScopeReference
from py_mtlf.core.adrf_discovery import AdrfResolver, normalize_api_root
from py_mtlf.core.nwdaf_context import NwdafContext, NwdafContextClient
from py_mtlf.core.seed_catalog import FamilyKey
from py_mtlf.models import TrainingDataDescriptor
from py_mtlf.wire.adrf import (
    DataSubscription,
    NadrfDataRetrievalNotification,
    NadrfDataRetrievalSubscription,
    NadrfDataStoreRecord,
    TimeWindow,
)

logger = logging.getLogger(__name__)


class DatasetJobState(StrEnum):
    PENDING = "PENDING"
    RESOLVING = "RESOLVING"
    RETRIEVING = "RETRIEVING"
    READY = "READY"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class DescriptorOrigin(StrEnum):
    CONSUMER_SUBSCRIPTION = "consumer_subscription"
    PRIVATE_API = "private_api"


@dataclass(frozen=True, order=True)
class DescriptorKey:
    origin: DescriptorOrigin
    correlation_id: str
    collection_group_id: str = ""


@dataclass(frozen=True)
class DescriptorEntry:
    key: DescriptorKey
    descriptor: TrainingDataDescriptor


@dataclass(frozen=True)
class ResolvedResource:
    identity: str
    scope_keys: tuple[str, ...]
    smf_data_sub: dict
    supi: str
    adrf_instance_id: str = ""
    available_start: datetime | None = None
    available_stop: datetime | None = None
    descriptor_origin: DescriptorOrigin = DescriptorOrigin.CONSUMER_SUBSCRIPTION
    collection_group_id: str = ""


@dataclass(frozen=True)
class DatasetRecord:
    identity: str
    scope_keys: tuple[str, ...]
    source: str
    measurement_time: datetime | None
    record: dict


@dataclass(frozen=True)
class DatasetSnapshot:
    job_id: str
    family_key: FamilyKey
    triggering_scope_key: str
    required_scope_keys: tuple[str, ...]
    time_window: TimeWindow
    source: str
    records: tuple[DatasetRecord, ...]
    scope_record_counts: dict[str, int]


@dataclass
class AdrfRoute:
    correlation_id: str
    resource: ResolvedResource
    peer_subscription_id: str = ""
    terminated: bool = False
    fetch_uri: str = ""
    pending_fetch_ids: list[str] = field(default_factory=list)
    seen_fetch_ids: set[str] = field(default_factory=set)
    callback_count: int = 0
    cleanup_complete: bool = False


@dataclass
class DatasetJob:
    job_id: str
    intent: RetrainIntent
    time_window: TimeWindow
    source: str
    state: DatasetJobState = DatasetJobState.PENDING
    resources: tuple[ResolvedResource, ...] = ()
    routes: dict[str, AdrfRoute] = field(default_factory=dict)
    records: list[DatasetRecord] = field(default_factory=list)
    failure: str = ""
    cleanup_failure: str = ""
    snapshot: DatasetSnapshot | None = None
    malformed_records: int = 0
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    last_progress: float = field(default_factory=time.monotonic)
    policy_owned: bool = True
    completion_handler: Callable[["DatasetJob"], None] | None = None
    descriptor_origin: DescriptorOrigin | None = None


class DatasetCoordinator:
    def __init__(
        self,
        settings: DatasetSettings,
        nwdaf_context: NwdafContextClient,
        policy: AccuracyPolicy,
        resolver: AdrfResolver,
        client: httpx.Client | None = None,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._nwdaf_context = nwdaf_context
        self._policy = policy
        self._resolver = resolver
        self._client = client or httpx.Client(timeout=settings.fetch_timeout_seconds)
        self._owns_client = client is None
        self._clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, DatasetJob] = {}
        self._routes: dict[str, str] = {}
        self._descriptors: dict[DescriptorKey, DescriptorEntry] = {}
        self._closing = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs,
            thread_name_prefix="dataset",
        )
        self._futures: set[Future] = set()
        self._ready_handler: Callable[[str], None] | None = None

    def set_ready_handler(self, handler: Callable[[str], None] | None) -> None:
        with self._lock:
            self._ready_handler = handler

    def accept_policy_intents(self) -> None:
        if self._closing.is_set():
            return
        for intent in self._policy.take_intents():
            stop = self._clock().astimezone(UTC)
            window = TimeWindow(
                startTime=stop - timedelta(seconds=self._settings.retrieval_window_seconds),
                stopTime=stop,
            )
            job = DatasetJob(str(uuid4()), intent, window, "unavailable")
            with self._lock:
                self._jobs[job.job_id] = job
            future = self._executor.submit(self._run_job, job)
            with self._lock:
                self._futures.add(future)
            future.add_done_callback(self._discard_future)

    def submit_external(
        self,
        intent: RetrainIntent,
        time_window: TimeWindow,
        completion_handler: Callable[[DatasetJob], None],
        collection_trigger: str = DescriptorOrigin.CONSUMER_SUBSCRIPTION,
    ) -> str:
        if self._closing.is_set():
            raise RuntimeError("dataset coordinator is shutting down")
        job = DatasetJob(
            str(uuid4()),
            intent,
            time_window,
            "unavailable",
            policy_owned=False,
            completion_handler=completion_handler,
            descriptor_origin=DescriptorOrigin(collection_trigger),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        future = self._executor.submit(self._run_job, job)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard_future)
        return job.job_id

    def validate_external_scope(
        self,
        intent: RetrainIntent,
        collection_trigger: str = DescriptorOrigin.CONSUMER_SUBSCRIPTION,
    ) -> None:
        origin = DescriptorOrigin(collection_trigger)
        self._resolve_entries(intent, self._descriptor_entry_snapshot(origin), origin)

    def put_training_data_descriptor(
        self,
        descriptor_id: str,
        descriptor: TrainingDataDescriptor,
    ) -> None:
        self.put_consumer_training_data_descriptor(descriptor_id, descriptor)

    def put_consumer_training_data_descriptor(
        self,
        descriptor_id: str,
        descriptor: TrainingDataDescriptor,
    ) -> None:
        self._put_descriptor(
            DescriptorKey(DescriptorOrigin.CONSUMER_SUBSCRIPTION, descriptor_id),
            descriptor,
        )

    def put_private_training_data_descriptor(
        self,
        descriptor_id: str,
        collection_group_id: str,
        descriptor: TrainingDataDescriptor,
    ) -> None:
        group_id = collection_group_id.strip()
        if not group_id:
            raise ValueError("private descriptor collection group must not be blank")
        self._put_descriptor(
            DescriptorKey(DescriptorOrigin.PRIVATE_API, descriptor_id, group_id),
            descriptor,
        )

    def _put_descriptor(
        self,
        key: DescriptorKey,
        descriptor: TrainingDataDescriptor,
    ) -> None:
        if key.correlation_id != descriptor.correlation_id:
            raise ValueError("descriptorId must match correlationId")
        copied = descriptor.model_copy(deep=True)
        with self._lock:
            self._descriptors[key] = DescriptorEntry(key, copied)

    def delete_training_data_descriptor(self, descriptor_id: str) -> bool:
        return self.delete_consumer_training_data_descriptor(descriptor_id)

    def delete_consumer_training_data_descriptor(self, descriptor_id: str) -> bool:
        with self._lock:
            key = DescriptorKey(DescriptorOrigin.CONSUMER_SUBSCRIPTION, descriptor_id)
            return self._descriptors.pop(key, None) is not None

    def delete_private_training_data_descriptor(
        self,
        descriptor_id: str,
        collection_group_id: str,
    ) -> bool:
        with self._lock:
            key = DescriptorKey(
                DescriptorOrigin.PRIVATE_API,
                descriptor_id,
                collection_group_id,
            )
            return self._descriptors.pop(key, None) is not None

    def _descriptor_entry_snapshot(
        self,
        origin: DescriptorOrigin | None = None,
    ) -> tuple[DescriptorEntry, ...]:
        now = self._clock().astimezone(UTC)
        with self._lock:
            expired = [
                key
                for key, entry in self._descriptors.items()
                if entry.descriptor.retain_until <= now
            ]
            for key in expired:
                self._descriptors.pop(key, None)
            return tuple(
                DescriptorEntry(key, self._descriptors[key].descriptor.model_copy(deep=True))
                for key in sorted(self._descriptors)
                if origin is None or key.origin is origin
            )

    def _descriptor_snapshot(
        self,
        origin: DescriptorOrigin | None = None,
    ) -> tuple[TrainingDataDescriptor, ...]:
        return tuple(entry.descriptor for entry in self._descriptor_entry_snapshot(origin))

    def shutdown(self) -> None:
        self._closing.set()
        with self._lock:
            jobs = tuple(self._jobs.values())
        for job in jobs:
            with job.condition:
                job.condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=False)
        for job in jobs:
            self._cleanup(job)
        self._resolver.close()
        if self._owns_client:
            self._client.close()

    def jobs(self) -> tuple[DatasetJob, ...]:
        with self._lock:
            return tuple(self._jobs.values())

    def ready_snapshots(self) -> tuple[DatasetSnapshot, ...]:
        return tuple(
            job.snapshot
            for job in self.jobs()
            if job.state == DatasetJobState.READY and job.snapshot is not None
        )

    def claim_ready(self, job_id: str) -> DatasetSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != DatasetJobState.READY or job.snapshot is None:
                return None
            job.state = DatasetJobState.CLAIMED
            return job.snapshot

    def finish_claim(
        self,
        job_id: str,
        *,
        success: bool,
        failure: str = "",
        cancelled: bool = False,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != DatasetJobState.CLAIMED:
                raise RuntimeError("dataset snapshot is not claimed")
            job.failure = failure
            job.state = (
                DatasetJobState.COMPLETED
                if success
                else (DatasetJobState.CANCELLED if cancelled else DatasetJobState.FAILED)
            )
            family_key = job.intent.family_key
        self._policy.complete_retrain(family_key)

    def receive_notification(self, notification: NadrfDataRetrievalNotification) -> None:
        with self._lock:
            job_id = self._routes.get(notification.notif_corr_id)
            job = self._jobs.get(job_id) if job_id else None
        if job is None:
            raise KeyError(notification.notif_corr_id)
        with job.condition:
            route = job.routes.get(notification.notif_corr_id)
            if route is None:
                raise KeyError(notification.notif_corr_id)
            route.callback_count += 1
            if notification.fetch_instruct is None:
                job.failure = "ADRF returned an unexpected retrieval notification profile"
            else:
                instruction = notification.fetch_instruct
                if instruction.expiry and instruction.expiry <= self._clock():
                    job.failure = "ADRF fetch instruction expired before retrieval"
                if route.fetch_uri and route.fetch_uri != instruction.fetch_uri:
                    job.failure = "ADRF changed fetchUri within one retrieval route"
                route.fetch_uri = instruction.fetch_uri
                for fetch_id in instruction.fetch_corr_ids:
                    if fetch_id not in route.seen_fetch_ids:
                        accepted_fetch_ids = sum(
                            len(item.seen_fetch_ids) for item in job.routes.values()
                        )
                        if accepted_fetch_ids >= self._settings.max_records_per_job:
                            job.failure = "ADRF fetch IDs exceed max_records_per_job"
                            break
                        route.seen_fetch_ids.add(fetch_id)
                        route.pending_fetch_ids.append(fetch_id)
            route.terminated = route.terminated or notification.termination_req
            job.last_progress = time.monotonic()
            job.condition.notify_all()
            if job.failure:
                raise RuntimeError(job.failure)

    def _run_job(self, job: DatasetJob) -> None:
        try:
            if self._closing.is_set():
                raise RuntimeError("dataset coordinator is shutting down")
            job.state = DatasetJobState.RESOLVING
            entries = self._descriptor_entry_snapshot(job.descriptor_origin)
            job.resources = self._resolve_entries(job.intent, entries, job.descriptor_origin)
            if not job.resources:
                raise RuntimeError("no accepted SMF collection resource matches the retrain scopes")
            self._ensure_window_coverage(job)
            job.source = self._select_source(job.resources)
            job.state = DatasetJobState.RETRIEVING
            if job.source == "adrf":
                self._retrieve_adrf(job, self._nwdaf_context.get())
            elif job.source == "mongodb":
                self._retrieve_mongo(job)
            else:
                raise RuntimeError("training data source is unavailable")
            self._complete(job)
        except Exception as error:
            job.failure = str(error)
            job.state = DatasetJobState.FAILED
            logger.warning("Dataset retrieval failed job_id=%s error=%s", job.job_id, error)
        finally:
            self._cleanup(job)
            if job.policy_owned and job.state == DatasetJobState.FAILED:
                self._policy.complete_retrain(job.intent.family_key)
            if job.completion_handler is not None:
                try:
                    job.completion_handler(job)
                except Exception:
                    logger.exception(
                        "External dataset completion handler failed job_id=%s",
                        job.job_id,
                    )

    def _discard_future(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _resolve(
        self,
        intent: RetrainIntent,
        descriptors: tuple[TrainingDataDescriptor, ...],
    ) -> tuple[ResolvedResource, ...]:
        entries = tuple(
            DescriptorEntry(
                DescriptorKey(
                    DescriptorOrigin.CONSUMER_SUBSCRIPTION,
                    descriptor.correlation_id,
                ),
                descriptor,
            )
            for descriptor in descriptors
        )
        return self._resolve_entries(intent, entries, None)

    def _resolve_entries(
        self,
        intent: RetrainIntent,
        entries: tuple[DescriptorEntry, ...],
        required_origin: DescriptorOrigin | None,
    ) -> tuple[ResolvedResource, ...]:
        now = self._clock().astimezone(UTC)
        resolved: list[ResolvedResource] = []
        for entry in entries:
            if required_origin is not None and entry.key.origin is not required_origin:
                continue
            descriptor = entry.descriptor
            if descriptor.retain_until <= now:
                continue
            event = descriptor.ml_event_subscription
            scopes = tuple(
                sorted(
                    scope.scope_key
                    for scope in intent.active_scopes
                    if scope.ml_event == event.ml_event
                    and (
                        scope.target_ue is None
                        or self._targets_match(scope.target_ue, event.target_ue)
                    )
                    and (
                        self._event_filters_match(
                            scope.ml_event_filter,
                            event.ml_event_filter or {},
                            entry.key.origin,
                        )
                    )
                )
            )
            if not scopes:
                continue
            smf_data_sub = descriptor.stored_data_spec.data_spec.smf_data_sub.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            )
            resolved.append(
                ResolvedResource(
                    identity=descriptor.correlation_id,
                    scope_keys=scopes,
                    smf_data_sub=smf_data_sub,
                    supi=descriptor.stored_data_spec.data_spec.smf_data_sub.supi,
                    adrf_instance_id=descriptor.adrf_instance_id or "",
                    available_start=descriptor.stored_data_spec.time_period.start_time,
                    available_stop=descriptor.stored_data_spec.time_period.stop_time,
                    descriptor_origin=entry.key.origin,
                    collection_group_id=entry.key.collection_group_id,
                )
            )
        private_groups = {
            resource.collection_group_id
            for resource in resolved
            if resource.descriptor_origin is DescriptorOrigin.PRIVATE_API
        }
        if len(private_groups) > 1:
            raise RuntimeError("multiple private collection groups match the requested dataset")
        for scope in intent.active_scope_keys:
            if not any(scope in resource.scope_keys for resource in resolved):
                reference = next(
                    item for item in intent.active_scopes if item.scope_key == scope
                )
                logger.warning(
                    "No training-data descriptor matches scope=%s event=%s filter=%s "
                    "target=%s candidates=%s",
                    scope,
                    reference.ml_event,
                    reference.ml_event_filter,
                    reference.target_ue,
                    [
                        {
                            "correlationId": item.correlation_id,
                            "state": item.state,
                            "mLEvent": item.ml_event_subscription.ml_event,
                            "mLEventFilter": item.ml_event_subscription.ml_event_filter,
                            "tgtUe": item.ml_event_subscription.target_ue,
                            "retainUntil": item.retain_until.isoformat(),
                        }
                        for item in (entry.descriptor for entry in entries)
                    ],
                )
                raise RuntimeError(f"scope {scope} has no usable training-data descriptor")
        return tuple(sorted(resolved, key=lambda item: item.identity))

    @staticmethod
    def _event_filters_match(
        expected: dict,
        actual: dict,
        origin: DescriptorOrigin,
    ) -> bool:
        if not expected:
            return True
        comparable = actual
        if (
            origin is DescriptorOrigin.PRIVATE_API
            and "networkArea" not in expected
            and "aoi" not in expected
        ):
            comparable = dict(actual)
            comparable.pop("networkArea", None)
        return expected == comparable

    @staticmethod
    def _matches_scope(scope: ScopeReference, subscription: dict) -> bool:
        for event in subscription.get("eventSubscriptions") or []:
            if event.get("event") != scope.ml_event:
                continue
            target = event.get("tgtUe") or subscription.get("tgtUe")
            if scope.target_ue is not None and not DatasetCoordinator._targets_match(
                scope.target_ue,
                target,
            ):
                continue
            event_filter = event.get("eventFilter") or event.get("analyticsFilter")
            if not isinstance(event_filter, dict):
                event_filter = {key: event[key] for key in scope.ml_event_filter if key in event}
            if scope.ml_event_filter and event_filter != scope.ml_event_filter:
                continue
            return True
        return False

    @staticmethod
    def _targets_match(expected: dict, actual: dict | None) -> bool:
        """Compare TargetUeInformation by its meaningful selection fields.

        Release 18 models may serialize unset target fields as ``false`` or an
        empty list while a peer sends the equivalent compact JSON form.  Those
        representation defaults must not split one analytics scope into two.
        """

        def normalized(value: dict | None) -> dict:
            if not isinstance(value, dict):
                return {}
            target: dict = {}
            if value.get("anyUe") is True:
                target["anyUe"] = True
            for key in ("supis", "intGroupIds"):
                members = value.get(key)
                if isinstance(members, list) and members:
                    target[key] = sorted(members)
            for key, item in value.items():
                if key in {"anyUe", "supis", "intGroupIds"}:
                    continue
                if item not in (None, False, "", [], {}):
                    target[key] = item
            return target

        return normalized(expected) == normalized(actual)

    def _select_source(self, resources: tuple[ResolvedResource, ...]) -> str:
        if any(not resource.adrf_instance_id for resource in resources):
            return "mongodb"
        adrf_ids = {
            resource.adrf_instance_id
            for resource in resources
            if resource.adrf_instance_id
        }
        if len(adrf_ids) > 1:
            raise RuntimeError("one dataset job cannot span multiple ADRF instances")
        try:
            if self._resolver.resolve(next(iter(adrf_ids), "")):
                return "adrf"
        except Exception as error:
            logger.warning("ADRF is unavailable; using MongoDB fallback: %s", error)
        return "mongodb"

    @staticmethod
    def _ensure_window_coverage(job: DatasetJob) -> None:
        for resource in job.resources:
            if (
                resource.available_start is None
                or resource.available_stop is None
                or resource.available_stop < job.time_window.start_time
                or resource.available_start > job.time_window.stop_time
            ):
                raise RuntimeError(
                    f"training-data descriptor {resource.identity} does not overlap "
                    "the requested dataset window"
                )

    def _retrieve_adrf(self, job: DatasetJob, context: NwdafContext) -> None:
        adrf_ids = {
            resource.adrf_instance_id
            for resource in job.resources
            if resource.adrf_instance_id
        }
        if len(adrf_ids) > 1:
            raise RuntimeError("one dataset job cannot span multiple ADRF instances")
        api_root = self._resolver.resolve(next(iter(adrf_ids), ""))
        if not api_root:
            raise RuntimeError("ADRF could not be resolved")
        callback = context.api_root + "/collector/retrieval-notify"
        go_base = context.internal_api_root
        create_url = go_base + "/internal/v1/adrf-data-management/data-retrieval-subscriptions"
        self._ensure_window_coverage(job)
        for resource in job.resources:
            correlation = str(uuid4())
            route = AdrfRoute(correlation, resource)
            with self._lock:
                job.routes[correlation] = route
                self._routes[correlation] = job.job_id
            payload = NadrfDataRetrievalSubscription(
                notifCorrId=correlation,
                notificationURI=callback,
                timePeriod=job.time_window,
                dataSub=DataSubscription(smfDataSub=resource.smf_data_sub),
                consTrigNotif=True,
            )
            response = self._client.post(
                create_url,
                headers={"Target-Api-Root": api_root},
                json=payload.model_dump(by_alias=True, exclude_none=True, mode="json"),
            )
            if response.status_code != 201 or not response.headers.get("Location"):
                raise RuntimeError(
                    f"ADRF retrieval subscription create failed with {response.status_code}"
                )
            route.peer_subscription_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
        deadline = time.monotonic() + self._settings.watchdog_timeout_seconds
        while True:
            with job.condition:
                if self._closing.is_set():
                    raise RuntimeError("dataset coordinator is shutting down")
                if job.failure:
                    raise RuntimeError(job.failure)
                pending = [
                    (route, fetch_id)
                    for route in job.routes.values()
                    for fetch_id in route.pending_fetch_ids
                ]
                for route, fetch_id in pending:
                    route.pending_fetch_ids.remove(fetch_id)
                complete = all(route.terminated for route in job.routes.values()) and not pending
                if complete:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("ADRF retrieval watchdog expired before termination")
                if not pending:
                    job.condition.wait(timeout=remaining)
                    if job.last_progress:
                        deadline = job.last_progress + self._settings.watchdog_timeout_seconds
                    continue
            for route, fetch_id in pending:
                self._fetch_one(job, route, api_root, fetch_id)
                deadline = time.monotonic() + self._settings.watchdog_timeout_seconds

    def _fetch_one(self, job: DatasetJob, route: AdrfRoute, api_root: str, fetch_id: str) -> None:
        url = self._fetch_url(api_root, fetch_id)
        last_status = 0
        last_error = ""
        for attempt in range(self._settings.max_retry_attempts):
            if self._closing.is_set():
                raise RuntimeError("dataset coordinator is shutting down")
            try:
                response = self._get_with_same_origin_redirects(url, api_root)
            except httpx.TransportError as error:
                last_error = str(error)
                response = None
            if response is None:
                retryable = True
            else:
                last_error = ""
                last_status = response.status_code
                retryable = response.status_code in {429, 500, 502, 503, 504}
            if response is None:
                pass
            elif response.status_code == 204:
                return
            elif response.status_code == 200:
                record = NadrfDataStoreRecord.model_validate(response.json())
                returned_supi = str(record.data_sub[0].smf_data_sub.get("supi", "")).strip()
                if returned_supi != route.resource.supi:
                    raise RuntimeError("ADRF returned a data-store record for a different SUPI")
                self._append_record(job, route.resource, record, "adrf", None, fetch_id)
                return
            elif not retryable:
                break
            if attempt + 1 == self._settings.max_retry_attempts:
                break
            self._wait_for_retry(attempt, honor_shutdown=True)
        if last_error:
            raise RuntimeError(f"ADRF fetch {fetch_id} failed: {last_error}")
        raise RuntimeError(f"ADRF fetch {fetch_id} failed with {last_status}")

    @staticmethod
    def _same_origin(left: str, right: str) -> bool:
        left_url = urlsplit(left)
        right_url = urlsplit(right)
        return (
            left_url.scheme.lower(),
            left_url.hostname,
            left_url.port,
        ) == (
            right_url.scheme.lower(),
            right_url.hostname,
            right_url.port,
        )

    @staticmethod
    def _fetch_url(api_root: str, fetch_id: str) -> str:
        return (
            normalize_api_root(api_root)
            + "/nadrf-datamanagement/v1/data-store-records?"
            + urlencode({"fetch-correlation-ids": fetch_id})
        )

    def _get_with_same_origin_redirects(self, url: str, api_root: str) -> httpx.Response:
        current = url
        for redirect_count in range(self._settings.max_redirects + 1):
            response = self._client.get(current, follow_redirects=False)
            if response.status_code not in {307, 308}:
                return response
            if redirect_count == self._settings.max_redirects:
                raise RuntimeError("ADRF fetch exceeded max_redirects")
            location = response.headers.get("Location", "")
            target = urljoin(current, location)
            if not location or not self._same_origin(target, api_root):
                raise RuntimeError("ADRF fetch redirect is missing or crosses origin")
            current = target
        raise RuntimeError("ADRF fetch redirect handling exhausted")

    def _retrieve_mongo(self, job: DatasetJob) -> None:
        from pymongo.errors import PyMongoError

        last_error: Exception | None = None
        for attempt in range(self._settings.max_retry_attempts):
            if self._closing.is_set():
                raise RuntimeError("dataset coordinator is shutting down")
            try:
                records, malformed = self._read_mongo_once(job)
            except PyMongoError as error:
                last_error = error
                if attempt + 1 == self._settings.max_retry_attempts:
                    break
                self._wait_for_retry(attempt, honor_shutdown=True)
                continue
            job.malformed_records += malformed
            for resource, record, measurement_time, native_id in records:
                self._append_record(
                    job,
                    resource,
                    record,
                    "mongodb",
                    measurement_time,
                    native_id,
                )
            return
        raise RuntimeError(f"Mongo dataset query failed: {last_error}")

    def _read_mongo_once(
        self,
        job: DatasetJob,
    ) -> tuple[
        list[tuple[ResolvedResource, NadrfDataStoreRecord, datetime, str]],
        int,
    ]:
        from pymongo import ASCENDING, MongoClient

        mongo = self._settings.mongodb
        client = MongoClient(
            mongo.url,
            serverSelectionTimeoutMS=mongo.connect_timeout_ms,
            connectTimeoutMS=mongo.connect_timeout_ms,
            socketTimeoutMS=mongo.read_timeout_ms,
            tz_aware=True,
        )
        records: list[tuple[ResolvedResource, NadrfDataStoreRecord, datetime, str]] = []
        malformed = 0
        try:
            client.admin.command("ping")
            supis = sorted({resource.supi for resource in job.resources})
            query = {
                "supi": {"$in": supis},
                "measurementTime": {
                    "$gte": job.time_window.start_time,
                    "$lte": job.time_window.stop_time,
                },
                "dataNotif.upfEventNotifs.0": {"$exists": True},
            }
            cursor = (
                client[mongo.database][mongo.collection]
                .find(query)
                .sort([("measurementTime", ASCENDING), ("_id", ASCENDING)])
            )
            count = 0
            by_supi: dict[str, list[ResolvedResource]] = {}
            for resource in job.resources:
                by_supi.setdefault(resource.supi, []).append(resource)
            for document in cursor:
                count += 1
                if count > self._settings.max_records_per_job:
                    raise RuntimeError("Mongo dataset exceeds max_records_per_job")
                supi = str(document.get("supi", "")).strip()
                matching_resources = by_supi.get(supi, [])
                if not matching_resources:
                    continue
                try:
                    measurement_time = document.get("measurementTime")
                    if not isinstance(measurement_time, datetime):
                        raise ValueError("measurementTime is not a datetime")
                    record = NadrfDataStoreRecord.model_validate(
                        {
                            "dataSub": document.get("dataSub"),
                            "dataNotif": document.get("dataNotif"),
                        }
                    )
                    record_supi = str(record.data_sub[0].smf_data_sub.get("supi", "")).strip()
                    if record_supi != supi:
                        raise ValueError("top-level and dataSub SUPI do not match")
                except (TypeError, ValueError) as error:
                    malformed += 1
                    logger.warning(
                        "Skipping malformed Mongo dataset record id=%s error=%s",
                        document.get("_id", ""),
                        error,
                    )
                    continue
                resource = ResolvedResource(
                    identity="|".join(item.identity for item in matching_resources),
                    scope_keys=tuple(
                        sorted({scope for item in matching_resources for scope in item.scope_keys})
                    ),
                    smf_data_sub=matching_resources[0].smf_data_sub,
                    supi=matching_resources[0].supi,
                    descriptor_origin=matching_resources[0].descriptor_origin,
                    collection_group_id=matching_resources[0].collection_group_id,
                )
                records.append(
                    (
                        resource,
                        record,
                        measurement_time,
                        str(document.get("_id", "")),
                    )
                )
        finally:
            client.close()
        return records, malformed

    def _append_record(
        self,
        job: DatasetJob,
        resource: ResolvedResource,
        record: NadrfDataStoreRecord,
        source: str,
        measurement_time: datetime | None,
        native_id: str,
    ) -> None:
        if not record.data_notif.upf_event_notifs:
            return
        payload = record.model_dump(by_alias=True, exclude_none=True, mode="json")
        identity = native_id or str(uuid4())
        if any(
            item.identity == identity
            and item.source == source
            for item in job.records
        ):
            return
        if len(job.records) >= self._settings.max_records_per_job:
            raise RuntimeError("dataset exceeds max_records_per_job")
        job.records.append(
            DatasetRecord(
                identity,
                resource.scope_keys,
                source,
                measurement_time,
                payload,
            )
        )

    def _complete(self, job: DatasetJob) -> None:
        counts = {
            scope: sum(scope in record.scope_keys for record in job.records)
            for scope in job.intent.active_scope_keys
        }
        missing = [scope for scope, count in counts.items() if count == 0]
        if missing:
            raise RuntimeError(f"required scopes have no valid records: {missing}")
        job.snapshot = DatasetSnapshot(
            job.job_id,
            job.intent.family_key,
            job.intent.triggering_scope_key,
            job.intent.active_scope_keys,
            job.time_window,
            job.source,
            tuple(job.records),
            counts,
        )
        job.state = DatasetJobState.READY
        logger.info(
            "Dataset ready job_id=%s source=%s records=%s scope_counts=%s",
            job.job_id,
            job.source,
            len(job.records),
            counts,
        )
        with self._lock:
            ready_handler = self._ready_handler
        if ready_handler is not None:
            ready_handler(job.job_id)

    def _cleanup(self, job: DatasetJob) -> None:
        try:
            context = self._nwdaf_context.get()
        except RuntimeError:
            return
        go_base = context.internal_api_root
        for route in job.routes.values():
            if route.peer_subscription_id and not route.cleanup_complete:
                terminal = False
                last_status = 0
                try:
                    for attempt in range(self._settings.max_retry_attempts):
                        try:
                            response = self._client.delete(
                                go_base + "/internal/v1/adrf-data-management/"
                                "data-retrieval-subscriptions/" + route.peer_subscription_id
                            )
                            last_status = response.status_code
                            if response.status_code in {204, 404}:
                                terminal = True
                                break
                            if response.status_code not in {429, 500, 502, 503, 504}:
                                break
                        except httpx.TransportError:
                            if attempt + 1 == self._settings.max_retry_attempts:
                                break
                        if attempt + 1 < self._settings.max_retry_attempts:
                            self._wait_for_retry(attempt, honor_shutdown=False)
                    if not terminal:
                        route_error = (
                            "ADRF cleanup did not converge "
                            f"status={last_status} subscription_id="
                            f"{route.peer_subscription_id}"
                        )
                        job.cleanup_failure = route_error
                        logger.warning("%s", route_error)
                finally:
                    if terminal:
                        route.cleanup_complete = True
                        route.peer_subscription_id = ""
                        if all(
                            item.cleanup_complete or not item.peer_subscription_id
                            for item in job.routes.values()
                        ):
                            job.cleanup_failure = ""
            with self._lock:
                self._routes.pop(route.correlation_id, None)

    def _wait_for_retry(self, attempt: int, *, honor_shutdown: bool) -> None:
        delay = min(
            self._settings.retry_initial_backoff_seconds * (2**attempt),
            self._settings.retry_max_backoff_seconds,
        )
        if honor_shutdown:
            if self._closing.wait(delay):
                raise RuntimeError("dataset coordinator is shutting down")
        else:
            time.sleep(delay)
