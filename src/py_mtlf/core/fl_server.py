import hashlib
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

import httpx

from py_mtlf.config import FederatedLearningSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.fl_artifacts import RoundLocalArtifact, validate_fl_artifact
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.seed_catalog import ModelCatalog
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.core.trainer import LoadedBundle, TrustedBundleLoader
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.ml_model import MLEventNotification, MLEventSubscription, MLModelAddress
from py_mtlf.wire.ml_model_training import (
    DataAvReq,
    DCCFEvent,
    MLModelTrainInfo,
    MLTrainReportInfo,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    TimeWindow,
    TrainingResourceIdentity,
    validate_fl_notification,
)
from py_mtlf.wire.private import SelectedTarget, selected_target_headers
from py_mtlf.wire.reporting import ReportingInformation

logger = logging.getLogger(__name__)


class FLServerState(StrEnum):
    CREATED = "CREATED"
    DISCOVERING = "DISCOVERING"
    PREPARATION_CREATING = "PREPARATION_CREATING"
    PREPARATION_WAITING = "PREPARATION_WAITING"
    READY = "READY"
    ROUND_DISPATCH = "ROUND_DISPATCH"
    ROUND_WAITING = "ROUND_WAITING"
    AGGREGATING = "AGGREGATING"
    CANDIDATE_READY = "CANDIDATE_READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FLClientCandidate:
    target: SelectedTarget
    tracking_areas: tuple[str, ...]


@dataclass
class FLParticipant:
    scope: ScopeReference
    candidate: FLClientCandidate
    notification_correlation_id: str
    resource_location: str = ""
    preparation_complete: bool = False
    expected_round: int | None = None
    notification: NwdafMLModelTrainNotif | None = None
    delay_extensions: int = 0
    requested_extension: int = 0
    granted_extension_seconds: int = 0
    expected_scope_digest: str = ""
    accepted_notification_digest: str = ""
    accepted_delay_notification_digest: str = ""

    @property
    def identity(self) -> TrainingResourceIdentity:
        return TrainingResourceIdentity(
            subscription_id=self.resource_location.rsplit("/", 1)[-1],
            ml_correlation_id="",
            notification_correlation_id=self.notification_correlation_id,
            expected_round_indicator=self.expected_round,
            notification_method="ON_EVENT_DETECTION",
        )


@dataclass
class FLProcess:
    process_id: str
    intent: RetrainIntent
    state: FLServerState = FLServerState.CREATED
    participants: list[FLParticipant] = field(default_factory=list)
    current_global_url: str = ""
    candidate_url: str = ""
    failure: str = ""
    cleanup_failure: str = ""
    base_artifact_key: str = ""
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )


class FLClientResolver:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        projection: SyncProjection,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._projection = projection
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds, follow_redirects=False
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def discover(
        self,
        scope: ScopeReference,
        model_interoperability: str,
    ) -> tuple[FLClientCandidate, ...]:
        snapshot = self._projection.snapshot()
        if snapshot is None:
            raise RuntimeError("backend sync is unavailable")
        owner_id = scope.consumer_id.strip()
        if not owner_id:
            raise RuntimeError(f"FL scope {scope.scope_key} has no monitor owner consumerId")
        tracking_areas = _scope_tracking_area_values(scope)
        if not tracking_areas:
            raise RuntimeError(f"FL scope {scope.scope_key} has no tracking area")
        base = snapshot.containing_nwdaf.internal_callback_base_uri.rstrip("/")
        response = self._client.get(
            base + "/internal/v1/nrf/nf-instances",
            params={
                "target-nf-type": "NWDAF",
                "requester-nf-type": "NWDAF",
                "target-nf-instance-id": owner_id,
                "service-names": "nnwdaf-mlmodeltraining",
                "ml-analytics-info-list": json.dumps(
                    [
                        {
                            "mlAnalyticsIds": [scope.ml_event],
                            "trackingAreaList": tracking_areas,
                            "flCapabilityType": "FL_CLIENT",
                            "mlModelInterInfo": {
                                "vendorList": [model_interoperability],
                            },
                        }
                    ],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        )
        response.raise_for_status()
        result = response.json()
        profiles = result.get("nfInstances")
        if not isinstance(profiles, list):
            raise RuntimeError("NRF discovery returned malformed nfInstances")
        candidates = []
        for profile in profiles:
            if not isinstance(profile, dict) or profile.get("nfStatus") not in {
                None,
                "",
                "REGISTERED",
            }:
                continue
            nf_id = str(profile.get("nfInstanceId", ""))
            areas = _fl_client_tracking_areas(profile, scope.ml_event, model_interoperability)
            if (
                not nf_id
                or nf_id != owner_id
                or nf_id == snapshot.containing_nwdaf.nf_instance_id
                or areas is None
            ):
                continue
            for service_id, service in _services(profile):
                if (
                    service.get("serviceName") != "nnwdaf-mlmodeltraining"
                    or service.get("nfServiceStatus") != "REGISTERED"
                ):
                    continue
                root = service.get("apiPrefix") or _derive_root(profile, service)
                if root:
                    candidates.append(
                        FLClientCandidate(
                            target=SelectedTarget(
                                nfInstanceId=nf_id,
                                nfServiceInstanceId=service_id,
                                serviceName="nnwdaf-mlmodeltraining",
                                apiRoot=str(root).rstrip("/"),
                                selectionSource="NRF",
                            ),
                            tracking_areas=areas,
                        )
                    )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.target.nf_instance_id,
                    item.target.nf_service_instance_id,
                    item.target.api_root,
                ),
            )
        )


class FLServerOrchestrator:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        projection: SyncProjection,
        policy: AccuracyPolicy,
        catalog: ModelCatalog,
        workspace: FLWorkspace,
        resolver: FLClientResolver,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._projection = projection
        self._policy = policy
        self._catalog = catalog
        self._workspace = workspace
        self._resolver = resolver
        self._loader = TrustedBundleLoader()
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds, follow_redirects=False
        )
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._processes: dict[str, FLProcess] = {}
        self._correlations: dict[str, str] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_active_server_processes,
            thread_name_prefix="fl-server",
        )
        self._futures: set[Future] = set()
        self._closing = threading.Event()

    def close(self) -> None:
        self._closing.set()
        with self._lock:
            processes = tuple(self._processes.values())
        for process in processes:
            with process.condition:
                process.condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._resolver.close()
        if self._owns_client:
            self._client.close()

    def accept_policy_intents(self) -> None:
        if self._closing.is_set():
            return
        for intent in self._policy.take_intents():
            process = FLProcess(process_id=str(uuid4()), intent=intent)
            with self._lock:
                self._processes[process.process_id] = process
                active = sum(
                    item.state not in {FLServerState.CANDIDATE_READY, FLServerState.FAILED}
                    for item in self._processes.values()
                )
                if active > self._settings.max_active_server_processes:
                    process.state = FLServerState.FAILED
                    process.failure = "FL Server process capacity is exhausted"
                    self._policy.complete_retrain(intent.family_key)
                    continue
            future = self._executor.submit(self._run, process)
            with self._lock:
                self._futures.add(future)
            future.add_done_callback(self._future_done)

    def receive_notification(self, notification: NwdafMLModelTrainNotif) -> None:
        with self._lock:
            process_id = self._correlations.get(notification.notification_correlation_id)
            process = self._processes.get(process_id or "")
        if process is None:
            raise KeyError(notification.notification_correlation_id)
        with process.condition:
            participant = next(
                (
                    item
                    for item in process.participants
                    if item.notification_correlation_id == notification.notification_correlation_id
                ),
                None,
            )
            if participant is None:
                raise KeyError(notification.notification_correlation_id)
            identity = participant.identity
            identity = TrainingResourceIdentity(
                subscription_id=identity.subscription_id,
                ml_correlation_id=process.process_id,
                notification_correlation_id=identity.notification_correlation_id,
                expected_round_indicator=identity.expected_round_indicator,
                notification_method=identity.notification_method,
            )
            validate_fl_notification(notification, identity)
            digest = hashlib.sha256(
                json.dumps(
                    notification.model_dump(
                        by_alias=True,
                        exclude_none=True,
                        mode="json",
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            active_preparation = process.state in {
                FLServerState.PREPARATION_CREATING,
                FLServerState.PREPARATION_WAITING,
            }
            active_round = process.state in {
                FLServerState.ROUND_DISPATCH,
                FLServerState.ROUND_WAITING,
            }
            if (
                notification.status_report is not None
                and not active_preparation
                and participant.accepted_notification_digest != digest
            ):
                process.failure = "preparation callback arrived outside the expected stage"
                process.condition.notify_all()
                raise ValueError(process.failure)
            if (
                notification.ml_model_infos
                and not active_round
                and participant.accepted_notification_digest != digest
            ):
                process.failure = "round callback arrived outside the expected stage"
                process.condition.notify_all()
                raise ValueError(process.failure)
            if notification.delay_event_notification is not None:
                if participant.accepted_delay_notification_digest == digest:
                    return
                if not (active_preparation or active_round):
                    process.failure = "delay callback arrived outside the expected stage"
                    process.condition.notify_all()
                    raise ValueError(process.failure)
                participant.accepted_delay_notification_digest = digest
                participant.requested_extension = (
                    notification.delay_event_notification.expected_completion_time or 0
                )
            elif notification.termination_request:
                process.failure = (
                    f"participant terminated training: {notification.termination_request}"
                )
            elif notification.status_report is not None:
                if participant.preparation_complete:
                    if participant.accepted_notification_digest != digest:
                        process.failure = "conflicting duplicate preparation callback"
                        raise ValueError(process.failure)
                    return
                participant.preparation_complete = True
                participant.accepted_notification_digest = digest
            elif notification.ml_model_infos:
                if participant.notification is not None:
                    if participant.accepted_notification_digest != digest:
                        process.failure = "conflicting duplicate round callback"
                        raise ValueError(process.failure)
                    return
                participant.notification = notification.model_copy(deep=True)
                participant.accepted_notification_digest = digest
            process.condition.notify_all()

    def discard_restored_routes(self, subscription_ids: tuple[str, ...]) -> None:
        if not subscription_ids:
            return
        future = self._executor.submit(self._delete_restored_routes, subscription_ids)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._future_done)

    def processes(self) -> tuple[FLProcess, ...]:
        with self._lock:
            return tuple(self._processes.values())

    def _run(self, process: FLProcess) -> None:
        try:
            logger.info(
                "Federated process started process_id=%s scopes=%s",
                process.process_id,
                process.intent.active_scope_keys,
            )
            process.state = FLServerState.DISCOVERING
            current = self._catalog.current(process.intent.family_key)
            if current is None:
                raise RuntimeError("FL base model is no longer current")
            model_interoperability = current.descriptor.model_interoperability
            if not model_interoperability:
                raise RuntimeError("FL base model has no model interoperability identifier")
            process.base_artifact_key = current.artifact.key
            discovered: dict[tuple[str, str, str], FLClientCandidate] = {}
            for scope in sorted(process.intent.active_scopes, key=lambda item: item.scope_key):
                for candidate in self._resolver.discover(scope, model_interoperability):
                    key = (
                        candidate.target.nf_instance_id,
                        candidate.target.nf_service_instance_id,
                        candidate.target.api_root,
                    )
                    discovered[key] = candidate
            candidates = tuple(discovered[key] for key in sorted(discovered))
            assignments = _assign(process.intent.active_scopes, candidates)
            process.participants = [
                FLParticipant(
                    scope=scope,
                    candidate=candidate,
                    notification_correlation_id=str(uuid4()),
                )
                for scope, candidate in assignments
            ]
            with self._lock:
                for participant in process.participants:
                    self._correlations[participant.notification_correlation_id] = process.process_id
            process.state = FLServerState.PREPARATION_CREATING
            for participant in process.participants:
                self._create_preparation(
                    process,
                    participant,
                    model_interoperability,
                    current.artifact.url,
                )
            process.state = FLServerState.PREPARATION_WAITING
            self._wait(
                process,
                lambda: all(item.preparation_complete for item in process.participants),
                self._settings.preparation_timeout_seconds,
            )
            logger.info(
                "Federated preparation complete process_id=%s participants=%s",
                process.process_id,
                [item.candidate.target.nf_instance_id for item in process.participants],
            )
            process.state = FLServerState.READY
            current = self._catalog.current(process.intent.family_key)
            if current is None or current.artifact.key != process.base_artifact_key:
                raise RuntimeError("FL base model changed while participants were preparing")
            process.current_global_url = current.artifact.url
            for round_indicator in range(self._settings.round_count):
                process.state = FLServerState.ROUND_DISPATCH
                for participant in process.participants:
                    participant.expected_round = round_indicator
                    participant.notification = None
                    participant.accepted_notification_digest = ""
                    participant.accepted_delay_notification_digest = ""
                    participant.delay_extensions = 0
                    participant.requested_extension = 0
                    participant.granted_extension_seconds = 0
                    self._patch_round(
                        process,
                        participant,
                        round_indicator,
                        process.current_global_url,
                    )
                process.state = FLServerState.ROUND_WAITING
                self._wait(
                    process,
                    lambda: all(item.notification is not None for item in process.participants),
                    self._settings.round_timeout_seconds,
                )
                process.state = FLServerState.AGGREGATING
                process.current_global_url = self._aggregate_round(
                    process, current.artifact, round_indicator
                )
                self._raise_if_failed(process)
                logger.info(
                    "Federated round aggregated process_id=%s round=%s artifact=%s",
                    process.process_id,
                    round_indicator,
                    process.current_global_url,
                )
            with process.condition:
                if process.failure:
                    raise RuntimeError(process.failure)
                process.candidate_url = process.current_global_url
                process.state = FLServerState.CANDIDATE_READY
            logger.info(
                "Federated candidate ready process_id=%s artifact=%s",
                process.process_id,
                process.candidate_url,
            )
        except Exception as error:
            with process.condition:
                process.state = FLServerState.FAILED
                process.failure = str(error)
                process.condition.notify_all()
            logger.exception("Federated process failed process_id=%s", process.process_id)
        finally:
            for participant in process.participants:
                if participant.resource_location:
                    failure = self._cleanup_participant(process, participant)
                    if failure:
                        process.cleanup_failure = f"{process.cleanup_failure}; {failure}".strip(
                            "; "
                        )
            with self._lock:
                for participant in process.participants:
                    self._correlations.pop(participant.notification_correlation_id, None)
            self._policy.complete_retrain(process.intent.family_key)

    def _create_preparation(
        self,
        process: FLProcess,
        participant: FLParticipant,
        model_interoperability: str,
        base_model_url: str,
    ) -> None:
        now = datetime.now(UTC)
        event = MLEventSubscription(
            mLEvent=participant.scope.ml_event,
            mLEventFilter=participant.scope.ml_event_filter,
            tgtUe=participant.scope.target_ue,
            modelInterInfo=model_interoperability,
        )
        value = NwdafMLModelTrainSubsc(
            mLEventSubscs=[event],
            notifUri=self._settings.callback_uri,
            notifCorreId=participant.notification_correlation_id,
            mlCorreId=process.process_id,
            mLPreFlag=True,
            mLModelInfos=[
                MLEventNotification(
                    event=participant.scope.ml_event,
                    mLFileAddr=MLModelAddress(mLModelUrl=base_model_url),
                )
            ],
            eventReq=ReportingInformation(notifMethod="ON_EVENT_DETECTION"),
            tgtRepUe=participant.scope.target_ue,
            mLModelTrainInfos=[
                MLModelTrainInfo(
                    dataAvReq=DataAvReq(
                        inpEvents=[DCCFEvent(upfEvent="USER_DATA_USAGE_TRENDS")],
                        minNumSamples=1,
                        timeWindows=[
                            TimeWindow(
                                startTime=now - timedelta(minutes=30),
                                stopTime=now,
                            )
                        ],
                    ),
                    timeAvReq=f"PT{self._settings.preparation_timeout_seconds}S",
                )
            ],
            mLTrainRepInfo=MLTrainReportInfo(maxResTime=self._settings.preparation_timeout_seconds),
        )
        go_base = self._go_base()
        response = self._client.post(
            go_base + "/internal/v1/ml-model-training/subscriptions",
            headers=selected_target_headers(participant.candidate.target),
            json=value.model_dump(by_alias=True, exclude_none=True, mode="json"),
        )
        if response.status_code != 201 or not response.headers.get("Location"):
            raise RuntimeError(
                "participant preparation create failed with "
                f"{response.status_code}: {response.text}"
            )
        participant.resource_location = response.headers["Location"]
        participant.expected_scope_digest = TrainingScopeDescriptor.from_training_request(
            value, 0
        ).scope_digest

    def _patch_round(
        self,
        process: FLProcess,
        participant: FLParticipant,
        round_indicator: int,
        artifact_url: str,
    ) -> None:
        patch = NwdafMLModelTrainSubscPatch(
            mLPreFlag=False,
            roundInd=round_indicator,
            mLModelInfos=[
                MLEventNotification(
                    event=participant.scope.ml_event,
                    mLFileAddr=MLModelAddress(mLModelUrl=artifact_url),
                )
            ],
            mLTrainRepInfo=MLTrainReportInfo(maxResTime=self._settings.round_timeout_seconds),
        )
        response = self._client.patch(
            participant.resource_location,
            headers={"Content-Type": "application/merge-patch+json"},
            content=patch.model_dump_json(by_alias=True, exclude_none=True),
        )
        if response.status_code not in {200, 204}:
            raise RuntimeError(f"participant round patch failed with {response.status_code}")

    def _wait(self, process: FLProcess, predicate, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        with process.condition:
            while True:
                if self._closing.is_set():
                    raise RuntimeError("FL Server is shutting down")
                if process.failure:
                    raise RuntimeError(process.failure)
                if predicate():
                    return
                for participant in process.participants:
                    if participant.requested_extension:
                        if participant.delay_extensions >= self._settings.max_delay_extensions:
                            raise RuntimeError("participant exceeded delay extension limit")
                        remaining_budget = (
                            self._settings.max_delay_extension_seconds
                            - participant.granted_extension_seconds
                        )
                        extension = min(
                            participant.requested_extension,
                            timeout,
                            remaining_budget,
                        )
                        if extension <= 0:
                            raise RuntimeError("participant delay extension budget is exhausted")
                        self._grant_extension(participant, extension)
                        participant.delay_extensions += 1
                        participant.granted_extension_seconds += extension
                        participant.requested_extension = 0
                        deadline = max(deadline, time.monotonic() + extension)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("federated stage deadline expired")
                process.condition.wait(timeout=remaining)

    @staticmethod
    def _raise_if_failed(process: FLProcess) -> None:
        with process.condition:
            if process.failure:
                raise RuntimeError(process.failure)

    def _cleanup_participant(
        self,
        process: FLProcess,
        participant: FLParticipant,
    ) -> str:
        last_error = ""
        for attempt in range(self._settings.cleanup_max_attempts):
            try:
                response = self._client.delete(participant.resource_location)
                if response.status_code in {204, 404}:
                    return ""
                last_error = f"cleanup returned {response.status_code}"
            except httpx.TransportError as error:
                last_error = str(error)
            if attempt + 1 < self._settings.cleanup_max_attempts:
                time.sleep(self._settings.cleanup_retry_backoff_seconds)
        logger.warning(
            "FL participant cleanup failed process_id=%s nf=%s error=%s",
            process.process_id,
            participant.candidate.target.nf_instance_id,
            last_error,
        )
        return (
            f"participant {participant.candidate.target.nf_instance_id} cleanup failed: "
            f"{last_error}"
        )

    def _grant_extension(self, participant: FLParticipant, extension: int) -> None:
        patch = NwdafMLModelTrainSubscPatch(mLTrainRepInfo=MLTrainReportInfo(maxResTime=extension))
        response = self._client.patch(
            participant.resource_location,
            headers={"Content-Type": "application/merge-patch+json"},
            content=patch.model_dump_json(by_alias=True, exclude_none=True),
        )
        if response.status_code not in {200, 204}:
            raise RuntimeError("participant delay extension PATCH failed")

    def _aggregate_round(
        self,
        process: FLProcess,
        base_artifact: ArtifactMetadata,
        round_indicator: int,
    ) -> str:
        if round_indicator > 0:
            base_artifact = self._workspace.download(
                process.current_global_url,
                process.process_id,
                f"round-{round_indicator}-global-input",
            )
        base = self._loader.load(base_artifact)
        base_digest = weights_digest(base.model)
        expected_model_contract = model_contract_digest(base.manifest)
        expected_preprocessing_contract = preprocessing_contract_digest(base.manifest)
        local_bundles: list[tuple[LoadedBundle, int]] = []
        participant_metadata = []
        for participant in process.participants:
            notification = participant.notification
            if notification is None or not notification.ml_model_infos:
                raise RuntimeError("participant local result is missing")
            address = notification.ml_model_infos[0].model_file_address
            if address is None or address.model_url is None:
                raise RuntimeError("participant local result has no model URL")
            artifact = self._workspace.download(
                str(address.model_url),
                process.process_id,
                f"round-{round_indicator}-{participant.candidate.target.nf_instance_id}",
            )
            bundle = self._loader.load(artifact)
            projection = _artifact_projection(bundle.manifest)
            contract = validate_fl_artifact(projection)
            if not isinstance(contract, RoundLocalArtifact):
                raise RuntimeError("participant returned a non-local FL artifact")
            metadata = contract.fl_metadata
            if (
                metadata.ml_corre_id != process.process_id
                or metadata.round_ind != round_indicator
                or metadata.participant_nf_instance_id
                != participant.candidate.target.nf_instance_id
                or metadata.scope_digest != participant.expected_scope_digest
                or metadata.input_global_weights_digest != base_digest
                or metadata.model_contract_digest != expected_model_contract
                or metadata.preprocessing_contract_digest != expected_preprocessing_contract
            ):
                raise RuntimeError("participant local artifact identity does not match assignment")
            local_bundles.append((bundle, metadata.training_sample_count))
            participant_metadata.append(
                {
                    "participant_nf_instance_id": metadata.participant_nf_instance_id,
                    "training_sample_count": metadata.training_sample_count,
                    "local_artifact_digest": artifact.key,
                }
            )
        aggregate = FederatedTrainer.aggregate(base, tuple(local_bundles))
        participant_metadata.sort(key=lambda item: item["participant_nf_instance_id"])
        output_digest = weights_digest(aggregate)
        published = self._workspace.publish(
            process_id=process.process_id,
            participant_id=self._server_id(),
            round_indicator=round_indicator,
            role="ROUND_GLOBAL",
            base=base,
            model=aggregate,
            metadata={
                "artifact_role": "ROUND_GLOBAL",
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": process.process_id,
                    "model_contract_digest": model_contract_digest(base.manifest),
                    "preprocessing_contract_digest": preprocessing_contract_digest(base.manifest),
                    "base_weights_digest": base_digest,
                    "weights_digest": output_digest,
                    "round_ind": round_indicator,
                    "participants": participant_metadata,
                    "aggregated_training_sample_count": sum(
                        item["training_sample_count"] for item in participant_metadata
                    ),
                },
            },
        )
        return published.url

    def _go_base(self) -> str:
        snapshot = self._projection.snapshot()
        if snapshot is None:
            raise RuntimeError("backend sync is unavailable")
        return snapshot.containing_nwdaf.internal_callback_base_uri.rstrip("/")

    def _server_id(self) -> str:
        snapshot = self._projection.snapshot()
        if snapshot is None:
            raise RuntimeError("backend sync is unavailable")
        return snapshot.containing_nwdaf.nf_instance_id

    def _future_done(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _delete_restored_routes(self, subscription_ids: tuple[str, ...]) -> None:
        base = self._go_base()
        for subscription_id in subscription_ids:
            last_error = ""
            for attempt in range(self._settings.cleanup_max_attempts):
                try:
                    response = self._client.delete(
                        base + "/internal/v1/ml-model-training/subscriptions/" + subscription_id
                    )
                    if response.status_code in {204, 404}:
                        last_error = ""
                        break
                    last_error = f"status {response.status_code}"
                except httpx.TransportError as error:
                    last_error = str(error)
                if attempt + 1 < self._settings.cleanup_max_attempts and self._closing.wait(
                    self._settings.cleanup_retry_backoff_seconds
                ):
                    break
            if last_error:
                logger.warning(
                    "Restored outbound FL route cleanup failed route=%s error=%s",
                    subscription_id,
                    last_error,
                )


def _assign(
    scopes: tuple[ScopeReference, ...],
    candidates: tuple[FLClientCandidate, ...],
) -> tuple[tuple[ScopeReference, FLClientCandidate], ...]:
    assignments = []
    used = set()
    for scope in sorted(scopes, key=lambda item: item.scope_key):
        owner_id = scope.consumer_id.strip()
        required = _scope_tais(scope)
        candidate = next(
            (
                item
                for item in candidates
                if item.target.nf_instance_id == owner_id
                and item.target.nf_instance_id not in used
                and (not required or required.intersection(item.tracking_areas))
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(
                f"monitor owner {owner_id or '<missing>'} is not an eligible "
                f"FL Client for scope {scope.scope_key}"
            )
        assignments.append((scope, candidate))
        used.add(candidate.target.nf_instance_id)
    if len(assignments) < 2:
        raise RuntimeError("the first FL profile requires two distinct clients")
    return tuple(assignments)


def _scope_tais(scope: ScopeReference) -> set[str]:
    return {_tai_key(item) for item in _scope_tracking_area_values(scope) if _tai_key(item)}


def _scope_tracking_area_values(scope: ScopeReference) -> list[dict]:
    area = scope.ml_event_filter.get("networkArea") or scope.ml_event_filter.get("aoi") or {}
    tais = area.get("tais") if isinstance(area, dict) else []
    return [dict(item) for item in tais or [] if isinstance(item, dict) and _tai_key(item)]


def _fl_client_tracking_areas(
    profile: dict,
    ml_event: str,
    model_interoperability: str,
) -> tuple[str, ...] | None:
    infos = []
    if isinstance(profile.get("nwdafInfo"), dict):
        infos.append(profile["nwdafInfo"])
    if isinstance(profile.get("nwdafInfoList"), dict):
        infos.extend(
            value for value in profile["nwdafInfoList"].values() if isinstance(value, dict)
        )
    areas = set()
    supported = False
    for info in infos:
        for entry in info.get("mlAnalyticsList") or []:
            if not isinstance(entry, dict):
                continue
            if ml_event not in (entry.get("mlAnalyticsIds") or []):
                continue
            if entry.get("flCapabilityType") not in {"FL_CLIENT", "FL_SERVER_AND_CLIENT"}:
                continue
            interoperability = entry.get("mlModelInterInfo") or {}
            if model_interoperability not in (interoperability.get("vendorList") or []):
                continue
            supported = True
            areas.update(
                _tai_key(item)
                for item in entry.get("trackingAreaList") or []
                if isinstance(item, dict) and _tai_key(item)
            )
    return tuple(sorted(areas)) if supported else None


def _services(profile: dict):
    values = [
        (str(item.get("serviceInstanceId", "")), item)
        for item in profile.get("nfServices") or []
        if isinstance(item, dict)
    ]
    values.extend(
        (str(item.get("serviceInstanceId") or key), item)
        for key, item in (profile.get("nfServiceList") or {}).items()
        if isinstance(item, dict)
    )
    return values


def _derive_root(profile: dict, service: dict) -> str:
    scheme = service.get("scheme")
    endpoints = service.get("ipEndPoints") or []
    endpoint = endpoints[0] if endpoints else {}
    host = service.get("fqdn") or profile.get("fqdn") or endpoint.get("ipv4Address")
    port = endpoint.get("port")
    if scheme not in {"http", "https"} or not host:
        return ""
    return f"{scheme}://{host}{f':{port}' if port else ''}"


def _tai_key(value: dict) -> str:
    plmn = value.get("plmnId") or {}
    mcc = str(plmn.get("mcc", ""))
    mnc = str(plmn.get("mnc", ""))
    tac = str(value.get("tac", ""))
    return f"{mcc}-{mnc}-{tac}" if mcc and mnc and tac else ""


def _artifact_projection(manifest: dict[str, object]) -> dict[str, object]:
    keys = {"bundle_schema_version", "file_digests", "artifact_role", "fl_metadata"}
    if "result_type" in manifest:
        keys.add("result_type")
    return {key: manifest[key] for key in keys}
