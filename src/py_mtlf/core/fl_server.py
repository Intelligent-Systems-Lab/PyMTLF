import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

import httpx

from py_mtlf.config import FederatedLearningSettings, FLServerSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy, RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.fl_artifacts import (
    HierarchyBranchValidation,
    HierarchyValidation,
    RoundGlobalArtifact,
    RoundInputArtifact,
    RoundLocalAccuracyCheckMetadata,
    RoundLocalArtifact,
    RoundLocalHierarchyAggregateMetadata,
    RoundLocalResultType,
    ValidationSummary,
    validate_fl_artifact,
    wape,
)
from py_mtlf.core.fl_experiment import (
    ExperimentAdmissionClosedError,
    ExperimentLifecycle,
    ExperimentRegistryError,
    FLExperimentRegistry,
)
from py_mtlf.core.fl_orchestration import (
    FlatExecutionRequest,
    FlatParticipantScope,
    MonitorParticipantSelection,
    TriggerSource,
)
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    FLWorkspaceArtifact,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.model_records import ParticipantSampleCount
from py_mtlf.core.notification_delivery import ProvisionNotificationDispatcher
from py_mtlf.core.nwdaf_context import NwdafContextClient
from py_mtlf.core.publication import PublicationCoordinator, ValidatedCandidate
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog
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
    PREPARATION_EVALUATING = "PREPARATION_EVALUATING"
    READY = "READY"
    ROUND_DISPATCH = "ROUND_DISPATCH"
    ROUND_WAITING = "ROUND_WAITING"
    ROUND_EVALUATING = "ROUND_EVALUATING"
    AGGREGATING = "AGGREGATING"
    FINAL_VALIDATION_DISPATCH = "FINAL_VALIDATION_DISPATCH"
    FINAL_VALIDATION_WAITING = "FINAL_VALIDATION_WAITING"
    FINAL_VALIDATION_EVALUATING = "FINAL_VALIDATION_EVALUATING"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    CANDIDATE_READY = "CANDIDATE_READY"
    PUBLISHING = "PUBLISHING"
    CUTOVER_PENDING = "CUTOVER_PENDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class FLServerAdmissionClosedError(RuntimeError):
    pass


class FLServerProcessConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class FLClientCandidate:
    target: SelectedTarget
    tracking_areas: tuple[str, ...]


@dataclass(frozen=True)
class HierarchyPreparationTarget:
    participant_nf_instance_id: str
    candidate: FLClientCandidate
    assignment_url: str


@dataclass
class FLParticipant:
    scope: FlatParticipantScope | ScopeReference
    candidate: FLClientCandidate
    notification_correlation_id: str
    resource_location: str = ""
    preparation_complete: bool = False
    preparation_notification: NwdafMLModelTrainNotif | None = None
    preparation_failure: str = ""
    expected_round: int | None = None
    notification: NwdafMLModelTrainNotif | None = None
    round_complete: bool = False
    round_failure: str = ""
    delay_extensions: int = 0
    requested_extension: int = 0
    granted_extension_seconds: int = 0
    expected_scope_digest: str = ""
    accepted_notification_digest: str = ""
    accepted_delay_notification_digest: str = ""
    training_sample_count: int = 0

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
    intent: RetrainIntent | None
    execution: FlatExecutionRequest | None = None
    generation: int = 0
    state: FLServerState = FLServerState.CREATED
    current_round: int | None = None
    completed_rounds: int = 0
    participants: list[FLParticipant] = field(default_factory=list)
    current_global_url: str = ""
    current_global_artifact: ArtifactMetadata | None = None
    candidate_url: str = ""
    failure: str = ""
    cleanup_failure: str = ""
    base_artifact_key: str = ""
    validation_summaries: tuple[ValidationSummary, ...] = ()
    gate_would_accept: bool | None = None
    gate_rejection_reasons: tuple[str, ...] = ()
    candidate_artifact: ArtifactMetadata | None = None
    published_model_id: int | None = None
    experiment_reservation_id: str = ""
    hierarchy_plan_id: str = ""
    hierarchy_family_key: FamilyKey | None = None
    hierarchy_active_scopes: tuple[ScopeReference, ...] = ()
    hierarchy_cleanup_complete: bool = False
    hierarchy_validation: HierarchyValidation | None = None
    hierarchy_state_observer: Callable[[FLServerState], None] | None = None
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )


@dataclass(frozen=True)
class HierarchyParticipantPreparationOutcome:
    participant_nf_instance_id: str
    resource_location: str
    assignment_url: str
    notification: NwdafMLModelTrainNotif | None
    failure: str
    delay_extensions: int
    granted_extension_seconds: int


@dataclass(frozen=True)
class HierarchyPreparationCollection:
    process_id: str
    plan_id: str
    participants: tuple[HierarchyParticipantPreparationOutcome, ...]
    timed_out_participant_nf_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class HierarchyValidationCollection:
    candidate_artifact: ArtifactMetadata
    validation_summaries: tuple[ValidationSummary, ...]
    hierarchy_branches: tuple[HierarchyBranchValidation, ...] = ()


@dataclass(frozen=True)
class _ValidatedHierarchyCandidate:
    artifact: ArtifactMetadata
    base_weights_digest: str
    candidate_weights_digest: str
    model_contract_digest: str
    preprocessing_contract_digest: str


@dataclass(frozen=True)
class HierarchyFinalizationResult:
    state: FLServerState
    candidate_digest: str
    published_model_id: int | None
    gate_would_accept: bool
    gate_rejection_reasons: tuple[str, ...]


class FLClientResolver:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        nwdaf_context: NwdafContextClient,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._nwdaf_context = nwdaf_context
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds, follow_redirects=False
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def discover(
        self,
        scope: FlatParticipantScope,
        model_interoperability: str,
    ) -> tuple[FLClientCandidate, ...]:
        context = self._nwdaf_context.get()
        owner_id = _participant_nf_instance_id(scope)
        if not owner_id:
            raise RuntimeError(f"FL scope {scope.scope_key} has no participant NF instance ID")
        tracking_areas = _scope_tracking_area_values(scope)
        if not tracking_areas:
            raise RuntimeError(f"FL scope {scope.scope_key} has no tracking area")
        base = context.internal_api_root
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
        required_areas = _scope_tais(scope)
        for profile in profiles:
            if not isinstance(profile, dict) or profile.get("nfStatus") != "REGISTERED":
                continue
            nf_id = str(profile.get("nfInstanceId", ""))
            areas = _fl_client_tracking_areas(profile, scope.ml_event, model_interoperability)
            if (
                not nf_id
                or nf_id != owner_id
                or nf_id == context.nf_instance_id
                or areas is None
                or not required_areas.issubset(areas)
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


class FLServerEngine:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        server_settings: FLServerSettings,
        nwdaf_context: NwdafContextClient,
        policy: AccuracyPolicy,
        catalog: ModelCatalog,
        workspace: FLWorkspace,
        resolver: FLClientResolver,
        client: httpx.Client | None = None,
        publication: PublicationCoordinator | None = None,
        provision_notifications: ProvisionNotificationDispatcher | None = None,
        experiments: FLExperimentRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._server_settings = server_settings
        self._nwdaf_context = nwdaf_context
        self._policy = policy
        self._catalog = catalog
        self._workspace = workspace
        self._resolver = resolver
        self._publication = publication
        self._provision_notifications = provision_notifications
        self._experiments = experiments or FLExperimentRegistry()
        self._loader = TrustedBundleLoader()
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds, follow_redirects=False
        )
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._processes: dict[str, FLProcess] = {}
        self._correlations: dict[str, str] = {}
        self._generation = 0
        self._executor = ThreadPoolExecutor(
            max_workers=server_settings.max_active_processes,
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
        for process in processes:
            if process.hierarchy_plan_id and not process.hierarchy_cleanup_complete:
                self.cancel_hierarchy_preparation(process.process_id, "FL Server is closing")
        self._resolver.close()
        if self._owns_client:
            self._client.close()

    def abort_generation(self, reason: str) -> None:
        """Discard all Training processes owned by the previous Go lifetime."""
        with self._lock:
            self._generation += 1
            processes = tuple(self._processes.values())
        for process in processes:
            with process.condition:
                process.state = FLServerState.FAILED
                if not process.failure:
                    process.failure = reason
                process.condition.notify_all()
        for process in processes:
            if process.hierarchy_plan_id:
                self.close_hierarchy_training(
                    process.process_id,
                    retain_for_adoption=False,
                )
            else:
                for participant in process.participants:
                    if participant.resource_location:
                        failure = self._cleanup_participant(process, participant)
                        if failure:
                            process.cleanup_failure = (
                                f"{process.cleanup_failure}; {failure}".strip("; ")
                            )
                        participant.resource_location = ""
                family_key = _flat_family_key(process)
                if family_key is not None:
                    self._policy.complete_retrain(family_key)
        with self._lock:
            self._processes.clear()
            self._correlations.clear()

    def start_flat(self, execution: FlatExecutionRequest) -> FLProcess:
        if self._closing.is_set():
            raise FLServerAdmissionClosedError("FL Server is closing")
        with self._lock:
            generation = self._generation
        process = FLProcess(
            process_id=str(uuid4()),
            intent=None,
            execution=execution,
            generation=generation,
        )
        try:
            reservation = self._experiments.reserve_server(process.process_id)
            process.experiment_reservation_id = reservation.reservation_id
        except ExperimentAdmissionClosedError as error:
            raise FLServerAdmissionClosedError(str(error)) from error
        except ExperimentRegistryError as error:
            raise FLServerProcessConflictError(str(error)) from error
        with self._lock:
            stale_generation = process.generation != self._generation
            if not stale_generation:
                self._processes[process.process_id] = process
                active = sum(
                    item.state
                    not in {
                        FLServerState.CANDIDATE_READY,
                        FLServerState.VALIDATION_REJECTED,
                        FLServerState.CUTOVER_PENDING,
                        FLServerState.COMPLETE,
                        FLServerState.FAILED,
                    }
                    for item in self._processes.values()
                )
        if stale_generation:
            process.state = FLServerState.FAILED
            process.failure = "containing NWDAF process generation changed"
            self._finish_experiment(process)
            raise FLServerAdmissionClosedError(process.failure)
        if active > self._server_settings.max_active_processes:
            process.state = FLServerState.FAILED
            process.failure = "FL Server process capacity is exhausted"
            self._finish_experiment(process)
            with self._lock:
                self._processes.pop(process.process_id, None)
            raise FLServerProcessConflictError(process.failure)
        try:
            future = self._executor.submit(self._run, process)
        except Exception as error:
            process.state = FLServerState.FAILED
            process.failure = "FL Server request executor is unavailable"
            self._finish_experiment(process)
            with self._lock:
                self._processes.pop(process.process_id, None)
            raise FLServerAdmissionClosedError(process.failure) from error
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._future_done)
        return process

    def start_hierarchy_preparation(
        self,
        *,
        plan_id: str,
        reservation_id: str,
        family_key: FamilyKey | None,
        model_id: int | None,
        ml_event: str,
        ml_event_filter: dict,
        target_ue: dict | None,
        model_interoperability: str,
        targets: tuple[HierarchyPreparationTarget, ...],
        active_scopes: tuple[ScopeReference, ...] = (),
    ) -> FLProcess:
        with self._lock:
            if self._closing.is_set():
                raise RuntimeError("FL Server is closing")
            generation = self._generation
        if not targets:
            raise ValueError("hierarchy preparation requires at least one participant target")
        participant_ids = tuple(item.participant_nf_instance_id for item in targets)
        if participant_ids != tuple(sorted(participant_ids)) or len(participant_ids) != len(
            set(participant_ids)
        ):
            raise ValueError(
                "hierarchy participant targets must be unique and canonically ordered"
            )
        for target in targets:
            if target.candidate.target.nf_instance_id != target.participant_nf_instance_id:
                raise ValueError(
                    "hierarchy participant target identity does not match its candidate"
                )
            if not target.assignment_url.strip():
                raise ValueError("hierarchy participant assignment URL must not be blank")

        process = FLProcess(
            process_id=str(uuid4()),
            intent=None,
            generation=generation,
            experiment_reservation_id=reservation_id,
            hierarchy_plan_id=plan_id,
            hierarchy_family_key=family_key,
            hierarchy_active_scopes=active_scopes,
        )
        self._experiments.attach_server(reservation_id, plan_id, process.process_id)
        process.participants = [
            FLParticipant(
                scope=ScopeReference(
                    scope_key=f"hierarchy:{plan_id}:{target.participant_nf_instance_id}",
                    consumer_id=target.participant_nf_instance_id,
                    model_ids=(model_id,) if model_id is not None else (),
                    ml_event=ml_event,
                    ml_event_filter=dict(ml_event_filter),
                    target_ue=dict(target_ue) if target_ue is not None else None,
                ),
                candidate=target.candidate,
                notification_correlation_id=str(uuid4()),
            )
            for target in targets
        ]
        with self._lock:
            stale_generation = generation != self._generation
            if not stale_generation:
                self._processes[process.process_id] = process
                for participant in process.participants:
                    self._correlations[participant.notification_correlation_id] = (
                        process.process_id
                    )
        if stale_generation:
            self._experiments.detach_server(reservation_id, process.process_id)
            raise RuntimeError("containing NWDAF process generation changed")

        try:
            process.state = FLServerState.PREPARATION_CREATING
            for participant, target in zip(process.participants, targets, strict=True):
                self._ensure_process_generation(process)
                self._create_preparation(
                    process,
                    participant,
                    model_interoperability,
                    target.assignment_url,
                )
            self._ensure_process_generation(process)
            process.state = FLServerState.PREPARATION_WAITING
            logger.info(
                "Hierarchy preparation dispatched plan_id=%s process_id=%s participants=%s",
                plan_id,
                process.process_id,
                participant_ids,
            )
            return process
        except Exception as error:
            process.state = FLServerState.FAILED
            process.failure = str(error)
            self._cleanup_hierarchy_process(process)
            if self._experiments.for_server_process(process.process_id) is not None:
                self._experiments.detach_server(reservation_id, process.process_id)
            with self._lock:
                self._processes.pop(process.process_id, None)
            raise

    def cancel_hierarchy_preparation(self, process_id: str, reason: str) -> None:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None:
            return
        if not process.hierarchy_plan_id:
            raise KeyError(process_id)
        with process.condition:
            process.state = FLServerState.FAILED
            if not process.failure:
                process.failure = reason
            process.condition.notify_all()
        self.close_hierarchy_training(process_id, retain_for_adoption=False)

    def close_hierarchy_training(
        self,
        process_id: str,
        *,
        retain_for_adoption: bool,
    ) -> None:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None:
            return
        if not process.hierarchy_plan_id:
            raise KeyError(process_id)
        self._cleanup_hierarchy_process(process)
        active = self._experiments.for_server_process(process.process_id)
        if active is not None:
            self._experiments.detach_server(
                process.experiment_reservation_id,
                process.process_id,
            )
        with self._lock:
            if not retain_for_adoption:
                self._processes.pop(process_id, None)
            else:
                # Adoption completion only needs the published model/family and
                # Root observer; drop the completed Training procedure payload.
                process.participants.clear()
                process.hierarchy_active_scopes = ()
                process.current_global_url = ""
                process.current_global_artifact = None
                process.candidate_url = ""
                process.candidate_artifact = None
                process.validation_summaries = ()

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
                FLServerState.FINAL_VALIDATION_DISPATCH,
                FLServerState.FINAL_VALIDATION_WAITING,
            }
            identity = participant.identity
            identity = TrainingResourceIdentity(
                subscription_id=identity.subscription_id,
                ml_correlation_id=process.process_id,
                notification_correlation_id=identity.notification_correlation_id,
                expected_round_indicator=identity.expected_round_indicator,
                notification_method=identity.notification_method,
            )
            try:
                validate_fl_notification(notification, identity)
            except ValueError as error:
                if process.hierarchy_plan_id and active_round:
                    participant.round_failure = str(error)
                    participant.round_complete = True
                    participant.accepted_notification_digest = digest
                    process.condition.notify_all()
                raise
            if process.hierarchy_plan_id and not (active_preparation or active_round):
                if digest in {
                    participant.accepted_notification_digest,
                    participant.accepted_delay_notification_digest,
                }:
                    return
                raise ValueError("hierarchy callback arrived after the active stage")
            if process.hierarchy_plan_id and active_round:
                if participant.round_complete:
                    if digest == participant.accepted_notification_digest:
                        return
                    participant.round_failure = "conflicting duplicate round callback"
                    process.condition.notify_all()
                    raise ValueError(participant.round_failure)
                if notification.delay_event_notification is not None and (
                    notification.ml_model_infos or notification.termination_request
                ):
                    participant.round_failure = (
                        "delay callback cannot also contain a terminal round outcome"
                    )
                    participant.round_complete = True
                    participant.accepted_notification_digest = digest
                    process.condition.notify_all()
                    raise ValueError(participant.round_failure)
            if active_preparation:
                if notification.round_indicator is not None:
                    participant.preparation_failure = (
                        "preparation callback must not include roundInd"
                    )
                    if not process.hierarchy_plan_id:
                        process.failure = participant.preparation_failure
                    participant.preparation_complete = True
                    process.condition.notify_all()
                    raise ValueError(participant.preparation_failure)
                if notification.delay_event_notification is not None:
                    if participant.accepted_delay_notification_digest == digest:
                        return
                    participant.accepted_delay_notification_digest = digest
                    participant.requested_extension = (
                        notification.delay_event_notification.expected_completion_time or 0
                    )
                    process.condition.notify_all()
                    return
                if not notification.ml_model_infos and not notification.termination_request:
                    participant.preparation_failure = (
                        "notification does not contain a preparation outcome"
                    )
                    if not process.hierarchy_plan_id:
                        process.failure = participant.preparation_failure
                    participant.preparation_complete = True
                    process.condition.notify_all()
                    raise ValueError(participant.preparation_failure)
                if participant.preparation_notification is not None:
                    if participant.accepted_notification_digest == digest:
                        return
                    participant.preparation_failure = (
                        "conflicting duplicate preparation callback"
                    )
                    if not process.hierarchy_plan_id:
                        process.failure = participant.preparation_failure
                    participant.preparation_complete = True
                    process.condition.notify_all()
                    raise ValueError(participant.preparation_failure)
                participant.preparation_notification = notification.model_copy(deep=True)
                participant.preparation_complete = True
                participant.accepted_notification_digest = digest
                if notification.termination_request and not process.hierarchy_plan_id:
                    process.failure = (
                        "participant terminated training: "
                        f"{notification.termination_request}"
                    )
                process.condition.notify_all()
                return
            if notification.delay_event_notification is not None:
                if participant.accepted_delay_notification_digest == digest:
                    return
                if not (active_preparation or active_round):
                    process.failure = "delay callback arrived outside the expected stage"
                    process.condition.notify_all()
                    raise ValueError(process.failure)
                if (
                    process.hierarchy_plan_id
                    and participant.accepted_delay_notification_digest
                ):
                    participant.round_failure = "conflicting duplicate delay callback"
                    participant.round_complete = True
                    participant.accepted_notification_digest = digest
                    process.condition.notify_all()
                    raise ValueError(participant.round_failure)
                participant.accepted_delay_notification_digest = digest
                participant.requested_extension = (
                    notification.delay_event_notification.expected_completion_time or 0
                )
            if notification.ml_model_infos:
                if active_round:
                    if participant.notification is not None:
                        if participant.accepted_notification_digest != digest:
                            process.failure = "conflicting duplicate round callback"
                            raise ValueError(process.failure)
                        return
                    participant.notification = notification.model_copy(deep=True)
                    participant.accepted_notification_digest = digest
                    if process.hierarchy_plan_id:
                        participant.round_complete = True
                else:
                    process.failure = "model callback arrived outside the expected stage"
                    process.condition.notify_all()
                    raise ValueError(process.failure)
            if notification.termination_request:
                if not active_round:
                    process.failure = "termination callback arrived outside the expected stage"
                    process.condition.notify_all()
                    raise ValueError(process.failure)
                if process.hierarchy_plan_id:
                    if participant.notification is not None:
                        if participant.accepted_notification_digest != digest:
                            participant.round_failure = (
                                "conflicting duplicate round callback"
                            )
                            participant.round_complete = True
                            process.condition.notify_all()
                            raise ValueError(participant.round_failure)
                    else:
                        participant.notification = notification.model_copy(deep=True)
                        participant.accepted_notification_digest = digest
                    participant.round_complete = True
                else:
                    process.failure = (
                        f"participant terminated training: {notification.termination_request}"
                    )
            elif notification.delay_event_notification is None and not notification.ml_model_infos:
                failure = "notification does not contain a stage outcome"
                if process.hierarchy_plan_id and active_round:
                    participant.round_failure = failure
                    participant.round_complete = True
                    participant.accepted_notification_digest = digest
                else:
                    process.failure = failure
                process.condition.notify_all()
                raise ValueError(failure)
            process.condition.notify_all()

    def processes(self) -> tuple[FLProcess, ...]:
        with self._lock:
            return tuple(self._processes.values())

    def collect_hierarchy_preparation(
        self,
        process_id: str,
    ) -> HierarchyPreparationCollection:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or not process.hierarchy_plan_id:
            raise KeyError(process_id)
        deadline = time.monotonic() + self._server_settings.preparation_timeout_seconds
        while True:
            extension_participant: FLParticipant | None = None
            extension_seconds = 0
            with process.condition:
                if self._closing.is_set():
                    raise RuntimeError("FL Server is shutting down")
                if process.failure:
                    raise RuntimeError(process.failure)
                if all(item.preparation_complete for item in process.participants):
                    break
                for participant in process.participants:
                    if not participant.requested_extension:
                        continue
                    remaining_budget = (
                        self._server_settings.delay_policy.max_extension_seconds
                        - participant.granted_extension_seconds
                    )
                    if (
                        participant.delay_extensions
                        >= self._server_settings.delay_policy.max_extensions
                        or remaining_budget <= 0
                    ):
                        participant.preparation_failure = (
                            "participant delay extension budget is exhausted"
                        )
                        participant.preparation_complete = True
                        participant.requested_extension = 0
                        continue
                    extension_participant = participant
                    extension_seconds = min(
                        participant.requested_extension,
                        self._server_settings.preparation_timeout_seconds,
                        remaining_budget,
                    )
                    participant.requested_extension = 0
                    break
                if extension_participant is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    process.condition.wait(timeout=remaining)
                    continue
            try:
                self._grant_extension(extension_participant, extension_seconds)
            except Exception as error:
                with process.condition:
                    extension_participant.preparation_failure = str(error)
                    extension_participant.preparation_complete = True
                    process.condition.notify_all()
            else:
                with process.condition:
                    extension_participant.delay_extensions += 1
                    extension_participant.granted_extension_seconds += extension_seconds
                    deadline = max(deadline, time.monotonic() + extension_seconds)
                    process.condition.notify_all()

        with process.condition:
            process.state = FLServerState.PREPARATION_EVALUATING
            participants = tuple(
                HierarchyParticipantPreparationOutcome(
                    participant_nf_instance_id=item.candidate.target.nf_instance_id,
                    resource_location=item.resource_location,
                    assignment_url=str(
                        item.preparation_notification.ml_model_infos[0]
                        .model_file_address.model_url
                    )
                    if (
                        item.preparation_notification is not None
                        and item.preparation_notification.ml_model_infos
                        and item.preparation_notification.ml_model_infos[0].model_file_address
                        is not None
                    )
                    else "",
                    notification=(
                        item.preparation_notification.model_copy(deep=True)
                        if item.preparation_notification is not None
                        else None
                    ),
                    failure=item.preparation_failure,
                    delay_extensions=item.delay_extensions,
                    granted_extension_seconds=item.granted_extension_seconds,
                )
                for item in process.participants
            )
            timed_out = tuple(
                item.participant_nf_instance_id
                for item in participants
                if item.notification is None and not item.failure
            )
        return HierarchyPreparationCollection(
            process_id=process.process_id,
            plan_id=process.hierarchy_plan_id,
            participants=participants,
            timed_out_participant_nf_instance_ids=timed_out,
        )

    def admit_hierarchy_preparation(self, process_id: str) -> None:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or not process.hierarchy_plan_id:
            raise KeyError(process_id)
        with process.condition:
            if process.state is not FLServerState.PREPARATION_EVALUATING:
                raise RuntimeError(
                    "hierarchy Server process is not awaiting preparation admission"
                )
            process.state = FLServerState.READY
            process.condition.notify_all()

    def execute_hierarchy_round(
        self,
        *,
        process_id: str,
        round_indicator: int,
        round_input_url: str,
        round_input_artifact: FLWorkspaceArtifact | None = None,
        expected_result_type: RoundLocalResultType,
        expected_subordinates: dict[str, tuple[str, ...]] | None = None,
        timeout_seconds: int | None = None,
        state_observer: Callable[[FLServerState], None] | None = None,
    ) -> FLWorkspaceArtifact:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or not process.hierarchy_plan_id:
            raise KeyError(process_id)
        if process.state is not FLServerState.READY:
            raise RuntimeError("hierarchy Server process is not ready for a round")
        if round_indicator < 0:
            raise ValueError("hierarchy round indicator must be non-negative")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("hierarchy round timeout must be positive")
        timeout = min(
            timeout_seconds or self._server_settings.round_timeout_seconds,
            self._server_settings.round_timeout_seconds,
        )
        try:
            process.state = FLServerState.ROUND_DISPATCH
            if state_observer is not None:
                state_observer(process.state)
            for participant in process.participants:
                self._raise_if_failed(process)
                participant.expected_round = round_indicator
                participant.notification = None
                participant.round_complete = False
                participant.round_failure = ""
                participant.accepted_notification_digest = ""
                participant.accepted_delay_notification_digest = ""
                participant.delay_extensions = 0
                participant.requested_extension = 0
                participant.granted_extension_seconds = 0
                self._patch_round(
                    process,
                    participant,
                    round_indicator,
                    round_input_url,
                    timeout_seconds=timeout,
                )
            self._raise_if_failed(process)
            process.state = FLServerState.ROUND_WAITING
            if state_observer is not None:
                state_observer(process.state)
            try:
                self._wait(
                    process,
                    lambda: all(item.round_complete for item in process.participants),
                    timeout,
                    collect_participant_failures=True,
                )
            except RuntimeError as error:
                if str(error) == "federated stage deadline expired":
                    for participant in process.participants:
                        if not participant.round_complete:
                            participant.round_failure = "round deadline expired"
                            participant.round_complete = True
                raise
            process.state = FLServerState.ROUND_EVALUATING
            if state_observer is not None:
                state_observer(process.state)
            failed = [
                item.candidate.target.nf_instance_id
                for item in process.participants
                if item.round_failure
                or (
                    item.notification is not None
                    and item.notification.termination_request is not None
                )
            ]
            if failed:
                raise RuntimeError(
                    "required hierarchy participants terminated: " + ",".join(failed)
                )
            process.state = FLServerState.AGGREGATING
            if state_observer is not None:
                state_observer(process.state)
            if round_input_artifact is None:
                raise RuntimeError("Server aggregation requires its owned ROUND_INPUT artifact")
            result = self._aggregate_round(
                process,
                round_input_url,
                round_indicator,
                round_input_artifact=round_input_artifact,
                expected_result_type=expected_result_type,
                expected_subordinates=expected_subordinates,
            )
            process.current_global_url = result.url
            process.state = FLServerState.READY
            return result
        except Exception as error:
            with process.condition:
                process.state = FLServerState.FAILED
                if not process.failure:
                    process.failure = str(error)
                process.condition.notify_all()
            self.cancel_hierarchy_preparation(process.process_id, str(error))
            raise

    def execute_hierarchy_validation(
        self,
        *,
        process_id: str,
        validation_round: int,
        candidate: FLWorkspaceArtifact,
        base_artifact: ArtifactMetadata,
        expected_candidate_process_id: str,
        expected_candidate_round: int,
        expected_subordinates: dict[str, tuple[str, ...]] | None = None,
        timeout_seconds: int | None = None,
        state_observer: Callable[[FLServerState], None] | None = None,
    ) -> HierarchyValidationCollection:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or not process.hierarchy_plan_id:
            raise KeyError(process_id)
        if process.state is not FLServerState.READY:
            raise RuntimeError("hierarchy Server process is not ready for final validation")
        if validation_round < 0 or expected_candidate_round < 0:
            raise ValueError("hierarchy validation rounds must be non-negative")
        if not expected_candidate_process_id:
            raise ValueError("hierarchy validation requires the candidate process identity")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("hierarchy validation timeout must be positive")
        timeout = min(
            timeout_seconds or self._server_settings.round_timeout_seconds,
            self._server_settings.round_timeout_seconds,
        )
        try:
            self._validate_hierarchy_candidate(
                candidate=candidate,
                base_artifact=base_artifact,
                expected_candidate_process_id=expected_candidate_process_id,
                expected_candidate_round=expected_candidate_round,
            )
            self._set_hierarchy_state(
                process,
                FLServerState.FINAL_VALIDATION_DISPATCH,
                state_observer,
            )
            process.candidate_url = candidate.url
            for participant in process.participants:
                self._raise_if_failed(process)
                participant.expected_round = validation_round
                participant.notification = None
                participant.round_complete = False
                participant.round_failure = ""
                participant.accepted_notification_digest = ""
                participant.accepted_delay_notification_digest = ""
                participant.delay_extensions = 0
                participant.requested_extension = 0
                participant.granted_extension_seconds = 0
                self._patch_validation(
                    process,
                    participant,
                    validation_round,
                    candidate.url,
                    timeout_seconds=timeout,
                )
            self._raise_if_failed(process)
            self._set_hierarchy_state(
                process,
                FLServerState.FINAL_VALIDATION_WAITING,
                state_observer,
            )
            try:
                self._wait(
                    process,
                    lambda: all(item.round_complete for item in process.participants),
                    timeout,
                    collect_participant_failures=True,
                )
            except RuntimeError as error:
                if str(error) == "federated stage deadline expired":
                    timed_out = []
                    for participant in process.participants:
                        if not participant.round_complete:
                            participant.round_failure = "validation deadline expired"
                            participant.round_complete = True
                            timed_out.append(
                                participant.candidate.target.nf_instance_id
                            )
                    raise RuntimeError(
                        "hierarchy validation deadline expired for participants: "
                        + ",".join(timed_out)
                    ) from error
                raise
            failed = [
                item.candidate.target.nf_instance_id
                for item in process.participants
                if item.round_failure
                or (
                    item.notification is not None
                    and item.notification.termination_request is not None
                )
            ]
            if failed:
                raise RuntimeError(
                    "required hierarchy validation participants terminated: "
                    + ",".join(failed)
                )
            self._set_hierarchy_state(
                process,
                FLServerState.FINAL_VALIDATION_EVALUATING,
                state_observer,
            )
            return self._collect_hierarchy_validation(
                process=process,
                candidate=candidate,
                base_artifact=base_artifact,
                validation_round=validation_round,
                expected_candidate_process_id=expected_candidate_process_id,
                expected_candidate_round=expected_candidate_round,
                expected_subordinates=expected_subordinates,
            )
        except Exception as error:
            with process.condition:
                process.state = FLServerState.FAILED
                if not process.failure:
                    process.failure = str(error)
                process.condition.notify_all()
            self.cancel_hierarchy_preparation(process.process_id, str(error))
            raise

    def finalize_hierarchy_candidate(
        self,
        *,
        process_id: str,
        validation_round: int,
        candidate: FLWorkspaceArtifact,
        base_artifact: ArtifactMetadata,
        expected_subordinates: dict[str, tuple[str, ...]],
        state_observer: Callable[[FLServerState], None] | None = None,
    ) -> HierarchyFinalizationResult:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None or not process.hierarchy_plan_id:
            raise KeyError(process_id)
        if process.hierarchy_family_key is None:
            raise RuntimeError("Root hierarchy process has no model family")
        if self._publication is None:
            raise RuntimeError("Root hierarchy publication owner is unavailable")
        process.hierarchy_state_observer = state_observer
        collection = self.execute_hierarchy_validation(
            process_id=process_id,
            validation_round=validation_round,
            candidate=candidate,
            base_artifact=base_artifact,
            expected_candidate_process_id=process_id,
            expected_candidate_round=validation_round - 1,
            expected_subordinates=expected_subordinates,
            state_observer=state_observer,
        )
        hierarchy_validation = HierarchyValidation(
            plan_id=process.hierarchy_plan_id,
            branches=collection.hierarchy_branches,
        )
        process.candidate_artifact = collection.candidate_artifact
        process.validation_summaries = collection.validation_summaries
        process.hierarchy_validation = hierarchy_validation
        leaf_summaries = tuple(
            summary
            for branch in hierarchy_validation.branches
            for summary in branch.subordinate_validation_summaries
        )
        self._evaluate_hierarchy_gate(process, leaf_summaries)
        if (
            self._server_settings.final_validation.enforce_performance_gate
            and not process.gate_would_accept
        ):
            self._set_hierarchy_state(
                process,
                FLServerState.VALIDATION_REJECTED,
                state_observer,
            )
            return HierarchyFinalizationResult(
                state=process.state,
                candidate_digest=candidate.digest,
                published_model_id=None,
                gate_would_accept=False,
                gate_rejection_reasons=process.gate_rejection_reasons,
            )

        self._set_hierarchy_state(
            process,
            FLServerState.CANDIDATE_READY,
            state_observer,
        )
        self._set_hierarchy_state(
            process,
            FLServerState.PUBLISHING,
            state_observer,
        )
        current = self._catalog.current(process.hierarchy_family_key)
        if current is None or current.artifact.key != base_artifact.key:
            raise RuntimeError("catalog base changed before hierarchy publication")
        published = self._publication.publish(
            ValidatedCandidate(
                process_id=process.process_id,
                family_key=process.hierarchy_family_key,
                base_artifact=base_artifact,
                candidate_artifact=collection.candidate_artifact,
                participants=tuple(
                    ParticipantSampleCount(
                        participantNfInstanceId=(
                            participant.candidate.target.nf_instance_id
                        ),
                        sampleCount=participant.training_sample_count,
                    )
                    for participant in sorted(
                        process.participants,
                        key=lambda item: item.candidate.target.nf_instance_id,
                    )
                ),
                validation_summaries=collection.validation_summaries,
                required_scope_keys=tuple(
                    item.scope_key for item in process.hierarchy_active_scopes
                ),
                gate_would_accept=bool(process.gate_would_accept),
                gate_rejection_reasons=process.gate_rejection_reasons,
                hierarchy_validation=hierarchy_validation,
            )
        )
        process.published_model_id = published.model_id
        required_scope_keys = tuple(
            item.scope_key for item in process.hierarchy_active_scopes
        )
        self._policy.begin_generation(
            process.hierarchy_family_key,
            self._catalog.version_key_for_id(current.model_id),
            published.version_key,
            required_scope_keys,
        )
        if self._provision_notifications is not None:
            self._provision_notifications.reconcile_family(process.hierarchy_family_key)
        terminal_state = (
            FLServerState.CUTOVER_PENDING
            if required_scope_keys
            else FLServerState.COMPLETE
        )
        self._set_hierarchy_state(process, terminal_state, state_observer)
        if terminal_state is FLServerState.COMPLETE:
            self._policy.complete_retrain(process.hierarchy_family_key)
        return HierarchyFinalizationResult(
            state=process.state,
            candidate_digest=candidate.digest,
            published_model_id=published.model_id,
            gate_would_accept=bool(process.gate_would_accept),
            gate_rejection_reasons=process.gate_rejection_reasons,
        )

    def mark_scope_adopted(
        self,
        family_key: tuple[str, str],
        model_id: int,
        scope_key: str,
    ) -> bool:
        if self._publication is None:
            return False
        completed = self._publication.mark_scope_adopted(family_key, model_id, scope_key)
        if not completed:
            return False
        with self._lock:
            process = next(
                (
                    item
                    for item in self._processes.values()
                    if item.published_model_id == model_id
                    and (
                        _flat_family_key(item) == family_key
                        or item.hierarchy_family_key == family_key
                    )
                ),
                None,
            )
        if process is not None:
            if process.hierarchy_plan_id:
                process.state = FLServerState.COMPLETE
                observer = process.hierarchy_state_observer
                if observer is not None:
                    try:
                        observer(process.state)
                    except Exception:
                        logger.exception(
                            "Failed to project hierarchy cutover completion process_id=%s",
                            process.process_id,
                        )
            else:
                process.state = FLServerState.COMPLETE
                self._finish_experiment(process)
        self._policy.complete_retrain(family_key)
        logger.info(
            "Federated model cutover complete model_id=%s family=%s",
            model_id,
            family_key,
        )
        return True

    def _run(self, process: FLProcess) -> None:
        execution = _flat_execution(process)
        scopes = execution.participant_selection.participants
        try:
            logger.info(
                "Federated process started process_id=%s scopes=%s",
                process.process_id,
                [scope.scope_key for scope in scopes],
            )
            process.state = FLServerState.DISCOVERING
            current = self._catalog.current(execution.model_family_id)
            if current is None:
                raise RuntimeError("FL base model is no longer current")
            model_interoperability = current.descriptor.model_interoperability
            if not model_interoperability:
                raise RuntimeError("FL base model has no model interoperability identifier")
            process.base_artifact_key = current.artifact.key
            discovered: list[FLClientCandidate] = []
            for scope in sorted(scopes, key=lambda item: item.scope_key):
                for candidate in self._resolver.discover(scope, model_interoperability):
                    discovered.append(candidate)
            candidates = tuple(
                sorted(
                    discovered,
                    key=lambda item: (
                        item.target.nf_instance_id,
                        item.target.nf_service_instance_id,
                        item.target.api_root,
                    ),
                )
            )
            assignments = _assign(scopes, candidates)
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
                self._server_settings.preparation_timeout_seconds,
            )
            logger.info(
                "Federated preparation complete process_id=%s participants=%s",
                process.process_id,
                [item.candidate.target.nf_instance_id for item in process.participants],
            )
            process.state = FLServerState.READY
            current = self._catalog.current(execution.model_family_id)
            if current is None or current.artifact.key != process.base_artifact_key:
                raise RuntimeError("FL base model changed while participants were preparing")
            process.current_global_artifact = current.artifact
            process.current_global_url = current.artifact.url
            for round_indicator in range(self._server_settings.round_count):
                process.current_round = round_indicator
                source_artifact = process.current_global_artifact
                if source_artifact is None:
                    raise RuntimeError("FL Server has no current global artifact")
                if source_artifact.url != process.current_global_url:
                    raise RuntimeError("FL Server global artifact URL does not match state")
                source_bundle = self._loader.load(source_artifact)
                round_input = self._workspace.publish_round_input(
                    process_id=process.process_id,
                    server_nf_instance_id=self._server_id(),
                    round_indicator=round_indicator,
                    base=source_bundle,
                    epochs=self._server_settings.client_training.epochs,
                    owner_plan_id=process.hierarchy_plan_id or None,
                )
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
                        round_input.url,
                    )
                process.state = FLServerState.ROUND_WAITING
                self._wait(
                    process,
                    lambda: all(item.notification is not None for item in process.participants),
                    self._server_settings.round_timeout_seconds,
                )
                process.state = FLServerState.AGGREGATING
                aggregate = self._aggregate_round(
                    process,
                    round_input.url,
                    round_indicator,
                    round_input_artifact=round_input,
                )
                process.current_global_artifact = _workspace_artifact_metadata(aggregate)
                process.current_global_url = process.current_global_artifact.url
                process.completed_rounds = round_indicator + 1
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
                candidate_artifact = process.current_global_artifact
                if candidate_artifact is None:
                    raise RuntimeError("FL Server has no final candidate artifact")
                if candidate_artifact.url != process.current_global_url:
                    raise RuntimeError("FL Server candidate artifact URL does not match state")
                process.candidate_url = candidate_artifact.url
            process.state = FLServerState.FINAL_VALIDATION_DISPATCH
            validation_round = self._server_settings.round_count
            process.current_round = validation_round
            for participant in process.participants:
                participant.expected_round = validation_round
                participant.notification = None
                participant.accepted_notification_digest = ""
                participant.accepted_delay_notification_digest = ""
                participant.delay_extensions = 0
                participant.requested_extension = 0
                participant.granted_extension_seconds = 0
                self._patch_validation(
                    process,
                    participant,
                    validation_round,
                    process.candidate_url,
                )
            process.state = FLServerState.FINAL_VALIDATION_WAITING
            self._wait(
                process,
                lambda: all(item.notification is not None for item in process.participants),
                self._server_settings.round_timeout_seconds,
            )
            process.state = FLServerState.FINAL_VALIDATION_EVALUATING
            self._evaluate_final_validation(
                process,
                current.artifact,
                candidate_artifact,
                validation_round,
            )
            if (
                self._server_settings.final_validation.enforce_performance_gate
                and not process.gate_would_accept
            ):
                process.state = FLServerState.VALIDATION_REJECTED
                logger.warning(
                    "Federated candidate rejected process_id=%s reasons=%s",
                    process.process_id,
                    process.gate_rejection_reasons,
                )
            else:
                process.state = FLServerState.CANDIDATE_READY
                if self._publication is not None:
                    if process.candidate_artifact is None:
                        raise RuntimeError("validated candidate artifact was not retained")
                    process.state = FLServerState.PUBLISHING
                    current_model = self._publication.publish(
                        ValidatedCandidate(
                            process_id=process.process_id,
                            family_key=execution.model_family_id,
                            base_artifact=current.artifact,
                            candidate_artifact=process.candidate_artifact,
                            participants=tuple(
                                ParticipantSampleCount(
                                    participantNfInstanceId=(
                                        participant.candidate.target.nf_instance_id
                                    ),
                                    sampleCount=participant.training_sample_count,
                                )
                                for participant in sorted(
                                    process.participants,
                                    key=lambda item: item.candidate.target.nf_instance_id,
                                )
                            ),
                            validation_summaries=process.validation_summaries,
                            required_scope_keys=execution.required_cutover_scope_keys,
                            gate_would_accept=bool(process.gate_would_accept),
                            gate_rejection_reasons=process.gate_rejection_reasons,
                        )
                    )
                    process.published_model_id = current_model.model_id
                    self._policy.begin_generation(
                        execution.model_family_id,
                        self._catalog.version_key_for_id(current.model_id),
                        current_model.version_key,
                        execution.required_cutover_scope_keys,
                    )
                    if self._provision_notifications is not None:
                        self._provision_notifications.reconcile_family(
                            execution.model_family_id
                        )
                    process.state = (
                        FLServerState.CUTOVER_PENDING
                        if execution.required_cutover_scope_keys
                        else FLServerState.COMPLETE
                    )
            logger.info(
                "Federated final validation complete process_id=%s state=%s artifact=%s",
                process.process_id,
                process.state,
                process.candidate_url,
            )
        except Exception as error:
            with process.condition:
                process.state = FLServerState.FAILED
                process.failure = str(error)
                process.condition.notify_all()
            logger.exception("Federated process failed process_id=%s", process.process_id)
        finally:
            release_experiment = process.state is not FLServerState.CUTOVER_PENDING
            if release_experiment:
                self._begin_experiment_cleanup(process)
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
            if process.state is not FLServerState.CUTOVER_PENDING:
                self._policy.complete_retrain(execution.model_family_id)
            if release_experiment:
                self._release_experiment(process)

    def _begin_experiment_cleanup(self, process: FLProcess) -> None:
        reservation_id = process.experiment_reservation_id
        if not reservation_id:
            return
        active = self._experiments.active()
        if active is None or active.reservation_id != reservation_id:
            return
        if active.lifecycle in {ExperimentLifecycle.PROVISIONAL, ExperimentLifecycle.ACTIVE}:
            active = self._experiments.mark_terminal(reservation_id, process.state.value)
        if active.lifecycle is ExperimentLifecycle.TERMINAL:
            self._experiments.begin_cleanup(reservation_id)

    def _release_experiment(self, process: FLProcess) -> None:
        active = self._experiments.active()
        if (
            process.experiment_reservation_id
            and active is not None
            and active.reservation_id == process.experiment_reservation_id
            and active.lifecycle is ExperimentLifecycle.CLEANING
        ):
            self._experiments.release(process.experiment_reservation_id)

    def _finish_experiment(self, process: FLProcess) -> None:
        self._begin_experiment_cleanup(process)
        self._release_experiment(process)

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
            notifUri=self._server_settings.callback_uri,
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
                                startTime=now
                                - timedelta(
                                    seconds=self._server_settings.preparation_data_window_seconds
                                ),
                                stopTime=now,
                            )
                        ],
                    ),
                    timeAvReq=f"PT{self._server_settings.preparation_timeout_seconds}S",
                )
            ],
            mLTrainRepInfo=MLTrainReportInfo(
                maxResTime=self._server_settings.preparation_timeout_seconds
            ),
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
        logger.info(
            "FL participant resource created process_id=%s nf=%s location=%s",
            process.process_id,
            participant.candidate.target.nf_instance_id,
            participant.resource_location,
        )
        participant.expected_scope_digest = TrainingScopeDescriptor.from_training_request(
            value, 0
        ).scope_digest
        with process.condition:
            aborted = bool(process.failure)
        if aborted:
            self._cleanup_participant(process, participant)
            participant.resource_location = ""
            raise RuntimeError(process.failure)

    def _patch_round(
        self,
        process: FLProcess,
        participant: FLParticipant,
        round_indicator: int,
        artifact_url: str,
        *,
        timeout_seconds: int | None = None,
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
            mLTrainRepInfo=MLTrainReportInfo(
                maxResTime=timeout_seconds or self._server_settings.round_timeout_seconds
            ),
        )
        response = self._client.patch(
            participant.resource_location,
            headers={"Content-Type": "application/merge-patch+json"},
            content=patch.model_dump_json(by_alias=True, exclude_none=True),
        )
        if response.status_code not in {200, 204}:
            raise RuntimeError(f"participant round patch failed with {response.status_code}")

    def _patch_validation(
        self,
        process: FLProcess,
        participant: FLParticipant,
        round_indicator: int,
        artifact_url: str,
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        patch = NwdafMLModelTrainSubscPatch(
            mLAccChkFlg=True,
            skipFlInd=True,
            roundInd=round_indicator,
            mLModelInfos=[
                MLEventNotification(
                    event=participant.scope.ml_event,
                    mLFileAddr=MLModelAddress(mLModelUrl=artifact_url),
                )
            ],
            mLTrainRepInfo=MLTrainReportInfo(
                maxResTime=timeout_seconds or self._server_settings.round_timeout_seconds
            ),
        )
        response = self._client.patch(
            participant.resource_location,
            headers={"Content-Type": "application/merge-patch+json"},
            content=patch.model_dump_json(by_alias=True, exclude_none=True),
        )
        if response.status_code not in {200, 204}:
            raise RuntimeError(
                f"participant final validation patch failed with {response.status_code}"
            )

    def _wait(
        self,
        process: FLProcess,
        predicate,
        timeout: int,
        *,
        collect_participant_failures: bool = False,
    ) -> None:
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
                        if (
                            participant.delay_extensions
                            >= self._server_settings.delay_policy.max_extensions
                        ):
                            if collect_participant_failures:
                                participant.round_failure = (
                                    "participant exceeded delay extension limit"
                                )
                                participant.round_complete = True
                                participant.requested_extension = 0
                                continue
                            raise RuntimeError("participant exceeded delay extension limit")
                        remaining_budget = (
                            self._server_settings.delay_policy.max_extension_seconds
                            - participant.granted_extension_seconds
                        )
                        extension = min(
                            participant.requested_extension,
                            timeout,
                            remaining_budget,
                        )
                        if extension <= 0:
                            if collect_participant_failures:
                                participant.round_failure = (
                                    "participant delay extension budget is exhausted"
                                )
                                participant.round_complete = True
                                participant.requested_extension = 0
                                continue
                            raise RuntimeError("participant delay extension budget is exhausted")
                        try:
                            self._grant_extension(participant, extension)
                        except Exception as error:
                            if collect_participant_failures:
                                participant.round_failure = (
                                    f"participant delay extension failed: {error}"
                                )
                                participant.round_complete = True
                                participant.requested_extension = 0
                                continue
                            raise
                        participant.delay_extensions += 1
                        participant.granted_extension_seconds += extension
                        participant.requested_extension = 0
                        deadline = max(deadline, time.monotonic() + extension)
                if predicate():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("federated stage deadline expired")
                process.condition.wait(timeout=remaining)

    @staticmethod
    def _raise_if_failed(process: FLProcess) -> None:
        with process.condition:
            if process.failure:
                raise RuntimeError(process.failure)

    def _ensure_process_generation(self, process: FLProcess) -> None:
        with self._lock:
            current = process.generation == self._generation
        if not current:
            raise RuntimeError("containing NWDAF process generation changed")
        self._raise_if_failed(process)

    @staticmethod
    def _set_hierarchy_state(
        process: FLProcess,
        state: FLServerState,
        observer: Callable[[FLServerState], None] | None,
    ) -> None:
        process.state = state
        if observer is not None:
            observer(state)

    def _cleanup_participant(
        self,
        process: FLProcess,
        participant: FLParticipant,
    ) -> str:
        last_error = ""
        for attempt in range(self._server_settings.cleanup.max_attempts):
            try:
                response = self._client.delete(participant.resource_location)
                if response.status_code in {204, 404}:
                    logger.info(
                        "FL participant resource deleted process_id=%s nf=%s location=%s status=%s",
                        process.process_id,
                        participant.candidate.target.nf_instance_id,
                        participant.resource_location,
                        response.status_code,
                    )
                    return ""
                last_error = f"cleanup returned {response.status_code}"
            except httpx.TransportError as error:
                last_error = str(error)
            if attempt + 1 < self._server_settings.cleanup.max_attempts:
                time.sleep(self._server_settings.cleanup.retry_backoff_seconds)
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

    def _cleanup_hierarchy_process(self, process: FLProcess) -> None:
        if process.hierarchy_cleanup_complete:
            return
        for participant in process.participants:
            if not participant.resource_location:
                continue
            failure = self._cleanup_participant(process, participant)
            if failure:
                process.cleanup_failure = f"{process.cleanup_failure}; {failure}".strip("; ")
        with self._lock:
            for participant in process.participants:
                self._correlations.pop(participant.notification_correlation_id, None)
        process.hierarchy_cleanup_complete = True

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
        round_input_url: str,
        round_indicator: int,
        *,
        round_input_artifact: FLWorkspaceArtifact,
        expected_result_type: RoundLocalResultType = RoundLocalResultType.TRAINING,
        expected_subordinates: dict[str, tuple[str, ...]] | None = None,
    ) -> FLWorkspaceArtifact:
        if round_input_artifact.url != round_input_url:
            raise RuntimeError("Server aggregation input URL does not match artifact")
        base_artifact = _workspace_artifact_metadata(round_input_artifact)
        base = self._loader.load(base_artifact)
        input_contract = validate_fl_artifact(_artifact_projection(base.manifest))
        if not isinstance(input_contract, RoundInputArtifact):
            raise RuntimeError("Server aggregation input is not a ROUND_INPUT artifact")
        base_digest = weights_digest(base.model)
        if (
            input_contract.fl_metadata.ml_corre_id != process.process_id
            or input_contract.fl_metadata.round_ind != round_indicator
            or input_contract.fl_metadata.weights_digest != base_digest
            or input_contract.fl_metadata.model_contract_digest
            != model_contract_digest(base.manifest)
            or input_contract.fl_metadata.preprocessing_contract_digest
            != preprocessing_contract_digest(base.manifest)
        ):
            raise RuntimeError("Server aggregation input identity does not match round")
        expected_model_contract = model_contract_digest(base.manifest)
        expected_preprocessing_contract = preprocessing_contract_digest(base.manifest)
        local_bundles: list[tuple[LoadedBundle, int]] = []
        participant_metadata = []
        for participant in process.participants:
            notification = participant.notification
            model_infos = notification.ml_model_infos if notification is not None else None
            if (
                notification is None
                or model_infos is None
                or len(model_infos) != 1
                or model_infos[0].event != participant.scope.ml_event
            ):
                raise RuntimeError("participant local result notification is missing or invalid")
            address = model_infos[0].model_file_address
            if address is None or address.model_url is None:
                raise RuntimeError("participant local result has no model URL")
            artifact = self._workspace.download(
                str(address.model_url),
                process.process_id,
                f"round-{round_indicator}-{participant.candidate.target.nf_instance_id}",
                owner_plan_id=process.hierarchy_plan_id or None,
            )
            bundle = self._loader.load(artifact)
            projection = _artifact_projection(bundle.manifest)
            contract = validate_fl_artifact(projection)
            if not isinstance(contract, RoundLocalArtifact):
                raise RuntimeError("participant returned a non-local FL artifact")
            if contract.result_type is not expected_result_type:
                raise RuntimeError("participant returned an unexpected local artifact type")
            metadata = contract.fl_metadata
            if expected_result_type is RoundLocalResultType.HIERARCHY_AGGREGATE:
                if not isinstance(metadata, RoundLocalHierarchyAggregateMetadata):
                    raise RuntimeError("Branch result lacks hierarchy aggregate metadata")
                expected = (expected_subordinates or {}).get(
                    participant.candidate.target.nf_instance_id
                )
                actual = tuple(
                    item.participant_nf_instance_id
                    for item in metadata.subordinate_participants
                )
                if expected is None or actual != expected:
                    raise RuntimeError(
                        "Branch hierarchy aggregate subordinate set does not match admission"
                    )
            if (
                metadata.ml_corre_id != process.process_id
                or metadata.round_ind != round_indicator
                or metadata.participant_nf_instance_id
                != participant.candidate.target.nf_instance_id
                or metadata.scope_digest != participant.expected_scope_digest
                or metadata.input_global_weights_digest != base_digest
                or metadata.base_weights_digest != base_digest
                or metadata.model_contract_digest != expected_model_contract
                or metadata.preprocessing_contract_digest != expected_preprocessing_contract
                or metadata.weights_digest != weights_digest(bundle.model)
            ):
                raise RuntimeError("participant local artifact identity does not match assignment")
            local_bundles.append((bundle, metadata.training_sample_count))
            participant.training_sample_count = metadata.training_sample_count
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
            owner_plan_id=process.hierarchy_plan_id or None,
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
        return published

    def _evaluate_final_validation(
        self,
        process: FLProcess,
        base_artifact: ArtifactMetadata,
        candidate_artifact: ArtifactMetadata,
        round_indicator: int,
    ) -> None:
        if candidate_artifact.url != process.candidate_url:
            raise RuntimeError("Server candidate URL does not match artifact")
        base = self._loader.load(base_artifact)
        candidate = self._loader.load(candidate_artifact)
        process.candidate_artifact = candidate_artifact
        base_digest = weights_digest(base.model)
        candidate_digest = weights_digest(candidate.model)
        expected_model_contract = model_contract_digest(base.manifest)
        expected_preprocessing_contract = preprocessing_contract_digest(base.manifest)
        if (
            model_contract_digest(candidate.manifest) != expected_model_contract
            or preprocessing_contract_digest(candidate.manifest) != expected_preprocessing_contract
        ):
            raise RuntimeError("final candidate changed the prepared model contract")

        summaries: list[tuple[FLParticipant, ValidationSummary]] = []
        for participant in process.participants:
            notification = participant.notification
            if notification is None or not notification.ml_model_infos:
                raise RuntimeError("participant final validation result is missing")
            address = notification.ml_model_infos[0].model_file_address
            if address is None or address.model_url is None:
                raise RuntimeError("participant final validation result has no model URL")
            artifact = self._workspace.download(
                str(address.model_url),
                process.process_id,
                f"validation-{participant.candidate.target.nf_instance_id}",
                owner_plan_id=process.hierarchy_plan_id or None,
            )
            bundle = self._loader.load(artifact)
            contract = validate_fl_artifact(_artifact_projection(bundle.manifest))
            if (
                not isinstance(contract, RoundLocalArtifact)
                or contract.result_type is not RoundLocalResultType.ACCURACY_CHECK
                or not isinstance(contract.fl_metadata, RoundLocalAccuracyCheckMetadata)
            ):
                raise RuntimeError("participant returned a non-validation local artifact")
            metadata = contract.fl_metadata
            evaluation = metadata.evaluation
            if (
                metadata.ml_corre_id != process.process_id
                or metadata.round_ind != round_indicator
                or metadata.participant_nf_instance_id
                != participant.candidate.target.nf_instance_id
                or metadata.scope_digest != participant.expected_scope_digest
                or metadata.input_global_weights_digest != candidate_digest
                or metadata.weights_digest != candidate_digest
                or weights_digest(bundle.model) != candidate_digest
                or metadata.model_contract_digest != expected_model_contract
                or metadata.preprocessing_contract_digest != expected_preprocessing_contract
                or evaluation.base_model_weights_digest != base_digest
                or evaluation.candidate_weights_digest != candidate_digest
            ):
                raise RuntimeError(
                    "participant final validation identity does not match assignment"
                )
            if (
                evaluation.base.absolute_actual_sum <= 0
                or evaluation.candidate.absolute_actual_sum <= 0
            ):
                raise RuntimeError("participant final validation has a zero denominator")
            summaries.append(
                (
                    participant,
                    ValidationSummary(
                        participant_nf_instance_id=metadata.participant_nf_instance_id,
                        scope_digest=metadata.scope_digest,
                        evaluation_sample_count=evaluation.evaluation_sample_count,
                        start_time=evaluation.start_time,
                        end_time=evaluation.end_time,
                        base_model_weights_digest=evaluation.base_model_weights_digest,
                        candidate_weights_digest=evaluation.candidate_weights_digest,
                        base=evaluation.base,
                        candidate=evaluation.candidate,
                    ),
                )
            )
        summaries.sort(key=lambda item: item[1].participant_nf_instance_id)
        base_error = sum(item.base.absolute_error_sum for _participant, item in summaries)
        base_actual = sum(item.base.absolute_actual_sum for _participant, item in summaries)
        candidate_error = sum(item.candidate.absolute_error_sum for _participant, item in summaries)
        candidate_actual = sum(
            item.candidate.absolute_actual_sum for _participant, item in summaries
        )
        aggregate_base = base_error / base_actual
        aggregate_candidate = candidate_error / candidate_actual
        reasons: list[str] = []
        triggering_scope_key = _triggering_scope_key(process)
        if triggering_scope_key is not None:
            triggering = next(
                (
                    summary
                    for participant, summary in summaries
                    if participant.scope.scope_key == triggering_scope_key
                ),
                None,
            )
            if triggering is None:
                raise RuntimeError("triggering scope has no final validation evidence")
            if wape(triggering.candidate) >= wape(triggering.base):
                reasons.append("triggering_scope_not_improved")
        if aggregate_candidate >= aggregate_base:
            reasons.append("aggregate_not_improved")
        for participant, summary in summaries:
            if participant.scope.scope_key == triggering_scope_key:
                continue
            regression = (wape(summary.candidate) or 0.0) - (wape(summary.base) or 0.0)
            if regression > self._server_settings.final_validation.max_scope_wape_regression:
                reasons.append(f"scope_regression_exceeded:{summary.scope_digest}")
        process.validation_summaries = tuple(item for _participant, item in summaries)
        process.gate_would_accept = not reasons
        process.gate_rejection_reasons = tuple(reasons)
        logger.info(
            "Federated final validation evaluated process_id=%s "
            "base_wape=%s candidate_wape=%s gate_would_accept=%s enforced=%s",
            process.process_id,
            aggregate_base,
            aggregate_candidate,
            process.gate_would_accept,
            self._server_settings.final_validation.enforce_performance_gate,
        )

    def _collect_hierarchy_validation(
        self,
        *,
        process: FLProcess,
        candidate: FLWorkspaceArtifact,
        base_artifact: ArtifactMetadata,
        validation_round: int,
        expected_candidate_process_id: str,
        expected_candidate_round: int,
        expected_subordinates: dict[str, tuple[str, ...]] | None,
    ) -> HierarchyValidationCollection:
        validated = self._validate_hierarchy_candidate(
            candidate=candidate,
            base_artifact=base_artifact,
            expected_candidate_process_id=expected_candidate_process_id,
            expected_candidate_round=expected_candidate_round,
        )
        candidate_artifact = validated.artifact
        base_digest = validated.base_weights_digest
        candidate_digest = validated.candidate_weights_digest
        expected_model_contract = validated.model_contract_digest
        expected_preprocessing_contract = validated.preprocessing_contract_digest

        summaries: list[ValidationSummary] = []
        hierarchy_branches: list[HierarchyBranchValidation] = []
        for participant in process.participants:
            notification = participant.notification
            model_infos = notification.ml_model_infos if notification is not None else None
            if (
                notification is None
                or model_infos is None
                or len(model_infos) != 1
                or model_infos[0].event != participant.scope.ml_event
            ):
                raise RuntimeError("participant final validation result is missing or invalid")
            address = model_infos[0].model_file_address
            if address is None or address.model_url is None:
                raise RuntimeError("participant final validation result has no model URL")
            artifact = self._workspace.download(
                str(address.model_url),
                process.process_id,
                f"validation-{participant.candidate.target.nf_instance_id}",
                owner_plan_id=process.hierarchy_plan_id or None,
            )
            bundle = self._loader.load(artifact)
            contract = validate_fl_artifact(_artifact_projection(bundle.manifest))
            if (
                not isinstance(contract, RoundLocalArtifact)
                or contract.result_type is not RoundLocalResultType.ACCURACY_CHECK
                or not isinstance(contract.fl_metadata, RoundLocalAccuracyCheckMetadata)
            ):
                raise RuntimeError("participant returned a non-validation local artifact")
            metadata = contract.fl_metadata
            evaluation = metadata.evaluation
            if (
                metadata.ml_corre_id != process.process_id
                or metadata.round_ind != validation_round
                or metadata.participant_nf_instance_id
                != participant.candidate.target.nf_instance_id
                or metadata.scope_digest != participant.expected_scope_digest
                or metadata.input_global_weights_digest != candidate_digest
                or metadata.weights_digest != candidate_digest
                or weights_digest(bundle.model) != candidate_digest
                or metadata.model_contract_digest != expected_model_contract
                or metadata.preprocessing_contract_digest
                != expected_preprocessing_contract
                or evaluation.base_model_weights_digest != base_digest
                or evaluation.candidate_weights_digest != candidate_digest
                or evaluation.base.absolute_actual_sum <= 0
                or evaluation.candidate.absolute_actual_sum <= 0
            ):
                raise RuntimeError(
                    "participant final validation identity does not match assignment"
                )
            summary = ValidationSummary(
                participant_nf_instance_id=metadata.participant_nf_instance_id,
                scope_digest=metadata.scope_digest,
                evaluation_sample_count=evaluation.evaluation_sample_count,
                start_time=evaluation.start_time,
                end_time=evaluation.end_time,
                base_model_weights_digest=evaluation.base_model_weights_digest,
                candidate_weights_digest=evaluation.candidate_weights_digest,
                base=evaluation.base,
                candidate=evaluation.candidate,
            )
            summaries.append(summary)
            if expected_subordinates is None:
                if metadata.subordinate_validation_summaries is not None:
                    raise RuntimeError(
                        "Leaf validation result must not contain subordinate evidence"
                    )
                continue
            expected = expected_subordinates.get(metadata.participant_nf_instance_id)
            subordinate = metadata.subordinate_validation_summaries
            actual = tuple(
                item.participant_nf_instance_id for item in subordinate or ()
            )
            if expected is None or subordinate is None or actual != expected:
                raise RuntimeError(
                    "Branch validation subordinate set does not match admission"
                )
            hierarchy_branches.append(
                HierarchyBranchValidation(
                    branch_nf_instance_id=metadata.participant_nf_instance_id,
                    subordinate_validation_summaries=subordinate,
                )
            )
        summaries.sort(key=lambda item: item.participant_nf_instance_id)
        hierarchy_branches.sort(key=lambda item: item.branch_nf_instance_id)
        if expected_subordinates is not None and set(expected_subordinates) != {
            item.branch_nf_instance_id for item in hierarchy_branches
        }:
            raise RuntimeError("hierarchy validation does not cover every admitted Branch")
        return HierarchyValidationCollection(
            candidate_artifact=candidate_artifact,
            validation_summaries=tuple(summaries),
            hierarchy_branches=tuple(hierarchy_branches),
        )

    def _validate_hierarchy_candidate(
        self,
        *,
        candidate: FLWorkspaceArtifact,
        base_artifact: ArtifactMetadata,
        expected_candidate_process_id: str,
        expected_candidate_round: int,
    ) -> _ValidatedHierarchyCandidate:
        if not isinstance(candidate.contract, RoundGlobalArtifact):
            raise RuntimeError("hierarchy final candidate is not a ROUND_GLOBAL artifact")
        candidate_metadata = candidate.contract.fl_metadata
        candidate_artifact = ArtifactMetadata(
            key=candidate.digest,
            size_bytes=candidate.path.stat().st_size,
            path=candidate.path,
            url=candidate.url,
        )
        base = self._loader.load(base_artifact)
        candidate_bundle = self._loader.load(candidate_artifact)
        base_digest = weights_digest(base.model)
        candidate_digest = weights_digest(candidate_bundle.model)
        expected_model_contract = model_contract_digest(base.manifest)
        expected_preprocessing_contract = preprocessing_contract_digest(base.manifest)
        if (
            candidate_metadata.ml_corre_id != expected_candidate_process_id
            or candidate_metadata.round_ind != expected_candidate_round
            or candidate_metadata.model_contract_digest != expected_model_contract
            or candidate_metadata.preprocessing_contract_digest
            != expected_preprocessing_contract
            or candidate_metadata.weights_digest != candidate_digest
            or model_contract_digest(candidate_bundle.manifest) != expected_model_contract
            or preprocessing_contract_digest(candidate_bundle.manifest)
            != expected_preprocessing_contract
        ):
            raise RuntimeError("hierarchy final candidate identity does not match the plan")
        return _ValidatedHierarchyCandidate(
            artifact=candidate_artifact,
            base_weights_digest=base_digest,
            candidate_weights_digest=candidate_digest,
            model_contract_digest=expected_model_contract,
            preprocessing_contract_digest=expected_preprocessing_contract,
        )

    def _evaluate_hierarchy_gate(
        self,
        process: FLProcess,
        summaries: tuple[ValidationSummary, ...],
    ) -> None:
        if not summaries:
            raise RuntimeError("hierarchy final validation has no Leaf evidence")
        base_error = sum(item.base.absolute_error_sum for item in summaries)
        base_actual = sum(item.base.absolute_actual_sum for item in summaries)
        candidate_error = sum(item.candidate.absolute_error_sum for item in summaries)
        candidate_actual = sum(item.candidate.absolute_actual_sum for item in summaries)
        if base_actual <= 0 or candidate_actual <= 0:
            raise RuntimeError("hierarchy final validation has a zero denominator")
        reasons: list[str] = []
        if candidate_error / candidate_actual >= base_error / base_actual:
            reasons.append("aggregate_not_improved")
        for summary in summaries:
            regression = (wape(summary.candidate) or 0.0) - (
                wape(summary.base) or 0.0
            )
            if regression > self._server_settings.final_validation.max_scope_wape_regression:
                reasons.append(f"scope_regression_exceeded:{summary.scope_digest}")
        process.gate_would_accept = not reasons
        process.gate_rejection_reasons = tuple(reasons)
        logger.info(
            "Hierarchy final validation evaluated process_id=%s "
            "base_wape=%s candidate_wape=%s gate_would_accept=%s enforced=%s",
            process.process_id,
            base_error / base_actual,
            candidate_error / candidate_actual,
            process.gate_would_accept,
            self._server_settings.final_validation.enforce_performance_gate,
        )

    def _go_base(self) -> str:
        return self._nwdaf_context.get().internal_api_root

    def _server_id(self) -> str:
        return self._nwdaf_context.get().nf_instance_id

    def _future_done(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

def _flat_execution(process: FLProcess) -> FlatExecutionRequest:
    if process.execution is not None:
        return process.execution
    if process.intent is None:
        raise RuntimeError("flat FL process requires an explicit execution request")
    return FlatExecutionRequest(
        model_family_id=process.intent.family_key,
        trigger_source=TriggerSource.DEGRADATION,
        participant_selection=MonitorParticipantSelection(
            participants=tuple(
                FlatParticipantScope.from_monitor_scope(scope)
                for scope in process.intent.active_scopes
            )
        ),
        required_cutover_scope_keys=process.intent.active_scope_keys,
        triggering_scope_key=getattr(process.intent, "triggering_scope_key", None),
    )


def _flat_family_key(process: FLProcess) -> FamilyKey | None:
    if process.execution is not None:
        return process.execution.model_family_id
    if process.intent is not None:
        return process.intent.family_key
    return None


def _triggering_scope_key(process: FLProcess) -> str | None:
    if process.execution is not None:
        return process.execution.triggering_scope_key
    if process.intent is not None:
        return getattr(process.intent, "triggering_scope_key", None)
    return None


def _assign(
    scopes: tuple[FlatParticipantScope | ScopeReference, ...],
    candidates: tuple[FLClientCandidate, ...],
) -> tuple[tuple[FlatParticipantScope | ScopeReference, FLClientCandidate], ...]:
    assignments = []
    used = set()
    for scope in sorted(scopes, key=lambda item: item.scope_key):
        owner_id = _participant_nf_instance_id(scope)
        required = _scope_tais(scope)
        eligible = tuple(
            item
            for item in candidates
            if item.target.nf_instance_id == owner_id
            and item.target.nf_instance_id not in used
            and (not required or required.intersection(item.tracking_areas))
        )
        if len(eligible) != 1:
            raise RuntimeError(
                f"configured participant {owner_id or '<missing>'} is not an eligible "
                f"unique FL Client for scope {scope.scope_key}"
            )
        candidate = eligible[0]
        assignments.append((scope, candidate))
        used.add(candidate.target.nf_instance_id)
    if len(assignments) < 2:
        raise RuntimeError("the first FL profile requires two distinct clients")
    return tuple(assignments)


def _scope_tais(scope: FlatParticipantScope | ScopeReference) -> set[str]:
    return {_tai_key(item) for item in _scope_tracking_area_values(scope) if _tai_key(item)}


def _scope_tracking_area_values(
    scope: FlatParticipantScope | ScopeReference,
) -> list[dict]:
    area = scope.ml_event_filter.get("networkArea") or scope.ml_event_filter.get("aoi") or {}
    tais = area.get("tais") if isinstance(area, dict) else []
    return [dict(item) for item in tais or [] if isinstance(item, dict) and _tai_key(item)]


def _participant_nf_instance_id(scope: FlatParticipantScope | ScopeReference) -> str:
    value = getattr(scope, "participant_nf_instance_id", None)
    if value is None:
        value = getattr(scope, "consumer_id", "")
    return str(value).strip()


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


def _workspace_artifact_metadata(artifact: FLWorkspaceArtifact) -> ArtifactMetadata:
    return ArtifactMetadata(
        key=artifact.digest,
        size_bytes=artifact.path.stat().st_size,
        path=artifact.path,
        url=artifact.url,
    )
