import copy
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

import httpx

from py_mtlf.config import (
    FederatedLearningSettings,
    FLClientSettings,
    ImageClassificationWorkloadSettings,
    LocalImageTrainingDataSettings,
    NotificationSettings,
)
from py_mtlf.core.accuracy_policy import RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.dataset import DatasetCoordinator, DatasetJob, DatasetJobState, DatasetSnapshot
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.fl_artifacts import (
    RoundGlobalArtifact,
    RoundInputArtifact,
    validate_fl_artifact_manifest,
)
from py_mtlf.core.fl_candidate_orchestration import (
    ClientLocalWork,
    IntermediateLocalWork,
    LocalContractDefaults,
    LocalExecutionRole,
    resolve_effective_node_contract,
)
from py_mtlf.core.fl_experiment import (
    ExperimentAdmissionClosedError,
    ExperimentLifecycle,
    ExperimentRegistryError,
    ExperimentRole,
    FLExperimentRegistry,
)
from py_mtlf.core.fl_round_model_distribution import RoundModelDistribution
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    validate_model_compatibility,
)
from py_mtlf.core.image_classification import ImageDatasetLoader
from py_mtlf.core.nwdaf_context import FLCapabilityType, NwdafContextClient
from py_mtlf.core.trainer import (
    LoadedBundle,
    LocalTrainer,
    TrustedBundleLoader,
    resolve_device,
    wape,
)
from py_mtlf.core.training_data import TrainingDatasetBuilder, dataset_evidence
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.core.workloads import (
    IMAGE_CLASSIFICATION_EVENT,
    ImageDatasetName,
    WorkloadProfile,
    image_training_contract,
    validate_image_manifest,
)
from py_mtlf.wire.adrf import TimeWindow as AdrfTimeWindow
from py_mtlf.wire.features import (
    HIERARCHICAL_FL_ORCHESTRATION_FEATURE,
    feature_intersection,
    feature_mask,
    includes_feature,
)
from py_mtlf.wire.ml_model import MLEventNotification, MLModelAddress
from py_mtlf.wire.ml_model_training import (
    DelayEventNotif,
    FlTopologyReport,
    InvalidParameter,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
    TrainingResourceIdentity,
    apply_subscription_patch,
    has_candidate_patch_fields,
    has_candidate_subscription_fields,
    split_candidate_operations,
    validate_candidate_subscription_receiver,
    validate_fl_patch,
    validate_fl_subscription,
)

logger = logging.getLogger(__name__)


class FLClientState(StrEnum):
    PROVISIONAL = "PROVISIONAL"
    PREPARING = "PREPARING"
    PREPARATION_RESULT_PENDING = "PREPARATION_RESULT_PENDING"
    PREPARED = "PREPARED"
    ROUND_RUNNING = "ROUND_RUNNING"
    VALIDATION_RUNNING = "VALIDATION_RUNNING"
    RESULT_PENDING = "RESULT_PENDING"
    READY = "READY"
    TERMINATING = "TERMINATING"
    FAILED = "FAILED"


class FLClientCapacityError(RuntimeError):
    pass


class BranchArtifactView(Protocol):
    url: str


class BranchPreparationDispatcher(Protocol):
    def prepare_protocol(
        self,
        *,
        representation: NwdafMLModelTrainSubsc,
        reservation_id: str,
    ) -> FlTopologyReport: ...

    def cancel(self, plan_id: str, reason: str) -> None: ...

    def execute_protocol_round(
        self,
        *,
        representation: NwdafMLModelTrainSubsc,
        upper_input: LoadedBundle,
        upper_input_artifact: ArtifactMetadata,
        upper_model: MLEventNotification,
        upper_client_subscription_id: str,
        upper_resource_revision: int,
        upper_training_scope: TrainingScopeDescriptor,
        callback_margin_seconds: int,
        local_work: IntermediateLocalWork,
    ) -> BranchArtifactView: ...

@dataclass
class FLClientResource:
    subscription_id: str
    representation: NwdafMLModelTrainSubsc
    state: FLClientState
    scope: TrainingScopeDescriptor
    dataset_snapshot: DatasetSnapshot | None = None
    dataset_job_id: str = ""
    last_error: str = ""
    revision: int = 1
    work_slot_owned: bool = True
    prepared_training_sample_count: int = 0
    preparation_base_artifact: ArtifactMetadata | None = None
    experiment_reservation_id: str = ""
    callback_slot_owned: bool = True
    candidate_contract: bool = False
    hierarchical_feature_negotiated: bool = False
    protocol_image_dataset: ImageDatasetName | None = None
    client_local_work: ClientLocalWork | None = None
    intermediate_local_work: IntermediateLocalWork | None = None

    @property
    def identity(self) -> TrainingResourceIdentity:
        method = (
            self.representation.event_request.notification_method
            if self.representation.event_request
            else None
        )
        expected_round = (
            None if self.representation.ml_preparation_flag else self.representation.round_indicator
        )
        return TrainingResourceIdentity(
            subscription_id=self.subscription_id,
            ml_correlation_id=self.representation.ml_correlation_id or "",
            notification_correlation_id=(self.representation.notification_correlation_id),
            expected_round_indicator=expected_round,
            notification_method=method,
        )


class FLClientEngine:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        client_settings: FLClientSettings,
        notification_settings: NotificationSettings,
        nwdaf_context: NwdafContextClient,
        datasets: DatasetCoordinator,
        workspace: FLWorkspace,
        client: httpx.Client | None = None,
        experiments: FLExperimentRegistry | None = None,
        branch_coordinator: BranchPreparationDispatcher | None = None,
        round_model_distribution: RoundModelDistribution | None = None,
        clock=time.monotonic,
    ) -> None:
        self._settings = settings
        self._client_settings = client_settings
        self._notification_settings = notification_settings
        self._nwdaf_context = nwdaf_context
        self._datasets = datasets
        self._workspace = workspace
        self._experiments = experiments or FLExperimentRegistry()
        self._branch_coordinator = branch_coordinator
        self._round_model_distribution = round_model_distribution
        self._trainer = FederatedTrainer(client_settings.training)
        self._device = resolve_device(client_settings.training.device)
        self._dataset_builder = TrainingDatasetBuilder(client_settings.training)
        self._image_dataset_loader = ImageDatasetLoader()
        self._loader = TrustedBundleLoader()
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._resources: dict[str, FLClientResource] = {}
        self._deleting: set[str] = set()
        self._cancelled_hierarchy_resources: dict[str, float] = {}
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=client_settings.max_concurrent_jobs,
            thread_name_prefix="fl-client",
        )
        self._outbox_executor = ThreadPoolExecutor(
            max_workers=min(2, client_settings.max_concurrent_jobs),
            thread_name_prefix="fl-client-outbox",
        )
        self._futures: set[Future] = set()
        self._outbox_futures: set[Future] = set()
        self._outbox_keys: set[tuple[str, int]] = set()
        self._abandoned_outbox_keys: set[tuple[str, int]] = set()
        self._abandoned_work_keys: set[tuple[str, int]] = set()
        self._delay_timers: dict[str, threading.Timer] = {}
        self._capacity = threading.BoundedSemaphore(client_settings.max_concurrent_jobs)
        self._outbox_capacity = threading.BoundedSemaphore(client_settings.callback_queue_size)
        self._closing = threading.Event()

    def close(self) -> None:
        self._closing.set()
        with self._lock:
            timers = tuple(self._delay_timers.values())
            self._delay_timers.clear()
        for timer in timers:
            timer.cancel()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._outbox_executor.shutdown(wait=True, cancel_futures=False)
        if self._owns_client:
            self._client.close()

    def abort_generation(self, reason: str) -> None:
        """Fence all old-Go resources while leaving the engine ready for new work."""
        active = self._experiments.active()
        plan_id = active.plan_id if active is not None else None
        release_callback_slots = 0
        release_work_slots = 0
        with self._lock:
            resources = tuple(self._resources.values())
            outbox_keys = set(self._outbox_keys)
            for resource in resources:
                key = (resource.subscription_id, resource.revision)
                if resource.work_slot_owned:
                    resource.work_slot_owned = False
                    self._abandoned_work_keys.add(key)
                    release_work_slots += 1
                if resource.callback_slot_owned:
                    if key in outbox_keys:
                        self._abandoned_outbox_keys.add(key)
                    resource.callback_slot_owned = False
                    release_callback_slots += 1
                resource.revision += 1
                self._cancel_delay(resource.subscription_id)
            self._resources.clear()
            self._deleting.clear()
            self._cancelled_hierarchy_resources.clear()
        for _ in range(release_callback_slots):
            self._outbox_capacity.release()
        for _ in range(release_work_slots):
            self._capacity.release()
        if self._branch_coordinator is not None:
            abort = getattr(self._branch_coordinator, "abort_generation", None)
            if callable(abort):
                abort(reason)
                if plan_id is not None:
                    self._branch_coordinator.cancel(plan_id, reason)
            elif plan_id is not None:
                self._branch_coordinator.cancel(plan_id, reason)
        if plan_id is not None:
            try:
                self._workspace.release_plan(plan_id)
            except RuntimeError:
                logger.exception(
                    "Failed to release FL Client workspace during generation reset plan_id=%s",
                    plan_id,
                )

    def create(self, value: NwdafMLModelTrainSubsc) -> FLClientResource:
        validate_fl_subscription(value)
        candidate_contract = has_candidate_subscription_fields(value)
        if candidate_contract:
            self._reject_retained_instruction(value)
            context = self._nwdaf_context.get()
            validate_candidate_subscription_receiver(
                value,
                context.nf_instance_id,
            )
            accepted_offer = (
                feature_mask(HIERARCHICAL_FL_ORCHESTRATION_FEATURE)
                if self._can_accept_protocol_resource(value, context)
                else ""
            )
            value = value.model_copy(
                update={
                    "supported_features": feature_intersection(
                        value.supported_features or "",
                        accepted_offer,
                    )
                },
                deep=True,
            )
        else:
            self._validate_preparation_admission(value)
        persistent, _operation = split_candidate_operations(value)
        if not self._capacity.acquire(blocking=False):
            raise FLClientCapacityError("FL client work capacity is exhausted")
        if not self._outbox_capacity.acquire(blocking=False):
            self._capacity.release()
            raise FLClientCapacityError("FL client callback outbox is full")
        resource_id = ""
        reservation_id = ""
        try:
            resource_id = str(uuid4())
            try:
                reservation = self._experiments.reserve_client(
                    resource_id,
                    value.ml_correlation_id or "",
                )
            except ExperimentAdmissionClosedError as error:
                raise RuntimeError(str(error)) from error
            except ExperimentRegistryError as error:
                raise FLClientCapacityError(str(error)) from error
            reservation_id = reservation.reservation_id
            resource = FLClientResource(
                subscription_id=resource_id,
                representation=persistent,
                state=FLClientState.PROVISIONAL,
                scope=TrainingScopeDescriptor.from_training_request(persistent, 0),
                experiment_reservation_id=reservation_id,
                candidate_contract=candidate_contract,
                hierarchical_feature_negotiated=includes_feature(
                    persistent.supported_features or "",
                    HIERARCHICAL_FL_ORCHESTRATION_FEATURE,
                ),
            )
            with self._lock:
                if any(
                    item.representation.notification_correlation_id
                    == value.notification_correlation_id
                    for item in self._resources.values()
                ):
                    raise ValueError("notifCorreId must be unique")
                self._resources[resource_id] = resource
            self._start_resource_operation(
                resource,
                candidate_operation=candidate_contract,
                initial_candidate_operation=candidate_contract,
            )
            return self.get(resource_id)
        except Exception:
            with self._lock:
                if resource_id:
                    self._cancel_delay(resource_id)
                    self._resources.pop(resource_id, None)
            if reservation_id:
                self._experiments.rollback_client(reservation_id, resource_id)
            self._capacity.release()
            self._outbox_capacity.release()
            raise

    def replace(self, subscription_id: str, value: NwdafMLModelTrainSubsc) -> FLClientResource:
        validate_fl_subscription(value)
        candidate_operation = has_candidate_subscription_fields(value)
        if candidate_operation:
            self._reject_retained_instruction(value)
            with self._lock:
                resource = self._required(subscription_id)
                self._ensure_mutable(resource)
                self._require_candidate_feature(resource)
        if value.fl_topology is not None:
            validate_candidate_subscription_receiver(
                value,
                self._nwdaf_context.get().nf_instance_id,
            )
        persistent, _operation = split_candidate_operations(value)
        return self._replace_resource(
            subscription_id,
            value,
            persistent,
            candidate_operation=candidate_operation,
            candidate_contract=candidate_operation,
        )

    def _replace_resource(
        self,
        subscription_id: str,
        requested: NwdafMLModelTrainSubsc,
        persistent: NwdafMLModelTrainSubsc,
        *,
        candidate_operation: bool,
        candidate_contract: bool,
    ) -> FLClientResource:
        scope = TrainingScopeDescriptor.from_training_request(persistent, 0)
        with self._lock:
            resource = self._required(subscription_id)
            self._ensure_mutable(resource)
            validate_fl_subscription(requested, resource.identity)
            if candidate_operation:
                self._require_candidate_feature(resource)
            if self._same_representation(persistent, resource.representation):
                return copy.deepcopy(resource)
            if resource.state in {
                FLClientState.PREPARING,
                FLClientState.ROUND_RUNNING,
                FLClientState.VALIDATION_RUNNING,
                FLClientState.RESULT_PENDING,
                FLClientState.PREPARATION_RESULT_PENDING,
            }:
                raise RuntimeError("training resource has an operation in progress")
            if not self._capacity.acquire(blocking=False):
                raise FLClientCapacityError("FL client work capacity is exhausted")
            if not self._outbox_capacity.acquire(blocking=False):
                self._capacity.release()
                raise FLClientCapacityError("FL client callback outbox is full")
            previous = copy.deepcopy(resource)
            try:
                resource.representation = persistent.model_copy(deep=True)
                resource.scope = scope
                resource.revision += 1
                resource.state = FLClientState.PROVISIONAL
                resource.callback_slot_owned = True
                resource.work_slot_owned = True
                resource.candidate_contract = candidate_contract
                resource.hierarchical_feature_negotiated = includes_feature(
                    persistent.supported_features or "",
                    HIERARCHICAL_FL_ORCHESTRATION_FEATURE,
                )
                self._start_resource_operation(
                    resource,
                    candidate_operation=candidate_operation,
                )
                return copy.deepcopy(resource)
            except Exception:
                self._resources[subscription_id] = previous
                self._capacity.release()
                self._outbox_capacity.release()
                raise

    def patch(self, subscription_id: str, patch: NwdafMLModelTrainSubscPatch) -> FLClientResource:
        candidate_operation = has_candidate_patch_fields(patch)
        with self._lock:
            resource = self._required(subscription_id)
            self._ensure_mutable(resource)
            validate_fl_patch(patch, resource.identity)
            if candidate_operation:
                self._require_candidate_feature(resource)
            update = patch.model_dump(
                by_alias=True,
                exclude_unset=True,
                exclude_none=False,
                mode="json",
            )
            value = apply_subscription_patch(resource.representation, patch)
            self._reject_retained_instruction(value)
            if value.fl_topology is not None:
                validate_candidate_subscription_receiver(
                    value,
                    self._nwdaf_context.get().nf_instance_id,
                )
            persistent, _operation = split_candidate_operations(value)
            if set(update) == {"mLTrainRepInfo"} and resource.state in {
                FLClientState.PREPARING,
                FLClientState.ROUND_RUNNING,
                FLClientState.VALIDATION_RUNNING,
            }:
                resource.representation = persistent
                self._schedule_delay(resource)
                return copy.deepcopy(resource)
            if self._same_representation(persistent, resource.representation):
                return copy.deepcopy(resource)
            if (
                resource.representation.round_indicator is not None
                and value.round_indicator is not None
                and value.round_indicator <= resource.representation.round_indicator
            ):
                raise RuntimeError("stale or conflicting FL round command")
            candidate_contract = resource.candidate_contract or candidate_operation
        return self._replace_resource(
            subscription_id,
            value,
            persistent,
            candidate_operation=candidate_operation,
            candidate_contract=candidate_contract,
        )

    @staticmethod
    def _reject_retained_instruction(value: NwdafMLModelTrainSubsc) -> None:
        _persistent, operation = split_candidate_operations(value)
        violations = []
        if operation.top_level_retained_result_request:
            violations.append(
                InvalidParameter(
                    "x-retainedResultReq",
                    "retained-result execution is not supported",
                )
            )
        violations.extend(
            InvalidParameter(
                "x-flTopology",
                f"retained-result execution is not supported for node {node_id}",
            )
            for node_id in operation.node_requests
        )
        if violations:
            raise RequirementsError(violations)

    def delete(self, subscription_id: str) -> None:
        with self._lock:
            self._prune_cancelled_resources_locked()
            if subscription_id in self._cancelled_hierarchy_resources:
                return
            resource = self._resources.get(subscription_id)
            if resource is not None and resource.state is FLClientState.TERMINATING:
                self._cancel_delay(subscription_id)
                self._resources.pop(subscription_id, None)
                self._deleting.discard(subscription_id)
                self._cancelled_hierarchy_resources[subscription_id] = (
                    self._clock() + self._settings.lifecycle.tombstone_ttl_seconds
                )
                return
        experiment = self._experiments.for_client_subscription(subscription_id)
        with self._lock:
            resource = self._required(subscription_id)
            hierarchy_bound = (
                experiment is not None
                and experiment.plan_id is not None
                and experiment.assigned_role in {ExperimentRole.BRANCH, ExperimentRole.LEAF}
            )
            if resource.state in {
                FLClientState.PREPARING,
                FLClientState.ROUND_RUNNING,
                FLClientState.VALIDATION_RUNNING,
                FLClientState.RESULT_PENDING,
                FLClientState.PREPARATION_RESULT_PENDING,
            } and not hierarchy_bound:
                raise RuntimeError("ML_TRAINING_NOT_COMPLETE")
            reservation_id = resource.experiment_reservation_id
            self._deleting.add(subscription_id)
            if hierarchy_bound:
                resource.revision += 1
                self._cancel_delay(subscription_id)
                release_callback_slot = (
                    resource.callback_slot_owned
                    and not any(key[0] == subscription_id for key in self._outbox_keys)
                )
                if release_callback_slot:
                    resource.callback_slot_owned = False
            else:
                release_callback_slot = False
        if hierarchy_bound:
            if release_callback_slot:
                self._outbox_capacity.release()
            try:
                current = experiment or self._experiments.active()
                plan_id = current.plan_id if current is not None else None
                if plan_id is not None and self._branch_coordinator is not None:
                    try:
                        self._branch_coordinator.cancel(
                            plan_id,
                            "parent cancelled preparation",
                        )
                    except RuntimeError:
                        logger.exception(
                            "Failed to cancel lower hierarchy resources plan_id=%s",
                            plan_id,
                        )
                    current = self._experiments.active()
                if current is not None:
                    if current.lifecycle in {
                        ExperimentLifecycle.PROVISIONAL,
                        ExperimentLifecycle.ACTIVE,
                    }:
                        current = self._experiments.mark_terminal(
                            reservation_id,
                            "CANCELLED",
                        )
                    if current.lifecycle is ExperimentLifecycle.TERMINAL:
                        current = self._experiments.begin_cleanup(reservation_id)
                    if subscription_id in current.upper_client_subscription_ids:
                        current = self._experiments.remove_client(
                            reservation_id,
                            subscription_id,
                        )
                    plan_id = current.plan_id if current is not None else plan_id
                    if plan_id is not None:
                        try:
                            self._workspace.release_plan(plan_id)
                        except RuntimeError:
                            logger.exception(
                                "Failed to release FL Client workspace plan_id=%s",
                                plan_id,
                            )
                    if (
                        current is not None
                        and not current.upper_client_subscription_ids
                        and current.server_process_id is None
                    ):
                        self._experiments.release(reservation_id)
            except Exception:
                with self._lock:
                    self._deleting.discard(subscription_id)
                raise
            with self._lock:
                self._resources.pop(subscription_id, None)
                self._deleting.discard(subscription_id)
                self._cancelled_hierarchy_resources[subscription_id] = (
                    self._clock() + self._settings.lifecycle.tombstone_ttl_seconds
                )
            return
        try:
            if reservation_id:
                self._experiments.remove_client(reservation_id, subscription_id)
        except Exception:
            with self._lock:
                self._deleting.discard(subscription_id)
            raise
        with self._lock:
            self._cancel_delay(subscription_id)
            self._resources.pop(subscription_id, None)
            self._deleting.discard(subscription_id)

    def get(self, subscription_id: str) -> FLClientResource:
        with self._lock:
            return copy.deepcopy(self._required(subscription_id))

    def _start_resource_operation(
        self,
        resource: FLClientResource,
        *,
        candidate_operation: bool = False,
        initial_candidate_operation: bool = False,
    ) -> None:
        if resource.candidate_contract and not resource.hierarchical_feature_negotiated:
            resource.state = FLClientState.READY
            resource.callback_slot_owned = False
            resource.work_slot_owned = False
            self._capacity.release()
            self._outbox_capacity.release()
            return
        if candidate_operation:
            self._start_preparation(
                resource,
                protocol_initial=initial_candidate_operation,
            )
            return
        self._start_operation(resource)

    @staticmethod
    def _require_candidate_feature(resource: FLClientResource) -> None:
        if resource.hierarchical_feature_negotiated:
            return
        raise RequirementsError(
            [
                InvalidParameter(
                    "suppFeats",
                    "HierarchicalFLOrch was not negotiated for this resource",
                )
            ]
        )

    def _can_accept_protocol_resource(self, value, context) -> bool:
        node = value.fl_topology
        if (
            value.ml_preparation_flag is not True
            or value.ml_model_infos
            or node is None
            or self._round_model_distribution is None
        ):
            return False
        event = value.ml_event_subscriptions[0]
        try:
            contract = image_training_contract(
                event.ml_event,
                event.model_interoperability,
            )
        except ValueError:
            return False
        intermediate_role = bool(node.children) or (
            node.policy is not None
            and node.policy.allow_additional_candidates is True
        )
        required_capabilities = (
            {FLCapabilityType.SERVER_AND_CLIENT}
            if intermediate_role
            else {
                FLCapabilityType.CLIENT,
                FLCapabilityType.SERVER_AND_CLIENT,
            }
        )
        if not any(
            event.ml_event in capability.ml_analytics_ids
            and capability.fl_capability_type in required_capabilities
            for capability in context.ml_analytics_capabilities
        ):
            return False
        if intermediate_role:
            if self._branch_coordinator is None or not hasattr(
                self._branch_coordinator,
                "prepare_protocol",
            ):
                return False
            try:
                resolve_effective_node_contract(
                    node,
                    role=LocalExecutionRole.INTERMEDIATE,
                    defaults=LocalContractDefaults(),
                )
            except (RequirementsError, ValueError):
                return False
            return True
        local_data = self._client_settings.training_data
        return (
            isinstance(
                self._client_settings.workload,
                ImageClassificationWorkloadSettings,
            )
            and isinstance(local_data, LocalImageTrainingDataSettings)
            and local_data.dataset == contract.name.value
            and event.model_interoperability
            in self._client_settings.model_interoperability_ids
            and node.strategy is not None
            and node.report_after is not None
            and node.report_after.unit == "epoch"
        )

    def _start_operation(self, resource: FLClientResource) -> None:
        value = resource.representation
        if value.ml_preparation_flag:
            self._start_preparation(resource)
            return
        if value.ml_accuracy_check_flag is not None or value.skip_fl_indicator is not None:
            self._start_validation(resource)
            return
        if value.ml_model_infos and value.round_indicator is not None:
            resource.state = FLClientState.ROUND_RUNNING
            self._schedule_delay(resource)
            self._submit(self._run_round, resource.subscription_id, resource.revision)
            return
        resource.state = FLClientState.READY
        resource.callback_slot_owned = False
        resource.work_slot_owned = False
        self._capacity.release()
        self._outbox_capacity.release()

    def _start_validation(self, resource: FLClientResource) -> None:
        value = resource.representation
        violations: list[InvalidParameter] = []
        if value.ml_accuracy_check_flag is not True:
            violations.append(InvalidParameter("mLAccChkFlg", "must be true for final validation"))
        if value.skip_fl_indicator is not True:
            violations.append(InvalidParameter("skipFlInd", "must be true for final validation"))
        if value.round_indicator is None:
            violations.append(InvalidParameter("roundInd", "is required for final validation"))
        if len(value.ml_model_infos or ()) != 1:
            violations.append(
                InvalidParameter(
                    "mLModelInfos",
                    "must provide exactly one final candidate for validation",
                )
            )
        if (
            resource.preparation_base_artifact is None
            or resource.dataset_snapshot is None
        ):
            violations.append(
                InvalidParameter(
                    "mLAccChkFlg",
                    "requires the frozen preparation dataset and base model",
                )
            )
        if violations:
            raise RequirementsError(violations)
        resource.state = FLClientState.VALIDATION_RUNNING
        self._schedule_delay(resource)
        self._submit(self._run_validation, resource.subscription_id, resource.revision)

    def _start_preparation(
        self,
        resource: FLClientResource,
        *,
        protocol_initial: bool = True,
    ) -> None:
        value = resource.representation
        window = _requested_window(value)
        event = value.ml_event_subscriptions[0]
        target = value.target_reporting_ue or event.target_ue
        scope = ScopeReference(
            scope_key=f"fl:{resource.subscription_id}:event:0",
            consumer_id="",
            model_ids=(),
            ml_event=event.ml_event,
            ml_event_filter=dict(event.ml_event_filter),
            target_ue=dict(target) if target is not None else None,
        )
        intent = RetrainIntent(
            family_key=(event.ml_event, event.model_interoperability),
            triggering_scope_key=scope.scope_key,
            active_scope_keys=(scope.scope_key,),
            triggering_scope=scope,
            active_scopes=(scope,),
            created_at=datetime.now(UTC),
        )
        resource.state = FLClientState.PREPARING
        self._schedule_delay(resource)
        self._submit(
            self._run_preparation,
            resource.subscription_id,
            resource.revision,
            intent,
            window,
            protocol_initial,
        )

    def _run_preparation(
        self,
        subscription_id: str,
        revision: int,
        intent: RetrainIntent,
        window: AdrfTimeWindow,
        protocol_initial: bool = True,
    ) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if resource.revision != revision:
                    self._release_work_slot(subscription_id, revision)
                    return
                value = resource.representation.model_copy(deep=True)
                candidate_contract = resource.candidate_contract
            if candidate_contract:
                self._run_protocol_preparation(
                    subscription_id,
                    revision,
                    value,
                    initial=protocol_initial,
                )
                return
            model_info = value.ml_model_infos[0]
            if model_info.model_file_address is None:
                raise RuntimeError("FL preparation base model must use mLFileAddr")
            downloaded = self._workspace.download_archive(
                str(model_info.model_file_address.model_url),
                value.ml_correlation_id or subscription_id,
                "preparation-base",
            )
            artifact = downloaded.metadata
            event = value.ml_event_subscriptions[0]
            base = self._loader.load(artifact)
            self._validate_workload_bundle(base, event.ml_event, event.model_interoperability)
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is None or current.revision != revision:
                    self._release_work_slot(subscription_id, revision)
                    return
                current.preparation_base_artifact = artifact
            if isinstance(self._client_settings.training_data, LocalImageTrainingDataSettings):
                with self._lock:
                    current = self._resources.get(subscription_id)
                    if current is None or current.revision != revision:
                        self._release_work_slot(subscription_id, revision)
                        return
                    current.state = FLClientState.PREPARATION_RESULT_PENDING
                    self._cancel_delay(subscription_id)
                    notification = NwdafMLModelTrainNotif(
                        notifCorreId=current.representation.notification_correlation_id,
                        mlCorreId=current.representation.ml_correlation_id,
                        mLModelInfos=[
                            current.representation.ml_model_infos[0].model_copy(deep=True)
                        ],
                    )
                self._enqueue_delivery(current, notification, FLClientState.PREPARED)
                self._release_work_slot(subscription_id, revision)
                return
            if self._client_settings.training_data is None:
                raise RuntimeError(
                    "image classification local training data is not configured"
                )
            collection_trigger = self._client_settings.training_data.collection_trigger
            self._datasets.validate_external_scope(intent, collection_trigger)
            job_id = self._datasets.submit_external(
                intent,
                window,
                lambda job: self._preparation_complete(
                    subscription_id,
                    revision,
                    job,
                    base.manifest,
                ),
                collection_trigger,
            )
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is not None and current.revision == revision:
                    current.dataset_job_id = job_id
        except Exception as error:
            logger.exception(
                "FL client preparation failed subscription_id=%s",
                subscription_id,
            )
            with self._lock:
                resource = self._resources.get(subscription_id)
                if resource is None or resource.revision != revision:
                    self._release_work_slot(subscription_id, revision)
                    return
                resource.state = FLClientState.FAILED
                resource.last_error = str(error)
                self._cancel_delay(subscription_id)
            self._enqueue_delivery(
                resource,
                _termination(resource),
                FLClientState.FAILED,
            )
            self._release_work_slot(subscription_id, revision)

    def _run_protocol_preparation(
        self,
        subscription_id: str,
        revision: int,
        value: NwdafMLModelTrainSubsc,
        *,
        initial: bool,
    ) -> None:
        violations = []
        if initial and value.ml_preparation_flag is not True:
            violations.append(
                InvalidParameter("mLPreFlag", "must be true for protocol preparation")
            )
        if initial and value.ml_model_infos:
            violations.append(
                InvalidParameter(
                    "mLModelInfos",
                    "must be omitted from protocol preparation",
                )
            )
        if value.fl_topology is None:
            violations.append(
                InvalidParameter("x-flTopology", "is required for protocol preparation")
            )
        if violations:
            raise RequirementsError(violations)
        node = value.fl_topology
        assert node is not None
        event = value.ml_event_subscriptions[0]
        contract = image_training_contract(event.ml_event, event.model_interoperability)
        context = self._nwdaf_context.get()
        intermediate_role = bool(node.children) or (
            node.policy is not None
            and node.policy.allow_additional_candidates is True
        )
        required_capabilities = (
            {FLCapabilityType.SERVER_AND_CLIENT}
            if intermediate_role
            else {
                FLCapabilityType.CLIENT,
                FLCapabilityType.SERVER_AND_CLIENT,
            }
        )
        if not any(
            event.ml_event in capability.ml_analytics_ids
            and capability.fl_capability_type in required_capabilities
            for capability in context.ml_analytics_capabilities
        ):
            raise RuntimeError(
                "containing NWDAF does not advertise the requested protocol FL capability"
            )
        with self._lock:
            current = self._resources.get(subscription_id)
            reservation_id = "" if current is None else current.experiment_reservation_id
        assigned_role = (
            ExperimentRole.BRANCH
            if intermediate_role
            else ExperimentRole.LEAF
        )
        experiment = self._experiments.for_client_subscription(subscription_id)
        if experiment is None or experiment.plan_id is None:
            self._experiments.bind_plan(
                reservation_id,
                value.ml_correlation_id or "",
                assigned_role,
            )
        elif (
            experiment.plan_id != value.ml_correlation_id
            or experiment.assigned_role is not assigned_role
        ):
            raise RuntimeError("protocol hierarchy binding changed")
        if intermediate_role:
            if node.report_after is None or node.report_after.unit != "round":
                raise RuntimeError(
                    "protocol Branch requires reportAfter with round unit"
                )
            intermediate_work = IntermediateLocalWork(
                lower_round_count=node.report_after.count
            )
            if self._branch_coordinator is None or not hasattr(
                self._branch_coordinator, "prepare_protocol"
            ):
                raise RuntimeError("protocol Branch preparation executor is unavailable")
            report = self._branch_coordinator.prepare_protocol(
                representation=value,
                reservation_id=reservation_id,
            )
        else:
            local_data = self._client_settings.training_data
            if not isinstance(
                self._client_settings.workload,
                ImageClassificationWorkloadSettings,
            ) or not isinstance(local_data, LocalImageTrainingDataSettings):
                raise RuntimeError(
                    "protocol image classification local training is not configured"
                )
            if local_data.dataset != contract.name.value:
                raise RuntimeError(
                    "protocol image dataset is incompatible with local configuration"
                )
            if (
                event.model_interoperability
                not in self._client_settings.model_interoperability_ids
            ):
                raise RuntimeError(
                    "protocol model interoperability is not supported locally"
                )
            if node.strategy is None or node.report_after is None:
                raise RuntimeError(
                    "protocol Leaf requires strategy and reportAfter"
                )
            if node.report_after.unit != "epoch":
                raise RuntimeError("protocol Leaf reportAfter unit must be epoch")
            client_work = ClientLocalWork(
                epochs=node.report_after.count,
                proximal_mu=node.strategy.method_parameters.proximal_mu,
            )
            report_values = {
                "nfInstanceId": context.nf_instance_id,
                **(
                    {"policy": node.policy.model_dump(by_alias=True, mode="json")}
                    if node.policy is not None
                    else {}
                ),
                **(
                    {
                        "strategy": node.strategy.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                    }
                    if node.strategy is not None
                    else {}
                ),
                **(
                    {
                        "reportAfter": node.report_after.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                    }
                    if node.report_after is not None
                    else {}
                ),
            }
            report = FlTopologyReport.model_validate(report_values)
        with self._lock:
            current = self._resources.get(subscription_id)
            if current is None or current.revision != revision:
                return
            current.protocol_image_dataset = contract.name
            if intermediate_role:
                current.intermediate_local_work = intermediate_work
            else:
                current.client_local_work = client_work
            current.state = FLClientState.PREPARATION_RESULT_PENDING
            self._cancel_delay(subscription_id)
            notification = NwdafMLModelTrainNotif(
                notifCorreId=value.notification_correlation_id,
                mlCorreId=value.ml_correlation_id,
                fl_topology_report=report,
            )
        self._enqueue_delivery(current, notification, FLClientState.PREPARED)
        self._release_work_slot(subscription_id, revision)

    def _preparation_complete(
        self,
        subscription_id: str,
        revision: int,
        job: DatasetJob,
        base_manifest: dict[str, object],
    ) -> None:
        preparation_error = ""
        training_sample_count = 0
        if job.state == DatasetJobState.READY and job.snapshot is not None:
            try:
                dataset = self._dataset_builder.build(job.snapshot, base_manifest)
                training_sample_count = sum(
                    scope.training_sample_count for scope in dataset.training_scopes
                )
            except Exception as error:
                preparation_error = str(error)
        with self._lock:
            resource = self._resources.get(subscription_id)
            if resource is None or resource.revision != revision:
                self._release_work_slot(subscription_id, revision)
                return
            self._cancel_delay(subscription_id)
            if (
                job.state == DatasetJobState.READY
                and job.snapshot is not None
                and not preparation_error
            ):
                minimum = _minimum_samples(resource.representation)
                if training_sample_count < minimum:
                    resource.state = FLClientState.FAILED
                    resource.last_error = "prepared training dataset does not meet minNumSamples"
                    notification = _termination(resource)
                    final = FLClientState.FAILED
                else:
                    resource.dataset_snapshot = job.snapshot
                    resource.prepared_training_sample_count = training_sample_count
                    resource.state = FLClientState.PREPARATION_RESULT_PENDING
                    notification = NwdafMLModelTrainNotif(
                        notifCorreId=resource.representation.notification_correlation_id,
                        mlCorreId=resource.representation.ml_correlation_id,
                        mLModelInfos=[
                            resource.representation.ml_model_infos[0].model_copy(deep=True)
                        ],
                    )
                    final = FLClientState.PREPARED
            else:
                resource.state = FLClientState.FAILED
                resource.last_error = preparation_error or job.failure
                notification = _termination(resource)
                final = FLClientState.FAILED
        self._enqueue_delivery(resource, notification, final)
        logger.info(
            "FL client preparation terminal subscription_id=%s state=%s "
            "records=%s samples=%s error=%s",
            subscription_id,
            final,
            len(job.snapshot.records) if job.snapshot is not None else 0,
            training_sample_count,
            resource.last_error or "none",
        )
        self._release_work_slot(subscription_id, revision)

    def _run_round(self, subscription_id: str, revision: int) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if resource.revision != revision:
                    raise RuntimeError("FL round resource revision is stale")
                value = resource.representation.model_copy(deep=True)
                snapshot = resource.dataset_snapshot
                preparation_base_artifact = resource.preparation_base_artifact
            model_info = value.ml_model_infos[0]
            if model_info.model_adrf is not None:
                if self._round_model_distribution is None:
                    raise RuntimeError("ADRF round model retrieval is unavailable")
                if model_info.model_unique_id is None:
                    raise RuntimeError("ADRF round input requires modelUniqueId")
                retrieved = self._round_model_distribution.retrieve(
                    reference=model_info.model_adrf,
                    model_unique_id=model_info.model_unique_id,
                    consumer_nf_instance_id=self._nwdaf_context.get().nf_instance_id,
                )
                downloaded = self._workspace.download_adrf_archive(
                    retrieved.model_url,
                    value.ml_correlation_id or subscription_id,
                    f"round-{value.round_indicator}-input",
                    expected_size=retrieved.model_size,
                    owner_plan_id=_owner_plan_id(resource),
                )
                artifact = downloaded.metadata
            else:
                if model_info.model_file_address is None:
                    raise RuntimeError("FL round input has no supported model transport")
                artifact = self._workspace.download(
                    str(model_info.model_file_address.model_url),
                    value.ml_correlation_id or subscription_id,
                    f"round-{value.round_indicator}-input",
                    owner_plan_id=_owner_plan_id(resource),
                )
            base = self._loader.load(artifact)
            round_input = validate_fl_artifact_manifest(base.manifest)
            if not isinstance(round_input, RoundInputArtifact):
                raise RuntimeError("FL round input is not a ROUND_INPUT artifact")
            if (
                round_input.fl_metadata.ml_corre_id != value.ml_correlation_id
                or (
                    not resource.candidate_contract
                    and round_input.fl_metadata.round_ind != value.round_indicator
                )
            ):
                raise RuntimeError("FL round input identity does not match the command")
            if preparation_base_artifact is None:
                if not resource.candidate_contract:
                    raise RuntimeError("FL round has no prepared base model")
                event = value.ml_event_subscriptions[0]
                if resource.intermediate_local_work is not None:
                    self._validate_protocol_intermediate_bundle(
                        base,
                        event.ml_event,
                        event.model_interoperability,
                    )
                else:
                    self._validate_workload_bundle(
                        base,
                        event.ml_event,
                        event.model_interoperability,
                    )
                with self._lock:
                    current = self._resources.get(subscription_id)
                    if current is None or current.revision != revision:
                        return
                    current.preparation_base_artifact = artifact
                    preparation_base_artifact = artifact
            prepared_base = self._loader.load(preparation_base_artifact)
            validate_model_compatibility(prepared_base, base)
            if (
                resource.candidate_contract
                and resource.intermediate_local_work is not None
            ):
                if self._branch_coordinator is None:
                    raise RuntimeError("protocol Branch round executor is unavailable")
                if resource.intermediate_local_work is None:
                    raise RuntimeError("protocol Branch local round contract is unavailable")
                published = self._branch_coordinator.execute_protocol_round(
                    representation=value,
                    upper_input=base,
                    upper_input_artifact=artifact,
                    upper_model=model_info,
                    upper_client_subscription_id=subscription_id,
                    upper_resource_revision=revision,
                    upper_training_scope=resource.scope,
                    callback_margin_seconds=(
                        self._client_settings.callback_deadline_margin_seconds
                    ),
                    local_work=resource.intermediate_local_work,
                )
                notification = NwdafMLModelTrainNotif(
                    notifCorreId=value.notification_correlation_id,
                    mlCorreId=value.ml_correlation_id,
                    roundInd=value.round_indicator,
                    mLModelInfos=[
                        MLEventNotification(
                            event=value.ml_event_subscriptions[0].ml_event,
                            mLFileAddr=MLModelAddress(mLModelUrl=published.url),
                        )
                    ],
                )
                with self._lock:
                    current = self._resources.get(subscription_id)
                    if current is not resource or current.revision != revision:
                        return
                    current.state = FLClientState.RESULT_PENDING
                    self._cancel_delay(subscription_id)
                self._enqueue_delivery(current, notification, FLClientState.READY)
                return
            local_image = self._client_settings.training_data
            if isinstance(local_image, LocalImageTrainingDataSettings):
                dataset = self._image_dataset_loader.load(
                    local_image.shard_path,
                    local_image.dataset,
                )
                training_sample_count = dataset.sample_count
                evidence = {
                    "workload_profile": "image_classification",
                    "dataset": local_image.dataset,
                    "training_sample_count": training_sample_count,
                }
            else:
                if isinstance(
                    self._client_settings.workload,
                    ImageClassificationWorkloadSettings,
                ):
                    raise RuntimeError(
                        "image classification local training data is not configured"
                    )
                if snapshot is None:
                    raise RuntimeError("FL round has no prepared ADRF dataset")
                dataset = self._dataset_builder.build(snapshot, base.manifest)
                training_sample_count = sum(
                    scope.training_sample_count for scope in dataset.training_scopes
                )
                if training_sample_count != resource.prepared_training_sample_count:
                    raise RuntimeError("FL round dataset changed after preparation")
                evidence = dataset_evidence(dataset).as_dict()
            proximal_mu = None
            epochs = round_input.fl_metadata.client_training.epochs
            if resource.client_local_work is not None:
                epochs = resource.client_local_work.epochs
                proximal_mu = resource.client_local_work.proximal_mu
            result = self._trainer.train(
                base,
                dataset,
                epochs=epochs,
                proximal_mu=proximal_mu,
            )
            if training_sample_count != result.training_sample_count:
                raise RuntimeError("local training sample count does not match dataset evidence")
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is not resource or current.revision != revision:
                    return
            participant_id = self._participant_id()
            metadata = {
                "artifact_role": "ROUND_LOCAL",
                "result_type": "TRAINING",
                "fl_metadata": {
                    "ml_corre_id": value.ml_correlation_id,
                    "round_ind": value.round_indicator,
                    "participant_nf_instance_id": participant_id,
                    "training_scope": resource.scope.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "training_sample_count": result.training_sample_count,
                    "dataset_evidence": evidence,
                },
            }
            published = self._workspace.publish(
                process_id=value.ml_correlation_id or subscription_id,
                participant_id=participant_id,
                round_indicator=value.round_indicator or 0,
                role="ROUND_LOCAL",
                base=base,
                model=result.model,
                owner_plan_id=_owner_plan_id(resource),
                metadata=metadata,
            )
            notification = NwdafMLModelTrainNotif(
                notifCorreId=value.notification_correlation_id,
                mlCorreId=value.ml_correlation_id,
                roundInd=value.round_indicator,
                mLModelInfos=[
                    MLEventNotification(
                        event=value.ml_event_subscriptions[0].ml_event,
                        mLFileAddr=MLModelAddress(mLModelUrl=published.url),
                    )
                ],
            )
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is not resource or current.revision != revision:
                    return
                current.state = FLClientState.RESULT_PENDING
                self._cancel_delay(subscription_id)
            self._enqueue_delivery(current, notification, FLClientState.READY)
            logger.info(
                "FL client local result ready subscription_id=%s round=%s samples=%s artifact=%s",
                subscription_id,
                value.round_indicator,
                result.training_sample_count,
                published.url,
            )
        except Exception as error:
            logger.exception("FL client round failed subscription_id=%s", subscription_id)
            with self._lock:
                resource = self._resources.get(subscription_id)
                if resource is not None:
                    resource.last_error = str(error)
                    resource.state = FLClientState.FAILED
                    self._cancel_delay(subscription_id)
            if resource is not None:
                self._enqueue_delivery(resource, _termination(resource), FLClientState.FAILED)
        finally:
            self._release_work_slot(subscription_id, revision)

    def _validate_workload_bundle(
        self,
        bundle: LoadedBundle,
        ml_event: str,
        model_interoperability: str,
    ) -> None:
        if bundle.manifest.get("model_interoperability") != model_interoperability:
            raise RuntimeError("FL preparation base model interoperability is incompatible")
        image_workload = isinstance(
            self._client_settings.workload,
            ImageClassificationWorkloadSettings,
        )
        if bundle.workload_profile is WorkloadProfile.IMAGE_CLASSIFICATION:
            if not image_workload:
                raise RuntimeError("FL preparation model workload is incompatible")
            if (
                ml_event != IMAGE_CLASSIFICATION_EVENT
                or bundle.manifest.get("analytics_event") != ml_event
            ):
                raise RuntimeError(
                    "FL preparation image analytics event is incompatible"
                )
            if not isinstance(
                self._client_settings.training_data,
                LocalImageTrainingDataSettings,
            ):
                raise RuntimeError(
                    "image classification local training data is not configured"
                )
            contract = validate_image_manifest(bundle.manifest)
            if contract.name.value != self._client_settings.training_data.dataset:
                raise RuntimeError("FL preparation image dataset is incompatible")
            return
        if image_workload:
            raise RuntimeError("FL preparation model workload is incompatible")
        if bundle.manifest.get("analytics_event") != ml_event:
            raise RuntimeError("FL preparation base model analytics event is incompatible")

    @staticmethod
    def _validate_protocol_intermediate_bundle(
        bundle: LoadedBundle,
        ml_event: str,
        model_interoperability: str,
    ) -> None:
        if bundle.manifest.get("model_interoperability") != model_interoperability:
            raise RuntimeError("FL preparation base model interoperability is incompatible")
        if bundle.workload_profile is not WorkloadProfile.IMAGE_CLASSIFICATION:
            raise RuntimeError("FL preparation model workload is incompatible")
        if (
            ml_event != IMAGE_CLASSIFICATION_EVENT
            or bundle.manifest.get("analytics_event") != ml_event
        ):
            raise RuntimeError("FL preparation image analytics event is incompatible")
        expected = image_training_contract(ml_event, model_interoperability)
        actual = validate_image_manifest(bundle.manifest)
        if actual != expected:
            raise RuntimeError("FL preparation image dataset is incompatible")

    def _run_validation(self, subscription_id: str, revision: int) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if (
                    resource.revision != revision
                    or resource.preparation_base_artifact is None
                ):
                    raise RuntimeError("final validation has no frozen preparation inputs")
                value = resource.representation.model_copy(deep=True)
                snapshot = resource.dataset_snapshot
                preparation_base_artifact = resource.preparation_base_artifact
            model_info = value.ml_model_infos[0]
            if model_info.model_file_address is None:
                raise RuntimeError("final validation candidate must use mLFileAddr")
            candidate_artifact = self._workspace.download(
                str(model_info.model_file_address.model_url),
                value.ml_correlation_id or subscription_id,
                f"validation-{value.round_indicator}-candidate",
                owner_plan_id=_owner_plan_id(resource),
            )
            base = self._loader.load(preparation_base_artifact)
            candidate = self._loader.load(candidate_artifact)
            candidate_contract = validate_fl_artifact_manifest(candidate.manifest)
            if not isinstance(candidate_contract, RoundGlobalArtifact):
                raise RuntimeError("final validation candidate is not a ROUND_GLOBAL artifact")
            validate_model_compatibility(base, candidate)
            if (
                value.round_indicator is None
                or candidate_contract.fl_metadata.round_ind != value.round_indicator - 1
            ):
                raise RuntimeError("final validation candidate identity does not match the command")
            if candidate_contract.fl_metadata.ml_corre_id != value.ml_correlation_id:
                raise RuntimeError(
                    "final validation candidate does not match the upper process"
                )
            if snapshot is None:
                raise RuntimeError("final validation has no frozen preparation dataset")
            dataset = self._dataset_builder.build(snapshot, base.manifest)
            training_sample_count = sum(
                scope.training_sample_count for scope in dataset.training_scopes
            )
            if training_sample_count != resource.prepared_training_sample_count:
                raise RuntimeError("final validation dataset changed after preparation")
            base_error = base_actual = candidate_error = candidate_actual = 0.0
            sample_count = 0
            for scope in dataset.evaluation_scopes:
                expected = scope.validation_targets
                if expected is None:
                    continue
                base_metric = wape(
                    expected,
                    LocalTrainer._predict(
                        base.model,
                        base.scaler,
                        scope,
                        dataset,
                        self._device,
                    ),
                )
                candidate_metric = wape(
                    expected,
                    LocalTrainer._predict(
                        candidate.model,
                        candidate.scaler,
                        scope,
                        dataset,
                        self._device,
                    ),
                )
                base_error += base_metric.error_sum
                base_actual += base_metric.actual_sum
                candidate_error += candidate_metric.error_sum
                candidate_actual += candidate_metric.actual_sum
                sample_count += scope.validation_sample_count
            if sample_count <= 0 or base_actual <= 0 or candidate_actual <= 0:
                raise RuntimeError("final validation requires non-zero evaluation evidence")
            participant_id = self._participant_id()
            published = self._workspace.publish(
                process_id=value.ml_correlation_id or subscription_id,
                participant_id=participant_id,
                round_indicator=value.round_indicator or 0,
                role="ROUND_LOCAL",
                base=candidate,
                model=candidate.model,
                owner_plan_id=_owner_plan_id(resource),
                metadata={
                    "artifact_role": "ROUND_LOCAL",
                    "result_type": "ACCURACY_CHECK",
                    "fl_metadata": {
                        "ml_corre_id": value.ml_correlation_id,
                        "round_ind": value.round_indicator,
                        "participant_nf_instance_id": participant_id,
                        "training_scope": resource.scope.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "evaluation": {
                            "evaluation_stage": "FINAL_VALIDATION",
                            "evaluation_sample_count": sample_count,
                            "start_time": snapshot.time_window.start_time.isoformat(),
                            "end_time": snapshot.time_window.stop_time.isoformat(),
                            "base": {
                                "absolute_error_sum": base_error,
                                "absolute_actual_sum": base_actual,
                            },
                            "candidate": {
                                "absolute_error_sum": candidate_error,
                                "absolute_actual_sum": candidate_actual,
                            },
                        },
                    },
                },
            )
            notification = NwdafMLModelTrainNotif(
                notifCorreId=value.notification_correlation_id,
                mlCorreId=value.ml_correlation_id,
                roundInd=value.round_indicator,
                mLModelInfos=[
                    MLEventNotification(
                        event=value.ml_event_subscriptions[0].ml_event,
                        mLFileAddr=MLModelAddress(mLModelUrl=published.url),
                    )
                ],
            )
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is None or current.revision != revision:
                    return
                current.state = FLClientState.RESULT_PENDING
                self._cancel_delay(subscription_id)
            self._enqueue_delivery(current, notification, FLClientState.READY)
            logger.info(
                "FL client final validation ready subscription_id=%s round=%s samples=%s",
                subscription_id,
                value.round_indicator,
                sample_count,
            )
        except Exception as error:
            logger.exception(
                "FL client final validation failed subscription_id=%s",
                subscription_id,
            )
            with self._lock:
                resource = self._resources.get(subscription_id)
                if resource is not None:
                    resource.last_error = str(error)
                    resource.state = FLClientState.FAILED
                    self._cancel_delay(subscription_id)
            if resource is not None:
                self._enqueue_delivery(resource, _termination(resource), FLClientState.FAILED)
        finally:
            self._release_work_slot(subscription_id, revision)

    def _enqueue_delivery(
        self,
        resource: FLClientResource,
        notification: NwdafMLModelTrainNotif,
        success_state: FLClientState,
    ) -> None:
        payload = notification.model_dump(by_alias=True, exclude_none=True, mode="json")
        key = (resource.subscription_id, resource.revision)
        with self._lock:
            current = self._resources.get(resource.subscription_id)
            if current is None or current.revision != resource.revision:
                return
            if key in self._outbox_keys:
                return
            self._outbox_keys.add(key)
        future = self._outbox_executor.submit(
            self._deliver_until_ack,
            resource.subscription_id,
            resource.revision,
            str(resource.representation.notification_uri),
            payload,
            success_state,
        )
        with self._lock:
            self._outbox_futures.add(future)
        future.add_done_callback(self._outbox_done)

    def _deliver_until_ack(
        self,
        subscription_id: str,
        revision: int,
        notification_uri: str,
        payload: dict,
        success_state: FLClientState,
    ) -> None:
        last_error = ""
        attempt = 0
        terminal = False
        while not self._closing.is_set():
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is None or current.revision != revision:
                    terminal = True
                    break
            try:
                response = self._client.post(notification_uri, json=payload)
                if response.status_code == 204:
                    superseded = ()
                    with self._lock:
                        current = self._resources.get(subscription_id)
                        if current is not None and current.revision == revision:
                            current.state = success_state
                            if success_state is FLClientState.PREPARED:
                                superseded = self._supersede_prepared_leaf_resource(
                                    subscription_id,
                                    revision,
                                )
                    for old_resource in superseded:
                        self._deliver_superseded_termination(old_resource)
                    terminal = True
                    break
                last_error = f"callback returned {response.status_code}"
                if 400 <= response.status_code < 500:
                    with self._lock:
                        current = self._resources.get(subscription_id)
                        if current is not None and current.revision == revision:
                            current.state = FLClientState.FAILED
                            current.last_error = f"callback rejected permanently: {last_error}"
                    terminal = True
                    break
            except httpx.TransportError as error:
                last_error = str(error)
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is None or current.revision != revision:
                    terminal = True
                    break
                current.state = FLClientState.RESULT_PENDING
                current.last_error = f"callback delivery pending: {last_error}"
            exponent = min(
                attempt,
                max(0, self._notification_settings.max_attempts - 1),
            )
            delay = min(
                self._notification_settings.initial_backoff_seconds * (2**exponent),
                self._notification_settings.max_backoff_seconds,
            )
            attempt += 1
            if self._closing.wait(delay):
                break
        key = (subscription_id, revision)
        with self._lock:
            self._outbox_keys.discard(key)
            abandoned = key in self._abandoned_outbox_keys
            self._abandoned_outbox_keys.discard(key)
            current = self._resources.get(subscription_id)
            if current is not None and current.revision == revision and terminal:
                current.callback_slot_owned = False
        if terminal and not abandoned:
            self._outbox_capacity.release()

    def _supersede_prepared_leaf_resource(
        self,
        subscription_id: str,
        revision: int,
    ) -> tuple[FLClientResource, ...]:
        release_work_slots = 0
        release_callback_slots = 0
        with self._lock:
            experiment = self._experiments.for_client_subscription(subscription_id)
            if (
                experiment is None
                or experiment.lifecycle is not ExperimentLifecycle.ACTIVE
                or experiment.assigned_role is not ExperimentRole.LEAF
            ):
                return ()
            superseded_ids = tuple(
                sorted(
                    value
                    for value in experiment.upper_client_subscription_ids
                    if value != subscription_id
                )
            )
            if not superseded_ids:
                return ()
            current = self._resources.get(subscription_id)
            if current is None or current.revision != revision:
                return ()
            for old_id in superseded_ids:
                old = self._resources.get(old_id)
                if (
                    old is None
                    or not old.candidate_contract
                    or old.experiment_reservation_id != experiment.reservation_id
                    or old.representation.ml_correlation_id
                    != current.representation.ml_correlation_id
                ):
                    raise RuntimeError(
                        "Leaf rebind resource group does not match the active procedure"
                    )
            self._experiments.supersede_clients(
                experiment.reservation_id,
                active_subscription_id=subscription_id,
                superseded_subscription_ids=superseded_ids,
            )
            superseded_resources = []
            for old_id in superseded_ids:
                old = self._resources.get(old_id)
                if old is None:
                    continue
                key = (old.subscription_id, old.revision)
                if old.work_slot_owned:
                    old.work_slot_owned = False
                    self._abandoned_work_keys.add(key)
                    release_work_slots += 1
                if old.callback_slot_owned:
                    if key in self._outbox_keys:
                        self._abandoned_outbox_keys.add(key)
                    old.callback_slot_owned = False
                    release_callback_slots += 1
                old.revision += 1
                old.state = FLClientState.TERMINATING
                old.last_error = ""
                old.dataset_snapshot = None
                old.dataset_job_id = ""
                old.prepared_training_sample_count = 0
                old.preparation_base_artifact = None
                old.experiment_reservation_id = ""
                old.protocol_image_dataset = None
                old.client_local_work = None
                old.intermediate_local_work = None
                self._cancel_delay(old.subscription_id)
                self._deleting.discard(old.subscription_id)
                superseded_resources.append(copy.deepcopy(old))
        for _ in range(release_callback_slots):
            self._outbox_capacity.release()
        for _ in range(release_work_slots):
            self._capacity.release()
        return tuple(superseded_resources)

    def _deliver_superseded_termination(
        self,
        resource: FLClientResource,
    ) -> None:
        payload = _termination(resource).model_dump(
            by_alias=True,
            exclude_none=True,
            mode="json",
        )
        last_error = ""
        for attempt in range(self._notification_settings.max_attempts):
            with self._lock:
                current = self._resources.get(resource.subscription_id)
                if (
                    current is None
                    or current.revision != resource.revision
                    or current.state is not FLClientState.TERMINATING
                ):
                    return
            try:
                response = self._client.post(
                    str(resource.representation.notification_uri),
                    json=payload,
                )
                if response.status_code == 204:
                    logger.info(
                        "Delivered superseded FL Client termination subscription_id=%s",
                        resource.subscription_id,
                    )
                    return
                last_error = f"termination delivery returned {response.status_code}"
                self._discard_terminal_resource(resource.subscription_id, resource.revision)
                logger.info(
                    "Discarded superseded FL Client resource after explicit delivery "
                    "failure subscription_id=%s status=%s",
                    resource.subscription_id,
                    response.status_code,
                )
                return
            except httpx.TransportError as error:
                last_error = str(error)
            if attempt + 1 < self._notification_settings.max_attempts:
                delay = min(
                    self._notification_settings.initial_backoff_seconds
                    * (2**attempt),
                    self._notification_settings.max_backoff_seconds,
                )
                if self._closing.wait(delay):
                    break
        with self._lock:
            current = self._resources.get(resource.subscription_id)
            if (
                current is not None
                and current.revision == resource.revision
                and current.state is FLClientState.TERMINATING
            ):
                current.last_error = (
                    "superseded termination delivery remains pending: "
                    f"{last_error}"
                )
        logger.warning(
            "Superseded FL Client termination delivery remains pending "
            "subscription_id=%s error=%s",
            resource.subscription_id,
            last_error,
        )

    def _discard_terminal_resource(self, subscription_id: str, revision: int) -> None:
        with self._lock:
            current = self._resources.get(subscription_id)
            if (
                current is None
                or current.revision != revision
                or current.state is not FLClientState.TERMINATING
            ):
                return
            self._resources.pop(subscription_id, None)
            self._deleting.discard(subscription_id)
            self._cancelled_hierarchy_resources[subscription_id] = (
                self._clock() + self._settings.lifecycle.tombstone_ttl_seconds
            )

    def _outbox_done(self, future: Future) -> None:
        with self._lock:
            self._outbox_futures.discard(future)

    def _release_work_slot(self, subscription_id: str, revision: int) -> None:
        key = (subscription_id, revision)
        with self._lock:
            if key in self._abandoned_work_keys:
                self._abandoned_work_keys.discard(key)
                return
            resource = self._resources.get(subscription_id)
            if (
                resource is None
                or resource.revision != revision
                or not resource.work_slot_owned
            ):
                return
            resource.work_slot_owned = False
        self._capacity.release()

    def _participant_id(self) -> str:
        context = self._nwdaf_context.get()
        if not context.nf_instance_id:
            raise RuntimeError("containing NWDAF identity is unavailable")
        return context.nf_instance_id

    def _schedule_delay(self, resource: FLClientResource) -> None:
        self._cancel_delay(resource.subscription_id)
        report = resource.representation.ml_training_report_info
        maximum = report.maximum_response_time if report is not None else None
        if maximum is None:
            maximum = (
                self._client_settings.fallback_deadlines.preparation_timeout_seconds
                if resource.state == FLClientState.PREPARING
                else self._client_settings.fallback_deadlines.round_timeout_seconds
            )
        delay = max(
            0.001,
            maximum - self._client_settings.callback_deadline_margin_seconds,
        )
        timer = threading.Timer(
            delay,
            self._send_delay_if_running,
            args=(resource.subscription_id, resource.revision, maximum),
        )
        timer.daemon = True
        self._delay_timers[resource.subscription_id] = timer
        timer.start()

    def _cancel_delay(self, subscription_id: str) -> None:
        timer = self._delay_timers.pop(subscription_id, None)
        if timer is not None:
            timer.cancel()

    def _send_delay_if_running(
        self,
        subscription_id: str,
        revision: int,
        expected_seconds: int,
    ) -> None:
        with self._lock:
            resource = self._resources.get(subscription_id)
            if (
                resource is None
                or resource.revision != revision
                or resource.state
                not in {
                    FLClientState.PREPARING,
                    FLClientState.ROUND_RUNNING,
                    FLClientState.VALIDATION_RUNNING,
                }
            ):
                return
            notification = NwdafMLModelTrainNotif(
                notifCorreId=resource.representation.notification_correlation_id,
                mlCorreId=resource.representation.ml_correlation_id,
                roundInd=(
                    resource.representation.round_indicator
                    if resource.state
                    in {FLClientState.ROUND_RUNNING, FLClientState.VALIDATION_RUNNING}
                    else None
                ),
                delayEventNotif=DelayEventNotif(
                    delayEventInd=True,
                    delayCause="NEED_MORE_TIME",
                    expCompTime=expected_seconds,
                ),
            )
        self._deliver_delay(resource, notification)

    def _deliver_delay(
        self,
        resource: FLClientResource,
        notification: NwdafMLModelTrainNotif,
    ) -> None:
        payload = notification.model_dump(by_alias=True, exclude_none=True, mode="json")
        last_error = ""
        for attempt in range(self._notification_settings.max_attempts):
            try:
                response = self._client.post(
                    str(resource.representation.notification_uri), json=payload
                )
                if response.status_code == 204:
                    return
                last_error = f"delay callback returned {response.status_code}"
            except httpx.TransportError as error:
                last_error = str(error)
            if attempt + 1 < self._notification_settings.max_attempts:
                delay = min(
                    self._notification_settings.initial_backoff_seconds * (2**attempt),
                    self._notification_settings.max_backoff_seconds,
                )
                if self._closing.wait(delay):
                    break
        with self._lock:
            current = self._resources.get(resource.subscription_id)
            if current is not None and current.revision == resource.revision:
                current.last_error = f"delay callback delivery failed: {last_error}"

    def _submit(self, function, *args) -> None:
        future = self._executor.submit(function, *args)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._future_done)

    def _future_done(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _required(self, subscription_id: str) -> FLClientResource:
        if subscription_id in self._deleting:
            raise KeyError(subscription_id)
        resource = self._resources.get(subscription_id)
        if resource is None:
            raise KeyError(subscription_id)
        return resource

    def _validate_preparation_admission(self, value: NwdafMLModelTrainSubsc) -> None:
        violations: list[InvalidParameter] = []
        if value.ml_preparation_flag is not True:
            violations.append(
                InvalidParameter(
                    "mLPreFlag",
                    "must be true when creating an FL Client resource",
                )
            )
        if len(value.ml_event_subscriptions) != 1:
            violations.append(
                InvalidParameter(
                    "mLEventSubscs",
                    "the current FL Client profile requires exactly one item",
                )
            )
        if len(value.ml_model_training_infos or ()) != len(value.ml_event_subscriptions):
            violations.append(
                InvalidParameter(
                    "mLModelTrainInfos",
                    "must correspond one-to-one with mLEventSubscs",
                )
            )
        for index, event in enumerate(value.ml_event_subscriptions):
            if event.ml_event != "UE_COMMUNICATION":
                violations.append(
                    InvalidParameter(
                        f"mLEventSubscs[{index}].mLEvent",
                        "only UE_COMMUNICATION training is supported",
                    )
                )
            if event.model_interoperability not in set(
                self._client_settings.model_interoperability_ids
            ):
                violations.append(
                    InvalidParameter(
                        f"mLEventSubscs[{index}].modelInterInfo",
                        "is not supported by this FL Client",
                    )
                )
            if (
                len(value.ml_model_infos or ()) != 1
                or value.ml_model_infos[0].event != event.ml_event
                or value.ml_model_infos[0].model_file_address is None
            ):
                violations.append(
                    InvalidParameter(
                        "mLModelInfos",
                        "must provide one URL-addressed completed base model "
                        f"for mLEventSubscs[{index}]",
                    )
                )
        for index, info in enumerate(value.ml_model_training_infos or ()):
            duration = info.time_availability_requirement
            if duration is not None and not _is_positive_iso_duration(duration):
                violations.append(
                    InvalidParameter(
                        f"mLModelTrainInfos[{index}].timeAvReq",
                        "must be a positive ISO 8601 day/time duration",
                    )
                )
        report = value.ml_training_report_info
        if (
            report is not None
            and report.maximum_response_time is not None
            and report.maximum_response_time
            <= self._client_settings.callback_deadline_margin_seconds
        ):
            violations.append(
                InvalidParameter(
                    "mLTrainRepInfo.maxResTime",
                    "does not leave a positive callback window",
                )
            )
        if violations:
            raise RequirementsError(violations)

    @staticmethod
    def _ensure_mutable(resource: FLClientResource) -> None:
        if resource.state in {FLClientState.FAILED, FLClientState.TERMINATING}:
            raise RuntimeError("NOT_AVAILABLE_FOR_FL_PROCESS_ANYMORE")

    def _prune_cancelled_resources_locked(self) -> None:
        now = self._clock()
        self._cancelled_hierarchy_resources = {
            subscription_id: deadline
            for subscription_id, deadline in self._cancelled_hierarchy_resources.items()
            if deadline > now
        }

    @staticmethod
    def _same_representation(
        left: NwdafMLModelTrainSubsc,
        right: NwdafMLModelTrainSubsc,
    ) -> bool:
        return left.model_dump(by_alias=True, exclude_none=True, mode="json") == right.model_dump(
            by_alias=True, exclude_none=True, mode="json"
        )


def _requested_window(value: NwdafMLModelTrainSubsc) -> AdrfTimeWindow:
    for info in value.ml_model_training_infos or []:
        requirement = info.data_availability_requirement
        if requirement and requirement.time_windows:
            window = requirement.time_windows[0]
            return AdrfTimeWindow(startTime=window.start_time, stopTime=window.stop_time)
    stop = datetime.now(UTC)
    return AdrfTimeWindow(startTime=stop - timedelta(minutes=30), stopTime=stop)


def _minimum_samples(value: NwdafMLModelTrainSubsc) -> int:
    return max(
        (
            info.data_availability_requirement.minimum_sample_count or 0
            for info in value.ml_model_training_infos or []
            if info.data_availability_requirement is not None
        ),
        default=0,
    )


def _owner_plan_id(resource: FLClientResource) -> str | None:
    if not resource.candidate_contract:
        return None
    return resource.representation.ml_correlation_id


def _termination(resource: FLClientResource) -> NwdafMLModelTrainNotif:
    return NwdafMLModelTrainNotif(
        notifCorreId=resource.representation.notification_correlation_id,
        mlCorreId=resource.representation.ml_correlation_id,
        roundInd=resource.representation.round_indicator,
        termTrainReq="NOT_AVAILABLE_ML_TRAIN",
    )


_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def _is_positive_iso_duration(value: str) -> bool:
    match = _ISO_DURATION.fullmatch(value.strip())
    if match is None:
        return False
    if "T" in value and not any(
        match.group(name) is not None for name in ("hours", "minutes", "seconds")
    ):
        return False
    components = [item for item in match.groupdict().values() if item is not None]
    return bool(components) and any(float(item) > 0 for item in components)
