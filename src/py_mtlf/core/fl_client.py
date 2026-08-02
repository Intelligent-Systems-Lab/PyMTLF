import copy
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

import httpx

from py_mtlf.config import FederatedLearningSettings, FLClientSettings, NotificationSettings
from py_mtlf.core.accuracy_policy import RetrainIntent, ScopeReference
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.dataset import DatasetCoordinator, DatasetJob, DatasetJobState, DatasetSnapshot
from py_mtlf.core.federated_trainer import FederatedTrainer
from py_mtlf.core.fl_workspace import (
    FLWorkspace,
    model_contract_digest,
    preprocessing_contract_digest,
    weights_digest,
)
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.core.trainer import LocalTrainer, TrustedBundleLoader, wape
from py_mtlf.core.training_data import TrainingDatasetBuilder
from py_mtlf.core.training_scope import TrainingScopeDescriptor
from py_mtlf.wire.adrf import TimeWindow as AdrfTimeWindow
from py_mtlf.wire.ml_model import MLEventNotification, MLModelAddress
from py_mtlf.wire.ml_model_training import (
    DelayEventNotif,
    InvalidParameter,
    NwdafMLModelTrainNotif,
    NwdafMLModelTrainSubsc,
    NwdafMLModelTrainSubscPatch,
    RequirementsError,
    StatusReportInfo,
    TrainDataInfo,
    TrainingResourceIdentity,
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
    FAILED = "FAILED"
    FAILED_RESTART = "FAILED_RESTART"


class FLClientCapacityError(RuntimeError):
    pass


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
    restart_terminal: bool = False
    prepared_training_sample_count: int = 0
    expected_model_contract_digest: str = ""
    expected_preprocessing_contract_digest: str = ""
    preparation_base_artifact: ArtifactMetadata | None = None

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


class FLClientService:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        client_settings: FLClientSettings,
        notification_settings: NotificationSettings,
        projection: SyncProjection,
        datasets: DatasetCoordinator,
        workspace: FLWorkspace,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client_settings = client_settings
        self._notification_settings = notification_settings
        self._projection = projection
        self._datasets = datasets
        self._workspace = workspace
        self._trainer = FederatedTrainer(client_settings.training)
        self._dataset_builder = TrainingDatasetBuilder(client_settings.training)
        self._loader = TrustedBundleLoader()
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._resources: dict[str, FLClientResource] = {}
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

    def create(self, value: NwdafMLModelTrainSubsc) -> FLClientResource:
        validate_fl_subscription(value)
        self._validate_preparation_admission(value)
        if not self._capacity.acquire(blocking=False):
            raise FLClientCapacityError("FL client work capacity is exhausted")
        if not self._outbox_capacity.acquire(blocking=False):
            self._capacity.release()
            raise FLClientCapacityError("FL client callback outbox is full")
        resource_id = ""
        try:
            resource_id = str(uuid4())
            resource = FLClientResource(
                subscription_id=resource_id,
                representation=value.model_copy(deep=True),
                state=FLClientState.PROVISIONAL,
                scope=TrainingScopeDescriptor.from_training_request(value, 0),
            )
            with self._lock:
                if any(
                    item.representation.notification_correlation_id
                    == value.notification_correlation_id
                    for item in self._resources.values()
                ):
                    raise ValueError("notifCorreId must be unique")
                self._resources[resource_id] = resource
            self._start_operation(resource)
            return self.get(resource_id)
        except Exception:
            with self._lock:
                if resource_id:
                    self._cancel_delay(resource_id)
                    self._resources.pop(resource_id, None)
            self._capacity.release()
            self._outbox_capacity.release()
            raise

    def replace(self, subscription_id: str, value: NwdafMLModelTrainSubsc) -> FLClientResource:
        validate_fl_subscription(value)
        scope = TrainingScopeDescriptor.from_training_request(value, 0)
        with self._lock:
            resource = self._required(subscription_id)
            self._ensure_mutable(resource)
            validate_fl_subscription(value, resource.identity)
            if self._same_representation(value, resource.representation):
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
                resource.representation = value.model_copy(deep=True)
                resource.scope = scope
                resource.revision += 1
                resource.state = FLClientState.PROVISIONAL
                self._start_operation(resource)
                return copy.deepcopy(resource)
            except Exception:
                self._resources[subscription_id] = previous
                self._capacity.release()
                self._outbox_capacity.release()
                raise

    def patch(self, subscription_id: str, patch: NwdafMLModelTrainSubscPatch) -> FLClientResource:
        with self._lock:
            resource = self._required(subscription_id)
            self._ensure_mutable(resource)
            validate_fl_patch(patch, resource.identity)
            update = patch.model_dump(
                by_alias=True,
                exclude_unset=True,
                exclude_none=True,
                mode="json",
            )
            effective = resource.representation.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            )
            effective.update(update)
            value = NwdafMLModelTrainSubsc.model_validate(effective)
            if set(update) == {"mLTrainRepInfo"} and resource.state in {
                FLClientState.PREPARING,
                FLClientState.ROUND_RUNNING,
                FLClientState.VALIDATION_RUNNING,
            }:
                resource.representation = value
                self._schedule_delay(resource)
                return copy.deepcopy(resource)
            if self._same_representation(value, resource.representation):
                return copy.deepcopy(resource)
            if (
                resource.representation.round_indicator is not None
                and value.round_indicator is not None
                and value.round_indicator <= resource.representation.round_indicator
            ):
                raise RuntimeError("stale or conflicting FL round command")
        return self.replace(subscription_id, value)

    def delete(self, subscription_id: str) -> None:
        with self._lock:
            resource = self._required(subscription_id)
            if resource.state in {
                FLClientState.PREPARING,
                FLClientState.ROUND_RUNNING,
                FLClientState.VALIDATION_RUNNING,
                FLClientState.RESULT_PENDING,
                FLClientState.PREPARATION_RESULT_PENDING,
            }:
                raise RuntimeError("ML_TRAINING_NOT_COMPLETE")
            self._cancel_delay(subscription_id)
            del self._resources[subscription_id]

    def get(self, subscription_id: str) -> FLClientResource:
        with self._lock:
            return copy.deepcopy(self._required(subscription_id))

    def restore_after_restart(
        self,
        subscription_id: str,
        representation: NwdafMLModelTrainSubsc,
    ) -> None:
        validate_fl_subscription(representation)
        resource = FLClientResource(
            subscription_id=subscription_id,
            representation=representation.model_copy(deep=True),
            state=FLClientState.FAILED_RESTART,
            scope=TrainingScopeDescriptor.from_training_request(representation, 0),
            last_error="in-flight FL operation cannot resume after backend restart",
            restart_terminal=True,
        )
        with self._lock:
            self._resources[subscription_id] = resource
        if self._outbox_capacity.acquire(blocking=False):
            self._enqueue_delivery(
                resource,
                NwdafMLModelTrainNotif(
                    notifCorreId=representation.notification_correlation_id,
                    mlCorreId=representation.ml_correlation_id,
                    termTrainReq="NOT_AVAILABLE_ML_TRAIN",
                ),
                FLClientState.FAILED,
            )
        else:
            with self._lock:
                resource.state = FLClientState.FAILED
                resource.last_error = "restart termination callback could not enter full outbox"

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
        if resource.dataset_snapshot is None or resource.preparation_base_artifact is None:
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

    def _start_preparation(self, resource: FLClientResource) -> None:
        value = resource.representation
        window = _requested_window(value)
        event = value.ml_event_subscriptions[0]
        target = value.target_reporting_ue or event.target_ue
        scope = ScopeReference(
            scope_key=resource.scope.scope_digest,
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
        self._datasets.validate_external_scope(intent)
        resource.state = FLClientState.PREPARING
        self._schedule_delay(resource)
        self._submit(
            self._run_preparation,
            resource.subscription_id,
            resource.revision,
            intent,
            window,
        )

    def _run_preparation(
        self,
        subscription_id: str,
        revision: int,
        intent: RetrainIntent,
        window: AdrfTimeWindow,
    ) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if resource.revision != revision:
                    return
                value = resource.representation.model_copy(deep=True)
            model_info = value.ml_model_infos[0]
            if model_info.model_file_address is None:
                raise RuntimeError("FL preparation base model must use mLFileAddr")
            artifact = self._workspace.download(
                str(model_info.model_file_address.model_url),
                value.ml_correlation_id or subscription_id,
                "preparation-base",
            )
            base = self._loader.load(artifact)
            event = value.ml_event_subscriptions[0]
            if base.manifest.get("analytics_event") != event.ml_event:
                raise RuntimeError("FL preparation base model analytics event is incompatible")
            if base.manifest.get("model_interoperability") != event.model_interoperability:
                raise RuntimeError("FL preparation base model interoperability is incompatible")
            expected_model_contract = model_contract_digest(base.manifest)
            expected_preprocessing_contract = preprocessing_contract_digest(base.manifest)
            with self._lock:
                current = self._resources.get(subscription_id)
                if current is None or current.revision != revision:
                    return
                current.preparation_base_artifact = artifact
            job_id = self._datasets.submit_external(
                intent,
                window,
                lambda job: self._preparation_complete(
                    subscription_id,
                    revision,
                    job,
                    base.manifest,
                    expected_model_contract,
                    expected_preprocessing_contract,
                ),
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
                    return
                resource.state = FLClientState.FAILED
                resource.last_error = str(error)
                self._cancel_delay(subscription_id)
            self._enqueue_delivery(
                resource,
                _termination(resource),
                FLClientState.FAILED,
            )
            self._capacity.release()

    def _preparation_complete(
        self,
        subscription_id: str,
        revision: int,
        job: DatasetJob,
        base_manifest: dict[str, object],
        expected_model_contract: str,
        expected_preprocessing_contract: str,
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
                self._capacity.release()
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
                    resource.expected_model_contract_digest = expected_model_contract
                    resource.expected_preprocessing_contract_digest = (
                        expected_preprocessing_contract
                    )
                    resource.state = FLClientState.PREPARATION_RESULT_PENDING
                    notification = NwdafMLModelTrainNotif(
                        notifCorreId=resource.representation.notification_correlation_id,
                        mlCorreId=resource.representation.ml_correlation_id,
                        statusReport=StatusReportInfo(
                            trainInDataInfo=TrainDataInfo(samplRatio=100)
                        ),
                    )
                    final = FLClientState.PREPARED
            else:
                resource.state = FLClientState.FAILED
                resource.last_error = preparation_error or job.failure
                notification = _termination(resource)
                final = FLClientState.FAILED
        self._enqueue_delivery(resource, notification, final)
        logger.info(
            "FL client preparation terminal subscription_id=%s state=%s records=%s",
            subscription_id,
            final,
            len(job.snapshot.records) if job.snapshot is not None else 0,
        )
        self._capacity.release()

    def _run_round(self, subscription_id: str, revision: int) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if resource.revision != revision or resource.dataset_snapshot is None:
                    raise RuntimeError("FL round has no prepared ADRF dataset")
                value = resource.representation.model_copy(deep=True)
                snapshot = resource.dataset_snapshot
            model_info = value.ml_model_infos[0]
            if model_info.model_file_address is None:
                raise RuntimeError("FL round input must use mLFileAddr")
            artifact = self._workspace.download(
                str(model_info.model_file_address.model_url),
                value.ml_correlation_id or subscription_id,
                f"round-{value.round_indicator}-input",
            )
            base = self._loader.load(artifact)
            if (
                model_contract_digest(base.manifest) != resource.expected_model_contract_digest
                or preprocessing_contract_digest(base.manifest)
                != resource.expected_preprocessing_contract_digest
            ):
                raise RuntimeError(
                    "FL round input changed the prepared model or preprocessing contract"
                )
            dataset = self._dataset_builder.build(snapshot, base.manifest)
            training_sample_count = sum(
                scope.training_sample_count for scope in dataset.training_scopes
            )
            if training_sample_count != resource.prepared_training_sample_count:
                raise RuntimeError("FL round dataset changed after preparation")
            result = self._trainer.train(base, dataset)
            participant_id = self._participant_id()
            base_digest = weights_digest(base.model)
            output_digest = weights_digest(result.model)
            metadata = {
                "artifact_role": "ROUND_LOCAL",
                "result_type": "TRAINING",
                "fl_metadata": {
                    "contract_version": "1.0",
                    "ml_corre_id": value.ml_correlation_id,
                    "model_contract_digest": model_contract_digest(base.manifest),
                    "preprocessing_contract_digest": preprocessing_contract_digest(base.manifest),
                    "base_weights_digest": base_digest,
                    "weights_digest": output_digest,
                    "round_ind": value.round_indicator,
                    "participant_nf_instance_id": participant_id,
                    "scope_digest": resource.scope.scope_digest,
                    "input_global_weights_digest": base_digest,
                    "training_sample_count": result.training_sample_count,
                },
            }
            published = self._workspace.publish(
                process_id=value.ml_correlation_id or subscription_id,
                participant_id=participant_id,
                round_indicator=value.round_indicator or 0,
                role="ROUND_LOCAL",
                base=base,
                model=result.model,
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
                resource.state = FLClientState.RESULT_PENDING
                self._cancel_delay(subscription_id)
            self._enqueue_delivery(resource, notification, FLClientState.READY)
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
            self._capacity.release()

    def _run_validation(self, subscription_id: str, revision: int) -> None:
        try:
            with self._lock:
                resource = self._required(subscription_id)
                if (
                    resource.revision != revision
                    or resource.dataset_snapshot is None
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
            )
            base = self._loader.load(preparation_base_artifact)
            candidate = self._loader.load(candidate_artifact)
            for bundle, label in ((base, "base"), (candidate, "candidate")):
                if (
                    model_contract_digest(bundle.manifest)
                    != resource.expected_model_contract_digest
                    or preprocessing_contract_digest(bundle.manifest)
                    != resource.expected_preprocessing_contract_digest
                ):
                    raise RuntimeError(f"final validation {label} changed the prepared contract")
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
                    LocalTrainer._predict(base.model, base.scaler, scope, dataset),
                )
                candidate_metric = wape(
                    expected,
                    LocalTrainer._predict(
                        candidate.model,
                        candidate.scaler,
                        scope,
                        dataset,
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
            base_digest = weights_digest(base.model)
            candidate_digest = weights_digest(candidate.model)
            published = self._workspace.publish(
                process_id=value.ml_correlation_id or subscription_id,
                participant_id=participant_id,
                round_indicator=value.round_indicator or 0,
                role="ROUND_LOCAL",
                base=candidate,
                model=candidate.model,
                metadata={
                    "artifact_role": "ROUND_LOCAL",
                    "result_type": "ACCURACY_CHECK",
                    "fl_metadata": {
                        "contract_version": "1.0",
                        "ml_corre_id": value.ml_correlation_id,
                        "model_contract_digest": model_contract_digest(candidate.manifest),
                        "preprocessing_contract_digest": preprocessing_contract_digest(
                            candidate.manifest
                        ),
                        "base_weights_digest": candidate_digest,
                        "weights_digest": candidate_digest,
                        "round_ind": value.round_indicator,
                        "participant_nf_instance_id": participant_id,
                        "scope_digest": resource.scope.scope_digest,
                        "input_global_weights_digest": candidate_digest,
                        "evaluation": {
                            "evaluation_stage": "FINAL_VALIDATION",
                            "evaluation_sample_count": sample_count,
                            "start_time": snapshot.time_window.start_time.isoformat(),
                            "end_time": snapshot.time_window.stop_time.isoformat(),
                            "base_model_weights_digest": base_digest,
                            "candidate_weights_digest": candidate_digest,
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
            self._capacity.release()

    def _enqueue_delivery(
        self,
        resource: FLClientResource,
        notification: NwdafMLModelTrainNotif,
        success_state: FLClientState,
    ) -> None:
        payload = notification.model_dump(by_alias=True, exclude_none=True, mode="json")
        key = (resource.subscription_id, resource.revision)
        with self._lock:
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
            try:
                response = self._client.post(notification_uri, json=payload)
                if response.status_code == 204:
                    with self._lock:
                        current = self._resources.get(subscription_id)
                        if current is not None and current.revision == revision:
                            current.state = success_state
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
        with self._lock:
            self._outbox_keys.discard((subscription_id, revision))
        if terminal:
            self._outbox_capacity.release()

    def _outbox_done(self, future: Future) -> None:
        with self._lock:
            self._outbox_futures.discard(future)

    def _participant_id(self) -> str:
        snapshot = self._projection.snapshot()
        if snapshot is None or not snapshot.containing_nwdaf.nf_instance_id:
            raise RuntimeError("containing NWDAF identity is unavailable")
        return snapshot.containing_nwdaf.nf_instance_id

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
        if resource.restart_terminal or resource.state in {
            FLClientState.FAILED,
            FLClientState.FAILED_RESTART,
        }:
            raise RuntimeError("NOT_AVAILABLE_FOR_FL_PROCESS_ANYMORE")

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
