import hashlib
import logging
import queue
import tempfile
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from py_mtlf.config import TrainingSettings
from py_mtlf.core.accuracy_policy import AccuracyPolicy
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.bundle_builder import CandidateBundleBuilder
from py_mtlf.core.dataset import DatasetCoordinator, DatasetSnapshot
from py_mtlf.core.notification_delivery import ProvisionNotificationDispatcher
from py_mtlf.core.provision_store import ProvisionResourceStore
from py_mtlf.core.seed_catalog import FamilyKey, ModelCatalog, StaleCatalogError
from py_mtlf.core.trainer import LocalTrainer, TrustedBundleLoader
from py_mtlf.core.training_data import TrainingDatasetBuilder

logger = logging.getLogger(__name__)


class TrainingJobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    REPROVISIONING = "REPROVISIONING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TrainingJob:
    job_id: str
    dataset_job_id: str
    family_key: FamilyKey
    state: TrainingJobState = TrainingJobState.PENDING
    stage: str = "PENDING"
    failure: str = ""
    candidate_artifact_key: str = ""
    promoted_generation: int = 0


class TrainingCoordinator:
    def __init__(
        self,
        settings: TrainingSettings,
        datasets: DatasetCoordinator,
        catalog: ModelCatalog,
        artifacts: ArtifactRepository,
        provisions: ProvisionResourceStore,
        notifications: ProvisionNotificationDispatcher,
        accuracy_policy: AccuracyPolicy,
    ) -> None:
        self._settings = settings
        self._datasets = datasets
        self._catalog = catalog
        self._artifacts = artifacts
        self._provisions = provisions
        self._notifications = notifications
        self._policy = accuracy_policy
        self._builder = TrainingDatasetBuilder(settings)
        self._trainer = LocalTrainer(settings)
        self._loader = TrustedBundleLoader()
        self._bundle_builder = CandidateBundleBuilder()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=settings.max_queue_size)
        self._lock = threading.RLock()
        self._jobs: dict[str, TrainingJob] = {}
        self._dataset_to_job: dict[str, str] = {}
        self._workers: list[threading.Thread] = []
        self._closing = False

    def open(self) -> None:
        with self._lock:
            if self._workers:
                return
            self._closing = False
            self._datasets.set_ready_handler(self.submit)
            for index in range(self._settings.max_concurrent_jobs):
                worker = threading.Thread(
                    target=self._run,
                    name=f"local-training-{index}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    def submit(self, dataset_job_id: str) -> None:
        snapshot = self._datasets.claim_ready(dataset_job_id)
        if snapshot is None:
            return
        job = TrainingJob(
            job_id=str(uuid4()),
            dataset_job_id=dataset_job_id,
            family_key=snapshot.family_key,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._dataset_to_job[dataset_job_id] = job.job_id
            closing = self._closing
        if closing or not self._settings.enabled:
            self._terminal(
                job,
                TrainingJobState.CANCELLED if closing else TrainingJobState.FAILED,
                "training coordinator is shutting down"
                if closing
                else "local training is disabled",
            )
            return
        try:
            self._queue.put_nowait(job.job_id)
        except queue.Full:
            self._terminal(
                job,
                TrainingJobState.FAILED,
                "local training queue is full",
            )

    def jobs(self) -> tuple[TrainingJob, ...]:
        with self._lock:
            return tuple(self._copy(job) for job in self._jobs.values())

    def job_for_dataset(self, dataset_job_id: str) -> TrainingJob | None:
        with self._lock:
            job_id = self._dataset_to_job.get(dataset_job_id)
            job = self._jobs.get(job_id) if job_id else None
            return self._copy(job) if job is not None else None

    def shutdown(self) -> None:
        self._datasets.set_ready_handler(None)
        with self._lock:
            if not self._workers:
                return
            self._closing = True
            workers = tuple(self._workers)
        for _worker in workers:
            self._queue.put(None)
        for worker in workers:
            worker.join()
        with self._lock:
            self._workers.clear()

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                with self._lock:
                    job = self._jobs.get(job_id)
                if job is not None:
                    self._execute(job)
            finally:
                self._queue.task_done()

    def _execute(self, job: TrainingJob) -> None:
        snapshot = self._claimed_snapshot(job.dataset_job_id)
        if snapshot is None:
            self._terminal(
                job,
                TrainingJobState.FAILED,
                "claimed dataset snapshot is unavailable",
            )
            return
        try:
            self._stage(job, TrainingJobState.RUNNING, "BUILDING_DATASET")
            base = self._catalog.current(snapshot.family_key)
            if base is None:
                self._terminal(
                    job,
                    TrainingJobState.CANCELLED,
                    "model no longer exists in current catalog",
                )
                return
            if not self._provisions.resources_for_family(snapshot.family_key):
                self._terminal(
                    job,
                    TrainingJobState.CANCELLED,
                    "model has no active provision demand",
                )
                return
            base_manifest = self._artifacts.manifest(base.artifact.key)
            dataset = self._builder.build(snapshot, base_manifest)

            self._stage(job, TrainingJobState.RUNNING, "FITTING")
            current_bundle = self._loader.load(base.artifact)
            candidate_base = self._loader.load(base.artifact)
            result = self._trainer.train(current_bundle, candidate_base, dataset)
            self._log_evaluation(job, result.evaluation)
            if not result.evaluation.accepted:
                self._terminal(
                    job,
                    TrainingJobState.FAILED,
                    "candidate rejected by performance gate: "
                    + ",".join(result.evaluation.rejection_reasons),
                )
                return

            self._stage(job, TrainingJobState.RUNNING, "PACKAGING")
            candidate_version = self._catalog.reserve_next_version(snapshot.family_key)
            with tempfile.TemporaryDirectory(prefix="py-mtlf-candidate-") as temporary:
                bundle = self._bundle_builder.build(
                    Path(temporary),
                    result=result,
                    dataset=dataset,
                    snapshot=snapshot,
                    generation=base.generation + 1,
                    parent_artifact_key=base.artifact.key,
                    model_version_key=candidate_version,
                )
                artifact = self._artifacts.publish(bundle)
            self._loader.load(artifact)
            with self._lock:
                job.candidate_artifact_key = artifact.key

            self._stage(job, TrainingJobState.REPROVISIONING, "PROMOTING")
            current_scopes = set(self._policy.active_scope_keys(snapshot.family_key))
            snapshot_scopes = set(snapshot.required_scope_keys)
            if current_scopes != snapshot_scopes:
                logger.warning(
                    "Training scope drift model=%s added=%s removed=%s",
                    snapshot.family_key,
                    self._scope_digests(current_scopes - snapshot_scopes),
                    self._scope_digests(snapshot_scopes - current_scopes),
                )
            if (
                self._policy.scope_reference(
                    snapshot.family_key,
                    snapshot.triggering_scope_key,
                )
                is None
            ):
                raise StaleCatalogError("triggering scope no longer belongs to the model")
            with self._provisions.hold_resources_for_family(
                snapshot.family_key
            ) as provision_resources:
                if not provision_resources:
                    self._terminal(
                        job,
                        TrainingJobState.CANCELLED,
                        "model provision demand disappeared during training",
                    )
                    return
                promoted = self._catalog.promote(
                    snapshot.family_key,
                    expected_generation=base.generation,
                    expected_artifact_key=base.artifact.key,
                    version_key=candidate_version,
                    artifact=artifact,
                )
                self._policy.begin_generation(
                    snapshot.family_key,
                    base.version_key,
                    promoted.version_key,
                    snapshot.required_scope_keys,
                )
                for resource in provision_resources:
                    self._notifications.enqueue(resource)
            with self._lock:
                job.promoted_generation = promoted.generation
            self._terminal(job, TrainingJobState.COMPLETED, "")
        except Exception as error:
            logger.exception(
                "Local training failed job_id=%s dataset_job_id=%s stage=%s",
                job.job_id,
                job.dataset_job_id,
                job.stage,
            )
            self._terminal(job, TrainingJobState.FAILED, str(error))

    def _claimed_snapshot(self, dataset_job_id: str) -> DatasetSnapshot | None:
        for dataset_job in self._datasets.jobs():
            if dataset_job.job_id == dataset_job_id:
                return dataset_job.snapshot
        return None

    def _terminal(
        self,
        job: TrainingJob,
        state: TrainingJobState,
        failure: str,
    ) -> None:
        with self._lock:
            if job.state in {
                TrainingJobState.COMPLETED,
                TrainingJobState.FAILED,
                TrainingJobState.CANCELLED,
            }:
                return
            job.state = state
            job.stage = state.value
            job.failure = failure
        self._datasets.finish_claim(
            job.dataset_job_id,
            success=state == TrainingJobState.COMPLETED,
            failure=failure,
            cancelled=state == TrainingJobState.CANCELLED,
        )
        logger.log(
            logging.INFO if state == TrainingJobState.COMPLETED else logging.WARNING,
            "Local training terminal job_id=%s state=%s failure=%s artifact=%s generation=%s",
            job.job_id,
            state,
            failure,
            job.candidate_artifact_key,
            job.promoted_generation,
        )

    def _stage(
        self,
        job: TrainingJob,
        state: TrainingJobState,
        stage: str,
    ) -> None:
        with self._lock:
            job.state = state
            job.stage = stage

    @staticmethod
    def _log_evaluation(job: TrainingJob, evaluation) -> None:
        logger.info(
            "Candidate evaluation job_id=%s accepted=%s current_wape=%s "
            "candidate_wape=%s delta=%s reasons=%s scopes=%s",
            job.job_id,
            evaluation.accepted,
            evaluation.aggregate_current.value,
            evaluation.aggregate_candidate.value,
            evaluation.aggregate_delta,
            evaluation.rejection_reasons,
            [
                {
                    "scope": item.scope_digest,
                    "triggering": item.triggering_scope,
                    "current": item.current.value,
                    "candidate": item.candidate.value,
                    "delta": item.delta,
                }
                for item in evaluation.scopes
            ],
        )

    @staticmethod
    def _scope_digests(scope_keys: set[str]) -> list[str]:
        return sorted(hashlib.sha256(scope.encode()).hexdigest() for scope in scope_keys)

    @staticmethod
    def _copy(job: TrainingJob) -> TrainingJob:
        return TrainingJob(
            job_id=job.job_id,
            dataset_job_id=job.dataset_job_id,
            family_key=job.family_key,
            state=job.state,
            stage=job.stage,
            failure=job.failure,
            candidate_artifact_key=job.candidate_artifact_key,
            promoted_generation=job.promoted_generation,
        )
