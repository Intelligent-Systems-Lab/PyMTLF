import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from py_mtlf.api import (
    adrf,
    artifacts,
    health,
    hierarchical_fl,
    ml_model_monitor,
    ml_model_provision,
    ml_model_training,
    training_data,
)
from py_mtlf.config import Settings
from py_mtlf.core.accuracy_policy import AccuracyPolicy
from py_mtlf.core.adrf_discovery import AdrfResolver
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.dataset import DatasetCoordinator
from py_mtlf.core.fl_branch import FLBranchPreparationCoordinator
from py_mtlf.core.fl_client import FLClientEngine
from py_mtlf.core.fl_experiment import FLExperimentRegistry
from py_mtlf.core.fl_hierarchy_artifacts import HierarchyArtifactService
from py_mtlf.core.fl_hierarchy_discovery import HierarchyNodeResolver
from py_mtlf.core.fl_root import FLRootCoordinator
from py_mtlf.core.fl_server import FLClientResolver, FLServerEngine
from py_mtlf.core.fl_topology import StaticTopologyPlanner
from py_mtlf.core.fl_workspace import FLWorkspace
from py_mtlf.core.model_records import (
    CompletedRevision,
    DurableModelState,
    DurableModelStateRepository,
    ModelCatalogRecord,
    RevisionOrigin,
)
from py_mtlf.core.monitor_reconciler import MonitorSubscriptionReconciler
from py_mtlf.core.monitor_store import (
    MonitorRegistrationStore,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.core.notification_delivery import ProvisionNotificationDispatcher
from py_mtlf.core.nwdaf_context import CapabilityConsistencyChecker, NwdafContextClient
from py_mtlf.core.nwdaf_discovery import NwdafMonitorResolver
from py_mtlf.core.provision_store import ProvisionResourceStore
from py_mtlf.core.publication import PublicationCoordinator
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.core.training_jobs import TrainingCoordinator
from py_mtlf.models import PrivateError

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    process_instance_id: str
    mode: str
    artifact_status: str = "starting"
    accepting_requests: bool = False

    @property
    def ready(self) -> bool:
        return self.artifact_status == "ready" and self.accepting_requests


def create_app(
    settings: Settings,
    *,
    artifact_repository: ArtifactRepository | None = None,
    capability_checker: CapabilityConsistencyChecker | None = None,
) -> FastAPI:
    artifact_repository = artifact_repository or ArtifactRepository(
        settings.storage.artifact_root, settings.artifact
    )
    state_lock = threading.RLock()
    seed_catalog = SeedCatalog(
        settings.model_provision,
        artifact_repository,
        state_lock,
    )
    model_state = DurableModelStateRepository(settings.model_state.directory, state_lock)
    accuracy_policy = AccuracyPolicy(
        settings.accuracy_policy,
        seed_catalog,
    )
    provision_store = ProvisionResourceStore(seed_catalog, state_lock)
    provision_notifications = ProvisionNotificationDispatcher(
        settings.notification,
        provision_store,
        seed_catalog,
    )
    runtime = RuntimeState(process_instance_id=str(uuid4()), mode=settings.runtime.mode)
    nwdaf_context = NwdafContextClient(
        settings.containing_nwdaf.internal_api_root,
        settings.containing_nwdaf.request_timeout_seconds,
    )
    capability_checker = capability_checker or CapabilityConsistencyChecker(
        nwdaf_context,
        configured_server=settings.federated_learning.server is not None,
        configured_client=settings.federated_learning.client is not None,
    )
    monitor_registrations = MonitorRegistrationStore(state_lock)
    monitor_subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    adrf_resolver = AdrfResolver(settings.adrf, nwdaf_context)
    fl_workspace = FLWorkspace(settings.federated_learning, settings.artifact)
    local_training = settings.local_training
    fl_client_settings = settings.federated_learning.client
    fl_server_settings = settings.federated_learning.server
    fl_experiments = FLExperimentRegistry()

    def resume_published_cutover(publication_record, model) -> None:
        family_key = seed_catalog.family_key_for_id(publication_record.family_id)
        if publication_record.previous_model_id is not None:
            accuracy_policy.restore_generation(
                family_key,
                seed_catalog.version_key_for_id(publication_record.previous_model_id),
                model.version_key,
                publication_record.required_cutover_scopes,
            )
        provision_notifications.reconcile_family(family_key)
        if not publication_record.required_cutover_scopes:
            accuracy_policy.complete_retrain(family_key)

    publication = PublicationCoordinator(
        settings.publication,
        model_state,
        seed_catalog,
        artifact_repository,
        fl_workspace,
        adrf_resolver,
        nwdaf_context,
        on_published=resume_published_cutover,
    )
    fl_server_holder: dict[str, FLServerEngine] = {}

    def monitor_subscription_created(registration) -> None:
        family_key = seed_catalog.family_for_version(
            seed_catalog.version_key_for_id(registration.model_id)
        )
        server = fl_server_holder.get("server")
        if family_key is not None and server is not None:
            server.mark_scope_adopted(
                family_key,
                registration.model_id,
                AccuracyPolicy.registration_scope_key(registration),
            )

    def monitor_subscription_timed_out(registration) -> None:
        accuracy_policy.remove_registration(registration)

    nwdaf_monitor_resolver = NwdafMonitorResolver(
        settings.model_monitor,
        nwdaf_context,
    )
    monitor_reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        nwdaf_context,
        monitor_registrations,
        monitor_subscriptions,
        nwdaf_monitor_resolver,
        state_lock,
        on_subscription_created=monitor_subscription_created,
        on_subscription_timeout=monitor_subscription_timed_out,
    )
    dataset_coordinator = DatasetCoordinator(
        settings.dataset,
        nwdaf_context,
        accuracy_policy,
        adrf_resolver,
    )
    training_coordinator = (
        TrainingCoordinator(
            local_training,
            dataset_coordinator,
            seed_catalog,
            artifact_repository,
            provision_store,
            provision_notifications,
            accuracy_policy,
        )
        if local_training is not None
        else None
    )
    fl_client_resolver = (
        FLClientResolver(
            settings.federated_learning,
            nwdaf_context,
        )
        if fl_server_settings is not None
        else None
    )
    fl_server = (
        FLServerEngine(
            settings.federated_learning,
            fl_server_settings,
            nwdaf_context,
            accuracy_policy,
            seed_catalog,
            fl_workspace,
            fl_client_resolver,
            publication=publication,
            provision_notifications=provision_notifications,
            experiments=fl_experiments,
        )
        if fl_server_settings is not None and fl_client_resolver is not None
        else None
    )
    if fl_server is not None:
        fl_server_holder["server"] = fl_server
    fl_branch = None
    if fl_client_settings is not None and fl_server is not None:
        fl_branch = FLBranchPreparationCoordinator(
            resolver=HierarchyNodeResolver(
                settings.federated_learning,
                nwdaf_context,
            ),
            nwdaf_context=nwdaf_context,
            artifact_service=HierarchyArtifactService(fl_workspace),
            server=fl_server,
        )
    fl_client = (
        FLClientEngine(
            settings.federated_learning,
            fl_client_settings,
            settings.notification,
            nwdaf_context,
            dataset_coordinator,
            fl_workspace,
            experiments=fl_experiments,
            branch_coordinator=fl_branch,
        )
        if fl_client_settings is not None
        else None
    )
    topology_settings = settings.federated_learning.topology
    strategy_settings = settings.federated_learning.strategy
    fl_root = None
    if topology_settings is not None:
        if fl_server is None or strategy_settings is None:
            raise RuntimeError("validated hierarchy configuration is incomplete")
        topology_planner = StaticTopologyPlanner.load(topology_settings.config_file)
        hierarchy_resolver = HierarchyNodeResolver(
            settings.federated_learning,
            nwdaf_context,
        )
        fl_root = FLRootCoordinator(
            strategy=strategy_settings,
            server_settings=fl_server_settings,
            planner=topology_planner,
            resolver=hierarchy_resolver,
            nwdaf_context=nwdaf_context,
            catalog=seed_catalog,
            artifact_service=HierarchyArtifactService(fl_workspace),
            workspace=fl_workspace,
            server=fl_server,
            policy=accuracy_policy,
            experiments=fl_experiments,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        logger.info("MTLF backend startup begin mode=%s", settings.runtime.mode)
        try:
            _prepare_workspace(settings.federated_learning.workspace_root)
            fl_workspace.open()
            artifact_repository.open()
            seed_catalog.open()
            durable_state = model_state.open(_initial_model_state(seed_catalog))
            seed_catalog.restore(durable_state)
            publication.open()
            if settings.runtime.mode == "local" or fl_server is not None:
                provision_notifications.open()
                monitor_reconciler.open()
            if training_coordinator is not None:
                training_coordinator.open()
            runtime.artifact_status = "ready"
            runtime.accepting_requests = True
            logger.info("MTLF backend startup complete ready=%s", runtime.ready)
            yield
        finally:
            runtime.accepting_requests = False
            if fl_root is not None:
                fl_root.close()
            if fl_branch is not None:
                fl_branch.close()
            fl_experiments.shutdown()
            if training_coordinator is not None:
                training_coordinator.shutdown()
            if fl_client is not None:
                fl_client.close()
            if fl_server is not None:
                fl_server.close()
            publication.close()
            fl_workspace.close()
            dataset_coordinator.shutdown()
            adrf_resolver.close()
            if settings.runtime.mode == "local" or fl_server is not None:
                monitor_reconciler.shutdown()
                provision_notifications.shutdown()
            nwdaf_monitor_resolver.close()
            nwdaf_context.close()
            runtime.artifact_status = "stopped"
            logger.info("MTLF backend shutdown complete")

    app = FastAPI(title="PyMTLF", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.artifacts = artifact_repository
    app.state.seed_catalog = seed_catalog
    app.state.model_catalog = seed_catalog
    app.state.model_state = model_state
    app.state.provision_store = provision_store
    app.state.provision_notifications = provision_notifications
    app.state.nwdaf_context = nwdaf_context
    app.state.capability_checker = capability_checker
    app.state.monitor_registrations = monitor_registrations
    app.state.monitor_subscriptions = monitor_subscriptions
    app.state.monitor_reconciler = monitor_reconciler
    app.state.nwdaf_monitor_resolver = nwdaf_monitor_resolver
    app.state.accuracy_policy = accuracy_policy
    app.state.dataset_coordinator = dataset_coordinator
    app.state.training_coordinator = training_coordinator
    app.state.fl_workspace = fl_workspace
    app.state.fl_client = fl_client
    app.state.fl_server = fl_server
    app.state.fl_branch = fl_branch
    app.state.fl_root = fl_root
    app.state.fl_experiments = fl_experiments
    app.state.publication = publication
    app.state.state_lock = state_lock
    app.include_router(health.router)
    app.include_router(artifacts.router)
    app.include_router(training_data.router)
    app.include_router(adrf.router)
    if settings.runtime.mode == "local" or fl_server is not None:
        app.include_router(ml_model_provision.router)
        app.include_router(ml_model_monitor.router)
    if fl_client is not None or fl_server is not None:
        app.include_router(ml_model_training.router)
    if (
        fl_root is not None
        and settings.federated_learning.training_trigger.private_api.enabled
    ):
        app.include_router(hierarchical_fl.router)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        if request.url.path.startswith(
            (
                "/internal/v1/ml-model-",
                "/internal/v1/adrf-data-management/",
                "/internal/v1/hierarchical-fl/",
            )
        ):
            from py_mtlf.api.problems import problem_response

            return problem_response(
                status.HTTP_400_BAD_REQUEST,
                "Bad Request",
                "request validation failed",
                cause="INVALID_MSG_FORMAT",
            )
        del exc
        payload = PrivateError(
            code="VALIDATION_ERROR",
            message="request validation failed",
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(mode="json"),
        )

    return app


def _prepare_workspace(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".write-probe-{uuid4()}"
    try:
        with probe.open("x", encoding="utf-8") as stream:
            stream.write("ready")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise RuntimeError(f"federated learning workspace is not writable: {root}") from error
    finally:
        probe.unlink(missing_ok=True)


def _initial_model_state(catalog: SeedCatalog) -> DurableModelState:
    families: dict[str, ModelCatalogRecord] = {}
    allocated = 0
    for model in catalog.snapshot():
        manifest = catalog.artifact_manifest(model.artifact.key)
        created_value = manifest.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created_value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            created_at = datetime.now(UTC)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        revision = CompletedRevision(
            modelUniqueId=model.model_id,
            origin=RevisionOrigin.SEED,
            artifactKey=model.artifact.key,
            artifactDigest=model.artifact.key,
            createdAt=created_at,
            generation=model.generation,
        )
        families[model.descriptor.family_id] = ModelCatalogRecord(
            schemaVersion="1.0",
            latestModelId=model.model_id,
            nextModelId=model.model_id + 1,
            revisions=(revision,),
        )
        allocated = max(allocated, model.model_id)
    return DurableModelState(
        schemaVersion="2.0",
        lastAllocatedModelId=allocated,
        families=families,
    )
