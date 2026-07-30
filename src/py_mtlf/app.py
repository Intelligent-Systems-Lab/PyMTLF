import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from py_mtlf.api import (
    adrf,
    artifacts,
    health,
    ml_model_monitor,
    ml_model_provision,
    ml_model_training,
    sync,
)
from py_mtlf.config import Settings
from py_mtlf.core.accuracy_policy import AccuracyPolicy
from py_mtlf.core.adrf_discovery import AdrfResolver
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.dataset import DatasetCoordinator
from py_mtlf.core.fl_client import FLClientService
from py_mtlf.core.fl_server import FLClientResolver, FLServerOrchestrator
from py_mtlf.core.fl_workspace import FLWorkspace
from py_mtlf.core.monitor_reconciler import MonitorSubscriptionReconciler
from py_mtlf.core.monitor_store import (
    MonitorRegistrationStore,
    MonitorSubscriptionProjectionStore,
)
from py_mtlf.core.notification_delivery import ProvisionNotificationDispatcher
from py_mtlf.core.nwdaf_discovery import NwdafMonitorResolver
from py_mtlf.core.provision_store import ProvisionResourceStore
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.core.sync_projection import SyncProjection
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
    sync_projection = SyncProjection(state_lock)
    monitor_registrations = MonitorRegistrationStore(state_lock)
    monitor_subscriptions = MonitorSubscriptionProjectionStore(state_lock)
    nwdaf_monitor_resolver = NwdafMonitorResolver(
        settings.model_monitor,
        sync_projection,
    )
    monitor_reconciler = MonitorSubscriptionReconciler(
        settings.model_monitor,
        sync_projection,
        monitor_registrations,
        monitor_subscriptions,
        nwdaf_monitor_resolver,
        state_lock,
    )
    adrf_resolver = AdrfResolver(settings.adrf, sync_projection)
    dataset_coordinator = DatasetCoordinator(
        settings.dataset,
        sync_projection,
        accuracy_policy,
        adrf_resolver,
    )
    training_coordinator = TrainingCoordinator(
        settings.training,
        dataset_coordinator,
        seed_catalog,
        artifact_repository,
        provision_store,
        provision_notifications,
        accuracy_policy,
    )
    fl_workspace = FLWorkspace(settings.federated_learning, settings.artifact)
    fl_client_resolver = FLClientResolver(
        settings.federated_learning,
        sync_projection,
    )
    fl_client = FLClientService(
        settings.federated_learning,
        settings.notification,
        settings.training,
        sync_projection,
        dataset_coordinator,
        fl_workspace,
    )
    fl_server = FLServerOrchestrator(
        settings.federated_learning,
        sync_projection,
        accuracy_policy,
        seed_catalog,
        fl_workspace,
        fl_client_resolver,
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
            if settings.runtime.mode in {"local", "fl_server"}:
                provision_notifications.open()
                monitor_reconciler.open()
            if settings.runtime.mode == "local":
                training_coordinator.open()
            runtime.artifact_status = "ready"
            runtime.accepting_requests = True
            logger.info("MTLF backend startup complete ready=%s", runtime.ready)
            yield
        finally:
            if settings.runtime.mode == "local":
                training_coordinator.shutdown()
            fl_client.close()
            fl_server.close()
            fl_workspace.close()
            dataset_coordinator.shutdown()
            if settings.runtime.mode in {"local", "fl_server"}:
                monitor_reconciler.shutdown()
                provision_notifications.shutdown()
            nwdaf_monitor_resolver.close()
            runtime.accepting_requests = False
            runtime.artifact_status = "stopped"
            logger.info("MTLF backend shutdown complete")

    app = FastAPI(title="PyMTLF", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.artifacts = artifact_repository
    app.state.seed_catalog = seed_catalog
    app.state.model_catalog = seed_catalog
    app.state.provision_store = provision_store
    app.state.provision_notifications = provision_notifications
    app.state.sync_projection = sync_projection
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
    app.state.state_lock = state_lock
    app.include_router(health.router)
    app.include_router(artifacts.router)
    app.include_router(sync.router)
    app.include_router(adrf.router)
    if settings.runtime.mode in {"local", "fl_server"}:
        app.include_router(ml_model_provision.router)
        app.include_router(ml_model_monitor.router)
    if settings.runtime.mode in {"fl_client", "fl_server"}:
        app.include_router(ml_model_training.router)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        if request.url.path.startswith(
            (
                "/internal/v1/ml-model-",
                "/internal/v1/adrf-data-management/",
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
