import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from py_mtlf.api import artifacts, health, sync
from py_mtlf.config import Settings
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.sync_projection import SyncProjection
from py_mtlf.models import PrivateError

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    process_instance_id: str
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
    runtime = RuntimeState(process_instance_id=str(uuid4()))
    sync_projection = SyncProjection()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        logger.info("MTLF backend startup begin")
        try:
            artifact_repository.open()
            runtime.artifact_status = "ready"
            runtime.accepting_requests = True
            logger.info("MTLF backend startup complete ready=%s", runtime.ready)
            yield
        finally:
            runtime.accepting_requests = False
            runtime.artifact_status = "stopped"
            logger.info("MTLF backend shutdown complete")

    app = FastAPI(title="PyMTLF", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.artifacts = artifact_repository
    app.state.sync_projection = sync_projection
    app.include_router(health.router)
    app.include_router(artifacts.router)
    app.include_router(sync.router)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request, exc
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
