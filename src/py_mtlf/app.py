import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from py_mtlf.api import artifacts, health
from py_mtlf.config import Settings
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.generation_journal import GenerationJournal
from py_mtlf.core.reconciliation import ReconciliationEngine, ReconciliationOutcome
from py_mtlf.models import PrivateError

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    database_status: str = "starting"
    artifact_status: str = "starting"
    reconciliation_status: str = "starting"
    accepting_state_changes: bool = False

    @property
    def ready(self) -> bool:
        return (
            self.database_status == "ready"
            and self.artifact_status == "ready"
            and self.reconciliation_status == "ready"
            and self.accepting_state_changes
        )


def create_app(
    settings: Settings,
    *,
    journal: GenerationJournal | None = None,
    artifact_repository: ArtifactRepository | None = None,
    reconciliation_engine: ReconciliationEngine | None = None,
) -> FastAPI:
    journal = journal or GenerationJournal(settings.storage.database_path)
    artifact_repository = artifact_repository or ArtifactRepository(
        settings.storage.artifact_root, settings.artifact
    )
    runtime = RuntimeState()

    async def run_reconciliation(engine: ReconciliationEngine) -> None:
        try:
            while True:
                pending = journal.list_pending()
                if not pending:
                    runtime.reconciliation_status = "ready"
                    runtime.accepting_state_changes = True
                    logger.info("MTLF backend reconciliation complete")
                    return
                outcomes = await asyncio.gather(*(engine.reconcile(event) for event in pending))
                if any(
                    outcome.outcome
                    in {ReconciliationOutcome.CONFLICT, ReconciliationOutcome.DIVERGED}
                    for outcome in outcomes
                ):
                    runtime.reconciliation_status = "unresolved"
                    runtime.accepting_state_changes = False
                    return
                if all(outcome.outcome == ReconciliationOutcome.APPLIED for outcome in outcomes):
                    continue
                runtime.reconciliation_status = "pending"
                await asyncio.sleep(settings.reconciliation.retry_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            runtime.reconciliation_status = "unresolved"
            runtime.accepting_state_changes = False
            logger.exception("MTLF backend reconciliation stopped after an unexpected error")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("MTLF backend startup begin")
        reconciliation_task: asyncio.Task[None] | None = None
        try:
            journal.open()
            journal.probe()
            runtime.database_status = "ready"
            artifact_repository.open()
            runtime.artifact_status = "ready"
            pending = journal.list_pending()
            if pending:
                if reconciliation_engine is None:
                    runtime.reconciliation_status = "unresolved"
                else:
                    runtime.reconciliation_status = "pending"
                    reconciliation_task = asyncio.create_task(
                        run_reconciliation(reconciliation_engine),
                        name="mtlf-startup-reconciliation",
                    )
            else:
                runtime.reconciliation_status = "ready"
            runtime.accepting_state_changes = runtime.reconciliation_status == "ready"
            logger.info("MTLF backend startup complete ready=%s", runtime.ready)
            yield
        finally:
            runtime.accepting_state_changes = False
            runtime.reconciliation_status = "stopped"
            if reconciliation_task is not None:
                reconciliation_task.cancel()
                with suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(
                        reconciliation_task,
                        timeout=settings.reconciliation.shutdown_timeout_seconds,
                    )
            journal.close()
            logger.info("MTLF backend shutdown complete")

    app = FastAPI(title="PyMTLF", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.journal = journal
    app.state.artifacts = artifact_repository
    app.include_router(health.router)
    app.include_router(artifacts.router)

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
