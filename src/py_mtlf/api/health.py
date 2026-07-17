import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    state = request.app.state.runtime
    database_status = state.database_status
    artifact_status = state.artifact_status
    if database_status == "ready":
        try:
            request.app.state.journal.probe()
        except Exception as error:
            database_status = "unavailable"
            logger.warning("MTLF backend database readiness probe failed: %s", type(error).__name__)
    if artifact_status == "ready":
        try:
            request.app.state.artifacts.probe()
        except Exception as error:
            artifact_status = "unavailable"
            logger.warning("MTLF backend artifact readiness probe failed: %s", type(error).__name__)
    ready = state.ready and database_status == "ready" and artifact_status == "ready"
    payload = {
        "status": "ready" if ready else "not_ready",
        "database": database_status,
        "artifacts": artifact_status,
        "reconciliation": state.reconciliation_status,
    }
    return JSONResponse(
        payload,
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
