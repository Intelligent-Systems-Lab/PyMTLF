import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    state = request.app.state.runtime
    artifact_status = state.artifact_status
    if artifact_status == "ready":
        try:
            request.app.state.artifacts.probe()
        except Exception as error:
            artifact_status = "unavailable"
            logger.warning("MTLF backend artifact readiness probe failed: %s", type(error).__name__)
    generation = request.app.state.generation_monitor.snapshot()
    capability = generation.verification
    ready = state.ready and artifact_status == "ready" and generation.ready
    payload = {
        "status": "ready" if ready else "not_ready",
        "artifacts": artifact_status,
        "processInstanceId": state.process_instance_id,
        "runtimeMode": state.mode,
        "enabledFlEngines": {
            "server": capability.configured_server,
            "client": capability.configured_client,
        },
        "advertisedFlEngines": {
            "server": capability.advertised_server,
            "client": capability.advertised_client,
        },
        "capabilityVerification": capability.status,
    }
    return JSONResponse(
        payload,
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
