from fastapi import APIRouter, Request

from py_mtlf.models import BackendSyncRequest, BackendSyncResponse, DataSourceSelection

router = APIRouter(prefix="/internal/v1", tags=["sync"])


@router.post("/sync", response_model=BackendSyncResponse)
def sync_backend(payload: BackendSyncRequest, request: Request) -> BackendSyncResponse:
    request.app.state.sync_projection.replace(payload)
    return BackendSyncResponse(
        processInstanceId=request.app.state.runtime.process_instance_id,
        snapshotAccepted=True,
        mongodbAvailable=False,
        sourceSelection=DataSourceSelection(),
    )
