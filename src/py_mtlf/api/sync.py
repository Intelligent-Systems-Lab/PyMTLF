from fastapi import APIRouter, HTTPException, Request, status

from py_mtlf.models import BackendSyncRequest, BackendSyncResponse, DataSourceSelection

router = APIRouter(prefix="/internal/v1", tags=["sync"])


@router.post("/sync", response_model=BackendSyncResponse)
def sync_backend(payload: BackendSyncRequest, request: Request) -> BackendSyncResponse:
    try:
        prepared_projection = request.app.state.sync_projection.prepare(payload)
        prepared_provisions = request.app.state.provision_store.prepare_from_sync(
            payload.ml_model_provision_subscriptions
        )
        prepared_registrations = (
            request.app.state.monitor_registrations.prepare_from_sync(
                payload.ml_model_monitor_registrations
            )
        )
        prepared_subscriptions = (
            request.app.state.monitor_subscriptions.prepare_from_sync(
                payload.ml_model_monitor_subscriptions
            )
        )
        prepared_monitor_restore = (
            request.app.state.monitor_reconciler.prepare_restore(
                tuple(prepared_registrations.values()),
                tuple(prepared_subscriptions.values()),
            )
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SNAPSHOT_RECONCILIATION_FAILED",
        ) from error

    with request.app.state.state_lock:
        request.app.state.sync_projection.commit(prepared_projection)
        restored = request.app.state.provision_store.commit_from_sync(
            prepared_provisions
        )
        request.app.state.monitor_registrations.commit_from_sync(
            prepared_registrations
        )
        request.app.state.monitor_subscriptions.commit_from_sync(
            prepared_subscriptions
        )
        request.app.state.monitor_reconciler.commit_restore(
            prepared_monitor_restore
        )

    for resource in restored:
        request.app.state.provision_notifications.enqueue(resource)
    request.app.state.monitor_reconciler.finalize_restore()
    return BackendSyncResponse(
        processInstanceId=request.app.state.runtime.process_instance_id,
        snapshotAccepted=True,
        mongodbAvailable=False,
        sourceSelection=DataSourceSelection(),
    )
