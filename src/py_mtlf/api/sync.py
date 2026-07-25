import logging

from fastapi import APIRouter, HTTPException, Request, status

from py_mtlf.models import BackendSyncRequest, BackendSyncResponse

router = APIRouter(prefix="/internal/v1", tags=["sync"])
logger = logging.getLogger(__name__)


@router.post("/sync", response_model=BackendSyncResponse)
def sync_backend(payload: BackendSyncRequest, request: Request) -> BackendSyncResponse:
    try:
        prepared_projection = request.app.state.sync_projection.prepare(payload)
        prepared_provisions = request.app.state.provision_store.prepare_from_sync(
            payload.ml_model_provision_subscriptions
        )
        prepared_registrations = request.app.state.monitor_registrations.prepare_from_sync(
            payload.ml_model_monitor_registrations
        )
        prepared_subscriptions = request.app.state.monitor_subscriptions.prepare_from_sync(
            payload.ml_model_monitor_subscriptions
        )
        prepared_monitor_restore = request.app.state.monitor_reconciler.prepare_restore(
            tuple(prepared_registrations.values()),
            tuple(prepared_subscriptions.values()),
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
        restored_model_ids = {
            resource.representation.model_id
            for resource in prepared_registrations.values()
        }
        restored_model_ids.update(
            model_id
            for resource in prepared_subscriptions.values()
            for model_id in resource.representation.model_ids
        )
        request.app.state.model_catalog.observe_external_model_ids(restored_model_ids)
        request.app.state.sync_projection.commit(prepared_projection)
        restored = request.app.state.provision_store.commit_from_sync(prepared_provisions)
        request.app.state.monitor_registrations.commit_from_sync(prepared_registrations)
        request.app.state.monitor_subscriptions.commit_from_sync(prepared_subscriptions)
        request.app.state.monitor_reconciler.commit_restore(prepared_monitor_restore)
        for resource in prepared_registrations.values():
            request.app.state.accuracy_policy.record_registration(
                resource.representation
            )

    for resource in restored:
        request.app.state.provision_notifications.enqueue(resource)
    request.app.state.monitor_reconciler.finalize_restore()
    logger.info(
        "Backend snapshot accepted training_data_source=%s",
        payload.training_data_source,
    )
    return BackendSyncResponse(
        processInstanceId=request.app.state.runtime.process_instance_id,
        snapshotAccepted=True,
    )
