from unittest.mock import Mock
from uuid import UUID

from fastapi.testclient import TestClient

from py_mtlf.app import create_app
from py_mtlf.config import ModelProvisionSettings, RuntimeSettings, SeedModelSettings
from py_mtlf.core.artifacts import ArtifactRepository


def test_sync_replaces_go_owned_projection(settings):
    payload = {
        "containingNwdaf": {
            "nfInstanceId": "nwdaf-1",
            "apiBaseUri": "http://127.0.0.1:8080",
            "internalCallbackBaseUri": "http://127.0.0.1:8091",
        },
        "eventsSubscriptions": [],
        "smfResources": [],
        "trainingDataSource": "adrf",
    }
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/internal/v1/sync", json=payload)

    assert response.status_code == 200
    assert response.json()["snapshotAccepted"] is True
    assert "trainingDataSource" not in response.json()
    UUID(response.json()["processInstanceId"])
    snapshot = app.state.sync_projection.snapshot()
    assert snapshot is not None
    assert snapshot.training_data_source == "adrf"


def test_training_restart_reconciliation_runs_only_on_first_process_sync(settings):
    payload = {
        "containingNwdaf": {
            "nfInstanceId": "nwdaf-1",
            "apiBaseUri": "http://127.0.0.1:8080",
            "internalCallbackBaseUri": "http://127.0.0.1:8091",
        },
        "eventsSubscriptions": [],
        "smfResources": [],
        "trainingDataSource": "adrf",
        "mlModelTrainingSubscriptions": [
            {
                "subscriptionId": "training-1",
                "direction": "INBOUND",
                "representation": {
                    "mLEventSubscs": [
                        {
                            "mLEvent": "UE_COMMUNICATION",
                            "mLEventFilter": {},
                            "modelInterInfo": "bundle-v1",
                        }
                    ],
                    "notifUri": "http://go.internal/training/callback",
                    "notifCorreId": "prep-client-a",
                    "mlCorreId": "fl-process-001",
                    "mLPreFlag": True,
                    "eventReq": {"notifMethod": "ON_EVENT_DETECTION"},
                    "mLModelTrainInfos": [
                        {
                            "dataAvReq": {
                                "inpEvents": [{"upfEvent": "USER_DATA_USAGE_TRENDS"}],
                            },
                            "timeAvReq": "PT5M",
                        }
                    ],
                },
            }
        ],
    }
    configured = settings.model_copy(update={"runtime": RuntimeSettings(mode="fl_client")})
    app = create_app(configured)
    app.state.fl_client.restore_after_restart = Mock()

    with TestClient(app) as client:
        assert client.post("/internal/v1/sync", json=payload).status_code == 200
        assert client.post("/internal/v1/sync", json=payload).status_code == 200

    app.state.fl_client.restore_after_restart.assert_called_once()


def test_sync_restores_provision_resources_and_reconciles_seed(
    settings,
    bundle_path,
):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    seeded = settings.model_copy(
        update={
            "model_provision": ModelProvisionSettings(
                seed_models=(
                    SeedModelSettings(
                        family_id="ue-communication-default",
                        model_id=1,
                        artifact_key=artifact.key,
                        event="UE_COMMUNICATION",
                    ),
                ),
            )
        }
    )
    payload = {
        "containingNwdaf": {
            "nfInstanceId": "nwdaf-1",
            "apiBaseUri": "http://127.0.0.1:8080",
            "internalCallbackBaseUri": "http://127.0.0.1:8091",
        },
        "eventsSubscriptions": [],
        "smfResources": [],
        "trainingDataSource": "adrf",
        "mlModelProvisionSubscriptions": [
            {
                "subscriptionId": "11111111-1111-4111-8111-111111111111",
                "representation": {
                    "mLEventSubscs": [
                        {
                            "mLEvent": "UE_COMMUNICATION",
                            "mLEventFilter": {},
                        }
                    ],
                    "notifUri": "http://go.internal/provision-callback",
                    "notifCorreId": "corr-restore",
                },
                "initiator": "ANLF_BACKEND",
                "destination": "ANLF_BACKEND",
            }
        ],
    }
    app = create_app(seeded, artifact_repository=repository)
    enqueued = []
    app.state.provision_notifications.enqueue = enqueued.append
    with TestClient(app) as client:
        response = client.post("/internal/v1/sync", json=payload)

    assert response.status_code == 200
    resources = app.state.provision_store.snapshot()
    assert len(resources) == 1
    assert resources[0].subscription_id == "11111111-1111-4111-8111-111111111111"
    assert len(enqueued) == 1


def test_sync_rejects_duplicate_monitor_identity_without_partial_commit(settings):
    initial = {
        "containingNwdaf": {
            "nfInstanceId": "nwdaf-1",
            "apiBaseUri": "http://127.0.0.1:8080",
            "internalCallbackBaseUri": "http://127.0.0.1:8091",
        },
        "eventsSubscriptions": [],
        "smfResources": [],
        "trainingDataSource": "adrf",
    }
    app = create_app(settings)
    with TestClient(app) as client:
        accepted = client.post("/internal/v1/sync", json=initial)
        assert accepted.status_code == 200
        projection_before = app.state.sync_projection.snapshot()
        assert projection_before is not None

        rejected = {
            **initial,
            "containingNwdaf": {
                **initial["containingNwdaf"],
                "nfInstanceId": "nwdaf-rejected",
            },
            "mlModelMonitorRegistrations": [
                {
                    "registrationId": "registration-duplicate",
                    "representation": {
                        "consumerId": "11111111-1111-4111-8111-111111111111",
                        "modelId": model_id,
                        "mLEvent": "UE_COMMUNICATION",
                    },
                    "initiator": "ANLF_BACKEND",
                }
                for model_id in (1, 2)
            ],
        }
        response = client.post("/internal/v1/sync", json=rejected)

    assert response.status_code == 409
    projection_after = app.state.sync_projection.snapshot()
    assert projection_after is not None
    assert (
        projection_after.containing_nwdaf.nf_instance_id
        == projection_before.containing_nwdaf.nf_instance_id
    )
    assert app.state.monitor_registrations.snapshot() == ()
