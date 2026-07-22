from uuid import UUID

from fastapi.testclient import TestClient

from py_mtlf.app import create_app


def test_sync_replaces_go_owned_projection(settings):
    payload = {
        "containingNwdaf": {
            "nfInstanceId": "nwdaf-1",
            "apiBaseUri": "http://127.0.0.1:8080",
            "internalCallbackBaseUri": "http://127.0.0.1:8090",
        },
        "eventsSubscriptions": [],
        "smfResources": [],
        "dataSourceAvailability": {"adrf": True, "mongodb": False},
        "mtlfSourceSelection": {"preferredSource": "", "effectiveSource": ""},
    }
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/internal/v1/sync", json=payload)

    assert response.status_code == 200
    assert response.json()["snapshotAccepted"] is True
    assert response.json()["sourceSelection"] == {
        "preferredSource": "",
        "effectiveSource": "",
    }
    UUID(response.json()["processInstanceId"])
    snapshot = app.state.sync_projection.snapshot()
    assert snapshot is not None
    assert snapshot.data_source_availability.adrf is True
