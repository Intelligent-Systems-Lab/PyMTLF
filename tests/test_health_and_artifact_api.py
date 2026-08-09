from uuid import UUID

from fastapi.testclient import TestClient

from py_mtlf.app import create_app


def test_health_is_ready_after_startup(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 404
        assert client.post("/internal/v1/sync", json={}).status_code == 404
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    UUID(response.json()["processInstanceId"])


def test_app_supports_repeated_startup_and_shutdown(settings):
    app = create_app(settings)

    for _ in range(2):
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200


def test_readiness_detects_artifact_storage_failure(settings, monkeypatch):
    app = create_app(settings)
    with TestClient(app) as client:

        def fail_probe():
            raise OSError("storage unavailable")

        monkeypatch.setattr(app.state.artifacts, "probe", fail_probe)
        response = client.get("/health/ready")

    assert response.status_code == 503
    UUID(response.json()["processInstanceId"])
    assert response.json()["artifacts"] == "unavailable"
    assert "database" not in response.json()
    assert "reconciliation" not in response.json()


def test_artifact_get_has_immutable_integrity_headers(settings, bundle_path):
    app = create_app(settings)
    with TestClient(app) as client:
        metadata = app.state.artifacts.publish(bundle_path)
        response = client.get(f"/internal/v1/artifacts/{metadata.key}")

    assert response.status_code == 200
    assert response.content == bundle_path.read_bytes()
    assert response.headers["content-length"] == str(metadata.size_bytes)
    assert response.headers["content-type"] == "application/gzip"
    assert response.headers["etag"] == f'"sha256:{metadata.key}"'
    assert response.headers["x-artifact-sha256"] == metadata.key
    assert "immutable" in response.headers["cache-control"]


def test_unknown_and_malformed_artifact_keys_are_not_found(settings):
    with TestClient(create_app(settings)) as client:
        unknown = client.get("/internal/v1/artifacts/" + "a" * 64)
        malformed = client.get("/internal/v1/artifacts/not-a-digest")

    assert unknown.status_code == 404
    assert unknown.json()["code"] == "ARTIFACT_NOT_FOUND"
    assert malformed.status_code == 404
    assert malformed.json()["code"] == "ARTIFACT_NOT_FOUND"


def test_contract_only_routes_are_not_registered(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post("/internal/v1/accuracy-reports", json={}).status_code == 404
        assert client.post("/internal/v1/model-apply-results", json={}).status_code == 404
        assert client.post("/internal/v1/data-source-selection", json={}).status_code == 404
