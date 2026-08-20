import hashlib
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from nwdaf_context import context_client, verified_capability_checker

from py_mtlf.app import create_app
from py_mtlf.core.fl_workspace import FLWorkspaceError
from py_mtlf.core.nwdaf_context import CapabilityVerification


def test_health_is_ready_after_startup(settings):
    with TestClient(
        create_app(
            settings,
            capability_checker=verified_capability_checker(),
            nwdaf_context_client=context_client(),
        )
    ) as client:
        assert client.get("/health/live").status_code == 404
        assert client.post("/internal/v1/sync", json={}).status_code == 404
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    UUID(response.json()["processInstanceId"])


def test_app_supports_repeated_startup_and_shutdown(settings):
    app = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )

    for _ in range(2):
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200


def test_new_app_lifetime_uses_fresh_backend_process_instance_id(settings):
    first = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )
    with TestClient(first) as client:
        first_id = client.get("/health/ready").json()["processInstanceId"]

    second = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )
    with TestClient(second) as client:
        second_id = client.get("/health/ready").json()["processInstanceId"]

    assert first_id != second_id


def test_workspace_startup_cleanup_failure_prevents_app_startup(settings):
    stale = settings.federated_learning.workspace_root / "stale" / "artifact"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    app = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )
    app.state.fl_workspace._delete_direct_child = Mock(side_effect=OSError("busy"))

    with (
        pytest.raises(FLWorkspaceError, match="startup cleanup failed"),
        TestClient(app),
    ):
        pass


def test_readiness_detects_artifact_storage_failure(settings, monkeypatch):
    app = create_app(
        settings,
        capability_checker=verified_capability_checker(),
        nwdaf_context_client=context_client(),
    )
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


def test_readiness_recovers_after_capability_unavailable_and_mismatch(settings):
    checker = SequenceCapabilityChecker(
        [
            CapabilityVerification(
                status="unavailable",
                configured_server=False,
                configured_client=False,
                advertised_server=None,
                advertised_client=None,
            ),
            CapabilityVerification(
                status="mismatch",
                configured_server=False,
                configured_client=False,
                advertised_server=True,
                advertised_client=False,
            ),
            CapabilityVerification(
                status="verified",
                configured_server=False,
                configured_client=False,
                advertised_server=False,
                advertised_client=False,
            ),
        ]
    )
    app = create_app(
        settings,
        capability_checker=checker,
        nwdaf_context_client=context_client(),
    )

    with TestClient(app) as client:
        unavailable = client.get("/health/ready")
        app.state.generation_monitor.refresh_once()
        mismatch = client.get("/health/ready")
        app.state.generation_monitor.refresh_once()
        recovered = client.get("/health/ready")

    assert unavailable.status_code == 503
    assert unavailable.json()["capabilityVerification"] == "unavailable"
    assert mismatch.status_code == 503
    assert mismatch.json()["capabilityVerification"] == "mismatch"
    assert recovered.status_code == 200
    assert recovered.json()["capabilityVerification"] == "verified"
    assert recovered.json()["enabledFlEngines"] == {"server": False, "client": False}
    assert recovered.json()["advertisedFlEngines"] == {"server": False, "client": False}


class SequenceCapabilityChecker:
    def __init__(self, verifications):
        self.verifications = list(verifications)

    def check(self):
        return self.verifications.pop(0)


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


def test_hierarchy_fl_artifact_uses_existing_serving_route(settings, bundle_path):
    content = bundle_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    process_id = "11111111-1111-4111-8111-111111111111"
    participant_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    role = "HIERARCHY_ASSIGNMENT"
    path = (
        settings.federated_learning.workspace_root
        / process_id
        / participant_id
        / "0"
        / role
        / f"{digest}.tar.gz"
    )
    with TestClient(create_app(settings)) as client:
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        response = client.get(
            f"/internal/v1/fl-artifacts/{process_id}/{participant_id}/0/{role}/{digest}"
        )

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["x-artifact-sha256"] == digest
    assert response.headers["etag"] == f'"sha256:{digest}"'


def test_contract_only_routes_are_not_registered(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post("/internal/v1/accuracy-reports", json={}).status_code == 404
        assert client.post("/internal/v1/model-apply-results", json={}).status_code == 404
        assert client.post("/internal/v1/data-source-selection", json={}).status_code == 404
