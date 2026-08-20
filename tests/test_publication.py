import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from nwdaf_context import context_client

from py_mtlf.config import PublicationSettings
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    HierarchyBranchValidation,
    HierarchyValidation,
    ValidationSummary,
    WapeComponents,
)
from py_mtlf.core.model_records import (
    CatalogValidationSummary,
    ParticipantSampleCount,
    PendingPublication,
    PublicationState,
)
from py_mtlf.core.publication import (
    PublicationAbandonedError,
    PublicationCoordinator,
    RecoverablePublicationError,
)
from py_mtlf.wire.private import SelectedTarget

NWDAF_ID = "11111111-1111-4111-8111-111111111111"
ADRF_ID = "22222222-2222-4222-8222-222222222222"
CLIENT_ID = "33333333-3333-4333-8333-333333333333"
LEAF_ID = "44444444-4444-4444-8444-444444444444"
DIGEST = "a" * 64


def _capture_error(target, operation):
    try:
        operation()
    except Exception as error:
        target.append(error)


def nwdaf_context_client():
    return context_client(
        nf_instance_id=NWDAF_ID,
        api_root="http://nwdaf-c.example",
        internal_api_root="http://go-c.example",
    )


def pending_publication() -> PendingPublication:
    start = datetime(2026, 7, 31, tzinfo=UTC)
    return PendingPublication(
        schemaVersion="1.0",
        publicationId="publication-a",
        state=PublicationState.FINAL_BUNDLE_READY,
        mlCorreId="process-a",
        reservedModelId=202607310001,
        previousModelId=1,
        familyId="ue-communication-default",
        expectedGeneration=1,
        expectedArtifactDigest="b" * 64,
        participantsAndSampleCounts=[
            ParticipantSampleCount(
                participantNfInstanceId=CLIENT_ID,
                sampleCount=100,
            )
        ],
        validationSummary=CatalogValidationSummary(globalGateAccepted=True),
        validationEvidence=[
            ValidationSummary(
                participant_nf_instance_id=CLIENT_ID,
                scope_digest="c" * 64,
                evaluation_sample_count=20,
                start_time=start,
                end_time=start + timedelta(minutes=1),
                base_model_weights_digest="d" * 64,
                candidate_weights_digest="e" * 64,
                base=WapeComponents(
                    absolute_error_sum=10,
                    absolute_actual_sum=100,
                ),
                candidate=WapeComponents(
                    absolute_error_sum=5,
                    absolute_actual_sum=100,
                ),
            )
        ],
        candidatePath="/tmp/candidate.tar.gz",
        candidateDigest="f" * 64,
        finalBundlePath="/tmp/final.tar.gz",
        finalBundleDigest=DIGEST,
        requiredCutoverScopes=["scope-a"],
        updatedAt=start,
    )


def test_store_in_adrf_uses_go_proxy_and_persists_exact_reference():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(204, request=request)
        return httpx.Response(
            201,
            request=request,
            headers={
                "Location": (
                    "http://adrf.example/nadrf-mlmodelmanagement/v1/mlmodel-store-records/store-a"
                )
            },
            json={
                "nfInstanceId": NWDAF_ID,
                "mlModelInfo": [
                    {
                        "modelUniqueId": 202607310001,
                        "mlFileAddr": {
                            "mLModelUrl": (
                                "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                                "mlmodel-store-records/store-a/model"
                            )
                        },
                        "mlStorageSize": 4096,
                        "allowConsumerList": [
                            {"nfInstanceId": NWDAF_ID},
                            {"nfInstanceId": CLIENT_ID},
                        ],
                    }
                ],
                "modelStoreResult": {
                    "modelUniqueId": 202607310001,
                    "storeResult": "ML_MODEL_FILE_STORED_IN_ADRF",
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    artifacts = Mock()
    artifacts.metadata.return_value = ArtifactMetadata(
        key=DIGEST,
        size_bytes=4096,
        path=Path("/tmp/final.tar.gz"),
        url=f"http://py-mtlf.example/internal/v1/artifacts/{DIGEST}",
    )
    resolver = Mock(
        resolve_model=Mock(
            return_value=SelectedTarget(
                nfInstanceId=ADRF_ID,
                nfServiceInstanceId="adrf-model",
                serviceName="nadrf-mlmodelmanagement",
                apiRoot="http://adrf.example",
                selectionSource="NRF",
            )
        )
    )
    coordinator = PublicationCoordinator(
        PublicationSettings(),
        Mock(),
        Mock(),
        artifacts,
        Mock(),
        resolver,
        nwdaf_context_client(),
        client,
    )
    coordinator._replace_publication = lambda value: value

    stored = coordinator._store_in_adrf(pending_publication())

    assert stored.state is PublicationState.STORE_ACCEPTED
    assert stored.selected_adrf_instance_id == ADRF_ID
    assert stored.store_trans_id == "store-a"
    assert requests[0].url.params["model-unique-ids"] == "202607310001"
    assert requests[1].headers["Target-Api-Root"] == "http://adrf.example"
    request_body = json.loads(requests[1].content)
    assert request_body["mlModelInfo"][0]["modelUniqueId"] == 202607310001
    assert request_body["mlModelInfo"][0]["allowConsumerList"] == [
        {"nfInstanceId": NWDAF_ID},
        {"nfInstanceId": CLIENT_ID},
    ]
    client.close()


def test_store_in_flight_probes_existing_record_without_reposting():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "nfInstanceId": NWDAF_ID,
                "mlModelInfo": [
                    {
                        "modelUniqueId": 202607310001,
                        "mlFileAddr": {
                            "mLModelUrl": (
                                "http://adrf.example/nadrf-mlmodelmanagement/v1/"
                                "mlmodel-store-records/store-a/model"
                            )
                        },
                        "mlStorageSize": 4096,
                        "allowConsumerList": [
                            {"nfInstanceId": NWDAF_ID},
                            {"nfInstanceId": CLIENT_ID},
                        ],
                    }
                ],
                "modelStoreResult": {
                    "modelUniqueId": 202607310001,
                    "storeResult": "ML_MODEL_FILE_STORED_IN_ADRF",
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    artifacts = Mock()
    artifacts.metadata.return_value = ArtifactMetadata(
        key=DIGEST,
        size_bytes=4096,
        path=Path("/tmp/final.tar.gz"),
        url=f"http://py-mtlf.example/internal/v1/artifacts/{DIGEST}",
    )
    coordinator = PublicationCoordinator(
        PublicationSettings(),
        Mock(),
        Mock(),
        artifacts,
        Mock(),
        Mock(),
        nwdaf_context_client(),
        client,
    )
    coordinator._replace_publication = lambda value: value
    in_flight = pending_publication().model_copy(
        update={
            "state": PublicationState.STORE_IN_FLIGHT,
            "selected_adrf_target": "http://adrf.example",
            "selected_adrf_instance_id": ADRF_ID,
        }
    )

    stored = coordinator._store_in_adrf(in_flight)

    assert stored.state is PublicationState.STORE_ACCEPTED
    assert [request.method for request in requests] == ["GET"]
    assert stored.store_trans_id == "store-a"
    client.close()


def test_restart_reannounces_cutover_pending_publication():
    cutover = pending_publication().model_copy(
        update={
            "state": PublicationState.CUTOVER_PENDING,
            "selected_adrf_target": "http://adrf.example",
            "selected_adrf_instance_id": ADRF_ID,
            "store_trans_id": "store-a",
            "resource_location": (
                "http://adrf.example/nadrf-mlmodelmanagement/v1/mlmodel-store-records/store-a"
            ),
        }
    )
    state = Mock()
    state.snapshot.return_value = SimpleNamespace(pending_publications=(cutover,))
    catalog = Mock()
    catalog.family_key_for_id.return_value = cutover.family_id
    model = Mock(model_id=cutover.reserved_model_id)
    catalog.current.return_value = model
    announced = threading.Event()
    coordinator = PublicationCoordinator(
        PublicationSettings(
            retry_interval_seconds=0.01,
            retry_max_interval_seconds=0.02,
        ),
        state,
        catalog,
        Mock(),
        Mock(),
        Mock(),
        nwdaf_context_client(),
        httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
        on_published=lambda publication, current: announced.set(),
    )

    coordinator.open()
    try:
        assert coordinator._worker is None
        coordinator.resume()
        assert announced.wait(timeout=1)
    finally:
        coordinator.close()


def test_production_open_does_not_resume_or_rewrite_unfinished_journal(tmp_path):
    state = Mock()
    coordinator = PublicationCoordinator(
        PublicationSettings(directory=tmp_path / "publications"),
        state,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        nwdaf_context_client(),
        Mock(),
    )

    coordinator.open()
    try:
        assert coordinator._worker is None
        assert coordinator._recovery_enabled is False
        state.snapshot.assert_not_called()
        state.update.assert_not_called()
    finally:
        coordinator.close()


def test_generation_reset_wakes_retry_and_abandons_without_terminal_rewrite():
    coordinator = PublicationCoordinator(
        PublicationSettings(
            retry_interval_seconds=30,
            retry_max_interval_seconds=30,
        ),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        nwdaf_context_client(),
        Mock(),
    )
    publication = pending_publication()
    attempted = threading.Event()
    coordinator._reserve = Mock(return_value=publication)

    def fail_recoverably(_publication, *, generation):
        del generation
        attempted.set()
        raise RecoverablePublicationError("ADRF unavailable")

    coordinator._advance = Mock(side_effect=fail_recoverably)
    coordinator._record_retry = Mock(return_value=publication)
    coordinator._mark_terminal = Mock()
    failures = []

    thread = threading.Thread(
        target=lambda: _capture_error(
            failures,
            lambda: coordinator.publish(Mock()),
        )
    )
    thread.start()
    assert attempted.wait(timeout=1)

    coordinator.abort_generation()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], PublicationAbandonedError)
    coordinator._mark_terminal.assert_not_called()


def test_final_bundle_receives_durable_hierarchy_validation(
    tmp_path,
    monkeypatch,
):
    publication = pending_publication().model_copy(
        update={
            "state": PublicationState.RESERVED,
            "final_bundle_path": None,
            "final_bundle_digest": None,
            "hierarchy_validation": HierarchyValidation(
                plan_id="55555555-5555-4555-8555-555555555555",
                branches=(
                    HierarchyBranchValidation(
                        branch_nf_instance_id=CLIENT_ID,
                        subordinate_validation_summaries=(
                            pending_publication()
                            .validation_evidence[0]
                            .model_copy(
                                update={
                                    "participant_nf_instance_id": LEAF_ID,
                                }
                            ),
                        ),
                    ),
                ),
            ),
        }
    )
    candidate_path = tmp_path / "candidate.tar.gz"
    candidate_path.write_bytes(b"candidate")
    publication = PendingPublication.model_validate(
        publication.model_copy(
            update={"candidate_path": str(candidate_path)}
        ).model_dump(by_alias=True, mode="json")
    )
    base_model = SimpleNamespace(digest="d" * 64)
    candidate_model = SimpleNamespace(digest="e" * 64)
    base_bundle = SimpleNamespace(manifest={"bundle": "base"}, model=base_model)
    candidate_bundle = SimpleNamespace(
        manifest={"bundle": "candidate"},
        model=candidate_model,
    )
    catalog = Mock()
    catalog.family_key_for_id.return_value = publication.family_id
    catalog.current.return_value = Mock(
        model_id=publication.previous_model_id,
        artifact=Mock(),
    )
    artifacts = Mock()
    artifacts.publish.return_value = ArtifactMetadata(
        key="f" * 64,
        size_bytes=1,
        path=tmp_path / "final.tar.gz",
        url="http://py-mtlf.example/final",
    )
    workspace = Mock()
    workspace.publish.return_value = SimpleNamespace(path=tmp_path / "workspace-final")
    coordinator = PublicationCoordinator(
        PublicationSettings(),
        Mock(),
        catalog,
        artifacts,
        workspace,
        Mock(),
        nwdaf_context_client(),
        Mock(),
    )
    coordinator._loader = Mock()
    coordinator._loader.load.side_effect = [base_bundle, candidate_bundle]
    coordinator._replace_publication = lambda value: value
    monkeypatch.setattr(
        "py_mtlf.core.publication.model_contract_digest",
        lambda _manifest: "1" * 64,
    )
    monkeypatch.setattr(
        "py_mtlf.core.publication.preprocessing_contract_digest",
        lambda _manifest: "2" * 64,
    )
    monkeypatch.setattr(
        "py_mtlf.core.publication.weights_digest",
        lambda model: model.digest,
    )

    updated = coordinator._build_final_bundle(publication)

    metadata = workspace.publish.call_args.kwargs["metadata"]["fl_metadata"]
    assert metadata["hierarchy_validation"] == (
        publication.hierarchy_validation.model_dump(mode="json")
    )
    assert updated.hierarchy_validation == publication.hierarchy_validation
    assert updated.state is PublicationState.FINAL_BUNDLE_READY
