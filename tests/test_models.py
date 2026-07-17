from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from py_mtlf.models import ApplyEvidence, ApplyStatus, ArtifactRecord, ModelIdentity


def test_model_identity_trims_and_rejects_blank_provider():
    assert ModelIdentity(provider_id=" local ", model_unique_id=1).provider_id == "local"
    with pytest.raises(ValidationError):
        ModelIdentity(provider_id=" ", model_unique_id=1)


def test_artifact_record_requires_lowercase_sha256():
    with pytest.raises(ValidationError):
        ArtifactRecord(
            url="http://127.0.0.1:9092/internal/v1/artifacts/invalid",
            digest="INVALID",
            size_bytes=1,
        )


def test_apply_evidence_requires_timezone():
    with pytest.raises(ValidationError):
        ApplyEvidence(
            event_id="event-1",
            model_identity=ModelIdentity(provider_id="local", model_unique_id=1),
            target_generation=1,
            status=ApplyStatus.FAILED,
            affected_runtime_count=0,
            completed_at=datetime(2026, 7, 17),
        )

    evidence = ApplyEvidence(
        event_id="event-1",
        model_identity=ModelIdentity(provider_id="local", model_unique_id=1),
        target_generation=1,
        status=ApplyStatus.FAILED,
        affected_runtime_count=0,
        completed_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    assert evidence.status == ApplyStatus.FAILED
