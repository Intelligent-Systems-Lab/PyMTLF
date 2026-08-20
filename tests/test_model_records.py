from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from py_mtlf.core.model_records import (
    DurableModelState,
    DurableModelStateRepository,
    ModelCatalogRecord,
    PendingPublication,
    migrate_seed_catalog,
    validate_catalog_publications,
)

DIGEST_A = "a" * 64
CLIENT_A = "00000000-0000-4000-8000-000000000001"
LEAF_A = "00000000-0000-4000-8000-000000000002"
ADRF = "00000000-0000-4000-8000-000000000010"


def seed_revision(model_id: int, previous: int | None = None) -> dict[str, object]:
    return {
        "modelUniqueId": model_id,
        "previousModelUniqueId": previous,
        "origin": "SEED",
        "artifactKey": DIGEST_A,
        "artifactDigest": DIGEST_A,
        "createdAt": datetime.now(UTC),
    }


def validation_evidence() -> list[dict[str, object]]:
    now = datetime.now(UTC)
    return [
        {
            "participant_nf_instance_id": CLIENT_A,
            "scope_digest": DIGEST_A,
            "evaluation_sample_count": 10,
            "start_time": now,
            "end_time": now + timedelta(seconds=1),
            "base_model_weights_digest": DIGEST_A,
            "candidate_weights_digest": DIGEST_A,
            "base": {"absolute_error_sum": 1, "absolute_actual_sum": 10},
            "candidate": {"absolute_error_sum": 1, "absolute_actual_sum": 10},
        }
    ]


def test_seed_migration_builds_first_durable_revision() -> None:
    catalog = migrate_seed_catalog(
        model_unique_id=3,
        artifact_key=DIGEST_A,
        created_at=datetime.now(UTC),
    )
    assert catalog.latest_model_id == 3
    assert catalog.next_model_id == 4
    assert catalog.revisions[0].origin == "SEED"
    assert ModelCatalogRecord.model_validate_json(catalog.model_dump_json()) == catalog


def test_catalog_rejects_unknown_schema_and_revision_cycle() -> None:
    with pytest.raises(ValidationError):
        ModelCatalogRecord.model_validate(
            {
                "schemaVersion": "2.0",
                "latestModelId": 1,
                "nextModelId": 2,
                "revisions": [seed_revision(1)],
            }
        )
    revisions = [seed_revision(1, 2), seed_revision(2, 1)]
    with pytest.raises(ValidationError, match="cycle"):
        ModelCatalogRecord.model_validate(
            {
                "schemaVersion": "1.0",
                "latestModelId": 2,
                "nextModelId": 3,
                "revisions": revisions,
            }
        )


def test_catalog_rejects_revision_outside_latest_linear_lineage() -> None:
    with pytest.raises(ValidationError, match="linear lineage"):
        ModelCatalogRecord.model_validate(
            {
                "schemaVersion": "1.0",
                "latestModelId": 2,
                "nextModelId": 4,
                "revisions": [
                    seed_revision(1),
                    seed_revision(2, 1),
                    seed_revision(3, 1),
                ],
            }
        )


def test_federated_revision_requires_adrf_reference() -> None:
    revision = seed_revision(2, 1)
    revision.update(
        {
            "origin": "FEDERATED",
            "mlCorreId": "fl-process-001",
            "participants": [
                {"participantNfInstanceId": CLIENT_A, "sampleCount": 20},
            ],
        }
    )
    with pytest.raises(ValidationError, match="ADRF reference"):
        ModelCatalogRecord.model_validate(
            {
                "schemaVersion": "1.0",
                "latestModelId": 2,
                "nextModelId": 3,
                "revisions": [seed_revision(1), revision],
            }
        )
    revision["adrfReference"] = {
        "adrfInstanceId": ADRF,
        "storeTransId": "store-1",
        "resourceLocation": "http://adrf.example/models/store-1",
    }
    catalog = ModelCatalogRecord.model_validate(
        {
            "schemaVersion": "1.0",
            "latestModelId": 2,
            "nextModelId": 3,
            "revisions": [seed_revision(1), revision],
        }
    )
    assert catalog.latest_model_id == 2


def test_store_accepted_journal_requires_record_locators() -> None:
    values = {
        "schemaVersion": "1.0",
        "publicationId": "publication-1",
        "state": "STORE_ACCEPTED",
        "mlCorreId": "fl-process-001",
        "reservedModelId": 2,
        "previousModelId": 1,
        "participantsAndSampleCounts": [{"participantNfInstanceId": CLIENT_A, "sampleCount": 20}],
        "validationSummary": {"globalGateAccepted": True},
        "validationEvidence": validation_evidence(),
        "candidatePath": "/durable/publication/candidate.tar.gz",
        "candidateDigest": DIGEST_A,
        "finalBundlePath": "/durable/publication/final.tar.gz",
        "finalBundleDigest": DIGEST_A,
        "selectedAdrfTarget": "http://adrf.example",
        "updatedAt": datetime.now(UTC),
    }
    with pytest.raises(ValidationError, match="record locators"):
        PendingPublication.model_validate(values)
    values.update(
        {
            "storeTransId": "store-1",
            "resourceLocation": "http://adrf.example/models/store-1",
        }
    )
    publication = PendingPublication.model_validate(values)
    assert publication.state == "STORE_ACCEPTED"

    catalog = migrate_seed_catalog(
        model_unique_id=1,
        artifact_key=DIGEST_A,
        created_at=datetime.now(UTC),
    )
    catalog = catalog.model_copy(update={"next_model_id": 3})
    validate_catalog_publications(catalog, (publication,))
    with pytest.raises(ValueError, match="greater than every reserved"):
        validate_catalog_publications(
            catalog.model_copy(update={"next_model_id": 2}),
            (publication,),
        )


def test_durable_model_state_repository_atomically_survives_restart(tmp_path) -> None:
    catalog = migrate_seed_catalog(
        model_unique_id=3,
        artifact_key=DIGEST_A,
        created_at=datetime.now(UTC),
    )
    initial = DurableModelState(
        schemaVersion="2.0",
        lastAllocatedModelId=3,
        families={"family-a": catalog},
    )
    repository = DurableModelStateRepository(tmp_path)
    assert repository.open(initial) == initial

    repository.update(lambda current: current.model_copy(update={"last_allocated_model_id": 10}))

    restored = DurableModelStateRepository(tmp_path).open(initial)
    assert restored.last_allocated_model_id == 10
    assert restored.families["family-a"].latest_model_id == 3
    assert not tuple(tmp_path.glob(".model-state.*"))


def test_durable_publication_preserves_hierarchy_validation_evidence(tmp_path) -> None:
    direct_evidence = validation_evidence()[0]
    leaf_evidence = {**direct_evidence, "participant_nf_instance_id": LEAF_A}
    publication = PendingPublication.model_validate(
        {
            "schemaVersion": "1.0",
            "publicationId": "publication-hierarchy",
            "state": "RESERVED",
            "mlCorreId": "root-process",
            "reservedModelId": 4,
            "previousModelId": 3,
            "familyId": "family-a",
            "expectedGeneration": 1,
            "expectedArtifactDigest": DIGEST_A,
            "participantsAndSampleCounts": [
                {"participantNfInstanceId": CLIENT_A, "sampleCount": 20}
            ],
            "validationSummary": {"globalGateAccepted": True},
            "validationEvidence": [direct_evidence],
            "hierarchyValidation": {
                "plan_id": "11111111-1111-4111-8111-111111111111",
                "branches": [
                    {
                        "branch_nf_instance_id": CLIENT_A,
                        "subordinate_validation_summaries": [leaf_evidence],
                    }
                ],
            },
            "candidatePath": "/durable/publication/candidate.tar.gz",
            "candidateDigest": DIGEST_A,
            "updatedAt": datetime.now(UTC),
        }
    )
    catalog = migrate_seed_catalog(
        model_unique_id=3,
        artifact_key=DIGEST_A,
        created_at=datetime.now(UTC),
    ).model_copy(update={"next_model_id": 5})
    initial = DurableModelState(
        schemaVersion="2.0",
        lastAllocatedModelId=4,
        families={"family-a": catalog},
        pendingPublications=(publication,),
    )

    DurableModelStateRepository(tmp_path).open(initial)
    restored = DurableModelStateRepository(tmp_path).open(initial)

    hierarchy = restored.pending_publications[0].hierarchy_validation
    assert hierarchy is not None
    assert hierarchy.plan_id == "11111111-1111-4111-8111-111111111111"
    assert (
        hierarchy.branches[0]
        .subordinate_validation_summaries[0]
        .participant_nf_instance_id
        == LEAF_A
    )
