from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from py_mtlf.core.model_records import (
    ModelCatalogRecord,
    PendingPublication,
    migrate_seed_catalog,
    validate_catalog_publications,
)

DIGEST_A = "a" * 64
CLIENT_A = "00000000-0000-4000-8000-000000000001"
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
