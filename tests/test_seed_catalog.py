import pytest
from conftest import build_bundle

from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import (
    ArtifactRepository,
    InvalidArtifactError,
)
from py_mtlf.core.model_records import AdrfReference
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.wire.ml_model import MLEventSubscription


def test_seed_catalog_validates_manifest_and_generic_applicability(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=1,
                    artifact_key=artifact.key,
                    event="UE_COMMUNICATION",
                ),
            ),
        ),
        repository,
    )
    catalog.open()

    group_a = MLEventSubscription(
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={"snssai": {"sst": 1}},
        tgtUe={"intGroupIds": ["group-a"]},
    )
    group_b = MLEventSubscription(
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={"snssai": {"sst": 2}},
        tgtUe={"intGroupIds": ["group-b"]},
    )

    assert catalog.resolve(group_a) == catalog.resolve(group_b)
    assert catalog.resolve(group_a).artifact == artifact


def test_seed_catalog_treats_omitted_interoperability_as_no_filter(
    settings,
    bundle_path,
):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=1,
                    artifact_key=artifact.key,
                    event="UE_COMMUNICATION",
                    model_interoperability="001122",
                ),
            ),
        ),
        repository,
    )
    catalog.open()

    without_filter = MLEventSubscription(
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
    )
    matching_filter = MLEventSubscription(
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        modelInterInfo="001122",
    )
    mismatching_filter = MLEventSubscription(
        mLEvent="UE_COMMUNICATION",
        mLEventFilter={},
        modelInterInfo="334455",
    )

    assert catalog.resolve(without_filter) is not None
    assert catalog.resolve(matching_filter) is not None
    assert catalog.resolve(mismatching_filter) is None


def test_seed_catalog_rejects_descriptor_manifest_identity_mismatch(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=2,
                    artifact_key=artifact.key,
                    event="UE_COMMUNICATION",
                ),
            ),
        ),
        repository,
    )

    with pytest.raises(InvalidArtifactError, match="identity"):
        catalog.open()


def test_catalog_promotes_candidate_with_new_model_identity(
    settings,
    bundle_path,
    tmp_path,
):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    seed = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            seed_models=(
                SeedModelSettings(
                    family_id="ue-communication-default",
                    model_id=1,
                    artifact_key=seed.key,
                    event="UE_COMMUNICATION",
                ),
            ),
        ),
        repository,
    )
    catalog.open()
    candidate_path = tmp_path / "candidate.tar.gz"
    build_bundle(
        candidate_path,
        mutate_manifest={
            "model_generation": 2,
            "model_identity": {"model_unique_id": 2},
        },
    )
    candidate = repository.publish(candidate_path)
    family_key = "ue-communication-default"
    version_key = catalog.restore_reserved_version(family_key, 2)

    promoted = catalog.promote(
        family_key,
        expected_generation=1,
        expected_artifact_key=seed.key,
        version_key=version_key,
        artifact=candidate,
        adrf_reference=AdrfReference(
            adrfInstanceId="00000000-0000-4000-8000-000000000010",
            storeTransId="store-2",
            resourceLocation="http://adrf.example/models/store-2",
        ),
    )

    assert promoted.generation == 2
    assert promoted.model_id == 2
    assert catalog.family_for_version(1) == family_key
    assert catalog.family_for_version(2) == family_key
    assert promoted.artifact.key == candidate.key
