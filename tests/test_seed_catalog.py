import pytest
from conftest import build_bundle

from py_mtlf.config import ModelProvisionSettings, SeedModelSettings
from py_mtlf.core.artifacts import (
    ArtifactRepository,
    InvalidArtifactError,
)
from py_mtlf.core.seed_catalog import SeedCatalog
from py_mtlf.wire.ml_model import MLEventSubscription


def test_seed_catalog_validates_manifest_and_generic_applicability(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            provider_namespace="local",
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


def test_seed_catalog_rejects_descriptor_manifest_identity_mismatch(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)
    catalog = SeedCatalog(
        ModelProvisionSettings(
            provider_namespace="different-provider",
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
            provider_namespace="local",
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
            "model_identity": {"provider_id": "local", "model_unique_id": 2},
        },
    )
    candidate = repository.publish(candidate_path)
    family_key = ("local", "ue-communication-default")
    version_key = catalog.reserve_next_version(family_key)

    promoted = catalog.promote(
        family_key,
        expected_generation=1,
        expected_artifact_key=seed.key,
        version_key=version_key,
        artifact=candidate,
    )

    assert promoted.generation == 2
    assert promoted.model_id == 2
    assert catalog.family_for_version(("local", 1)) == family_key
    assert catalog.family_for_version(("local", 2)) == family_key
    assert promoted.artifact.key == candidate.key
