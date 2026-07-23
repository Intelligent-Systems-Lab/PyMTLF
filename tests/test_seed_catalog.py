import pytest

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
