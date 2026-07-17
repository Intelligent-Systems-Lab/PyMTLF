import hashlib
import io
import tarfile

import pytest
from conftest import build_bundle

from py_mtlf.core.artifacts import (
    ArtifactConflictError,
    ArtifactRepository,
    InvalidArtifactError,
)


def test_publish_is_content_addressed_and_idempotent(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    first = repository.publish(bundle_path)
    second = repository.publish(bundle_path)

    assert first == second
    assert first.key == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert first.path.read_bytes() == bundle_path.read_bytes()
    assert repository.inventory() == [first]


def test_protected_delete_requires_unreferenced_key(settings, bundle_path):
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    artifact = repository.publish(bundle_path)

    with pytest.raises(ArtifactConflictError):
        repository.protected_delete(artifact.key, {artifact.key})

    assert repository.protected_delete(artifact.key, set())
    assert not repository.protected_delete(artifact.key, set())


def test_reopen_retains_published_artifact(settings, bundle_path):
    first = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    first.open()
    artifact = first.publish(bundle_path)

    reopened = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    reopened.open()

    assert reopened.metadata(artifact.key) == artifact
    assert reopened.inventory() == [artifact]


def test_publish_rejects_unsafe_or_unexpected_entry(settings, tmp_path):
    path = tmp_path / "unsafe.tar.gz"
    info = tarfile.TarInfo("../escape")
    info.size = 1
    build_bundle(path, extra_entries=[(info, b"x")])
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    with pytest.raises(InvalidArtifactError, match="unsafe"):
        repository.publish(path)


def test_publish_rejects_symlink(settings, tmp_path):
    path = tmp_path / "symlink.tar.gz"
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "config.json"
    info.size = 0
    build_bundle(path, extra_entries=[(info, b"")])
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    with pytest.raises(InvalidArtifactError, match="unsafe"):
        repository.publish(path)


def test_publish_rejects_duplicate_entry(settings, bundle_path, tmp_path):
    duplicate = tmp_path / "duplicate.tar.gz"
    with tarfile.open(bundle_path, "r:gz") as source, tarfile.open(duplicate, "w:gz") as output:
        for member in source.getmembers():
            content = source.extractfile(member).read()
            output.addfile(member, io.BytesIO(content))
        content = b"duplicate"
        info = tarfile.TarInfo("model.py")
        info.size = len(content)
        output.addfile(info, io.BytesIO(content))
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    with pytest.raises(InvalidArtifactError, match="duplicate"):
        repository.publish(duplicate)


def test_publish_rejects_manifest_digest_mismatch(settings, tmp_path):
    path = tmp_path / "bad-digest.tar.gz"
    build_bundle(path, mutate_manifest={"file_digests": {}})
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    with pytest.raises(InvalidArtifactError, match="digest inventory"):
        repository.publish(path)


@pytest.mark.parametrize("manifest", [[], "invalid", 1])
def test_publish_rejects_non_object_manifest(settings, tmp_path, manifest):
    path = tmp_path / "non-object-manifest.tar.gz"
    build_bundle(path, manifest_value=manifest)
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()

    with pytest.raises(InvalidArtifactError, match="JSON object"):
        repository.publish(path)
