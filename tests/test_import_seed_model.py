import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from py_mtlf.config import ArtifactSettings
from py_mtlf.core.artifacts import (
    ArtifactRepository,
    InvalidArtifactError,
)
from py_mtlf.core.seed_import import build_seed_bundle


def manifest(path: Path) -> tuple[dict[str, object], set[str]]:
    with tarfile.open(path, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        stream = archive.extractfile("config.json")
        assert stream is not None
        return json.load(stream), names


def rewrite_bundle(
    source: Path,
    destination: Path,
    *,
    mutate_manifest: dict[str, object] | None = None,
    remove: str | None = None,
    extra: tuple[str, bytes] | None = None,
) -> None:
    with tarfile.open(source, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            if not member.isreg() or member.name == remove:
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            files[member.name] = stream.read()
    if mutate_manifest:
        value = json.loads(files["config.json"])
        value.update(mutate_manifest)
        files["config.json"] = json.dumps(value, sort_keys=True).encode()
    if extra:
        files[extra[0]] = extra[1]
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, BytesIO(content))


def test_import_traffic_seed_uses_current_contract(tmp_path):
    output = tmp_path / "traffic.tar.gz"
    build_seed_bundle(
        Path("seed_models/initial"),
        output,
        model_id=1,
        event="UE_COMMUNICATION",
        model_interoperability="001122",
    )

    value, names = manifest(output)

    assert value["workload_profile"] == "ue_communication_forecasting"
    assert value["analytics_event"] == "UE_COMMUNICATION"
    assert "bundle_schema_version" not in value
    assert "file_digests" not in value
    assert names == {"config.json", "model.py", "model.npy", "scaler.pkl"}
    repository = ArtifactRepository(tmp_path / "artifacts", ArtifactSettings())
    repository.open()
    repository.publish(output)


def test_import_image_seed_omits_standard_event_and_scaler(tmp_path):
    output = tmp_path / "image.tar.gz"
    build_seed_bundle(
        Path("seed_models/image_classification/mnist"),
        output,
        model_id=2,
        event=None,
        model_interoperability="image-classification-pytorch",
    )

    value, names = manifest(output)

    assert value["workload_profile"] == "image_classification"
    assert value["initialization_seed"] == 101
    assert "analytics_event" not in value
    assert "SCALER_PATH" not in value
    assert names == {"config.json", "model.py", "model.npy"}
    repository = ArtifactRepository(tmp_path / "artifacts", ArtifactSettings())
    repository.open()
    repository.publish(output)


def test_import_image_seed_rejects_analytics_event(tmp_path):
    with pytest.raises(ValueError, match="does not accept"):
        build_seed_bundle(
            Path("seed_models/image_classification/mnist"),
            tmp_path / "image.tar.gz",
            model_id=2,
            event="UE_COMMUNICATION",
            model_interoperability="image-classification-pytorch",
        )


@pytest.mark.parametrize(
    ("rewrite", "expected"),
    [
        ({"remove": "model.py"}, "file set"),
        ({"extra": ("scaler.pkl", b"not-used")}, "file set"),
        ({"mutate_manifest": {"workload_profile": "unknown"}}, "unsupported"),
        (
            {"mutate_manifest": {"model": {"input_channels": 3, "num_classes": 10}}},
            "input_channels",
        ),
    ],
)
def test_image_bundle_rejects_invalid_profile_inventory_or_model_contract(
    tmp_path,
    rewrite,
    expected,
):
    source = tmp_path / "source.tar.gz"
    candidate = tmp_path / "candidate.tar.gz"
    build_seed_bundle(
        Path("seed_models/image_classification/mnist"),
        source,
        model_id=2,
        event=None,
        model_interoperability="image-classification-pytorch",
    )
    rewrite_bundle(source, candidate, **rewrite)
    repository = ArtifactRepository(tmp_path / "artifacts", ArtifactSettings())
    repository.open()

    with pytest.raises(InvalidArtifactError, match=expected):
        repository.publish(candidate)
