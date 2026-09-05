import hashlib
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from py_mtlf.config import ArtifactSettings
from py_mtlf.models import SHA256_PATTERN, ModelIdentity

REQUIRED_BUNDLE_FILES = {"config.json", "model.py", "model.npy", "scaler.pkl"}


class ArtifactError(RuntimeError):
    pass


class InvalidArtifactError(ArtifactError):
    pass


class ArtifactConflictError(ArtifactError):
    pass


class ArtifactNotFoundError(ArtifactError):
    pass


@dataclass(frozen=True)
class ArtifactMetadata:
    key: str
    size_bytes: int
    path: Path
    url: str
    media_type: str = "application/gzip"


class ArtifactRepository:
    def __init__(self, root: Path, settings: ArtifactSettings):
        self._root = Path(root)
        self._settings = settings

    @property
    def root(self) -> Path:
        return self._root

    def open(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self.probe()

    def probe(self) -> None:
        probe_fd, probe_name = tempfile.mkstemp(prefix=".write-probe-", dir=self._root)
        probe_path = Path(probe_name)
        try:
            os.close(probe_fd)
            probe_path.unlink()
        finally:
            probe_path.unlink(missing_ok=True)

    def publish(self, candidate: str | Path) -> ArtifactMetadata:
        source = Path(candidate)
        if not source.is_file():
            raise InvalidArtifactError("candidate artifact is not a regular file")
        size = source.stat().st_size
        if size <= 0 or size > self._settings.max_compressed_bytes:
            raise InvalidArtifactError("compressed artifact size is outside the configured limit")

        digest = self._hash_file(source)
        self._validate_bundle(source)
        destination = self._path_for_key(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            if destination.stat().st_size != size or self._hash_file(destination) != digest:
                raise ArtifactConflictError(
                    "existing content-addressed artifact does not match key"
                )
            return self.metadata(digest)

        fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
                self._copy(input_stream, output)
                output.flush()
                os.fsync(output.fileno())
            if self._hash_file(temporary) != digest:
                raise ArtifactConflictError("artifact changed while publishing")
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                if destination.stat().st_size != size or self._hash_file(destination) != digest:
                    raise ArtifactConflictError(
                        "concurrent content-addressed artifact does not match key"
                    ) from exc
            self._fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.metadata(digest)

    def metadata(self, key: str) -> ArtifactMetadata:
        self._validate_key(key)
        path = self._path_for_key(key)
        if not path.is_file():
            raise ArtifactNotFoundError("artifact was not found")
        return ArtifactMetadata(
            key=key,
            size_bytes=path.stat().st_size,
            path=path,
            url=f"{self._settings.public_base_url}/internal/v1/artifacts/{key}",
        )

    def inventory(self) -> list[ArtifactMetadata]:
        artifacts = []
        if not self._root.exists():
            return artifacts
        for path in sorted(self._root.glob("[0-9a-f][0-9a-f]/*")):
            if path.is_file() and SHA256_PATTERN.fullmatch(path.name):
                artifacts.append(self.metadata(path.name))
        return artifacts

    def manifest(self, key: str) -> dict[str, object]:
        metadata = self.metadata(key)
        try:
            with tarfile.open(metadata.path, "r:gz") as archive:
                member = archive.getmember("config.json")
                stream = archive.extractfile(member)
                if stream is None:
                    raise InvalidArtifactError("artifact config.json cannot be read")
                value = json.loads(stream.read(self._settings.max_single_file_bytes + 1))
        except (KeyError, tarfile.TarError, OSError, json.JSONDecodeError) as exc:
            raise InvalidArtifactError("artifact manifest cannot be read") from exc
        if not isinstance(value, dict):
            raise InvalidArtifactError("artifact manifest must contain an object")
        return value

    def protected_delete(self, key: str, protected_keys: set[str]) -> bool:
        self._validate_key(key)
        if key in protected_keys:
            raise ArtifactConflictError("artifact is protected by durable state")
        path = self._path_for_key(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _path_for_key(self, key: str) -> Path:
        self._validate_key(key)
        return self._root / key[:2] / key

    @staticmethod
    def _validate_key(key: str) -> None:
        if not SHA256_PATTERN.fullmatch(key):
            raise InvalidArtifactError("artifact key must be lowercase SHA-256 hex")

    @staticmethod
    def _copy(source: BinaryIO, destination: BinaryIO) -> None:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_bundle(self, path: Path) -> None:
        names: set[str] = set()
        extracted_size = 0
        config_bytes: bytes | None = None
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                if len(members) > self._settings.max_entries:
                    raise InvalidArtifactError("artifact has too many archive entries")
                for member in members:
                    self._validate_member(member, names)
                    names.add(member.name)
                    extracted_size += member.size
                    if extracted_size > self._settings.max_extracted_bytes:
                        raise InvalidArtifactError("artifact exceeds extracted size limit")
                    if member.size > self._settings.max_single_file_bytes:
                        raise InvalidArtifactError("artifact entry exceeds single-file size limit")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise InvalidArtifactError("artifact entry cannot be read")
                    content = stream.read(self._settings.max_single_file_bytes + 1)
                    if len(content) != member.size:
                        raise InvalidArtifactError(
                            "artifact entry size does not match archive metadata"
                        )
                    if member.name == "config.json":
                        config_bytes = content
        except (tarfile.TarError, OSError) as exc:
            raise InvalidArtifactError("artifact is not a valid gzip tar archive") from exc

        if names != REQUIRED_BUNDLE_FILES:
            missing = sorted(REQUIRED_BUNDLE_FILES - names)
            unexpected = sorted(names - REQUIRED_BUNDLE_FILES)
            raise InvalidArtifactError(
                f"artifact file set is invalid; missing={missing}, unexpected={unexpected}"
            )
        self._validate_manifest(config_bytes)

    @staticmethod
    def _validate_member(member: tarfile.TarInfo, names: set[str]) -> None:
        pure = PurePosixPath(member.name)
        if member.name in names:
            raise InvalidArtifactError("artifact contains a duplicate entry")
        if (
            not member.isreg()
            or pure.is_absolute()
            or len(pure.parts) != 1
            or pure.name in {"", ".", ".."}
            or ".." in pure.parts
        ):
            raise InvalidArtifactError("artifact contains an unsafe entry")

    @staticmethod
    def _validate_manifest(config_bytes: bytes | None) -> None:
        if config_bytes is None:
            raise InvalidArtifactError("artifact is missing config.json")
        try:
            config = json.loads(config_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidArtifactError("config.json is not valid JSON") from exc
        if not isinstance(config, dict):
            raise InvalidArtifactError("config.json must contain a JSON object")
        if "file_digests" in config:
            raise InvalidArtifactError("unsupported manifest field: file_digests")
        try:
            ModelIdentity.model_validate(config["model_identity"])
        except (KeyError, ValueError) as exc:
            raise InvalidArtifactError("bundle model identity is invalid") from exc
        if not isinstance(config.get("analytics_event"), str) or not config["analytics_event"]:
            raise InvalidArtifactError("bundle analytics_event is required")
        if not isinstance(config.get("runtime_compatibility"), dict):
            raise InvalidArtifactError("bundle runtime_compatibility is required")
        inference = config.get("inference")
        if not isinstance(inference, dict) or not inference.get("feature_order"):
            raise InvalidArtifactError("bundle inference feature_order is required")
        if not inference.get("output_fields") or not isinstance(inference.get("seq_length"), int):
            raise InvalidArtifactError("bundle inference output_fields and seq_length are required")
        expected_names = {
            config.get("MODEL_SCRIPT"),
            config.get("MODEL_PATH"),
            config.get("SCALER_PATH"),
        }
        if expected_names != {"model.py", "model.npy", "scaler.pkl"}:
            raise InvalidArtifactError("bundle component filenames are invalid")
    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
