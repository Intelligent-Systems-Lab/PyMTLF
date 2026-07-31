import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import torch

from py_mtlf.config import ArtifactSettings, FederatedLearningSettings
from py_mtlf.core.artifacts import REQUIRED_BUNDLE_FILES, ArtifactMetadata
from py_mtlf.core.fl_artifacts import validate_fl_artifact
from py_mtlf.core.trainer import LoadedBundle
from py_mtlf.models import SHA256_PATTERN, ModelIdentity


@dataclass(frozen=True)
class FLWorkspaceArtifact:
    process_id: str
    participant_id: str
    round_indicator: int
    role: str
    digest: str
    path: Path
    url: str


class FLWorkspace:
    def __init__(
        self,
        settings: FederatedLearningSettings,
        artifact_settings: ArtifactSettings,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._artifact_settings = artifact_settings
        self._root = settings.workspace_root
        self._client = client or httpx.Client(
            timeout=settings.artifact_download.timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None

    def open(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def cleanup_expired(self) -> None:
        cutoff = time.time() - self._settings.workspace_ttl_seconds
        for child in self._root.iterdir() if self._root.exists() else ():
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)

    def download(self, url: str, process_id: str, label: str) -> ArtifactMetadata:
        allowed = set(self._settings.artifact_download.allowed_origins)
        origin = _origin(url)
        if allowed and origin not in allowed:
            raise RuntimeError("FL artifact origin is not allowed")
        directory = self._root / _safe(process_id) / "downloads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(label)}.tar.gz"
        temporary = path.with_name(f".{path.name}.{os.getpid()}")
        digest = hashlib.sha256()
        size = 0
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"FL artifact download failed with {response.status_code}")
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self._artifact_settings.max_compressed_bytes:
                            raise RuntimeError("FL artifact download exceeds the configured limit")
                        digest.update(chunk)
                        output.write(chunk)
            if size == 0:
                raise RuntimeError("FL artifact download is empty")
            self._validate_archive(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactMetadata(
            key=digest.hexdigest(),
            size_bytes=size,
            path=path,
            url=url,
        )

    def _validate_archive(self, path: Path) -> None:
        extracted = 0
        names = set()
        manifest_bytes: bytes | None = None
        component_digests: dict[str, str] = {}
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                if len(members) > self._artifact_settings.max_entries:
                    raise RuntimeError("FL artifact has too many entries")
                for member in members:
                    if (
                        not member.isreg()
                        or "/" in member.name
                        or member.name.startswith(".")
                        or member.name in names
                    ):
                        raise RuntimeError("FL artifact contains an unsafe entry")
                    names.add(member.name)
                    extracted += member.size
                    if member.size > self._artifact_settings.max_single_file_bytes:
                        raise RuntimeError("FL artifact entry exceeds the configured limit")
                    if extracted > self._artifact_settings.max_extracted_bytes:
                        raise RuntimeError("FL artifact exceeds the extracted size limit")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise RuntimeError("FL artifact entry cannot be read")
                    content = stream.read(self._artifact_settings.max_single_file_bytes + 1)
                    if len(content) != member.size:
                        raise RuntimeError("FL artifact entry size does not match archive metadata")
                    if member.name == "config.json":
                        manifest_bytes = content
                    else:
                        component_digests[member.name] = hashlib.sha256(content).hexdigest()
        except (OSError, tarfile.TarError) as error:
            raise RuntimeError("FL artifact is not a valid gzip tar archive") from error
        if names != REQUIRED_BUNDLE_FILES:
            missing = sorted(REQUIRED_BUNDLE_FILES - names)
            unexpected = sorted(names - REQUIRED_BUNDLE_FILES)
            raise RuntimeError(
                f"FL artifact file set is invalid; missing={missing}, unexpected={unexpected}"
            )
        manifest = _validated_manifest(manifest_bytes, component_digests)
        role = manifest.get("artifact_role")
        if role is None:
            try:
                ModelIdentity.model_validate(manifest["model_identity"])
            except (KeyError, ValueError) as error:
                raise RuntimeError("completed FL input model identity is invalid") from error
            return
        try:
            projection = {
                key: manifest[key]
                for key in (
                    "bundle_schema_version",
                    "file_digests",
                    "artifact_role",
                    "fl_metadata",
                )
            }
            if "result_type" in manifest:
                projection["result_type"] = manifest["result_type"]
            if "model_identity" in manifest:
                projection["model_identity"] = manifest["model_identity"]
            validate_fl_artifact(projection)
        except (KeyError, ValueError) as error:
            raise RuntimeError("FL artifact role contract is invalid") from error

    def publish(
        self,
        *,
        process_id: str,
        participant_id: str,
        round_indicator: int,
        role: str,
        base: LoadedBundle,
        model: torch.nn.Module,
        metadata: dict[str, object],
    ) -> FLWorkspaceArtifact:
        directory = (
            self._root
            / _safe(process_id)
            / _safe(participant_id)
            / str(round_indicator)
            / _safe(role)
        )
        directory.mkdir(parents=True, exist_ok=True)
        weights = np.empty(len(model.state_dict()), dtype=object)
        weights[:] = [value.detach().cpu().numpy() for value in model.state_dict().values()]
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            temp = Path(temporary)
            (temp / "model.py").write_bytes(base.model_source)
            np.save(temp / "model.npy", weights, allow_pickle=True)
            if not base.scaler_source:
                raise RuntimeError("FL base bundle has no preserved scaler source")
            (temp / "scaler.pkl").write_bytes(base.scaler_source)
            components = {
                name: (temp / name).read_bytes() for name in ("model.py", "model.npy", "scaler.pkl")
            }
            manifest = dict(base.manifest)
            manifest.pop("model_identity", None)
            manifest.update(metadata)
            manifest["bundle_schema_version"] = "1.0"
            manifest["file_digests"] = {
                name: hashlib.sha256(content).hexdigest() for name, content in components.items()
            }
            projection = {
                key: manifest[key]
                for key in (
                    "bundle_schema_version",
                    "file_digests",
                    "artifact_role",
                    "fl_metadata",
                )
            }
            if "result_type" in manifest:
                projection["result_type"] = manifest["result_type"]
            if "model_identity" in manifest:
                projection["model_identity"] = manifest["model_identity"]
            validate_fl_artifact(projection)
            files = {
                "config.json": json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
                **components,
            }
            candidate = temp / "artifact.tar.gz"
            _write_bundle(candidate, files)
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            destination = directory / f"{digest}.tar.gz"
            if not destination.exists():
                os.replace(candidate, destination)
        base_url = self._settings.public_base_url.rstrip("/")
        url = (
            f"{base_url}/internal/v1/fl-artifacts/{quote(_safe(process_id))}/"
            f"{quote(_safe(participant_id))}/{round_indicator}/{quote(_safe(role))}/{digest}"
        )
        return FLWorkspaceArtifact(
            process_id, participant_id, round_indicator, role, digest, destination, url
        )

    def resolve(
        self,
        process_id: str,
        participant_id: str,
        round_indicator: int,
        role: str,
        digest: str,
    ) -> Path | None:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            return None
        path = (
            self._root
            / _safe(process_id)
            / _safe(participant_id)
            / str(round_indicator)
            / _safe(role)
            / f"{digest}.tar.gz"
        )
        return path if path.is_file() else None


def model_contract_digest(manifest: dict[str, object]) -> str:
    file_digests = manifest.get("file_digests")
    model_source_digest = file_digests.get("model.py") if isinstance(file_digests, dict) else None
    payload = {
        "model": manifest.get("model"),
        "analytics_event": manifest.get("analytics_event"),
        "model_interoperability": manifest.get("model_interoperability"),
        "runtime_compatibility": manifest.get("runtime_compatibility"),
        "model_source_digest": model_source_digest,
    }
    return _digest_json(payload)


def preprocessing_contract_digest(manifest: dict[str, object]) -> str:
    file_digests = manifest.get("file_digests")
    scaler_digest = file_digests.get("scaler.pkl") if isinstance(file_digests, dict) else None
    return _digest_json(
        {
            "inference": manifest.get("inference"),
            "scaler_digest": scaler_digest,
        }
    )


def weights_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.detach().cpu()).tobytes())
    return digest.hexdigest()


def _write_bundle(path: Path, files: dict[str, bytes]) -> None:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))


def _origin(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("FL artifact URL is invalid")
    if parsed.username or parsed.password or parsed.fragment:
        raise RuntimeError("FL artifact URL contains unsupported components")
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _safe(value: str) -> str:
    value = value.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(char not in allowed for char in value):
        raise RuntimeError("FL workspace identifier is invalid")
    return value


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validated_manifest(
    manifest_bytes: bytes | None,
    component_digests: dict[str, str],
) -> dict[str, object]:
    if manifest_bytes is None:
        raise RuntimeError("FL artifact is missing config.json")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("FL artifact config.json is invalid") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("FL artifact config.json must contain an object")
    if manifest.get("bundle_schema_version") != "1.0":
        raise RuntimeError("FL artifact bundle schema version is unsupported")
    if not isinstance(manifest.get("analytics_event"), str) or not manifest["analytics_event"]:
        raise RuntimeError("FL artifact analytics_event is required")
    if (
        not isinstance(manifest.get("model_interoperability"), str)
        or not manifest["model_interoperability"].strip()
    ):
        raise RuntimeError("FL artifact model_interoperability is required")
    if not isinstance(manifest.get("runtime_compatibility"), dict):
        raise RuntimeError("FL artifact runtime_compatibility is required")
    if not isinstance(manifest.get("model"), dict):
        raise RuntimeError("FL artifact model contract is required")
    if not isinstance(manifest.get("inference"), dict):
        raise RuntimeError("FL artifact inference contract is required")
    expected_names = {
        manifest.get("MODEL_SCRIPT"),
        manifest.get("MODEL_PATH"),
        manifest.get("SCALER_PATH"),
    }
    if expected_names != {"model.py", "model.npy", "scaler.pkl"}:
        raise RuntimeError("FL artifact component filenames are invalid")
    declared = manifest.get("file_digests")
    if not isinstance(declared, dict) or set(declared) != set(component_digests):
        raise RuntimeError("FL artifact component digest inventory is invalid")
    for name, actual in component_digests.items():
        expected = declared.get(name)
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            raise RuntimeError("FL artifact component digest is invalid")
        if expected != actual:
            raise RuntimeError(f"FL artifact component digest mismatch: {name}")
    return manifest
