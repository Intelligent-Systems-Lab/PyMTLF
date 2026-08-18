import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import torch

from py_mtlf.config import ArtifactSettings, FederatedLearningSettings
from py_mtlf.core.artifacts import REQUIRED_BUNDLE_FILES, ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    FLArtifactContract,
    HierarchyAssignmentArtifact,
    HierarchyPreparationResultArtifact,
    validate_fl_artifact_manifest,
)
from py_mtlf.core.fl_hierarchy import (
    HierarchyMessageType,
    normalize_nf_instance_id,
    normalize_plan_id,
)
from py_mtlf.core.trainer import LoadedBundle
from py_mtlf.models import SHA256_PATTERN, ModelIdentity


class FLWorkspaceError(RuntimeError):
    pass


class FLArtifactUnavailableError(FLWorkspaceError):
    pass


class FLArtifactIntegrityError(FLWorkspaceError):
    pass


class FLArtifactContractError(FLWorkspaceError):
    pass


class FLArtifactIdentityError(FLWorkspaceError):
    pass


@dataclass(frozen=True)
class FLWorkspaceArtifact:
    process_id: str
    participant_id: str
    round_indicator: int
    role: str
    digest: str
    path: Path
    url: str
    manifest: dict[str, object]
    contract: FLArtifactContract


@dataclass(frozen=True)
class ValidatedArchive:
    manifest: dict[str, object]
    contract: FLArtifactContract | None


@dataclass(frozen=True)
class ValidatedHierarchyArtifact:
    metadata: ArtifactMetadata
    manifest: dict[str, object]
    contract: HierarchyAssignmentArtifact | HierarchyPreparationResultArtifact


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
            if child.name == ".staging":
                for staged in child.iterdir() if child.is_dir() else ():
                    if staged.stat().st_mtime < cutoff:
                        if staged.is_dir():
                            shutil.rmtree(staged, ignore_errors=True)
                        else:
                            staged.unlink(missing_ok=True)
                continue
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

    def download_hierarchy(
        self,
        url: str,
        *,
        expected_role: ArtifactRole,
        expected_message_type: HierarchyMessageType,
        expected_publisher_nf_instance_id: str,
        intended_recipient_nf_instance_id: str,
        expected_plan_id: str | None = None,
    ) -> ValidatedHierarchyArtifact:
        return self._download_hierarchy(
            url,
            expected_role=expected_role,
            expected_message_type=expected_message_type,
            expected_publisher_nf_instance_id=expected_publisher_nf_instance_id,
            intended_recipient_nf_instance_id=intended_recipient_nf_instance_id,
            expected_plan_id=expected_plan_id,
        )

    def download_assignment(
        self,
        url: str,
        *,
        intended_recipient_nf_instance_id: str,
        expected_plan_id: str | None = None,
    ) -> ValidatedHierarchyArtifact:
        return self._download_hierarchy(
            url,
            expected_role=ArtifactRole.HIERARCHY_ASSIGNMENT,
            expected_message_type=None,
            expected_publisher_nf_instance_id=None,
            intended_recipient_nf_instance_id=intended_recipient_nf_instance_id,
            expected_plan_id=expected_plan_id,
        )

    def inspect_artifact(self, artifact: ArtifactMetadata) -> ValidatedArchive:
        return self._validate_archive(artifact.path)

    def _download_hierarchy(
        self,
        url: str,
        *,
        expected_role: ArtifactRole,
        expected_message_type: HierarchyMessageType | None,
        expected_publisher_nf_instance_id: str | None,
        intended_recipient_nf_instance_id: str,
        expected_plan_id: str | None,
    ) -> ValidatedHierarchyArtifact:
        if expected_role not in {
            ArtifactRole.HIERARCHY_ASSIGNMENT,
            ArtifactRole.HIERARCHY_PREPARATION_RESULT,
        }:
            raise ValueError("hierarchy download requires a hierarchy artifact role")
        expected_publisher = (
            normalize_nf_instance_id(expected_publisher_nf_instance_id)
            if expected_publisher_nf_instance_id is not None
            else None
        )
        intended_recipient = normalize_nf_instance_id(intended_recipient_nf_instance_id)
        normalized_plan_id = (
            normalize_plan_id(expected_plan_id) if expected_plan_id is not None else None
        )

        try:
            allowed = {
                _origin(value) for value in self._settings.artifact_download.allowed_origins
            }
            origin = _origin(url)
            expected_digest = _artifact_url_digest(url)
        except (RuntimeError, ValueError) as error:
            raise FLArtifactIntegrityError(str(error)) from error
        if allowed and origin not in allowed:
            raise FLArtifactIdentityError("FL artifact origin is not allowed")
        staging = self._root / ".staging"
        file_descriptor: int | None = None
        temporary: Path | None = None
        try:
            staging.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".hierarchy-download-",
                suffix=".tar.gz",
                dir=staging,
            )
            temporary = Path(temporary_name)
            os.close(file_descriptor)
            file_descriptor = None
        except OSError as error:
            if file_descriptor is not None:
                with suppress(OSError):
                    os.close(file_descriptor)
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
            raise FLWorkspaceError("FL hierarchy download staging is unavailable") from error
        if temporary is None:
            raise FLWorkspaceError("FL hierarchy download staging was not created")
        digest = hashlib.sha256()
        size = 0
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise FLArtifactUnavailableError(
                        f"FL artifact download failed with {response.status_code}"
                    )
                digest_headers = response.headers.get_list("X-Artifact-SHA256")
                if len(digest_headers) != 1 or not SHA256_PATTERN.fullmatch(digest_headers[0]):
                    raise FLArtifactIntegrityError(
                        "FL artifact digest response header is invalid"
                    )
                response_digest = digest_headers[0]
                if response_digest != expected_digest:
                    raise FLArtifactIntegrityError(
                        "FL artifact URL and response digest do not match"
                    )
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self._artifact_settings.max_compressed_bytes:
                            raise FLArtifactIntegrityError(
                                "FL artifact download exceeds the configured limit"
                            )
                        digest.update(chunk)
                        output.write(chunk)
            if size == 0:
                raise FLArtifactIntegrityError("FL artifact download is empty")
            if digest.hexdigest() != expected_digest:
                raise FLArtifactIntegrityError(
                    "FL artifact downloaded archive digest does not match"
                )

            validated = self._validate_archive(temporary)
            contract = validated.contract
            if not isinstance(
                contract,
                (HierarchyAssignmentArtifact, HierarchyPreparationResultArtifact),
            ):
                raise FLArtifactContractError("FL artifact is not a hierarchy artifact")
            metadata = contract.hierarchy_metadata
            if contract.artifact_role is not expected_role:
                raise FLArtifactContractError(
                    "FL hierarchy artifact role does not match expectation"
                )
            if (
                expected_message_type is not None
                and metadata.message_type is not expected_message_type
            ):
                raise FLArtifactContractError(
                    "FL hierarchy message type does not match expectation"
                )
            if (
                expected_publisher is not None
                and metadata.publisher_nf_instance_id != expected_publisher
            ):
                raise FLArtifactIdentityError(
                    "FL hierarchy publisher does not match expected peer"
                )
            if metadata.intended_recipient_nf_instance_id != intended_recipient:
                raise FLArtifactIdentityError(
                    "FL hierarchy artifact has the wrong intended recipient"
                )
            if normalized_plan_id is not None and metadata.plan_id != normalized_plan_id:
                raise FLArtifactIdentityError(
                    "FL hierarchy plan ID does not match expectation"
                )

            directory = self._root / metadata.plan_id / "downloads"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{expected_digest}.tar.gz"
            if destination.exists():
                if _hash_file(destination) != expected_digest:
                    raise FLArtifactIntegrityError(
                        "existing FL hierarchy download conflicts with digest"
                    )
            else:
                os.replace(temporary, destination)
            artifact_metadata = ArtifactMetadata(
                key=expected_digest,
                size_bytes=size,
                path=destination,
                url=url,
            )
            return ValidatedHierarchyArtifact(
                metadata=artifact_metadata,
                manifest=validated.manifest,
                contract=contract,
            )
        except httpx.HTTPError as error:
            raise FLArtifactUnavailableError("FL artifact transport failed") from error
        except OSError as error:
            raise FLWorkspaceError("FL hierarchy workspace operation failed") from error
        finally:
            _remove_hierarchy_staging(temporary)

    def release_plan(self, plan_id: str) -> None:
        normalized = normalize_plan_id(plan_id)
        directory = self._root / normalized
        try:
            if directory.is_file():
                raise FLWorkspaceError("FL plan workspace path is not a directory")
            with suppress(FileNotFoundError):
                shutil.rmtree(directory)
        except OSError as error:
            raise FLWorkspaceError("FL plan workspace release failed") from error

    def _validate_archive(self, path: Path) -> ValidatedArchive:
        extracted = 0
        names = set()
        manifest_bytes: bytes | None = None
        component_digests: dict[str, str] = {}
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                if len(members) > self._artifact_settings.max_entries:
                    raise FLArtifactIntegrityError("FL artifact has too many entries")
                for member in members:
                    if (
                        not member.isreg()
                        or "/" in member.name
                        or member.name.startswith(".")
                        or member.name in names
                    ):
                        raise FLArtifactIntegrityError("FL artifact contains an unsafe entry")
                    names.add(member.name)
                    extracted += member.size
                    if member.size > self._artifact_settings.max_single_file_bytes:
                        raise FLArtifactIntegrityError(
                            "FL artifact entry exceeds the configured limit"
                        )
                    if extracted > self._artifact_settings.max_extracted_bytes:
                        raise FLArtifactIntegrityError(
                            "FL artifact exceeds the extracted size limit"
                        )
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise FLArtifactIntegrityError("FL artifact entry cannot be read")
                    content = stream.read(self._artifact_settings.max_single_file_bytes + 1)
                    if len(content) != member.size:
                        raise FLArtifactIntegrityError(
                            "FL artifact entry size does not match archive metadata"
                        )
                    if member.name == "config.json":
                        manifest_bytes = content
                    else:
                        component_digests[member.name] = hashlib.sha256(content).hexdigest()
        except (OSError, tarfile.TarError) as error:
            raise FLArtifactIntegrityError(
                "FL artifact is not a valid gzip tar archive"
            ) from error
        if names != REQUIRED_BUNDLE_FILES:
            missing = sorted(REQUIRED_BUNDLE_FILES - names)
            unexpected = sorted(names - REQUIRED_BUNDLE_FILES)
            raise FLArtifactIntegrityError(
                f"FL artifact file set is invalid; missing={missing}, unexpected={unexpected}"
            )
        manifest = _validated_manifest(manifest_bytes, component_digests)
        role = manifest.get("artifact_role")
        if role is None:
            try:
                ModelIdentity.model_validate(manifest["model_identity"])
            except (KeyError, ValueError) as error:
                raise FLArtifactContractError(
                    "completed FL input model identity is invalid"
                ) from error
            return ValidatedArchive(manifest=manifest, contract=None)
        try:
            contract = validate_fl_artifact_manifest(manifest)
        except ValueError as error:
            raise FLArtifactContractError("FL artifact role contract is invalid") from error
        return ValidatedArchive(manifest=manifest, contract=contract)

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
        weights = np.empty(len(model.state_dict()), dtype=object)
        weights[:] = [value.detach().cpu().numpy() for value in model.state_dict().values()]
        weights_stream = io.BytesIO()
        np.save(weights_stream, weights, allow_pickle=True)
        if not base.scaler_source:
            raise RuntimeError("FL base bundle has no preserved scaler source")
        components = {
            "model.py": base.model_source,
            "model.npy": weights_stream.getvalue(),
            "scaler.pkl": base.scaler_source,
        }
        manifest = dict(base.manifest)
        for key in (
            "artifact_role",
            "fl_metadata",
            "hierarchy_metadata",
            "model_identity",
            "result_type",
        ):
            manifest.pop(key, None)
        manifest.update(metadata)
        manifest["bundle_schema_version"] = "1.0"
        manifest["file_digests"] = {
            name: hashlib.sha256(content).hexdigest() for name, content in components.items()
        }
        contract = validate_fl_artifact_manifest(manifest)
        files = {
            "config.json": json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            **components,
        }

        directory = (
            self._root
            / _safe(process_id)
            / _safe(participant_id)
            / str(round_indicator)
            / _safe(role)
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                temp = Path(temporary)
                candidate = temp / "artifact.tar.gz"
                _write_bundle(candidate, files)
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                destination = directory / f"{digest}.tar.gz"
                if destination.exists():
                    if _hash_file(destination) != digest:
                        raise FLArtifactIntegrityError(
                            "existing FL artifact publication conflicts with digest"
                        )
                else:
                    os.replace(candidate, destination)
        except (OSError, tarfile.TarError) as error:
            raise FLWorkspaceError(
                "FL artifact publication workspace operation failed"
            ) from error
        base_url = self._settings.public_base_url.rstrip("/")
        url = (
            f"{base_url}/internal/v1/fl-artifacts/{quote(_safe(process_id))}/"
            f"{quote(_safe(participant_id))}/{round_indicator}/{quote(_safe(role))}/{digest}"
        )
        return FLWorkspaceArtifact(
            process_id=process_id,
            participant_id=participant_id,
            round_indicator=round_indicator,
            role=role,
            digest=digest,
            path=destination,
            url=url,
            manifest=manifest,
            contract=contract,
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
    try:
        parsed_port = parsed.port
    except ValueError as error:
        raise RuntimeError("FL artifact URL has an invalid port") from error
    port = f":{parsed_port}" if parsed_port else ""
    hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{hostname}{port}"


def _artifact_url_digest(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if parsed.query or parsed.fragment:
        raise RuntimeError("FL artifact URL contains unsupported components")
    digest = parsed.path.rsplit("/", 1)[-1]
    if not SHA256_PATTERN.fullmatch(digest):
        raise RuntimeError("FL artifact URL has an invalid digest")
    return digest


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_hierarchy_staging(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise FLWorkspaceError("FL hierarchy download staging cleanup failed") from error


def _validated_manifest(
    manifest_bytes: bytes | None,
    component_digests: dict[str, str],
) -> dict[str, object]:
    if manifest_bytes is None:
        raise FLArtifactIntegrityError("FL artifact is missing config.json")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FLArtifactContractError("FL artifact config.json is invalid") from error
    if not isinstance(manifest, dict):
        raise FLArtifactContractError("FL artifact config.json must contain an object")
    if manifest.get("bundle_schema_version") != "1.0":
        raise FLArtifactContractError("FL artifact bundle schema version is unsupported")
    if not isinstance(manifest.get("analytics_event"), str) or not manifest["analytics_event"]:
        raise FLArtifactContractError("FL artifact analytics_event is required")
    if (
        not isinstance(manifest.get("model_interoperability"), str)
        or not manifest["model_interoperability"].strip()
    ):
        raise FLArtifactContractError("FL artifact model_interoperability is required")
    if not isinstance(manifest.get("runtime_compatibility"), dict):
        raise FLArtifactContractError("FL artifact runtime_compatibility is required")
    if not isinstance(manifest.get("model"), dict):
        raise FLArtifactContractError("FL artifact model contract is required")
    if not isinstance(manifest.get("inference"), dict):
        raise FLArtifactContractError("FL artifact inference contract is required")
    expected_names = {
        manifest.get("MODEL_SCRIPT"),
        manifest.get("MODEL_PATH"),
        manifest.get("SCALER_PATH"),
    }
    if expected_names != {"model.py", "model.npy", "scaler.pkl"}:
        raise FLArtifactContractError("FL artifact component filenames are invalid")
    declared = manifest.get("file_digests")
    if not isinstance(declared, dict) or set(declared) != set(component_digests):
        raise FLArtifactIntegrityError("FL artifact component digest inventory is invalid")
    for name, actual in component_digests.items():
        expected = declared.get(name)
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            raise FLArtifactIntegrityError("FL artifact component digest is invalid")
        if expected != actual:
            raise FLArtifactIntegrityError(f"FL artifact component digest mismatch: {name}")
    return manifest
