import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import httpx
import numpy as np
import torch

from py_mtlf.config import ArtifactSettings, FederatedLearningSettings
from py_mtlf.core.artifacts import ArtifactMetadata
from py_mtlf.core.fl_artifacts import (
    ArtifactRole,
    FLArtifactContract,
    RoundGlobalArtifact,
    validate_fl_artifact_manifest,
)
from py_mtlf.core.fl_hierarchy import normalize_nf_instance_id, normalize_plan_id
from py_mtlf.core.trainer import LoadedBundle
from py_mtlf.core.workloads import (
    IMAGE_CLASSIFICATION_EVENT,
    WorkloadProfile,
    required_bundle_files,
    validate_image_manifest,
    workload_profile,
)
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
class DownloadedArchive:
    metadata: ArtifactMetadata
    validated: ValidatedArchive


class FLArtifactReader:
    def __init__(self, workspace: "FLWorkspace", path: Path, stream: BinaryIO) -> None:
        self._workspace = workspace
        self._path = path
        self._stream = stream
        self._closed = False

    def iter_bytes(self, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        try:
            while chunk := self._stream.read(chunk_size):
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._workspace._release_reader(self._path)


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
        self._lock = threading.RLock()
        self._owned_directories: dict[str, set[Path]] = {}
        self._released_plan_ids: dict[str, float] = {}
        self._pending_release: set[Path] = set()
        self._cleanup_failures: dict[Path, float] = {}
        self._active_readers: dict[Path, int] = {}

    def open(self) -> None:
        self._validate_workspace_root()
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            for child in tuple(self._root.iterdir()):
                self._delete_direct_child(child)
        except (OSError, FLWorkspaceError) as error:
            raise FLWorkspaceError("FL workspace startup cleanup failed") from error
        with self._lock:
            self._owned_directories.clear()
            self._released_plan_ids.clear()
            self._pending_release.clear()
            self._cleanup_failures.clear()
            self._active_readers.clear()

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
        self._retry_cleanup_failures(cutoff=cutoff)
        with self._lock:
            self._prune_released_plans_locked()

    def download(
        self,
        url: str,
        process_id: str,
        label: str,
        *,
        owner_plan_id: str | None = None,
    ) -> ArtifactMetadata:
        return self.download_archive(
            url,
            process_id,
            label,
            owner_plan_id=owner_plan_id,
        ).metadata

    def download_archive(
        self,
        url: str,
        process_id: str,
        label: str,
        *,
        owner_plan_id: str | None = None,
    ) -> DownloadedArchive:
        self.cleanup_expired()
        allowed = set(self._settings.artifact_download.allowed_origins)
        origin = _origin(url)
        if allowed and origin not in allowed:
            raise RuntimeError("FL artifact origin is not allowed")
        directory = self._root / _safe(process_id) / "downloads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(label)}.tar.gz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{_safe(label)}.",
            suffix=".tar.gz",
            dir=directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        expected_digest = _artifact_url_digest(url)
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
            if digest.hexdigest() != expected_digest:
                raise FLArtifactIntegrityError(
                    "FL artifact downloaded archive digest does not match URL"
                )
            validated = self._validate_archive(temporary)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            _remove_empty_download_parent(temporary, self._root)
            raise
        temporary.unlink(missing_ok=True)
        artifact = ArtifactMetadata(
            key=expected_digest,
            size_bytes=size,
            path=path,
            url=url,
        )
        if owner_plan_id is not None:
            self._register_owned_directory(owner_plan_id, directory.parent)
        return DownloadedArchive(metadata=artifact, validated=validated)

    def download_adrf_archive(
        self,
        url: str,
        process_id: str,
        label: str,
        *,
        expected_size: int,
        owner_plan_id: str | None = None,
    ) -> DownloadedArchive:
        """Download an ADRF-owned URL whose path does not expose the source digest."""
        self.cleanup_expired()
        allowed = set(self._settings.artifact_download.allowed_origins)
        origin = _origin(url)
        if allowed and origin not in allowed:
            raise RuntimeError("FL artifact origin is not allowed")
        if expected_size <= 0 or expected_size > self._artifact_settings.max_compressed_bytes:
            raise RuntimeError("ADRF model size is outside the configured limit")
        directory = self._root / _safe(process_id) / "downloads"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(label)}.tar.gz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{_safe(label)}.",
            suffix=".tar.gz",
            dir=directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"ADRF model artifact download failed with {response.status_code}"
                    )
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self._artifact_settings.max_compressed_bytes:
                            raise RuntimeError(
                                "ADRF model artifact download exceeds the configured limit"
                            )
                        digest.update(chunk)
                        output.write(chunk)
            if size != expected_size:
                raise FLArtifactIntegrityError(
                    "ADRF model artifact size does not match the store record"
                )
            validated = self._validate_archive(temporary)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            _remove_empty_download_parent(temporary, self._root)
            raise
        temporary.unlink(missing_ok=True)
        artifact = ArtifactMetadata(
            key=digest.hexdigest(),
            size_bytes=size,
            path=path,
            url=url,
        )
        if owner_plan_id is not None:
            self._register_owned_directory(owner_plan_id, directory.parent)
        return DownloadedArchive(metadata=artifact, validated=validated)

    def inspect_artifact(self, artifact: ArtifactMetadata) -> ValidatedArchive:
        return self._validate_archive(artifact.path)

    def release_plan(self, plan_id: str) -> None:
        self.cleanup_expired()
        normalized = normalize_plan_id(plan_id)
        plan_directory = self._direct_child(normalized)
        with self._lock:
            self._prune_released_plans_locked()
            self._released_plan_ids[normalized] = (
                time.monotonic() + self._settings.lifecycle.tombstone_ttl_seconds
            )
            directories = self._owned_directories.pop(normalized, set())
            directories.add(plan_directory)
            self._pending_release.update(directories)
        failures = self._release_directories(directories)
        if failures:
            raise FLWorkspaceError("FL plan workspace release failed")

    def reset_generation(self) -> None:
        """Release every scratch artifact owned by the previous containing Go boot."""
        if not self._root.exists():
            with self._lock:
                self._owned_directories.clear()
                self._pending_release.clear()
                self._cleanup_failures.clear()
            return
        try:
            directories = {
                self._direct_child(child.name)
                for child in self._root.iterdir()
            }
        except (OSError, FLWorkspaceError) as error:
            raise FLWorkspaceError("FL workspace generation reset failed") from error
        with self._lock:
            self._owned_directories.clear()
            self._pending_release.update(directories)
        failures = self._release_directories(directories)
        if failures:
            raise FLWorkspaceError("FL workspace generation reset failed")

    def claim_artifact(self, owner_plan_id: str, artifact: ArtifactMetadata) -> None:
        try:
            relative = artifact.path.resolve().relative_to(self._root.resolve())
        except ValueError as error:
            raise FLWorkspaceError("FL artifact is outside the workspace") from error
        if len(relative.parts) < 2:
            raise FLWorkspaceError("FL artifact has no process directory")
        self._register_owned_directory(owner_plan_id, self._root / relative.parts[0])

    def republish_validation_candidate(
        self,
        *,
        source: ArtifactMetadata,
        plan_id: str,
        participant_id: str,
        round_indicator: int,
    ) -> FLWorkspaceArtifact:
        normalized_plan = normalize_plan_id(plan_id)
        normalized_participant = normalize_nf_instance_id(participant_id)
        if round_indicator < 0:
            raise ValueError("validation round indicator must be non-negative")
        try:
            expected_digest = _artifact_url_digest(source.url)
        except RuntimeError as error:
            raise FLArtifactIntegrityError(str(error)) from error
        actual_digest = _hash_file(source.path)
        if source.key != expected_digest or actual_digest != expected_digest:
            raise FLArtifactIntegrityError(
                "validation candidate URL, metadata, and archive digest do not match"
            )
        validated = self._validate_archive(source.path)
        if not isinstance(validated.contract, RoundGlobalArtifact):
            raise FLArtifactContractError(
                "validation candidate is not a ROUND_GLOBAL artifact"
            )

        directory = (
            self._root
            / normalized_plan
            / normalized_participant
            / str(round_indicator)
            / ArtifactRole.ROUND_GLOBAL.value
        )
        destination = directory / f"{expected_digest}.tar.gz"
        temporary: Path | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _hash_file(destination) != expected_digest:
                    raise FLArtifactIntegrityError(
                        "existing validation candidate conflicts with digest"
                    )
            else:
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".validation-candidate-",
                    suffix=".tar.gz",
                    dir=directory,
                )
                os.close(file_descriptor)
                temporary = Path(temporary_name)
                shutil.copyfile(source.path, temporary)
                if _hash_file(temporary) != expected_digest:
                    raise FLArtifactIntegrityError(
                        "republished validation candidate changed archive bytes"
                    )
                os.replace(temporary, destination)
        except FLWorkspaceError:
            raise
        except OSError as error:
            raise FLWorkspaceError(
                "validation candidate publication workspace operation failed"
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._register_owned_directory(normalized_plan, directory.parents[2])

        base_url = self._settings.public_base_url.rstrip("/")
        url = (
            f"{base_url}/internal/v1/fl-artifacts/{quote(normalized_plan)}/"
            f"{quote(normalized_participant)}/{round_indicator}/"
            f"{ArtifactRole.ROUND_GLOBAL.value}/{expected_digest}"
        )
        if url == source.url:
            raise FLArtifactIdentityError(
                "Branch validation candidate URL must differ from the Root URL"
            )
        return FLWorkspaceArtifact(
            process_id=normalized_plan,
            participant_id=normalized_participant,
            round_indicator=round_indicator,
            role=ArtifactRole.ROUND_GLOBAL.value,
            digest=expected_digest,
            path=destination,
            url=url,
            manifest=validated.manifest,
            contract=validated.contract,
        )

    def _validate_archive(self, path: Path) -> ValidatedArchive:
        extracted = 0
        names = set()
        manifest_bytes: bytes | None = None
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
        except (OSError, tarfile.TarError) as error:
            raise FLArtifactIntegrityError(
                "FL artifact is not a valid gzip tar archive"
            ) from error
        manifest = _validated_manifest(manifest_bytes)
        try:
            expected_files = required_bundle_files(manifest)
        except ValueError as error:
            raise FLArtifactContractError(str(error)) from error
        if names != expected_files:
            missing = sorted(expected_files - names)
            unexpected = sorted(names - expected_files)
            raise FLArtifactIntegrityError(
                f"FL artifact file set is invalid; missing={missing}, unexpected={unexpected}"
            )
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
        owner_plan_id: str | None = None,
    ) -> FLWorkspaceArtifact:
        self.cleanup_expired()
        weights = np.empty(len(model.state_dict()), dtype=object)
        weights[:] = [value.detach().cpu().numpy() for value in model.state_dict().values()]
        weights_stream = io.BytesIO()
        np.save(weights_stream, weights, allow_pickle=True)
        components = {
            "model.py": base.model_source,
            "model.npy": weights_stream.getvalue(),
        }
        expected_files = required_bundle_files(base.manifest)
        if "scaler.pkl" in expected_files:
            if not base.scaler_source:
                raise RuntimeError("FL base bundle has no preserved scaler source")
            components["scaler.pkl"] = base.scaler_source
        manifest = dict(base.manifest)
        for key in (
            "artifact_role",
            "fl_metadata",
            "model_identity",
            "result_type",
        ):
            manifest.pop(key, None)
        manifest.update(metadata)
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
        if owner_plan_id is not None:
            self._register_owned_directory(owner_plan_id, directory.parents[2])
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

    def publish_round_input(
        self,
        *,
        process_id: str,
        server_nf_instance_id: str,
        round_indicator: int,
        base: LoadedBundle,
        epochs: int,
        owner_plan_id: str | None = None,
    ) -> FLWorkspaceArtifact:
        return self.publish(
            process_id=process_id,
            participant_id=server_nf_instance_id,
            round_indicator=round_indicator,
            role="ROUND_INPUT",
            base=base,
            model=base.model,
            owner_plan_id=owner_plan_id,
            metadata={
                "artifact_role": "ROUND_INPUT",
                "fl_metadata": {
                    "ml_corre_id": process_id,
                    "round_ind": round_indicator,
                    "client_training": {"epochs": epochs},
                },
            },
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
        with self._lock:
            if self._is_pending_release(path):
                return None
            return path if path.is_file() else None

    def open_artifact(
        self,
        process_id: str,
        participant_id: str,
        round_indicator: int,
        role: str,
        digest: str,
    ) -> FLArtifactReader | None:
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
        with self._lock:
            if self._is_pending_release(path):
                return None
            try:
                stream = path.open("rb")
            except (OSError, FLWorkspaceError):
                return None
            self._active_readers[path] = self._active_readers.get(path, 0) + 1
        return FLArtifactReader(self, path, stream)

    def _register_owned_directory(self, plan_id: str, directory: Path) -> None:
        normalized = normalize_plan_id(plan_id)
        direct_child = self._direct_child(directory.name)
        if directory.resolve() != direct_child:
            raise FLWorkspaceError("FL artifact ownership path is not a direct workspace child")
        with self._lock:
            self._prune_released_plans_locked()
            released = normalized in self._released_plan_ids
            if released:
                self._pending_release.add(direct_child)
            if direct_child in self._pending_release:
                error = FLWorkspaceError("FL artifact owner was already released")
            else:
                self._owned_directories.setdefault(normalized, set()).add(direct_child)
                return
        self._release_directories({direct_child})
        raise error

    def _prune_released_plans_locked(self) -> None:
        now = time.monotonic()
        self._released_plan_ids = {
            plan_id: deadline
            for plan_id, deadline in self._released_plan_ids.items()
            if deadline > now
        }

    def _release_reader(self, path: Path) -> None:
        with self._lock:
            count = self._active_readers.get(path, 0)
            if count <= 1:
                self._active_readers.pop(path, None)
            else:
                self._active_readers[path] = count - 1
            ready = {
                directory
                for directory in self._pending_release
                if not self._has_active_reader(directory)
            }
        self._release_directories(ready)

    def _release_directories(self, directories: set[Path]) -> set[Path]:
        failures: set[Path] = set()
        for directory in directories:
            with self._lock:
                if self._has_active_reader(directory):
                    continue
            try:
                self._delete_direct_child(directory)
            except (OSError, FLWorkspaceError):
                failures.add(directory)
                with self._lock:
                    self._cleanup_failures[directory] = time.time()
                continue
            with self._lock:
                self._pending_release.discard(directory)
                self._cleanup_failures.pop(directory, None)
        return failures

    def _retry_cleanup_failures(self, *, cutoff: float | None = None) -> None:
        threshold = time.time() if cutoff is None else cutoff
        with self._lock:
            ready = {
                path
                for path, failed_at in self._cleanup_failures.items()
                if failed_at <= threshold and not self._has_active_reader(path)
            }
        self._release_directories(ready)

    def _has_active_reader(self, directory: Path) -> bool:
        return any(
            count > 0 and path.is_relative_to(directory)
            for path, count in self._active_readers.items()
        )

    def _is_pending_release(self, path: Path) -> bool:
        return any(path.is_relative_to(directory) for directory in self._pending_release)

    def _direct_child(self, name: str) -> Path:
        child = (self._root / name).resolve()
        root = self._root.resolve()
        if child.parent != root:
            raise FLWorkspaceError("FL workspace path is not a direct child")
        return child

    def _delete_direct_child(self, child: Path) -> None:
        direct_child = self._direct_child(child.name)
        if child.resolve() != direct_child:
            raise FLWorkspaceError("FL workspace cleanup target is not a direct child")
        if direct_child.is_dir():
            shutil.rmtree(direct_child)
        else:
            direct_child.unlink(missing_ok=True)

    def _validate_workspace_root(self) -> None:
        root = self._root.resolve()
        repository_root = Path(__file__).resolve().parents[3]
        if (
            root == Path(root.anchor)
            or Path.cwd().resolve().is_relative_to(root)
            or repository_root.is_relative_to(root)
        ):
            raise FLWorkspaceError("FL workspace root is unsafe")


def validate_model_compatibility(base: LoadedBundle, candidate: LoadedBundle) -> None:
    contract_fields = (
        "workload_profile",
        "analytics_event",
        "model_interoperability",
        "runtime_compatibility",
        "model",
        "inference",
    )
    if any(base.manifest.get(field) != candidate.manifest.get(field) for field in contract_fields):
        raise FLArtifactContractError("FL artifact model contract is incompatible")
    base_state = base.model.state_dict()
    candidate_state = candidate.model.state_dict()
    if set(base_state) != set(candidate_state):
        raise FLArtifactContractError("FL artifact parameter keys are incompatible")
    for name, value in base_state.items():
        other = candidate_state[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise FLArtifactContractError("FL artifact parameter shape or dtype is incompatible")


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_download_parent(path: Path, workspace_root: Path) -> None:
    parent = path.parent
    process_directory = parent.parent
    for candidate in (parent, process_directory):
        if candidate == workspace_root:
            break
        try:
            candidate.rmdir()
        except OSError:
            break


def _validated_manifest(manifest_bytes: bytes | None) -> dict[str, object]:
    if manifest_bytes is None:
        raise FLArtifactIntegrityError("FL artifact is missing config.json")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FLArtifactContractError("FL artifact config.json is invalid") from error
    if not isinstance(manifest, dict):
        raise FLArtifactContractError("FL artifact config.json must contain an object")
    unsupported = sorted(
        field for field in ("bundle_schema_version", "file_digests") if field in manifest
    )
    if unsupported:
        raise FLArtifactContractError(f"unsupported manifest field: {unsupported[0]}")
    try:
        profile = workload_profile(manifest)
    except ValueError as error:
        raise FLArtifactContractError(str(error)) from error
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
    if profile is WorkloadProfile.UE_COMMUNICATION_FORECASTING:
        if not isinstance(manifest.get("analytics_event"), str) or not manifest["analytics_event"]:
            raise FLArtifactContractError("FL artifact analytics_event is required")
        expected_names = {
            manifest.get("MODEL_SCRIPT"),
            manifest.get("MODEL_PATH"),
            manifest.get("SCALER_PATH"),
        }
        if expected_names != {"model.py", "model.npy", "scaler.pkl"}:
            raise FLArtifactContractError("FL artifact component filenames are invalid")
    else:
        if manifest.get("analytics_event") != IMAGE_CLASSIFICATION_EVENT:
            raise FLArtifactContractError(
                f"image classification artifact requires {IMAGE_CLASSIFICATION_EVENT}"
            )
        if (
            manifest.get("MODEL_SCRIPT") != "model.py"
            or manifest.get("MODEL_PATH") != "model.npy"
            or "SCALER_PATH" in manifest
        ):
            raise FLArtifactContractError("FL artifact component filenames are invalid")
        try:
            validate_image_manifest(manifest)
        except ValueError as error:
            raise FLArtifactContractError(str(error)) from error
    return manifest
