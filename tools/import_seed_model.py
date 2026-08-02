import argparse
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path

from py_mtlf.config import load_settings
from py_mtlf.core.artifacts import REQUIRED_BUNDLE_FILES, ArtifactRepository


def _read_source(source: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    files = {}
    for name in REQUIRED_BUNDLE_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"seed source is missing {name}")
        files[name] = path.read_bytes()
    try:
        config = json.loads(files.pop("config.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("seed config.json is invalid") from error
    if not isinstance(config, dict):
        raise ValueError("seed config.json must contain an object")
    return config, files


def _build_bundle(
    source: Path,
    output: Path,
    *,
    model_id: int,
    event: str,
    model_interoperability: str,
) -> None:
    config, components = _read_source(source)
    config.update(
        {
            "bundle_schema_version": "1.0",
            "model_identity": {"model_unique_id": model_id},
            "analytics_event": event,
            "model_interoperability": model_interoperability,
            "runtime_compatibility": config.get(
                "runtime_compatibility",
                {"python": ">=3.12", "framework": "torch"},
            ),
            "file_digests": {
                name: hashlib.sha256(content).hexdigest() for name, content in components.items()
            },
        }
    )
    files = {
        "config.json": json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        **components,
    }
    with (
        output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(content))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and publish a deterministic seed model bundle"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-id", required=True, type=int)
    parser.add_argument("--event", default="UE_COMMUNICATION")
    parser.add_argument("--model-interoperability", required=True)
    args = parser.parse_args()

    settings = load_settings(args.config)
    repository = ArtifactRepository(
        settings.storage.artifact_root,
        settings.artifact,
    )
    repository.open()
    with tempfile.TemporaryDirectory(prefix="py-mtlf-seed-") as temporary:
        bundle = Path(temporary) / "seed.tar.gz"
        _build_bundle(
            args.source,
            bundle,
            model_id=args.model_id,
            event=args.event,
            model_interoperability=args.model_interoperability,
        )
        metadata = repository.publish(bundle)
    print(
        json.dumps(
            {
                "artifact_key": metadata.key,
                "model_id": args.model_id,
                "url": metadata.url,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
