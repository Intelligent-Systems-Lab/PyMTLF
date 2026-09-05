import gzip
import io
import json
import tarfile
from pathlib import Path

from py_mtlf.core.workloads import (
    IMAGE_CLASSIFICATION_EVENT,
    WorkloadProfile,
    required_bundle_files,
    workload_profile,
)


def read_seed_source(source: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError("seed source is missing config.json")
    try:
        config = json.loads(config_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("seed config.json is invalid") from error
    if not isinstance(config, dict):
        raise ValueError("seed config.json must contain an object")
    files = {}
    for name in required_bundle_files(config) - {"config.json"}:
        path = source / name
        if not path.is_file():
            raise ValueError(f"seed source is missing {name}")
        files[name] = path.read_bytes()
    return config, files


def build_seed_bundle(
    source: Path,
    output: Path,
    *,
    model_id: int,
    event: str | None,
    model_interoperability: str,
) -> None:
    config, components = read_seed_source(source)
    profile = workload_profile(config)
    config.update(
        {
            "model_identity": {"model_unique_id": model_id},
            "model_interoperability": model_interoperability,
            "runtime_compatibility": config.get(
                "runtime_compatibility",
                {"python": ">=3.12", "framework": "torch"},
            ),
        }
    )
    if profile is WorkloadProfile.UE_COMMUNICATION_FORECASTING:
        config["analytics_event"] = event or "UE_COMMUNICATION"
    elif event not in {None, IMAGE_CLASSIFICATION_EVENT}:
        raise ValueError(
            f"image classification seed requires {IMAGE_CLASSIFICATION_EVENT}"
        )
    else:
        config["analytics_event"] = IMAGE_CLASSIFICATION_EVENT
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
