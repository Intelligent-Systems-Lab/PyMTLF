import argparse
import json
import tempfile
from pathlib import Path

from py_mtlf.config import load_settings
from py_mtlf.core.artifacts import ArtifactRepository
from py_mtlf.core.seed_import import build_seed_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and publish a deterministic seed model bundle"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-id", required=True, type=int)
    parser.add_argument("--event")
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
        build_seed_bundle(
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
