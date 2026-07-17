import argparse
import json
from pathlib import Path

from py_mtlf.config import load_settings
from py_mtlf.core.artifacts import ArtifactRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a trusted local model artifact")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    repository = ArtifactRepository(settings.storage.artifact_root, settings.artifact)
    repository.open()
    metadata = repository.publish(args.artifact)
    print(
        json.dumps(
            {
                "key": metadata.key,
                "url": metadata.url,
                "size_bytes": metadata.size_bytes,
                "media_type": metadata.media_type,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
