import argparse
import logging

import uvicorn

from py_mtlf.app import create_app
from py_mtlf.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the internal MTLF backend")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.log.level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.server.binding_host,
        port=settings.server.port,
    )


if __name__ == "__main__":
    main()
