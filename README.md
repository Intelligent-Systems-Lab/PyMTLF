# PyMTLF

PyMTLF is the private MTLF backend used by the local NWDAF implementation. It is
not a standalone 3GPP network function. NWDAF Go owns standard SBI routing and
the standard ADRF subscription and callback procedures. PyMTLF will consume
ADRF fetch instructions and retrieve the referenced data directly in a later
phase.

The current foundation provides the service lifecycle, health endpoints,
storage-mode selection handshake, and immutable model artifact repository.
Accuracy policy, direct data retrieval, and training are intentionally not
active yet.

## Development

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python run.py --config config/config.yaml
```

The default listener is `127.0.0.1:9092`. Runtime state is stored below
`data/`, which is excluded from git.
