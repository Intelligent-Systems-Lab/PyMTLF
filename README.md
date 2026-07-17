# PyMTLF

PyMTLF is the private MTLF backend used by the local NWDAF implementation. It is
not a standalone 3GPP network function. NWDAF Go remains the owner of all
standard SBI and ADRF communication.

Phase 1 provides the service lifecycle, internal durable-state models, durable
generation journal, startup reconciliation seam, and immutable model artifact
repository. Accuracy policy and training are intentionally not active yet.

## Development

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python run.py --config config/config.yaml
```

The default listener is `127.0.0.1:9092`. Runtime state is stored below
`data/`, which is excluded from git.
