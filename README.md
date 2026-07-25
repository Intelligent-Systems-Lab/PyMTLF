# PyMTLF

PyMTLF is the private MTLF backend used by the local NWDAF implementation. It is
not a standalone 3GPP network function. NWDAF Go owns standard SBI routing and
the standard ADRF subscription and callback procedures. PyMTLF consumes ADRF
fetch instructions and retrieves the referenced records directly.

The current runtime provides the service lifecycle, health and sync endpoints,
an immutable model artifact repository, a configured seed-model catalog,
standard-shaped Model Provision and ML Model Monitor resources, and a
degradation-only WAPE policy, scope-aware historical dataset retrieval, and
read-only MongoDB fallback access. A bounded CPU trainer consumes READY
datasets, warm-starts the current model, evaluates per-scope and aggregate
WAPE, publishes a new immutable bundle, and reprovisions the current artifact
through the existing Model Provision resource.

Before using the sample configuration, import the initial seed bundle:

```bash
uv run python tools/import_seed_model.py \
  --config config/config.yaml \
  --source seed_models/initial \
  --provider-namespace local-mtlf \
  --model-id 1
```

The version-controlled source bundle is owned by this repository under
`seed_models/initial`. It contains `config.json`, `model.py`, `model.npy`, and
`scaler.pkl`. The import command packages those components into the immutable
runtime repository below `data/artifacts`. Copy the emitted `artifact_key` into
`model_provision.seed_models[].artifact_key` and assign that seed a unique,
non-empty `family_id`. Startup fails if a configured
seed artifact is missing or its manifest identity does not match the
descriptor.

## Development

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python run.py --config config/config.yaml
```

The default listener is `127.0.0.1:9092`. Runtime state is stored below
`data/`, which is excluded from git. The tracked source seed remains under
`seed_models/`; PyAnLF is only a provisioned artifact consumer and does not
provide the initial model source.

See [`docs/api.md`](docs/api.md) for the complete private HTTP surface and the
standard-shaped operations PyMTLF sends through the containing Go NWDAF.
