# PyMTLF

PyMTLF is the private Model Training Logical Function backend used by this
project's Go NWDAF. It is not an independently registered 3GPP network
function. The containing Go NWDAF owns public SBI resources, NRF interaction,
and peer-NWDAF transport; PyMTLF owns model lifecycle, training, federated
learning coordination, local artifacts, and experiment records.

## Current Capabilities

- local retraining for UE communication forecasting;
- flat federated learning with Server and Client engines;
- recursive hierarchical federated learning with Root, Intermediate, and Leaf
  behavior assigned by protocol relationships rather than fixed process types;
- MNIST and CIFAR-10 image-classification workloads backed by local `.npz`
  shards;
- protocol-driven topology formation, participant policy, branch replacement,
  and direct Leaf reparenting;
- model provision and accuracy-monitoring resources behind the containing Go
  NWDAF;
- ADRF-backed data retrieval and ML model publication;
- immutable model bundles and temporary round artifacts; and
- node-local JSONL experiment recording, validation metrics, and final-model
  export.

## Runtime Architecture

One PyMTLF process belongs to one Go NWDAF process. At startup it reads the
containing NWDAF identity and advertised FL capability from:

```http
GET /internal/v1/nwdaf-context
```

PyMTLF refuses readiness when its enabled FL engines do not match the
containing NWDAF capability. A new PyMTLF process starts with empty volatile
subscription, round, and FL procedure state. Durable model artifacts, model
catalog state, and publication checkpoints are stored separately.

The FL roles are enabled by configuration:

| Configuration | Enabled behavior |
| --- | --- |
| `runtime.mode: local` | Local data retrieval, training, validation, model publication, provision, and monitoring. |
| `federated_learning.server` | FL Server engine, participant coordination, aggregation, publication, provision, and monitoring. |
| `federated_learning.client` | FL Client engine, local preparation, training, validation, and upstream notifications. |
| Both `server` and `client` | Intermediate-capable process: Client toward its parent and Server toward its direct children. |
| `federated_learning.orchestration` | Top-level flat or hierarchical coordinator and optional manual training trigger. |

Hierarchy roles are not fixed configuration modes. The Root sends a recursive
`flTopology` instruction through Model Training subscriptions. An Intermediate
process applies the instruction to its direct children, reports the realized
subtree with `flTopologyReport`, aggregates child results, and reports its
domain result upstream.

## Configuration Profiles

Tracked profiles are examples and must be adapted for a real deployment:

| File | Purpose |
| --- | --- |
| `config/local.yaml` | Standalone local retraining. |
| `config/fl-server.yaml` | Flat FL Server. |
| `config/fl-client.yaml` | FL Client using consumer-subscription data collection. |
| `config/fl-server-client.yaml` | Combined Server and Client engines for an Intermediate node. |
| `config/fl-server-hierarchy.yaml` | Root-side static hierarchical orchestration with a private manual trigger. |
| `config/fl-client-image-classification.yaml` | Local-shard MNIST example; change the dataset and interoperability identifier for CIFAR-10. |
| `config/fl-client-private-collection.yaml` | FL Client with private operator-triggered data collection. |
| `config/fl-server-client-private-collection.yaml` | Combined Intermediate-capable profile with private data collection. |
| `config/topology/hierarchical-topology.yaml` | Example Root topology, participant policies, FedProx strategy, reporting cadence, and branch-failure action. |

`federated_learning.server.max_active_processes` is currently required to be
`1`. The FL workspace is temporary and must not overlap durable artifact,
publication, model-state, or experiment-record directories.

## Hierarchical FL Contract

PyMTLF accepts the Release 18-shaped Model Training fields implemented in
`src/py_mtlf/wire/ml_model_training.py` and the following project-defined JSON
properties:

- `flTopology`: recursive topology instruction;
- `flTopologyReport`: recursive realized-topology report;
- `retainedResultReq`: request for a retained result; and
- `retainedResultStatus`: retained-result outcome.

The current experiments do not rely on retained-result recovery. The fields
remain implemented in the current protocol model.

Each direct parent-child relationship has its own Model Training subscription
ID and notification correlation. A shared `mlCorreId` correlates the local
relationships that belong to one hierarchical FL procedure. The receiving Go
NWDAF creates the subscription resource ID and passes it to its PyMTLF in the
`X-NWDAF-Subscription-Id` header.

The example topology supports:

- ordered candidates through `priority`;
- optional additional discovery;
- minimum available and training-node requirements;
- fractional participant selection;
- completion and failure acceptance policy;
- FedProx with sample-weighted aggregation;
- per-node `reportAfter` cadence; and
- `replace_branch` or `reparent_leaves_to_root` behavior after a Branch
  failure.

## Workloads and Model Bundles

UE communication forecasting uses the existing bundle family containing model
configuration, weights, Python model definition, and scaler state.

MNIST and CIFAR-10 use a shared small-CNN architecture with dataset-specific
initial weights. Image clients read a deployment-mounted `.npz` file containing
`images` and `labels`; the runtime does not download or repartition the
dataset. Image-classification bundles do not contain a scaler.

Import an image seed before using it in a Root profile:

```bash
uv run python tools/import_seed_model.py \
  --config config/fl-client-image-classification.yaml \
  --source seed_models/image_classification/mnist \
  --model-id 1001 \
  --model-interoperability pymtlf-image-classification-mnist
```

The command writes an immutable bundle under `storage.artifact_root` and prints
its artifact key. Configure that key in the matching Root seed-model entry.

## Experiment Records

When `federated_learning.experiment_recording` is configured, each node writes:

```text
<record-directory>/<mlCorreId>/observations.jsonl
```

Records include Model Training operations, topology and repair decisions,
accepted-round participant sets, model evaluations, and final-artifact events.
Each file contains observations made by that node only. Optional local
validation records accuracy and loss without changing Model Training messages.

A successful Root procedure also copies the final accepted model to:

```text
<record-directory>/<mlCorreId>/final-model.tar.gz
```

## Development

PyMTLF requires Python 3.12 or later. The Linux x86-64 dependency profile uses
PyTorch 2.5.1 from the CUDA 12.1 wheel index.

```bash
uv sync --dev
uv run python run.py --config config/local.yaml
uv run pytest -q
uv run ruff check .
```

The configured listener defaults to `127.0.0.1:9092`. FastAPI exposes
`/openapi.json`, `/docs`, and `/redoc` while the process is running.

See [`docs/api.md`](docs/api.md) for the complete current private HTTP surface,
route availability, Model Training resource semantics, and outbound Go NWDAF
dependencies.
