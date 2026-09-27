# PyMTLF HTTP API

## Scope and Ownership

PyMTLF is a private backend of one containing Go NWDAF. These routes are not
public 3GPP SBI endpoints. The Go NWDAF owns public SBI paths, peer transport,
subscription resource creation, NRF access, and ADRF forwarding. PyMTLF owns
the backend state and decisions reached through the private routes below.

The implementation is authoritative:

- route registration: `src/py_mtlf/app.py`;
- route handlers: `src/py_mtlf/api/`;
- wire models: `src/py_mtlf/wire/`; and
- runtime owners: `src/py_mtlf/core/`.

FastAPI exposes the exact running schema through `/openapi.json`, `/docs`, and
`/redoc`.

## Route Availability

| Runtime capability | Routes enabled |
| --- | --- |
| Always | readiness, artifact download, training-data descriptors, ADRF retrieval notifications |
| `runtime.mode: local` | Model Provision and Model Monitor routes |
| FL Server engine | Model Provision, Model Monitor, Model Training notification ingress |
| FL Client engine | Model Training subscription create, replace, patch, and delete |
| Private training-data collection | collection-control and UPF callback routes |
| Top-level coordinator with private trigger enabled | federated training-request routes |

The shared Model Training router is mounted when either FL engine is enabled.
An operation returns `503 SERVICE_NOT_AVAILABLE` when its required engine is
not enabled in that process.

## Complete Inbound Route Summary

### Readiness and artifacts

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| Operator or containing Go NWDAF | `GET /health/ready` | `200` or `503` | Report process identity, runtime mode, artifact readiness, enabled engines, advertised engines, and capability consistency. |
| Model consumer | `GET /internal/v1/artifacts/{artifact_key}` | `200` | Download an immutable completed-model bundle. |
| FL peer | `GET /internal/v1/fl-artifacts/{process_id}/{participant_id}/{round_indicator}/{role}/{digest}` | `200` | Download a temporary round artifact by its exact identity and digest. |

Completed bundles use immutable cache headers and a SHA-256 ETag. Round
artifacts are temporary workspace objects and are removed with their FL
lifecycle.

### Training-data inputs

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| PyAnLF through Go | `PUT /internal/v1/anlf/training-data-descriptors/{descriptor_id}` | `204` | Create or replace a local training-data descriptor. |
| PyAnLF through Go | `DELETE /internal/v1/anlf/training-data-descriptors/{descriptor_id}` | `204` | Remove a descriptor. |
| Go NWDAF | `POST /internal/v1/adrf-data-management/retrieval-notifications` | `204` | Deliver an ADRF retrieval notification to the dataset coordinator. |

### Private training-data collection

These routes exist only when an FL Client selects the `private_api` collection
trigger.

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| Operator | `POST /internal/v1/training-data-collections` | `202` | Start one configured collection request. |
| Operator | `GET /internal/v1/training-data-collections/{request_id}` | `200` | Read collection progress and outcome. |
| Operator | `DELETE /internal/v1/training-data-collections/{request_id}` | `202` | Request collection cancellation and cleanup. |
| Event Exposure producer | `POST /callbacks/upf-event-exposure` | `204` | Deliver data collected for an active request. |

### Model Provision

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| PyAnLF through Go | `POST /internal/v1/ml-model-provision/subscriptions` | `201` | Create a provision resource. |
| PyAnLF through Go | `PUT /internal/v1/ml-model-provision/subscriptions/{subscription_id}` | `200` | Replace a provision resource. |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-provision/subscriptions/{subscription_id}` | `204` | Delete a provision resource and cancel pending delivery. |

The request and response use the implemented Release 18-shaped
`NwdafMLModelProvSubsc` representation. Immediate reporting may include the
matching model notification in the response; otherwise PyMTLF dispatches the
notification asynchronously through Go.

### Model Monitor

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| PyAnLF through Go | `POST /internal/v1/ml-model-monitor/registrations` | `201` | Register one local model-use scope. |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-monitor/registrations/{registration_id}` | `204` | Remove the scope and reconcile its downstream monitor subscription. |
| Go NWDAF | `POST /internal/v1/ml-model-monitor/notifications` | `204` | Deliver an accuracy-monitoring notification. |

Monitor notifications are correlated by `notifCorrId`. In local mode a
degradation decision may start local retraining. In federated mode it may
trigger the configured top-level coordinator only when the federated
degradation trigger is enabled.

### Model Training

| Caller | Method and path | Success | Required engine |
| --- | --- | --- | --- |
| Peer NWDAF through Go | `POST /internal/v1/ml-model-training/subscriptions` | `201` | FL Client |
| Peer NWDAF through Go | `PUT /internal/v1/ml-model-training/subscriptions/{subscription_id}` | `200` | FL Client |
| Peer NWDAF through Go | `PATCH /internal/v1/ml-model-training/subscriptions/{subscription_id}` | `200` | FL Client |
| Peer NWDAF through Go | `DELETE /internal/v1/ml-model-training/subscriptions/{subscription_id}` | `204` | FL Client |
| Peer NWDAF through Go | `POST /internal/v1/ml-model-training/notifications` | `204` | FL Server |

The receiving Go NWDAF creates the public Model Training subscription resource
ID. On create it calls PyMTLF with:

```http
POST /internal/v1/ml-model-training/subscriptions
X-NWDAF-Subscription-Id: <canonical UUIDv4>
Content-Type: application/json
```

PyMTLF validates the identifier, uses it as the resource key, and returns the
accepted representation with a private `Location` ending in the same
identifier. It does not generate a second Model Training subscription ID.
Replace, patch, and delete use that same identifier in the private path.

The request representation includes the implemented Release 18 Model Training
fields plus the current project-defined properties:

| Property | Meaning |
| --- | --- |
| `flTopology` | Recursive node instruction carrying candidates, direct-child policy, training strategy, reporting cadence, and descendants. |
| `retainedResultReq` | Requests the latest retained result for the correlated procedure. The current formal experiments do not depend on it. |

Model Training notifications may add:

| Property | Meaning |
| --- | --- |
| `flTopologyReport` | Recursive report of realized direct-child and descendant state. |
| `retainedResultStatus` | Result of a retained-result request. |

The current topology policy can express additional-candidate authority,
additional-candidate priority, selection method, minimum available nodes,
training fraction, minimum training nodes, failure acceptance, and minimum
completion rate. The current strategy requires `method: fedProx`, an
aggregation name, and `methodParameters.proximalMu`. `reportAfter` carries a
positive count and a unit.

Each direct parent-child edge has its own `subscriptionId` and `notifCorreId`.
`mlCorreId` correlates the edge-local resources that belong to one hierarchical
procedure. `roundInd` remains local to the Model Training relationship that
carries it.

### Federated training trigger

These routes exist only when a flat or hierarchical top-level coordinator is
configured and `training_trigger.private_api.enabled` is true.

| Caller | Method and path | Success | Purpose |
| --- | --- | --- | --- |
| Experiment controller | `POST /internal/v1/federated-learning/training-requests` | `202` | Submit one model family for training with a caller-provided UUIDv4 request ID. |
| Experiment controller | `GET /internal/v1/federated-learning/training-requests/{request_id}` | `200` | Read request state, plan identity, progress, result digest, or failure. |

## Outbound Dependencies Through Go NWDAF

PyMTLF does not perform public SBI routing itself. Depending on its enabled
features, it calls private routes on the containing Go NWDAF for:

| Private Go route family | Purpose |
| --- | --- |
| `GET /internal/v1/nwdaf-context` | Read the containing NF identity, API roots, and advertised FL capability. |
| `GET /internal/v1/nrf/nf-instances` | Resolve ADRF, FL participants, or model-monitor peers through NRF discovery. |
| `/internal/v1/ml-model-training/subscriptions...` | Create, update, or remove peer Model Training resources. |
| `/internal/v1/ml-model-monitor/subscriptions...` | Reconcile public Model Monitor subscriptions. |
| `/internal/v1/adrf-data-management/...` | Create and remove retrieval subscriptions or store collected records. |
| `/internal/v1/adrf-mlmodelmanagement/mlmodel-store-records...` | Publish, query, or remove ADRF model records. |
| `/internal/v1/udm-sdm/...` and `/internal/v1/udm-uecm/...` | Resolve group members and serving-SMF registrations for private collection. |
| `/internal/v1/smf-event-exposure/subscriptions...` | Establish and remove Event Exposure subscriptions for private collection. |

Callbacks and peer Model Training messages use the callback URI or peer target
provided by the active resource and topology state. Go owns the actual external
request and returns the peer result to PyMTLF.

## Error Representation

Standard-shaped private operations return `application/problem+json` errors.
Malformed request bodies and candidate extension violations return `400` with
`INVALID_MSG_FORMAT` and, when available, `invalidParams`. Missing resources
return `404 RESOURCE_NOT_FOUND`. Disabled roles or unavailable runtime state
return `503` with the corresponding cause.

Routes outside these standard-shaped families may return the private
`PrivateError` representation. Request validation for standard-shaped private
paths is normalized to HTTP `400`; other FastAPI validation failures use HTTP
`422`.

## Experiment Output

When experiment recording is enabled, PyMTLF writes node-local append-only
records to `<directory>/<mlCorreId>/observations.jsonl`. Record families
include:

- `MODEL_TRAINING_OPERATION` for sent and received create, replace, patch,
  delete, and notification operations;
- candidate, edge, repair, and topology-acceptance decisions;
- `ROUND_AGGREGATION` participant and acceptance outcomes;
- `MODEL_EVALUATION` validation loss and accuracy; and
- `MODEL_ARTIFACT_SAVED` for the Root final model.

The recorder preserves `subscriptionId`, peer NF instance identity, operation
direction, request start time, result time, outcome, cause, relevant message
fields, `roundInd`, and `mlCorreId` when those values exist at the recording
point. Every file contains observations from its local PyMTLF process only.
