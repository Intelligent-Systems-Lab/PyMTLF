# PyMTLF HTTP API

PyMTLF is the private MTLF backend of a containing Go NWDAF. It is not an
independently registered 3GPP NF. Go owns public
`Nnwdaf_MLModelProvision`/`Nnwdaf_MLModelMonitor` SBI exposure and routes the
same Release 18-shaped representations to the backend.

The implementation is authoritative:

- routes: `src/py_mtlf/api/`
- wire models: `src/py_mtlf/wire/`
- application wiring: `src/py_mtlf/app.py`
- provision and monitor state: `src/py_mtlf/core/`

FastAPI exposes `/docs`, `/redoc`, and `/openapi.json` while running.

Route availability depends on `runtime.mode`:

| Mode | Provision/Monitor routes | Local trainer | Training routes |
| --- | --- | --- | --- |
| `local` | yes | yes | no |
| `fl_server` | yes | no | yes; also coordinates peer clients |
| `fl_client` | no | no | yes |

Readiness, artifact, incremental training-data descriptor, and ADRF callback
routes remain available in all three modes.

## Endpoint Summary

| Caller | Method and path | Purpose | Success |
| --- | --- | --- | --- |
| Go | `GET /health/ready` | Artifact readiness and process identity | `200` or `503` |
| PyAnLF through Go | `PUT /internal/v1/anlf/training-data-descriptors/{id}` | Publish or replace a stored training-data descriptor | `204` |
| PyAnLF through Go | `DELETE /internal/v1/anlf/training-data-descriptors/{id}` | Remove a data descriptor | `204` |
| PyAnLF through Go | `POST /internal/v1/ml-model-provision/subscriptions` | Create a provision resource | `201` |
| PyAnLF through Go | `PUT /internal/v1/ml-model-provision/subscriptions/{id}` | Replace a provision resource | `200` |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-provision/subscriptions/{id}` | Delete a provision resource | `204` |
| PyAnLF through Go | `POST /internal/v1/ml-model-monitor/registrations` | Register one READY model-use scope | `201` |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-monitor/registrations/{id}` | Deregister a model-use scope | `204` |
| Go | `POST /internal/v1/ml-model-monitor/notifications` | Deliver a correlated accuracy notification | `204` |
| Go | `POST /internal/v1/adrf-data-management/retrieval-notifications` | Deliver a complete ADRF retrieval notification | `204` |
| PyAnLF | `GET /internal/v1/artifacts/{sha256}` | Download an immutable model bundle | `200` |
| Peer NWDAF through Go | `/internal/v1/ml-model-training/subscriptions...` | Model Training resource lifecycle | standard create/update/patch/delete results |

JSON Model Provision and Monitor errors use `application/problem+json`.
Malformed standard-shaped bodies return `400`; unknown resources or
correlations return `404`. Resource creation returns an owner-generated UUID
in `Location`.

`GET /health/live` and `POST /internal/v1/sync` are deliberately absent.

## Readiness And Containing NWDAF Context

`GET /health/ready` returns the current `processInstanceId`, `runtimeMode`, and
artifact status. Both ready `200` and not-ready `503` retain the same UUID for
the lifetime of the process. A configured seed catalog is validated during
startup; a missing or invalid artifact prevents readiness.

PyMTLF reads immutable containing-NWDAF information from the corresponding Go
MTLF edge:

```http
GET /internal/v1/nwdaf-context
```

```json
{
  "nfInstanceId": "11111111-1111-4111-8111-111111111111",
  "apiRoot": "http://127.0.0.1:8000",
  "internalApiRoot": "http://127.0.0.1:8091"
}
```

This endpoint supplies identity and origins only. It does not carry resource
snapshots, storage selection, raw data, policy state, or model bytes. A new
process starts with empty volatile provision, monitor, retrieval, training,
and FL state; durable completed model artifacts remain available locally.

## Initial Model Provision

The request and accepted representation use the Release 18
`NwdafMLModelProvSubsc` shape. PyMTLF resolves each `mLEventSubscs` entry
against its configured seed catalog. Every seed has an explicit internal
`family_id`; the family remains stable across retraining while every promoted
artifact receives a new standard `modelUniqueId`.

When `eventReq.immRep` is true and a compatible seed exists, the `201` or `200`
representation includes `mLEventNotifs` with:

- the requested event and notification correlation
- `modelUniqueId`
- the seed's applicability filter and target, when configured
- `mLFileAddr.mLModelUrl` pointing to the immutable artifact endpoint

If one generic seed covers multiple active-demand entries, PyMTLF reports that
model once rather than duplicating the same model notification for every
covered entry.

Without immediate reporting, the accepted resource is returned first and a
standard `NwdafMLModelProvNotif` is delivered asynchronously through Go to the
original notification destination. A no-match request remains a valid
subscription but does not invent an address or start training.

Artifact responses use `application/gzip`, exact `Content-Length`, a strong
SHA-256 ETag, `X-Artifact-SHA256`, `nosniff`, and immutable cache semantics.
There is no mutable `latest` alias or directory listing.

## ML Model Monitoring

PyMTLF owns Model Monitor registrations. Each local registration represents a
READY AnLF model-use scope. A reconciliation worker creates one corresponding
standard `MLModelMonitorSub` through Go; Go routes it to PyAnLF. Registration
create/delete is not blocked on that downstream resource operation, and
transport failures retry with bounded backoff.

Incoming `MLModelMonitorNotify` is located by `notifCorrId`. A valid
notification must contain at least one `modelAccuInfos` or `anaFeedbacks`
entry. The current policy consumes `modelAccuInfos[].deviation` as a WAPE error
ratio:

- missing `deviation` is a liveness report and does not update the baseline
- only the degradation path is active
- reference samples, population standard deviation with `min_std`, the fixed
  WAPE floor, strict z-score comparison, and N-in-M decisions are configured
  under `accuracy_policy`
- each canonical event/filter/target/consumer scope has independent state
- any degraded scope claims one model-level in-flight retrain intent and
  records all active scopes

One model-level retrain intent snapshots the triggering scope and every active
scope for that model. The dataset coordinator resolves those scopes through
current incremental training-data descriptors or the MongoDB fallback, fixes
one historical time window, and never merges the two sources. Every required scope
must contain at least one valid UPF record before a `READY` snapshot is
published. `READY` is atomically claimed by the bounded local-training
coordinator and keeps the model retrain-in-flight until a terminal outcome.

The local trainer:

- converts ADRF-aligned raw notifications into the bundle's fixed ten-feature
  order, summing volume/packet fields and averaging throughput fields per
  timestamp and scope
- reserves the older 20% of each scope as reference validation, uses the newer
  80% for training, and applies a purge gap between the two regions
- fits a new `StandardScaler` only on training-period observations
- warm-starts the current Torch model on CPU with deterministic seeds, Adam,
  and Huber loss
- always records current/candidate per-scope and aggregate WAPE; the
  `training.enforce_performance_gate` switch decides whether regression blocks
  promotion

The triggering scope must be eligible for both training and evaluation.
Other active scopes participate when eligible and otherwise remain recorded
with an exclusion reason in logs and the candidate manifest. Scope drift
during training is logged but does not discard an otherwise valid candidate;
a stale base generation or removed model demand does.

An accepted candidate reserves a provider-wide model ID, is packaged with the
same four-file bundle contract,
reloaded for validation, published under a content-addressed immutable URL,
and atomically promoted in the process-local family catalog. Retired IDs remain
indexed to the family but are never reused during the process lifetime. The
existing Model Provision resources then resolve the new URL at send time.
Notification delivery keeps only the latest desired artifact per resource,
retries retryable failures with capped exponential backoff, and cancels a
stale resource revision. Job completion does not wait for PyAnLF activation
because standard callback `204` only acknowledges acceptance.

After promotion, reports for the retired model ID are ignored. Each scope
starts a fresh baseline only after PyAnLF registers the new model identity and
PyMTLF establishes the corresponding owned subscription/correlation.
Liveness-only reports still describe insufficient data and never signal
activation.

### Periodic monitor watchdog

Each active periodic Monitor subscription records its negotiated `repPeriod`.
A valid notification, including a liveness-only notification without
`deviation`, resets the watchdog. By default the relationship expires after
two missed report periods plus a 30-second grace interval. Expiry performs one
best-effort standard DELETE, clears the local subscription and registration,
and removes the associated accuracy-policy state. It does not stop AnLF
analytics or invalidate a model already loaded by AnLF.

## Historical Dataset Retrieval

A successfully stored collection is announced incrementally with:

```http
PUT /internal/v1/anlf/training-data-descriptors/{descriptor_id}
```

The path ID equals `correlationId`. Its representation carries the standard
`dataSpec`, stored time period, ML event and target scope, source NWDAF,
lifecycle state, and retention time. `adrfInstanceId` is present for
ADRF-backed data and absent for MongoDB-backed data. DELETE removes that
descriptor. The descriptor exists only in PyMTLF memory and is not replayed
after restart.

When a matching current descriptor exists, PyMTLF independently resolves `nadrf-datamanagement` through the
containing Go NWDAF's generic NRF proxy or uses `adrf.configured_endpoint`.
For each accepted SMF collection resource it creates a Release 18-shaped
retrieval subscription through Go, accepts complete callbacks on the endpoint
listed above, and directly issues
`GET /nadrf-datamanagement/v1/data-store-records?fetch-correlation-ids=...`.
The request carries the descriptor's complete `dataSub` plus the requested
`timePeriod`. The workspace ADRF V0 currently selects records using only
`dataSub.smfDataSub.supi`, the time window, and the subscription snapshot
cutoff; it does not structurally match `notifId`, `notifUri`, or the complete
`eventSubs` value.

One callback may contain multiple `fetchCorrIds`. The current interoperability
profile fetches them sequentially, with one identifier in each collection GET,
and expects one `NadrfDataStoreRecord` response. The Release 18 query parameter
can represent multiple identifiers, so this is an ADRF V0 profile restriction
rather than a standard cardinality restriction.

The retrieval target is built from the selected ADRF API root; the callback's
mandatory `fetchUri` is not dereferenced because TS 29.575 defines the
`data-store-records` resource and notes that this URI is not needed by the
consumer for ADRF retrieval. Go never receives dataset bytes. The workspace
ADRF V0 terminal callback with
an empty ID list is accepted only when `terminationReq=true`; it produces a
zero-data result and does not relax the required-scope completeness rule.

The pinned workspace free5GC NRF accepts ADRF registration but its older NF
Discovery schema rejects `target-nf-type=ADRF`. Use configured mode with that
build. NRF mode remains available for Release 18-compatible NRF
implementations and does not silently fall back to NF Management listing.

When no usable ADRF descriptor exists, PyMTLF opens the configured MongoDB
collection read-only and queries
distinct accepted SUPIs with the same inclusive time window. Only documents
with a non-empty standard `dataNotif.upfEventNotifs` alternative qualify.
PyMTLF does not create indexes, write records, query legacy correlation IDs,
or merge data from ADRF.

## Outbound Dependency

PyMTLF calls the containing Go NWDAF for monitor and ADRF control resources:

| Purpose | Method and Go path | Required success |
| --- | --- | --- |
| Read containing NWDAF | `GET /internal/v1/nwdaf-context` | `200` |
| Create monitor subscription | `POST /internal/v1/ml-model-monitor/subscriptions` | `201`, `Location`, JSON |
| Delete monitor subscription | `DELETE /internal/v1/ml-model-monitor/subscriptions/{id}` | `204`; `404` is terminal cleanup |
| Create ADRF retrieval subscription | `POST /internal/v1/adrf-data-management/data-retrieval-subscriptions` | `201`, `Location`, JSON |
| Delete ADRF retrieval subscription | `DELETE /internal/v1/adrf-data-management/data-retrieval-subscriptions/{id}` | `204`; peer `404` is terminal cleanup |

Create also sends the private
`X-NWDAF-Monitor-Registration-Id` header. The request body remains the
Release 18 `MLModelMonitorSub` representation; Go stores the header only in
its process-local route ledger. No registration or subscription is restored
to a replacement PyMTLF process.

The callback URI in the standard subscription points back to
`/internal/v1/ml-model-monitor/notifications`. Go replaces it with its own
internal callback while routing, then restores the PyMTLF URI in the accepted
representation. PyMTLF never calls PyAnLF directly.

## Deployment Boundary

The default listener is `127.0.0.1:9092` over ordinary HTTP. TLS, OAuth
delegation, independent NRF registration, and cross-Go-restart persistence are
outside the current deployment. Runtime artifacts live below `data/`, which is
excluded from git. The reproducible, version-controlled initial bundle source
is owned by PyMTLF under `seed_models/initial`; importing it publishes a
content-addressed runtime artifact. PyAnLF receives only the resulting Model
Provision metadata and downloads the artifact from this service.

Go polls readiness and treats the UUID as the process generation. A same-ID
`503` temporarily blocks new work without resetting routes. A changed UUID or
two consecutive transport failures confirms process loss. Go drains admitted
operations, clears volatile MTLF routes, and waits for a ready process. It does
not terminate existing AnLF analytics and does not replay old runtime state to
the replacement process.
