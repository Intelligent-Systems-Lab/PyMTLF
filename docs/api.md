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

## Endpoint Summary

| Caller | Method and path | Purpose | Success |
| --- | --- | --- | --- |
| Go or operator | `GET /health/live` | Process liveness | `200` |
| Go | `GET /health/ready` | Artifact readiness and process identity | `200` or `503` |
| Go | `POST /internal/v1/sync` | Replace the recoverable containing-NWDAF snapshot | `200` |
| PyAnLF through Go | `POST /internal/v1/ml-model-provision/subscriptions` | Create a provision resource | `201` |
| PyAnLF through Go | `PUT /internal/v1/ml-model-provision/subscriptions/{id}` | Replace a provision resource | `200` |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-provision/subscriptions/{id}` | Delete a provision resource | `204` |
| PyAnLF through Go | `POST /internal/v1/ml-model-monitor/registrations` | Register one READY model-use scope | `201` |
| PyAnLF through Go | `DELETE /internal/v1/ml-model-monitor/registrations/{id}` | Deregister a model-use scope | `204` |
| Go | `POST /internal/v1/ml-model-monitor/notifications` | Deliver a correlated accuracy notification | `204` |
| Go | `POST /internal/v1/adrf-data-management/retrieval-notifications` | Deliver a complete ADRF retrieval notification | `204` |
| PyAnLF | `GET /internal/v1/artifacts/{sha256}` | Download an immutable model bundle | `200` |

JSON Model Provision and Monitor errors use `application/problem+json`.
Malformed standard-shaped bodies return `400`; unknown resources or
correlations return `404`. Resource creation returns an owner-generated UUID
in `Location`.

## Health And Sync

`GET /health/ready` returns the current `processInstanceId` and artifact
status. A configured seed catalog is validated during startup. A missing
artifact, invalid archive, or manifest identity mismatch prevents readiness
instead of producing a fake model URL.

`POST /internal/v1/sync` carries:

- containing NWDAF identity and Go internal callback base URI
- accepted Events Subscription and SMF collection-resource snapshots
- current `trainingDataSource` (`adrf`, `mongodb`, or `unavailable`)
- Model Provision subscription snapshots
- Model Monitor registration snapshots
- MTLF-destined Model Monitor subscription projections, including the private
  `ownerRegistrationId` needed to distinguish an active resource from an
  orphan after process restart

Sync does not carry MongoDB credentials, ADRF endpoints, raw observations,
model bytes, fetch instructions, or accuracy-policy baseline state. Provision and monitor
control intent is restored; the volatile WAPE baseline intentionally restarts
empty.

## Initial Model Provision

The request and accepted representation use the Release 18
`NwdafMLModelProvSubsc` shape. PyMTLF resolves each `mLEventSubscs` entry
against its configured seed catalog.

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
the synced Events and SMF resources, fixes one historical time window, and
uses the synced source without cross-source fallback. Every required scope
must contain at least one valid UPF record before a `READY` snapshot is
published. `READY` keeps the model retrain-in-flight until the local-training
workflow consumes it. Local training, artifact publication, generation
advancement, and updated-model reprovision are not active yet.

## Historical Dataset Retrieval

In ADRF mode, PyMTLF independently resolves `nadrf-datamanagement` through the
containing Go NWDAF's generic NRF proxy or uses `adrf.configured_endpoint`.
For each accepted SMF collection resource it creates a Release 18-shaped
retrieval subscription through Go, accepts complete callbacks on the endpoint
listed above, and directly issues
`GET /nadrf-datamanagement/v1/data-store-records?fetch-correlation-ids=...`.
Go never receives dataset bytes. The workspace ADRF V0 terminal callback with
an empty ID list is accepted only when `terminationReq=true`; it produces a
zero-data result and does not relax the required-scope completeness rule.

The pinned workspace free5GC NRF accepts ADRF registration but its older NF
Discovery schema rejects `target-nf-type=ADRF`. Use configured mode with that
build. NRF mode remains available for Release 18-compatible NRF
implementations and does not silently fall back to NF Management listing.

In MongoDB mode, PyMTLF opens the configured collection read-only and queries
distinct accepted SUPIs with the same inclusive time window. Only documents
with a non-empty standard `dataNotif.upfEventNotifs` alternative qualify.
PyMTLF does not create indexes, write records, query legacy correlation IDs,
or merge data from ADRF.

## Outbound Dependency

PyMTLF calls the containing Go NWDAF for monitor and ADRF control resources:

| Purpose | Method and Go path | Required success |
| --- | --- | --- |
| Create monitor subscription | `POST /internal/v1/ml-model-monitor/subscriptions` | `201`, `Location`, JSON |
| Delete monitor subscription | `DELETE /internal/v1/ml-model-monitor/subscriptions/{id}` | `204`; `404` is terminal cleanup |
| Create ADRF retrieval subscription | `POST /internal/v1/adrf-data-management/data-retrieval-subscriptions` | `201`, `Location`, JSON |
| Delete ADRF retrieval subscription | `DELETE /internal/v1/adrf-data-management/data-retrieval-subscriptions/{id}` | `204`; peer `404` is terminal cleanup |

Create also sends the private
`X-NWDAF-Monitor-Registration-Id` header. The request body remains the
Release 18 `MLModelMonitorSub` representation; Go stores the header only in
its process-local sync mirror. On restart, PyMTLF accepts a restored
subscription as active only when this owner identity still exists and its
standard scope still matches. Otherwise the resource is isolated as an
orphan and deleted through the same Go path before it can update policy.

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
