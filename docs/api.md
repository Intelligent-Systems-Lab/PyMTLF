# Internal API

PyMTLF uses `/internal/v1` for private APIs. This API is not a 3GPP SBI.

The current foundation registers:

- `GET /health/live`
- `GET /health/ready`
- `GET /internal/v1/artifacts/{sha256}`
- `POST /internal/v1/data-source-selection`

Accuracy reports, datasets, model-ready events, model apply results, and the
active-state query remain semantic inventory in the transition plan. Their
runtime models and routes are added only in the phase that activates each
operation.

Private errors use this shape:

```json
{
  "code": "ARTIFACT_NOT_FOUND",
  "message": "artifact was not found",
  "retryable": false,
  "correlation_id": "request-or-resource-id"
}
```

Artifact GET responses include exact `Content-Length`, `application/gzip`, a
strong SHA-256 ETag, `X-Artifact-SHA256`, and immutable cache semantics. Artifact
keys are lowercase 64-character SHA-256 values; there is no mutable `latest`
alias or directory listing.

The default deployment binds to loopback and does not provide application-level
authentication or TLS. Non-loopback deployment requires a separate trust and
network-policy decision.

The data-source selection request carries only `availableDataSources`; it does
not expose MongoDB credentials or ADRF fetch instructions. PyMTLF returns its
configured `storageMode` when the available source set satisfies that mode.
