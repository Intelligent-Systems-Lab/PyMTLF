# Internal API

PyMTLF uses `/internal/v1` for private APIs. This API is not a 3GPP SBI.

The current foundation registers:

- `GET /health/live`
- `GET /health/ready`
- `GET /internal/v1/artifacts/{sha256}`
- `POST /internal/v1/sync`

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

The sync request carries the containing NWDAF identity and typed data-source
availability. It does not expose MongoDB credentials, raw data, model artifacts,
or ADRF fetch instructions. Source preference and effective selection remain
empty until the retrieval phase activates that policy.
