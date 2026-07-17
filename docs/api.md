# Internal API

PyMTLF uses `/internal/v1` for private versioned APIs. Future state-changing
payloads will carry `contract_version: "1.0"` when their owning phase activates
them. This API is not a 3GPP SBI.

Phase 1 registers only:

- `GET /health/live`
- `GET /health/ready`
- `GET /internal/v1/artifacts/{sha256}`

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

Generation-journal and readiness semantics are documented in `state.md`.
