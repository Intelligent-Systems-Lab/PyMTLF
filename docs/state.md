# Durable State and Reconciliation

PyMTLF uses SQLite schema version 1 at the configured `database_path`. The
default is `data/mtlf-state.sqlite3`. Runtime state and published artifacts are
excluded from git.

The journal owns three durable concerns:

- `model_generation_state` stores last allocated and confirmed active
  generation per `(provider_id, model_unique_id)`.
- `provision_events` binds stable event IDs and monotonically allocated target
  generations to immutable artifact digests.
- `training_job_terminal_state` preserves terminal training evidence.

Artifact bytes are validated and atomically published before a generation is
allocated. SQLite allocation uses `BEGIN IMMEDIATE`, performs no network or
model work while holding the transaction, and never reuses a committed target
generation. A rollback reuses an older digest under a newly allocated
generation. Only an `APPLIED` result updates confirmed active generation and
digest.

Provision states are `PENDING_DELIVERY`, `PENDING_APPLY`, `APPLIED`, `FAILED`,
`STALE`, `NO_MATCH`, and `CONFLICT`. Duplicate event IDs with the same canonical
payload or complete terminal result are idempotent; changed evidence is a
conflict. Terminal evidence persists the observed active generation and digest,
affected runtime count, completion time, and a canonical result digest in
addition to status and failure information.

At startup, no pending events means reconciliation is ready. Pending events are
queried through an injected internal state-query protocol when that capability
is available. Phase 1 supplies only fakes for this protocol; the real NWDAF HTTP
client is added with the Phase 3 route. A confirmed target generation and
digest closes the event as `APPLIED`; base generation, no match, or an
unavailable dependency stays pending. A target digest mismatch or a generation
ahead of the target records durable conflict evidence. Diverged runtimes fail
closed without choosing a winner.

Liveness reports only process responsiveness. Readiness additionally requires
the current database schema, writable artifact root, and resolved startup
reconciliation. Database and artifact probes run on each readiness request so a
post-startup storage failure returns `503`. A pending row without a state-query
implementation also returns `503`. Reconciliation work is app-owned, cancelled
at shutdown, and bounded by the configured shutdown timeout. An unexpected
reconciliation error is logged and leaves the service unresolved and not ready.

Phase 1 does not run automatic artifact garbage collection. The repository
exposes only a protected-delete primitive; all successfully published artifacts
are conservatively retained until later model-apply and rollback retention work
defines safe cleanup.
