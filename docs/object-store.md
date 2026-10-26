# Object-store-backed Mem++

This optional backend addresses [issue #21](https://github.com/AIDAChip-Inc/mem-plus-plus/issues/21).
It keeps the corpus outside compute instances, so a writer or reader can restart on
a fresh container without losing acknowledged memory. PostgreSQL + pgvector remains
the default and is recommended for large corpora, frequent writes, or high query volume.

## What this adds

- S3 and S3-compatible object stores through the normal AWS credential chain.
- Immutable, content-addressed SQLite snapshots and a conditionally updated publication pointer.
- One ingestion writer, many read-only replicas, disposable in-memory databases.
- The existing Python recall/write, browse, supersede, merge and consolidation APIs.
- The authenticated `/add` and `/search` HTTP service, including durable request receipts.
- A filesystem adapter for local development or a volume with POSIX locking, atomic
  rename and fsync guarantees. Generic object-backed/FUSE mounts are **not** assumed
  to satisfy those guarantees. Access those stores directly through the S3 adapter.

This reduces the infrastructure needed for smaller organizational assistants and
batch/task deployments: durable memory no longer requires an operated PostgreSQL
service or an irreplaceable local volume. Replicas share a corpus by using the same
bucket prefix. Retained snapshots also make versions portable and recoverable.
It does not guarantee lower cost or better latency than PostgreSQL: each session
downloads a whole snapshot, and each writer commit uploads a whole new snapshot.

## Configure a backend

```bash
uv sync --extra object-store
export MEMORY_STORAGE_BACKEND=object_store
export MEMORY_OBJECT_STORE_URI=s3://YOUR_PRIVATE_BUCKET/mempp/organization-a
export MEMORY_OBJECT_STORE_ROLE=writer
export MEMORY_USER=alice
```

Use an existing private bucket. Configure AWS credentials with your normal AWS
profile, workload role, or environment. No credentials belong in the URI or source
files. The extra includes the AWS CRT dependency needed by AWS login profiles.
For a compatible service, additionally set `MEMORY_S3_ENDPOINT_URL` to its endpoint
and configure its credentials/region through the AWS SDK environment.

The service must implement conditional `PutObject` with both `If-Match` and
`If-None-Match`, atomic object replacement, and read-after-write consistency. These
are requirements, not optional optimizations. There is no unconditional-write
fallback when a compatible endpoint does not support them. See
[S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

For a local test, replace the URI with `file:///absolute/path/to/corpus`. This
directory is durable state, not a disposable cache. Do not share a live SQLite
database on a mount: the adapter writes immutable snapshots and publishes pointers.
The filesystem adapter requires POSIX and is not available on native Windows.

Unset `MEMORY_STORAGE_BACKEND`, or set it to `postgres`, to use the existing
`MEMORY_DATABASE_URL` configuration. A backend switch selects a separate corpus;
this release does not migrate an existing PostgreSQL corpus automatically.

## Minimal Python example

Run ingestion once with the writer environment above:

```bash
uv run --extra object-store python - <<'PY'
from memory import store_facts_verbatim

print(store_facts_verbatim("assistant", [{
    "summary": "Project Atlas approved the revised launch plan.",
    "tags": ["atlas", "launch"],
    "occurred_at": "2026-10-26",
}]))
PY
```

On a fresh instance with the same URI, credentials and contributor identity:

```bash
export MEMORY_OBJECT_STORE_ROLE=reader
uv run --extra object-store python - <<'PY'
from memory import recall_facts
print(recall_facts("assistant", "What is the Atlas launch plan?"))
PY
```

There is no persistent local database to copy. The embedding model may download on
first use; its local cache is disposable. Use the existing `project_scope(...)`
context for explicit project/customer scopes. Contributor identity, project
sharing and source tags retain their existing behavior; a scope selector is not
authentication. Separate trust domains should use distinct prefixes and credentials.

Reader is the default role. A missing publication is an error for readers; initialize
the writer first. Writer ingestion commits publish automatically. Direct ORM users
must commit with `session.commit()` or a `with session.begin():` block:

```python
from memory.db import get_session

# In writer mode, this also initializes an empty corpus for readers.
with get_session() as session:
    session.commit()
```

Use `get_session()` for this backend. A process-wide `get_engine()` cannot represent
independent published snapshots and is intentionally rejected. Benchmark-specific
integrations that explicitly require PostgreSQL still require PostgreSQL.

## HTTP deployment

Use the existing authenticated evidence API with the same storage configuration:

```bash
export MEMORY_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MEMORY_OBJECT_STORE_ROLE=writer
uv run --extra object-store python -m memory.aml_api --host 127.0.0.1 --port 8787
```

Writer startup publishes an empty corpus if needed. Start reader replicas after
that, with the same URI and `MEMORY_OBJECT_STORE_ROLE=reader`. `/search` and
`/health` work on replicas; `/add` returns 403. Route all ingestion to the writer.
Successful `/add` responses cover the entire request, including its idempotency
receipt. Retry a 409 publication conflict with the **same** request ID and payload;
reuse with a different payload is rejected. Network/server failures may have an
unknown outcome, so use the same retry rule for those failures too.

For containers:

```bash
docker build -f Dockerfile.snapshots -t mempp-snapshots .
docker run --rm -p 127.0.0.1:8787:8787 \
  -e MEMORY_OBJECT_STORE_URI -e MEMORY_API_TOKEN \
  -e MEMORY_OBJECT_STORE_ROLE=writer \
  mempp-snapshots
```

Supply credentials through your platform's workload role or inject its AWS
credential configuration at runtime. The image defaults to reader mode. It needs
network access to object storage and, on a cold model cache, Hugging Face. For
remote clients, put the HTTP service behind your authenticated TLS ingress as
with the PostgreSQL deployment. No database container or durable compute volume
is required. Reader replicas use the same image with `ROLE=reader`.

## Publication, durability and freshness

Objects under a configured prefix are:

```text
latest.json                     # format, embedding profile, SHA-256 of snapshot
snapshots/<sha256>.sqlite        # full immutable corpus + SQL indexes + vectors
```

1. A session reads `latest.json` once, then downloads and verifies that snapshot.
   It uses a private, in-memory SQLite database for the entire session.
2. On commit, the writer uploads the complete new snapshot with a create-only
   conditional PUT. The snapshot includes original text, event times, attribution,
   all scope/provenance columns, tags, embeddings, supersession/merge relations,
  consolidation run history and HTTP idempotency receipts.
  Timestamps and query bounds are normalized to UTC, preserving instants even
  when incoming timestamps contain nonzero offsets (naive inputs mean UTC).
3. Only after that upload succeeds does the writer replace `latest.json`, requiring
   the ETag it originally read. Initial publication requires the pointer not to exist.
4. Success is acknowledged only after publication succeeds. An upload failure
   leaves the old version authoritative. A stale writer receives
   `PublicationConflict` (HTTP 409) and cannot overwrite the winner. Close a failed
   session and repeat the operation in a new session; never reuse its local state.

There is deliberately no distributed lock service or multiwriter merge protocol.
Operate one ingestion writer and serialize its work. Conditional publication also
protects against accidentally overlapping writers, but requires callers to retry
conflicts. The public `store_facts_verbatim` API retains its existing in-scope
exact-summary deduplication. A list passed to that API commits each fact separately;
it is not an atomic batch. The HTTP API publishes a complete request in one commit.

An acknowledgement lost after publication has an unknown outcome to the caller.
The durable request receipt (HTTP), source-message identity (store API), or existing
summary deduplication (verbatim API) makes retries of the same input idempotent.
Arbitrary manual mutation operations are not made exactly-once by this backend.

Readers are pinned to one version until their session closes. A newly opened
session sees the latest successfully published pointer under the object store's
consistency guarantees. A concurrent publish may leave an already-open reader on
the previous complete version, never a mixture. Every public API invocation opens
a fresh session; there is no background cache invalidation or freshness interval.
Failures to fetch a published object fail the operation rather than substituting
an empty/stale corpus. The whole database lives in memory and can be discarded at
any time. There are no irreplaceable local caches.

Unpublished snapshots can remain after failed publications. Published older
snapshots also remain for recovery; this release does not garbage-collect them.
Plan a retention policy that keeps the current snapshot and any versions required
by live readers. Do not apply independent short object expiration to `snapshots/`:
it could delete the still-current corpus. S3 bucket versioning is recommended for
recovering older pointer versions. Bucket durability, encryption and retention
settings determine durability beyond successful object-store acknowledgements.

For recovery, stop ingestion, obtain a known-good retained snapshot and its matching
manifest/profile, and restore the pointer with an ETag-conditional update. Never
restore using an unconditional overwrite while ingestion is running. Snapshot
checksums detect accidental corruption, not a malicious writer with bucket access.

## Retrieval differences and current limits

| Capability | PostgreSQL (default) | Object-store snapshots v1 |
|---|---|---|
| Original rows, attribution, scopes, audit relations | Existing schema | Same schema, preserved in each version |
| Lexical recall | PostgreSQL English full-text / cover-density rank | Existing case-insensitive substring candidates, graded query-term overlap; no stemming or PostgreSQL full-text operators |
| Semantic recall | pgvector ANN | Exact cosine scan over every eligible embedded row |
| Tags, weighted RRF, event-time filter, recent-evidence slots | Existing engine | Same pipeline and scope/time predicates |
| Semantic browse UI | pgvector order | Existing SQLite lexical fallback; semantic ranking is supported in **recall**, not browse |
| Hit statistics | Durable updates | Readers do not write hits/salience; writer sessions can persist them |
| Write concurrency | Database transactions | One ingestion writer; conflicting publication rejected |
| Snapshot size / embeddings | Database configuration | Maximum 256 MiB, 384-dimensional schema only |
| Query/commit cost | Indexed database work | Entire snapshot GET per session, local scans, entire snapshot PUT per commit |

The snapshot format records its embedding model/backend/dimension profile and
rejects a mismatched profile. Enabled embeddings must be available for writes and
vector queries; those failures raise instead of silently disabling the semantic
leg. Explicit `MEMORY_EMBEDDINGS_ENABLED=0` is still an intentional ablation; such
rows have no vectors and need backfilling/re-ingestion before semantic retrieval.

All data plus query work must fit comfortably in instance memory; allow substantial
headroom beyond snapshot size for deserialization, ORM objects and embeddings.
Snapshots persist SQL indexes and embedding values, but this v1 has no separate
full-text inverted index or ANN index. It prioritizes portability and correctness
for small corpora over throughput. Benchmark results published for PostgreSQL do
not transfer to this different lexical ranker, and reader-only hit behavior can
also change ranking across repeated queries. S3 Vectors is not used in v1.

## Permissions and checks

Readers need `s3:GetObject` for the corpus prefix. Writers also need `s3:PutObject`
for `latest.json` and `snapshots/*`; deletion is not needed by the backend. Give
initialization callers sufficient `s3:ListBucket` permission to distinguish a
missing key from access denied. KMS-encrypted buckets additionally require the
corresponding KMS key permissions. Denied access and missing buckets are errors,
not interpreted as empty corpora. Keep public access blocked and avoid granting
readers write permissions; the process role is an additional software guard.

Offline contract tests cover recovery, conflicts, failed uploads, lost responses,
reader isolation, audit preservation, semantic candidates and HTTP request receipts:

```bash
uv run --extra test --extra object-store pytest -q tests_engine/test_snapshots.py
```

The opt-in live test creates a random prefix inside a bucket you supply and removes
only that prefix's objects. Use a private test bucket without retention/versioning
for automatic complete cleanup; on versioned buckets, delete markers/old versions
need a separate retention policy.

```bash
MEMORY_TEST_S3_URI=s3://YOUR_PRIVATE_TEST_BUCKET/test-prefix \
  uv run --extra test --extra object-store pytest -q tests_engine/test_snapshots_s3.py
```
