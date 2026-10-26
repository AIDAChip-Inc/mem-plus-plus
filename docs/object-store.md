# Object-store-backed Mem++

This optional backend addresses [issue #21](https://github.com/AIDAChip-Inc/mem-plus-plus/issues/21).
It keeps organizational memory outside compute instances, allowing disposable
containers, concurrent ingestion writers and read-only replicas without operating
PostgreSQL. PostgreSQL + pgvector remains the default and is recommended when
large-corpus search latency or transactional throughput matters most.

## What this adds

- S3 or S3-compatible object storage and a POSIX filesystem adapter.
- Immutable, content-addressed SQLite chunks and versioned manifests, published
  through an ETag-conditional `latest.json` pointer.
- Concurrent writers: independent record changes automatically rebase onto the
  newest published version; incompatible edits to the same row are rejected.
- Incremental transfers: commits upload changed chunks; readers reuse verified
  immutable chunks in disposable memory/disk caches.
- No fixed application-level corpus size cap. An optional operator budget can
  bound memory use; the current search engine still needs the corpus in memory.
- The existing Python recall/write, browse, supersede, merge and consolidation
  APIs, plus the authenticated `/add` and `/search` HTTP service.

Records, source/event timestamps, contributor tags, project scopes, embeddings,
supersession/merge relations, consolidation runs and HTTP request receipts share
one consistent published version. No generative LLM is called for verbatim ingest.

This removes a database-service dependency for smaller organizational assistants,
ephemeral tasks and portable deployments. Chunk reuse reduces repeated network
traffic and retained snapshot storage. It does not guarantee lower cost or faster
search than PostgreSQL, and cold readers still need to load a complete corpus.

## Configuration

```bash
uv sync --extra object-store
export MEMORY_STORAGE_BACKEND=object_store
export MEMORY_OBJECT_STORE_URI=s3://YOUR_PRIVATE_BUCKET/mempp/organization-a
export MEMORY_OBJECT_STORE_ROLE=writer
export MEMORY_USER=alice
```

Use a private bucket and your normal AWS profile, workload role or credential
configuration. The extra includes AWS CRT for AWS login profiles. Credentials do
not belong in the URI or source files. For a compatible service, additionally set
`MEMORY_S3_ENDPOINT_URL` and configure its region/credentials through the AWS SDK.

The object store must provide atomic replacement, read-after-write consistency,
and conditional `PutObject` with both `If-Match` and `If-None-Match`. There is no
unsafe unconditional-write fallback. See
[S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

For local development or a suitable shared volume, use
`file:///absolute/path/to/corpus`. This directory is durable state, not a cache.
The filesystem adapter requires POSIX `flock`, atomic rename and fsync. Generic
S3/FUSE mounts are not assumed to provide those guarantees; use the S3 adapter
for those object stores. Native Windows does not support the POSIX adapter.

Unset `MEMORY_STORAGE_BACKEND`, or set it to `postgres`, for the existing
`MEMORY_DATABASE_URL` backend. Switching backends selects a separate corpus; no
PostgreSQL corpus migration is performed automatically. Existing format-1 object
snapshots remain readable and upgrade to chunked format 2 on the next writer commit.

Additional controls:

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_OBJECT_STORE_ROLE` | `reader` | `reader` or `writer`; readers cannot ingest |
| `MEMORY_OBJECT_STORE_CACHE_BYTES` | `67108864` | Shared process memory cache budget; 0 disables it |
| `MEMORY_OBJECT_STORE_CACHE_DIR` | unset | Optional disposable disk cache, namespaced by store location |
| `MEMORY_OBJECT_STORE_MAX_BYTES` | `0` | Maximum serialized corpus bytes; 0 means no application cap |
| `MEMORY_OBJECT_STORE_WRITE_RETRIES` | `8` | Maximum optimistic publication attempts |
| `MEMORY_OBJECT_STORE_MAX_OBJECT_BYTES` | `1073741824` | S3 GET allocation guard for one object; independent of corpus size |

New snapshots use 64 KiB chunks, so the individual-object guard does not limit
corpus size. Raising the corpus budget cannot create more instance memory: size
instances for the full corpus, SQLite, query results and serialization/rebase work.

## Python example

Ingest with writer configuration:

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

Query on another instance with the same URI and identity:

```bash
export MEMORY_OBJECT_STORE_ROLE=reader
# Optional, particularly for corpora larger than the memory cache:
export MEMORY_OBJECT_STORE_CACHE_DIR=/tmp/mempp-cache
uv run --extra object-store python - <<'PY'
from memory import recall_facts
print(recall_facts("assistant", "What is the Atlas launch plan?"))
PY
```

Readers fail clearly if there is no publication; initialize a writer first. A
writer can initialize an empty corpus with `get_session().commit()` inside a
session context. Direct ORM users can use `session.commit()` or
`with session.begin():`; releasing a savepoint does not publish before its outer
transaction commits. Use `get_session()` rather than a process-wide `get_engine()`.
Benchmark adapters that explicitly require PostgreSQL still require PostgreSQL.

Use the existing `project_scope(...)` context for explicit project/customer
scopes. Scopes are not authentication. Separate trust domains need separate
prefixes/credentials. The local chunk cache contains corpus data, so restrict
its directory permissions and use encrypted ephemeral storage when appropriate.
The embedding-model cache is also disposable; cold instances may download the
model from Hugging Face.

## HTTP/container deployment

```bash
export MEMORY_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MEMORY_OBJECT_STORE_ROLE=writer
uv run --extra object-store python -m memory.aml_api --host 127.0.0.1 --port 8787
```

Writer startup initializes an empty corpus when necessary. Multiple writer
instances may share a prefix. Reader replicas use `ROLE=reader`; `/search` and
`/health` work, while `/add` returns 403. Successful `/add` responses cover all
records in the request and its durable idempotency receipt. Retry a 409 or an
unknown network outcome with the same request ID and payload. Reusing an ID with
a changed payload is rejected.

```bash
docker build -f Dockerfile.snapshots -t mempp-snapshots .
docker run --rm -p 127.0.0.1:8787:8787 \
  -e MEMORY_OBJECT_STORE_URI -e MEMORY_API_TOKEN \
  -e MEMORY_OBJECT_STORE_ROLE=writer \
  mempp-snapshots
```

Supply AWS credentials through a workload role or runtime credential
configuration. The image defaults to reader mode. Optional local caches can be
removed on every restart. Remote clients should use your authenticated TLS ingress
as with the PostgreSQL deployment. No durable compute volume is required.

## Publication, concurrency and recovery

```text
latest.json                       # current format/profile/manifest hash
manifests/<sha256>.json            # immutable ordered chunk hashes + size/profile
chunks/<sha256>                    # immutable SQLite byte chunks
snapshots/<sha256>.sqlite          # legacy format-1 objects, if present
```

Each session reads `latest.json` from the store, verifies the immutable manifest
and chunks, and opens an independent in-memory SQLite database. Up to 16 chunk
fetches run in parallel on a cold load. Unchanged warm sessions fetch only the
pointer; immutable data comes from the process/disk cache. A corrupt cached object
is discarded/refetched, and the entire cache can be lost without losing memory.
A cold restart without a cache still downloads the full corpus.

On commit the writer splits its serialized database into chunks, uploads only
hashes absent from its known published chunks, then uploads an immutable manifest.
Only afterward does it conditionally replace the pointer using the ETag originally
read. Success is acknowledged after pointer publication succeeds. Interrupted
uploads cannot expose a partial version. Timestamps and query bounds are normalized
to UTC so offset-bearing inputs retain their event instants.

If another writer published first, the losing writer reloads the latest version
and performs a three-way merge against its original base. Independent inserts and
edits to different rows are preserved. Tags use deterministic scoped IDs, so
concurrent inserts of the same tag share one registry entry. Source-ID writes and
identical HTTP receipts recognize retries instead of duplicating records.
Conflicting edits/deletes to the same row, incompatible receipt payloads, or merges
that would break unique/foreign-key constraints raise `PublicationConflict` (HTTP
409). No last-write-wins replacement is used. After the configured retry budget is
exhausted, close the failed session and repeat the operation in a fresh session.

This is optimistic row-level conflict detection, not serializable predicate
locking. Cross-record business invariants still need application coordination.
High contention can return retryable conflicts. The public verbatim API retains
its in-scope exact-summary retry deduplication, but concurrent callers without a
shared source/request identity can create equivalent independent records. For
cross-writer exactly-once ingestion, use the HTTP request ID or a store source ID.
Verbatim lists commit per fact; HTTP request batches commit atomically. Arbitrary
manual mutations are not made exactly-once by this backend.

A reader stays pinned to one complete version until its session closes. Newly
opened sessions observe the latest pointer under the store's consistency guarantees;
there is no stale-pointer cache or background invalidation interval. A valid cached
immutable object need not be fetched again. If a required uncached durable object
is missing, corrupt or inaccessible, the operation fails instead of falling back
to an empty/stale corpus.

All manifests/chunks are retained, including possible orphan objects after failed
publications. There is no automatic garbage collection. A retention job must trace
all versions it intends to keep, including versions used by live readers, before
removing unreferenced chunks. Age-based chunk expiration alone is unsafe: a current
version may still reference very old chunks. S3 bucket versioning is useful for
pointer recovery. To restore, stop ingestion and conditionally republish the pointer
to a known-good retained manifest/profile. Checksums detect accidental corruption,
not malicious writers. Bucket encryption, retention and durability settings still apply.

## Search behavior and remaining resource costs

| Capability | PostgreSQL default | Object-store backend |
|---|---|---|
| Records, scopes, attribution, provenance | Existing schema | Same schema across published versions |
| Lexical recall | English full-text / cover-density rank | Graded case-insensitive substring/term overlap, without PostgreSQL stemming/operators |
| Semantic recall | pgvector ANN | Exact cosine scan over eligible embedded records |
| Tags, weighted RRF, time filters, recent slots | Existing engine | Same scope/time predicates and pipeline |
| Semantic browse | pgvector ordering | Existing SQLite lexical fallback; semantic **recall** is supported |
| Hit statistics | Durable updates | Readers do not write statistics; writers can persist them |
| Write concurrency | Database transactions | Automatic merge of independent row changes; incompatible changes conflict |
| Network transfers | Database protocol | Changed chunks on writes, missing chunks on reads |
| Corpus limit | Database resources | No fixed application cap; instance memory and operator budget apply |

The current schema requires 384-dimensional embeddings. Snapshot profiles record
model/backend/dimensions and reject mismatches. Enabled embeddings must be available;
write/query failures do not silently drop the semantic leg. Explicit
`MEMORY_EMBEDDINGS_ENABLED=0` remains an intentional ablation whose records need
backfilling/re-ingestion before semantic retrieval.

Chunking removes whole-corpus network transfers for warm readers/small writes,
but sessions still deserialize the full corpus and serialize it on commit. A
concurrent rebase scans base/changed/latest rows and needs additional memory.
Local lexical/vector work still scans eligible records. There is no S3 Vectors or
separate ANN service in this implementation, and PostgreSQL benchmark numbers do
not describe this lexical ranker or reader-only hit behavior.

## Permissions and verification

Readers need `s3:GetObject` on the prefix. Writers also need `s3:PutObject` on
`latest.json`, `manifests/*` and `chunks/*`. Backend operation does not need deletion.
Initialization callers need enough `s3:ListBucket` permission to distinguish missing
keys from access denied. KMS-encrypted buckets need corresponding key permissions.
Keep public access blocked and restrict reader credentials to read access.

```bash
uv run --extra test --extra object-store pytest -q \
  tests_engine/test_snapshots.py tests_engine/test_snapshot_scaling.py

# Actual >256 MiB round trip; uses roughly 1 GiB of transient memory:
MEMORY_TEST_LARGE_SNAPSHOT=1 uv run --extra test --extra object-store \
  pytest -q tests_engine/test_snapshots_large.py

# Creates a random prefix and deletes only that prefix's objects:
MEMORY_TEST_S3_URI=s3://YOUR_PRIVATE_TEST_BUCKET/test-prefix \
  uv run --extra test --extra object-store pytest -q tests_engine/test_snapshots_s3.py
```

Use a non-versioned test bucket without retention for complete automatic cleanup.
Versioned buckets additionally retain old versions/delete markers according to
bucket policy. The live tests cover real ONNX inference as well as storage contracts.
