# Mem++ for MemoryArena

This local evaluation facade implements MemoryArena's `initialize`, `add`, and
`wrap_user_prompt` HTTP contract. It supports the memory side of web-search and
other agent tasks. It does not perform web search, run an answering/judge model,
or establish an accepted MemoryArena backend or score.

## Start the memory service

From the Mem++ checkout, configure a **dedicated**, migrated PostgreSQL/pgvector
database and a `MEMORY_API_TOKEN` of at least 32 printable ASCII characters
without whitespace. The local ONNX MiniLM-384 profile is required, as for the
Textual Add/Search API. No generative model is called by this facade.

```sh
uv sync
uv run alembic upgrade head
# Set MEMORY_DATABASE_URL and MEMORY_API_TOKEN through your environment first.
export MEMORY_ARENA_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
export MEMORY_ARENA_TOP_K=10
uv run python -m memory.memoryarena_api --host 127.0.0.1 --port 8000
```

Create a fresh run UUID for every independent evaluation run. Reusing a run UUID
intentionally resumes its data; it does not clear it. Re-running a task inside
the same run UUID therefore sees that task's earlier evidence. Within a run, each upstream
`user_id` identifies an independent task world. Clients/sessions using the same
world share evidence. Repeated initialization preserves history. After a service
restart, initialize the user again before querying. This CLI uses a single process;
its initialized-user registry is in memory.

## Configure the upstream client

The upstream [MemoryArena client](https://github.com/ZexueHe/MemoryArena/blob/main/memory/client.py)
accepts a `requests.Session`. For a direct Python invocation:

```python
import os
import requests
from client import MemoryClient  # MemoryArena's memory/client.py

session = requests.Session()
session.headers["Authorization"] = "Bearer " + os.environ["MEMORY_API_TOKEN"]
client = MemoryClient(user_id="task-1", memory_system_name="mempp",
                      base_url="http://127.0.0.1:8000", session=session)
client.add("action: searched project notes\nobs: Atlas launch approved")
print(client.wrap_user_prompt("What happened to Atlas?"))
```

For upstream scripts that construct their own MemoryClient (including search-agent
subprocesses), the supplied patch adds the token header in the common `_post`
method. From the **MemoryArena** checkout:

```sh
git apply --check /absolute/path/to/mem-plus-plus/integrations/memoryarena/client-auth.patch
git apply /absolute/path/to/mem-plus-plus/integrations/memoryarena/client-auth.patch
```

Export the same `MEMORY_API_TOKEN` in the environment-server/search-agent
processes, then set the upstream search configuration's memory section to:

```json
{"memory_system_name": "mempp", "memory_url": "http://127.0.0.1:8000", "no_memory": false, "use_step_memory": true}
```

The facade replaces the upstream memory server for this configuration; no change
to the upstream server factory is needed. Check the patch against the current
upstream revision before applying. Keep the service on loopback for local runs;
remote use requires TLS and private database access. Do not send the token to an
untrusted host. The upstream agent and evaluator have separate model requirements
and may use paid APIs; running the memory facade does not authorize those costs.

## Data and retry behavior

Chunks are preserved as source text with the transport prefix `user:`. This
prefix describes the upstream ingestion channel, not the original contributor;
authors/roles already present in the chunk are retained. The upstream contract
provides no event timestamp, so none is invented. Each retrieved record has a
stable evidence ID, and its text is escaped inside `<memory>` tags in the prompt.
The original query follows the `<memory_context>` block.

The upstream API has no request ID. Exact duplicate chunks within a task world
are therefore intentionally deduplicated using a content hash, including retries
after a lost response. The facade retries the atomic canonical-tag race up to three attempts because
the upstream client has no built-in retry. If contention persists, a 409 is
returned; retry the identical chunk. Other failures remain visible. All records and the retry receipt
commit atomically. Source data and receipts require the same retention policy.
Use a dedicated database, separate from demo resets and consolidation experiments.

## Validate without paid models

```sh
uv sync --extra test
uv run pytest tests_engine/test_memoryarena_api.py tests_engine/test_aml_api.py -q
```

PostgreSQL cases require `MEMORY_DATABASE_URL`; otherwise they skip. Unit tests
use deterministic vectors. Real HTTP/client/ONNX smoke evidence is recorded
separately; neither kind of test is a MemoryArena answer-accuracy result.
