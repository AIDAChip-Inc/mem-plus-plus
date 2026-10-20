# MemoryBench website package preparation

The public [leaderboard](https://memorybench.thuir.cn/leaderboard) has a
[ZIP upload form](https://memorybench.thuir.cn/submit), inspected 2026-10-15.
This utility validates and packages existing model answers. It does not implement
a MemoryBench agent, generate answers, run evaluation, submit anything, or imply
that Mem++ appears on that leaderboard.

The form accepts a subset of official datasets, but every included dataset must
contain exactly its official test indices. These are not necessarily consecutive:
Locomo-0 at revision `3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d` has indices
129, 189, 163, 143, 188. Do not substitute another LoCoMo split or retrieval traces.

## Prepare a pinned index manifest

Use the repository Python environment. Fetching needs optional `huggingface-hub`
and `pyarrow`; packaging itself uses only the standard library. Install those in
a disposable environment if needed. No token is sent for the public dataset.

```sh
python integrations/memorybench/package_submission.py fetch-manifest \
  --revision 3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d \
  --dataset Locomo-0 --output memorybench-indices.json
```

The explicit commit pins all test shards referenced by the official test state.
The manifest retains test IDs and source file hashes, not prompts or gold answers.
Repeat `--dataset` for additional datasets. Recheck the official data and form
before a real run; the example pin is reproducible, not a promise of future venue
eligibility. A locally edited manifest cannot establish official coverage.

## Package real answers

Provide a UTF-8 JSON array containing `dataset` (exact official name), integer
`test_idx`, and string `response` for every required test ID of each included
dataset. This local helper allows empty strings as explicit abstentions, but
the organizer website has since rejected empty responses. Preserve raw refusals
and agree on or disclose any submission rendering; do not silently invent an
answer. See the [V5 procedure](../../docs/evaluations/2026-10-19-memorybench-v5/PROCEDURE.md). Booleans are not IDs.
Duplicate IDs/keys, missing or extra IDs, unknown datasets, and nonstandard JSON
constants fail validation. Extra row fields are omitted from the ZIP, preventing
accidental inclusion of prompts, labels, or local run metadata.

```sh
python integrations/memorybench/package_submission.py package \
  --manifest memorybench-indices.json --predictions actual-model-responses.json \
  --output mempp-memorybench.zip
```

Outputs never overwrite existing files. The ZIP contains only
`dataset/<name>/predict.json`, ordered deterministically, with a conservative
500,000,000-byte compressed size limit. Keep the printed ZIP SHA-256, manifest,
actual run configuration and complete response provenance together outside the
upload ZIP. Structural validation cannot verify how answers were generated.

## Before website submission

A compatible Mem++ agent and real answer-generation run are still required.
Preserve sharing within a task's organization/project and separate independent
benchmark worlds explicitly. Use the official protocol for training, feedback,
and test access; a test-index manifest does not authorize reading labels during
answer generation. Do not replace missing answers with synthetic test fixtures.

The form also requires the actual base model, system name, contact email and
acceptance of its guidelines; description and submission notes are optional.
Choose a free local model or obtain an explicit inference budget before a paid
run. Review the completed answers and claims with Claude, Kimi and independent
Codex before upload. Evaluation results are emailed and **not automatically
published** to the public leaderboard. A separate controlled answer-only run
now has [organizer-email results and a submission history](../../docs/evaluations/2026-10-19-memorybench-v5/README.md); the helper itself does not generate or evaluate answers.
The website displayed no deadline when inspected.

## Native MemoryBench off-policy integration

`install_native.py` installs an optional `mempp` agent/solver into upstream
MemoryBench commit `247ba8c1e327fa297c0d57aaad0e3cbd66bac656`. It is a local
integration for the website workflow, not an upstream inclusion request or a
completed score submission. On-policy execution is deliberately disabled in the
registry until its complete training lifecycle is validated.

Use Python 3.13+ and install the pinned engine separately from the plugin:

```sh
uv pip install 'memory-research @ git+https://github.com/AIDAChip-Inc/mem-plus-plus.git@d952129834f1749c412a04528a4f9138d94a7d74'
uv pip install 'anthropic==0.84.0' openai tqdm
python integrations/memorybench/install_native.py /path/to/MemoryBench
```

The installer requires that exact upstream commit and refuses existing plugin
files. It copies the agent/store/solver, registers `mempp`, and adds its config.
It makes two small compatibility changes to the native Anthropic provider:
omit `top_p` when unset, and strip internal message annotations such as
`origin_content` before API transport. Prompts and the native feedback protocol
are retained. Apply the upstream dependencies needed for your chosen runner;
the minimal import/contract-test environment is not the full evaluation stack.

Provide a migrated private PostgreSQL/pgvector database with vector dimension
384, then configure before starting Python:

```sh
export MEMORY_DATABASE_URL='postgresql+psycopg2://...'
export MEMORY_EMBEDDING_BACKEND=onnx
export MEMORY_EMBEDDINGS_ENABLED=1
export MEMORY_RRF_W_FUZZY=1
export MEMORY_RRF_W_TAG=1
export MEMORY_RRF_W_VECTOR=2
export MEMORY_CANDIDATE_LIMIT=200
export MEMORY_LEXICAL_OR=1
export MEMPP_MEMORYBENCH_RUN_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
```

These are an explicit local MiniLM-384 retrieval profile, not a reproduction of
paper runs using OpenAI embeddings. The adapter checks the installed engine
commit, database dialect/dimension, embedding backend and retrieval profile.
No credential, database URL or model configuration is written into memory
snapshots. Supply model credentials externally; do not copy an upstream `.env`
into the controlled execution environment.

### Sharing and cache behavior

The native runner intentionally pools training dialogs within an experiment,
including cross-dataset training transfer in grouped experiments. Mem++ keeps
that sharing. Source identity includes dataset plus dialog index in pooled
training, avoiding collisions when different datasets reuse indices. Exact
dialog retries deduplicate; changed observations append instead of replacing
prior source history. As in native BM25 and Embedder, generated dialogs retain
the augmented first-round content, including its retrieved context. Grouped
experiments therefore intentionally transfer that observed training context
across datasets; separate corpus forks do not remove inherited training data.
Supplied roles, speakers, turn IDs and dates remain text;
the adapter does not invent event dates from unparsed source strings.

The native off-policy corpus runner copies its completed training cache before
adding a corpus. A copied Mem++ snapshot forks into a separate database world
based on the same run UUID and new cache path. Both corpus branches inherit
training memory, while later writes do not cross branches. Paths and run UUIDs
must remain stable when resuming the same world. A new run UUID is required for
an independent experiment; an old snapshot from another run is rejected.

Snapshots use atomic file replacement and a content checksum. Loading the same
world checks the actual database records and embeddings against the snapshot.
Missing data, extra history, a wrong database, checksum drift, or incompatible
run/profile metadata fail closed. Interrupted snapshot forks can retry matching
partial ingestion without duplicating records. Snapshots contain source text;
keep them with the experiment's artifacts, not in credential files. A checksum
is an integrity check, not a signature proving who generated the data.

### Protocol and remaining execution steps

The response prompt follows the native BM25-style en/zh context prompt. The
augmented first-round history remains available for native feedback follow-up
rounds. Failed requests leave the original history intact for native retries.
The solver propagates infrastructure/model errors instead of silently
turning them into `Error: Unable to generate response.` prediction strings.
A context overflow also stops the run; the adapter does not silently reduce
`retrieve_k`. Pass `--retrieve_k 50` to the native off-policy CLI to override its
default of 5, and record the effective value. Native dialog generation caps
output at 2048 tokens by default; the installed test config allows 4096.

Corpus off-policy runs require freshly generated `dialog_mempp` training fields.
The official dataset does not currently include them. Generate those through
the pinned native reading/feedback protocol and preserve them locally; do not
rename another baseline's generated dialogs. Allowed training feedback is
adaptation data, while test labels must not be ingested. A complete controlled
answer-generation run is still needed before the website ZIP can be uploaded.

The website requires all exact official test indices per included dataset,
a base-model name and a contact email. Its evaluation is emailed and not
automatically listed on the public leaderboard. No deadline was displayed on
the submission form when rechecked on 2026-10-15.

Before any paid execution, wire all native model/feedback calls into the shared
authorized benchmark-budget ledger, with distinct benchmark/run/purpose labels.
Do not start a new budget pool or let retry/fallback logic swallow budget errors.
The integration itself does not enforce API spending limits.

### Free verification

After installing the plugin into the pinned checkout:

```sh
MEMORYBENCH_CHECKOUT=/path/to/MemoryBench python -m pytest \
  integrations/memorybench/test_native.py -q
```

The native contract tests import real upstream classes and use synthetic
storage/LLM fixtures. A separate real local ONNX/PostgreSQL smoke test verified
training sharing, separate corpus forks, stable IDs and snapshot reload across
three synthetic worlds (eight records), with zero paid model calls. This is
integration validation, not MemoryBench answer accuracy or submitted predictions.
