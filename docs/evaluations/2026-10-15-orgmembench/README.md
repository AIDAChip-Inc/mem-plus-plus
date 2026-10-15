# Mem++ official-harness OrgMemBench run — 2026-10-15

Full small and medium runs used actual API-generated answers and the unchanged
OrgMemBench neutral answerer and judge. These are self-run results published in the Mem++ repository only. They have
not been submitted to OrgMemBench and are not an accepted ranking or a
reproduction of the historical Mem++ paper's evaluation setup.

| Tier | Artifacts | Questions | Mean rubric | Semantic mean | Judge exact-match | Judge faithfulness | Answerer + judge cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 126 | 11 | **0.6496** | 0.4745 | 0.0000 | 0.0000 | $0.8269 |
| Medium | 443 | 73 | **0.5567** | 0.4112 | 0.1096 | 0.1644 | $5.5031 |

The rubric mean is average weighted partial credit on a 0–1 scale, not the
fraction of questions fully correct. These are single runs with no confidence
intervals. Full category scores and all other official metrics are retained in
the unmodified result JSONs. Exact-match and faithfulness are judge-assessed
fields; the harness defines hallucination rate as one minus faithfulness.

## Protocol

- Upstream base: `f17fc92128c560de5c5db0bf982c33845dd98c58`; adapter/harness run commit: `5841da38f1f8a3838de13fefc0c58519582d4861`.
- Mem++ engine: `9a04af72d949a486e71aa7094150ae1a537ab0fe`, Python 3.13, PostgreSQL/pgvector, local ONNX all-MiniLM-L6-v2 embeddings (384 dimensions).
- Retrieval: k=50, candidate limit 50, fuzzy/tag/vector RRF weights 1/1/4, lexical OR off. Event-date filtering uses the upstream runner's supplied `as_of` value.
- Answerer and judge: `claude-sonnet-4-6`, temperature 1, explicit thinking budget 4,000, maximum output 8,000, Anthropic SDK 0.84.0. Official answerer context cap 40,000 characters; official judge answer-input cap 6,000 characters.
- Authors and threads share within one organizational world. A fresh run UUID and separate company/tier scopes keep unrelated runs and tiers apart; pre-existing history was left untouched.
- Full loaded datasets: 126/443 artifacts and 11/73 questions. Parallelism 3, no question limit, no checkpoint reuse, no score-based rerun or answer selection.
- The local execution wrapper called `runner.run_system` with a budgeted client (`max_retries=0`, timeout 180 seconds) and a dedicated output directory. Model settings, prompts, rubric and scoring code were unchanged.

The official latency fields measure adapter retrieval, excluding generation and
judging. `cost_per_query_usd` includes the answerer but excludes the judge; the
table above includes both. Zero storage-footprint fields are uninstrumented
defaults, not measured zero storage. Total token-priced API usage for the two
official runs was **$6.330030**; no paid ingestion or embedding calls were used.

## Artifacts and verification

- [Small: all answers, evidence, judgments and metrics](helix-small.json)
- [Medium: all answers, evidence, judgments and metrics](helix-medium.json)
- [Execution audit and provenance hashes](audit.json)

The audit verifies full question coverage, 168 actual API prompt hashes,
correspondence between API text and answers, replay of every judgment, and
metric recomputation. It checks 569 stored source texts/identities against the
normalized corpus, reconstructs all 84 answerer contexts, and checks 4,183 source
appearances (783 date bounded): zero foreign-world, source-ID or date-bound
violations. This validates execution and provenance, not independent answer
correctness. Normalized corpus hashes, result hashes, source-code hashes and
embedding revision/hashes are preserved in the audit.

The previous 17 adapter tests passed with SQLite/PostgreSQL. Three upstream
fixture failures were reproduced on pristine upstream (3 failures / 89 passes)
and the adapter branch (3 failures / 106 passes); those are disclosed in the
adapter submission. No new benchmark inference was needed to publish this run.

## Reproduction

Use the pinned adapter commit in the [pinned adapter source](https://github.com/abdoaom/OrgMemBench/tree/5841da38f1f8a3838de13fefc0c58519582d4861)
and its `docker/mempp.md` instructions. With the documented private PostgreSQL,
ONNX profile, a fresh `MEMPP_ORGMEMBENCH_RUN_ID`, and an externally configured
`ANTHROPIC_API_KEY`, pin `anthropic==0.84.0` and execute the full small and medium
runs. SDK 0.84.0 accepts the official harness's temperature argument.

```sh
orgmembench run --system mempp --tier small --execute
orgmembench run --system mempp --tier medium --execute
```

Those commands reproduce the core protocol, not the local billing ledger or
response-recording wrapper. The audit's private DB/API-response checks were
rerun locally; the published audit JSON is an evidence report, not a portable
maintainer-side audit command. Preserve fresh output paths to avoid stale
checkpoint reuse.
