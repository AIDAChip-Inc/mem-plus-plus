# MemoryBench LoCoMo-family local evaluation — 2026-10-15

Mem++ produced answers for all 50 official test items across `Locomo-0` through `Locomo-9` in MemoryBench. Mean local F1 using the pinned native scorer was **0.3950822** (0–1 scale). This is a local evaluation of that family only: it is not the entire MemoryBench suite, an organizer-issued website score, a public leaderboard placement, or a reproduction of the paper's original LoCoMo evaluation. No comparative rank is claimed.

## Protocol

Each dataset was a separate native off-policy experiment with its own memory scope, 20 newly generated training dialogs and five official test questions. Dialog retention intentionally preserves observed retrieval context from training; a separate corpus fork then adds that dataset's corpus. Sharing within each experiment is retained. All ten datasets used fixed settings; no test answers were replaced or selectively excluded.

- Answer model: `claude-sonnet-4-6`, temperature 1, at most three training rounds, 2,048 training / 4,096 test output tokens.
- Memory: local ONNX `all-MiniLM-L6-v2`, 384 dimensions, PostgreSQL/pgvector, retrieval k=50.
- MemoryBench harness commit: `247ba8c1e327fa297c0d57aaad0e3cbd66bac656`.
- Dataset revision: `3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d`.
- Mem++ engine commit: `d952129834f1749c412a04528a4f9138d94a7d74`.
- Native integration commit: `45328156584c4519811ceaa0422fbb4375f6d957`; see [integration instructions](../../../integrations/memorybench/README.md).

## Results

| Dataset | Test items | Mean local native F1 (0–1) |
|---|---:|---:|
| Locomo-0 | 5 | 0.469565 |
| Locomo-1 | 5 | 0.505495 |
| Locomo-2 | 5 | 0.627666 |
| Locomo-3 | 5 | 0.035484 |
| Locomo-4 | 5 | 0.240404 |
| Locomo-5 | 5 | 0.116254 |
| Locomo-6 | 5 | 0.800000 |
| Locomo-7 | 5 | 0.252632 |
| Locomo-8 | 5 | 0.303323 |
| Locomo-9 | 5 | 0.600000 |
| **Combined** | **50** | **0.395082** |

These are single runs without confidence intervals. Local token-overlap F1 is not a percentage of fully correct answers. It is distinct from any subsequent organizer scoring. The small per-dataset samples do not establish comparative superiority.

## Evidence and limits

[Machine-readable results](results.json) preserve per-item F1, exact counts, fixed settings, prediction hashes and the hashes of ten saved audit reports. The controlled local drivers retained API requests/responses, original retrieval contexts and source records. Read-only audits checked the saved predictions against actual API outputs, reconstructed source records, checked 12,500 retrieved source appearances and recomputed 13,948 stored-vector appearances. No foreign-scope records were found in the audited retrieval traces. These checks establish provenance and scope for this run; they do not establish answer correctness independently of the scorer.

The run used 596 actual answer-generation API calls costing $20.552814. Audits made no paid API calls. Raw request/source traces and the controlled run drivers are retained locally, rather than included in this public summary; this report alone is not a complete independently rerunnable experiment bundle. No credentials or contact address are included.

The exact prepared website ZIP has SHA256 `bf9077824282a8abcf3a5506c87b9207e3e3c3da501dd1e4df8077870f48ce2c`; combined predictions SHA256 is `317109c53eee7ad294827361c84578a01e02f14bdcb0cf6571b390f03d282747`. Website evaluation is separate and does not automatically establish a public ranking.
