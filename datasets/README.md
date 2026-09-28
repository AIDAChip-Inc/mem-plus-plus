---
title: memory-research evaluation datasets
owner: qiyas
status: active
updated: 2026-07-23
---

# Evaluation datasets

`fetch.py` downloads the four memory / long-context benchmarks into **local,
gitignored** folders under `datasets/`. **No dataset file is ever committed** —
`datasets/.gitignore` ignores all data and keeps only `fetch.py`, this README,
and the ignore file itself. The populated folder is shared with the student
out-of-band (the fetch must actually work; do not hand over an empty folder).

Run from the `memory-research/` project root:

```bash
python datasets/fetch.py all              # locomo + longmemeval(_s,_oracle) + longbench-E
python datasets/fetch.py locomo
python datasets/fetch.py longmemeval       # _s + _oracle
python datasets/fetch.py longmemeval --include-m   # opt-in, multi-GB
python datasets/fetch.py longbench         # LongBench-E subset (default)
python datasets/fetch.py longbench --full  # all 21 v1 configs
python datasets/fetch.py membench          # both agents (participation + observation)
python datasets/fetch.py membench --third-only   # lighter observation half only
```

## What each target pulls

| Target | Source | Files landed | Approx. size | License |
|--------|--------|--------------|--------------|---------|
| `locomo` | [snap-research/locomo](https://github.com/snap-research/locomo) raw `data/locomo10.json` | `locomo/locomo10.json`, `locomo/LICENSE.txt` | ~15–25 MB | See fetched `LICENSE.txt` — **check before any redistribution** |
| `longmemeval` (default) | HF dataset [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval/longmemeval_s_cleaned.json`, `longmemeval/longmemeval_oracle.json` | `_s` ~200–400 MB (~115K tokens/instance, ~40 sessions), `_oracle` ~10–20 MB | MIT |
| `longmemeval --include-m` | same HF dataset | `+ longmemeval/longmemeval_m_cleaned.json` | **multi-GB** (~500 sessions/instance) — opt-in only | MIT |
| `longbench` (default) | HF dataset [`THUDM/LongBench`](https://huggingface.co/datasets/THUDM/LongBench) | `longbench/<config>.jsonl` (LongBench-E subset, 13 configs) | ~100–300 MB | MIT |
| `longbench --full` | same HF dataset | `longbench/<config>.jsonl` (all 21 v1 configs) | ~500 MB–1 GB | MIT |
| `membench` (default) | [import-myself/Membench](https://github.com/import-myself/Membench) raw `MemData/` (arXiv [2506.21605](https://arxiv.org/abs/2506.21605)) | `membench/{FirstAgent,ThirdAgent}/<qatype>.json` | ~600 MB (FirstAgent ~500 MB, ThirdAgent ~115 MB) | MIT (declared in README badge; **no standalone LICENSE file** — fetch-only, do not redistribute) |
| `membench --third-only` | same repo | `membench/ThirdAgent/<qatype>.json` only | ~115 MB | same |

Sizes are order-of-magnitude — the actual bytes depend on the HF revision.

## Dependencies

| Target | Needs |
|--------|-------|
| `locomo` | stdlib only (`urllib`) |
| `longmemeval` | `huggingface_hub` |
| `longbench` | `datasets` |
| `membench` | stdlib only (`urllib`) |

These are declared in the project `pyproject.toml` (owned outside `eval/` and
`datasets/`). If a HF dependency is missing, that sub-fetch prints an
actionable install hint and is skipped — it does not crash the whole run.

## Benchmark shapes (what the adapters in `eval/adapters/` consume)

- **LoCoMo** — `session_N` turn lists per conversation + a `qa` list. QA
  categories: single-hop, multi-hop, temporal, open-domain, adversarial
  (adversarial = unanswerable; correct behavior is to abstain). Evidence
  `dia_id`s are the gold retrieval targets.
- **LongMemEval** — per instance: `question_id` (`_abs` suffix = abstention),
  `question_type`, `question`, `answer`, `haystack_sessions` (+ ids/dates),
  `answer_session_ids` (session-level gold). Abilities: information-extraction,
  multi-session, temporal-reasoning, knowledge-update, abstention.
- **LongBench v1** — per config: `input`, `context`, `answers`,
  `all_classes`, `length`. Task categories: single-doc QA, multi-doc QA,
  summarization, few-shot, synthetic, code. Metric is per-task (F1 / ROUGE-L /
  accuracy / edit-sim / EM) — see the LongBench paper.
- **MemBench** — `{scenario: [trajectory]}` per `<qatype>.json`; each trajectory
  has a `tid`, a `message_list`, and one multiple-choice `QA` (`question`,
  `choices` A–D, `ground_truth` letter, `answer` text, `target_step_id` gold
  evidence). Two agents: **FirstAgent** = participation (`{mid, user, assistant}`
  dialogue sessions), **ThirdAgent** = observation (`{mid, message}` statement
  streams). `mid` is global across sub-lists and is the turn provenance id;
  `target_step_id` is a `mid` (flat) or a `[mid, outer]` pair. The category is
  `<agent>:<qatype>`. Native metric is exact-letter accuracy — this harness scores
  it under its own frozen token-F1 + LLM-judge protocol (choices rendered into the
  question), so numbers are comparable to the other three benchmarks here, not to
  the MemBench leaderboard.

## Reproducibility note

Benchmark leaderboard numbers do **not** transfer across judges. Every score
this harness produces is measured under our own frozen protocol (pinned judge
model + published judge prompt — see `eval/judge_prompt.txt` and
`eval/judge.py`). Record the dataset revision alongside any reported number.
