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
dataset. Empty strings are allowed as explicit abstentions. Booleans are not IDs.
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
published** to the public leaderboard. No scored Mem++ submission has been made
through this utility. The website displayed no deadline when inspected.
