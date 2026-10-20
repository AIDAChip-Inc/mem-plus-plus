# MemoryBench V5: procedure and submission history

This records the 819-item, 28-dataset run submitted as
`mem++_full_v5_refusal_format`, submission
`693dfc58d3eb4fe49f1311b1bf9a6fe6`. The organizer reported evaluation complete
and **Overall average 0.6261**; see [results.json](results.json) for the seven
case-group Average, Weighted Avg and Z-Score values. Those overlapping groups
describe the same 819 submitted items. They are not seven disjoint test sets or seven
independently reproduced pooled-training experiments.

The result comes from a user-provided organizer email. The attached per-item
result archive has not been inspected, including how the two refusal markers
were scored. The overall aggregation formula has not been independently
verified. This is not a local F1, accuracy, Elo, public rank, paper reproduction
or demonstrated improvement over the earlier subset.

## 1. Pin the native integration and data

The committed [integration guide](../../../integrations/memorybench/README.md)
documents installation, environment variables, cache behavior and contract
tests. Its installer and package utility are public. The bounded V5 execution,
selector, recovery and audit scripts described below are **private execution
artifacts, not committed runnable reproduction tools**. Installing the plugin
alone does not reproduce this score.

The recorded pins are:

| Component | Pin |
| --- | --- |
| Native MemoryBench code | [`THUIR/MemoryBench`](https://github.com/THUIR/MemoryBench/tree/247ba8c1e327fa297c0d57aaad0e3cbd66bac656), commit `247ba8c1e327fa297c0d57aaad0e3cbd66bac656` |
| Official dataset | [`THUIR/MemoryBench`](https://huggingface.co/datasets/THUIR/MemoryBench/tree/3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d), revision `3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d` |
| Engine install pin documented by the public integration | `d952129834f1749c412a04528a4f9138d94a7d74` |
| Provider SDK used for controlled transport | `anthropic==0.84.0` |
| V5 selector embedding asset | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, ONNX, 384 dimensions |
| Official test-index manifest SHA-256 | `ee72b718ef79a863d9e2e5fb87706e062efa0e949c1141c7049507e44edf28df` |

The wrapper checkout used for V5 was `a684eb9796d8a5386819c5bcda52ba2be94f5109`.
That checkout and the private selector/runner are separate from the engine install
pin; the original training adapter also preceded this wrapper version. The
execution manifest bound the V5 files by their actual source bytes. The local embedding
helper SHA-256 was
`ed7239440cc54886232b32391a033a444e4e0506d7f1417c727eee031a75cc8f`.

For a new experiment, follow the committed guide to install into the pinned
upstream checkout and configure a migrated UTF-8 PostgreSQL/pgvector database.
The recorded retrieval profile used local ONNX embeddings, fuzzy/tag/vector
RRF weights `1/1/2`, candidate limit 200, lexical OR enabled and `retrieve_k=50`.
Use fresh run/world identifiers for independent experiments; retain intended
sharing within each experiment and its training/corpus forks. Never point a new
experiment at an old world and call it a fresh run.

Use the committed `package_submission.py fetch-manifest` command to obtain the
official indices at the pinned revision, repeating `--dataset` for every
included dataset. Check exact index sets, not consecutive ranges. Retain the
index/shard hashes and effective imported source hashes; package metadata alone
does not prove that the intended engine code was imported.

## 2. Establish and freeze the off-policy memory states

The original run used 28 independent single-dataset off-policy experiments,
with native memory forks within each:

- LoCoMo: 200 generated training dialogs across ten datasets, with the native
  training/feedback loop and local LoCoMo F1 feedback.
- DialSim: 199 generated training dialogs across three datasets. The native
  loop allowed at most three answer rounds, with native exact-match shortcuts,
  Correct/Wrong judge prompts and feedback/early-stop semantics. The controlled
  judge route explicitly used Sonnet 4.6, temperature 0.2 and a 1,024-token cap;
  training generation used a 2,048-token cap.
- The other 15 datasets: replay of 2,845 official neutral training-dialog rows,
  rather than relabeling another baseline's corpus-specific generated dialogs.

Training feedback is adaptation data. Test gold labels were not added to
generation requests; development nevertheless used prior outputs and diagnostics
from this same cohort. Native observed training context was retained,
including augmented first-round messages where the native protocol stores
them. Source IDs, text, roles, worlds, stored vectors and retrieved contexts
were audited against the pinned inputs. V5 **reused those audited memory states
and frozen top-50 candidate pools**; it did not rerun native training or issue
new retrieval queries for each corrected answer.

## 3. Freeze the V5 answer-only replay

Earlier outputs and intermediate corrected variants were preserved. Reviews
found instruction/context pressure, topic drift and evidence-selection losses;
the V5 protocol was frozen as a separate full-cohort replay. It was not a patch
that replaced only answers judged wrong. The fixed development items had been
reviewed in earlier versions, so neither the cohort nor the 28-item development
pilot is an unseen holdout.

For each item's own frozen 50 records, V5 extracted whole provenance-linked
utterances or paragraphs and deduplicated exact units while retaining source
rank/span attribution. Facts were not pooled across unrelated items or worlds.
Selected text remained exact source substrings; oversized units could be
omitted, not head/tail-clipped into fragments. The serialized evidence object
was bounded to **12,000 UTF-8 bytes**, separate from the complete current task
and other request fields.

The selector combined local MiniLM semantic ranking with BM25 using RRF weights
**2 semantic : 1 BM25**, smoothing 60. All query/unit token windows were scored:
254 content tokens, stride 190, with attention-masked mean/L2 embeddings and
mean-over-query-windows of the best unit-window cosine. There was no silent
256-token truncation or provider fallback.

Exact native conversation templates enabled question-plus-persona/date or
participant focus for **scoring only**, with certified primary dated utterances
prioritized ahead of historical records. This applied to 80 items. The other
739 retained the prior whole-unit fallback, including 21 official LoCoMo MCQ
variants not recognized by that template parser. The full official current
task, options and qualifiers remained in the model request in both paths.

This selection is lossy. Primary-first allocation can crowd out useful history,
and BM25/semantic ranking can miss a directly relevant turn. Historical
assistant uncertainty is not proof that the underlying source lacks evidence.
Mechanical source fidelity does not establish answer correctness; unsupported
details, topic drift and ordinary answer errors remain possible.

All frozen V5 requests used `claude-sonnet-4-6`, temperature **0**, and these
measured output-token caps:

| Datasets | Cap |
| --- | ---: |
| LimitGen-Syn | 2,048 |
| JRE-L | 4,096 |
| DialSim, IdeaBench, JuDGE, LexEval, Locomo-0…9, NFCats, WritingPrompts | 8,192 |
| WritingBench datasets | 16,000 |
| HelloBench datasets | 16,384 |

V5 LoCoMo used 8,192; the earlier 50-item LoCoMo run used 4,096. Do not conflate
their protocols or scores. The frozen V5 request-cohort SHA-256 is
`8f124686d6a1d6313702a605f233992df8d1821003bd3dca2daf9d793b3d6cba`.

## 4. Execute without silently retrying or rewriting outcomes

The private runner used two workers and one shared **$500 total ledger**,
retaining earlier charges and unresolved reservations. Each logical request
was bound to an official item, purpose, request hash and reservation before
dispatch; actual usage settled its charge. The public integration itself does
not implement that spending guard. A future runner must provide and review its
own explicit budget enforcement.

Two V5 calls ended in `APIConnectionError` without a returned response. Their
processing/billing remained unknown and their combined **$0.664194** reservation
was retained. They were not silently treated as zero-cost failures or retried.
The next phase processed only unattempted items.

Transport then used the unchanged logical request through SDK streaming with
zero automatic retries. Private evidence retained raw SSE events and a complete
message. Completion required a valid message start, consistent block indices,
closed blocks, terminal stop/usage, explicit `message_stop`, and agreement with
the final message. A clean EOF or accumulated SDK snapshot alone was not enough
to settle a partial answer. Post-run auditors reconstructed raw text-block
starts/deltas and checked them against the saved/API text.

The first continuation stopped on a fully received, charged empty refusal.
After review, a separate scheduler continued unrelated unattempted items while
preserving such refusals as blocked raw outcomes. It did not regenerate them.
The user explicitly authorized one identical-request child attempt for each
of the two lost-response items. Execution waited until **817 authentic
outcomes** had been audited. Prospective parent-linked purposes and receipts fixed which children
would supply those indices before their text was seen; no best-of selection was
used. Original unknown calls and holds remained in the ledger.

The final accounting was **821 V5 attempts: 819 settled outcomes plus two
original unknown calls**. Raw outcomes contained 812 `end_turn`, five
`max_tokens` and two empty `refusal` results. Capped answers were retained rather
than regenerated for quality. Audits checked official coverage, actual
request/API/saved-text hashes, SSE completion, model/usage/cost bindings and
preservation of every earlier ledger entry. No text was repaired during these
audits.

## 5. Package, validate and retain the transport history

Keep the upload archive minimal: exactly 28
`dataset/<official-name>/predict.json` members, each a JSON array with only
`dataset`, integer `test_idx` and string `response`. Preserve literal dataset
names, including `&`, and exact official indices. Verify unique 819-item
coverage, strict JSON, CRC, paths, member/count/size limits and each decoded
response's hash. Keep provenance, prompts, ledger, database snapshots and
credentials outside the ZIP. Local package validity is not website validation.

Earlier full-suite uploads encountered a size-limit failure for an approximately
2.20 MB archive despite the advertised 500 MiB limit. Two UI attempts with
different compression and one genuine direct form-endpoint POST failed; the
direct attempt returned HTTP 413. This did not establish that ZIP contents or
empty answers caused the size rejection. The organizer subsequently reported
increasing the API limit in [MemoryBench issue 9](https://github.com/THUIR/MemoryBench/issues/9).

A temporary three-part, complete-dataset route evaluated one older subset;
two other parts failed validation over five empty outputs. Those older outputs
and that subset result were preserved. They are not a comparable V5 baseline,
nor do separate subset receipts establish one full-suite result.

The later raw V5 ZIP was received as submission
`8c08a57f46c34a74b92b76817be2aa49`, then failed organizer validation:
`invalid_response — response must be a non-empty string` for two entries in
`dataset/IdeaBench/predict.json`. The local audit identifies indices **995 and
1881** as the actual empty provider refusals; the email did not enumerate those
indices.

## 6. Disclose the two refusal-format changes and the result

After explicit user authorization, a separate submission adapter replaced only
those two empty submission strings with the same literal marker:

```text
[No answer: provider returned an empty refusal.]
```

This is **adapter text, not model-generated text**. No new inference occurred.
The 817 other decoded response strings, all 28 member paths and all official
indices were unchanged; the original API records and raw ZIP remain preserved.
The marker was disclosed in the submission notes. An organizer-approved
refusal convention was not obtained, and later evaluation completion does not
prove how these individual entries were handled.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Raw V5 ZIP: 817 nonempty + two actual empty refusals | 2,031,670 | `ee30fe9640dd604d71fa4d20482a2c8a530e460fa0ba6c84b131d477d583521b` |
| Evaluated refusal-format ZIP: 817 raw strings + two adapter markers | 2,031,741 | `3cb6f3fbf2b76390d06f72a9ff3baf4e5f26be004e129d45526677b917ae8f43` |

The latter was received as `693dfc58d3eb4fe49f1311b1bf9a6fe6` and subsequently
reported evaluated by the organizer. Its reported metrics belong to that exact
refusal-format submission. An initial upload receipt, completed evaluation and
public leaderboard inclusion are separate states; no public rank is asserted.

## 7. What a future reproducer must retain

Use the committed guide for the native setup and package commands. To reproduce
this exact V5 execution, the private selector/orchestrators, frozen candidate
pools, complete task/request/provenance manifests, trained memory snapshots and
provider records would additionally be needed. Their fingerprints identify this
run but do not make those artifacts publicly available. This document is a
protocol and incident record, not a claim of a public one-command score rerun.

For a new, separately identified run:

1. Pin code/data/model assets and exact official index sets; record imported
   sources, retrieval settings, training route, model parameters and every cap.
2. Use fresh worlds while preserving intended within-experiment sharing; freeze
   allowed training/corpus sources and exclude test golds from generation.
3. Freeze the evidence policy and full request cohort before dispatch. Label
   development exposure honestly; retain prior versions instead of overwriting
   outputs or selecting favorable cases across versions.
4. Audit source/span/vector provenance, request fidelity, actual usage and the
   complete shared ledger. Stop unknown transport outcomes; retain their holds.
   Any exceptional retry needs its own explicit authority and parent linkage.
5. Review raw coverage and output-format blocks separately. Preserve refusals,
   truncations and mistakes. Keep raw model output distinct from any disclosed
   submission-format rendering.
6. Recheck the live official form/limits and obtain a clear refusal convention
   where possible. Review the exact archive, notes and claims with the panel
   before submission; retain receipt, validation and evaluation separately.

Private logs, contacts, screenshots, raw responses, credentials and database
snapshots are not published with these notes. See the linked result record for
the measured result and its remaining verification limits.
