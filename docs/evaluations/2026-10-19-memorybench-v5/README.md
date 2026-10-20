# MemoryBench: organizer-evaluated V5 replay

The organizer reported **Overall average 0.6261** for submission
`693dfc58d3eb4fe49f1311b1bf9a6fe6`, system `mem++_full_v5_refusal_format`.
The source is a user-provided evaluation-complete email; its private screenshot
and contact details are not published.

| Case group | Average | Weighted Avg | Z-Score |
| --- | ---: | ---: | ---: |
| domain/Academic&Knowledge | 0.7129 | 0.6671 | 1.2178 |
| domain/Legal | 0.5458 | 0.5388 | 0.2974 |
| domain/Open-Domain | 0.5376 | 0.7133 | 1.2492 |
| task/Long-Long | 0.7901 | 0.7562 | 1.7990 |
| task/Long-Short | 0.2741 | 0.3743 | 0.0235 |
| task/Short-Long | 0.7376 | 0.7320 | 0.8069 |
| task/Short-Short | 0.7947 | 0.7947 | 0.9101 |

These are the organizer's reported fields, not accuracy, F1 or Elo. The overall
0.6261 is transcribed rather than recomputed from these seven rows. Its
aggregation formula, the attached per-item result archive and individual
refusal-marker scoring have not been independently verified. No public rank or
automatic leaderboard publication is established. The public heatmap uses a
`weighted_average` field, but identical normalization/reference statistics and
weighting between that table and this email have not been confirmed; the
published base models also differ from Sonnet 4.6. No comparative superiority
or memory-only gain is claimed.

Submitted coverage was **819 unique official indices, 28 datasets**, represented
by exactly 28 `dataset/<name>/predict.json` files. Domains and task groups are
overlapping labels, not seven independent experiments. These are submitted
counts, not independently inspected scored-item counts.

This was a separately authorized **answer-only replay** using Sonnet 4.6,
temperature 0, local ONNX MiniLM-384 / PostgreSQL memory, frozen top-50 candidate
pools and at most 12,000 UTF-8 evidence bytes. It reused audited native memory
states and did not rerun their training lifecycle. The cohort was exposed during
development; it is not an unseen holdout. Raw outcomes were 812 `end_turn`, five
`max_tokens`, and two actual empty provider refusals. To satisfy website
nonempty-string validation, the submitted variant renders only those two
refusals as disclosed adapter markers. Its other 817 response strings are
unchanged. The marker is not model-generated text or a claimed substantive
answer, and an organizer-approved convention has not been obtained.

Evaluated ZIP: **2,031,741 bytes**, SHA-256
`3cb6f3fbf2b76390d06f72a9ff3baf4e5f26be004e129d45526677b917ae8f43`.
The original raw ZIP and all earlier versions remain preserved privately.

- [Procedure, source pins, caps, recovery and submission issues](PROCEDURE.md)
- [Structured result, dataset counts, raw stop reasons and verification limits](results.json)
- [Exact official test indices and shard hashes](test-indices.json)
- [Public native setup and packaging guide](../../../integrations/memorybench/README.md)

The bounded selector, execution/recovery tools and frozen run artifacts remain
private. These notes make the configuration and incident history explicit;
they do not provide a public one-command reproduction of this exact score.
Earlier paper scores, the original full-suite attempt and the 50-item subset
remain separate results with different protocols and coverage.
