# MemoryBench full coverage — 2026-10-16

Local generation completed for **819 official test items across 28 datasets**. This is coverage and provenance evidence, not a new accuracy score. The initial full archive upload was rejected with “Upload exceeds the server size limit.” Its 2,195,145 bytes were below the published 524,288,000-byte limit. No receipt, accepted full submission, organizer score or public rank is established. A lossless BZIP2 ZIP reduced transport size to 1,686,428 bytes while preserving every prediction-file byte, but the website rejected it with the same size-limit message. After those two browser failures, one reviewed direct-client attempt sent the same DEFLATE archive to the exact public form endpoint (`POST /api/v1/submissions`), with no redirects or automatic retries. The 2,197,475-byte multipart request returned **HTTP 413, “Request Entity Too Large”**, with an nginx/1.18.0 (Ubuntu) HTML error page. This reproduces rejection outside the in-app browser; it is consistent with a request-size restriction, but does not establish the configured threshold or responsible layer. Attempts stopped after this third rejection. Actual limits and empty-response acceptance remain unknown. See [sanitized HTTP diagnostic](upload-diagnostic.json).

The run preserves the original 50 evaluated LoCoMo outputs and adds 769 outputs. The existing 50-item native local F1 (0.3950822) and organizer email average (0.6560) apply only to that original subset; neither is a full-suite result. [Original subset evidence](../2026-10-15-memorybench-locomo-family/README.md).

## Protocol

28 independent native SINGLE-dataset off-policy experiments use pinned data `THUIR/MemoryBench` revision `3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d`. The website aggregates these datasets into seven case groups. This does not reproduce the seven pooled native domain/task CLI experiments, which produce overlapping predictions.

Claude Sonnet 4.6 (`claude-sonnet-4-6`), temperature 1, local ONNX MiniLM-L6-v2 384-dimensional embeddings, PostgreSQL/pgvector, and retrieval k=50 stay fixed. Engine pin: `d952129834f1749c412a04528a4f9138d94a7d74`; native MemoryBench pin: `247ba8c1e327fa297c0d57aaad0e3cbd66bac656`. The 769 new outputs use native configured test caps; the original 50 retain their previously evaluated 4096-token test cap (the native LoCoMo configuration is 8192). Effective per-dataset caps appear below. Own training responses use 2048 tokens maximum. Native contextualized dialog retention and separate memory-world forks are preserved; organizational sharing remains enabled within each experiment.

The 15 noncorpus datasets replay 2,845 official universal training dialogs. The three DialSim datasets generate 199 own training dialogs through the native up-to-three-round process, with explicit Sonnet 4.6 Correct/Wrong feedback (1024-token maximum, temperature 0.2). The ten original LoCoMo datasets retain their 200 own training dialogs and native local F1 feedback. No optional test scorers were run for the 769 new outputs.

## Retained outputs and verification

All 819 official `(dataset, test_idx)` pairs are present exactly once, with no extra indices. The original ten LoCoMo ZIP members remain byte-for-byte unchanged. The new outputs contain 714 `end_turn`, 48 `max_tokens`, and seven explicit `refusal` stop reasons. Five refusals are empty strings and two contain text. Every original output is retained without regeneration, cap increases, replacement or selective exclusion. Empty-response website acceptance remains unverified and is a separate upload gate.

All 769 new outputs match their frozen API requests, retrieved contexts and settled response records. The 15 noncorpus provenance audits checked 13,041 stored vectors and 35,900 retrieved-source appearances. DialSim audits replayed all 199 training dialogs and 390 feedback decisions against recorded API requests, checked 116,226 stored-vector appearances across nine distinct memory worlds and 12,500 retrieved-source appearances. No foreign records were found in the checked scopes. These checks establish provenance and scope isolation, not answer correctness or an independent semantic review of training judgments.

The extension made 1,739 paid calls costing $259.265091, including training, feedback and 769 test calls. The original 596 calls cost $20.552814; 718 free input-token-count calls are separate preparation. The shared budget also covers earlier work and is not a separate budget for each dataset.

Initial rejected DEFLATE ZIP SHA-256: `ed324ed17f17063afb59e67da5467dd7b5ecc3f262580819d43a2652f1fb3bc3`. Original subset ZIP SHA-256: `bf9077824282a8abcf3a5506c87b9207e3e3c3da501dd1e4df8077870f48ce2c`. Rejected lossless BZIP2 ZIP SHA-256: `0e6d73ccf4fdef303caa4498b5fcbba5ff5fb03ad0f6b75561ac35dbedabf0cc`. Machine-readable coverage, limits, output exceptions and audit hashes: [results.json](results.json). Private raw API records and prompts are retained locally.

## Exact split counts

Training counts below describe replayed official dialogs for noncorpus datasets and own generated dialogs for LoCoMo/DialSim. Test counts are the complete pinned official test splits.

| Dataset | Training dialogs | Test items | Test output cap |
|:---|---:|---:|---:|
| DialSim-bigbang | 66 | 17 | 8192 |
| DialSim-friends | 66 | 17 | 8192 |
| DialSim-theoffice | 67 | 17 | 8192 |
| HelloBench-Academic&Knowledge-QA | 170 | 43 | 16384 |
| HelloBench-Academic&Knowledge-Writing | 65 | 17 | 16384 |
| HelloBench-Creative&Design | 181 | 47 | 16384 |
| IdeaBench | 200 | 50 | 8192 |
| JRE-L | 200 | 50 | 4096 |
| JuDGE | 200 | 50 | 2048 |
| LexEval-Judge | 200 | 50 | 8192 |
| LexEval-QA | 200 | 50 | 8192 |
| LexEval-Summarization | 200 | 50 | 8192 |
| LimitGen-Syn | 200 | 50 | 500 |
| Locomo-0 | 20 | 5 | 4096 |
| Locomo-1 | 20 | 5 | 4096 |
| Locomo-2 | 20 | 5 | 4096 |
| Locomo-3 | 20 | 5 | 4096 |
| Locomo-4 | 20 | 5 | 4096 |
| Locomo-5 | 20 | 5 | 4096 |
| Locomo-6 | 20 | 5 | 4096 |
| Locomo-7 | 20 | 5 | 4096 |
| Locomo-8 | 20 | 5 | 4096 |
| Locomo-9 | 20 | 5 | 4096 |
| NFCats | 200 | 50 | 8192 |
| WritingBench-Academic&Engineering | 133 | 34 | 16000 |
| WritingBench-Creative&Design | 336 | 86 | 16000 |
| WritingBench-Politics&Law | 160 | 41 | 16000 |
| WritingPrompts | 200 | 50 | 8192 |
