# Engineering Evidence Report (PROOF.md)

> **Candidate Note:** This report documents the investigation, architectural decisions, and empirical results. Findings are tied to saved traces, tests, and diagnostic plots; unmeasured effects are identified explicitly.

---

## Candidate Information

- **Candidate Name:** Farnaz Babaloo (@FarnazBabaloo)
- **Submission Date:** 2026-10-10 (report preparation date)
- **Execution Target:** macOS ARM64, Python 3.11.5, CPU
- **Track Scope Completed:**
  - [x] Required Core (A–D: diagnosis, fusion, normalization, chunking)
  - [x] Elective 1: E — Evidence Reranking & Diversity
  - [x] Elective 2: F — Adversarial & Noise Document Defense
  - [x] Staff Track: Adaptive System-1 Decision Judge (implemented and evaluated; generalization limits below)

The detailed task specification defines A–D and two of E–H; the template's three-core-task wording is an earlier summary. Additional clarification work addresses the general rubric. The optional Staff judge is integrated; elective H completion is not claimed.

---

## Executive Summary & Scorecard

| Metric | Naive Baseline | Your Final Pipeline | Delta ($\Delta$) | Target |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 30% (6/20 visible) | 100% (20/20 visible) | +70 percentage points | ≥65% held-out; unverified |
| **Retrieval Recall@5** | 55% (11/20 raw) | 100% (20/20 ranked) | +45 percentage points | ≥75% on evaluated set |
| **Clarification Accuracy** | 40% (8/20; pre-fix v1) | 90% (18/20; pre-Staff fresh v2) | Not paired: different sets | >50% rubric; met on manual sample |
| **Mean Query Latency (CPU)** | 8.3 ms | 28.5 ms | +20.2 ms | <100 ms |
| **Distractor / Poison Ingestion** | Original citation rate not measured | 0/40 final top-3 citations | Not established | 0% |

Accuracy sources: `traces/metrics_baseline.json`, `traces/metrics_v65.json` (intermediate 65%), and `traces/metrics.json`. Recall@5 is recomputed from existing traces in `traces/proof_scorecard.json`: original raw top-five versus final ranked top-five, with line references/IDs. Original raw Hit@3 was 40%; that differs from the original answer evaluator's 30%. Final citation Hit@1 and Hit@3 are both 100% on the visible set.

The visible evaluator measures gold-document citation success, not full semantic answer correctness. Public corpus questions overlap this set; hidden accuracy remains unknown. Manual clarification sets are balanced (20 ambiguous and 20 specific each) and authored by the same assistant after seeing implementation. Exact fresh queries were frozen before execution, but scenarios overlap development; this is not independent annotation or a population estimate. The original v1 set is now development data. Its paired cached regression improved from 8/20 to 20/20 ambiguity detections; that is not unseen-data evidence.

Final adaptive p50/p95: 22.1/29.5 ms. Latest live mean after routing/calibration revisions is 28.5 ms; the latency-only revision measured 28.0 ms. The first Staff implementation averaged 72.7 ms (`traces/metrics_pre_latency.json`); bounded normalization memoization and precompiled bank patterns reduced the latest live mean to 28.0 ms (61.5% lower). Pre-Staff mean was 34.8 ms (`traces/metrics_pre_staff.json`). Host load and startup cost were not controlled, so timing changes cannot be attributed to a single fix. Full test suite: **103 passed**, no exclusions. Current quota: **262/1000**, with **738 remaining**; no key rotation was used.

---

## Investigation & Architectural Changes

### Investigation 1: Retrieval Characterization, Hybrid Fusion & Candidate Coverage (A/B)

#### 1. Observed Symptom

The original CLI bypassed the registered retriever, so retrieval changes did not reach `bench`/`bepar`. Original visible answer accuracy was 6/20. Even the remote-only revision reached only 13/20; seven gold IDs were absent from the captured remote candidate pools.

#### 2. Falsifiable Hypothesis

If missing candidates cause the remaining errors, reranking the same remote pools cannot recover them, whereas a general lexical index of the public corpus should increase gold-document coverage. Rank fusion should be insensitive to score scaling, but its accuracy benefit must be measured rather than assumed.

#### 3. Diagnostic Probes & Exact Trace IDs

Executed paths: `python -m pytest tests/ -q` invokes the real CLI `bench`; `python rubbish.py bekesh` plots recorded observations. The following 20 pre-Staff visible traces include short, long and entity-specific cases and exceed the required trace counts. They document the core investigation; current adaptive trace IDs are in `traces/requirements_audit.json` and detailed per-round records in `traces/trace.jsonl`. The complete client log now contains 104 distinct queries. These are not all independently designed controlled probes.

| Trace ID | Query | Remote Top1–Top5 fused spread | Decision |
|---|---|---:|---|
| `client_c805c0e199d0429e9e307872008ede43` | من یه بار فقط جریمه مالیاتی برام بریدن، چه طوری حذفش کنم؟ | 0.4156 | answer |
| `client_5376e64ed5b74676b3526b595a33108f` | اگر شرکت محکومیت مالی داشته باشد، چقدر رتبه‌اش کاهش پیدا میکنه؟ | 0.0728 | answer |
| `client_7631ebaa2c4543de827bdd80dde320ed` | اگر یک چک برگشتی داشته باشم اما بلافاصله (ظرف ۲۴ ساعت) مبلغ را تأمین و چک را رفع سوءاثر کنم، باز هم در سابقه اعتباری من ثبت می‌شود؟ | 0.3236 | answer |
| `client_0a2a9b3ca7674854b8988fb6ba1869c4` | ۴۸۰ شدم. چرا؟ | 1.0000 | answer |
| `client_3973b52c468e4d4e9622864ec7318fb6` | برای اشخاص حقوقی، رخدادهای عملیاتی مانند «تعطیلی موقت» یا «توقف تولید» در گزارش اعتباری بازتاب دارد؟ | 0.6019 | answer |
| `client_c523375923fe4775861a6e5e903a1d9a` | اگر در قراردادهایی که شرکت به‌عنوان ضامن حضور دارد، اقساط با تأخیر بازپرداخت شوند اما پرداخت‌ در نهایت انجام شود، آیا این تأخیر بر رتبه اعتباری شرکت ضامن تأثیر دارد؟ | 0.2179 | answer |
| `client_f915c6fe410b4f498188a0b848cff532` | شرکت سودآوره ولی امتیاز نداره چرا؟ | 0.2219 | answer |
| `client_ac52308f954b4ceda9d0f4b2e126a97d` | اگر شرکت در یک بانک خاص سابقه تاخیر در پرداخت بدهی داشته باشد، آیا این موضوع بر دریافت تسهیلات از سایر بانک‌ها تاثیر می‌گذارد؟ | 0.2167 | answer |
| `client_c194a17ad5454362be61cfd0ce6232bc` | آیا امکان مقایسه امتیاز اعتباری یک شخص با شخص دیگر وجود دارد؟ | 0.1782 | answer |
| `client_8530de97b5714e6c81314ea2abbb169d` | نمی‌خوام این چکم که برگشت خورده تو سابقه‌م بیاد، چه طوری حذفش کنم؟ | 0.3564 | answer |
| `client_3f7340762d034e49800fa645911e6126` | من ضامن یک وام شدم. اگر شخص تاخیر زیادی در پرداخت قسط های وام داشته باشد، آیا امتیاز من به شدت کم می شود؟ | 0.2540 | answer |
| `client_d1ca96ae42d74b82ae495c2730cfac1f` | چرا با وجود بازپرداخت به‌موقع اقساط، امتیاز اعتباری من هنوز افزایش نیافته است؟ | 0.3180 | answer |
| `client_7d28ec6d85c848bba0c727e5895390be` | عدد صادرات/واردات توی رتبه اعتباری موثره؟ | 0.1660 | answer |
| `client_fede24f825424ca8a68c9cbb8aaaba4d` | آیا داشتن چند ضامن برای یک وام، در مقابل یک ضامن، تاثیر متفاوتی بر رتبه اعتباری ضامنان دارد؟ | 0.2617 | answer |
| `client_5900d17383aa4e219bd6907458f5eec5` | داشتن تعداد زیاد اعتبارات اسنادی همیشه خوب است؟ | 0.4783 | answer |
| `client_4f26e646752743368eb8029eb34b06a3` | رتبه C3 یعنی چی؟ | 0.3421 | answer |
| `client_02e48a22ae2f42c8b0779d4037eb4c04` | اگر شرکت بدهی به تامین‌کنندگان داشته باشد، آیا این بدهی در گزارش اعتباری ثبت می‌شود؟ | 0.1429 | answer |
| `client_20b78d175d0b4731a84e43e9a5444bf9` | من بدهی مالیاتی پرداخت نشده داشتم و رتبه ام E3 شده است. اگر آن را پرداخت کنم، چه اتفاقی می افتد؟ | 0.2832 | answer |
| `client_c3d8b084aef44de0bcb6abb4a7f66ea9` | چرا وقتی شرکت هیچ سابقه تاخیری ندارد، بالاترین امتیاز رو نداره؟ | 0.2089 | answer |
| `client_ea9457be67d0455b8d278f6e7dc12293` | تغییر محل قانونی و آدرس شرکت چه تأثیری بر امتیاز اعتباری دارد؟ | 0.1415 | answer |

#### 4. Quantitative Evidence & Sub-score Analysis

| Configuration / metric | Measured result |
|---|---:|
| Original raw Hit@3 | 8/20 = 40% |
| Original raw Recall@5 | 11/20 = 55% |
| Latest remote candidate-pool gold recall | 65% |
| Merged remote + local candidate-pool gold recall | 100% |
| Final ranked Recall@5 | 100% |

For trace `client_c805c0e199d0429e9e307872008ede43`, remote document 66 has BM25 11.0639, dense 0.1260 and fused 1.0000; the final first citation is reference 173. Its remote Top1–Top5 fused spread is 0.4156. Raw remote scores are retained, not replaced by local scores.

Evidence: `traces/requirements_audit.json`, `traces/replay_analysis.json`, `traces/proof_scorecard.json`; plots `score_spread.png`, `length_vs_bm25.png` and `rrf_ablation.png` in `plots/`.

#### 5. Root Cause Analysis

CLI execution and the registry were disconnected. A candidate-only reranker cannot recover an absent document. Remote channel magnitudes also differ, although the sealed server's internal implementation was not inspected. The measured length/BM25 relationship is descriptive, not proof of a causal length penalty.

#### 6. Architectural Modification

- **Module:** `rubbish_rag/execution.py`, `orchestrator.py`, `pipeline.py`, `local_search.py` and `rubbish.py`.
- **Mechanism:** Share registered stages across CLI/orchestrator; issue one bounded remote call per factual query. Fuse channel ranks with reciprocal 1/(60+rank), average available channels and multiply by two. The public-corpus BM25 index weights question terms twice, includes keyword/BriefAnswer terms, uses k1=1.2 and b=0.75, admits coverage ≥0.35 and at most six local candidates. Corpus row positions remain IDs; the actual CSV has 442 rows. Production retrieval does not load benchmark labels.

#### 7. Ablation Experiment

| Fixed-pool ablation | Visible Hit@3 |
|---|---:|
| Full pipeline | 100% |
| Remove RRF | 100% |
| Remove local supplementation | 65% |
| Remote raw fusion only, without local search/reranking | 50% |
| Remote RRF only, same pools and exclusions | 40% |

`tools/replay_evidence.py` and `tools/audit_pdf_requirements.py` use cached remote replies and consume zero calls. RRF alone is worse on these pools and has no measurable gain in the full pipeline. The 35-point supplementation gain must not be attributed to RRF. An offline scaling test verifies rank-fusion invariance to a 100× BM25 rescaling.

#### 8. Production Trade-offs

- **Latency Overhead:** Final adaptive pipeline mean 28.5 ms versus 8.3 ms originally; per-component overhead was not isolated.
- **Memory Overhead:** Cached public corpus and lexical postings; memory was not benchmarked.
- **Edge Cases Considered:** Missing candidates, unequal channel availability, score provenance and zero-score evidence. Lexical supplementation can miss low-overlap paraphrases. The corpus must be available locally; no sealed index was inspected or modified.

---

### Investigation 2: Persian Character Normalization & Slot Preservation (C)

#### 1. Observed Symptom

The original normalizer was not idempotent for colloquial forms. An ASCII-only rewrite of `۴۸۰ شدم. چرا؟` lost a useful numeric match. Bank-name substrings and rank digits also risked incorrect slot extraction.

#### 2. Falsifiable Hypothesis

Normalizing script and whitespace consistently should remove superficial variants without changing numeric meaning. Keeping both original and canonical query forms in one request should retain server matches that depend on Persian digits.

#### 3. Diagnostic Probes & Exact Trace IDs

The numeric query appears in real benchmark traces `client_ede22140ffdf4f57bbb532bc3df7702f` (failed rewrite), `client_968f3e88026b470697a7e9cd9c32e8b6` (original + canonical) and `client_0a2a9b3ca7674854b8988fb6ba1869c4` (final pipeline). These were captured through benchmark execution, not separate invented probe commands. Inspect them with `python rubbish.py inspect-trace <id>`. Normalization/slot fixtures run via `python -m pytest tests/test_normalization.py tests/test_slots.py tests/test_engineering.py -q`.

#### 4. Quantitative Evidence & Sub-score Analysis

The first stored candidate in each numeric trace is shown below; these are remote channel scores.

| Trace ID | Query Variant | Top-1 Doc ID | BM25 Score | Dense Score | Fused Score | Outcome |
|---|---|---:|---:|---:|---:|---|
| `client_ede22140ffdf4f57bbb532bc3df7702f` | ASCII-only rewrite | 0 | 0.0000 | 0.0000 | 0.0000 | Clarified; useful numeric match absent |
| `client_968f3e88026b470697a7e9cd9c32e8b6` | Original + canonical | 110 | 7.0553 | 0.0772 | 1.0000 | D2 match restored; citation 110 |
| `client_0a2a9b3ca7674854b8988fb6ba1869c4` | Final combined pipeline | 110 | 7.0553 | 0.0772 | 1.0000 | Local reference 311 first; visible gold hit |

Restoring remote document 110 alone did not satisfy the visible gold-ID contract; local supplementation supplied the distinct expected reference.

A later CPU profile of 20 cached queries recorded 41,343 normalization calls and repeated sorting/normalization of the fixed bank vocabulary. The latency-only revision memoizes the pure string normalizer (8192-entry LRU) and prepares longest-first bank regexes once at import. Full-text keys ensure that forged text under a valid ID cannot reuse another excerpt’s normalization. Verification, reranking, retrieval budgets and decision rules remain active.

`tools/benchmark_latency.py` compared the same captured pools across one initial pass and five warm passes (120 queries per version), including trace writes but excluding process imports. Initial-pass mean fell from **59.03 to 13.84 ms**; warm mean **52.82 to 4.96 ms**, warm p95 **63.25 to 6.53 ms**. Complete predictions, routing, selected evidence and per-round decisions are equivalent (only score roundoff up to 1e-12 is allowed). This mocked-transport timing is distinct from the **72.7 → 28.0 ms** live benchmark. Full suite remains 96/96 and visible accuracy 20/20. Evidence: `traces/latency_before.json`, `traces/latency_after.json`, and `traces/metrics_pre_latency.json`.

#### 5. Root Cause Analysis

Colloquial replacement inserted ZWNJ after whitespace folding, changing the result on a second normalization pass. ASCII-only canonicalization also removed a representation the service matched. Loose entity matching confused shorter bank names with longer names and rank digits with numeric amounts.

#### 6. Architectural Modification

- **Module:** `rubbish_rag/normalize_fa.py`, `slots.py`, `pipeline.py`.
- **Mechanism:** NFKC, Arabic/Persian digit mapping, Arabic-letter folding, colloquial expansion before final ZWNJ/whitespace folding, and preservation of decimal meaning. Match longest bank names at word boundaries; distinguish export/import volume from Bank Saderat. Exclude rank codes from amounts and distinguish legal from individual persona. Send original and normalized forms together when they differ.

#### 7. Ablation Experiment

The two historical numeric traces expose a rewrite regression and its correction, but other changes occurred between runs; they are not a controlled global normalization ablation. No isolated overall accuracy gain for normalization or slot filtering is claimed. Regression tests verify idempotence, numeric meaning, bank variants and rank/persona conflicts.

#### 8. Production Trade-offs

- **Latency Overhead:** Not separately measured; string normalization uses standard-library operations.
- **Memory Overhead:** Replacement/regex tables and an LRU of at most 8192 input/result string pairs; retained bytes are text-length-dependent and were not profiled. There is no response or evidence-verdict cache.
- **Edge Cases Considered:** Arabic letters/digits, half-spaces, spaced ranks, decimals and negation preservation. The historical fresh-set year/score confusion is documented in Investigation 5. The latest guard requires a rank or score expression, so a year or loan amount alone cannot satisfy the score slot.

---

### Investigation 3: Boundary-Aware Chunking & Metadata Preservation (D)

#### 1. Observed Symptom

The original fixed 300-character slicing could cut a sentence or numeric expression and omit category context. The remote D2 candidate in the numeric trace ends mid-sentence at the service boundary.

#### 2. Falsifiable Hypothesis

Sentence packing with word-boundary fallback and a metadata prefix should preserve all local source words and domain anchors while keeping ordinary body segments bounded.

#### 3. Diagnostic Probes & Exact Trace IDs

Remote example: `client_0a2a9b3ca7674854b8988fb6ba1869c4`, document 110. Local validation command: `python -m pytest tests/test_engineering.py -q -k chunk_coverage`. This is an offline synthetic chunking test, not a new retrieval probe; no API trace ID is fabricated for it.

#### 4. Quantitative Evidence & Sub-score Analysis

The coverage test reconstructs a long repeated answer modulo whitespace, checks the `[Category | Keyword]` prefix on every chunk, and retains `۲۴ ساعت` in its sentence-sized sample. Tested body segments are ≤400 characters. There is no before/after retrieval measurement attributable to the candidate chunker because the sealed index was not rebuilt.

#### 5. Root Cause Analysis

Unconditional character offsets ignore sentence/word boundaries and do not carry metadata into text. Increasing the slice length alone would not address either mechanism.

#### 6. Architectural Modification

- **Module:** `rubbish_rag/pipeline.py`, `chunk()`.
- **Mechanism:** Split on sentence delimiters/newlines, fall back to word boundaries for long sentences, pack adjacent segments up to 400 body characters, and prefix category/keyword while preserving document/chunk IDs, question, category and body.

#### 7. Ablation Experiment

No retrieval-quality ablation of chunking was performed. Offline coverage and metadata assertions validate its local behavior only. The final visible score cannot be credited to changes in sealed indexing.

#### 8. Production Trade-offs

- **Latency / Memory Overhead:** Local chunk construction was not separately benchmarked.
- **Edge Cases Considered:** Long answers and whitespace reconstruction. Header length is metadata-dependent; a single word longer than 400 characters can exceed the body bound. Number/unit pairs are preserved when their source sentence fits, not guaranteed across every long-sentence fallback boundary.

---

### Investigation 4: Verified Evidence, Slot-Aware Reranking & Deduplication (E/F)

#### 1. Observed Symptom

Unknown synthetic records and repeated document chunks appeared in remote evidence. Legacy trace line 1 ranks document 9004 above legitimate C1 record 106; legacy line 11 repeats document 47. Those legacy entries lack client IDs, so their line references are retained rather than invented.

#### 2. Falsifiable Hypothesis

Checking both document membership and text against the public reference should reject unknown IDs and forged content attached to legitimate IDs. Deduplicating after ranking should prevent multiple chunks of one document from occupying the citation window.

#### 3. Diagnostic Probes & Exact Trace IDs

The real CLI C1 smoke test (`python rubbish.py bepar "رتبه C1 یعنی چی؟" --topk 3`, executed by pytest) is trace `client_8857aacb20904c3bb84961ea0ca00432`. A corporate-conviction query is trace `client_5376e64ed5b74676b3526b595a33108f`. Offline forgery/dedup checks are in `tests/test_engineering.py`.

#### 4. Quantitative Evidence & Sub-score Analysis

| Trace ID | Query Variant | Top-1 Doc ID | BM25 Score | Dense Score | Fused Score | Outcome |
|---|---|---:|---:|---:|---:|---|
| `client_8857aacb20904c3bb84961ea0ca00432` | C1; first merged candidate | 9004 | 8.9152 | 0.2500 | 1.0000 | Rejected; final first citation 106 |
| `client_5376e64ed5b74676b3526b595a33108f` | Corporate conviction; first merged candidate | 9106 | 21.0357 | 0.2491 | 0.9371 | Rejected; final first citation 56 |

These rows describe the stored candidate order, not a claim that RRF changes raw scores. In the latest visible pools, unknown remote records are 5% of hits and 0% after ranking. Mean remote top-five redundancy is 3%, versus 0% after ranking. Final top-three citation poison rate is 0/40. See `traces/replay_analysis.json`, `traces/requirements_audit.json` and `plots/repetition_vs_bm25.png`.

#### 5. Root Cause Analysis

Raw lexical overlap can reward keyword spam. Corpus membership alone cannot reject a forged excerpt under a valid ID, and chunk-level ranking permits one document to occupy several positions. An initial verifier also incorrectly compared BriefAnswer and Answer separately, rejecting legitimate chunks spanning their join; its failed run scored 10% and remains in the logs.

#### 6. Architectural Modification

- **Module:** `rubbish_rag/pipeline.py`, `grounded()`, `rerank()`, `resolve()`.
- **Mechanism:** Match normalized returned content against concatenated BriefAnswer + Answer; no magic synthetic-ID cutoff. Filter explicit rank/bank/persona conflicts, score IDF-weighted relevance and RRF, deduplicate on document ID, and return a verified reference brief with at most three distinct citations.

#### 7. Ablation Experiment

Synthetic regression combines an unknown ID, forged known-ID text, a conflicting rank and duplicate valid evidence; only the valid C1 document survives. The before/after pool statistics quantify filtering and redundancy, but an isolated deduplication Hit@3 gain was not measured. The RRF/local-search ablations are in Investigation 1.

#### 8. Production Trade-offs

- **Latency / Memory Overhead:** Verification uses the cached public corpus; separate costs were not measured.
- **Edge Cases Considered:** Valid-ID forgery, duplicate chunks and explicit slot conflict. Exact normalized-content verification can reject valid paraphrases. Resolver expansion to the complete public reference goes beyond the literal retrieved fragment; valid citations alone do not prove relevance or that every cited record supports every clause.

---

### Investigation 5: Ambiguity Detection & Missing-Information Gating

#### 1. Observed Symptom

A separate 40-query manual evaluation detected only 8/20 ambiguous inputs (40%), below the general rubric target. For example, asking whether Bank Mellat would lend produced an answer about banks being unable to change credit scores. A question about an unspecified report item produced an answer about penalties.

#### 2. Falsifiable Hypothesis

Distinguishing personal eligibility/classification requests from general facts, checking missing slots, and detecting unanchored references should increase ambiguity recall without broadly blocking complete questions.

#### 3. Diagnostic Probes & Exact Trace IDs

Original live command: `python tools/evaluate_clarification.py --live`. Failure examples: a07 `client_ddfac190be0e454ebf8dc09bd4ad7856` and a16 `client_e9290d0d3835482cb592dca7c43e56d3`.

After fixes: `python tools/replay_clarification_regression.py` replays original replies without API calls. Fresh live command: `python tools/evaluate_clarification.py --live --dataset evaluation/clarification_fresh_v2.json --output-prefix clarification_fresh_v2`.

Labels, rationales and frozen SHA256 files are in `evaluation/`. Full outputs and runtime hashes are in `traces/clarification_evaluation.json` and `traces/clarification_fresh_v2_evaluation.json`; case trace IDs also appear in their corresponding `_cases.jsonl` files and the main trace.

#### 4. Quantitative Evidence & Sub-score Analysis

| Evaluation | Ambiguous cases clarified | False clarifications / specific cases | Overall routing |
|---|---:|---:|---:|
| Original manual v1 | 8/20 (40%) | 0/20 | 28/40 (70%) |
| Fixed v1, cached development regression | 20/20 (100%) | 0/20 | 40/40 (100%) |
| Fixed fresh v2, real execution | 18/20 (90%) | 1/20 | 37/40 (92.5%) |

Fresh precision/F1: 94.7%/0.923. All clarifications have a question and at least two nonempty options; structural validity does not measure full conversational quality. The descriptive 95% Wilson recall interval is [69.9%, 97.2%]. The original fresh run had three errors (all fixed in the subsequent exposed-data regression; original results are preserved):

| Case | Query | Expected | Actual | Trace ID |
|---|---|---|---|---|
| fresh_08 | امکان دریافت وام برای من هست؟ | clarify | answer | `client_f1e73baad46b4906a82007d21ce42961` |
| fresh_14 | در سال ۱۴۰۵ رتبه من بد شده؛ سطح ریسک من چیست؟ | clarify | answer | `client_8a330d3af9884ae6a68d23c530697c2d` |
| fresh_25 | بدهی چیست و چه نوع بدهی‌هایی در اعتبارسنجی بررسی می‌شوند؟ | answer | clarify | `client_1acb849c36c6446499c8d9c4d2fde08b` |

#### 5. Root Cause Analysis

Narrow phrase matching missed eligibility variants and contextless references. Any numeric value could stand in for a credit score. Installment wording incorrectly implied a loan debt. A verified answer document still supplied no evidence about what an unspecified pronoun referred to.

#### 6. Architectural Modification

- **Module:** `rubbish_rag/clarification.py`, called by the shared execution path and resolver; `tests/test_clarification.py`.
- **Mechanism:** Check debt source separately from installment wording; request only missing bank/rank information; distinguish personal rank interpretation from general repayment effects; require a referent for deictic phrases unless an explicit topic is named. Tests cover nearby complete questions and avoid asking for an already supplied grade. The original labels/results remain unchanged.

#### 7. Ablation Experiment

The old dataset is now development material: its paired cached result changes from 8/20 to 20/20 ambiguity detections with no retrieval calls. This is regression evidence, not an independent test. A separate frozen v2 evaluated the pre-Staff code once. Subsequent Staff changes mean that report is historical. The current pre-routing regression reaches 40/40 after the general guards were revised (`traces/staff_clarification_regression.json`); this rerun is not held-out evidence. The two datasets' 40% and 90% figures are not a paired causal estimate. No guard-by-guard ablation was performed.

#### 8. Production Trade-offs

- **Latency / Retrieval:** Original manual run used 32 calls; fresh run used 21 because 19 inputs clarified before retrieval. The sample mix differs, so the difference is not an isolated savings estimate. Last full test suite used 21 calls.
- **Evaluation Limits:** Same assistant authored labels after inspecting code; exact fresh queries were unseen but domain patterns were familiar. Routing success does not establish factual accuracy, independent annotation, hidden performance or multi-turn completion.
- **Historical Errors and Current Fixes:** The original run missed `امکان دریافت وام`, mistook year 1405 for a score, and over-clarified debt definitions. General rules now distinguish score expressions from other numbers, personal eligibility requests from factual questions, and definitions from personal debt. All 40 exposed cases route correctly; historical failures remain in their original reports. This is development regression, not a new 100% held-out claim.

---

## Deliberate Non-Interventions

1. **No external LLM or embedding service.** The solution uses the local public corpus and CPU pipeline. No model-size, latency or memory comparison was measured, so hypothetical numerical penalties are not claimed.
2. **No sealed server inspection or index reconstruction.** Only public APIs and permitted reference/evaluation material were used; chunking improvements do not imply a changed private index.
3. **No hardcoded answers or benchmark IDs in production retrieval.** The local index covers all public records; benchmark labels are used only by evaluation tools/tests.
4. **No claimed RRF accuracy gain or independent dedup Hit@3 gain.** The measured negative/neutral results are retained. Negation tokens are preserved, but full polarity matching is not implemented. Staff probability calibration is measured below, including its negative held-out result.
5. **No claim of fresh test accuracy after Staff error analysis.** The original 15-query Staff test exposed five routing failures. General guards were then revised; the same split is now regression data. Its perfect rerun does not establish generalization. The three pre-Staff clarification errors were subsequently fixed; current exposed-set routing is 40/40. The new 24-query calibration audit is preserved without subsequent rule tuning.

---

## Bonus Track: Adaptive System-1 Decision Judge *(Optional)*

The CPU heuristic judge is integrated into the shared CLI/orchestrator execution path. Implementation and reproducible evaluation are submitted; strong out-of-sample calibration is **not** established.

- **Judge Architecture:** `judge.py` produces typed pre/post decisions. Pre-routing selects no retrieval, clarification, or hybrid retrieval with planned k ∈ {0, 2, 3, 5, 8}; the caller's `topk` is a context cap. The retriever still obtains at least 10 remote candidates plus public-corpus supplementation, so shorter selected context is not equivalent to cheaper remote retrieval. Runtime reads only fitted probability tables for six observable route/domain groups, never evaluation labels or query examples. A Dirichlet(1,1,1) prior smooths each group; unseen groups fall back to a uniform distribution. This adds no model dependency. The optional model adapter delegates to the CPU policy and loads no model.
- **Routing Accuracy on `data/judge_dataset.json`:** First held-out test: **10/15 = 66.7%**, below the provided static baseline's **11/15 = 73.3%** (`traces/staff_test.json`). After examining errors and revising general rules, current train/dev/test regression is **20/20, 15/15, 15/15** (`traces/staff_regression.json`). Test is now exposed; 100% is not a new held-out result. Errors involved help requests, missing installment/company details, causal income questions, and guaranteed-loan spam. The train error also revealed that “report shows a loan” was mistaken for loan eligibility.
- **Clarification F1:** First held-out **0.333** (TP=1, FP=2, FN=2); exposed-test regression **1.000** (TP=3, FP=0, FN=0). These are distinct from the manual 40-query clarification study above.
- **Token / Latency Savings:** On identical captured remote pools, fixed k=5, fixed k=10 and adaptive selection all score **20/20**. Selected excerpts contain **3214, 6975 and 919** word tokens respectively: **71.4% reduction versus fixed-5**. Counts use `normalize_fa.tokens`, not an LLM tokenizer; no generation model is called. They exclude full-reference expansion by the resolver, so no billed-generation savings are claimed. Fixed-10 averages 9.5 available documents. Source trace IDs and all three policies are in `traces/staff_replay_trace.jsonl`; totals in `traces/staff_context.json`. The initial Staff implementation increased live mean from **34.8 to 72.7 ms**. A subsequent computation-reuse optimization reduced the live mean to **28.0 ms**, below the 100 ms rubric bound. The separate paired cached-pool experiment is detailed in Investigation 2; host load remains a limitation of the live comparison. First-run pre-judge mean was **1.14 ms**. The latest full suite after confidence revisions measures **28.5 ms** live mean, with unchanged 20/20 visible accuracy.
- **Probability Calibration:**
  - **Historical experiment retained:** Original temperature 0.95 was fitted on 20 train rows. On the original 15-query held-out split it worsened retrieval Brier from 0.3042 to 0.3084 and ECE from 0.2813 to 0.2895. Original selected-decision Brier/ECE were 0.3022/0.2828. The old scalar artifact is preserved in `evaluation/staff_temperature_v1.json`, with results in `traces/staff_test.json`.
  - **Revised method:** Replace constant chosen-action confidence with smoothed empirical class frequencies grouped by predicted route and presence of a credit-domain anchor. Fit only the **40 calibration rows**, then evaluate once on the **24 separate audit rows** in `evaluation/staff_calibration_v2.json`; SHA256 freezes the complete labelled file before execution. The same assistant authored both splits knowing prior failures, so this is not independent validation. Classes are artificially balanced and domain coverage is narrow. Runtime stores only aggregate counts/probabilities; no test text or labels are loaded.
  - **Same-query comparison on the 24 audit rows:** raw and fitted policies both route **17/24 = 70.8%** correctly. Calibration measures confidence, not a change of route accuracy. Decision Brier **0.2647 → 0.0326**; decision ECE **0.2411 → 0.1714**. Retrieval Brier **0.2640 → 0.0208**; retrieval ECE **0.2695 → 0.1357**. Lower is better; ECE uses 10 fixed-width bins. These values must not be directly compared with the earlier 15-query experiment as a paired improvement.
  - **Selective risk:** At confidence ≥0.75, coverage is **16/24 = 66.7%**, with **0/16** observed routing errors. All seven audit errors have selected-action confidence 0.20. This is a small-sample observation, not a guarantee. Five missed social phrases and two vague requests still route to retrieval; low confidence expands their context budget rather than correcting their route. All seven stay in scoring. The thresholds' risk/coverage curve is diagnostic; runtime does not discard every low-confidence decision.
  - **Trade-offs:** Group estimates are coarse and data-limited. Small correct groups become underconfident; aggregate ECE remains nonzero. Missing confidence groups use uniform probabilities; retrieval confidence below 0.75 expands the context cap. The two-round bound and public-text verification still apply. These are routing probabilities only: post-retrieval support scores remain heuristic and are not calibrated semantic truth. Independent labels, broader linguistic coverage and semantic sufficiency evaluation are still needed.
  - **Evidence:** `traces/route_calibration_fit.json`, `traces/route_calibration_audit.json`, `plots/route_calibration.png`; historical negative results remain in `plots/staff_evaluation.png`. No rule or parameter changes followed the audit result. The original public 50-query routing regression still passes; it is exposed data.
- **Key Insight:** The post-judge verifies evidence against the public corpus, removes forged/unknown records, detects explicit rank conflicts and requires both rank facets for comparisons. It accumulates verified documents and stops after at least two when support passes its heuristic threshold. Weak/missing coverage triggers one broader retrieval; execution stops after at most **two calls**, or on no novel evidence, and abstains without citations. Abstention is represented as `status=clarify, decision=abstain` for compatibility with the existing response schema. Heuristic evidence probabilities are not calibrated semantic-truth probabilities. Nine controlled evidence fixtures cover valid, empty, forged, unknown, conflicting, partial/complete comparison, round limit and spam cases: **9/9** (`traces/staff_evidence.json`). These are controlled fixtures, not independent sufficiency labels. Tests additionally exercise second-round recovery and bounded failure. Multi-turn slot resolution and broad semantic contradiction detection are not implemented.

---

## Reproduction Checklist

- [x] `python -m pytest tests/ -q` passes: 103 tests, including both live CLI smoke tests. It consumes 21 retrieval calls.
- [x] `python rubbish.py bench` completed within the full-suite run in under 3 minutes on CPU; saved visible result is 20/20.
- [x] `python rubbish.py bekesh` generates the diagnostic figures. Then run `python tools/audit_pdf_requirements.py` to retain the final two-panel length/fusion ablation. The template's `plot` command is not provided by this CLI.
- [x] `python rubbish.py bastesh` packages code, evidence, plots and per-file SHA256 hashes without retrieval.

Use the original supplied repository at base commit `426333db4730a94afb263db0cfe02fadc63932b9` with its corpus and sealed runtime. The ZIP is an overlay, not a standalone distribution. Create a Python 3.11 environment and install the lock file:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Quota-free checks and evidence regeneration:

```bash
python -m pytest tests/ -q -k 'not cli_bepar and not cli_bench'
python tools/validate_cached_runtime.py
python tools/evaluate_staff.py --regression
python tools/staff_context_evidence.py
python tools/plot_staff.py
python tools/plot_route_calibration.py
python tools/replay_clarification_regression.py
python tools/replay_evidence.py
python rubbish.py bekesh
python tools/audit_pdf_requirements.py
python rubbish.py compare traces/metrics_baseline.json traces/metrics.json
```

Replay timing is not real retrieval timing. `tools/benchmark_latency.py --label before` and `--label after` preserve their existing output files; archive the delivered reports before a new experiment. The historical before measurement requires the pre-optimization code, not the current cached implementation. Replay tools overwrite their own generated artifacts; preserve delivered evidence if exact historical trace IDs are needed. `python tools/calibrate_routes.py --fit` followed by `--audit` reproduces the separate calibration experiment in a clean experiment directory; delivered reports are protected against overwriting/refitting. The Staff `--fit`/`--test` commands preserve the original experiment and refuse reuse once its report exists; use `--regression` for current code. Real manual evaluation scripts refuse to overwrite existing results to prevent accidental repeat calls. Preserve the quota ledger and key; total recorded usage is 262/1000. Staff implementation used 42 calls through two full-suite runs; latency optimization used 21 and the later routing/calibration revision used another 21 for a final full-suite run. The 40/24 calibration experiment itself used zero retrieval calls. Routing, evidence fixtures, profiling and paired context/latency comparisons used zero calls.

Package and verify:

```bash
python rubbish.py bastesh
(cd submissions && shasum -a 256 -c submission.zip.sha256)
```
