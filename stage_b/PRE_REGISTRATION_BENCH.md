# PRE-REGISTRATION — Benchmark Extension BENCH (post-seal)

**Version:** v1.3, 2026-07-12 — **FROZEN (MK sign-off 2026-07-12) + Amendment A1.**
v1.3 = v1.2 + the single disclosed Amendment A1 (§9): subword-prefix commitment rule,
adopted after Phase-3 smokes and before any strict cell. Spec version `bench/1.3`.
**v1.0 → v1.1:** same day, after a Codex CLI adversarial review of the v1.0 draft
(verdict 4/10 FIX; filed at `wiki/paper/cc-bench-prereg-review.md`). All of that review's
TIGHTENED items are incorporated; §10 records the honest per-fix status.
**v1.1 → v1.2:** same day, after a second static Codex adversarial pass (6/10 FIX).
v1.2 freezes the replication-task fusion aliases, the exact A2 estimator, byte-level
dataset inputs and sampling algorithm, strict-control disposition, resume provenance,
and comparability language; it also removes an impossible claim that the seal recorded a
complete runtime-version tuple. §10.3 records the fixes.
**Relationship to the seal:** labeled post-seal extension. The `prereg-seal-20260612`
verdict (geometric 18/20 PASS / strict full-panel 18/20 FAIL), `stage_b/profiles/`, and the
"Ten Models × two tasks" paper headline are frozen and are never re-opened by anything in
this document. New results land in `stage_b/profiles_bench/` and, in the paper, in a
clearly-labeled extension section.
**Supersedes as execution spec:** `wiki/paper/cc-benchmark-proposal-v2.md` §6/§9 (this
document freezes the decisions that proposal left open, at the narrowed scope chosen by MK
2026-07-11: HaluEval + ANLI R2 + n=1000 replication; FEVER / TruthfulQA / SimpleQA / P3 are
NOT part of this extension and remain future work).
**Adversarial-review lineage:** `cc-benchmark-review.md` (v1, FIX) and
`cc-benchmark-review-v2.md` (v2, FIX). The v2 review's 8 required fixes are resolved here
(§10 audit table).

---

## 1. Cohort, seed, machinery (all frozen)

- **Cohort:** the sealed 10-model MLX cohort, verbatim from `run_seal.py::COHORT`
  (Llama-3.2-3B, Llama-3.1-8B, Mistral-7B-v0.3, Mistral-Nemo-2407, Phi-3.5-mini,
  Phi-4-mini, Qwen2.5-7B, Qwen3-1.7B, Qwen3-8B, gemma-3-4b-it — all `mlx-community` 4-bit).
  No additions, no drops. A model that fails a task's smoke or length-intersection gate is
  reported as a failed/abstained cell **inside** the denominator, never silently removed.
- **Seed:** `20260711` (fresh; not in `PILOT_SEEDS`; never used by seal or EXT).
- **Selector:** the sealed 29-cell panel and in-bag selection / OOB evaluation rule,
  `nboot=2000`. The row-resampling implementation remains the seal-comparison path; the
  registered cluster endpoints deliberately change only the exchangeable unit from row to
  stem (§5). They are not described as selector-byte-identical to the seal.
- **Loci:** unchanged and reported per family — ACE at t=0/prefix-last; PRI/RPV/Confidence
  at gen_step=1; fusion averages both.
- **Extraction:** `max_new_tokens=1`; strict cells `max_dropped=0`.
- **Frozen fusion task aliases and new-task fallback (closes the exact-key trap):**
  `append_fusion_columns` currently looks up orientations by the exact key
  `slug|benchmark`. The extension MUST pass the following canonical key to that lookup:

  | BENCH task key | fusion-sign key |
  |---|---|
  | `anli_r1_rep` | `anli_r1` |
  | `triviaqa_paired_rep` | `triviaqa_paired` |
  | `halueval_qa`, `anli_r2`, `halueval_dialogue`, `halueval_summarization` | no alias; use the frozen modal fallbacks |

  Thus the replication tasks reuse the seal's exact per-(model, task) component
  orientations rather than silently falling back. New tasks use the existing
  `fusion/1.0` modal signs: ACE `-1`, `null_ratio_post_rank1` `+1`,
  `fisher_eff_rank` `+1` (and surprise `+1` for the non-geometric fusion). The Phase-0
  `fusion_signs.json` sha256 is
  `92b5468bd241b517dd2d5cf70ad28556157424deb54b5f85f88af0305ff35372` and MUST NOT
  change in Phase 1.
- **Comparability labeling is split rather than overloaded:**
  - **Feature/extraction code-and-weight matched** iff:

    1. `module_hashes()` **T0-subset** equals the seal's recorded values. The T0 subset is
       enumerated (no shorthand): `pri_calibrator.py`, `comprehensive_run.py`,
       `pri_runtime.py`, `pri_v2_io_plugins.py`, `pri_v2_mlx_pipeline.py`,
       `model_adapters.py`, `test_shadow_ambiguity.py`,
       `diagnose_inter_head_disagreement.py` — i.e., every `module_hashes()` entry except
       `confluence_calibrator.py` and `fusion_signs.json`;
    2. per-model `model_snapshot_sha.resolved_revision` equals the seal's; and
    3. a Phase-1 parity sentinel re-scores `sample_idx=0` from the frozen 20260612 ANLI
       file for each cohort model through the unchanged T0 extraction path and reproduces
       every finite pre-fusion value in that model's committed
       `stage_b/profiles/anli_r1/<slug>.matrix.npz` byte-for-byte. A failed sentinel does
       not get explained away by AUROC closeness: that model's BENCH cells receive a
       feature-version-delta caveat.
  - **Extension-inference matched** iff `confluence_calibrator.py` and every harness hash
    equal the Phase-1 extension baseline. Cluster endpoints are internally comparable
    within BENCH but are never called inference-byte-identical to the sealed row endpoint.
  - **Runtime provenance limitation:** static inspection found no complete seal-time tuple
    for Python, `mlx`, `mlx-lm`, and NumPy in the committed seal profiles/summary. The only
    contemporaneous repo record is `mlx-lm 0.29.1`; the other exact seal versions are not
    reconstructible. Phase 1 therefore records the complete BENCH runtime tuple but MUST
    NOT claim version equality to an unrecorded seal tuple. The parity sentinel above is
    the operational byte-comparability check, and the paper discloses the missing historical
    runtime tuple.
- **Phase-1 compute baseline:** before any Phase-1 edit,
  `confluence_calibrator.py` sha256 is
  `f55279162eb15a3806a0698e1540e898a13f70dbdc9e99820a969bb2bf563bbb`.
  The Phase-1 diff against that baseline is limited to the stem persistence, cluster
  resampling, endpoint stamping, and alias plumbing enumerated in §7. Any other compute
  change requires a prior Amendments entry. The post-edit extension hash is then frozen in
  the manifest before Phase 2.
- **Harness hash manifest (fixing the v2 review's hash-coverage trap):** the launcher,
  gate, builders, and analysis scripts are NOT in `module_hashes()`. At Phase-1 end we
  record into `stage_b/profiles_bench/EXTENSION_MANIFEST.json` the sha256 of **every**
  harness file the extension executes — the frozen file set is exactly:
  `run_bench.py` (the extension launcher — frozen name, no siblings),
  `generate_bench_data.py` (the extension builders — frozen name, no siblings),
  `check_fresh_data.py`, `analyze_universality.py`, `confluence_calibrator.py`,
  `fusion_signs.json`, and THIS document — plus the complete BENCH runtime version values
  and the pre/post Phase-1 diff summary. If Phase 1 creates or executes any harness file
  outside this set, it must be added to the manifest via an Amendments entry before
  Phase 2. Only with that
  manifest may the extension be described as "separately hashed."

## 2. Registered task families and denominators (frozen)

| Family | Task key | Status | n (rows) | Cells |
|---|---|---|---|---|
| A — breadth | `halueval_qa` | **CONFIRMATORY** | 1000 (500 stems × 2) | 10 |
| B — replication | `anli_r1_rep` | **CONFIRMATORY** | 1000 (500/500) | 10 |
| B — replication | `triviaqa_paired_rep` | **CONFIRMATORY** | 1000 (500 q × 2) | 10 |
| C — exploratory | `anli_r2` | exploratory (robustness) | 1000 (500/500) | 10 |
| C — exploratory | `halueval_dialogue` | exploratory (length-gated) | target 1000 | ≤10 |
| C — exploratory | `halueval_summarization` | exploratory (length-gated) | target 1000 | ≤10 |

- The confirmatory denominator is exactly **30 cells** (families A+B). No fallback
  substitution: if a family-A/B cell cannot run, it is a reported failure, not a swap.
- ANLI R2 is registered as **robustness only** — same task, harder round; it does NOT
  count toward any breadth claim, and no paper sentence may cite it as a "new benchmark."
- Family-C cells carry no bars; they may narrow but never upgrade any headline claim.

## 3. Data sources — verified facts and frozen build specs

All supply counts verified live 2026-07-11 from the local HF cache / GitHub API.

### 3.1 HaluEval (families A, C)
- **Source & license (resolved — was the v2 blocker):** build from the GitHub source
  `RUCAIBox/HaluEval`, pinned commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`
  (HEAD since 2024-02-12), license **MIT** (GitHub API spdx `MIT`, verified 2026-07-11).
  The apache-2.0 HF mirror (`pminervini/HaluEval`) is NOT used. Downloaded raw file
  sha256 recorded in each data manifest.
- Subsets/fields (verified): `qa_data.json` 10k (`knowledge/question/right_answer/
  hallucinated_answer`); `dialogue_data.json` 10k (`knowledge/dialogue_history/
  right_response/hallucinated_response`); `summarization_data.json` 10k
  (`document/right_summary/hallucinated_summary`).
- **Emission:** paired Framing-A — each sampled source item emits exactly two rows
  (right→label 0, hallucinated→label 1), exact 0.50 balance, ≤2 rows/stem by construction.
  Stem id = source-item index; recorded per row for the cluster bootstrap.
- **Frozen prompt templates** (verbatim; `{}` are field substitutions; prompt ends at the
  one-token cue `Answer:`; label 0 → YES, 1 → NO):
  - `halueval_qa`:
    ```
    You are given reference knowledge, a question, and a candidate answer. Decide
    whether the candidate answer is faithful to the reference knowledge. Answer YES
    if the candidate answer is supported by the knowledge, NO if it contains
    hallucinated or fabricated content.

    Knowledge: {knowledge}
    Question: {question}
    Candidate answer: {right_answer | hallucinated_answer}

    Is the candidate answer faithful to the knowledge? Answer:
    ```
  - `halueval_dialogue` (verbatim; candidate = `right_response` for label 0,
    `hallucinated_response` for label 1):
    ```
    You are given reference knowledge, a dialogue history, and a candidate response.
    Decide whether the candidate response is faithful to the reference knowledge.
    Answer YES if the candidate response is supported by the knowledge, NO if it
    contains hallucinated or fabricated content.

    Knowledge: {knowledge}
    Dialogue history: {dialogue_history}
    Candidate response: {candidate}

    Is the candidate response faithful to the knowledge? Answer:
    ```
  - `halueval_summarization` (verbatim; candidate = `right_summary` for label 0,
    `hallucinated_summary` for label 1):
    ```
    You are given a document and a candidate summary. Decide whether the candidate
    summary is faithful to the document. Answer YES if the candidate summary is
    supported by the document, NO if it contains hallucinated or fabricated content.

    Document: {document}
    Candidate summary: {candidate}

    Is the candidate summary faithful to the document? Answer:
    ```
  These template strings are byte-frozen at sign-off; the builder must emit them
  character-for-character (single `\n` line breaks as shown, one blank line between
  instruction and fields and between fields and the final question).
- **Contamination note:** HaluEval hallucinations are ChatGPT-generated and filtered;
  the paper must state that this family measures *recognition of LLM-written
  hallucinations* (RAGTruth remains the superior, protocol-incompatible target — the
  v2 proposal's exclusion language is incorporated by reference).

### 3.2 ANLI (families B, C) — frozen local `facebook/anli` artifact

The upstream revision was not preserved in the local datasets cache, so v1.2 does not
pretend that an HF revision can be selected later without researcher degrees of freedom.
The registered input is the already-cached `plain_text/0.0.0` artifact with datasets
fingerprint `8e4813d81f46d313dac7892e1c28076917cfcdf9`; the exact Arrow inputs are:

| Split file | sha256 |
|---|---|
| `anli-train_r1.arrow` | `b32df9e1ee446fa9d34c6996f788dbce7fbbe9ec682d0672cb340837904ee40a` |
| `anli-dev_r2.arrow` | `6ff4c3bac8b0ae917cf89dd73cf9966107d5888232d8e423ecde8da8555486fd` |
| `anli-test_r2.arrow` | `d63398b51f5c29f92b251b1f5b54c9a1a5c9772a1b2a7ed96a047cee0221e655` |

The builder must read these bytes (path configurable; fingerprint + sha256 binding), not
redownload a mutable default revision. A missing/mismatched file triggers §8.1 source abort.

Verified label counts (0=entail, 1=neutral, 2=contradiction):
`dev_r1` 1000 {334/333/333}; `train_r1` 16,946 {5371/7052/4523};
`dev_r2` 1000 {334/333/333}; `test_r2` 1000 {334/333/333}; `train_r2` 45,460.
- **`anli_r1_rep` draws from `train_r1`** (registered convention change, with reason):
  dev_r1 has ≤667 usable rows after the neutral drop, of which ~400 are consumed by the
  20260526 sealed file and the 20260612 seal file — n=1000 balanced is arithmetically
  impossible from dev_r1. train_r1 supplies 5371/4523 per class. **Caveat frozen into the
  claim:** train-split examples are noisier/less-verified than dev; this cell is a
  "re-test of the sealed construct at 5× n on the train distribution," and the paper must
  say so — it may NOT be described as a same-distribution replication (see §5 B-endpoints
  for the language rule).
  **Exclusion set (enumerated):** the union of the 20260526 sealed files
  (`t0-morphology-furnace/experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl`,
  `triviaqa_paired_seed20260526_n100.jsonl`) and the 20260612 seal files
  (`stage_b/data/anli_R1_seed20260612_n200.jsonl`,
  `stage_b/data/triviaqa_paired_seed20260612_n200.jsonl`). Zero overlap against this
  union is required by normalized-prompt hash AND, where the row carries one, by
  stem/question id. Both the builder and the gate check it (the current builder knows
  only the 20260526 reference and the gate accepts exactly one sealed file — extending
  both to the enumerated union is Phase-1 work, §7).
- **`anli_r2` draws from `dev_r2` ∪ `test_r2` pooled** (registered convention extension):
  668 entail / 666 contradiction usable → 500/500 sampled. Test-split gold labels are
  publicly released by the ANLI authors (verified present and balanced above). No prior
  MLX-line run has touched R2; the enumerated exclusion union above is still asserted
  (expected vacuous, checked anyway). **Allocation frozen:** stratified equally by split —
  exactly 250 entail + 250 contradiction drawn from `dev_r2` and the same from `test_r2`
  (supply verified: ≥334/333 per class per split). Each row's provenance records its
  source split, and split-specific descriptive AUROCs are reported alongside the pooled
  cell so pooling cannot conceal split heterogeneity.
- Template, neutral-drop, balancing, and shuffling: byte-identical logic to the sealed
  `build_anli` (same instruction text, same label convention 0=YES/entail, 1=NO/contra).

### 3.3 TriviaQA (family B) — frozen local `trivia_qa` `rc.wikipedia` artifact

The registered input is the already-cached `rc.wikipedia/0.0.0` artifact with datasets
fingerprint `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`, specifically
`trivia_qa-validation.arrow` sha256
`8e95a5f9ce34a037cc3dd0d2e544961a20470cb6c415f6ab48a1e115ed5a7a90`.
As with ANLI, the path is configurable but the fingerprint + bytes are frozen and a
mismatch triggers §8.1 source abort.

Verified: 7,993 validation questions. `triviaqa_paired_rep` samples 500 questions,
disjoint from the enumerated exclusion union of §3.2 (both the 20260526 and 20260612
TriviaQA files) by normalized-prompt hash AND `question_id`, and emits paired
correct/wrong rows with the sealed builder's template and cross-sampled wrong answers →
n=1000, exact 0.50, stem = question_id.

### 3.4 Deterministic sampling and row order (frozen)

A seed without an algorithm is not a reproducible draw. Every task uses the following
procedure after the enumerated exclusion-union and 2048-token common-intersection predicates
have been defined. Source order always means original row order in the frozen artifact;
duplicate stable source identities are a hard failure.

1. **ANLI R1 replication (sealed-builder generalization):** initialize
   `numpy.random.RandomState(20260711)`, shuffle `range(len(train_r1))` once, scan that order,
   drop neutral/excluded/duplicate prompts, append the first 500 unique entailments and 500
   unique contradictions, stop when both quotas are full, concatenate entailments then
   contradictions, and call `rng.shuffle(rows)` once. This is the existing `build_anli`
   algorithm with only split, quota, and exclusion-union generalized.
2. **ANLI R2:** initialize a fresh `numpy.random.RandomState(20260711)`. In split order
   `dev_r2`, then `test_r2`, shuffle each split's full row-index list once using that same
   RNG; within each split scan/filter as above and take exactly 250 entailments + 250
   contradictions. Concatenate in order `(dev entail, dev contradiction, test entail,
   test contradiction)`, then call `rng.shuffle(rows)` once.
3. **HaluEval subsets:** initialize a fresh `numpy.random.RandomState(20260711)` per subset,
   form eligible pinned source-item indices in ascending order, call `rng.permutation` once,
   and take the first required number of stems (500 for confirmatory QA; the §4 attainable
   target for exploratory subsets). Emit label 0 then label 1 within each selected stem,
   then call `rng.shuffle(rows)` once.
4. **TriviaQA replication (sealed builder verbatim apart from quota/exclusion union):** use
   `random.Random(20260711)`; set `pool_size=min(len(validation), 5000)`; obtain the pool by
   `rng.sample(range(len(validation)), pool_size)`; copy and `rng.shuffle` the donor pool;
   scan the pool in order using the sealed alias-collision, unique-wrong-answer, donor-index,
   and prompt/question exclusion logic until 500 pairs exist; emit correct then wrong per
   pair; finally call `rng.shuffle(records)` once. No preliminary sorted-stem sampling or
   second RNG is permitted.
5. The manifest records candidate counts after each filter, RNG class + seed, the ordered
   shuffled indices/pool, selected source ids before final row shuffle, and final JSONL
   sha256. Any implementation that cannot produce these records fails the data gate.

## 4. Admissibility contract (applies to every cell)

The seven clauses of proposal-v2 §1.3 are incorporated verbatim: one-token commit cue;
per-model `io_plugins.get_prompt_strategy` wrapping; length limits; strict no-drop
(`max_dropped=0`, all `drops` reasons zero); first-token sanity; smoke gate; common
intersection set. Additions frozen here:

- **Length policy (families A, C):** a stem is admissible iff its FULL wrapped prompt
  (per-model chat template applied) tokenizes to **≤ 2048 tokens for every cohort model**
  scoring that task. Sampling draws only from the admissible intersection.
  **Cap set by measurement, not taste (census 2026-07-12):** tokenizer-only census over
  the pinned HaluEval files (2,000 stems sampled per subset, census seed 999 ≠ the
  registered seed; all 10 cohort tokenizers; frozen §3.1 templates; Mistral-Nemo
  chat-wrapped per `pri_v2_io_plugins.py`; no model forwards, no labels read):
  QA max-across-models p99 = 366 / max = 457 tokens → 100% admissible at 2048;
  dialogue p99 = 336 / max = 391 → 100%; summarization p50 = 1066 / p90 = 1888 /
  p99 = 2727 / max = 3429 → **92.3% admissible at 2048** (est. ~9,230 of 10,000 stems —
  the 500-stem confirmatory floor for QA and the 400-stem exploratory floor are both
  cleared by an order of magnitude). 3072 would admit ~100% of summarization but at
  ~2.25× the O(L²) attention-capture cost on the worst stems; 2048 is retained.
  - `halueval_qa` / `anli_*` / `triviaqa_*`: non-binding per the census (QA max 457);
    still verified at build, not assumed.
  - `halueval_dialogue` / `halueval_summarization`: if the admissible intersection
    < 500 stems, the cell set runs at the largest achievable paired n ≥ 400
    (recorded in the manifest); if < 400 stems (200 paired rows short of n=400), the
    subset is **abandoned and reported** (exploratory, so no denominator impact) —
    MK red-line 2026-07-12 raised the floor from 300 to 400 so a marginal cell
    (predicted CI-lo ≈ 0.54–0.55) is not run underpowered. Per-model intersections are
    never used — one common stem set per task.
- **Smoke gate (strengthened per review fix #6):** per (model, task), a stratified
  16-row smoke — 8 stems crossed over {short, long} × {label 0, label 1} (for
  summarization/dialogue, "long" = top length quartile of admissible stems) — must show:
  model loads; ACE 16/16 + readout 16/16 usable; produced `panel` byte-identical to the
  seal's panel (MISSING=[], EXTRA=[]); shuffled-label control passes; first committed
  token is a meaningful YES/NO under the actual prompt strategy for ≥ 15/16 rows.
  Smokes produce no registered metric and can never be `--resume`d into a cell.
- **Full-cell commitment audit (new in v1.1; closes the "smoke only checks 16 rows"
  hole):** in every STRICT cell, each row's first generated token id is decoded and must
  normalize to a canonical `YES`/`NO` commit under the frozen normalization rule: strip
  leading whitespace/newlines, case-fold, strip trailing punctuation; accept any
  **non-empty prefix of `yes` or `no`** (Amendment A1, §9 — subword tokenizers such as
  Mistral's split "YES" into `Y`+`ES`, so the first token of a genuine yes-commit can be
  a strict prefix; the prefix sets of the two forms are disjoint, so no ambiguity).
  Whitespace-only tokens (`"\n"`) and non-prefix tokens (`" To"`) fail. Any row failing this
  marks the cell **COMMITMENT-FAIL** (counted as a failed cell in its family denominator —
  same standing as a behavioral smoke fail). The decoded-commitment tally is stamped into
  the cell's provenance so the audit is checkable after the fact.
- **Strict shuffled-label control and disposition (frozen in v1.2):** every completed
  strict cell runs K=3 null permutations under each bootstrap unit used by a registered
  endpoint. Ungrouped tasks globally permute row labels. Paired/grouped tasks independently
  swap the two labels within each stem with probability 0.5, preserving the paired design,
  then run the stem-cluster selector. Each permutation uses a fresh
  `numpy.random.RandomState(20260711 + 90210 + k)`, `k∈{0,1,2}`.
  For each unit separately, if ≥2/3 permuted geometric CIs have `CI_lo > 0.50`, that
  endpoint receives terminal status **CONTROL-FAIL[unit]** and cannot count as deployable.
  A1 uses the cluster control; B1-procedural uses the row control; B1-valid uses the cluster
  control for TriviaQA and row control for ANLI. Thus a control failure invalidates exactly
  the endpoint whose null calibration failed, without silently contaminating the other
  comparison. No post-result adjudication can restore it inside this registration. A
  full-panel-only control flag is reported but does not affect the geometric endpoints;
  a geometric control flag does. Control method, seeds, and all three results are stamped.

## 5. Statistical gates (frozen — resolves review fixes #2 and #5)

- **Cluster bootstrap — exact frozen algorithm** (correcting v1.0's "outer and inner/OOB
  resampling" misdescription; the sealed selector performs ONE bootstrap draw per
  replicate, selects in-bag, and evaluates OOB — no inner bootstrap exists): *Each
  replicate samples stem ids with replacement; all rows belonging to each sampled stem
  enter the in-bag set with that stem's sampled multiplicity; the OOB set contains all
  rows of stems absent from the in-bag draw; cell selection and sign-locking occur
  in-bag; the selected cell is evaluated OOB. The deployability statistic is the 2.5th
  percentile of the OOB AUROC distribution over `nboot=2000` replicates, exactly as in
  the sealed row version but with stem replacing row as the exchangeable unit.* For
  ungrouped tasks (ANLI), stem = row and this reduces to the sealed procedure
  identically. Prerequisite (Phase 1, §7): `stem_id` must be persisted per row through
  the loaders, ACE/readout alignment, `merge_matrices`, and the `.npz`, with an equality
  check against the source JSONL's stem metadata.
- **Per-cell deployability gate:**
  - **Family A (`halueval_qa`) and family C grouped cells:** the confirmatory gate is the
    nested-OOB selected-cell OOB AUROC 95% CI lower bound > 0.50 under the **stem-cluster
    bootstrap** above. The row-bootstrap CI is reported alongside, descriptively — a
    row-pass/cluster-fail cell is NOT deployable.
  - **Family B:** both gates are computed and the endpoint is split (below):
    row-bootstrap CI_lo (the seal's procedure, for historical comparability) and the
    cluster CI_lo (valid inference; for `anli_r1_rep`, rows are ungrouped so the two
    coincide by construction).
- **Effective n:** every grouped cell reports groups (=stems) as effective n; the paper
  never writes "n=1000 independent samples" for paired tasks (effective n = 500).
- **Endpoints per family:**
  - **A1 (primary, task bar):** `halueval_qa` geometric-only deployable (cluster gate) on
    **≥ 8/10** models.
  - **A2 (primary, universal-floor probe, NAMED FIXED CELL):** the frozen cell
    **`fusion_rank_mean_geom`** holds LOMO holdout AUROC > 0.55 on **≥ 8/10** holdouts on
    `halueval_qa`. **Planned-denominator rule (closes the LOMO escape hatch —
    `analyze_universality.py` currently sets its denominator to however many matrices
    exist):** all A2 counts use the planned ten-model cohort; a missing, behavioral-fail,
    commitment-fail, or HaluEval cluster-control-fail holdout contributes zero passes and
    the denominator is always 10;
    if fewer than 3 usable training models remain for any holdout fit, A2 **aborts and
    is scored FAIL**. This endpoint is a NEW registered code path (Phase 1, §7.5) — the
    existing post-run `fixed_cell_max_survival` landscape is computed but **descriptive
    only** and may not be quoted as a confirmed endpoint.
    **Exact A2 estimator:** for each usable model, construct `fusion_rank_mean_geom` using
    the new-task modal component orientations frozen in §1, then rank-transform that one
    column *within that model* with `_rank01`. For each planned holdout, concatenate the
    ranked rows and labels of the other usable models (each usable model contributes its
    full 1000 rows, so model weights are equal); fit **only one global sign** on that pooled
    training vector using `_score_candidate`; do not select a cell or threshold. Apply that
    frozen sign to the held-out model's within-model ranks and compute point AUROC with
    `auroc_fixed`. Strict `AUROC > 0.55` passes; equality, non-finite AUROC, a failed/missing
    holdout, or a zero fitted sign fails that holdout. No CI is used for A2. A model whose
    HaluEval cluster endpoint is CONTROL-FAIL is unusable for both training and holdout
    evaluation. If fewer than three usable training models remain, the already-frozen A2
    abort=FAIL rule applies. The output stamps
    every training slug, fitted sign, pool AUROC, holdout AUROC, and failure reason.
  - **B1 — split into two endpoints (a single "replication" bar would conflate corrected
    and uncorrected estimands):**
    - **B1-procedural (primary):** geometric-only deployable on **≥ 17/20** replication
      cells under the seal's own row-bootstrap gate. This is procedure-identical to the
      sealed 18/20 and licenses ONLY the sentence "the sealed procedure reproduces at
      5× n on fresh draws" — it may NOT be called a statistically-independent
      replication (the TriviaQA row gate inherits the seal's known pairing inflation,
      and `anli_r1_rep` is a train-distribution re-test, not a same-distribution
      replication).
    - **B1-valid (primary):** geometric-only deployable on **≥ 17/20** under the valid
      gates — cluster bootstrap for `triviaqa_paired_rep`, row bootstrap for
      `anli_r1_rep` (ungrouped). Only B1-valid may support replication-strength language
      in the paper, and the only licensed sentence is: **"The geometric procedure remained
      deployable on at least 17/20 larger fresh-draw cells under pair-aware inference;
      this combines a train-distribution ANLI re-test with a clustered TriviaQA
      replication and is not a same-distribution replication of the original 20-cell
      cohort."** Any cell whose verdict differs between the two gates is reported as an
      inflation finding against the sealed gate. No shorter generic claim that "the seal
      replicated" is licensed by B1-valid.
  - **B2 (registered orphan probe):** `gemma-3-4b/anli_r1_rep` and
    `Llama-3.1-8B/anli_r1_rep` — deployable or not at n=1000 is reported either way;
    a pass **narrows** the orphan reading (small-n artifact component), a fail
    **strengthens** it. Neither outcome alters the sealed verdict.
- **Falsification / paper-language rule (frozen):**
  - A1 ∧ A2 → the paper may say the geometric floor **extends to HaluEval-QA**.
  - exactly one of A1/A2 → "partially extends," with the failing endpoint named.
  - ¬A1 ∧ ¬A2 → the extension section reports that the floor **does not extend** to
    HaluEval-QA; the title/abstract scope stays the original two tasks, verbatim.
  - Either B1 endpoint fails (< 17/20) → the paper reports the miss prominently in the
    extension section, naming which gate missed (the sealed 18/20 stands as sealed, but
    the extension must not be published without the miss stated).
  - "Universal," wherever used post-extension, ranges over: the 10 cohort models × the
    tasks where the frozen endpoints held. Nothing else.
- **Endpoint semantics vs the sealed launcher (deliberate redefinition, not
  inheritance):** the sealed `run_seal.py` forbids any endpoint PASS while any planned
  cell is incomplete. This extension REPLACES that rule: failed cells (behavioral-fail,
  commitment-fail, control-fail, abort) do not block endpoint scoring — they score as failures inside
  the fixed planned denominators (A1/A2 over 10; B1 over 20). An endpoint may therefore
  PASS with failed cells present, because the bars already price failures in. A cell that
  is merely *unrun* (interrupted, not yet executed) still blocks scoring — endpoints are
  scored only when every planned cell has a terminal status.
- **Multiplicity:** primary endpoints are exactly {A1, A2, B1-procedural, B1-valid};
  everything else (per-cell landscapes, E1/E2/E3 analyses, family C) is
  descriptive/exploratory.

## 6. Frozen predictions (EXT discipline; no prediction gates publication)

- `halueval_qa` A1: **LEAN PASS** (~8/10). Basis: 32B torch cell geom CI-lo 0.809, but the
  cohort is 1.7B–8B 4-bit; likely orphans: Qwen3-1.7B, gemma-3-4b.
- `halueval_qa` A2 (fusion LOMO floor): **GENUINELY OPEN** (~50%). Basis: QA was the one
  HaluEval subset where Fusion won at 32B; cross-family transfer at small scale unproven.
- `anli_r1_rep` B: **LEAN 8/10 deployable**, with both sealed orphans
  (gemma-3-4b, Llama-3.1-8B) predicted to REMAIN non-deployable at n=1000
  (we predict the orphans are model-capability holes, not small-n noise) — confidence LOW;
  train-split shift adds variance in both directions.
- `triviaqa_paired_rep` B: **STRONG PASS 10/10** (sealed cells were comfortable passes).
- B1 (both endpoints): **LEAN PASS ≥17/20 on each** (18/20 point prediction on
  B1-procedural, mirroring the seal; B1-valid predicted to match or drop at most one
  TriviaQA cell to the cluster gate).
- `anli_r2` (exploratory): ~7–8/10 deployable; same two models weakest.
- `halueval_dialogue` / `halueval_summarization` (exploratory): **LEAN widespread
  marginal/fail** — 32B CI-lo was 0.539/0.553, and these subsets shifted locus to
  readout/surprise at gen_step=1; at 7B-class scale we predict < 5/10 deployable, and we
  flag in advance that a readout-family (not ACE) winner pattern here would echo the
  Llama-70B locus dissociation.

## 7. Harness edits authorized by this pre-registration

All in `commit-confluence`; none touch the T0 modules; itemized so the extension-baseline
hash delta is fully accounted for:
1. **`generate_bench_data.py`** (new file — frozen name, in the §1 manifest):
   `build_halueval_{qa,dialogue,summarization}` (per-subset field branching, pinned
   GitHub raw download + sha256, byte-frozen templates of §3.1, paired emission, per-row
   `stem_id`, length-intersection filter), `build_anli_generic` (split parameter incl.
   `train_r1` and the split-stratified `dev_r2+test_r2` pooling, the **enumerated
   exclusion union** of §3.2), `build_triviaqa_rep` (500-question target, same exclusion
   union), all using the exact cached-artifact hashes and deterministic draw/order algorithm
   of §3.2–§3.4. The current builder knows only the 20260526 sealed reference — the union
   machinery is new work, acknowledged here.
2. `check_fresh_data.py` generic task path: schema/balance/intra-dup kept;
   single-sealed-reference requirement replaced by the enumerated exclusion union
   (asserted for replication tasks, vacuously checked for new benchmarks); added
   length-cap check, one-token-cue check, stem-cap (≤2) check, per-row `stem_id`
   presence check.
3. **`run_bench.py`** (new launcher — frozen name, in the §1 manifest): arbitrary
   (task → file) pairs, `profiles_bench/` output dir, per-family bars and the §5
   endpoint semantics from THIS document (no hardcoded 19/20; failed cells score as
   failures in fixed denominators), stratified smoke mode, full-cell commitment audit,
   paired-design shuffled controls, and fusion task aliases (§1/§4) stamped into provenance.
   Resume validation MUST compare, in addition to the seal fields: `bootstrap_unit`, exact
   ordered `stem_id` vector sha256, canonical fusion-sign task key/fallback, panel sha256,
   endpoint/spec version (`bench/1.2`), commitment status+tally, control status+seeds,
   extension-manifest sha256, and presence of every required row/cluster endpoint. A row
   profile cannot resume as a cluster profile merely because data/code/seed match; any
   mismatch is terminal ERROR and recomputation is required.
4. `confluence_calibrator.py`: persist `stem_id` per row **through the loaders,
   ACE/readout alignment, `merge_matrices`, and the `.npz`**, with an equality check
   against the source JSONL metadata (the current path keeps only scores/labels/
   `sample_idx` and the selector resamples row indices — this is a coordinated change,
   not a keyword flag); implement the exact cluster algorithm frozen in §5 inside the
   selector path; implement the paired-design null permutation of §4; accept the frozen
   canonical task key separately from the reported BENCH task key. These are the ONLY
   authorized edits to hashed compute files; together they define the extension-baseline
   hash. Existing `sealed_selector="... imported, not modified"` provenance must be replaced
   for cluster results by an honest resampling-unit-specific value.
5. `analyze_universality.py`: a registered fixed-cell LOMO path for
   `fusion_rank_mean_geom` implementing the exact §5 estimator and planned-denominator rule
   (denominator always 10; missing/failed holdouts count as fails; <3 usable training models
   → abort=FAIL), surfaced separately from the descriptive max-survival landscape.

## 8. Execution phases (order is binding)

1. **Phase 0 (this document):** freeze. MK sign-off recorded below.
2. **Phase 1 — harness:** edits of §7 + unit-level checks; run the ten-model frozen-row
   parity sentinel of §1 using only pre-existing seal data/matrices; inspect the diff from
   the Phase-0 calibrator hash and reject edits outside §7; then record and freeze
   `EXTENSION_MANIFEST.json` (hash manifest of §1) before Phase 2. No BENCH data drawn.
3. **Phase 2 — data:** build all task files with seed 20260711; run gates; record data
   manifests (source sha256/frozen Arrow fingerprints, per-class counts, intersection sizes,
   effective n). **Amendment discipline (replaces v1.0's blank cheque):**
   implementation-only corrections (a builder bug producing output that violates THIS
   document's frozen spec) require an Amendments entry filed BEFORE regeneration,
   stating the bug and why the fix is spec-restoring; any change to source, template
   bytes, sampling frame, exclusion sets, length cap, label convention, bars, or
   endpoint logic **voids Phase 0 and requires a new pre-registration**.
4. **Phase 3 — smokes:** stratified 16-row smokes per (model, task). A smoke failure marks
   that cell BEHAVIORAL-FAIL in the denominator (it is not run strict, not swapped out).
5. **Phase 4 — strict cells:** families A, B first (confirmatory), then C. `--resume`
   allowed across interruptions; pilot seeds banned; per-cell provenance stamped as in
   the seal.
6. **Phase 5 — analysis & propagation:** endpoints A1/A2/B1 scored against §5;
   results page `wiki/results/bench-extension-<date>.md`; paper extension section per the
   frozen language rules; vault propagation per canon (result page → candidates → summary
   → index → log; root orientation only if the frontier changed).

### 8.1 Abort rules (frozen; each abort is an Amendments entry + a reported outcome)

- 🧱 **Confirmatory supply abort:** if `halueval_qa`'s admissible 2048-token intersection
  yields < 500 stems, or any family-B task cannot reach its frozen n under the exclusion
  union, that family's cells are NOT run at reduced n — the family aborts and the
  extension re-registers (confirmatory n is not adjustable post-hoc).
- 🔏 **Hash-drift abort:** any T0-subset hash mismatch, frozen `fusion_signs.json` mismatch,
  extension-baseline mismatch after Phase 1, or model `resolved_revision` drift → the
  affected cells abort (no "run-anyway-with-caveat" for confirmatory cells). A parity
  sentinel mismatch does not masquerade as equality: it triggers the frozen feature-version
  caveat; the scientific cell may run only if code and model hashes still match.
- 📦 **Source-drift abort:** pinned HaluEval commit/file bytes or any frozen ANLI/TriviaQA
  Arrow fingerprint/sha256 unavailable or mismatched at build → abort; changing the source
  requires a new pre-registration.
- 🗣️ **Systematic commitment failure:** if ≥ 3 cohort models fail the §4 full-cell
  commitment audit on the same task, the task (not just the cells) is declared
  behaviorally infeasible for this cohort and reported as such — its cells all score
  as failures in the fixed denominators; no template retuning after data is drawn.

## 9. Amendments

(append-only)

### A1 — 2026-07-12 — Subword-prefix commitment rule (bench/1.2 → bench/1.3)

**Timing:** filed after Phase-3 smokes surfaced the issue and BEFORE any strict cell was
run or any registered metric computed. Phase-4 had not started.
**What the smokes found (44/60 cells complete at filing):** 9 COMMITMENT-FAILs with
three signatures — (a) Mistral-7B decodes first token `'Y'` on 5/5 of its smoked tasks:
its tokenizer splits `YES` into `Y`+`ES`, so the model IS committing YES at token 1 and
the v1.2 exact-form rule (`yes`/`no` only) misclassifies a genuine commit — a tokenizer
artifact, not a behavioral failure; (b) Qwen2.5 decodes `' To'` on 9/16 ANLI-R1/R2 smoke
rows (chain-of-thought instead of commit) — a REAL behavioral failure; (c) Phi-3.5
decodes `'\n'` on 4/16 ANLI rows (commit-locus offset) — a REAL failure under the
first-token contract.
**Change (MK decision, option "c" of three presented):** the canonical-commit test
becomes "normalized token is a non-empty prefix of `yes` or `no`" (rescues only the
subword artifact (a)). Signatures (b) and (c) remain failures — no behavioral rescue.
**What does NOT change:** templates, data (drawn and sha-pinned before the amendment),
sampling, bars, denominators, endpoints, controls, cluster algorithm — only the
normalizer predicate and `SPEC_VERSION` → `bench/1.3`.
**Materiality:** under v1.2, `anli_r1_rep` had 3 commitment-failing models, tripping the
§8.1 systematic-commitment rule → task infeasible → B1 mathematically dead partly on a
tokenizer artifact. Under v1.3, Mistral-7B is rescued everywhere; Qwen2.5 and Phi-3.5
ANLI failures stand (anli_r1_rep: 2 behavioral fails — below the §8.1 threshold; both
B1 endpoints remain reachable at ≥17/20 with zero slack beyond these two).
**Disclosure rule for the paper:** the extension section must state that the commitment
normalizer was amended between smokes and strict cells, citing this entry.
**Re-audit protocol:** completed smoke verdicts are re-scored from their stamped
`commitment_audit.rows` (token ids/decodes are immutable facts recorded under v1.2);
model outputs are not regenerated. Re-scored profiles are stamped
`reaudited_under=bench/1.3-A1`.

### A2 — 2026-07-14 — Restore the registered A2 analysis path; stem-aware E3

**Timing:** filed while Phase 4 was paused, before any strict cell completed and before any
registered BENCH metric was computed. This entry is not backdated and does not modify A1's
historical text.
**Defect:** A1 correctly changed execution profiles and summaries from `bench/1.2` to
`bench/1.3`, but the hash-frozen `analyze_universality.py` still strictly required
`bench/1.2` at both A2 input gates. It would therefore reject every profile produced by the
run it was registered to score and then reject the strict summary. A1's "what does NOT
change" clause omitted the analysis path, and the extension manifest's original A1
`unchanged` string named "analysis". Although literally accurate as a statement about file
bytes, that provenance statement was consequentially misleading because leaving analysis
unchanged made the registered estimator inoperable. The manifest retains that original
wording and adds a visible `unchanged_correction` linked to A2.
**A2 repair:** `analyze_universality.py` now has one analysis-local constant,
`ACCEPTED_SPEC_VERSION = "bench/1.3"`, and both profile and summary gates use strict equality
to it. It is deliberately mirrored instead of imported from `run_bench.py`: importing the
execution harness would couple matrix-only analysis to its MLX/runtime dependency boundary.
The extension manifest freezes and re-stamps both files together; a future spec bump must
amend both. The gate does not accept `bench/1.2`, a version range, or a prefix match.
**Estimator output schema:** `bench-a2/1.2` is retained. That string versions the estimator's
output contract, which this repair does not change; it is independent of the input execution
spec version.
**E3 correction (descriptive, non-gating):** paired-task label-efficiency subsamples now draw
whole stems, so one row of a TriviaQA/HaluEval pair cannot enter while its twin is left out.
For ungrouped tasks, where each stem is one row, the implementation takes the preserved
row-stratified RNG path identically and asserts the unique-stem condition. Reported sample
sizes remain label budgets. This correction is separate from the registered A2 endpoint and
cannot change an endpoint verdict.
**Scope:** this amendment is spec-restoring, not spec-altering. It changes no bar,
denominator, endpoint, estimator, fixed cell, cell set, sign convention, data, template,
sampling frame, control, or inference threshold. The `fusion_rank_mean_geom` LOMO estimator
and strict `> 0.55` / 8-of-10 rule are unchanged.
**Additive sealed sensitivity:** `cluster_sensitivity.py` is an explicitly descriptive,
non-gating stem-cluster sensitivity for the ten published sealed TriviaQA cells. It changes
only the resampling unit for that report and does not alter the sealed 18/20 result.
**Disclosure rule for the paper:** the BENCH extension must disclose the A1→A2 version-gate
desynchronization and pre-Phase-4 repair, the unchanged `bench-a2/1.2` output schema, and the
stem-aware E3 correction. Any label-cost claim must name the label budget and paired-stem
subsampling rule. The sealed cluster sensitivity must be labeled descriptive and non-gating.

### A3 — 2026-07-14 — Resolve frozen BENCH inputs by content, not builder-machine path

**Timing:** filed with A2 while Phase 4 was paused, before any strict cell completed and before
any registered BENCH metric was computed. Phase-2 data manifests remain byte-frozen.
**Defect:** the six frozen manifests truthfully record the absolute paths used on the builder
machine, but `run_bench.py` treated those provenance strings as portable identities. It
compared exclusion paths to the current dependency root and required Arrow files at the
recorded `/Users/msrk/...` locations. A vendored dependency or a reviewer cache therefore
failed despite containing the exact registered bytes. Rewriting the frozen manifests would
destroy provenance and still would not make absolute paths portable.
**Repair:** the recorded paths remain untouched as provenance. Exclusion-reference identity
is now the exact sha256 multiset of the enumerated union, resolved first at the recorded path
and then at the corresponding current/vendored reference. Frozen Arrow artifacts resolve
through `$CONFLUENCE_HF_CACHE`, then the platform Hugging Face datasets-cache default, then
the recorded path, and must match `FROZEN_ARROW_HASHES`. Missing bytes fail closed with the
environment variable and expected sha256 in the error.
**Scope:** A3 changes source resolution only. It changes no source bytes, datum, manifest,
exclusion union, sampling frame, seed, RNG, task, bar, denominator, endpoint, estimator,
panel, sign convention, control, or result interpretation. `SPEC_VERSION` remains
`bench/1.3`.
**Phase-3 consequence:** changing the load-bearing `run_bench.py` hash changes the extension
manifest sha256. Existing `SMOKE_SUMMARY.json` is not silently re-stamped. The strict launcher
continues to fail closed on that mismatch; the executor must perform and document the Phase-3
provenance re-audit required by the harness before Phase 4 resumes.
**Disclosure rule for the paper/review packet:** state that frozen manifests retain original
builder paths as provenance while execution resolves and verifies content-addressed inputs;
name `$CONFLUENCE_HF_CACHE` as the portability override. Do not call BENCH resume-ready until
the post-A3 smoke provenance gate passes.

## 10. Audit tables

### 10.1 v2-review (proposal-level) required fixes — status after v1.2

Codex's independent audit of the v1.0 draft scored several of these "partial"; the v1.1/v1.2
edits below are what upgraded them. Fixes marked (Phase-1-verified) only fully close when
the named code exists and the extension manifest is recorded.

| Review fix | v1.0 status (per Codex) | v1.2 resolution |
|---|---|---|
| 1. Hash coverage exact | partial | §1: T0 subset enumerated; frozen file set with no "sibling" escapes; current fusion/calibrator hashes frozen; BENCH runtime recorded without falsely claiming equality to an absent seal tuple; ten-model byte-parity sentinel added (Phase-1-verified) |
| 2. LOMO endpoint frozen | partial | §5 A2: named cell + planned-denominator rule (always /10; missing holdout = fail; <3 training models → abort=FAIL) as a new registered code path (Phase-1-verified) |
| 3. Task set + denominator frozen | partial | §2 unchanged (30 cells, no substitution) + §3.2 enumerated exclusion union + §8.1 abort rules replace the silent gaps |
| 4. P3 primary status | resolved (excluded) | unchanged |
| 5. Cluster bootstrap confirmatory | **failed** (not executable: `stem_id` discarded by loaders/merge/npz; selector resamples rows) | §5 exact frozen algorithm + §7.4 coordinated stem-persistence changes; B1 split into procedural/valid so no invalid gate supports replication language (Phase-1-verified) |
| 6. Stratified smoke | partial | §4: smoke kept + **full-cell commitment audit** on every strict row with frozen YES/NO normalization; dialogue/summ templates now byte-frozen |
| 7. FEVER freeze | resolved (excluded) | unchanged |
| 8. SimpleQA exploratory rule | resolved (excluded) | unchanged |

### 10.2 Bench-review (Phase-0 draft audit, Codex 2026-07-11, 4/10 FIX) → v1.1/v1.2 fixes

Full review: `wiki/paper/cc-bench-prereg-review.md`.

| Finding | Fix by v1.2 |
|---|---|
| Cluster gate not executable (`stem_id` discarded end-to-end) | §5 exact algorithm; §7.4 loader/merge/npz/selector changes with source-metadata equality check |
| "Outer and inner/OOB" misdescribes the selector | §5 corrected: one draw per replicate, in-bag selection, OOB evaluation |
| Declared frozen while sign-off pending | Header: DRAFT PENDING MK SIGN-OFF; freezes at sign-off |
| A2 denominator escape (`len(slugs)`) | §5 planned-denominator rule + abort=FAIL |
| B1 conflates estimands; train-R1 is not a same-distribution replication | §5 B1-procedural vs B1-valid; §3.2 language rule ("re-test", never "replication") |
| `run_seal.py` PASS semantics contradict the denominator policy | §5 explicit redefinition (terminal-status scoring over fixed denominators) in the new `run_bench.py` |
| Exclusion union machinery does not exist | §3.2 enumerates both file sets; §7.1/7.2 name the build work honestly |
| Dialogue/summ templates not byte-frozen; 16-row smoke insufficient | §3.1 verbatim templates; §4 full-cell commitment audit |
| "Sibling" hash escapes; venv-path runtime rule | §1 freezes file names and BENCH versions; because the seal tuple is incomplete, it requires an explicit limitation + byte-parity sentinel rather than fabricated equality |
| Amendment blank cheque; no abort rules | §8 Phase-2 amendment discipline; §8.1 abort rules |
| R2 pooling allocation unfrozen | §3.2: 250/250 per class per split, split recorded, split-specific descriptives reported |

### 10.3 Second static adversarial pass (Codex 2026-07-11, 6/10 FIX) → v1.2 fixes

| Finding | Fix in v1.2 |
|---|---|
| Replication task keys silently trigger modal fusion fallbacks | §1 freezes `*_rep` → sealed-task aliases; new tasks' modal signs and `fusion_signs.json` hash are explicit |
| A2 names a cell but not a reproducible LOMO estimator | §5 freezes within-model ranks, nine-model row pooling, sign-only training, strict threshold/missing rules, and stamped outputs |
| Seal runtime equality is impossible to verify from committed artifacts | §1 states the missing historical tuple, records BENCH versions, and substitutes a ten-model byte-parity sentinel; no fabricated equality claim |
| ANLI/TriviaQA revisions deferred until build | §3.2/§3.3 freeze exact cached artifact fingerprints and Arrow sha256 values before sign-off |
| Resume can mix row/cluster or differently oriented profiles | §7.3 freezes the additional configuration/provenance fields and exact resume rejection rule |
| Strict shuffled-control failure has no endpoint disposition | §4 makes geometric CONTROL-FAIL terminal and denominator-preserving; grouped nulls preserve pairs |
| Seed does not define a unique data draw | §3.4 freezes per-task RNG class/seed, source/index order, quotas, sealed-compatible ANLI/TriviaQA scans, HaluEval sampling, emission order, and manifest records |
| "Byte-comparable" conflates extraction with changed inference | §1 separates feature/extraction matching from extension-inference matching and scopes the cluster claim |
| B1-valid leaves generic replication language open | §5 freezes the only licensed B1-valid sentence and its train-split/cluster caveats |

## Sign-off

- Authored + facts verified (license, splits, counts, commit pin): Claude (Fable 5),
  2026-07-11 (v1.0).
- Adversarial review: Codex CLI, 2026-07-11, 4/10 FIX → all TIGHTENED items folded into
  v1.1 same day (`wiki/paper/cc-bench-prereg-review.md`).
- Second static adversarial pass: Codex, 2026-07-11, 6/10 FIX → all blockers/must-fixes
  folded into v1.2 same day (§10.3). No project code or tests were run by Codex.
- **MK red-line, 2026-07-12:** A1 bar **8/10 confirmed**; A2 bar **8/10 @ 0.55
  confirmed** (seal precedent); B1 bars **17/20 (each) confirmed** (seal precedent);
  dialogue/summ minimum n **raised 300 → 400**; length cap **to be set by tokenizer
  census** (tokenizer-only length measurement over the pinned HaluEval files with the
  frozen templates and per-model prompt strategies — no model forwards, no labels used,
  no data drawn; census seed distinct from the registered 20260711). The census result
  fixes the cap value in §4 before sign-off.
- **Census run 2026-07-12 (results in §4):** cap **2048 confirmed by measurement** —
  QA/dialogue 100% admissible; summarization 92.3% (~9,230 stems), all floors cleared
  by ≥18×. Every open integer is now resolved: length cap 2048 (measured), A1 8/10,
  A2 8/10 @ 0.55, B1 17/20 each, dialogue/summ min n 400.
- **MK sign-off: RECORDED 2026-07-12 ("signed off") — Phase 0 is FROZEN.** All
  integers as red-lined above (length cap 2048 measured; A1 8/10; A2 8/10 @ 0.55;
  B1 17/20 each; dialogue/summ min n 400). §7 Phase-1 scope accepted (Codex work order
  `CODEX_WORKORDER_BENCH_PHASE1.md`). From this point, changes go to §9 only.
