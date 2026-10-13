# Amendment A1 — standalone behavioural Endpoint (B)

**Date:** 2026-10-12. **Protocol:** `TEMP-v1.3-A1-behavioural/1-pre-output-audit-2026-10-12`.
**Parent:** `PRE_REGISTRATION_TEMP.md`, v1.3 draft, NOT frozen and unsigned; unchanged.
**Status:** pre-output draft revised after audit, pilot-informed and post-pilot. These revised files have not been committed or run. Execution/verification: **not run by Codex**.

## Authorization, version boundary, and pre-output commitment

MK explicitly authorizes running the full behavioural Endpoint (B) now, locally on the Apple M4 Mac mini (16 GB), as a standalone descriptive run **before Endpoint (A) exists**. This departs from §8.1's launch condition and §5.6's phase order. It also changes the §6.3-style gate dispositions, §8.2's extraction work, §9 stop rules, and §2.8's output location. Under §10 these are scientific-design changes, not implementation-only corrections: A1 is the separately versioned behavioural protocol for this phase, not a claim that the parent v1.3 was signed/frozen or completed. Endpoint (A) and its eventual confirmatory design remain unexecuted and unchanged by A1.

**Binding launch condition: this amendment and its matching harness must be committed before any full-run output exists.** Claude Code/MK must review and commit the files and supply that commit to `--amendment-commit`. Before creating the run directory or producing outputs, the driver uses Git at runtime to verify that the supplied 40-hex ID names an existing commit, resolve its tree, and compare the committed bytes of every launch-hashed file with the working tree. The launch set includes this amendment, the sampler, its inherited `sampling_pilot.py`, `stage_b/run_bench.py`, task data, and historical matrices. Any missing commit/file or byte mismatch refuses launch; the manifest records the verified commit and tree hashes. This enforces ordering for outputs of this driver, not an attestation that no external or earlier output exists; Claude Code/MK must verify that broader pre-output condition. Commit, execution review, and execution remain pending; Codex has executed neither Git nor the harness. This document must not be retrospectively described as a pre-pilot registration.

The 2026-10-12 pilot outputs were visible when these choices were made. Its first 20 ANLI rows and first 10 HaluEval stems (20 rows) are a subset of the full run, not an independent holdout. The full run reuses their original uniforms. No full-run-dependent threshold adjustment, Phi removal, prompt replacement, seed change, or selective rerun is permitted. A later correction requires another dated version, disclosure of visible outputs, and a stated regeneration scope; preserve superseded artifacts. A reporting-only correction may rerun the tabulator against unchanged raw output and launch files, with its analysis hash and change disclosure recorded; it must not silently change the scientific protocol.

## Revision after audit (2026-10-12)

The seven round-1 findings from the adversarial DeepSeek audit were addressed in this still-pre-output draft, with finding 5 remaining partial as clarified by round 2 below:

1. Any exact HaluEval historical-ID mismatch gives every affected cell `FAILED: historical_identity`, independently of the 1% infrastructure allowance; quarantine and full reporting remain, with mismatch and near-tie mismatch counts separate.
2. The post-pilot 25% sampled-invalid and 5% greedy-invalid thresholds are descriptive format-usability flags only. The inherited §6.3 ANLI ≥190/200 canonical-greedy rule is the only behavioural PASS/FAIL gate.
3. Exact P(invalid), sampled-token surprise, and strict-non-YES now report observed-only and all-available-distribution diagnostics, each with included/planned counts and explicit units.
4. Immutable launch provenance is separate from the tabulator's analysis hash. Reporting fixes may rebuild summaries with hash disclosure instead of an implementation-mismatch refusal.
5. Every generated draw records `cdf_lo`, `cdf_hi`, vocabulary size and the V−1 clip indicator. Accounting checks its uniform against these bounds, including quarantined draws, and treats violations as harness failures.
6. Runtime Git checks verify the supplied commit, tree and byte identity of all launch files before any run output is created.
7. All file-hash keys are resolved paths relative to the repository root, independent of the caller's working directory; paths resolving outside that root are rejected.

- **Round 2 — NEW-1:** Any `sampling_verification_failures > 0` gives the affected cell `FAILED: sampling_verification`, with a separate reason/count and zero tolerance outside the 1% infrastructure allowance. Independent defects retain their own counts/reasons; if historical identity also fails, its reason/count remains alongside the sampling status. This never stops measurement or changes other cells.
- **Round 2 — NEW-2:** Before rebuilding, move any existing `ACCOUNTED_OUTPUTS.jsonl`, `PROMPT_AUDIT.jsonl`, `CELL_SUMMARY.json`, and `SUMMARY.md` into a unique UTC-timestamped `superseded/` subdirectory of the run directory; record that archive path in the rebuilt reports. Raw outputs, `manifest.json`, and `uniforms.npy` remain untouched.
- **Round 2 — round-1 finding 5 (PARTIAL):** The tabulator checks the assigned uniform and recorded CDF bounds, but without the logits it cannot confirm that `token_id` owns that interval. A wrong token ID with the same canonical disposition can still pass these checks. No logit reconstruction is added; this is a bounded consistency check, not independent verification of the sampled token.

MK's four pilot fixes remain binding: temperature-0 identity uses historical BENCH answers, not `mlx_lm`, with top-two margins and near-tie flags; invalid, valid-wrong and harness failures remain separate with coverage by model × task × temperature; Phi stays in by advance decision for the reasons below; and per-cell allowances replace task-wide zeroing. Verification: **not run by Codex**.

## Evidence informing A1

Read sources: `pilot/runs/2026-10-12-pilot/SUMMARY.md`, `manifest.json`, and the adjacent pilot log. Four models × 40 prompts × (one greedy + 15 draws) yielded 2,560 records. The pilot reported one harness-bucket record, zero parser disagreements, and 80/80 HaluEval greedy token IDs equal to historical BENCH commitments. The one harness record was Qwen3-8B, HaluEval `sample_idx=4`: the harness chose token 14080 (`" YES"`), equal to BENCH, while `mlx_lm.stream_generate` chose token 5664 (`" NO"`). The reported T=1 YES/NO masses were approximately 0.51/0.45. These bucket masses are not themselves the top-two token logits; the pilot did not log the latter.

Invalid draws at T=0.7/1.0/1.5 (each /200, tasks pooled) were Qwen3 3/8/32, Llama-3.1 2/7/80, Phi 39/42/63, and Mistral 0/0/10. Phi additionally had 5/40 invalid greedy commits. At T=0.7, its mean exact invalid probability was 0.2079 versus <0.01 for each other model. Its most common invalid tokens were newline (103) and `The` (31); pilot continuations indicate response-writing instead of the required commit. This is model format failure, not evidence of a sampler bug.

MK's supplied historical lesson is that 14 noncanonical TriviaQA one-token commits (Gemma-3-4B 12/1,000, Llama-3.1-8B 1, Qwen3-1.7B 1) triggered the BENCH rule that a task with at least three COMMITMENT-FAIL cells is zeroed, causing all ten TriviaQA cells to fail. That account is supplied context, not independently re-executed or audited here. A1 explicitly forbids such task-wide propagation. The supplied vault record is commit `beeac50`, with four harness fixes in the backlog and a 2026-10-12 pilot log entry; Codex has not accessed or changed the vault.

## Cohort, exact scope, and randomness

**Binding decision: retain Phi-3.5-mini.** It is the deliberately registered low-decidedness stress model and preserves one of the four architecture families. Dropping it after observing approximately 20% invalid draws at T=0.7 would select on the outcome and obscure the format-failure behaviour this experiment needs to measure. Keeping it requires separating format errors from valid-but-wrong answers and allowing a stress cell to be FAILED yet fully analysed descriptively. There is no Phi-specific exemption or relaxed budget.

All four §2.2 models remain, in order, at their pinned revisions:

| Model (`mlx-community/` prefix) | Revision |
|---|---|
| Qwen3-8B-4bit | `545dc4251c05440727734bcd94334791f6ab0192` |
| Llama-3.1-8B-Instruct-4bit | `90215b22ec18e72f623dde2ea7af4097025160e2` |
| Phi-3.5-mini-instruct-4bit | `7b2052fd882fe017300d4d42a4eb06a27b816af4` |
| Mistral-7B-Instruct-v0.3-4bit | `a4b8f870474b0eb527f466a03fbc187830d271f5` |

Scope is **4 models × 2 tasks × 3 temperatures**, T={0.7,1.0,1.5}, K=5: all 200 ANLI rows and all 1,000 HaluEval rows (500 paired stems), plus one greedy commit per prompt. This is 24 sampled cells with 72,000 draws, plus 8 greedy diagnostic cells with 4,800 commits, or **76,800 planned records**. Per sampled cell N=1,000 for ANLI and N=5,000 for HaluEval; per greedy cell N=200 and N=1,000 respectively. No model/task/T/row/draw is removed from those denominators.

Use the unchanged §2.3 datasets and SHA-256 hashes already bound in `pilot/sampling_pilot.py`, including their source order, labels (0=YES, 1=NO), and HaluEval stem membership. Data-selection seeds remain ANLI `20260612` and HaluEval `20260711`; there is no new prompt draw. The harness checks source hashes/counts, valid labels/prompts, 500 stems with exactly one row per label, and one-to-one historical HaluEval indices, labels, and stems. Source labels test data alignment and correctness; they are never an identity reference for the model's commitment.

The sampling stream is unchanged: one `numpy.random.RandomState(20260826)` produces 72,000 float64 uniforms, nested in model → task (ANLI, HaluEval) → T → ascending source `sample_idx` → draw 0…4 order. The inherited `uniforms()` generates the entire design regardless of a failure; no skipped cell shifts subsequent uniforms. Write `uniforms.npy` before model calls and require its pilot manifest hash `d3b9cdb288ff9ecb9746f76edd1dea53ffe28ba4eeccc7d7eba0aee58a31ada6`. Sample over ascending token ID with `searchsorted(cumsum(q), u, side="right")`, clipped to V−1 exactly as §2.6. No top-k/top-p, rejection of invalid tokens, renormalization onto YES/NO, or extra draw is allowed.

For each generated draw store the cumulative mass immediately below and at its selected token as `cdf_lo` and `cdf_hi`, plus `vocab_size` and `cdf_clipped`. The tabulator verifies the assigned uniform and `cdf_lo ≤ u < cdf_hi`. The sole exception is §2.6's V−1 clip: `cdf_clipped=true`, token ID V−1, and `cdf_hi ≤ u < 1` when the cumulative sum rounds short of u (equality also yields searchsorted rank V). Bounds must be finite, ordered and within probability range (upper rounding tolerance 1e−12); a token-0 lower bound must be zero. Check every available draw, including historical quarantine; a missing selection witness or violated bound is an `a_harness` sampling-verification failure with zero tolerance, counted separately from ordinary infrastructure failures. A slot with no generated token remains a missing/infrastructure failure. These recorded bounds audit selection consistency; they do not independently reconstruct the vocabulary logits or establish that the token ID owns the recorded interval.

All four models use raw prompt passthrough and the pilot's BOS encoding rule. A commit is exactly one token. Keep the BENCH normalizer and nonempty YES/NO prefix rule unchanged, including canonical subword prefixes; “invalid” below means noncanonical under that rule, not merely a token whose literal text differs from `YES`/`NO`.

## Greedy reference, near ties, and ANLI limitations

For every prompt compute greedy `argmax(z_prefix)`, resolving exact ties to the lowest token ID. Log the first and second token IDs in descending logit order (ascending ID for tied logits), their unscaled logits, and margin `z_top1 − z_top2`. A **near tie is margin ≤0.15**, fixed now as a descriptive convention, including exact ties. Every available prompt margin/ID pair goes to raw records and `PROMPT_AUDIT.jsonl`; the summary reports counts by model/task, available/planned denominators, and near-tie historical mismatches. Failed/unattempted prompts have explicit unavailable/null audit entries. Near ties neither excuse identity mismatches nor change sampling or budgets.

**HaluEval reference:** the historical BENCH matrix's `gen_token_ids`, joined one-to-one by `sample_idx`, with its §3.6 hash checked. Compare exact IDs for every fresh greedy commit. Also record canonical-disposition agreement diagnostically. `mlx_lm` generation is not the reference and cannot cause a harness failure merely by disagreeing. The original pilot's `mlx_lm` field remains a historical diagnostic; the full runner omits that extra generation entirely.

An exact HaluEval mismatch makes that prompt's greedy record and all 15 draws `a_harness`, because their shared prefix fails the historical reference check. Preserve any generated token/text, probabilities, `sample_error`, and `observed_bucket` as quarantined diagnostic evidence; none contributes to trusted model-outcome rates. The comparison has zero token-ID tolerance, including near ties. **Any mismatch gives each affected cell the distinct status `FAILED: historical_identity`, even below 1% of its planned records.** Historical mismatch alone never stops the run: finish measuring and report everything. Count mismatching prompts separately from near-tie mismatching prompts (the latter is a subset), as well as affected slots per cell. Missing/duplicate/extra historical indices, hash failures, or label/stem misalignment are structural preflight failures and stop execution.

**ANLI replacement:** no historical per-row token record exists. Use the §6.3 behavioural canonicality rule (≥190/200 fresh greedy commits canonical), complete source-row/label alignment, and the row audit: attach the same greedily computed token ID and margin metadata unchanged to that prompt's greedy record and all three T × five sampled records. Sampled token IDs are allowed to differ. A broken row audit is a harness defect; a faithfully observed noncanonical greedy commit is a model format failure. The ≥190/200 rule applies to the greedy ANLI diagnostic cell only and does not automatically fail sampled cells. Each sampled cell has its own budget.

The ANLI row audit is bookkeeping, not historical identity. **ANLI has no historical per-row token record; A1 uses a separate ≥95% greedy behavioural canonicality check and row audit, with no full-panel tolerance parity or transitive identity protection.** This limitation appears in the full-run report. No claim of §6.4 parity is permitted. A1 suspends the extractor-dependent parity/smoke requirements for this behavioural phase, rather than pretending they passed.

## Outcomes, denominators, gates, and descriptive flags

Each planned slot belongs to exactly one exclusive bucket:

| Bucket | Meaning |
|---|---|
| a | Harness failure: missing/unattempted output, structural/record defect, numeric failure, or failed historical reference. |
| b | Trusted model output, invalid/noncanonical. |
| c | Trusted canonical output, wrong relative to source label. |
| d | Trusted canonical output, correct relative to source label. |

For every (model, task, T), including greedy diagnostics, report N and all four counts and **a/N, b/N, c/N, d/N** separately. Also report observed outcomes `O=N−a`, valid outcomes `V=c+d`, and invalid rate `b/O`. The **primary model error rate is `(b+c)/O`**; it counts both format failure and wrong canonical answers. The **conditional-on-valid error rate is `c/V`**, a secondary description subject to selection by format compliance. Zero denominators produce NA, never zero.

Retain §8.2's draw-level `sample_error`: 0 only for canonical-correct, 1 for canonical-wrong or invalid. Retain `strict_non_yes`: 0 only for canonical YES, 1 for NO or invalid. Harness failures have no trusted behavioural outcome; retained fields on such records are excluded from trusted outcome rates but included, when available, in the separately labelled diagnostic series below. Never silently assign a model error to a missing draw.

The §8.3 **full-denominator `sample_error_rate`** is the unweighted mean of prompt/stem unit error means. With complete output it equals `(b+c)/N` and the primary rate, because all ANLI prompts have five draws and all HaluEval stems have ten. With any harness failures it is NA; report the observed primary rate and full-design bounds `[(b+c)/N, (a+b+c)/N]` instead. These bounds assign all unknown outcomes success versus failure. Neither `sample_error_rate` nor its observed analogue may appear alone: show the invalid, valid-wrong, valid-correct and harness rates beside it. Strict-non-YES is answer bias, not an error rate.

Gates and flags are uniform across all four models and fixed before the full run:

| Check | Threshold and denominator | Disposition |
|---|---|---|
| Historical identity integrity gate | Zero exact HaluEval greedy ID mismatches. | Any mismatch fails affected cells with reason `historical_identity`, regardless of the infrastructure allowance; status `FAILED: historical_identity` unless sampling verification also fails. |
| Sampling-verification integrity gate | Zero inverse-CDF selection-witness failures per cell. | Any violation marks only that cell `FAILED: sampling_verification`; retain all other failure reasons/counts, including historical identity. Never stops the run. |
| Missing/infrastructure integrity gate | `I/N ≤0.01` per cell, where I counts missing/infrastructure/record/numeric failures, excluding historical-only quarantine and sampling-verification-only failures. At most 10/1,000 ANLI sampled or 50/5,000 HaluEval sampled; greedy at most 2/200 or 10/1,000. | Exceedance marks that cell FAILED with reason. |
| Outcome availability gate | `O > 0`. | O=0 marks the cell FAILED; rates with zero denominators are NA. |
| ANLI greedy behavioural gate | `c+d ≥190` out of all 200 planned prompts. | Below 190 marks only that greedy cell FAILED. This is the only behavioural gate. |
| Model invalid, positive T: descriptive flag | `b/O >0.25` per cell. | Add `format_usability`; no change to PASS/FAIL. |
| Model invalid, greedy: descriptive flag | `b/O >0.05` per greedy cell. | Add `format_usability`; no change to PASS/FAIL. |

Equality at the 1% infrastructure limit and the 190/200 ANLI bar passes; equality at a format threshold does not raise its flag. The 1% allowance limits infrastructure contamination; it is not evidence that missing outcomes are random. Full-design bounds remain mandatory even below it. Historical-only and sampling-verification-only failures do not consume I; a slot with independent defects in multiple categories counts in each applicable category but only once in a/N. When both historical identity and sampling verification fail, the status is `FAILED: sampling_verification` and both reasons/counts are retained. The 25% and 5% format thresholds were chosen after the pilot, with no independent pre-registered basis for general PASS/FAIL use. They are therefore descriptive usability flags, not scientific acceptance gates or population guarantees. The actual pre-registered §6.3 rule is ≥190/200 canonical ANLI greedy commits over the planned prompts; extending it to every model/task greedy cell via a 5% observed-invalid cutoff is not justified. Near-tie margin ≤0.15 and independent-parser disagreement are also diagnostic flags only.

Failed or flagged cells remain in every table and retain all raw and accounted records, including their correct outputs. **No failed cell zeroes, removes, or changes another cell or task.** Per-model, per-task, per-temperature allowances replace task-wide zeroing. Invalid-rate flags never stop sampling or trigger a replacement cohort. Historical mismatches independently fail each affected cell but never stop otherwise executable measurement. Any spread across T from a common bad prefix reflects failures of those actual draws, not a blanket propagation rule. A complete cell with format flags still contributes its descriptive behavioural bootstrap. A raw file with unassignable corrupt/extra records is untrusted for that model's cells; preserve the lines in the audit and count its slots as harness failures, without affecting other models.

## Reported analysis and deliberately unavailable readouts

The inference unit is **one ANLI prompt** (all five draws) or **one HaluEval stem** (both labelled rows, all ten draws). Draws are never independent inferential units. Report all per-cell rates, flags/reasons, coverage O/N, greedy historical/row audits, near-tie counts and parser diagnostics. The independent letters-only parser is a diagnostic, not an alternative to the registered BENCH normalizer.

For each cell, report **both observed-only and all-available-distribution values** for exact mean P(invalid), mean sampled-token surprise, and strict-non-YES, with **included/planned counts and units beside every value**. Observed-only excludes a_harness; all-available-distribution includes retained finite diagnostic values from quarantined records as well. Exact P(invalid)=1−p_valid is averaged once per distinct prompt at that T, never five times because it was copied onto five draws; its planned denominator is 200 ANLI prompts or 1,000 HaluEval prompts. Surprise and strict-non-YES are averaged over available records, with the cell's planned N as coverage denominator. Strict-non-YES remains the actual token indicator (including retained quarantined tokens), not the distribution expectation 1−P(YES). Available distribution diagnostics are descriptive and do not restore trust to quarantined outcomes. Nonfinite/out-of-range values and missing fields are unavailable, not zero. Greedy has no q_T or surprise: both report NA with 0/planned included; its actual-token strict-non-YES remains defined. No available values yields NA.

Keep the behavioural part of §8.4: pair indices 0…7 in cohort/task order, `RandomState(20260826+40001+pair_index)`, 2,000 calls to `rng.randint(0,n_clusters,size=n_clusters)`. Apply each multiset jointly across all three T anchors. Report sampled-error point estimates and T-wise 2.5/97.5 percentile intervals, and `R_1.0=(M(1.0)−M(0.7))*(M(1.5)−M(1.0))`, point R, bootstrap median and 2.5/97.5 percentiles, attempted=2,000 and usable. HaluEval stem order is first source occurrence. With any harness slot in the three sampled cells, mark the full behavioural contrast/intervals unavailable with usable=0, rather than dropping prompts/stems or filling in outcomes. A format-usability flag alone does not suppress numerical descriptive analysis. There is no confirmatory verdict or multiplicity claim.

Draw-level **surprise** is descriptive: `−log q_T(sampled_token)`, natural-log units, with float64 `q_T=softmax(z_prefix/T)`. Implement as `−[(z_i−max(z))/T − log(sum_j exp((z_j−max(z))/T))]` to avoid underflow from taking a log of a rounded-to-zero probability. There is no epsilon floor, post-commit probability, or greedy-temperature division. Store surprise only for positive-T draws, with exact canonical YES/NO masses; greedy surprise is not defined in A1.

No ACE/readout extractor exists yet: §3.7 artifacts are unbuilt. Therefore the 31-cell panel, geometry, post-commit metrics, both fusion scores, selector/deployability/coverage analysis, and the **`fusion_rank_mean_geom` reversal cannot be computed**. A1 also does not compute surprise detector AUROC or its detector reversal; adding cheap raw surprise does not complete the registered detector analysis. Only the behavioural error reversal is computed. Endpoint (A), full-panel parity and smokes, extractor provenance equivalence, and H-TEMP-1 falsification are NOT COMPUTED, not successful and not silently replaced by behavioural output.

## Local execution, artifacts, runtime estimate, and stops

New sibling files `pilot/sampling_endpoint_b.py` and `pilot/summarize_endpoint_b.py` reuse the pilot's cohort, task pins, uniform stream and free-memory function. `pilot/sampling_pilot.py` and `pilot/summarize_pilot.py` are unchanged so the original pilot behaviour remains reproducible. The full sampler uses one prefix forward per prompt shared by greedy and all three temperatures, with no stochastic model forwards, ACE capture, post-commit forward, diagnostic continuations, or `mlx_lm` parity generation. That is **4,800 study prefix forwards plus four vocabulary-size probe forwards**; it explicitly replaces §8.1's extraction budget for this behavioural-only phase.

Local M4 Mac mini only (`Mac16,10`, Apple Silicon); one model per subprocess. Require at least 30% free memory before each model load, as in the pilot. Use only locally cached pinned snapshots with offline loading. No Modal, cloud worker, paid compute, or automatic download fallback. If completion would need paid compute, **stop and report the expected cost to MK**; absent a quote report cost unknown, not zero, and do not proceed. Expected incremental external compute spend for this authorized local route is zero; local electricity/hardware costs have not been estimated.

The pilot log gives the following measured wall/load seconds; the manifest spans about 946 seconds (15.8 minutes). The last column is the **estimate** `load + 30*(wall−load)`, where 30=1,200/40 prompts:

| Model | Pilot wall seconds | Pilot load seconds | Estimated full hours |
|---|---:|---:|---:|
| Qwen3-8B | 250.3 | 116.8 | 1.14 |
| Llama-3.1-8B | 300.5 | 113.7 | 1.59 |
| Phi-3.5-mini | 205.1 | 53.9 | 1.27 |
| Mistral-7B | 190.0 | 102.8 | 0.76 |

This load-adjusted estimate totals about **4.8 hours**. Direct 30× scaling is about **7.9 hours**. Plan roughly **5–8 hours sequentially**, an estimate, not a measured full-run runtime or guarantee. ANLI expands 10× while HaluEval expands 50×; prompt-length mix, cache/load state and memory pressure differ. The full runner removes pilot continuations and parity generation, so these scalings do not predict work exactly. A fixed four-hour timeout per model replaces the pilot's 20-minute timeout; a timeout is a stop, never an automatic paid-compute escalation. Record actual model wall times in the full manifest.

Outputs are confined to a fresh directory under **`pilot/runs/`**, already ignored by `pilot/runs/.gitignore`; this overrides §2.8 only for A1. Refuse to overwrite an existing run. Write uniforms/manifest before calls; per-model JSONL is exclusive-created and flushed after each prompt; model logs, vocabulary buckets, and timings are separate. The manifest records `amendment_commit`, `amendment_tree`, `launch_files_sha256`, `analysis_files_sha256_at_launch`, separate `gates` and `format_flags`, package versions, host, pinned revisions, timings and stops. All hashed file-path keys are resolved repository-relative paths; relative inputs are interpreted from the repository root, not the working directory. Historical model/module environment equivalence is not certified by this manifest; exact HaluEval identity remains the available empirical check.

Launch provenance excludes `summarize_endpoint_b.py`; its hash is recorded separately at launch. The tabulator still requires unchanged launch hashes and the uniform-file hash, but permits changes to its own reporting code. Each `SUMMARY.md` and `CELL_SUMMARY.json` records the current analysis hash, the launch analysis hash and whether they differ. Before opening any derived report for rebuilding, the tabulator moves the four existing derived reports named below (or any available subset) into a unique UTC-timestamped directory under the run's `superseded/` directory and records its relative path in the new summaries. An archival error aborts rebuilding before any report is overwritten. Disclose the correction and visible outputs under the version rule above. The launch manifest, raw records, and `uniforms.npy` remain unchanged. This separation permits reporting fixes without silently changing data collection or requiring model regeneration.

After completion or interruption, the tabulator writes `SUMMARY.md`, `CELL_SUMMARY.json`, `PROMPT_AUDIT.jsonl`, and `ACCOUNTED_OUTPUTS.jsonl`. The last file contains all 76,800 planned slots, synthesizing explicit harness records for absent slots; duplicate records fail their slot instead of inflating its denominator. Raw files and unexpected/corrupt lines remain available for audit. A crash preserves completed models and flushed prompts; this is per-model durability, not automatic resume. Regeneration requires a new output directory and the §10-style disclosure above; do not pool attempts or silently replace failures.

Stop before model work for a nonexistent/unverified commit, a launch file differing from that commit, wrong host, missing cached revision, source/reference/module/uniform hash mismatch, historical schema/row/label/stem defect, or inability to record provenance. Stop further model launches on insufficient memory, timeout, or a failed child process. Numeric probability/logit/shape failures or model exceptions preserve the failing prompt and earlier records, stop that subprocess and subsequent launches, and leave every unattempted slot a reported harness failure. Cell gate failures and descriptive format flags alone are reporting dispositions, not execution stops. A shared numeric/provenance defect must be resolved and amended before regeneration. No output can establish that an unattempted cell passed.

## Handoff and evidence required for completion

Claude Code owns execution verification after this write/audit-only handoff. Review the A1-to-code correspondence, run the accompanying small self-check suite and any necessary local import/schema checks, verify the pre-output commit requirement, and then execute the authorized local run. No test, import, Python, MLX, Git, model, or other code was executed by Codex.

The CLI is unchanged; `--amendment-commit` now enforces commit existence and byte identity instead of accepting an attestation alone. Illustrative commands for Claude Code **after the required commit**, from this workspace (not executed by Codex):

```sh
python -m unittest discover -s pilot -p 'test_endpoint_b.py'
python pilot/sampling_endpoint_b.py --out runs/<new-full-run-id> --amendment-commit <reviewed-40-hex-commit>
python pilot/summarize_endpoint_b.py runs/<new-full-run-id>
```

Run the tabulator even after a stopped run. Completion evidence is the committed amendment/harness, verification report, full manifest and per-model logs, 76,800 accounted slots, 24 sampled plus 8 greedy cell statuses, greedy audits, and eight behavioural contrast reports with attempted/usable counts. Until that evidence exists, this deliverable is **drafted and statically audited; verification: not run by Codex**.
