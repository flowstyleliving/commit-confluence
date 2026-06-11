# Stage B - Sealed Unified-Panel Pre-Registration (v2, hardened 2026-06-10)

**Status:** reviewed + hardened; cohort confirmed (P + Llama-3.1-8B). Gated only on the
`confluence_calibrator` build + byte-level cross-check below. No fresh-seed run launched yet.

Supersedes `PRE_REGISTRATION_DRAFT.md`. Changes from the draft are flagged `[R#]` against the
review in `stage_b/REVIEW.md`.

## Question

When a single dispatcher fits **all in-house geometric families plus the confidence base at a
commit moment** (ACE at t=0 prefill-last position; readout at gen_step=1, the first generated
token - adjacent commit-moment positions, per-sample aligned, NOT identical), and a per-(model,
task) **nested-OOB** selector chooses the operating
cell **without oracle knowledge** (train-fold selection + sign-lock only), is at least one panel
cell deployable on every model in the cohort - and does that coverage replicate out-of-sample under
a fresh seed? This is a test of the *dispatcher*, not a search for a universal cell (retired).

## Build prerequisite (must land + verify before sealing)  [R1][R2]

`confluence_calibrator.py` lives in THIS repo and **imports the sealed machinery read-only** - it
must not edit `t0-morphology-furnace` (the frozen ACE/T0 core). Contract:

1. **Attention pass** (ACE): collect per-sample scores for `ATTENTION_PANEL_T0` via the sealed
   calibrator's collection path. -> score sub-matrix `A` (n x 21), panel labels.
2. **Readout pass** (RPV + residual + confidence, ONE source) [R3]: at the gen_step=1 commit instant
   (the readout's native locus, via the imported `trace_pair_features`), compute
   `{null_ratio_post_rank1, fisher_eff_rank, spectral_entropy, neg_shadow_logvol_r1,
   surprise, p_max}`. -> score sub-matrix `B` (n x 6). null_ratio is sourced HERE only, never
   also from the attention pass, to avoid a double definition. (ACE's t=0 and the readout's
   gen_step=1 are adjacent commit-moment positions; the merge aligns them per-sample.)
3. **Merge** by `sample_idx` with a hard label-alignment assert (same data file hash, same order,
   identical label vector). hstack -> merged `M` (n x ~27 after dropping non-finite columns),
   merged panel list.
4. **Select** with the SEALED `_nested_bootstrap_oob_auroc(M, labels, panel, n_bootstrap, seed)` -
   imported, not reimplemented, so train-fold sign-lock + OOB evaluation are inherited verbatim.
5. **Emit** one `CalibrationProfile` JSON per (model, task): selected cell, OOB AUROC median + CI,
   winner_stability, warnings, drift hashes, model snapshot SHA, data hash.

**Cross-check gate (no model run needed):** restrict `M` to the readout columns only and reproduce,
on the existing sealed RPV rows, the per-cell selection story (e.g. Qwen3-8B/anli: `fisher_eff_rank`
selected, OOB CI_lo > 0.5; null_ratio NOT selected). Numbers must match the comprehensive run's
marginal AUROCs within bootstrap tolerance before any fresh forward pass.

## Cohort (CONFIRMED: P + Llama-3.1-8B)  [R-scope]

10 models x {ANLI R1, TriviaQA paired}, n=200, **fresh seed** (not 20260526 / 20260611) = **20 cells**:
Llama-3.2-3B, Llama-3.1-8B, Mistral-7B-v0.3, Mistral-Nemo, Phi-3.5-mini, Phi-4-mini, Qwen2.5-7B,
Qwen3-1.7B, Qwen3-8B, gemma-3-4b.

Note: Llama-3.1-8B has no prior ACE seal, so its ACE cells are fresh-only (no sealed cross-check
reference) - acceptable for a fresh run, flagged for interpretation.

**Excluded from the sealed claim** (documented, routed to the gate/parser lane, listed exploratory):
dolphin-2.9.3-mistral-nemo-12b, gemma-3-1b, DeepSeek-R1-Distill-Qwen-7B - measurement-orphan on
>= 1 task in Stage A (surprise itself at chance; known chat-template / CoT-overflow breakage).

## Endpoints + registered claims  [R5]

The panel INCLUDES confidence cells; falling back to surprise when geometry is blind is a designed
feature. Two pre-registered claims:

- **PRIMARY (product coverage):** the full-panel nested-OOB-selected cell has OOB CI_lo > 0.50 on
  **>= 19/20** cohort cells. The readout-only cross-check (2026-06-10) already shows the likely
  single failure is `gemma-3-4b/anli`: nested-OOB is STRICTER than Stage A's marginal bootstrap
  (Stage A surprise CI_lo 0.55 deployable -> nested-OOB selected-winner CI_lo 0.45, NOT deployable),
  so that cell is a genuine orphan even WITH confidence, not surprise-backstopped. It is the allowed
  <= 1/20 primary failure.
- **SECONDARY (geometric science):** a dispatcher restricted to the geometric families
  (ACE + null_ratio + RPV, confidence EXCLUDED) has OOB CI_lo > 0.50 on **>= 17/20** cells; the
  known geometric gap `gemma-3-4b/anli` is documented in advance.

Pre-registered descriptive (non-gating): the win-map (which family is selected where); the backstop
check (on null_ratio-collapse cells e.g. Qwen3-8B/anli, RPV or ACE is the geometric winner); and the
corroboration-vs-complementarity correlation (families agree per-sample yet each win somewhere).

## Multiplicity is priced in, not pruned  [R4]

The merged panel is ~27 cells. Larger panels carry more selection multiplicity; nested-OOB absorbs
exactly this by re-selecting inside each in-bag resample and scoring OOB. `winner_unstable` warnings
are therefore EXPECTED and NON-FATAL; deployability is judged ONLY on OOB CI_lo > 0.50, never on the
in-sample point AUROC. We deliberately do not prune to the sealed ACE winner (that would inject
oracle knowledge); the honest dispatcher sees the full candidate set.

## Falsification

- PRIMARY full-panel dispatcher OOB CI_lo <= 0.50 on >= 2/20 cells -> the honest selector does not
  recover Stage A coverage; panel is an in-sample artifact. NO-GO.
- SECONDARY geometric-only dispatcher deployable on < 17/20 -> geometry does not stand without
  confidence on this cohort. Geometric thesis weakened; report as honest-negative.
- If surprise is the selected winner on EVERY cell -> geometry adds nothing under honest selection.
  NO-GO for the geometric thesis (the product claim could still hold, reported as such).
- Fresh-seed mean OOB AUROC drops > 0.05 vs Stage A full-sample point estimates on shared cells ->
  selection-bias inflation; re-scope and re-seal.

## Controls (pre-registered)

- Shuffled-label control per cell: selected-cell OOB CI must include 0.50.
- Rotation-invariance control (RPV): median |stat delta| ~ 3e-10 in the source run; re-assert.
- Drift hashes on all imported sealed modules must match the cross-check artifacts.

## Seed / provenance

Fresh seed stamped at run time (not 20260526 / 20260611; pilot seeds not reused). Per-cell
`data_hash_sha256`, model snapshot SHA, and imported-module hashes recorded. One CalibrationProfile
per (model, task) under `stage_b/profiles/<task>/<slug>.profile.json`.

---

# Amendments v3 (2026-06-10, second adversarial pass - BEFORE any fresh data exists)

All amendments below were registered before fresh data was generated or seen; none were
informed by fresh-run results. Findings C1-C7 in `stage_b/SELF_REVIEW.md` (second pass).

## A1 - Per-task n made exact (fixes a v2 inconsistency)  [C4]

v2 said "n=200" blanket while the sealed TriviaQA file is 100 rows. Registered counts:
**ANLI R1 = 200 rows; TriviaQA paired = 100 pairs = 200 rows** (label-balanced 0.50 exactly by
pairing). This conforms to the v2 "n=200" text, removes the power asymmetry (an n=100 cell has
~37-sample OOB sets - needlessly brutal against CI_lo > 0.50), and is registered before data
generation. The fresh-vs-StageA drift check compares per-cell AUROCs and is n-agnostic.

## A2 - CI semantics stated  [C5]

The nested-OOB CI aggregates resamples whose in-bag winners may differ; the reported `winner`
is the modal in-bag pick. **`deployable` certifies the selection PROCEDURE on that panel, not
the named cell in isolation.** Profiles carry `ci_semantics` + `winner_marginal` (the modal
winner's full-sample marginal, reference only). No numeric change to any endpoint.

## A3 - Controls implementation precision  [C2]

- **Shuffled-label control (per cell, per endpoint):** v2 said the permuted CI "must include
  0.50" - statistically naive, since a single permutation's 95% CI excludes 0.5 upward ~2.5%
  of the time by chance (~1 false alarm expected over 40 endpoint-cells). Registered version:
  **K=3 permutations, full nested-OOB each; the cell is FLAGGED if >= 2/3 permuted CIs have
  CI_lo > 0.50.** A flagged cell does not flip deployability; it invalidates that cell's
  result pending investigation and is reported in SUMMARY (`control_failures`).
- **Rotation-invariance control (RPV):** AMENDED from "re-assert" to **asserted in the
  source sealed-era run** (median |stat delta| ~ 3e-10). The fresh readout pass imports the
  same compute path whose sha256 is recorded per profile (`module_hashes`), so the property
  is inherited by code identity, not re-measured. Re-measurement would require modifying the
  frozen exploratory compute; declined.
- **Drift hashes:** implemented - every profile records sha256 of `pri_calibrator.py`,
  `comprehensive_run.py`, `diagnose_inter_head_disagreement.py`, `confluence_calibrator.py`,
  and `fusion_signs.json`, plus the HF snapshot revision of the model weights.

## A4 - Endpoint denominator integrity  [C1]

Endpoints are evaluated over the **20 PLANNED cohort cells**. An errored/uncomputed cell
counts as **NOT deployable** (the denominator never shrinks), and SUMMARY carries
`incomplete: true`. An incomplete run cannot claim the registered endpoints; the prior
harness silently dropped errored cells from n, which could print PASS in a state the
registration calls NO-GO.

## A5 - Data-provenance guards  [C3, C7]

- The harness refuses **sealed-CONTENT data files** (sha256 match, not path match) and
  **pilot seeds** {20260512, 20260526, 20260610, 20260611} unless `--allow-sealed-data`
  (which forces `is_preview`). "SEALED" can no longer be stamped on sealed-era data by
  passing the files explicitly.
- **Fresh seed != fresh examples.** A registered run's data files must PASS
  `stage_b/check_fresh_data.py` against the sealed files: schema, exact n (A1), label
  balance, zero intra-file duplicates, and **zero prompt overlap with the sealed 20260526
  examples** (normalized-text sha256). Overlap would quietly dilute the out-of-sample claim.

## A6 - Fusion cells join the panel (panel 27 -> 29)  [E4]

Two pre-registered cross-locus fusion candidates are appended to the merged panel before
selection (the dispatcher MAY pick them; nothing privileges them):

- `fusion_rank_mean_full` = mean of NaN-propagating per-cell rank transforms
  ((avg_rank - 0.5)/n_finite) of the oriented components
  {ACE_modal, surprise, null_ratio_post_rank1, fisher_eff_rank}. Contains confidence ->
  **excluded from the geometric endpoint**.
- `fusion_rank_mean_geom` = same over {ACE_modal, null_ratio_post_rank1, fisher_eff_rank};
  eligible for both endpoints.

Honesty constraints: fusion columns are precomputed before the bootstrap, so component
orientations are locked **per (model, task) from SEALED-ERA artifacts only**
(`stage_b/fusion_signs.json`, built by `stage_b/build_fusion_spec.py` from the 18 sealed ACE
profiles + the 20260526 RPV rows; committed before fresh data). Missing entries (Llama-3.1-8B)
fall back to the cohort-modal sign; surprise is hard-clamped +1 (canonical). ACE_modal =
`attention::last_minus_1_js_no_bos`, the modal sealed winner - NOTE the sealed winner tally is
dispersed (11 distinct winners / 18 profiles; modal wins only 3), so the fusion's ACE component
is a weak compromise by construction; this dispersion is itself per-deployment evidence.
A mis-oriented fusion on some model simply loses the selection - endpoints are unaffected.
Registered descriptive question (non-gating): does fusion ever win the dispatcher, and where?

## A7 - Pre-registered descriptive analyses (zero new forwards)  [E1, E2, E3]

The harness persists each cell's merged score matrix (`<slug>.matrix.npz`);
`stage_b/analyze_universality.py` (committed now, before data) runs:

- **E1 LOMO universality probe** (per task): rank-transform each cell within each model;
  select (cell, sign) on the 9-model pool; evaluate on the holdout. Report (i) the
  LOMO-winner row per holdout and (ii) the full cell x holdout landscape (flagged
  multiplicity-prone). **Interpretation guide, registered in advance:** a pool-selected cell
  holding AUROC > 0.55 on >= 8/10 holdouts = first evidence FOR partial universality;
  nothing surviving = cements the per-deployment framing. Plus a per-cell SIGN-STABILITY
  audit: sign-universal cells would license a fixed-orientation universal screener with
  per-deployment thresholds - a registered middle-ground hypothesis between "universal
  detector" and "per-deployment calibration".
- **E2 Task-transfer matrix:** per model, apply the task-A winner (cell + sign) to task-B
  scores, both directions, primary + geometric winners. Measures whether per-MODEL
  calibration suffices (the "/task type" clause of the goal).
- **E3 Label-efficiency curve:** stratified subsamples n in {50, 100, 150}, 10 repeats,
  nested-OOB at nboot=1000; report fraction deployable per size. Prices the labeling cost
  of per-deployment calibration.

All three are descriptive; none gate the seal. None require model forwards beyond the run.
