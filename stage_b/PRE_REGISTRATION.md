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
