# Stage B - Sealed Unified-Panel Pre-Registration (DRAFT)

**Status:** DRAFT - not sealed. Gated on (a) user confirmation of cohort scope, and
(b) the unified single-pass calibrator build (prerequisite below). No fresh-seed run has
been launched from this page.

## Question

When a single calibrator fits **all three in-house geometric families plus the confidence base
in one forward pass** at the commit locus, and a per-(model, task) nested-OOB dispatcher selects
the operating cell **without oracle knowledge**, is at least one panel cell deployable on every
model in the registered cohort - and does that coverage replicate out-of-sample under a fresh seed?

This is not a search for a universal cell (retired). It is a test of the *dispatcher*: does honest,
selection-bias-free model selection over the merged panel recover the coverage Stage A saw in-sample?

## Prerequisite build (must land before sealing)

A unified `confluence_calibrator` that, from one t=0 / commit-locus forward pass per sample, emits:
- ACE attention panel cells (reuse `t0-morphology-furnace/pri_calibrator.py` `--t0-commit` path)
- `null_ratio_post_rank1` (residual-motion, v3)
- `fisher_eff_rank` (RPV primary) + `spectral_entropy`, `neg_shadow_logvol_r1` (secondary)
- `surprise`, `p_max` (base)

then runs the **existing** nested-OOB selection (schema >= 1.1) over the merged candidate panel,
emitting one `CalibrationProfile` per (model, task) with sign-locked-from-train direction, drift
hashes, and deployability rails. The contract: identical numbers to the source runs when restricted
to a single family (byte-level cross-check on >= 1 model before any fresh run).

## Registered cohort (CONFIRM BEFORE SEAL)

Recommended **primary cohort P** = the 9 ACE-sealed models x {ANLI R1, TriviaQA paired}, n=200,
**fresh seed** (not 20260526, not the RPV 20260611):
Llama-3.2-3B, Mistral-7B-v0.3, Mistral-Nemo, Phi-3.5-mini, Phi-4-mini, Qwen2.5-7B,
Qwen3-1.7B, Qwen3-8B, gemma-3-4b  ->  18 cells.

Optional **+1**: add Llama-3.1-8B (Stage-A-clean; extends ACE to a within-family scale point) -> 20 cells.

**Excluded from the sealed claim** (documented, routed to the gate/parser lane, listed as exploratory):
dolphin-2.9.3-mistral-nemo-12b, gemma-3-1b, DeepSeek-R1-Distill-Qwen-7B - all measurement-orphan on
>= 1 task in Stage A (surprise itself at chance; known chat-template / CoT-overflow gate breakage).

## Primary endpoint + registered claim

For each cohort cell, the dispatcher returns the nested-OOB-selected panel cell and its OOB AUROC CI.
**Registered claim (one-sided):** on >= 17/18 primary-cohort cells (allowing the single documented
`gemma-3-4b/anli` surprise-backstopped gap), the dispatcher's selected cell has OOB CI_lo > 0.50
**without oracle knowledge** (selection uses train folds only).

Pre-registered secondary (descriptive, not gating):
- geometric-only coverage: fraction of cells where a *non-surprise* family is the selected winner.
- backstop check: on the null_ratio-collapse cells (e.g. Qwen3-8B/anli), RPV or ACE is the selected winner.
- agreement: do the three families' per-sample risk scores correlate (corroboration) yet each win
  somewhere (complementarity)? Report the win-map, not just the mean.

## Falsification

- If the dispatcher's selected-cell OOB CI_lo <= 0.50 on >= 3/18 primary cells -> the honest selector
  does **not** recover Stage A's in-sample coverage; the panel is an in-sample artifact. NO-GO.
- If, on every cell, surprise alone is the selected winner -> the geometry adds nothing under honest
  selection; the "panel" collapses to plain confidence. NO-GO for the geometric thesis.
- If the fresh-seed coverage drops by > 0.05 mean OOB AUROC vs Stage A's full-sample point estimates
  on the shared cells -> selection-bias inflation; re-scope.

## Controls (pre-registered)

- shuffled-label control per cell must be flat (selected-cell OOB CI must include 0.50).
- rotation-invariance control already clean in the RPV run (median |stat delta| ~ 3e-10); re-assert.
- drift hashes on all source modules must match the byte-level cross-check artifacts.

## Seed / provenance

Fresh seed, stamped at run time (not reused from 20260526 / 20260611). Pilot seeds not reused.
Per-cell `data_hash_sha256` recorded. One CalibrationProfile JSON per (model, task) under
`stage_b/profiles/<task>/<slug>.profile.json`.
