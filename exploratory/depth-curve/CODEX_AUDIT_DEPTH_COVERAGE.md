# Codex audit order — `depth_coverage.py` (depth-coverage discriminator)

**Mode: STATIC REVIEW ONLY. Do NOT execute anything.** No `python3 depth_coverage.py`, no
tests, no harnesses, no model code, no package commands. Read the source, read the emitted
JSON, and report. Where verification would require execution, write the exact command and
mark it "not run by Codex" — an executor will supply artifacts.

## Files
- `exploratory/depth-curve/depth_coverage.py` — the analysis under audit
- `exploratory/depth-curve/DEPTH_COVERAGE.json` — its output (read it against the code)
- `exploratory/depth-curve/rescore_grid_a.py` — the cross-fit convention being imitated
- `exploratory/depth-curve/colocation_analysis.py` — the prior analysis this follows from
- `confluence_calibrator.py` §`append_fusion_columns` / `_rank01` — the house fusion rule
- `exploratory/depth-curve/DC_DATA_CONTRACT.md` line 120 — the verified rung mapping

## What the analysis claims
That a per-model **depth-targeted single** attention instrument (instrument and block chosen
per fold on training rows) matches or beats the **fixed-rung** arms the ACE panel deploys
(2 instruments x rungs `N//2, N-2, N-1`), across 17 banked depth-curve cells. Headline
numbers to check against the code: fixed-rung FUSION is worse than the best SINGLE fixed rung
(median -0.038 both grids); depth-targeted single beats best single fixed rung by median
+0.120 in grid A (5/8 CIs exclude 0) but only +0.011 in grid B (1/9); shuffled-label control
lands every arm at 0.49-0.51.

## Attack surface — in priority order

1. **Bootstrap validity around a selection statistic.** The contrast re-runs the ENTIRE
   cross-fit (including argmax selection) per resample, then takes a percentile CI. Several
   cells show an interval visibly not centred on its point estimate (e.g. grid A
   Llama-3.3-70B/anli: delta +0.016, CI90 [-0.029, +0.225]). Is the percentile bootstrap
   valid here, or does resampling-with-replacement systematically shift the selection
   distribution relative to the point estimate? If it is biased, name the direction and
   whether any reported conclusion flips. Suggest the correct interval if one exists.

2. **Is the arm set fair?** `fixed6` fuses 6 columns; `target1` picks 1 of ~2L. Both are
   cross-fitted, so selection is paid for on held-out rows — but confirm there is no
   asymmetry in HOW MUCH each arm's selection is charged. Specifically: does the fusion arm
   get a free lunch or a free penalty from being unselected?

3. **Transductive within-fold ranking.** `fuse()` computes `rank01` inside the 40-row
   held-out fold. Single-column arms are rank-invariant so this cannot affect them, but the
   fusion arms' scores depend on the held-out set's own ranking. Does that advantage or
   penalise fusion, and is n_ho=40 too coarse for a 6-column rank mean?

4. **Pooling choice.** Cell value = mean of 5 per-fold held-out AUROCs (each on 20/20 rows).
   The alternative is pooling held-out predictions into one 200-row AUROC. Does the
   fold-mean understate or overstate, and does it interact with (3)?

5. **Fold map and seeds.** `make_folds` is copied from `rescore_grid_a.py`; the fold map is
   frozen per TASK by sorted-name index, and labels are asserted identical across models
   within a task. Bootstrap resamples within (fold x label) strata with the fold map frozen
   — confirm a duplicated row can never straddle train/held-out. Confirm no remaining
   dependence on Python's per-process string hash salt or on dict insertion order.

6. **Deviation from the registered convention, declared in the docstring.** Peak selection
   is a plain training argmax; the rescore additionally required `A_tr >= 0.65` AND above a
   per-fold shuffled-label envelope. The docstring argues that gate served the E5 endpoint
   DEFINITION, not bias control. Is that argument sound, and does dropping it change
   anything beyond which cells are declared evaluable?

7. **Scope discipline.** Confirm the code cannot pool grid A with grid B, touches no sealed
   artifact, and reads the npz trees read-only. Flag any statement in the docstring that
   over-claims relative to what the code computes.

## Deliverable
Findings ranked by severity, each with: the concrete failure scenario, whether it changes a
reported number or only its interpretation, and the minimal fix. Explicitly list anything you
could NOT verify without executing code.
