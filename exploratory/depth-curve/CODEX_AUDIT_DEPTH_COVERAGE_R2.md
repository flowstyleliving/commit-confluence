# Codex audit order — ROUND 2 — `depth_coverage.py` + `stability_diagnostic.py`

**Mode: STATIC REVIEW ONLY. Do NOT execute anything.** No `python3 depth_coverage.py`, no
`stability_diagnostic.py`, no tests, harnesses, model code, or package commands. Read the
source and the emitted JSON and report. Where verification would require execution, write the
exact command and mark it "not run by Codex" — an executor supplies artifacts.

Round 1's order is `CODEX_AUDIT_DEPTH_COVERAGE.md` (same directory). Read it first: this round
audits the REWRITE that responded to it, plus one new artifact that round 1 never saw.

## What round 1 changed (verify each landed, and landed correctly)

| # | Round-1 finding | Response in this version |
|---|---|---|
| 1 | ordinal-vs-average tie ranking biases results (rated High) | Tie rule replaced with midranks (`midrank_columns`). **The predicted consequence was measured and FALSIFIED**: 0 natural ties in point estimates; single-column inflation exactly 0.0000; fusion difference max 0.0017. Mechanism cannot fire because bootstrap ties form only between duplicate copies of one row, which share a label. Fixed for fidelity, not for effect. **Audit that reasoning, not just the code.** |
| 2 | percentile bootstrap around an argmax is not established; 3 cells flip classification under the mirrored convention | Estimand split. **Primary is now CONDITIONAL** — selections and training calibrations frozen from the real fit, only held-out rows resampled, so the statistic contains no argmax. Full-algorithm intervals still emitted in BOTH conventions with a per-contrast `full_convention_disagrees` flag. |
| 3 | "production-mirroring" / "reproduces the house rule" over-claimed | Binding NAMING DISCIPLINE block in the module docstring; `fixed6` declared a local construction; `rung_best1` declared not the production selector. |
| 5 | transductive within-fold ranking | Held-out values now mapped onto the TRAINING empirical CDF (`train_percentile`). |
| 6 | dropping the registered `>= 0.65` qualifying gate changes the estimand | `target1_gated` arm added. It reports **0/8 and 0/9 wins, median delta exactly 0.0000, and `gate_changes_selection` false in 85/85 folds.** |
| 7 | fold-mean vs pooled out-of-fold | Both emitted; medians differ by ~0.002. |
| 8 | JSON `sign` field emitted `"target"`/`"fixed"` regardless of which arms the contrast compared | Replaced with `winner: a/b/tie`; `target_wins` renamed `first_arm_wins`. |

## Attack surface — round 2, in priority order

1. **Is `target1_gated` genuinely inert, or is it silently a no-op?** A gate that never once
   changes a selection across 85 folds and returns a delta of *exactly* 0.0000 in all 17 cells
   is the signature of both "correctly inert" and "not actually wired in". Trace the gate
   through `crossfit` and confirm the restriction is applied to the candidate set before the
   argmax, that it can in principle bind, and that a training AUROC below 0.65 is reachable
   given orientation is also fit on training. **This conclusion is currently load-bearing** —
   it is the basis for saying the deviation from the registered convention is harmless.

2. **Does the CONDITIONAL estimand answer the question its docstring claims?** Selections are
   frozen from the real fit and only held-out rows are resampled within (fold x label) strata.
   Confirm: (a) nothing downstream of the freeze re-selects; (b) the training CDF used for
   calibration is the frozen real-fit one, not recomputed from resampled rows; (c) the
   docstring's characterisation — "given the columns this procedure actually selected, how well
   do they separate" — is neither stronger nor weaker than what the code computes. State
   plainly what this interval does NOT charge for, and whether any prose in the file lets a
   reader mistake it for procedure-level inference.

3. **`train_percentile` correctness and edge cases.** It maps held-out values onto the sorted
   training vector via `searchsorted` left/right averaged over `2*len(s)`. Check: values below
   all / above all training values; the midrank convention's consistency with
   `midrank_columns`; whether the claimed AUROC-invariance for single-column arms actually
   holds (the map is monotone NON-DECREASING, not strictly increasing — does the collapsing of
   distinct held-out values that fall in the same training gap ever change a single-column
   AUROC, and if so is that a bug or the intended cost of honest calibration?); and whether
   ties introduced by that collapsing interact with (1) of round 1.

4. **The pooled-OOF legitimacy argument.** The docstring asserts pooling 200 held-out rows into
   one AUROC is "legitimate only because every arm is percentile-calibrated against its own
   training fold, making folds commensurable." Is that sound, or does it still mix five
   differently-fit scoring functions into one ranking? If unsound, say whether it changes any
   reported conclusion (fold-mean is the reported statistic; pooled OOF is corroborative).

5. **NEW ARTIFACT — `stability_diagnostic.py` and the selection-stability finding.** This is a
   post-hoc stratification: cells split by whether the modal per-fold (instrument, block) pick
   recurs in >= 4/5 folds. Reported result: STABLE (n=10) median +0.1188, **minimum +0.0000,
   all >= 0**; UNSTABLE (n=7) median -0.0082, min -0.0770, containing all four negative cells;
   midrank Spearman(stability, delta) = +0.588.
   Audit specifically:
   - **Circularity.** Is this finding partly or wholly a tautology (unstable argmax -> noisier
     selected column -> lower held-out AUROC)? The docstring concedes it is "partly mechanical"
     and claims the non-trivial content is the stable group's FLOOR being >= 0. **Is that
     defence valid, or is the floor also mechanically forced?**
   - **Post-hoc selection.** The 4/5 threshold was chosen after seeing the deltas. Is the
     STABLE/UNSTABLE split robust to 3/5 or 5/5? Would you accept any quantitative claim from
     this, or only a directional one?
   - **Independence.** `delta` and `stability` are computed from the same cross-fit. Is the
     correlation contaminated beyond the mechanical channel above?
   - **Degenerate members.** The stable group's minimum is Qwen2.5-7B/halueval at exactly
     +0.0000 — its peak block IS the N-2 rung, so both arms are the same column. Does including
     a definitionally-zero cell in the "all >= 0" claim make that claim weaker than it reads?

6. **A reproducibility failure worth generalising.** An earlier scratch version of this
   diagnostic reported Spearman 0.657; the committed script reports 0.588, and 0.657 is not
   reproducible from `DEPTH_COVERAGE.json` under ordinal (0.549), midrank (0.588), or raw
   Pearson (0.577) conventions. The group medians were identical, so the discrepancy is in the
   correlation only. **Is there any code path in the committed script that could produce
   0.657** (a different stability definition — e.g. modal BLOCK ignoring instrument — or a
   different cell set)? If not, confirm the scratch figure should be treated as unreproducible
   and discarded rather than reconciled.

7. **Provenance completeness.** `provenance` carries script sha256, input npz sha256s, and the
   numpy version. Is anything missing that would prevent an independent rerun from being
   checked against this JSON — and does `stability_diagnostic.py` need its own provenance
   given it consumes only the JSON?

8. **Scope discipline, re-check.** Confirm the rewrite still cannot pool grid A with grid B,
   touches no sealed artifact, reads npz read-only, and that the NAMING DISCIPLINE block is not
   contradicted anywhere else in either file.

## Deliverable
Findings ranked by severity, each with: the concrete failure scenario, whether it changes a
reported number or only its interpretation, and the minimal fix. Explicitly separate
"changes a number" from "changes what the number may be called". List anything you could NOT
verify without executing code. Where round 1's finding was answered by a MEASUREMENT rather
than a code change (items 1 and 6 of the table above), say whether you accept the measurement
as settling it.
