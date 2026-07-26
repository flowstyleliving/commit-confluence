# E3-HaluEval descriptive label-efficiency sweep — SPEC

STATUS: **DESCRIPTIVE / POST-HOC — NOT A REGISTERED ENDPOINT.** The ten
`halueval_qa` score matrices already exist and are published
(`stage_b/profiles_bench/halueval_qa/*.matrix.npz`), so this sweep cannot be
blind-confirmatory. It produces a HaluEval-QA label-cost *estimate*, never a
registered endpoint, and it cannot upgrade any BENCH headline claim (A1/A2/B1
stand as registered). Motivation: the merged CC paper truthfully states that
HaluEval-QA label cost is unmeasured — A1 calibrated on the full 1000 rows,
giving an upper bound and no lower bound. This sweep supplies the descriptive
lower-bound curve.

## Frozen choices (fixed before execution; no post-hoc variation)

1. **Cells.** Exactly the ten `halueval_qa` (model, task) matrices in
   `stage_b/profiles_bench/`. No other task enters.
2. **Machinery.** The production `analyze_universality.label_efficiency`
   verbatim — no reimplementation. Subsampling is therefore stem-aware by
   construction (Amendment A2 path: complete two-row {0,1} stems, even budgets
   only, `subsample_unit="stem"`); the nested OOB bootstrap *inside* each
   subsample remains row-level, exactly as in the sealed E3. Deployability per
   repeat = OOB AUROC 95% CI lower bound > 0.50; the reported statistic per
   (cell, budget) is the fraction of repeats deployable, full-panel and
   geometric-only.
3. **Budgets.** `{50, 100, 150, 300, 500}` labels. The first three mirror the
   sealed E3 grid for cross-task comparability; 300 and 500 extend the axis to
   ask where the curve flattens (possible here because n=1000; the sealed
   n=200 cells could not measure past 150).
4. **Repeats / bootstraps / seed.** `repeats=10`, `nboot=1000`,
   `seed=20260613` — identical to the sealed E3 convention.
5. **Output.** One JSON, `stage_b/profiles_bench/E3_HALUEVAL_DESCRIPTIVE.json`,
   carrying the per-cell per-budget table plus this spec's identity; committed
   alongside this spec. No existing artifact is modified.

## Reading rules

- Comparisons to the sealed E3 numbers are **descriptive across different
  tasks and n**; they share machinery and seed convention, not distribution.
- "Deployable at budget b" here is a subsample statistic, not a fresh-data
  claim; a registered label-cost endpoint would require a new registration
  with fresh data.
- If the curve is still rising at 500, the honest statement is a lower bound,
  exactly as ≥150 is for the sealed tasks. No knee may be claimed unless the
  curve is flat (within repeat noise) across two consecutive budgets.

executor: Claude (repo .venv); spec committed before execution.
