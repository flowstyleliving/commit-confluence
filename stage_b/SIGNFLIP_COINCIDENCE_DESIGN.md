# A2 sign-flip coincidence: designed-retrospective screen

## Status and observation

This is a **DESIGNED-RETROSPECTIVE** screen. It is **NOT a finding** and **NOT
blind-confirmatory**: the relevant matrices already exist. The motivating observation is
that the A2 flip trio—`Mistral-7B-Instruct-v0.3-4bit`,
`Mistral-Nemo-Instruct-2407-4bit`, and `Qwen2.5-7B-Instruct-4bit`—coincides with the v4
sealed `E_A2` partial-transfer trio identified in research-candidates §5. The screen asks
only whether that fixed membership predictor is associated with auxiliary-task sign
orientation; it cannot upgrade the coincidence into confirmation.

## Frozen design

1. **Nine-model common cohort.** Freeze these models in this order:
   `Llama-3.2-3B-Instruct-4bit`, `Llama-3.1-8B-Instruct-4bit`,
   `Mistral-7B-Instruct-v0.3-4bit`, `Mistral-Nemo-Instruct-2407-4bit`,
   `Phi-4-mini-instruct-4bit`, `Qwen2.5-7B-Instruct-4bit`, `Qwen3-1.7B-4bit`,
   `Qwen3-8B-4bit`, and `gemma-3-4b-it-4bit`. These are the BENCH models with at least
   two existing matrices among the three frozen auxiliary tasks. `Phi-3.5-mini-instruct-4bit`
   is outside the common cohort because it has fewer than two such matrices; it may not be
   substituted or added.

2. **Predictor fixed before analysis.** Predictor `trio_member=1` exactly for
   `Mistral-7B-Instruct-v0.3-4bit`, `Mistral-Nemo-Instruct-2407-4bit`, and
   `Qwen2.5-7B-Instruct-4bit`; it is `0` for the other six models. The predictor comes
   from v4 partial-transfer membership, not from any auxiliary-task sign result.

3. **Tasks.** The discovery task `halueval_qa` is **EXCLUDED** from all fitting,
   classification, and reporting for this screen. The auxiliary tasks fixed in advance
   are exactly, and only, `anli_r2`, `halueval_dialogue`, and
   `halueval_summarization`.

4. **Fixed cell and per-task sign.** The only cell is `fusion_rank_mean_geom`. For each
   model and each usable auxiliary task, independently fit its sign with the existing
   `_score_candidate` rule on that model-task matrix and its labels. Record `+1` or `-1`
   as returned; do not use a sign learned from another model or task. A model-task is
   usable iff its matrix exists, contains exactly one cell with that name, has at least
   four finite cell/label pairs spanning both labels, and returns a nonzero finite fitted
   result. Otherwise record `NA` plus the mechanical reason.

5. **Model outcome and missingness.** A model receives a binary positive-orientation
   majority outcome only when at least two of the three auxiliary tasks are usable.
   Set `positive_majority=1` iff the number of `+1` signs is strictly greater than the
   number of `-1` signs; set it to `0` otherwise, including a one-to-one tie with exactly
   two usable tasks. If any frozen cohort model has fewer than two usable tasks, report
   its row and all task-level reasons but abort the primary Fisher calculation rather
   than shrinking the cohort, imputing a sign, or replacing a task.

6. **Primary statistic.** Form the fixed 2×2 table crossing `trio_member` with
   `positive_majority` over all nine models and report the two-sided Fisher-exact odds
   ratio and p-value. This single Fisher-exact association is the primary statistic.
   Report the table's four counts and the nine binary model outcomes alongside it; do not
   convert this retrospective p-value into a confirmatory claim.

7. **Mandatory full reporting and prohibited flexibility.** Report the **FULL
   model×task sign table** for all nine models and all three auxiliary tasks regardless
   of result, including `NA` cells and reasons. There is **NO task dropping, family
   regrouping, cell substitution, or threshold tuning**. There is also no model
   substitution, predictor revision, alternate majority rule, one-sided Fisher test, or
   secondary task search in this design.

## Exact executor command

The following command reads the existing BENCH matrices and prints one JSON object
containing the complete sign table, model outcomes, fixed 2×2 table, and Fisher result. It
does not write a result file.

**executor: user/Claude**

```bash
python - <<'PY'
import json
import math
import sys
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, "stage_b")
from analyze_universality import load_cells, SEAL

cohort = [
    "Llama-3.2-3B-Instruct-4bit",
    "Llama-3.1-8B-Instruct-4bit",
    "Mistral-7B-Instruct-v0.3-4bit",
    "Mistral-Nemo-Instruct-2407-4bit",
    "Phi-4-mini-instruct-4bit",
    "Qwen2.5-7B-Instruct-4bit",
    "Qwen3-1.7B-4bit",
    "Qwen3-8B-4bit",
    "gemma-3-4b-it-4bit",
]
trio = {
    "Mistral-7B-Instruct-v0.3-4bit",
    "Mistral-Nemo-Instruct-2407-4bit",
    "Qwen2.5-7B-Instruct-4bit",
}
tasks = ["anli_r2", "halueval_dialogue", "halueval_summarization"]
fixed_cell = "fusion_rank_mean_geom"
cells = load_cells("stage_b/profiles_bench")
sign_table = []
outcomes = []

for model in cohort:
    row = {"model": model, "trio_member": int(model in trio), "tasks": {}}
    signs = []
    for task in tasks:
        cell = cells.get((model, task))
        result = {"sign": None, "usable": False, "reason": None}
        if cell is None:
            result["reason"] = "missing matrix"
        else:
            indices = [i for i, panel_cell in enumerate(cell["panel"])
                       if panel_cell[2] == fixed_cell]
            if len(indices) != 1:
                result["reason"] = f"fixed-cell count={len(indices)}"
            else:
                x = np.asarray(cell["M"][:, indices[0]], dtype=np.float64)
                y = np.asarray(cell["y"])
                finite = np.isfinite(x) & np.isfinite(y)
                if int(finite.sum()) < 4:
                    result["reason"] = "fewer than four finite pairs"
                elif len(np.unique(y[finite])) < 2:
                    result["reason"] = "fewer than two labels"
                else:
                    auc, sign, _ = SEAL._score_candidate(x[finite], y[finite])
                    if sign not in (-1, 1) or not math.isfinite(float(auc)):
                        result["reason"] = "nonfinite or zero fitted sign"
                    else:
                        result.update(sign=int(sign), usable=True, reason=None)
                        signs.append(int(sign))
        row["tasks"][task] = result
    plus = signs.count(1)
    minus = signs.count(-1)
    positive_majority = None if len(signs) < 2 else int(plus > minus)
    row["n_usable"] = len(signs)
    row["positive_majority"] = positive_majority
    sign_table.append(row)
    outcomes.append({
        "model": model,
        "trio_member": int(model in trio),
        "positive_majority": positive_majority,
    })

abort = any(row["positive_majority"] is None for row in outcomes)
report = {
    "status": "DESIGNED-RETROSPECTIVE; NOT A FINDING; NOT BLIND-CONFIRMATORY",
    "discovery_task_excluded": "halueval_qa",
    "auxiliary_tasks": tasks,
    "fixed_cell": fixed_cell,
    "sign_table": sign_table,
    "model_outcomes": outcomes,
    "primary_aborted": abort,
}
if abort:
    report["abort_reason"] = "at least one frozen model has fewer than two usable auxiliary tasks"
else:
    a = sum(r["trio_member"] == 1 and r["positive_majority"] == 1 for r in outcomes)
    b = sum(r["trio_member"] == 1 and r["positive_majority"] == 0 for r in outcomes)
    c = sum(r["trio_member"] == 0 and r["positive_majority"] == 1 for r in outcomes)
    d = sum(r["trio_member"] == 0 and r["positive_majority"] == 0 for r in outcomes)
    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    report["fisher_exact"] = {
        "layout": [["trio+ and majority+", "trio+ and majority-not+"],
                   ["trio- and majority+", "trio- and majority-not+"]],
        "table": table,
        "alternative": "two-sided",
        "odds_ratio": None if not math.isfinite(float(odds_ratio)) else float(odds_ratio),
        "odds_ratio_nonfinite": not math.isfinite(float(odds_ratio)),
        "p_value": float(p_value),
    }
print(json.dumps(report, indent=2, sort_keys=False))
PY
```

hash + commit this design before any executor runs the screen; a real finding requires fresh-task/fresh-data registered replication with the trio predictor sealed before labels are examined.
