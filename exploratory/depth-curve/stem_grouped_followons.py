"""Stem-grouped re-run of the depth-lane analyses the 2026-10-08 rescore did not cover — UNREGISTERED.

Covers E7, E8 and E1'' (score_grid_b.py) and the depth-coverage, three-instrument and
attention-family follow-ons. Each is re-run with STEM-GROUPED folds: HaluEval's two rows per
`source_idx` stem, and ANLI's five repeated premises, always land in the same fold. Nothing
else changes — permutation nulls, bootstraps, seeds and thresholds are the banked machinery.

Safety: the scorers write fixed filenames next to themselves, so this script copies the lane
into a temporary directory, swaps the fold builder at runtime in every module that binds it
(no source file is edited), runs there, and copies the outputs into ./stem_grouped_followons/.
It refuses to run if that directory exists (single look). No banked file can be overwritten.

Also runs two fold-draw checks for results that moved under the grouped map: the P9 outcome
(Llama-3.1-8B/anli, < 5 qualifying mid-blocks in a majority of folds) and the cross-fitted
peak of Llama-3.3-70B/anli, each over 100 row-stratified and 100 stem-grouped fold maps.
These separate a grouping effect from ordinary fold-draw sensitivity.

Not covered, by design: colocation and instrument-redundancy (no folds), the stability
diagnostic (PARKED).

Run: ../../.venv/bin/python stem_grouped_followons.py   (numpy 2.0.2; an unpatched control
in this environment reproduces DEPTH_COVERAGE/THREE_INSTRUMENT/ATTENTION_FAMILY exactly)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stem_grouped_followons")
OUTPUTS = ("GRID_B_RESULTS.json", "GRID_B_RESULTS.md", "DEPTH_COVERAGE.json",
           "THREE_INSTRUMENT.json", "ATTENTION_FAMILY.json", "FOLD_DRAW_CHECKS.json")
SKIP = {"_gridb_logs", "_vnorm_logs", "__pycache__", "manifests_gridb",
        "stem_grouped_followons"}

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import rescore_grid_a as RG
import depth_coverage as DC
from stem_grouped_sensitivity import load_groups, units_by_type, make_group_folds
import score_grid_b as SB
from depth_score import load_cell

LABELS = {t: SB.load_task_labels(t) for t in SB.TASKS}
UNITS = {t: units_by_type(load_groups(t, LABELS[t]), LABELS[t]) for t in SB.TASKS}
ORIGINAL = RG.make_folds


def grouped_folds(y, rng):
    y = np.asarray(y, dtype=np.int64)
    match = [t for t in SB.TASKS if np.array_equal(LABELS[t], y)]
    assert len(match) == 1, "label vector does not identify a frozen task"
    fold_of = make_group_folds(UNITS[match[0]], len(y), rng)
    for f in range(RG.K_FOLDS):
        assert np.bincount(y[fold_of == f], minlength=2).tolist() == [20, 20]
    return fold_of


RG.make_folds = SB.make_folds = DC.make_folds = grouped_folds
import three_instrument as TI   # noqa: E402  (reads DC.make_folds by attribute)
import attention_family as AF   # noqa: E402

t0 = time.time()
sys.argv = ["score_grid_b.py", "--npz-dir", "npz/depth_grid_b"]
SB.main(); DC.main(); TI.main(); AF.main()
print(f"grouped re-runs done {time.time() - t0:.0f}s", flush=True)


def inner_design(task, fold):
    return [np.stack([RG._rng(task, "inner-env", f).permutation(int((fold != f).sum()))
                      for _ in range(RG.NPERM_INNER)]) for f in range(RG.K_FOLDS)]


def fold_maps(task, tag, m):
    y = LABELS[task]
    return {"row": ORIGINAL(y, RG._rng(task, tag + "-row", m)),
            "group": make_group_folds(UNITS[task], len(y), RG._rng(task, tag + "-group", m))}


checks = {"n_maps_per_arm": 100}
task = "anli_r1"
y = LABELS[task]
prim, L, _meta, _p = SB.load_cell_b("npz/depth_grid_b", task, "Llama-3.1-8B-Instruct", y)
lo, hi = int(np.ceil(0.4 * L)), int(np.floor(0.9 * L))
p9 = {"row": [], "group": []}
for m in range(100):
    for arm, fold in fold_maps(task, "p9", m).items():
        quals, _ = SB.fold_qual_masks(prim, y, fold, inner_design(task, fold))
        counts = [int(q[lo:hi + 1].sum()) for q in quals]
        p9[arm].append({"counts": counts, "hit": sum(c < 5 for c in counts) >= 3})
checks["P9_llama31_8b_anli"] = {
    arm: {"hits": sum(r["hit"] for r in v),
          "count_median": float(np.median([c for r in v for c in r["counts"]])),
          "count_range": [int(min(c for r in v for c in r["counts"])),
                          int(max(c for r in v for c in r["counts"]))]}
    for arm, v in p9.items()}
scores, _y, metrics, _m = load_cell("npz/depth_curve", task, "Llama-3.3-70B-Instruct")
prim = scores[:, :, metrics.index("final_js_no_bos")]
peaks = {"row": [], "group": []}
for m in range(100):
    for arm, fold in fold_maps(task, "pk", m).items():
        cf = RG.crossfit_cell(prim, y, fold, inner_design(task, fold))
        if cf["ok"]:
            peaks[arm].append(float(np.median([fd["peak"] for fd in cf["folds"]])))
checks["peak_cf_llama33_70b_anli"] = {
    arm: {"n_defined": len(v), "median": float(np.median(v)),
          "p5_p95": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
          "share_ge_60": float(np.mean(np.asarray(v) >= 60))}
    for arm, v in peaks.items()}
with open("FOLD_DRAW_CHECKS.json", "w") as fh:
    json.dump(checks, fh, indent=1, allow_nan=False)
print(f"fold-draw checks done {time.time() - t0:.0f}s", flush=True)
'''


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING TO RUN: {OUT} exists — single look.")
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "depth-curve")
        shutil.copytree(HERE, work, ignore=lambda d, names: [
            n for n in names if n in SKIP or n.endswith((".log", ".png"))])
        for name in OUTPUTS:
            path = os.path.join(work, name)
            if os.path.exists(path):
                os.remove(path)
        subprocess.run([sys.executable, "-W", "ignore", "-c", WORKER], cwd=work, check=True)
        os.makedirs(OUT)
        for name in OUTPUTS:
            shutil.copy2(os.path.join(work, name), os.path.join(OUT, name))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
