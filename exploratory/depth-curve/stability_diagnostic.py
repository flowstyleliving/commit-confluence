"""Post-hoc diagnostic — does SELECTION STABILITY predict the depth-targeting benefit?

Reads DEPTH_COVERAGE.json only. No GPU, no npz, no refit.

MOTIVATION. The primary contrast `target1 - rung_best1` is strongly positive in grid A
(median +0.119) and near-zero in grid B (median +0.011) with cells pointing BOTH ways.
This asks whether the split is explained by how consistently the per-fold argmax lands on
the same (instrument, block).

DEFINITION. For each cell, `selected` holds the 5 per-fold training picks. Stability =
(count of the MODAL (instrument, block) pair) / 5. A cell is STABLE at >= 4/5.

KNOWN WEAKNESSES — all must be reported with any use of this output (Codex R2 finding 3):
  1. POST-HOC. The stratification AND the 4/5 threshold were chosen after seeing the
     deltas. This describes these 17 cells; it is not a registered test and no p-value
     here would be honest.
  2. THRESHOLD-SENSITIVE. The headline "stable cells never lose" holds at 4/5 and 5/5 but
     BREAKS at 3/5, where two negative cells enter the stable group. Only the ordering of
     the group medians survives all three thresholds. Sensitivity is printed below; quote
     it, never the 4/5 line alone.
  3. PARTLY MECHANICAL. An unstable argmax means a noisier selected column, which lowers
     held-out AUROC almost by construction. The floor is not logically forced — a selector
     can stably pick a column that generalises badly — but the correlation mostly is.
  4. ENDOGENOUS. Each row shapes four folds' selections and the fifth fold's held-out
     AUROC, so stability and delta are computed from the same cross-fit.
  5. GRID-CONFOUNDED. At 4/5 the stable group is 7 grid-A / 3 grid-B and the unstable group
     1 grid-A / 6 grid-B, so the pooled correlation partly restates the grid difference.
     Per-grid figures are printed below.
  6. ONE DEGENERATE MEMBER. The stable group's minimum (Qwen2.5-7B/halueval, exactly 0.000)
     is a cell whose peak block IS the N-2 rung, so both arms are the same column. "All
     >= 0" is true but includes a definitional zero; "all stable cells benefit" is false.
  7. n = 17.

Usage: python3 stability_diagnostic.py
"""

import collections
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = "target1-rung_best1"
STABLE_AT = 4  # of 5 folds


def spearman(a, b):
    """Spearman rho via Pearson on midranks (ties averaged)."""
    def mr(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = mr(a), mr(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    doc = json.load(open(os.path.join(HERE, "DEPTH_COVERAGE.json")))
    rows = []
    for c in doc["cells"]:
        if not c.get("usable", True):
            continue
        con = c["contrasts"].get(PRIMARY)
        if con is None:
            continue
        picks = [(s["instrument"], s["block"]) for s in c["selected"]]
        modal_n = collections.Counter(picks).most_common(1)[0][1]
        rows.append(dict(
            grid=c["grid"], task=c["task"], model=c["model"], n_rows=c["n_rows"],
            delta=float(con["delta"]),
            ci=con["ci90_conditional"],
            stability=modal_n / 5.0,
            margin=float(np.median([s["train_top_margin"] for s in c["selected"]])),
        ))

    print(f"{'gr':2} {'task':11} {'model':28} {'N':>3} {'stab':>4} {'margin':>7} {'delta':>8}  CI90 cond")
    for r in sorted(rows, key=lambda r: (-r["stability"], r["delta"])):
        lo, hi = r["ci"]
        print(f"{r['grid']:2} {r['task']:11} {r['model'][:28]:28} {r['n_rows']:3} "
              f"{r['stability']:4.1f} {r['margin']:7.4f} {r['delta']:+8.4f}  [{lo:.4f}, {hi:.4f}]")

    # --- threshold sensitivity (R2 finding 3, weakness 2)
    print("\nTHRESHOLD SENSITIVITY — the 4/5 split is post-hoc; all three are shown:")
    sens = {}
    for thr in (3, 4, 5):
        st_ = [r["delta"] for r in rows if r["stability"] * 5 >= thr]
        un_ = [r["delta"] for r in rows if r["stability"] * 5 < thr]
        sens[f"{thr}/5"] = dict(
            stable_n=len(st_), stable_median=round(float(np.median(st_)), 4),
            stable_min=round(float(min(st_)), 4), stable_all_nonneg=bool(all(x >= 0 for x in st_)),
            unstable_n=len(un_),
            unstable_median=(round(float(np.median(un_)), 4) if un_ else None))
        um = f"{np.median(un_):+.4f}" if un_ else "   n/a"
        print(f"  >= {thr}/5 folds  stable n={len(st_):2} median {np.median(st_):+.4f} "
              f"min {min(st_):+.4f} all>=0 {str(all(x >= 0 for x in st_)):5} | "
              f"unstable n={len(un_):2} median {um}")

    st = [r for r in rows if r["stability"] * 5 >= STABLE_AT]
    un = [r for r in rows if r["stability"] * 5 < STABLE_AT]
    for name, grp in ((f"STABLE  (>={STABLE_AT}/5)", st), ("UNSTABLE", un)):
        d = [r["delta"] for r in grp]
        ga = sum(1 for r in grp if r["grid"] == "A")
        print(f"\n{name} n={len(grp):2} | median {np.median(d):+.4f} | min {min(d):+.4f} "
              f"| max {max(d):+.4f} | all>=0: {all(x >= 0 for x in d)} "
              f"| grid A/B: {ga}/{len(grp)-ga}")

    # --- per-grid stratification (R2 finding 3, weakness 5)
    print("\nPER-GRID (the pooled correlation partly restates the grid difference):")
    for grid in ("A", "B"):
        gr = [r for r in rows if r["grid"] == grid]
        gs = [r["delta"] for r in gr if r["stability"] * 5 >= STABLE_AT]
        gu = [r["delta"] for r in gr if r["stability"] * 5 < STABLE_AT]
        f_ = lambda v: f"n={len(v)} median {np.median(v):+.4f}" if v else "n=0"
        print(f"  grid {grid}: stable {f_(gs)} | unstable {f_(gu)} | "
              f"rho {spearman([r['stability'] for r in gr], [r['delta'] for r in gr]):+.3f}")

    neg = [(r["grid"], r["model"], r["task"], round(r["delta"], 4), r["stability"])
           for r in rows if r["delta"] < 0]
    print("\nnegative-delta cells:", neg)

    m = [r["margin"] for r in rows]
    print(f"\nSpearman(stability, delta) = {spearman([r['stability'] for r in rows], [r['delta'] for r in rows]):+.3f}")
    print(f"Spearman(margin, delta)    = {spearman(m, [r['delta'] for r in rows]):+.3f}")
    print(f"train top-vs-runnerup margin: median {np.median(m):.4f}, range [{min(m):.4f}, {max(m):.4f}]")
    degenerate = [r for r in rows if r["delta"] == 0.0]
    if degenerate:
        print("\nDEGENERATE cells (both arms are the SAME column; delta is definitional, "
              "not a measured tie):")
        for r in degenerate:
            print(f"  {r['grid']} {r['task']} {r['model']}")

    print(f"\nn cells = {len(rows)} (post-hoc; see module docstring weaknesses 1-7)")

    # --- provenance (R2 finding 6)
    src = os.path.join(HERE, "DEPTH_COVERAGE.json")
    out = {
        "provenance": {
            "diagnostic_sha256": _sha256(os.path.abspath(__file__)),
            "depth_coverage_json_sha256": _sha256(src),
            "numpy": np.__version__,
        },
        "threshold_reported": f"{STABLE_AT}/5",
        "threshold_sensitivity": sens,
        "spearman_stability_delta_midrank": round(
            spearman([r["stability"] for r in rows], [r["delta"] for r in rows]), 4),
        "caveats": "post-hoc, threshold-sensitive, partly mechanical, endogenous, "
                   "grid-confounded, one degenerate member; see module docstring",
        "cells": [{k: r[k] for k in ("grid", "task", "model", "stability", "delta", "ci")}
                  for r in rows],
    }
    with open(os.path.join(HERE, "STABILITY_DIAGNOSTIC.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote STABILITY_DIAGNOSTIC.json")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
