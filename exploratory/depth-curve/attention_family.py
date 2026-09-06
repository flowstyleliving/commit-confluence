"""All seven ACE attention metrics at depth — does "two signals" survive the full family?

[[results/three-instrument-2026-09-06]] examined THREE metrics (`js_no_bos`, `bos_mass`,
`v_norm_lastq_weighted`) and found two signals: bos and v_norm nearly one curve, js apart.
But the deployed ACE attention panel is `ATTENTION_PANEL_T0_WITH_V_NORMS` = 3 layers x SEVEN
metrics (sealed `pri_calibrator.ATTENTION_METRICS` + `ATTENTION_METRICS_V_NORMS`). Four of the
seven had never been looked at down the stack. This script scores all seven.

The registered depth artifacts carry the four weight-only metrics per block; the 2026-09-02
sidecar carries the three value-norm metrics. Together that is 7/7 with depth data.

NOTE ON `fixed21`. Seven metrics at the three panel rungs is the SAME COLUMN SET as the
deployed ACE t=0 panel. It is still NOT the deployed arm — house fusion locks signs from
sealed-era per-cell specs, ranks the whole cohort, and the production calibrator SELECTS among
cells rather than equal-weight fusing them. `fixed21` is a local construction with a matching
column set, and must not be described as the deployed aggregate.

The three-metric subset is recomputed here so the prior page's numbers reproduce as a
cross-check.

Scope: descriptive, NOT registered; grids never pooled; torch/Modal, NON-byte-comparable.

Usage: python3 attention_family.py  ->  ATTENTION_FAMILY.json
"""

import glob
import hashlib
import json
import os

import numpy as np

import depth_coverage as dc
import three_instrument as ti

# sealed order: ATTENTION_METRICS + ATTENTION_METRICS_V_NORMS
WEIGHT = ("final_js", "final_js_kv_groups", "final_js_no_bos", "final_bos_mass")
VNORM = ("final_v_norm_bos", "final_v_norm_max", "final_v_norm_lastq_weighted")
METRICS = WEIGHT + VNORM
SHORT = {"final_js": "js", "final_js_kv_groups": "js_kv", "final_js_no_bos": "js_nobos",
         "final_bos_mass": "bos", "final_v_norm_bos": "vn_bos",
         "final_v_norm_max": "vn_max", "final_v_norm_lastq_weighted": "vn_lastq"}
NAMES = [SHORT[m] for m in METRICS]
M = len(METRICS)
TRIO = ("js_nobos", "bos", "vn_lastq")     # the 2026-09-06 three-metric subset
QUAL = 0.65


def load(depth_path, vnorm_path):
    d = np.load(depth_path, allow_pickle=True)
    v = np.load(vnorm_path, allow_pickle=True)
    for k in ti.MIRROR_EXACT:
        if not np.array_equal(np.asarray(d[k]), np.asarray(v[k])):
            return None, f"row identity mismatch on {k}"
    ca, cb = np.asarray(d[ti.MIRROR_TOL], float), np.asarray(v[ti.MIRROR_TOL], float)
    if float(np.max(np.abs(ca - cb))) > ti.COMMIT_P_ATOL:
        return None, "row identity mismatch on commit_p"
    dm = json.loads(str(d["meta"]))
    L = int(dm["n_layers"])
    dmetrics = json.loads(str(d["metrics"]))
    vmetrics = json.loads(str(v["v_norm_metrics"]))
    S, V = d["scores"], v["v_norm_scores"]
    cols = []
    for m in METRICS:
        if m in dmetrics:
            cols.append(S[:, :, dmetrics.index(m)])
        elif m in vmetrics:
            cols.append(V[:, :, vmetrics.index(m)])
        else:
            return None, f"metric {m} present in neither artifact"
    X = np.column_stack(cols).astype(float)
    if not np.isfinite(X).all():
        return None, "non-finite after join"
    return {"X": X, "y": d["labels"].astype(int), "L": L}, None


ARMS = ("fixed21", "fixed9_trio", "rung_best1_7", "rung_best1_3",
        "target1_7", "target1_3", "target7", "target3")


def crossfit(X, y, fold_of, rungs, L, want_state=False):
    """Same convention as depth_coverage.crossfit; 7 metric blocks of width L."""
    blocks = {NAMES[i]: (i * L, (i + 1) * L) for i in range(M)}
    rung_cols = {n: [blocks[n][0] + r for r in rungs] for n in NAMES}
    all21 = [c for n in NAMES for c in rung_cols[n]]
    trio9 = [c for n in TRIO for c in rung_cols[n]]
    trio_cols = np.concatenate([np.arange(*blocks[n]) for n in TRIO])

    per_fold, acc = [], {k: [] for k in ARMS}
    for f in range(dc.K_FOLDS):
        tr, ho = fold_of != f, fold_of == f
        y_tr, y_ho = y[tr], y[ho]
        if y_tr.min() == y_tr.max() or y_ho.min() == y_ho.max():
            return None, None
        X_tr, X_ho = X[tr], X[ho]
        a = dc.auroc_from_ranks(dc.midrank_columns(X_tr), y_tr)
        dirs = np.where(a >= 0.5, 1.0, -1.0)
        A = np.maximum(a, 1.0 - a)

        peak_of = {n: blocks[n][0] + int(np.argmax(A[blocks[n][0]:blocks[n][1]])) for n in NAMES}
        cols_of = {
            "fixed21": all21,
            "fixed9_trio": trio9,
            "rung_best1_7": [all21[int(np.argmax(A[all21]))]],
            "rung_best1_3": [trio9[int(np.argmax(A[trio9]))]],
            "target1_7": [int(np.argmax(A))],
            "target1_3": [int(trio_cols[int(np.argmax(A[trio_cols]))])],
            "target7": [peak_of[n] for n in NAMES],
            "target3": [peak_of[n] for n in TRIO],
        }
        fs = {"y_ho": y_ho, "score": {}, "cols": cols_of,
              "winner_7": NAMES[cols_of["target1_7"][0] // L],
              "winner_3": NAMES[cols_of["target1_3"][0] // L]}
        for arm, cols in cols_of.items():
            cal = np.column_stack([
                dc.train_percentile(dirs[c] * X_tr[:, c], dirs[c] * X_ho[:, c]) for c in cols
            ]).mean(axis=1)
            score = (dirs[cols[0]] * X_ho[:, cols[0]]) if len(cols) == 1 else cal
            fs["score"][arm] = score
            acc[arm].append(dc.auroc(score, y_ho))
        per_fold.append(fs)
    return {k: float(np.mean(v)) for k, v in acc.items()}, (per_fold if want_state else None)


CONTRASTS = (("target1_7", "target1_3"),        # PRIMARY: do the other four metrics add?
             ("rung_best1_7", "rung_best1_3"),
             ("fixed21", "fixed9_trio"),
             ("target7", "target3"),
             ("target1_7", "rung_best1_7"),
             ("target7", "rung_best1_7"),
             ("fixed21", "rung_best1_7"))


def cond_ci(state, ti_, L):
    rng = np.random.default_rng((dc.SEED, ti_, L, 11))
    strata = [[np.flatnonzero(fs["y_ho"] == c) for c in (0, 1)] for fs in state]
    draws = {k: [] for k in ARMS}
    for _ in range(dc.N_BOOT):
        idx = [np.concatenate([s[rng.integers(0, len(s), len(s))] for s in st]) for st in strata]
        for arm in ARMS:
            draws[arm].append(np.mean([dc.auroc(state[f]["score"][arm][idx[f]],
                                                state[f]["y_ho"][idx[f]])
                                       for f in range(dc.K_FOLDS)]))
    point = {k: float(np.mean([dc.auroc(fs["score"][k], fs["y_ho"]) for fs in state]))
             for k in ARMS}
    out = {}
    for a, b in CONTRASTS:
        v = np.array(draws[a]) - np.array(draws[b])
        ci = [round(float(np.percentile(v, 5)), 4), round(float(np.percentile(v, 95)), 4)]
        d = point[a] - point[b]
        out[f"{a}-{b}"] = {"delta": round(d, 4), "ci90_conditional": ci,
                           "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return out


def main():
    files, tasks = [], set()
    for grid, dt, vt in ti.TREES:
        for p in sorted(glob.glob(os.path.join(dc.HERE, dt, "*", "*.depth.npz"))):
            task = p.split(os.sep)[-2]
            model = os.path.basename(p).replace(".depth.npz", "")
            vp = os.path.join(dc.HERE, vt, task, model + ".vnorm.npz")
            if os.path.exists(vp):
                tasks.add(task)
                files.append((grid, task, model, p, vp))
    ti_of = {t: i for i, t in enumerate(sorted(tasks))}

    cells, refused, cormats = [], [], {"A": [], "B": []}
    for grid, task, model, dp, vp in files:
        cell, why = load(dp, vp)
        if cell is None:
            refused.append({"grid": grid, "task": task, "model": model, "why": why})
            continue
        X, y, L = cell["X"], cell["y"], cell["L"]
        rungs = [L // 2, L - 2, L - 1]

        R = dc.midrank_columns(X)
        a = dc.auroc_from_ranks(R, y)
        a = np.maximum(a, 1.0 - a)
        cv = {NAMES[i]: a[i * L:(i + 1) * L] for i in range(M)}
        peak = {n: int(np.argmax(cv[n])) for n in NAMES}
        pauc = {n: round(float(cv[n][peak[n]]), 4) for n in NAMES}
        C = np.corrcoef(np.vstack([cv[n] for n in NAMES]))
        cormats[grid].append(C)
        blind = [f"{NAMES[i]}~{NAMES[j]}" for i in range(M) for j in range(i + 1, M)
                 if cv[NAMES[i]][peak[NAMES[j]]] < QUAL and cv[NAMES[j]][peak[NAMES[i]]] < QUAL]

        fold_of = dc.make_folds(y, np.random.default_rng((dc.SEED, ti_of[task])))
        point, state = crossfit(X, y, fold_of, rungs, L, want_state=True)
        cells.append({
            "grid": grid, "task": task, "model": model, "n_layers": L,
            "peak_block": peak, "peak_auroc": pauc,
            "corr": {f"{NAMES[i]}~{NAMES[j]}": round(float(C[i, j]), 4)
                     for i in range(M) for j in range(i + 1, M)},
            "mutually_blind_pairs": blind, "n_blind": len(blind),
            "winners_7": [fs["winner_7"] for fs in state],
            "winners_3": [fs["winner_3"] for fs in state],
            "arms_foldmean": {k: round(v, 4) for k, v in point.items()},
            "contrasts": cond_ci(state, ti_of[task], L),
        })
        print(f"{grid} {task[:8]:8} {model[:24]:24} blind {len(blind):2}/21 | "
              f"t1_7 {point['target1_7']:.3f} t1_3 {point['target1_3']:.3f} | "
              f"win7 {max(set(cells[-1]['winners_7']), key=cells[-1]['winners_7'].count)}")

    out = {"config": {"metrics": list(METRICS), "trio_subset": list(TRIO),
                      "panel_note": "fixed21 matches the DEPLOYED ACE t=0 column set but is "
                                    "NOT the deployed arm (signs, cohort, selection differ)",
                      "seed": dc.SEED, "n_boot": dc.N_BOOT, "qualifying_bar": QUAL,
                      "registered": False},
           "provenance": {"script_sha256": _sha(os.path.abspath(__file__)),
                          "numpy": np.__version__},
           "refused_cells": refused, "cells": cells}
    for grid in ("A", "B"):
        if cormats[grid]:
            med = np.median(np.stack(cormats[grid]), axis=0)
            out.setdefault("median_correlation", {})[grid] = {
                f"{NAMES[i]}~{NAMES[j]}": round(float(med[i, j]), 4)
                for i in range(M) for j in range(i + 1, M)}
    allc = np.median(np.stack(cormats["A"] + cormats["B"]), axis=0)
    out["median_correlation_all17"] = {f"{NAMES[i]}~{NAMES[j]}": round(float(allc[i, j]), 4)
                                       for i in range(M) for j in range(i + 1, M)}
    with open(os.path.join(dc.HERE, "ATTENTION_FAMILY.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    report(cells, allc)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def report(cells, allc):
    print("\n=== MEDIAN CURVE CORRELATION, all 17 cells ===")
    print("            " + "".join(f"{n:>10}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"{n:>11} " + "".join(f"{allc[i, j]:>10.3f}" for j in range(M)))
    for grid in ("A", "B"):
        g = [c for c in cells if c["grid"] == grid]
        if not g:
            continue
        print(f"\ngrid {grid} (n={len(g)}):")
        for a, b in CONTRASTS:
            k = f"{a}-{b}"
            ds = [c["contrasts"][k]["delta"] for c in g]
            print(f"  {k:28} wins {sum(d>0 for d in ds):2}/{len(g)} | "
                  f"median {np.median(ds):+.4f} | excl0 "
                  f"{sum(c['contrasts'][k]['excludes_zero'] for c in g):2}/{len(g)}")
        med = {k: round(float(np.median([c['arms_foldmean'][k] for c in g])), 4) for k in ARMS}
        print("  median fold-mean:", med)
        w = [x for c in g for x in c["winners_7"]]
        print("  target1_7 fold winners:", {n: w.count(n) for n in NAMES if w.count(n)})
        print(f"  mutually blind pairs (of 21): median {np.median([c['n_blind'] for c in g]):.0f}")


if __name__ == "__main__":
    main()
