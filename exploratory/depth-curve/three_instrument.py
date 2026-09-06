"""Three-instrument depth analysis — the first read that is NOT one instrument short.

Candidate #16 has been two-instrument since 2026-08-29 because
`final_v_norm_lastq_weighted` had no per-layer data. The additive v-norm capture channel
(commit 1b9da4b) ran 2026-09-02 and wrote a `<slug>.vnorm.npz` sidecar for all 17 usable
cells. This script joins those sidecars to the registered depth artifacts and redoes both
prior reads with the third instrument present.

ROW-IDENTITY JOIN IS ENFORCED, NOT ASSUMED. The sidecar mirrors `labels`, `gen_token_ids`,
`commit_p` and `yes_no`. The three integer columns must match BIT-EXACTLY; `commit_p` is a
float and is matched at a calibrated tolerance (see COMMIT_P_ATOL, which documents the
measurement that set it). Any cell failing either test is REFUSED, not repaired. This is the direct fix for the finding that `sample_idx` was a vacuous guard
(it is arange(n) in every file and cannot detect a permutation). `sample_idx` is therefore
NOT used as evidence of anything here.

TWO PARTS
  A. CO-LOCATION, three-way (descriptive, in-sample, matches colocation_analysis.py
     conventions): per-block sign-free AUROC, peak block, pairwise curve correlation,
     cross-evaluation at each other instrument's peak, and mutual blindness at the
     registered 0.65 bar.
  B. DEPTH COVERAGE, three-instrument (cross-fitted, matches depth_coverage.py exactly):
     the fixed panel grows to 3 instruments x 3 rungs and the targetable set grows to 3L
     columns. Two-instrument arms are recomputed alongside so the delta attributable to the
     third instrument is readable directly.

SCOPE. Descriptive, NOT registered; cannot upgrade or downgrade E1/E5/E6. Grids never
pooled. Torch/Modal lane, NON-byte-comparable with the sealed MLX panels. The v-norm channel
is unregistered by construction (`registered: false` in its own meta).

Usage: python3 three_instrument.py  ->  THREE_INSTRUMENT.json
"""

import glob
import hashlib
import json
import os

import numpy as np

import depth_coverage as dc

HERE = dc.HERE
VN = "final_v_norm_lastq_weighted"
INSTR = ("js", "bos", "vnorm")
MIRROR_EXACT = ("labels", "gen_token_ids", "yes_no")   # integer columns: bit-exact required
MIRROR_TOL = "commit_p"                                 # float column: see COMMIT_P_ATOL
# CALIBRATED TOLERANCE, not a loosened guard. Measured on this data:
#   * `commit_p` is the ONLY mirrored column with row-level resolution — 200 distinct values
#     per cell, against 2 for labels, 2 for yes_no and 3 for gen_token_ids. Dropping it would
#     leave no column able to detect a permutation, which is the whole point of the check.
#   * Exact equality refused 5 of 17 cells. The disagreement is 1-2 rows per cell at
#     |delta| <= 2.220e-16 — one to two ULP of a float64 near 1.0, i.e. floating-point
#     non-determinism between the registered run and the sidecar recomputation.
#   * The SMALLEST gap between two genuinely distinct commit_p values in a cell is 5.936e-08,
#     and a permutation moves a typical value by ~1e-3.
# 1e-12 sits ~4 orders above the observed artifact and ~5 orders below the smallest real gap.
# A permuted row would exceed it by roughly nine orders of magnitude.
COMMIT_P_ATOL = 1e-12
QUAL = 0.65

TREES = (("A", "npz/depth_curve", "npz_vnorm/depth_curve_vnorm"),
         ("B", "npz/depth_grid_b", "npz_vnorm/depth_grid_b_vnorm"))


def load_cell(depth_path, vnorm_path):
    """Join a registered depth artifact to its v-norm sidecar. Refuses on any mismatch."""
    d = np.load(depth_path, allow_pickle=True)
    v = np.load(vnorm_path, allow_pickle=True)
    for k in MIRROR_EXACT + (MIRROR_TOL,):
        if k not in d or k not in v:
            return None, f"mirror column {k} absent"
    for k in MIRROR_EXACT:
        if not np.array_equal(np.asarray(d[k]), np.asarray(v[k])):
            return None, f"ROW IDENTITY MISMATCH on {k} (exact) — join refused"
    ca = np.asarray(d[MIRROR_TOL], float)
    cb = np.asarray(v[MIRROR_TOL], float)
    if ca.shape != cb.shape:
        return None, f"ROW COUNT MISMATCH on {MIRROR_TOL} — join refused"
    worst = float(np.max(np.abs(ca - cb))) if ca.size else 0.0
    if worst > COMMIT_P_ATOL:
        return None, (f"ROW IDENTITY MISMATCH on {MIRROR_TOL}: max |delta| {worst:.3e} "
                      f"> atol {COMMIT_P_ATOL:.0e} — join refused")
    dmeta, vmeta = json.loads(str(d["meta"])), json.loads(str(v["meta"]))
    if int(dmeta["n_layers"]) != int(vmeta["n_layers"]):
        return None, "n_layers disagree between artifact and sidecar"
    diag = vmeta.get("additive_capture_diagnostics", {})
    if diag.get("n_nonfinite", 0) or diag.get("n_blocks_failed", 0):
        return None, f"capture diagnostics non-clean: {diag.get('n_nonfinite')} nonfinite"

    metrics = json.loads(str(d["metrics"]))
    vmetrics = json.loads(str(v["v_norm_metrics"]))
    L = int(dmeta["n_layers"])
    S, V = d["scores"], v["v_norm_scores"]
    cols = {
        "js": S[:, :, metrics.index(dc.JS)],
        "bos": S[:, :, metrics.index(dc.BOS)],
        "vnorm": V[:, :, vmetrics.index(VN)],   # BY NAME — index 0 is v_norm_bos, not this
    }
    X = np.column_stack([cols[k] for k in INSTR]).astype(float)
    if not np.isfinite(X).all():
        return None, "non-finite block scores after join"
    return {"X": X, "y": d["labels"].astype(int), "L": L,
            "n_scored": diag.get("n_scored"), "mode": vmeta.get("v_norm_capture_mode"),
            "commit_p_max_delta": worst}, None


def curves(X, y, L):
    """Per-block sign-free AUROC for each instrument (in-sample, matches co-location)."""
    R = dc.midrank_columns(X)
    a = dc.auroc_from_ranks(R, y)
    a = np.maximum(a, 1.0 - a)
    return {k: a[i * L:(i + 1) * L] for i, k in enumerate(INSTR)}


def main():
    task_names, files = set(), []
    for grid, dtree, vtree in TREES:
        for p in sorted(glob.glob(os.path.join(HERE, dtree, "*", "*.depth.npz"))):
            task = p.split(os.sep)[-2]
            model = os.path.basename(p).replace(".depth.npz", "")
            vp = os.path.join(HERE, vtree, task, model + ".vnorm.npz")
            if os.path.exists(vp):
                task_names.add(task)
                files.append((grid, task, model, p, vp))
    ti_of = {t: i for i, t in enumerate(sorted(task_names))}

    cells, refused = [], []
    for grid, task, model, dp, vp in files:
        cell, why = load_cell(dp, vp)
        if cell is None:
            refused.append({"grid": grid, "task": task, "model": model, "why": why})
            continue
        X, y, L = cell["X"], cell["y"], cell["L"]
        rungs = [L // 2, L - 2, L - 1]

        # ---- PART A: three-way co-location (in-sample, descriptive)
        cv = curves(X, y, L)
        peak = {k: int(np.argmax(cv[k])) for k in INSTR}
        pk_auc = {k: round(float(cv[k][peak[k]]), 4) for k in INSTR}
        corr, cross, blind = {}, {}, []
        for i, a in enumerate(INSTR):
            for b in INSTR[i + 1:]:
                corr[f"{a}~{b}"] = round(float(np.corrcoef(cv[a], cv[b])[0, 1]), 4)
                ab = float(cv[a][peak[b]])       # a evaluated at b's peak
                ba = float(cv[b][peak[a]])
                cross[f"{a}@{b}peak"] = round(ab, 4)
                cross[f"{b}@{a}peak"] = round(ba, 4)
                if ab < QUAL and ba < QUAL:
                    blind.append(f"{a}~{b}")

        # ---- PART B: cross-fitted arms, 3-instrument vs 2-instrument
        fold_of = dc.make_folds(y, np.random.default_rng((dc.SEED, ti_of[task])))
        point, state = crossfit3(X, y, fold_of, rungs, L, want_state=True)
        cond = conditional_ci(state, ti_of[task], L)

        cells.append({
            "grid": grid, "task": task, "model": model, "n_layers": L,
            "n_rows": int(len(y)), "rungs": rungs,
            "vnorm_cells_scored": cell["n_scored"], "capture_mode": cell["mode"],
            "commit_p_max_delta": cell["commit_p_max_delta"],
            "peak_block": peak, "peak_auroc": pk_auc,
            "curve_pearson": corr, "cross_eval": cross,
            "mutually_blind_pairs": blind,
            "arms_foldmean": {k: round(v, 4) for k, v in point.items()},
            "contrasts": cond,
        })
        print(f"{grid} {task:11} {model[:26]:26} peaks js{peak['js']:>3} bos{peak['bos']:>3} "
              f"vn{peak['vnorm']:>3} | blind {len(blind)}/3 | "
              f"t1_3 {point['target1_3i']:.3f} vs t1_2 {point['target1_2i']:.3f}")

    out = {
        "config": {
            "instruments": list(INSTR), "third_instrument_metric": VN,
            "k_folds": dc.K_FOLDS, "n_boot": dc.N_BOOT, "seed": dc.SEED,
            "qualifying_bar": QUAL,
            "row_identity": "ENFORCED: labels+gen_token_ids+yes_no bit-exact; commit_p at "
                            f"atol {COMMIT_P_ATOL:.0e} (only column with row-level resolution, "
                            "200 distinct values; artifact 2.2e-16 vs smallest real gap 5.9e-8); "
                            "sample_idx deliberately NOT used (vacuous)",
            "commit_p_atol": COMMIT_P_ATOL,
            "note": "descriptive, NOT registered; grids never pooled; v-norm channel unregistered",
        },
        "provenance": {"script_sha256": _sha(os.path.abspath(__file__)),
                       "depth_coverage_sha256": _sha(os.path.join(HERE, "depth_coverage.py")),
                       "numpy": np.__version__},
        "refused_cells": refused,
        "cells": cells,
    }
    with open(os.path.join(HERE, "THREE_INSTRUMENT.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    summarise(cells, refused)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


ARMS3 = ("fixed9", "fixed6_2i", "rung_best1_3i", "rung_best1_2i",
         "target1_3i", "target1_2i", "target3", "target2_2i", "target1_vnorm")


def crossfit3(X, y, fold_of, rungs, L, want_state=False):
    """Same convention as depth_coverage.crossfit, with a third instrument block.

    X is [n, 3L] = [js | bos | vnorm]. Two-instrument arms are restricted to the first 2L
    columns so the 3i-vs-2i contrast isolates the third instrument exactly.
    """
    r_js = list(rungs)
    r_bos = [L + r for r in rungs]
    r_vn = [2 * L + r for r in rungs]
    nine, six = r_js + r_bos + r_vn, r_js + r_bos

    per_fold, acc = [], {k: [] for k in ARMS3}
    for f in range(dc.K_FOLDS):
        tr, ho = fold_of != f, fold_of == f
        y_tr, y_ho = y[tr], y[ho]
        if y_tr.min() == y_tr.max() or y_ho.min() == y_ho.max():
            return None, None
        X_tr, X_ho = X[tr], X[ho]
        a_tr = dc.auroc_from_ranks(dc.midrank_columns(X_tr), y_tr)
        dirs = np.where(a_tr >= 0.5, 1.0, -1.0)
        A = np.maximum(a_tr, 1.0 - a_tr)

        cols_of = {
            "fixed9": nine,
            "fixed6_2i": six,
            "rung_best1_3i": [nine[int(np.argmax(A[nine]))]],
            "rung_best1_2i": [six[int(np.argmax(A[six]))]],
            "target1_3i": [int(np.argmax(A))],
            "target1_2i": [int(np.argmax(A[:2 * L]))],
            "target1_vnorm": [2 * L + int(np.argmax(A[2 * L:]))],
            "target3": [int(np.argmax(A[:L])),
                        L + int(np.argmax(A[L:2 * L])),
                        2 * L + int(np.argmax(A[2 * L:]))],
            "target2_2i": [int(np.argmax(A[:L])), L + int(np.argmax(A[L:2 * L]))],
        }
        fs = {"y_ho": y_ho, "score": {}, "cols": cols_of}
        for arm, cols in cols_of.items():
            cal = np.column_stack([
                dc.train_percentile(dirs[c] * X_tr[:, c], dirs[c] * X_ho[:, c]) for c in cols
            ]).mean(axis=1)
            # single-column arms score on RAW oriented values (depth_coverage R2 finding 2)
            score = (dirs[cols[0]] * X_ho[:, cols[0]]) if len(cols) == 1 else cal
            fs["score"][arm] = score
            acc[arm].append(dc.auroc(score, y_ho))
        per_fold.append(fs)
    return {k: float(np.mean(v)) for k, v in acc.items()}, (per_fold if want_state else None)


CONTRASTS3 = (("target1_3i", "target1_2i"),        # PRIMARY: what the 3rd instrument adds
              ("target1_3i", "rung_best1_3i"),
              ("rung_best1_3i", "rung_best1_2i"),
              ("fixed9", "fixed6_2i"),
              ("fixed9", "rung_best1_3i"),
              ("target3", "target2_2i"),
              ("target3", "rung_best1_3i"),
              ("target1_vnorm", "rung_best1_2i"))


def conditional_ci(state, ti, L):
    """Evaluation-row-only interval, conditional on the frozen cross-fit (as depth_coverage)."""
    rng = np.random.default_rng((dc.SEED, ti, L, 3))
    strata = [[np.flatnonzero(fs["y_ho"] == c) for c in (0, 1)] for fs in state]
    draws_acc = {k: [] for k in ARMS3}
    for _ in range(dc.N_BOOT):
        draws = [np.concatenate([s[rng.integers(0, len(s), len(s))] for s in st]) for st in strata]
        for arm in ARMS3:
            draws_acc[arm].append(np.mean([
                dc.auroc(state[f]["score"][arm][draws[f]], state[f]["y_ho"][draws[f]])
                for f in range(dc.K_FOLDS)]))
    point = {k: float(np.mean([dc.auroc(fs["score"][k], fs["y_ho"]) for fs in state]))
             for k in ARMS3}
    out = {}
    for a, b in CONTRASTS3:
        v = np.array(draws_acc[a]) - np.array(draws_acc[b])
        ci = [round(float(np.percentile(v, 5)), 4), round(float(np.percentile(v, 95)), 4)]
        d = point[a] - point[b]
        out[f"{a}-{b}"] = {"delta": round(d, 4), "ci90_conditional": ci,
                           "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                           "winner": "a" if d > 0 else ("b" if d < 0 else "tie")}
    return out


def summarise(cells, refused):
    print(f"\ncells joined: {len(cells)} | refused: {len(refused)}")
    for r in refused:
        print("  REFUSED", r)
    for grid in ("A", "B"):
        g = [c for c in cells if c["grid"] == grid]
        if not g:
            continue
        print(f"\ngrid {grid} (n={len(g)}):")
        for a, b in CONTRASTS3:
            k = f"{a}-{b}"
            ds = [c["contrasts"][k]["delta"] for c in g]
            ex = sum(c["contrasts"][k]["excludes_zero"] for c in g)
            w = sum(d > 0 for d in ds)
            print(f"  {k:32} wins {w:2}/{len(g)} | median {np.median(ds):+.4f} | excl0 {ex:2}/{len(g)}")
        med = {k: round(float(np.median([c['arms_foldmean'][k] for c in g])), 4) for k in ARMS3}
        print("  median fold-mean:", med)
        nb = [len(c["mutually_blind_pairs"]) for c in g]
        print(f"  mutually blind pairs per cell: median {np.median(nb):.1f}, "
              f"cells with >=1: {sum(1 for x in nb if x)}/{len(g)}")
        for pair in ("js~bos", "js~vnorm", "bos~vnorm"):
            rs = [c["curve_pearson"][pair] for c in g]
            print(f"  curve r {pair:11} median {np.median(rs):+.3f}")


if __name__ == "__main__":
    main()
