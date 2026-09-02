"""Depth-coverage test — does a depth-TARGETED single instrument beat the FIXED-rung arms?

Read-only, no GPU. Operates on the banked depth-curve artifacts (`npz/depth_curve/`,
`npz/depth_grid_b/`), which record per-block scores for two of the three ACE attention
instruments (`final_js_no_bos`, `final_bos_mass`) on the same 200 rows the ACE panel was
scored on.

HYPOTHESIS UNDER TEST (candidate #16, [[results/instrument-colocation-2026-08-29]]): the
fixed ACE attention aggregate may transfer partly through DEPTH COVERAGE rather than through
a shared latent. If so, aiming ONE instrument at a model's own peak block should match or
beat several instruments pinned to the three panel rungs.

Rung mapping is the empirically verified one (DC_DATA_CONTRACT.md line 120): 0-indexed
mid = N//2, last_minus_1 = N-2, final = N-1; depth-curve values at those indices reproduce
the sealed ACE panel's attention marginals to 4 dp.

── NAMING DISCIPLINE (Codex audit 2026-08-31, finding 3) ────────────────────────────────
The fused arms here are HOUSE-STYLE FOLD-LOCAL RANK FUSIONS. They are NOT the deployed ACE
arm, and this file must not be quoted as reproducing it. Differences from
`confluence_calibrator.append_fusion_columns`:
  * house orientations are locked from sealed-era per-cell specs (or a modal fallback);
    here EVERY component sign is fit on the current training folds;
  * house fusion ranks the whole cohort; here components are calibrated per fold;
  * `fixed6` — an equal mean of six attention columns — is CONSTRUCTED HERE. Nothing
    establishes it as an arm the ACE panel deploys.
Likewise `rung_best1` is "best of the six constructed fixed columns", NOT the production
calibrator's selection. Do not call either "production-mirroring".

── ESTIMANDS (Codex audit finding 2) ────────────────────────────────────────────────────
An argmax is non-smooth, so an n-out-of-n bootstrap that reruns the selector is not known to
be valid. Two DIFFERENT questions are therefore reported separately, and the primary one is
the regular estimand:

  CONDITIONAL (primary interval) — an EVALUATION-ROW-ONLY interval, conditional on the
  ENTIRE frozen cross-fit: selected columns, fitted orientations, fitted calibrations and
  the fold map are all held at their real-fit values, and only held-out rows are resampled
  within (fold x label) strata. The statistic contains no argmax, so the percentile
  bootstrap is regular here. It charges NOTHING for column selection, orientation fitting,
  fold assignment, or training-sample variation (R2 finding 5).

  FULL-ALGORITHM (secondary, both conventions) — "how well does the whole fit-and-select
  procedure do?" The entire cross-fit including selection is rerun per resample. Reported as
  percentile AND basic/mirrored intervals, because the audit showed classifications flip
  between conventions; disagreement between them is emitted per contrast as a warning flag.

── TIE RULE (Codex audit finding 1) ─────────────────────────────────────────────────────
Ranks are MIDRANKS (ties share their mean rank), matching the house `_rank01`. The prior
version used stable ordinal ranks. Measured impact on this data was ~1e-4 (bootstrap ties
form only between duplicate copies of one row, which share a label, so tie-breaking cannot
move AUROC) — but ordinal ranks are the wrong rule and are not relied on here.
R2 finding 2 corrected that mechanism: the training-CDF step map can ALSO tie distinct
opposite-label rows. Measured across 255 fold-arm fits: 752 such ties created, 205/255
single-column fold AUROCs moved, max 0.010. Cell-level effect on the primary contrast:
max 0.0035, mean 0.0012, 0/17 sign flips. Single-column arms now bypass the map entirely.

── THE 0.65 QUALIFYING GATE (R1 finding 6, corrected by R2 finding 1) ───────────────
An earlier version carried a `target1_gated` arm meant to test what dropping the registered
`A_tr >= 0.65` qualifier costs. That arm was DEGENERATE: restricting an argmax to a
superlevel set that always contains the unrestricted maximum returns the same column, and
the empty case fell back to it. Its zero contrast measured nothing and has been removed.
The real evidence that the threshold never binds here is the independent per-fold record
`training_peak_qualifies_0.65`, which is 5/5 in every cell (85/85 folds).
KNOWN GAP: the registered rescore ALSO required the training peak to beat a per-fold
shuffled-label envelope (rescore_grid_a.py). That limb is NOT implemented here. This file
therefore does not test the full registered qualifier, only the bare 0.65 threshold.

CROSS-FITTING follows the grid-A rescore convention (rescore_grid_a.py): stratified 5-fold
map frozen from the real labels and shared across models within a task; per fold, instrument
choice / block choice / orientation are fit on the 4 TRAINING folds only; the arm is scored
on the held-out fold. Cell value = mean of the 5 held-out AUROCs. Pooled out-of-fold AUROC
over all 200 rows is reported alongside (audit finding 7) as a CORROBORATIVE statistic only.
It compares scores from five differently selected, oriented and calibrated rules; percentile
calibration puts them on a common numerical range but does not align their class-conditional
meaning (R2 finding 4). Fold-mean is the reported statistic.

Held-out scores are calibrated against the TRAINING empirical CDF (audit finding 5), not
ranked within the 40-row held-out fold as before. Single-column arms are unaffected (AUROC
is invariant to a monotone map); fusion arms no longer depend on the held-out cohort.

ARMS
  fixed6      2 instruments x 3 panel rungs, rank-mean fused    (constructed; see naming)
  fixed3_js   js_no_bos at the 3 rungs, fused
  fixed3_bos  bos_mass at the 3 rungs, fused
  rung_best1  best SINGLE of those 6 fixed columns, training-chosen
  target1     best SINGLE (instrument, block) over ALL blocks, training-chosen
  target1_js  js_no_bos at its own training-chosen peak block
  target2     js and bos each at their own peak block, fused

Single-column arms (rung_best1, target1, target1_js) are scored on RAW oriented held-out
values; fused arms are scored on training-CDF percentiles, which they need for a common
scale. Pooled out-of-fold AUROC always uses the percentile version.

Usage: python3 depth_coverage.py    ->  DEPTH_COVERAGE.json
"""

import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JS = "final_js_no_bos"
BOS = "final_bos_mass"
K_FOLDS = 5
N_BOOT = 1000
N_CONTROL = 100
SEED = 20260831
QUAL_AUROC = 0.65


# ── rank / AUROC primitives ───────────────────────────────────────────────────
def midrank_columns(x):
    """Column-wise MIDRANKS 1..n — tied values share their mean rank.

    Matches the house `_rank01` tie policy. Ordinal ranks must not be used: bootstrap
    resampling duplicates rows, so ties are guaranteed, and any positional tie-break
    would encode array order (which is label-sorted) rather than score order.
    """
    n = x.shape[0]
    order = np.argsort(x, axis=0, kind="stable")
    xs = np.take_along_axis(x, order, axis=0)
    idx = np.arange(n, dtype=np.float64)[:, None]
    is_new = np.empty(xs.shape, bool)
    is_new[0] = True
    is_new[1:] = xs[1:] != xs[:-1]
    is_end = np.empty(xs.shape, bool)
    is_end[-1] = True
    is_end[:-1] = xs[1:] != xs[:-1]
    first = np.maximum.accumulate(np.where(is_new, idx, 0.0), axis=0)
    last = np.minimum.accumulate(np.where(is_end, idx, n - 1.0)[::-1], axis=0)[::-1]
    r = np.empty(xs.shape, dtype=np.float64)
    np.put_along_axis(r, order, (first + last) / 2.0 + 1.0, axis=0)
    return r


def auroc_from_ranks(R, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (R[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auroc(score, y):
    return float(auroc_from_ranks(midrank_columns(np.asarray(score, float)[:, None]), y)[0])


def train_percentile(v_tr, v_ho):
    """Map held-out values onto the TRAINING empirical CDF, midrank convention -> [0,1].

    NOTE (R2 finding 2): this is a STEP map — non-decreasing, NOT strictly increasing.
    Distinct held-out values inside one training gap collapse to the same percentile.
    Fusion arms accept that cost for a common scale; single-column arms do not use it.
    """
    s = np.sort(v_tr)
    lo = np.searchsorted(s, v_ho, side="left")
    hi = np.searchsorted(s, v_ho, side="right")
    return (lo + hi) / (2.0 * len(s))


def make_folds(y, rng):
    """Stratified fold assignment frozen from the real labels (rescore_grid_a.make_folds)."""
    fold_of = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        for f, chunk in enumerate(np.array_split(idx, K_FOLDS)):
            fold_of[chunk] = f
    return fold_of


ARMS = ("fixed6", "fixed3_js", "fixed3_bos", "rung_best1",
        "target1", "target1_js", "target2")
SINGLE_ARMS = frozenset({"rung_best1", "target1", "target1_js"})


# ── one cross-fit over one (possibly resampled) row set ───────────────────────
def crossfit(X, y, fold_of, rungs, L, want_state=False):
    """X [n, 2L] = [js blocks | bos blocks].

    Returns (arms -> mean held-out AUROC, state). `state` is per-fold frozen material —
    selected columns, signs, sorted training values, held-out labels — which the
    CONDITIONAL bootstrap resamples against without re-running any selection.
    """
    js_rungs = list(rungs)
    bos_rungs = [L + r for r in rungs]
    six = js_rungs + bos_rungs

    per_fold, acc = [], {k: [] for k in ARMS}
    for f in range(K_FOLDS):
        tr, ho = fold_of != f, fold_of == f
        y_tr, y_ho = y[tr], y[ho]
        if y_tr.min() == y_tr.max() or y_ho.min() == y_ho.max():
            return None, None
        X_tr, X_ho = X[tr], X[ho]

        a_tr = auroc_from_ranks(midrank_columns(X_tr), y_tr)
        dirs = np.where(a_tr >= 0.5, 1.0, -1.0)      # orientation locked on training
        A_tr = np.maximum(a_tr, 1.0 - a_tr)          # sign-free training strength

        def cal(cols):
            """Percentile-calibrate held-out scores of `cols` against TRAINING CDFs."""
            return np.column_stack([
                train_percentile(dirs[c] * X_tr[:, c], dirs[c] * X_ho[:, c]) for c in cols])

        # --- training-side selection (never sees held-out rows)
        p_all = int(np.argmax(A_tr))
        p_js = int(np.argmax(A_tr[:L]))
        p_bos = L + int(np.argmax(A_tr[L:]))
        best_rung = six[int(np.argmax(A_tr[six]))]

        cols_of = {"fixed6": six, "fixed3_js": js_rungs, "fixed3_bos": bos_rungs,
                   "rung_best1": [best_rung], "target1": [p_all], "target1_js": [p_js],
                   "target2": [p_js, p_bos]}

        fold_state = {"y_ho": y_ho, "score": {}, "pool": {}, "cols": cols_of,
                      "top_margin": float(np.sort(A_tr)[-1] - np.sort(A_tr)[-2]),
                      "top_qualifies": bool(A_tr[p_all] >= QUAL_AUROC)}
        for arm, cols in cols_of.items():
            calibrated = cal(cols).mean(axis=1)
            # Single-column arms are scored on RAW oriented values (R2 finding 2): the
            # training ECDF is a STEP map, so it can collapse distinct opposite-label
            # held-out values into ties and move a single-column AUROC. Fusion needs a
            # common scale and keeps the calibration.
            score = (dirs[cols[0]] * X_ho[:, cols[0]]) if arm in SINGLE_ARMS else calibrated
            fold_state["score"][arm] = score
            fold_state["pool"][arm] = calibrated   # pooling always needs a common scale
            acc[arm].append(auroc(score, y_ho))
        per_fold.append(fold_state)

    means = {k: float(np.mean(v)) for k, v in acc.items()}
    return means, (per_fold if want_state else None)


CONTRASTS = (("target1", "rung_best1"),      # PRIMARY: closest isolation of depth freedom
             ("target1", "fixed6"),
             ("fixed6", "rung_best1"),
             ("target2", "fixed6"),
             ("target2", "target1"),
             ("target2", "rung_best1"),
             ("target1_js", "fixed3_js"))


def _ci(v, q=(5, 95)):
    return [round(float(np.percentile(v, q[0])), 4), round(float(np.percentile(v, q[1])), 4)]


def _excl(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def analyse(path, grid, task_index):
    d = np.load(path, allow_pickle=True)
    S, y = d["scores"], d["labels"].astype(int)
    metrics = json.loads(str(d["metrics"]))
    meta = json.loads(str(d["meta"]))
    L = int(meta["n_layers"])
    task = path.split(os.sep)[-2]
    model = os.path.basename(path).replace(".depth.npz", "")

    # --- input validation (audit finding 9)
    assert set(np.unique(y)) <= {0, 1}, f"non-binary labels in {path}"
    assert len(metrics) == len(set(metrics)), f"duplicate metric names in {path}"
    assert S.shape[1] == L, f"block count {S.shape[1]} != meta n_layers {L} in {path}"
    assert S.shape[0] == len(y), f"row/label mismatch in {path}"

    X = np.column_stack([S[:, :, metrics.index(JS)], S[:, :, metrics.index(BOS)]]).astype(float)
    if not np.isfinite(X).all():
        return {"grid": grid, "task": task, "model": model,
                "usable": False, "why": "non-finite block scores"}

    rungs = [L // 2, L - 2, L - 1]
    ti = task_index[task]
    fold_of = make_folds(y, np.random.default_rng((SEED, ti)))
    point, state = crossfit(X, y, fold_of, rungs, L, want_state=True)

    # ── pooled out-of-fold AUROC (finding 7) — valid because every arm is
    # percentile-calibrated against its own training fold, so folds are commensurable.
    y_oof = np.concatenate([fs["y_ho"] for fs in state])
    pooled = {arm: round(auroc(np.concatenate([fs["pool"][arm] for fs in state]), y_oof), 4)
              for arm in ARMS}

    # ── CONDITIONAL interval (primary): selections FROZEN, resample held-out rows only.
    crng = np.random.default_rng((SEED, ti, L, 3))
    ho_strata = [[np.flatnonzero(fs["y_ho"] == c) for c in (0, 1)] for fs in state]
    cond = {arm: [] for arm in ARMS}
    for _ in range(N_BOOT):
        draws = [np.concatenate([s[crng.integers(0, len(s), len(s))] for s in fs_str])
                 for fs_str in ho_strata]
        for arm in ARMS:
            cond[arm].append(np.mean([auroc(state[f]["score"][arm][draws[f]],
                                            state[f]["y_ho"][draws[f]])
                                      for f in range(K_FOLDS)]))

    # ── FULL-ALGORITHM interval (secondary): rerun the whole cross-fit incl. selection.
    frng = np.random.default_rng((SEED, ti, L, len(y)))
    strata = [np.flatnonzero((fold_of == f) & (y == c))
              for f in range(K_FOLDS) for c in (0, 1)]
    bfold = np.concatenate([np.full(len(strata[2 * f + c]), f)
                            for f in range(K_FOLDS) for c in (0, 1)])
    full = {arm: [] for arm in ARMS}
    for _ in range(N_BOOT):
        bidx = np.concatenate([s[frng.integers(0, len(s), len(s))] for s in strata])
        r, _ = crossfit(X[bidx], y[bidx], bfold, rungs, L)
        if r is None:
            continue
        for arm in ARMS:
            full[arm].append(r[arm])

    # ── shuffled-label control: whole cross-fit incl. selection, labels permuted in-fold.
    srng = np.random.default_rng((SEED, ti, L, 7))
    ctrl = {arm: [] for arm in ARMS}
    for _ in range(N_CONTROL):
        yp = y.copy()
        for f in range(K_FOLDS):
            rows = np.flatnonzero(fold_of == f)
            yp[rows] = y[rows[srng.permutation(len(rows))]]
        r, _ = crossfit(X, yp, fold_of, rungs, L)
        if r is not None:
            for arm in ARMS:
                ctrl[arm].append(r[arm])

    out = {
        "grid": grid, "task": task, "model": model, "usable": True,
        "n_layers": L, "n_rows": int(len(y)), "rungs": rungs,
        "arms_foldmean": {k: round(v, 4) for k, v in point.items()},
        "arms_pooled_oof": pooled,
        "selected": [{"instrument": "js" if fs["cols"]["target1"][0] < L else "bos",
                      "block": fs["cols"]["target1"][0] % L,
                      "train_top_margin": round(fs["top_margin"], 4)} for fs in state],
        "training_peak_qualifies_0.65": f"{sum(fs['top_qualifies'] for fs in state)}/{K_FOLDS}",
        "shuffled_label_control": {k: round(float(np.median(v)), 4) for k, v in ctrl.items()},
        "shuffled_label_control_range": {
            k: [round(float(np.min(v)), 4), round(float(np.max(v)), 4)] for k, v in ctrl.items()},
        "contrasts": {},
    }
    for a, b in CONTRASTS:
        delta = point[a] - point[b]
        c_v = np.array(cond[a]) - np.array(cond[b])
        f_v = np.array(full[a]) - np.array(full[b])
        ci_cond = _ci(c_v)
        ci_pct = _ci(f_v)
        # basic/mirrored interval for the same replicates: 2*theta_hat - percentiles
        ci_basic = [round(2 * delta - _ci(f_v)[1], 4), round(2 * delta - _ci(f_v)[0], 4)]
        out["contrasts"][f"{a}-{b}"] = {
            "delta": round(delta, 4),
            "winner": "a" if delta > 0 else ("b" if delta < 0 else "tie"),
            "ci90_conditional": ci_cond,
            "conditional_excludes_zero": _excl(ci_cond),
            "ci90_full_percentile": ci_pct,
            "ci90_full_basic": ci_basic,
            "full_convention_disagrees": bool(_excl(ci_pct) != _excl(ci_basic)),
        }
    return out


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = []
    for grid, pat in (("A", "npz/depth_curve"), ("B", "npz/depth_grid_b")):
        for f in sorted(glob.glob(os.path.join(HERE, pat, "*", "*.depth.npz"))):
            files.append((grid, f))

    task_labels = {}
    for _, f in files:
        d = np.load(f, allow_pickle=True)
        t = f.split(os.sep)[-2]
        y = d["labels"].astype(int)
        if t in task_labels:
            assert np.array_equal(task_labels[t], y), f"label mismatch within task {t}: {f}"
        else:
            task_labels[t] = y
    # NOTE (audit finding 9): identical label VECTORS do not prove identical row identity;
    # rows could in principle be permuted within same-label positions. `sample_idx` is
    # banked in each npz and is checked here to close that gap.
    task_rows = {}
    for _, f in files:
        d = np.load(f, allow_pickle=True)
        t = f.split(os.sep)[-2]
        if "sample_idx" in d:
            si = np.asarray(d["sample_idx"])
            if t in task_rows:
                assert np.array_equal(task_rows[t], si), f"row identity mismatch in {t}: {f}"
            else:
                task_rows[t] = si

    task_index = {t: i for i, t in enumerate(sorted(task_labels))}
    out = [analyse(f, g, task_index) for g, f in files]
    ok = [r for r in out if r["usable"]]

    print("%-2s %-11s %-26s %3s %6s %6s %6s %6s %6s %7s %-17s %-17s %s" % (
        "gr", "task", "model", "N", "fix6", "rung1", "targ1", "targ2", "pool1",
        "t1-r1", "CI90 cond", "CI90 full pct", "flags"))
    for r in ok:
        a, c = r["arms_foldmean"], r["contrasts"]["target1-rung_best1"]
        flags = ("C" if c["conditional_excludes_zero"] else ".") + \
                ("!" if c["full_convention_disagrees"] else ".")
        print("%-2s %-11s %-26s %3d %6.3f %6.3f %6.3f %6.3f %6.3f %+7.3f %-17s %-17s %s" % (
            r["grid"], r["task"], r["model"][:26], r["n_layers"], a["fixed6"], a["rung_best1"],
            a["target1"], a["target2"], r["arms_pooled_oof"]["target1"], c["delta"],
            str(c["ci90_conditional"]), str(c["ci90_full_percentile"]), flags))

    summary = {}
    for grid in ("A", "B"):
        g = [r for r in ok if r["grid"] == grid]
        s = {"n_cells": len(g)}
        for a, b in CONTRASTS:
            k = f"{a}-{b}"
            s[k] = {
                "first_arm_wins": f"{sum(r['contrasts'][k]['winner'] == 'a' for r in g)}/{len(g)}",
                "median_delta": round(float(np.median([r["contrasts"][k]["delta"] for r in g])), 4),
                "conditional_excludes_zero":
                    f"{sum(r['contrasts'][k]['conditional_excludes_zero'] for r in g)}/{len(g)}",
                "full_convention_disagrees":
                    f"{sum(r['contrasts'][k]['full_convention_disagrees'] for r in g)}/{len(g)}",
            }
        s["median_arms_foldmean"] = {a: round(float(np.median([r["arms_foldmean"][a] for r in g])), 4)
                                     for a in ARMS}
        s["median_arms_pooled_oof"] = {a: round(float(np.median([r["arms_pooled_oof"][a] for r in g])), 4)
                                       for a in ARMS}
        s["shuffled_control_median_of_cell_medians"] = {
            a: round(float(np.median([r["shuffled_label_control"][a] for r in g])), 4) for a in ARMS}
        s["shuffled_control_extreme_over_cells"] = {
            a: [round(float(min(r["shuffled_label_control_range"][a][0] for r in g)), 4),
                round(float(max(r["shuffled_label_control_range"][a][1] for r in g)), 4)] for a in ARMS}
        summary[f"grid_{grid}"] = s
        print(f"\ngrid {grid} (n={len(g)}):")
        for a, b in CONTRASTS:
            k = f"{a}-{b}"
            print(f"  {k:<26} wins {s[k]['first_arm_wins']:>5} | median {s[k]['median_delta']:+.4f}"
                  f" | cond excl0 {s[k]['conditional_excludes_zero']:>5}"
                  f" | conv-disagree {s[k]['full_convention_disagrees']:>5}")
        print("  median fold-mean:", s["median_arms_foldmean"])
        print("  median pooled OOF:", s["median_arms_pooled_oof"])
        print("  shuffled ctrl (median / extreme):",
              s["shuffled_control_median_of_cell_medians"]["target1"],
              s["shuffled_control_extreme_over_cells"]["target1"])

    prov = {"script_sha256": _sha256(os.path.abspath(__file__)),
            "numpy": np.__version__,
            "inputs": {os.path.relpath(f, HERE): _sha256(f) for _, f in files}}
    with open(os.path.join(HERE, "DEPTH_COVERAGE.json"), "w") as fh:
        json.dump({"config": {"instruments": [JS, BOS], "k_folds": K_FOLDS,
                              "n_boot": N_BOOT, "n_control": N_CONTROL, "seed": SEED,
                              "ranks": "midranks (house _rank01 tie policy)",
                              "calibration": "held-out mapped onto TRAINING empirical CDF",
                              "rungs": "0-indexed N//2, N-2, N-1 (verified vs panel, 4dp)",
                              "primary_contrast": "target1-rung_best1",
                              "primary_interval": "ci90_conditional (regular; selection frozen)",
                              "naming": "fused arms are house-STYLE fold-local fusions, NOT the "
                                        "deployed ACE arm; rung_best1 is not the production selector",
                              "note": "grids never pooled; all selection on training folds only"},
                   "provenance": prov, "summary": summary, "cells": out}, fh, indent=2)


if __name__ == "__main__":
    main()
