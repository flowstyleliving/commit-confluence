"""Instrument co-location analysis over EXISTING depth-curve artifacts.

Read-only. No model forwards, no Modal spend. Recomputes per-block sign-free AUROC
for the two ACE attention instruments that were both recorded per-layer
(`final_js_no_bos` = inter-head disagreement, `final_bos_mass` = attention-sink mass)
and asks whether they peak at the same depth.

Paired bootstrap over rows gives a CI on the peak-location GAP; a shuffled-label
permutation envelope gives a per-block null. Grids A and B are scored separately and
never pooled (both are torch/Modal, NON-byte-comparable).

Usage: python3 colocation_analysis.py
Writes COLOCATION.json next to this file.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = "final_js_no_bos"
SECOND = "final_bos_mass"
N_BOOT = 1000
N_PERM = 200
SEED = 20260829


def curves(scores, labels):
    """Sign-free per-block AUROC for a (rows, blocks) score matrix."""
    n = scores.shape[0]
    n1 = int(labels.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return np.full(scores.shape[1], np.nan)
    order = np.argsort(scores, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, n + 1, dtype=np.float64)[:, None], axis=0)
    a = (ranks[labels == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    a = np.maximum(a, 1.0 - a)
    a[np.isnan(scores).any(axis=0)] = np.nan
    return a


def peak(c):
    return int(np.nanargmax(c)) if not np.isnan(c).all() else -1


def analyse(path, grid):
    d = np.load(path, allow_pickle=True)
    S, y = d["scores"], d["labels"]
    metrics = json.loads(str(d["metrics"]))
    meta = json.loads(str(d["meta"]))
    ip, isec = metrics.index(PRIMARY), metrics.index(SECOND)
    Sp, Ss = S[:, :, ip], S[:, :, isec]

    cp, cs = curves(Sp, y), curves(Ss, y)
    lp, ls = peak(cp), peak(cs)

    rng = np.random.default_rng(SEED)
    n = len(y)
    boot_p, boot_s, boot_gap = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        bp, bs = peak(curves(Sp[idx], yb)), peak(curves(Ss[idx], yb))
        if bp < 0 or bs < 0:
            continue
        boot_p.append(bp)
        boot_s.append(bs)
        boot_gap.append(bp - bs)

    perm_p, perm_s = [], []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        perm_p.append(curves(Sp, yp))
        perm_s.append(curves(Ss, yp))
    env_p = np.nanpercentile(np.array(perm_p), 97.5, axis=0)
    env_s = np.nanpercentile(np.array(perm_s), 97.5, axis=0)

    ok = ~np.isnan(cp) & ~np.isnan(cs)
    return {
        "grid": grid,
        "task": path.split(os.sep)[-2],
        "model": os.path.basename(path).replace(".depth.npz", ""),
        "n_layers": int(meta["n_layers"]),
        "js_peak_block": lp,
        "js_peak_auc": round(float(cp[lp]), 4),
        "js_peak_ci90": [int(np.percentile(boot_p, 5)), int(np.percentile(boot_p, 95))],
        "bos_peak_block": ls,
        "bos_peak_auc": round(float(cs[ls]), 4),
        "bos_peak_ci90": [int(np.percentile(boot_s, 5)), int(np.percentile(boot_s, 95))],
        "gap": lp - ls,
        "gap_ci90": [int(np.percentile(boot_gap, 5)), int(np.percentile(boot_gap, 95))],
        "gap_excludes_zero": bool(
            np.percentile(boot_gap, 5) > 0 or np.percentile(boot_gap, 95) < 0
        ),
        "js_at_bos_peak": round(float(cp[ls]), 4),
        "bos_at_js_peak": round(float(cs[lp]), 4),
        "js_envelope_at_peak": round(float(env_p[lp]), 4),
        "bos_envelope_at_peak": round(float(env_s[ls]), 4),
        "curve_pearson_r": round(float(np.corrcoef(cp[ok], cs[ok])[0, 1]), 4) if ok.sum() > 3 else None,
        "bos_beats_js": bool(cs[ls] > cp[lp]),
    }


def main():
    out = []
    for grid, pat in (("A", "npz/depth_curve"), ("B", "npz/depth_grid_b")):
        for f in sorted(glob.glob(os.path.join(HERE, pat, "*", "*.depth.npz"))):
            out.append(analyse(f, grid))

    hdr = ("grid", "task", "model", "N", "js*", "ci90", "jsAUC", "bos*", "ci90", "bosAUC",
           "gap", "gapCI", "sep", "js@bos*", "bos@js*", "r")
    print("%-2s %-11s %-26s %3s %4s %-9s %6s %5s %-9s %6s %5s %-10s %4s %7s %7s %6s" % hdr)
    for r in out:
        print("%-2s %-11s %-26s %3d %4d %-9s %6.3f %5d %-9s %6.3f %5d %-10s %4s %7.3f %7.3f %6.3f" % (
            r["grid"], r["task"], r["model"][:26], r["n_layers"],
            r["js_peak_block"], str(r["js_peak_ci90"]), r["js_peak_auc"],
            r["bos_peak_block"], str(r["bos_peak_ci90"]), r["bos_peak_auc"],
            r["gap"], str(r["gap_ci90"]), "YES" if r["gap_excludes_zero"] else "no",
            r["js_at_bos_peak"], r["bos_at_js_peak"], r["curve_pearson_r"]))

    for grid in ("A", "B"):
        g = [r for r in out if r["grid"] == grid]
        print(f"\ngrid {grid}: n={len(g)} cells; gap CI excludes 0 in "
              f"{sum(r['gap_excludes_zero'] for r in g)}/{len(g)}; "
              f"bos_mass peak > js peak in {sum(r['bos_beats_js'] for r in g)}/{len(g)}")

    with open(os.path.join(HERE, "COLOCATION.json"), "w") as fh:
        json.dump({"config": {"primary": PRIMARY, "second": SECOND, "n_boot": N_BOOT,
                              "n_perm": N_PERM, "seed": SEED,
                              "note": "sign-free in-sample AUROC; grids never pooled"},
                   "cells": out}, fh, indent=2)


if __name__ == "__main__":
    main()
