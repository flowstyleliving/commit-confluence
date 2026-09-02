"""Question A — are the three ACE attention instruments one signal or three?

Read-only, zero-compute. Operates on the BANKED torch-panel profile matrices
(`<archive>/<task>/<model>.matrix.npz`, each `score_matrix (200, 27)` + `labels`),
which record all seven attention metrics at all three panel depths.

The three headline instruments:
    js_no_bos              inter-head Jensen-Shannon disagreement (sink column stripped)
    bos_mass               attention-sink mass
    v_norm_lastq_weighted  V-norm-weighted attention

For each (model, task, depth) it asks two things:

  1. SHARED STRUCTURE — pairwise Spearman among the three, and the variance fraction
     carried by the first principal component of the three standardized instruments.
     One shared latent => high |rho| and PC1 fraction near 1.

  2. UNIQUE CONTRIBUTION (the discriminator) — residualize each instrument on the other
     two (least squares on ranks) and score the residual against the label. If the part
     of instrument i that the other two CANNOT explain still separates hallucinated from
     faithful, instrument i carries information the others do not, and "three coarse
     measurements of one quantity" is false.

Orientation is fit per (cell, depth, instrument) so AUROC >= 0.5 — in-sample sign
fitting, same convention as the sealed panel marginals. Precision variants (__bf16,
__int8, __fp32) are excluded from headline counts to avoid pseudo-replication, and
reported separately.

Usage: python3 instrument_redundancy.py [archive_dir]
Writes REDUNDANCY.json next to this file.
"""

import ast
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVE = os.path.expanduser(
    "~/Documents/furnace-guard/artifacts/modal_profiles_ext/profiles_ext"
)
INSTRUMENTS = ("js_no_bos", "bos_mass", "v_norm_lastq_weighted")
DEPTHS = ("mid", "last_minus_1", "final")
UNIQUE_BAR = 0.60  # residual AUROC above this = instrument carries its own signal


def ranks(a):
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(score, y):
    r = ranks(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse_cell(path):
    d = np.load(path, allow_pickle=True)
    S, y = d["score_matrix"], d["labels"]
    # `panel` is a JSON array of STRINGS, each the repr of a (block, family, metric)
    # tuple — so it needs a second parse per element.
    names = [ast.literal_eval(e)[2] for e in json.loads(str(d["panel"]))]
    col = {n: i for i, n in enumerate(names)}
    meta = json.loads(str(d["meta"]))

    out = []
    for depth in DEPTHS:
        keys = [f"{depth}_{ins}" for ins in INSTRUMENTS]
        if not all(k in col for k in keys):
            continue
        X = np.column_stack([S[:, col[k]] for k in keys]).astype(float)
        if np.isnan(X).any() or (X.std(axis=0) == 0).any():
            continue

        # orient each instrument so its own AUROC >= 0.5 (in-sample sign fit)
        signs, aucs = [], []
        for j in range(3):
            a = auc(X[:, j], y)
            s = 1.0 if a >= 0.5 else -1.0
            signs.append(s)
            aucs.append(max(a, 1 - a))
            X[:, j] *= s

        R = np.column_stack([ranks(X[:, j]) for j in range(3)])
        Z = (R - R.mean(axis=0)) / R.std(axis=0)

        rho = {"js~bos": spearman(X[:, 0], X[:, 1]),
               "js~vnorm": spearman(X[:, 0], X[:, 2]),
               "bos~vnorm": spearman(X[:, 1], X[:, 2])}
        ev = np.linalg.eigvalsh(np.cov(Z, rowvar=False))[::-1]
        pc1 = float(ev[0] / ev.sum())

        # unique contribution: residualize each on the other two, score residual
        unique = []
        for j in range(3):
            others = [k for k in range(3) if k != j]
            A = np.column_stack([np.ones(len(Z)), Z[:, others]])
            beta, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
            resid = Z[:, j] - A @ beta
            r2 = 1.0 - resid.var() / Z[:, j].var()
            ra = auc(resid, y)
            unique.append({"instrument": INSTRUMENTS[j],
                           "solo_auc": round(float(aucs[j]), 4),
                           "sign": int(signs[j]),
                           "r2_from_others": round(float(r2), 4),
                           "residual_auc": round(float(max(ra, 1 - ra)), 4)})

        out.append({
            "model": os.path.basename(path).replace(".matrix.npz", ""),
            "task": os.path.basename(os.path.dirname(path)),
            "depth": depth,
            "precision": meta.get("precision"),
            "is_variant": "__" in os.path.basename(path),
            "spearman": {k: (round(v, 4) if v == v else None) for k, v in rho.items()},
            "pc1_var_frac": round(pc1, 4),
            "instruments": unique,
            "n_unique_above_bar": int(sum(u["residual_auc"] > UNIQUE_BAR for u in unique)),
        })
    return out


def main():
    archive = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARCHIVE
    cells = []
    for f in sorted(glob.glob(os.path.join(archive, "*", "*.matrix.npz"))):
        cells.extend(analyse_cell(f))

    main_cells = [c for c in cells if not c["is_variant"]]
    print(f"{len(cells)} (model, task, depth) rows; {len(main_cells)} excluding precision variants\n")

    hdr = ("model", "task", "depth", "rho js~bos", "js~vn", "bos~vn", "PC1",
           "solo js/bos/vn", "resid js/bos/vn", "uniq")
    print("%-24s %-12s %-13s %9s %6s %6s %5s  %-20s %-20s %4s" % hdr)
    for c in main_cells:
        s = c["spearman"]
        so = "/".join(f"{u['solo_auc']:.2f}" for u in c["instruments"])
        re_ = "/".join(f"{u['residual_auc']:.2f}" for u in c["instruments"])
        print("%-24s %-12s %-13s %9s %6s %6s %5.2f  %-20s %-20s %4d" % (
            c["model"][:24], c["task"][:12], c["depth"],
            f"{s['js~bos']:+.2f}" if s["js~bos"] is not None else "  n/a",
            f"{s['js~vnorm']:+.2f}" if s["js~vnorm"] is not None else " n/a",
            f"{s['bos~vnorm']:+.2f}" if s["bos~vnorm"] is not None else " n/a",
            c["pc1_var_frac"], so, re_, c["n_unique_above_bar"]))

    pc1 = [c["pc1_var_frac"] for c in main_cells]
    uniq = [c["n_unique_above_bar"] for c in main_cells]
    print(f"\nPC1 variance fraction: median {np.median(pc1):.3f}, "
          f"range [{min(pc1):.3f}, {max(pc1):.3f}]  (1.00 = one shared factor)")
    print(f"Rows where ALL THREE residuals clear {UNIQUE_BAR}: "
          f"{sum(u == 3 for u in uniq)}/{len(main_cells)}")
    print(f"Rows where at least one residual clears {UNIQUE_BAR}: "
          f"{sum(u >= 1 for u in uniq)}/{len(main_cells)}")
    for j, ins in enumerate(INSTRUMENTS):
        rs = [c["instruments"][j]["residual_auc"] for c in main_cells]
        r2 = [c["instruments"][j]["r2_from_others"] for c in main_cells]
        print(f"  {ins:<22} residual AUROC median {np.median(rs):.3f}; "
              f"R^2 explained by the other two median {np.median(r2):.3f}")

    with open(os.path.join(HERE, "REDUNDANCY.json"), "w") as fh:
        json.dump({"config": {"instruments": list(INSTRUMENTS), "depths": list(DEPTHS),
                              "unique_bar": UNIQUE_BAR, "archive": archive,
                              "note": "in-sample sign fit per instrument; n=200; "
                                      "torch/Modal panel, NON-byte-comparable"},
                   "cells": cells}, fh, indent=2)


if __name__ == "__main__":
    main()
