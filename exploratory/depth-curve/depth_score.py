"""Depth-curve scorer — implements PRE_REGISTRATION.md endpoints EXACTLY. Local, no GPU.

Run ONCE, after ALL 8 extraction cells exist (fail-closed on missing cells):
  python depth_score.py --npz-dir <dir>   # <dir>/<task>/<slug>.depth.npz layout
Writes RESULTS.json + RESULTS.md next to this file.

Frozen constants below mirror the prereg. One implementation decision recorded
pre-inspection: inside bootstrap resamples the per-block shuffled-label envelope is the
FULL-SAMPLE envelope (a fixed qualifier mask); recomputing 200 permutations inside each
of 1000 resamples is disproportionate and the envelope is a property of the block, not
the resample. Point-estimate ℓ* uses full-sample AUROC + the same mask.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from scipy.stats import rankdata


def _stable_seed(*parts):
    """Deterministic across processes (python hash() is salted per process)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return SEED + int(h[:8], 16) % (2 ** 16)

SEED = 20260816
NBOOT = 1000
NPERM = 200
ENVELOPE_Q = 97.5
QUAL_AUROC = 0.65
PRIMARY = "final_js_no_bos"
E1_ABS_NL_SPAN = 2      # absolute: span of median(N-l*) <= 2 blocks
E1_ABS_FR_SPAN = 0.03   # ... AND span of median(l*/N) >= 0.03
E1_REL_FR_SPAN = 0.015  # relative: span of median(l*/N) <= 0.015
E1_REL_NL_SPAN = 3      # ... AND span of median(N-l*) >= 3
E2_CLIFF_J = 0.15
E2_GRADUAL_J = 0.08
E3_FRONT = 0.4
E3_AUROC = 0.60
E4_DIP = 0.05
P3_CEILING = 0.75
P4_MID_MEDIAN = 0.60

QWEN_TRIO = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct"]
CONTROL = "Llama-3.3-70B-Instruct"
TASKS = ["anli_r1", "halueval_qa"]


def signfree_auroc_matrix(scores_lm, y):
    """scores_lm: [n, L] for one metric; y: [n] in {0,1}. Returns [L] sign-free AUROCs."""
    n, L = scores_lm.shape
    n1 = int((y == 1).sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("degenerate labels")
    out = np.empty(L, dtype=np.float64)
    for li in range(L):
        r = rankdata(scores_lm[:, li])
        a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        out[li] = max(a, 1.0 - a)
    return out


def load_cell(npz_dir, task, slug):
    path = os.path.join(npz_dir, task, f"{slug}.depth.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MISSING CELL: {path} — scoring requires all 8 cells (prereg discipline)")
    z = np.load(path, allow_pickle=False)
    metrics = json.loads(str(z["metrics"]))
    meta = json.loads(str(z["meta"]))
    scores = z["scores"]  # [n, L, K]
    y = z["labels"].astype(np.int64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: non-finite scores")
    if scores.shape[0] != y.shape[0] or scores.shape[1] != int(meta["n_layers"]):
        raise ValueError(f"{path}: shape mismatch {scores.shape} vs n_layers={meta['n_layers']}")
    return scores, y, metrics, meta


def analyze_cell(scores, y, metrics, meta, rng_env, rng_boot):
    n, L, K = scores.shape
    pk = metrics.index(PRIMARY)
    prim = scores[:, :, pk]  # [n, L]

    curves = {m: signfree_auroc_matrix(scores[:, :, k], y).tolist() for k, m in enumerate(metrics)}
    A = np.asarray(curves[PRIMARY])

    # shuffled-label envelope (primary metric), same sign-free statistic
    env = np.empty((NPERM, L), dtype=np.float64)
    for p in range(NPERM):
        yp = rng_env.permutation(y)
        env[p] = signfree_auroc_matrix(prim, yp)
    env_q = np.percentile(env, ENVELOPE_Q, axis=0)

    qual_mask = (A >= QUAL_AUROC) & (A > env_q)
    if qual_mask.any():
        lstar = int(np.argmax(np.where(qual_mask, A, -np.inf)))
    else:
        lstar = None

    # bootstrap l* (primary), full-sample envelope as fixed qualifier mask
    boot_lstars = []
    n_noqual = 0
    for b in range(NBOOT):
        idx = rng_boot.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            n_noqual += 1
            continue
        Ab = signfree_auroc_matrix(prim[idx], yb)
        mb = (Ab >= QUAL_AUROC) & (A > env_q)
        if not mb.any():
            n_noqual += 1
            continue
        boot_lstars.append(int(np.argmax(np.where(mb, Ab, -np.inf))))
    boot_lstars = np.asarray(boot_lstars, dtype=np.int64)

    def q(v, p):
        return float(np.percentile(v, p)) if len(v) else None

    N = L
    res = {
        "model": meta["model"], "task": meta["task"], "n_layers": N, "n_rows": n,
        "curves_signfree_auroc": curves, "envelope_q97_5_primary": env_q.tolist(),
        "lstar": lstar,
        "lstar_boot_median": q(boot_lstars, 50), "lstar_boot_ci": [q(boot_lstars, 5), q(boot_lstars, 95)],
        "boot_noqual_frac": round(n_noqual / NBOOT, 4),
        "peak_auroc": float(A[lstar]) if lstar is not None else None,
        "N_minus_lstar_median": (N - q(boot_lstars, 50)) if len(boot_lstars) else None,
        "lstar_frac_median": (q(boot_lstars, 50) / N) if len(boot_lstars) else None,
    }

    # E2 rise shape
    if lstar is not None:
        lo, hi = int(np.floor(E3_FRONT * N)), int(np.ceil(0.6 * N))
        b_base = float(np.median(A[lo:hi])) if hi > lo else float("nan")
        start = int(np.floor(0.5 * N))
        seg = A[start:lstar + 1]
        J = float(np.max(np.abs(np.diff(seg)))) if len(seg) >= 2 else 0.0
        R = float(A[lstar] - b_base)
        shape = ("CLIFF" if (J >= E2_CLIFF_J and J >= 0.5 * R) else
                 "GRADUAL" if J <= E2_GRADUAL_J else "MIXED")
        res["E2"] = {"baseline_mid": b_base, "rise": R, "max_adjacent_jump": J, "shape": shape}
    else:
        res["E2"] = None

    # E3 early layers
    front = int(np.floor(E3_FRONT * N))
    early = [int(li) for li in range(front) if A[li] >= E3_AUROC and A[li] > env_q[li]]
    res["E3_early_blocks"] = early

    # E4 terminal dip
    res["E4_terminal_dip"] = (lstar is not None and lstar <= N - 2
                              and float(A[N - 1]) <= float(A[lstar]) - E4_DIP)
    res["mid_region_median"] = float(np.median(A[int(np.floor(0.4 * N)):int(np.ceil(0.6 * N))]))
    res["max_auroc_any_block"] = float(A.max())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    cells = {}
    for task in TASKS:
        for slug in QWEN_TRIO + [CONTROL]:
            rng_env = np.random.default_rng(_stable_seed(task, slug, "env"))
            rng_boot = np.random.default_rng(_stable_seed(task, slug, "boot"))
            scores, y, metrics, meta = load_cell(args.npz_dir, task, slug)
            cells[(task, slug)] = analyze_cell(scores, y, metrics, meta, rng_env, rng_boot)
            print(f"scored {task}/{slug}")

    # E1 decision per task (Qwen trio)
    verdicts = {}
    for task in TASKS:
        rows = [cells[(task, s)] for s in QWEN_TRIO]
        with_peak = [r for r in rows if r["lstar"] is not None and r["lstar_boot_median"] is not None]
        if len(with_peak) < 2:
            verdicts[task] = {"E1": "NO-PEAKS", "n_with_peak": len(with_peak)}
            continue
        nl = [r["N_minus_lstar_median"] for r in with_peak]
        fr = [r["lstar_frac_median"] for r in with_peak]
        nl_span = max(nl) - min(nl)
        fr_span = max(fr) - min(fr)
        if nl_span <= E1_ABS_NL_SPAN and fr_span >= E1_ABS_FR_SPAN:
            e1 = "ABSOLUTE"
        elif fr_span <= E1_REL_FR_SPAN and nl_span >= E1_REL_NL_SPAN:
            e1 = "RELATIVE"
        else:
            e1 = "UNDECIDED"
        verdicts[task] = {"E1": e1, "n_with_peak": len(with_peak),
                          "N_minus_lstar_medians": nl, "lstar_frac_medians": fr,
                          "nl_span": nl_span, "fr_span": round(fr_span, 4)}
    e1_overall = (verdicts[TASKS[0]]["E1"] if verdicts[TASKS[0]]["E1"] == verdicts[TASKS[1]]["E1"]
                  else "MIXED")

    # predictions
    qwen_cells = [cells[(t, s)] for t in TASKS for s in QWEN_TRIO]
    ctrl_cells = [cells[(t, CONTROL)] for t in TASKS]
    P = {
        "P1_all_qwen_peak_last4": all(c["lstar"] is not None and c["lstar"] >= c["n_layers"] - 4
                                      for c in qwen_cells),
        "P3_control_no_peak": all(c["lstar"] is None and c["max_auroc_any_block"] < P3_CEILING
                                  for c in ctrl_cells),
        "P4_qwen_mid_quiet_32_72": all(cells[(t, s)]["mid_region_median"] < P4_MID_MEDIAN
                                       for t in TASKS
                                       for s in ["Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct"]),
        "P5_terminal_dip_32_72": all(cells[(t, s)]["E4_terminal_dip"]
                                     for t in TASKS
                                     for s in ["Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct"]),
    }

    out = {"prereg": "PRE_REGISTRATION.md", "seed": SEED, "nboot": NBOOT, "nperm": NPERM,
           "primary_metric": PRIMARY,
           "E1_per_task": verdicts, "E1_OVERALL": e1_overall, "predictions": P,
           "cells": {f"{t}/{s}": c for (t, s), c in cells.items()}}
    jpath = os.path.join(args.out_dir, "RESULTS.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)

    lines = ["# Depth-curve results (auto-generated by depth_score.py)", "",
             f"**E1 OVERALL: {e1_overall}**  ({TASKS[0]}: {verdicts[TASKS[0]]['E1']}, "
             f"{TASKS[1]}: {verdicts[TASKS[1]]['E1']})", "",
             "| task | model | N | l* | boot CI | N−l* | l*/N | peak | mid-med | E2 | E4 dip |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (t, s), c in cells.items():
        e2 = c["E2"]["shape"] if c["E2"] else "—"
        ls = c["lstar"] if c["lstar"] is not None else "none"
        ci = c["lstar_boot_ci"] if c["lstar_boot_median"] is not None else "—"
        nml = c["N_minus_lstar_median"]; frm = c["lstar_frac_median"]
        lines.append(f"| {t} | {s} | {c['n_layers']} | {ls} | {ci} | "
                     f"{nml if nml is not None else '—'} | "
                     f"{round(frm, 3) if frm is not None else '—'} | "
                     f"{round(c['peak_auroc'], 3) if c['peak_auroc'] else '—'} | "
                     f"{round(c['mid_region_median'], 3)} | {e2} | {'Y' if c['E4_terminal_dip'] else 'N'} |")
    lines += ["", f"Predictions: {json.dumps(P)}", ""]
    mpath = os.path.join(args.out_dir, "RESULTS.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {jpath}\nwrote {mpath}\nE1 OVERALL: {e1_overall}")


if __name__ == "__main__":
    main()
