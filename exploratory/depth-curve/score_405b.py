"""405B STRETCH-CELL scorer — DESCRIPTIVE ONLY (PRE_REGISTRATION_EXPANSION.md §1/§7).

The Llama-3.1-405B pair is registered as a stretch OUTSIDE every confirmatory
denominator: it runs only on MK's explicit go (given 2026-08-17), is reported
descriptively, and can neither strengthen nor weaken the sealed grid-B verdict
(GRID_B_RESULTS.* is never read or written here). No CONFIRM/WEAKEN/FALSIFY
vocabulary appears in this output.

What it reports, per cell (same frozen machinery and fold designs as grid B —
rescore_grid_a.py verified against the freeze pin; labels from the local frozen
data files):
  - cross-fitted dip Δ_cf with fold detail, a per-cell within-fold permutation p
    ((1+k)/(B+1)), and a conditional bootstrap CI;
  - cross-fitted cliff quantities J_cf / R_cf (descriptive, no success calls);
  - cross-fitted peak location + fraction (the E1'' panel point at N=126);
  - the E8-style band count: qualifying blocks (train AUROC >= 0.65 AND above the
    training-rows-only envelope) in [0.4N, 0.9N] per fold — does the Llama
    mid-stack band persist at 405B, beside 8B (3-8), 70B (16-25), 3.3-70B (25-31)?

Run ONCE after both 405B cells reach terminal status:
  .venv/bin/python score_405b.py --npz-dir npz/depth_grid_b
Writes STRETCH_405B_RESULTS.json + .md (refuses to run if they exist; atomic).
"""
import argparse
import hashlib
import json
import os

import numpy as np

from depth_score import QUAL_AUROC, ENVELOPE_Q
from rescore_grid_a import (
    RS_SEED, K_FOLDS, NPERM_INNER, NPERM_OUTER, NBOOT,
    _rng, _sha256_file, mc_p, rank_columns,
    make_folds, within_fold_permutation, crossfit_cell,
    e5_from_crossfit, e6_from_crossfit,
)
from score_grid_b import (
    RESCORE_PIN_SHA256, PRIMARY, EXPECT_METRICS, EXPECT_SCHEMA, EXPECT_N_ROWS,
    TASKS, load_task_labels, fold_qual_masks,
)

SLUG = "Llama-3.1-405B-Instruct"
EXP_LAYERS = 126
EXP_PRECISION = "nf4"
FROZEN_REVISION = "be673f326cab4cd22ccfef76109faf68e41aa5f1"


def load_405b_cell(npz_dir, task, y_task):
    path = os.path.join(npz_dir, task, f"{SLUG}.depth.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"npz missing: {path}")
    z = np.load(path, allow_pickle=False)
    metrics = json.loads(str(z["metrics"]))
    meta = json.loads(str(z["meta"]))
    scores = z["scores"]
    y = z["labels"].astype(np.int64)

    def req(cond, msg):
        if not cond:
            raise ValueError(f"{path}: {msg}")

    req(meta.get("schema") == EXPECT_SCHEMA, f"schema {meta.get('schema')!r}")
    req(str(meta.get("model", "")).endswith(SLUG), "model id")
    req(meta.get("task") == task, "task")
    req(meta.get("precision") == EXP_PRECISION, f"precision {meta.get('precision')!r}")
    req(int(meta.get("n_layers", -1)) == EXP_LAYERS, f"n_layers {meta.get('n_layers')}")
    req(meta.get("revision_pinned") == FROZEN_REVISION, "revision_pinned")
    req(meta.get("provenance", {}).get("hf_model_revision") == FROZEN_REVISION,
        "loaded revision")
    from score_grid_b import FROZEN_DATA_SHA256 as _FD
    req(meta.get("data_sha256") == _FD[task], "data sha mismatch")
    req(metrics == EXPECT_METRICS, "metrics")
    req(scores.shape == (EXPECT_N_ROWS, EXP_LAYERS, len(EXPECT_METRICS)), "shape")
    req(np.array_equal(y, y_task), "labels != frozen data")
    req(np.array_equal(z["sample_idx"], np.arange(EXPECT_N_ROWS)), "sample_idx")
    gate = meta.get("gate") or {}
    req(bool(gate.get("GATE_cos_ok")) and bool(gate.get("GATE_yes_no_ok")), "gates")
    req(float(meta.get("yes_no_commit_rate", 0.0)) >= 0.5, "yes/no rate")
    req(np.isfinite(scores).all(), "non-finite")
    req(meta.get("capture_mode") == "full_eager_retention", "capture mode")
    return scores[:, :, metrics.index(PRIMARY)], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    live_pin = _sha256_file(os.path.join(here, "rescore_grid_a.py"))
    if live_pin != RESCORE_PIN_SHA256:
        raise SystemExit(f"STAT MACHINERY DRIFT: {live_pin}")

    jpath = os.path.join(args.out_dir, "STRETCH_405B_RESULTS.json")
    mpath = os.path.join(args.out_dir, "STRETCH_405B_RESULTS.md")
    for p in (jpath, mpath):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING TO RUN: {p} exists — single look.")

    y_by_task = {t: load_task_labels(t) for t in TASKS}

    # terminal-status enforcement for BOTH stretch cells
    cells = {}
    for task in TASKS:
        spath = os.path.join(args.npz_dir, task, f"{SLUG}.status.json")
        if not os.path.exists(spath):
            raise SystemExit(f"NOT TERMINAL: missing {spath}")
        status = json.load(open(spath))
        if status.get("status") != "ok":
            cells[task] = {"evaluable": False,
                           "reason": f"aborted: {str(status.get('reason', '?'))[:300]}"}
            continue
        try:
            prim, meta = load_405b_cell(args.npz_dir, task, y_by_task[task])
            cells[task] = {"evaluable": True, "prim": prim,
                           "npz_sha256": _sha256_file(
                               os.path.join(args.npz_dir, task, f"{SLUG}.depth.npz")),
                           "gpu": meta.get("load_notes", {}).get("gpu_names"),
                           "yes_no_rate": meta.get("yes_no_commit_rate")}
        except Exception as e:  # noqa: BLE001
            cells[task] = {"evaluable": False,
                           "reason": f"invalid artifact: {type(e).__name__}: {e}"}

    # frozen designs (identical construction + seed to grid B / calibration)
    out_cells = {}
    for task in TASKS:
        c = cells[task]
        if not c["evaluable"]:
            out_cells[task] = {"evaluable": False, "reason": c["reason"]}
            print(f"{task}: not evaluable", flush=True)
            continue
        y = y_by_task[task]
        fold_of = make_folds(y, _rng(task, "folds"))
        inner = []
        for f in range(K_FOLDS):
            n_tr = int((fold_of != f).sum())
            g = _rng(task, "inner-env", f)
            inner.append(np.stack([g.permutation(n_tr) for _ in range(NPERM_INNER)]))
        prim = c["prim"]
        cache = []
        for f in range(K_FOLDS):
            tr = fold_of != f
            cache.append((rank_columns(prim[tr]), rank_columns(prim[~tr])))

        cf = crossfit_cell(prim, y, fold_of, inner, rank_cache=cache)
        e5 = e5_from_crossfit(cf, EXP_LAYERS)
        e6 = e6_from_crossfit(cf, EXP_LAYERS)

        # per-cell within-fold permutation null (descriptive p)
        g = _rng(task, "outer-perm-withinfold")
        nd = np.full(NPERM_OUTER, -np.inf)
        for p in range(NPERM_OUTER):
            yp = within_fold_permutation(y, fold_of, g)
            cfp = crossfit_cell(prim, yp, fold_of, inner, rank_cache=cache)
            e5p = e5_from_crossfit(cfp, EXP_LAYERS)
            if e5p["defined"]:
                nd[p] = e5p["delta"]
            if (p + 1) % 500 == 0:
                print(f"perm {task}: {p + 1}/{NPERM_OUTER}", flush=True)

        # bootstrap (conditional CI), same stratified construction
        gb = _rng(task, "boot")
        bdelta = np.full(NBOOT, np.nan)
        for b in range(NBOOT):
            take = np.empty(EXPECT_N_ROWS, dtype=np.int64)
            pos = 0
            for f in range(K_FOLDS):
                for cls in (0, 1):
                    stratum = np.flatnonzero((fold_of == f) & (y == cls))
                    take[pos:pos + len(stratum)] = stratum[
                        gb.integers(0, len(stratum), size=len(stratum))]
                    pos += len(stratum)
            cfb = crossfit_cell(prim[take], y[take], fold_of[take], inner)
            e5b = e5_from_crossfit(cfb, EXP_LAYERS)
            if e5b["defined"]:
                bdelta[b] = e5b["delta"]
            if (b + 1) % 500 == 0:
                print(f"boot {task}: {b + 1}/{NBOOT}", flush=True)
        fin = bdelta[~np.isnan(bdelta)]

        # band count (E8-style), full qualification rule
        quals, curves = fold_qual_masks(prim, y, fold_of, inner)
        lo, hi = int(np.ceil(0.4 * EXP_LAYERS)), int(np.floor(0.9 * EXP_LAYERS))
        counts = [int(q[lo:hi + 1].sum()) for q in quals]

        entry = {
            "evaluable": True,
            "delta_cf": e5["delta"] if e5["defined"] else None,
            "delta_defined": e5["defined"],
            "fold_peaks": e5.get("fold_peaks"),
            "fold_contrasts": e5.get("fold_contrasts"),
            "perm_p_delta": (mc_p(nd, e5["delta"]) if e5["defined"] else None),
            "null_undefined_frac": float(np.isneginf(nd).mean()),
            "boot_ci_5_95_conditional": ([float(np.percentile(fin, 5)),
                                          float(np.percentile(fin, 95))]
                                         if len(fin) else None),
            "boot_undefined_frac": float(np.isnan(bdelta).mean()),
            "J_cf": e6["J"] if e6["defined"] else None,
            "R_cf": e6["R"] if e6["defined"] else None,
            "e6_undefined_why": (None if e6["defined"] else e6.get("why")),
            "peak_cf": (float(np.median(e5["fold_peaks"])) if e5["defined"] else None),
            "peak_frac": (float(np.median(e5["fold_peaks"])) / EXP_LAYERS
                          if e5["defined"] else None),
            "band_window": [lo, hi],
            "band_counts_per_fold": counts,
            "median_train_curve": np.median(np.stack(curves), axis=0).round(4).tolist(),
            "yes_no_rate": c["yes_no_rate"],
            "gpu": c["gpu"],
            "npz_sha256": c["npz_sha256"],
        }
        out_cells[task] = entry
        print(f"described: {task}", flush=True)

    out = {
        "banner": "405B STRETCH CELLS — DESCRIPTIVE ONLY, outside every registered "
                  "denominator; the sealed grid-B verdict is unaffected.",
        "model": SLUG, "revision": FROZEN_REVISION, "n_layers": EXP_LAYERS,
        "config": {"rs_seed": RS_SEED, "k_folds": K_FOLDS,
                   "nperm_inner": NPERM_INNER, "nperm_outer": NPERM_OUTER,
                   "nboot": NBOOT, "primary": PRIMARY,
                   "qual": [QUAL_AUROC, ENVELOPE_Q]},
        "provenance": {"machinery_sha256": live_pin,
                       "scorer_sha256": _sha256_file(os.path.abspath(__file__)),
                       "numpy": np.__version__},
        "context_band_counts": {"llama31_8b": "3-8", "llama31_70b": "16-25",
                                "llama33_70b(grid-A)": "25-31"},
        "cells": out_cells,
    }
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, jpath)

    lines = ["# 405B stretch cells — descriptive report (single look)", "",
             out["banner"], ""]
    def _fmt(v, spec):
        return format(v, spec) if v is not None else "UNDEF"

    for task in TASKS:
        e = out_cells[task]
        if not e.get("evaluable"):
            lines.append(f"- **{task}**: NOT EVALUABLE — {e['reason']}")
            continue
        # None-safe rendering (round-10 B1): an evaluable artifact whose cross-fit
        # is undefined (no qualifying training peak in some fold) must still render.
        lines.append(
            f"- **{task}**: Δ_cf {_fmt(e['delta_cf'], '.4f')} "
            f"(perm p {_fmt(e['perm_p_delta'], '.4g')}, "
            f"boot CI {e['boot_ci_5_95_conditional']}); "
            f"peak_cf {_fmt(e['peak_cf'], '.0f')}/126 "
            f"(frac {_fmt(e['peak_frac'], '.3f')}); "
            f"J_cf {_fmt(e['J_cf'], '.4f')}, R_cf {_fmt(e['R_cf'], '.4f')}"
            + (f" (E6 undefined: {e['e6_undefined_why']})" if e['e6_undefined_why'] else "")
            + f"; band counts/fold {e['band_counts_per_fold']} in window {e['band_window']}"
            + f"; yes/no rate {e['yes_no_rate']}")
    lines += ["", f"Band context: 8B 3–8, 3.1-70B 16–25, 3.3-70B(grid-A) 25–31.", ""]
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, mpath)
    print(f"wrote {jpath}\nwrote {mpath}")


if __name__ == "__main__":
    main()
