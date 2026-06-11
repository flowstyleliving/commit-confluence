#!/usr/bin/env python3
"""
analyze_universality - the pre-registered descriptive analyses E1/E2/E3 (amendment A7).

Runs ONLY on the per-cell matrices (.npz) + profiles emitted by run_seal.py - zero new model
forwards. All three are DESCRIPTIVE (non-gating): they inform the universal-vs-per-deployment
question; they cannot pass or fail the seal.

E1  LOMO universality probe: per task, rank-transform each cell within each model, pool the
    other 9 models, select (cell, sign) on the pool, evaluate on the held-out model.
    Two tables: (i) the LOMO-winner per holdout (honest dispatcher universality);
    (ii) the full cell x holdout landscape (descriptive, multiplicity-prone - flagged).
    Interpretation guide (pre-registered): a cell that, when pool-selected, holds
    AUROC > 0.55 on >= 8/10 holdouts = first evidence FOR partial universality; nothing
    surviving = cements the per-deployment framing. Includes a per-cell SIGN-STABILITY audit
    (sign-universal cells would license a fixed-orientation screener with per-deployment
    thresholds - a middle ground between universal and per-deployment).

E2  Task-transfer matrix: per model, apply the task-A profile winner (cell + full-sample
    sign) to the task-B matrix and report the transfer AUROC, both directions, for the
    primary and geometric winners. Measures whether per-MODEL calibration suffices.

E3  Label-efficiency curve: per (model, task), stratified subsamples n in {50,100,150},
    R repeats, re-run the sealed nested-OOB selector (nboot 1000), report the fraction of
    repeats deployable. Prices the labeling cost of per-deployment calibration.

Usage:
  python stage_b/analyze_universality.py --profiles-dir stage_b/profiles --out stage_b/universality.json
"""
import sys, os, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC
SEAL = CC.SEAL


def auroc_fixed(y, s, sign):
    """AUROC of s oriented by a FIXED sign (no refitting on the eval side)."""
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s, dtype=np.float64)
    f = np.isfinite(s)
    if f.sum() < 4 or len(np.unique(y[f])) < 2:
        return None
    return float(roc_auc_score(y[f], s[f] * sign))


def load_cells(profiles_dir):
    """-> {(slug, task): {M, y, panel(with fusion), labels_by_cell, profile}}"""
    cells = {}
    for npz in sorted(glob.glob(os.path.join(profiles_dir, "*", "*.matrix.npz"))):
        task = os.path.basename(os.path.dirname(npz))
        slug = os.path.basename(npz).replace(".matrix.npz", "")
        d = np.load(npz, allow_pickle=False)
        M = d["score_matrix"]; y = d["labels"]
        panel = [tuple(c) for c in json.loads(str(d["panel"]))]
        # append the pre-registered fusion cells so all analyses see the deployed 29-cell panel
        M, panel, _ = CC.append_fusion_columns(M, panel, slug, task)
        profp = npz.replace(".matrix.npz", ".profile.json")
        prof = json.load(open(profp)) if os.path.exists(profp) else None
        cells[(slug, task)] = {"M": M, "y": y, "panel": panel,
                               "labels": [SEAL._cell_label(c) for c in panel], "profile": prof}
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# E1 - LOMO + sign stability
# ──────────────────────────────────────────────────────────────────────────────
def lomo(cells, task):
    slugs = sorted({s for (s, t) in cells if t == task})
    if len(slugs) < 3:
        return {"skipped": f"only {len(slugs)} models with matrices for {task}"}
    ref_labels = cells[(slugs[0], task)]["labels"]
    for s in slugs:
        if cells[(s, task)]["labels"] != ref_labels:
            return {"error": f"panel mismatch across models ({s})"}
    n_cells = len(ref_labels)
    ranked = {s: np.column_stack([CC._rank01(cells[(s, task)]["M"][:, j])
                                  for j in range(n_cells)]) for s in slugs}
    ys = {s: cells[(s, task)]["y"] for s in slugs}

    winners, landscape = [], {lab: {} for lab in ref_labels}
    for hold in slugs:
        pool = [s for s in slugs if s != hold]
        Mp = np.vstack([ranked[s] for s in pool])
        yp = np.concatenate([ys[s] for s in pool])
        best = (None, -1.0, 0, None)  # (j, |auc-.5|, sign, auc)
        for j in range(n_cells):
            auc, sign, _ = SEAL._score_candidate(Mp[:, j], yp)
            if np.isfinite(auc):
                landscape[ref_labels[j]][hold] = auroc_fixed(ys[hold], ranked[hold][:, j], sign)
                if abs(auc - 0.5) > best[1]:
                    best = (j, abs(auc - 0.5), sign, auc)
        j, _, sign, pool_auc = best
        winners.append({"holdout": hold, "winner": ref_labels[j] if j is not None else None,
                        "pool_auroc": None if pool_auc is None else round(pool_auc, 4),
                        "sign": sign,
                        "holdout_auroc": auroc_fixed(ys[hold], ranked[hold][:, j], sign)
                        if j is not None else None})

    # sign-stability audit on the RAW columns (sign is rank-invariant; raw for transparency)
    sign_audit = {}
    for j, lab in enumerate(ref_labels):
        signs = []
        for s in slugs:
            _, sgn, _ = SEAL._score_candidate(cells[(s, task)]["M"][:, j], cells[(s, task)]["y"])
            if sgn != 0:
                signs.append(int(sgn))
        if signs:
            agree = max(signs.count(1), signs.count(-1)) / len(signs)
            sign_audit[lab] = {"n_models": len(signs), "agreement": round(agree, 3),
                               "modal_sign": 1 if signs.count(1) >= signs.count(-1) else -1}
    surv = {lab: sum(1 for v in d.values() if v is not None and v > 0.55)
            for lab, d in landscape.items()}
    return {"holdout_winners": winners,
            "winner_survival": sum(1 for w in winners
                                   if w["holdout_auroc"] is not None and w["holdout_auroc"] > 0.55),
            "n_holdouts": len(slugs),
            "landscape_cells_gt055": {k: v for k, v in sorted(surv.items(), key=lambda kv: -kv[1])
                                      if v >= max(1, len(slugs) - 2)},
            "landscape_note": "full cell x holdout matrix is multiplicity-prone; descriptive only",
            "landscape": landscape, "sign_stability": sign_audit}


# ──────────────────────────────────────────────────────────────────────────────
# E2 - task transfer
# ──────────────────────────────────────────────────────────────────────────────
def transfer(cells):
    tasks = sorted({t for (_, t) in cells})
    if len(tasks) != 2:
        return {"skipped": f"need exactly 2 tasks, have {tasks}"}
    ta, tb = tasks
    out = []
    for slug in sorted({s for (s, _) in cells}):
        if (slug, ta) not in cells or (slug, tb) not in cells:
            continue
        row = {"slug": slug}
        for src, dst in [(ta, tb), (tb, ta)]:
            prof = cells[(slug, src)]["profile"]
            if not prof:
                continue
            for ep_key, ep_name in [("primary_full_panel", "primary"),
                                    ("secondary_geometric_only", "geometric")]:
                ep = prof.get(ep_key) or {}
                w = ep.get("winner")
                wm = (ep.get("full_sample_marginals") or {}).get(w) or {}
                sign = wm.get("sign")
                if not w or sign in (None, 0):
                    continue
                dcell = cells[(slug, dst)]
                if w not in dcell["labels"]:
                    continue
                j = dcell["labels"].index(w)
                row[f"{ep_name}:{src}->{dst}"] = {
                    "cell": w, "src_auroc": (wm or {}).get("auroc"),
                    "transfer_auroc": auroc_fixed(dcell["y"], dcell["M"][:, j], sign)}
        out.append(row)
    vals = [v["transfer_auroc"] for r in out for k, v in r.items()
            if isinstance(v, dict) and v.get("transfer_auroc") is not None]
    return {"rows": out,
            "median_transfer_auroc": round(float(np.median(vals)), 4) if vals else None,
            "frac_transfer_gt055": round(float(np.mean([v > 0.55 for v in vals])), 3) if vals else None}


# ──────────────────────────────────────────────────────────────────────────────
# E3 - label efficiency
# ──────────────────────────────────────────────────────────────────────────────
def label_efficiency(cells, sizes=(50, 100, 150), repeats=10, nboot=1000, seed=20260613):
    out = {}
    for (slug, task), c in sorted(cells.items()):
        M, y, panel = c["M"], c["y"], c["panel"]
        geom_keys = {p[2] for p in panel if p[2] not in CC.NON_GEOMETRIC_KEYS}
        gcols = [j for j, p in enumerate(panel) if p[2] in geom_keys]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        res = {}
        for nsub in sizes:
            if nsub >= len(y):
                continue
            dep_full, dep_geom = 0, 0
            for r in range(repeats):
                rng = np.random.RandomState(seed + 1000 * nsub + r)
                kp = max(2, int(round(nsub * len(pos) / len(y))))
                kn = nsub - kp
                if kp > len(pos) or kn > len(neg) or kn < 2:
                    continue
                idx = np.concatenate([rng.choice(pos, kp, replace=False),
                                      rng.choice(neg, kn, replace=False)])
                full = SEAL._nested_bootstrap_oob_auroc(M[idx], y[idx], panel, nboot, seed + r)
                geom = SEAL._nested_bootstrap_oob_auroc(M[idx][:, gcols], y[idx],
                                                        [panel[j] for j in gcols], nboot, seed + r)
                lo_f, lo_g = full.get("oob_auroc_ci_lo"), geom.get("oob_auroc_ci_lo")
                dep_full += bool(lo_f is not None and np.isfinite(lo_f) and lo_f > 0.50)
                dep_geom += bool(lo_g is not None and np.isfinite(lo_g) and lo_g > 0.50)
            res[str(nsub)] = {"frac_deployable_full": round(dep_full / repeats, 3),
                              "frac_deployable_geom": round(dep_geom / repeats, 3)}
        out[f"{slug}|{task}"] = res
        print(f"[E3] {slug}|{task}: {res}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--nboot-labeleff", type=int, default=1000)
    ap.add_argument("--skip", default="", help="comma list from {lomo,transfer,labeleff}")
    a = ap.parse_args()
    skip = set(s for s in a.skip.split(",") if s)

    cells = load_cells(a.profiles_dir)
    print(f"loaded {len(cells)} (model, task) matrices from {a.profiles_dir}", flush=True)
    report = {"n_cells_loaded": len(cells),
              "cells": sorted(f"{s}|{t}" for (s, t) in cells)}
    tasks = sorted({t for (_, t) in cells})
    if "lomo" not in skip:
        report["E1_lomo"] = {t: lomo(cells, t) for t in tasks}
        for t in tasks:
            r = report["E1_lomo"][t]
            if "holdout_winners" in r:
                print(f"\n[E1 {t}] LOMO winner survival: {r['winner_survival']}/{r['n_holdouts']} "
                      f"holdouts > 0.55", flush=True)
                for w in r["holdout_winners"]:
                    print(f"   {w['holdout']:42s} {str(w['winner']):46s} "
                          f"pool={w['pool_auroc']} hold={None if w['holdout_auroc'] is None else round(w['holdout_auroc'], 4)}")
    if "transfer" not in skip:
        report["E2_transfer"] = transfer(cells)
        print(f"\n[E2] median transfer AUROC: {report['E2_transfer'].get('median_transfer_auroc')} "
              f"| frac > 0.55: {report['E2_transfer'].get('frac_transfer_gt055')}", flush=True)
    if "labeleff" not in skip:
        report["E3_label_efficiency"] = label_efficiency(
            cells, repeats=a.repeats, nboot=a.nboot_labeleff)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
