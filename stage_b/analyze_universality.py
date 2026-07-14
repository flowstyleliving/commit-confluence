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

A2  BENCH v1.3 registered fixed-cell LOMO: `fusion_rank_mean_geom` only, exact planned
    ten-model denominator, pooled-training sign fit, and strict holdout AUROC > 0.55.

Usage:
  python stage_b/analyze_universality.py --profiles-dir stage_b/profiles --out stage_b/universality.json
"""
import sys, os, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC
SEAL = CC.SEAL
BENCH_PLANNED_SLUGS = [
    "Llama-3.2-3B-Instruct-4bit", "Llama-3.1-8B-Instruct-4bit",
    "Mistral-7B-Instruct-v0.3-4bit", "Mistral-Nemo-Instruct-2407-4bit",
    "Phi-3.5-mini-instruct-4bit", "Phi-4-mini-instruct-4bit",
    "Qwen2.5-7B-Instruct-4bit", "Qwen3-1.7B-4bit", "Qwen3-8B-4bit",
    "gemma-3-4b-it-4bit",
]

# bench/1.3 Amendment A2: single source of truth inside the analysis boundary.  This is
# intentionally mirrored, rather than imported from run_bench.py: importing the execution
# harness would couple matrix-only analysis to its MLX/runtime imports.  The extension manifest
# freezes both files, and A2 requires them to be re-stamped together after any future bump.
ACCEPTED_SPEC_VERSION = "bench/1.3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auroc_fixed(y, s, sign):
    """AUROC of s oriented by a FIXED sign (no refitting on the eval side)."""
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s, dtype=np.float64)
    f = np.isfinite(s)
    if f.sum() < 4 or len(np.unique(y[f])) < 2:
        return None
    return float(roc_auc_score(y[f], s[f] * sign))


def matrix_stem_ids(matrix, labels):
    """Load persisted BENCH stems or recover sealed TriviaQA stems from source provenance."""
    if "stem_ids" in matrix.files:
        return matrix["stem_ids"].astype(str)
    if "meta" not in matrix.files or "sample_idx" not in matrix.files:
        return None
    meta = json.loads(str(matrix["meta"]))
    data_path = meta.get("data_path")
    if not data_path:
        return None
    path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    indices = matrix["sample_idx"].astype(np.int64)
    if len(indices) != len(labels) or any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError("matrix sample_idx cannot be aligned to its source rows")
    if not np.array_equal(labels, np.asarray([rows[i]["label"] for i in indices])):
        raise ValueError("matrix labels disagree with source rows at sample_idx")
    stems = []
    for i in indices:
        row = rows[i]
        row_meta = row.get("meta") or {}
        value = row.get("stem_id", row_meta.get("stem_id", row_meta.get("question_id")))
        if value is None:
            return None
        stems.append(str(value))
    return np.asarray(stems, dtype=np.str_)


def load_cells(profiles_dir):
    """-> {(slug, task): {M, y, panel(with fusion), labels_by_cell, profile}}"""
    cells = {}
    summary_path = os.path.join(profiles_dir, "SUMMARY.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    systematic_commitment_fail_tasks = set(
        ((summary.get("endpoints") or {}).get(
            "systematic_commitment_fail_tasks") or []))
    summary_status = {
        (row.get("slug"), row.get("task")): row.get("terminal_status")
        for row in (summary.get("cells") or [])
    }
    for npz in sorted(glob.glob(os.path.join(profiles_dir, "*", "*.matrix.npz"))):
        task = os.path.basename(os.path.dirname(npz))
        slug = os.path.basename(npz).replace(".matrix.npz", "")
        d = np.load(npz, allow_pickle=False)
        M = d["score_matrix"]; y = d["labels"]
        panel = [tuple(c) for c in json.loads(str(d["panel"]))]
        # append the pre-registered fusion cells so all analyses see the deployed 29-cell panel
        canonical_task = CC.canonical_fusion_task_key(task)
        M, panel, fusion_meta = CC.append_fusion_columns(
            M, panel, slug, task, canonical_benchmark=canonical_task)
        profp = npz.replace(".matrix.npz", ".profile.json")
        prof = json.load(open(profp)) if os.path.exists(profp) else None
        stem_ids = matrix_stem_ids(d, y)
        cells[(slug, task)] = {"M": M, "y": y, "panel": panel,
                               "stem_ids": stem_ids,
                               "labels": [SEAL._cell_label(c) for c in panel],
                               "profile": prof, "fusion": fusion_meta,
                               "summary_terminal_status": summary_status.get((slug, task)),
                               "task_behaviorally_infeasible": (
                                   task in systematic_commitment_fail_tasks)}
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
    # SF2: `holdout_winner_survival` below counts holdouts where the (possibly DIFFERENT each
    # time) pooled winner cleared 0.55 - it can read high even if no single cell generalizes.
    # The honest universality number is whether ONE fixed cell survives across holdouts: the max
    # over cells of its >0.55 count. That is the metric the LOMO interpretation guide (A7) keys on.
    fixed_lab = max(surv, key=surv.get) if surv else None
    return {"holdout_winners": winners,
            "holdout_winner_survival": sum(1 for w in winners
                                           if w["holdout_auroc"] is not None and w["holdout_auroc"] > 0.55),
            "fixed_cell_max_survival": {"cell": fixed_lab,
                                        "n_holdouts_gt055": surv.get(fixed_lab, 0) if fixed_lab else 0,
                                        "of_n_holdouts": len(slugs)},
            "n_holdouts": len(slugs),
            "landscape_cells_gt055": {k: v for k, v in sorted(surv.items(), key=lambda kv: -kv[1])
                                      if v >= max(1, len(slugs) - 2)},
            "landscape_note": "full cell x holdout matrix is multiplicity-prone; descriptive only",
            "landscape": landscape, "sign_stability": sign_audit}


def registered_a2(cells, task="halueval_qa", planned_slugs=None):
    """BENCH v1.3 A2 estimator; never substitutes the descriptive max landscape."""
    planned = list(planned_slugs or BENCH_PLANNED_SLUGS)
    if len(planned) != 10 or len(set(planned)) != 10:
        raise ValueError("A2 planned denominator must be exactly ten unique models")

    usable, failures = {}, {}
    for slug in planned:
        cell = cells.get((slug, task))
        if cell is None:
            failures[slug] = "missing matrix"
            continue
        profile = cell.get("profile") or {}
        if cell.get("summary_terminal_status") != "OK":
            failures[slug] = cell.get("summary_terminal_status") or "missing strict summary cell"
            continue
        if cell.get("task_behaviorally_infeasible"):
            failures[slug] = "task systematic commitment failure"
            continue
        if profile.get("spec_version") != ACCEPTED_SPEC_VERSION:
            failures[slug] = "missing/mismatched bench profile"
            continue
        if profile.get("terminal_status") != "OK":
            failures[slug] = profile.get("terminal_status") or "missing terminal status"
            continue
        if (profile.get("endpoint_status_by_unit") or {}).get("cluster") != "OK":
            failures[slug] = "CONTROL-FAIL[cluster]"
            continue
        if len(cell["y"]) != 1000:
            failures[slug] = f"n={len(cell['y'])}, expected 1000"
            continue
        indices = [j for j, panel_cell in enumerate(cell["panel"])
                   if panel_cell[2] == "fusion_rank_mean_geom"]
        if len(indices) != 1:
            failures[slug] = f"fusion_rank_mean_geom columns={len(indices)}"
            continue
        j = indices[0]
        usable[slug] = {"ranked": CC._rank01(cell["M"][:, j]), "y": cell["y"]}

    if len(usable) < 4:
        rows = [{"holdout": slug, "pass": False,
                 "failure_reason": failures.get(slug, "fewer than 3 usable training models")}
                for slug in planned]
        return {"schema_version": "bench-a2/1.2", "task": task,
                "cell": "fusion_rank_mean_geom", "threshold": 0.55,
                "planned_denominator": 10, "n_usable_models": len(usable),
                "aborted": True, "abort_reason": "fewer than 3 usable training models",
                "n_holdouts_gt055": 0, "bar": 8, "pass": False,
                "failures": failures, "holdouts": rows}

    rows = []
    for holdout in planned:
        if holdout not in usable:
            rows.append({"holdout": holdout, "training_slugs": [], "fitted_sign": 0,
                         "pool_auroc": None, "holdout_auroc": None, "pass": False,
                         "failure_reason": failures[holdout]})
            continue
        training = [slug for slug in planned if slug != holdout and slug in usable]
        if len(training) < 3:
            return {"schema_version": "bench-a2/1.2", "task": task,
                    "cell": "fusion_rank_mean_geom", "threshold": 0.55,
                    "planned_denominator": 10, "n_usable_models": len(usable),
                    "aborted": True,
                    "abort_reason": f"holdout {holdout} has fewer than 3 training models",
                    "n_holdouts_gt055": 0, "bar": 8, "pass": False,
                    "failures": failures, "holdouts": rows}
        pooled_scores = np.concatenate([usable[slug]["ranked"] for slug in training])
        pooled_labels = np.concatenate([usable[slug]["y"] for slug in training])
        pool_auc, sign, n_pool = SEAL._score_candidate(pooled_scores, pooled_labels)
        if sign == 0 or not np.isfinite(pool_auc):
            hold_auc, reason = None, "zero/non-finite pooled sign fit"
        else:
            hold_auc = auroc_fixed(
                usable[holdout]["y"], usable[holdout]["ranked"], sign)
            reason = None if hold_auc is not None else "non-finite holdout AUROC"
        passed = bool(hold_auc is not None and hold_auc > 0.55)
        rows.append({"holdout": holdout, "training_slugs": training,
                     "n_training_models": len(training), "n_pool_rows": int(n_pool),
                     "fitted_sign": int(sign),
                     "pool_auroc": None if not np.isfinite(pool_auc) else float(pool_auc),
                     "holdout_auroc": hold_auc, "threshold_strict_gt": 0.55,
                     "pass": passed, "failure_reason": reason})
    n_pass = sum(row["pass"] for row in rows)
    return {"schema_version": "bench-a2/1.2", "task": task,
            "cell": "fusion_rank_mean_geom", "estimator": (
                "within-model _rank01; concatenate other usable models; fit one sign with "
                "_score_candidate; apply fixed sign with auroc_fixed"),
            "planned_slugs": planned, "planned_denominator": 10,
            "n_usable_models": len(usable), "aborted": False,
            "threshold": 0.55, "n_holdouts_gt055": n_pass, "bar": 8,
            "pass": bool(n_pass >= 8), "failures": failures, "holdouts": rows}


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
def _legacy_row_subsample(y, nsub, rng):
    """The pre-A2 row-stratified E3 draw, kept intact for ungrouped tasks."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    kp = max(2, int(round(nsub * len(pos) / len(y))))
    kn = nsub - kp
    if kp > len(pos) or kn > len(neg) or kn < 2:
        return None
    return np.concatenate([rng.choice(pos, kp, replace=False),
                           rng.choice(neg, kn, replace=False)])


def _e3_subsample(y, stem_ids, nsub, rng):
    """Draw a label-budget subsample without splitting a paired stem (A2)."""
    if stem_ids is None:
        stem_ids = np.asarray([f"row:{i}" for i in range(len(y))], dtype=np.str_)
    else:
        stem_ids = np.asarray(stem_ids).astype(str)
    if len(stem_ids) != len(y):
        raise ValueError("E3 stem_ids length must match labels")

    ordered_stems = list(dict.fromkeys(stem_ids.tolist()))
    if len(ordered_stems) == len(y):
        # Unique stem per row is the ungrouped case.  Calling the preserved helper is the
        # exact pre-A2 RNG path; this assertion prevents a mislabeled grouped input from
        # silently entering it.
        assert len(set(stem_ids.tolist())) == len(y)
        return _legacy_row_subsample(y, nsub, rng), "row"

    rows_by_stem = {stem: np.flatnonzero(stem_ids == stem).astype(np.int64)
                    for stem in ordered_stems}
    if any(len(rows) != 2 or set(y[rows].tolist()) != {0, 1}
           for rows in rows_by_stem.values()):
        raise ValueError("grouped E3 currently requires paired two-row {0,1} stems")
    if nsub % 2:
        raise ValueError("paired-stem E3 requires an even label budget")
    n_stems = nsub // 2
    if n_stems > len(ordered_stems):
        return None, "stem"
    chosen = rng.choice(len(ordered_stems), n_stems, replace=False)
    idx = np.concatenate([rows_by_stem[ordered_stems[int(j)]] for j in chosen])
    assert len(idx) == nsub and len(set(stem_ids[idx].tolist())) == n_stems
    return idx, "stem"


def label_efficiency(cells, sizes=(50, 100, 150), repeats=10, nboot=1000, seed=20260613):
    out = {}
    for (slug, task), c in sorted(cells.items()):
        M, y, panel = c["M"], c["y"], c["panel"]
        geom_keys = {p[2] for p in panel if p[2] not in CC.NON_GEOMETRIC_KEYS}
        gcols = [j for j, p in enumerate(panel) if p[2] in geom_keys]
        stem_ids = c.get("stem_ids")
        units = set()
        res = {}
        for nsub in sizes:
            if nsub >= len(y):
                continue
            dep_full, dep_geom = 0, 0
            for r in range(repeats):
                rng = np.random.RandomState(seed + 1000 * nsub + r)
                idx, unit = _e3_subsample(y, stem_ids, nsub, rng)
                units.add(unit)
                if idx is None:
                    continue
                full = SEAL._nested_bootstrap_oob_auroc(M[idx], y[idx], panel, nboot, seed + r)
                geom = SEAL._nested_bootstrap_oob_auroc(M[idx][:, gcols], y[idx],
                                                        [panel[j] for j in gcols], nboot, seed + r)
                lo_f, lo_g = full.get("oob_auroc_ci_lo"), geom.get("oob_auroc_ci_lo")
                dep_full += bool(lo_f is not None and np.isfinite(lo_f) and lo_f > 0.50)
                dep_geom += bool(lo_g is not None and np.isfinite(lo_g) and lo_g > 0.50)
            res[str(nsub)] = {"label_budget": nsub,
                              "subsample_unit": next(iter(units)) if len(units) == 1 else None,
                              "frac_deployable_full": round(dep_full / repeats, 3),
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
    ap.add_argument("--bench-a2", action="store_true",
                    help="score registered BENCH v1.3 A2 on halueval_qa")
    a = ap.parse_args()
    skip = set(s for s in a.skip.split(",") if s)
    if a.bench_a2:
        summary_path = os.path.join(a.profiles_dir, "SUMMARY.json")
        if not os.path.exists(summary_path):
            raise ValueError("registered A2 requires the strict BENCH SUMMARY.json")
        summary = json.load(open(summary_path))
        if summary.get("spec_version") != ACCEPTED_SPEC_VERSION:
            raise ValueError(
                f"registered A2 requires a {ACCEPTED_SPEC_VERSION} strict summary")

    cells = load_cells(a.profiles_dir)
    print(f"loaded {len(cells)} (model, task) matrices from {a.profiles_dir}", flush=True)
    report = {"n_cells_loaded": len(cells),
              "cells": sorted(f"{s}|{t}" for (s, t) in cells)}
    if a.bench_a2:
        report["A2_registered"] = registered_a2(cells)
        a2 = report["A2_registered"]
        print(f"[A2] {a2['n_holdouts_gt055']}/10 > 0.55; bar=8; "
              f"pass={a2['pass']} aborted={a2['aborted']}", flush=True)
    tasks = sorted({t for (_, t) in cells})
    if "lomo" not in skip:
        report["E1_lomo"] = {t: lomo(cells, t) for t in tasks}
        for t in tasks:
            r = report["E1_lomo"][t]
            if "holdout_winners" in r:
                fx = r["fixed_cell_max_survival"]
                print(f"\n[E1 {t}] LOMO holdout-winner survival: "
                      f"{r['holdout_winner_survival']}/{r['n_holdouts']} > 0.55 | "
                      f"best FIXED cell {fx['cell']}: {fx['n_holdouts_gt055']}/{fx['of_n_holdouts']} "
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
