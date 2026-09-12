#!/usr/bin/env python3
"""Operating points for the CC selector: detection rate at a fixed false-alarm rate.

AUROC says how well a score ranks. A deployer needs to know what happens at a
threshold: of the items that should be flagged, how many are, and how many
clean items are flagged along the way. This script reports, per deployment,
the detection rate (TPR) and realized false-alarm rate (FPR) of the
geometric-only selector at target false-alarm rates of 5% and 10%, with
precision.

No label of a scored row is used to fit anything. The script replays the
registered nested out-of-bag bootstrap with the SAME draws: inside each resample
the signal and its sign are chosen on the in-bag rows exactly as the sealed
selector chooses them, the alarm threshold is set on the in-bag negatives, and
all three are then applied unchanged to the out-of-bag rows.

One transductive detail, inherited from the registered pipeline and NOT
introduced here: the two fusion columns rank-transform each component over all
rows of the deployment before any resampling, so they depend on the scores
(never the labels) of rows that later fall out of bag.
Like the paper's OOB AUROC, the result describes the selection PROCEDURE, not a
single named signal, because in-bag winners can differ across resamples.

Self-check: the same loop recomputes each deployment's OOB AUROC median, 95% CI
lower bound, resample count and in-bag winner tally, and requires all four to
equal the published profile values. The winner tally is the strong one: it is a
per-cell histogram over resamples, so matching it means the draws and the in-bag
selections agree, not merely two summary statistics. If any deployment does not
reproduce, the script exits without writing results.

Positive class = label 1 (contradiction / wrong answer / hallucinated
candidate), the class the selector's locked sign orients upward.

Descriptive and post-registration: no endpoint, bar or verdict depends on it.
No model run; it reads the published matrices and profiles only.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import confluence_calibrator as CC  # noqa: E402

SEAL = CC.SEAL
TARGET_FPRS = (0.05, 0.10)
PROJECTED_PREVALENCE = 0.10  # precision re-expressed at a 10% base rate (projection only)
REPRO_TOL = 1e-12


def row_resamples(n, n_boot, seed):
    """Draws of the sealed row bootstrap (_nested_bootstrap_oob_auroc)."""
    rng = np.random.RandomState(seed + 1)
    for _ in range(n_boot):
        in_bag = rng.randint(0, n, size=n)
        in_bag_set = set(in_bag.tolist())
        oob = np.array([i for i in range(n) if i not in in_bag_set], dtype=np.int64)
        yield in_bag, oob


def cluster_resamples(stem_ids, n_boot, seed):
    """Draws of the BENCH stem-cluster bootstrap (_cluster_bootstrap_oob_auroc)."""
    stem_ids = np.asarray(stem_ids).astype(str)
    ordered = list(dict.fromkeys(stem_ids.tolist()))
    if len(ordered) == len(stem_ids):
        # The registered selector reduces exactly to the row path here.
        yield from row_resamples(len(stem_ids), n_boot, seed)
        return
    rows_by_stem = {s: np.flatnonzero(stem_ids == s).astype(np.int64) for s in ordered}
    n_groups = len(ordered)
    rng = np.random.RandomState(seed + 1)
    for _ in range(n_boot):
        sampled = rng.randint(0, n_groups, size=n_groups)
        sampled_set = set(sampled.tolist())
        in_bag = np.concatenate([rows_by_stem[ordered[g]] for g in sampled])
        oob_groups = [g for g in range(n_groups) if g not in sampled_set]
        oob = (np.concatenate([rows_by_stem[ordered[g]] for g in oob_groups])
               if oob_groups else np.array([], dtype=np.int64))
        yield in_bag, oob


def load_deployment(profile_path):
    prof = json.load(open(profile_path))
    npz = np.load(profile_path.replace(".profile.json", ".matrix.npz"), allow_pickle=True)
    M = npz["score_matrix"].astype(np.float64)
    y = npz["labels"].astype(np.int64)
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError(f"{profile_path}: labels are not binary 0/1")
    panel = [tuple(c) for c in json.loads(str(npz["panel"]))]

    prov = prof.get("provenance") or {}
    M, panel, _ = CC.append_fusion_columns(
        M, panel, prof["slug"], prof["benchmark"],
        canonical_benchmark=prov.get("canonical_fusion_benchmark"))
    geom_keys = {c[2] for c in panel if c[2] not in CC.NON_GEOMETRIC_KEYS}
    cols = [j for j, c in enumerate(panel) if c[2] in geom_keys]

    is_bench = "profiles_bench" in profile_path
    if is_bench:
        published = prof["endpoints_by_unit"]["cluster"]["secondary_geometric_only"]
        rows = [json.loads(line) for line in open(os.path.join(REPO, prof["data_path"]))]
        idx = npz["sample_idx"].astype(int)
        if not np.array_equal(np.array([int(rows[i]["label"]) for i in idx]), y):
            raise ValueError(f"{profile_path}: jsonl labels do not match the matrix")
        stems = [str(rows[i]["stem_id"]) for i in idx]
        draws = cluster_resamples(stems, prov["n_bootstrap"], prov["seed"])
        unit = "cluster"
    else:
        published = prof["secondary_geometric_only"]
        draws = row_resamples(len(y), prov["n_bootstrap"], prov["seed"])
        unit = "row"

    return {
        "task": prof["benchmark"], "model": prof["slug"], "unit": unit, "n": int(len(y)),
        "sm": M[:, cols], "y": y, "draws": draws, "published": published,
        "panel_cols": [panel[j] for j in cols],
    }


def analyse(profile_path):
    d = load_deployment(profile_path)
    sm, y = d["sm"], d["y"]
    aucs = []
    winner_counts = {}
    ops = {a: {"tpr": [], "fpr": [], "prec": [], "prec_proj": []} for a in TARGET_FPRS}

    for in_bag, oob in d["draws"]:
        if len(oob) < 4 or len(np.unique(y[oob])) < 2:
            continue
        # Cell selection and sign-lock on the in-bag rows, as the sealed selector does.
        best_j, best_distance, best_sign = -1, -1.0, 0
        for j in range(sm.shape[1]):
            auc, sign, _ = SEAL._score_candidate(sm[in_bag, j], y[in_bag])
            if np.isfinite(auc):
                distance = abs(auc - 0.5)
                if distance > best_distance:
                    best_j, best_distance, best_sign = j, distance, sign
        if best_j < 0:
            continue
        # Counted here, before the OOB validity check, exactly where the sealed
        # selector counts it -- otherwise the tallies would not be comparable.
        winner_counts[best_j] = winner_counts.get(best_j, 0) + 1

        s_oob = sm[oob, best_j] * best_sign
        y_oob = y[oob]
        finite = np.isfinite(s_oob)
        if finite.sum() < 4 or len(np.unique(y_oob[finite])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_oob[finite], s_oob[finite])))

        # Threshold from the in-bag negatives only, then applied unchanged out-of-bag.
        s_in = sm[in_bag, best_j] * best_sign
        neg_in = s_in[(y[in_bag] == 0) & np.isfinite(s_in)]
        so, yo = s_oob[finite], y_oob[finite]
        n_pos, n_neg = int((yo == 1).sum()), int((yo == 0).sum())
        for a in TARGET_FPRS:
            flag = so > np.quantile(neg_in, 1.0 - a)
            tp = int((flag & (yo == 1)).sum())
            fp = int((flag & (yo == 0)).sum())
            tpr, fpr = tp / n_pos, fp / n_neg
            ops[a]["tpr"].append(tpr)
            ops[a]["fpr"].append(fpr)
            ops[a]["prec"].append(tp / (tp + fp) if tp + fp else np.nan)
            pi = PROJECTED_PREVALENCE
            denom = tpr * pi + fpr * (1 - pi)
            ops[a]["prec_proj"].append(tpr * pi / denom if denom else np.nan)

    arr = np.asarray(aucs)
    med, lo = float(np.median(arr)), float(np.percentile(arr, 2.5))
    pub = d["published"]
    pub_med, pub_lo = pub["oob_auroc_median"], pub["oob_auroc_ci_lo"]
    tally = {SEAL._cell_label(d["panel_cols"][j]): c for j, c in winner_counts.items() if c > 0}
    tally_matches = tally == pub.get("winner_counts")
    reproduces = (abs(med - pub_med) <= REPRO_TOL and abs(lo - pub_lo) <= REPRO_TOL
                  and tally_matches and len(arr) == pub.get("oob_n_bootstrap_used"))

    out = {
        "task": d["task"], "model": d["model"], "unit": d["unit"], "n": d["n"],
        "n_resamples_used": int(len(arr)),
        "oob_auroc_median": med, "oob_auroc_ci_lo": lo,
        "published_oob_auroc_median": pub_med, "published_oob_auroc_ci_lo": pub_lo,
        "reproduces_published": bool(reproduces),
        "winner_tally_matches": bool(tally_matches),
        # NOT the registered deployability endpoint: this is the AUROC criterion
        # alone, without the commitment-audit and control gates BENCH applies.
        "ci_lo_above_half": bool(lo > 0.50),
        "operating_points": {},
    }
    for a in TARGET_FPRS:
        tpr = np.asarray(ops[a]["tpr"])
        out["operating_points"][f"fpr_target_{a:.2f}"] = {
            "tpr_median": float(np.median(tpr)),
            "tpr_ci": [float(np.percentile(tpr, 2.5)), float(np.percentile(tpr, 97.5))],
            "realized_fpr_median": float(np.median(ops[a]["fpr"])),
            "precision_median_at_sample_base_rate": float(np.nanmedian(ops[a]["prec"])),
            "precision_median_projected_at_10pct": float(np.nanmedian(ops[a]["prec_proj"])),
        }
    return out


def summarize(rows):
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    summary = {}
    for task, rs in sorted(by_task.items()):
        entry = {"n_deployments": len(rs)}
        for a in TARGET_FPRS:
            k = f"fpr_target_{a:.2f}"
            tprs = [r["operating_points"][k]["tpr_median"] for r in rs]
            fprs = [r["operating_points"][k]["realized_fpr_median"] for r in rs]
            projs = [r["operating_points"][k]["precision_median_projected_at_10pct"] for r in rs]
            entry[k] = {
                "tpr_median_across_deployments": float(np.median(tprs)),
                "tpr_min": float(np.min(tprs)), "tpr_max": float(np.max(tprs)),
                "realized_fpr_median_across_deployments": float(np.median(fprs)),
                "projected_precision_at_10pct_median": float(np.nanmedian(projs)),
            }
        summary[task] = entry
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True,
                    help="JSON output path (required, so nothing is written by default)")
    ap.add_argument("--only", default="",
                    help="substring filter on profile paths, for a quick self-check")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    core = sorted(glob.glob(os.path.join(REPO, "stage_b/profiles/*/*.profile.json")))
    bench = sorted(p for p in glob.glob(os.path.join(REPO, "stage_b/profiles_bench/*/*.profile.json"))
                   if "_smoke" not in p)
    paths = [p for p in core + bench if args.only in p]
    if not paths:
        print("No profiles matched.")
        return 1
    print(f"Replaying the registered bootstrap on {len(paths)} deployments "
          f"({args.workers} workers)...", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(analyse, paths))

    failed = [r for r in rows if not r["reproduces_published"]]
    print(f"\nReproduction check: {len(rows) - len(failed)}/{len(rows)} deployments "
          "match their published OOB median and CI lower bound exactly.")
    if failed:
        for r in failed:
            print(f"  MISMATCH {r['task']}/{r['model']}: median {r['oob_auroc_median']:.6f} "
                  f"vs {r['published_oob_auroc_median']:.6f}, ci_lo {r['oob_auroc_ci_lo']:.6f} "
                  f"vs {r['published_oob_auroc_ci_lo']:.6f}")
        print("Refusing to write results: the resamples differ from the registered run.")
        return 1

    summary = summarize(rows)
    print(f"\n{'task':24s} {'n':>3s}  {'TPR@5%FPR':>10s} {'TPR@10%FPR':>11s} "
          f"{'real.FPR@10%':>12s} {'prec@10%prev':>12s}")
    for task, e in summary.items():
        f5, f10 = e["fpr_target_0.05"], e["fpr_target_0.10"]
        print(f"{task:24s} {e['n_deployments']:3d}  {f5['tpr_median_across_deployments']:10.3f} "
              f"{f10['tpr_median_across_deployments']:11.3f} "
              f"{f10['realized_fpr_median_across_deployments']:12.3f} "
              f"{f10['projected_precision_at_10pct_median']:12.3f}")

    with open(args.out, "w") as f:
        json.dump({
            "status": "DESCRIPTIVE, post-registration. Not an endpoint; moves no verdict.",
            "selector": "geometric-only nested OOB, replayed with the registered draws",
            "positive_class": "label 1 (contradiction / wrong answer / hallucinated candidate)",
            "threshold_rule": "quantile (1 - target FPR) of in-bag negative scores; flag if score > threshold (strict, so ties do not alarm)",
            "precision_note": "medians over resamples; a resample that raised no alarm contributes no precision value",
            "ci_lo_above_half_note": "AUROC criterion only; NOT the registered deployability endpoint, which also applies commitment and control gates",
            "fusion_note": "fusion columns are rank-transformed over all rows before resampling (registered pipeline behaviour), so they see out-of-bag scores, never out-of-bag labels",
            "target_fprs": list(TARGET_FPRS),
            "projected_prevalence": PROJECTED_PREVALENCE,
            "summary_by_task": summary,
            "deployments": rows,
        }, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
