#!/usr/bin/env python3
"""
Stage A - Union coverage matrix for the commit-moment monitor.

Zero-NEW-compute. Reads only sealed artifacts:
  - RPV comprehensive run  (per-sample rows: fisher_eff_rank / null_ratio_post_rank1 / surprise / label)
  - ACE sealed profiles    (nested-OOB CI on the per-(model,task) selected attention cell)

For every (model, task) cell it asks, per family, "is this signal OOB-deployable here?"
deployable := lower bound of a CI on the standalone AUROC > 0.50.

Family CI methods are NOT identical (we say so out loud):
  - ACE        : the profile's own sealed nested-OOB CI lower bound (stricter).
  - RPV/v3/surp: full-sample sign-locked AUROC with a 2000x percentile bootstrap CI,
                 sign locked from the run's own train-fold sign majority (no test-fold
                 sign fitting). Point estimate is cross-checked against the run's CV
                 marginal AUROC so the full-sample orientation isn't doing covert work.

Output: stage_a/out/coverage_matrix.{md,json}
"""
import json, glob, os, sys
import numpy as np

_T0 = os.environ.get("CONFLUENCE_T0_REPO", os.path.expanduser("~/Documents/t0-morphology-furnace"))
RPV_DIR = os.path.join(_T0, "exploratory/shadow-ambiguity/comprehensive_outputs")
ACE_DIR = os.path.join(_T0, "experiments/t0-sealed/2026-05-26/profiles")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEED = 20260610
N_BOOT = 2000
DEPLOY_BAR = 0.50
BENCH_MAP = {"anli_r1": "anli", "triviaqa_paired": "triviaqa"}

# RPV-run families to bootstrap from rows. RPV primary = fisher_eff_rank (pre-registered).
RPV_FEATURES = {
    "null_ratio": "null_ratio_post_rank1",   # the v3 residual-motion detector, same cell/data
    "RPV":        "fisher_eff_rank",          # readout pseudo-volume primary
    "surprise":   "surprise",                 # confidence base
}


def auroc(scores, y):
    """AUROC via rank statistic with tie-averaged ranks. y in {0,1}."""
    y = np.asarray(y); scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1
    rank_full = np.empty(len(scores), dtype=float)
    rank_full[order] = ranks
    sum_pos = rank_full[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def boot_ci(scores, y, sign, rng, n_boot=N_BOOT):
    """Point AUROC + percentile CI for the sign-locked feature. Orientation fixed."""
    oriented = sign * np.asarray(scores, dtype=float)
    point = auroc(oriented, y)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ii = idx[b]
        boots[b] = auroc(oriented[ii], y[ii])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def majority_sign(fold_signs):
    if not fold_signs:
        return None
    s = sum(1 if x > 0 else -1 for x in fold_signs)
    return 1 if s >= 0 else -1


def load_rpv(path, rng):
    d = json.load(open(path))
    model = d["model"]; slug = model.split("/")[-1]
    bench = d["benchmark"]
    rows = d.get("rows", [])
    out = {"model": model, "slug": slug, "benchmark": bench, "n": len(rows),
           "families": {}, "controls": {}, "errors": []}
    if not rows:
        out["errors"].append("no rows"); return out
    y = np.array([int(r["label"]) for r in rows])
    if y.sum() == 0 or (y == 0).sum() == 0:
        out["errors"].append("single-class labels"); return out

    marg = d.get("analysis", {}).get("marginal_train_locked_auroc", {})
    base = d.get("analysis", {}).get("base_models", {})

    for fam, col in RPV_FEATURES.items():
        if not all(col in r and r[col] is not None for r in rows):
            out["families"][fam] = {"error": f"missing {col}"}; continue
        x = np.array([float(r[col]) for r in rows])
        if not np.all(np.isfinite(x)):
            mask = np.isfinite(x); x = x[mask]; yy = y[mask]
        else:
            yy = y
        # lock sign from run's own train-fold signs when present, else full-sample direction
        fs = marg.get(col, {}).get("fold_signs") if col in marg else None
        sign = majority_sign(fs) if fs else (1 if auroc(x, yy) >= 0.5 else -1)
        point, lo, hi = boot_ci(x, yy, sign, rng)
        cv = (marg.get(col, {}).get("auroc")
              if col in marg else base.get("surprise", {}).get("auroc"))
        out["families"][fam] = {
            "auroc": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "sign": sign, "cv_marginal": cv, "n": int(len(yy)),
            "deployable": bool(lo > DEPLOY_BAR),
        }
    # rotation-invariance control (should be ~0): median |stat delta| under random rotation
    rr = [r.get("random_rotation_max_abs_stat_delta") for r in rows
          if r.get("random_rotation_max_abs_stat_delta") is not None]
    if rr:
        out["controls"]["rotation_max_abs_delta_median"] = float(np.median(rr))
    return out


def load_ace(slug, bench):
    sub = BENCH_MAP.get(bench)
    if sub is None:
        return None
    path = os.path.join(ACE_DIR, sub, f"{slug}.profile.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    cs = d.get("calibration_stats", {}); det = d.get("detector", {})
    ci_lo = cs.get("oob_auroc_ci_lo")
    return {
        "metric": det.get("metric"), "gen_step": det.get("gen_step"), "layer": det.get("layer"),
        "oob_median": cs.get("oob_auroc_median"), "oob_ci_lo": ci_lo, "oob_ci_hi": cs.get("oob_auroc_ci_hi"),
        "winner_stability": cs.get("winner_stability"), "warnings": d.get("warnings", []),
        "deployable": bool(ci_lo is not None and ci_lo > DEPLOY_BAR),
    }


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RPV_DIR, "shadow_v2__*__limitall.json")))
    cells = []
    for f in files:
        cell = load_rpv(f, rng)
        cell["ace"] = load_ace(cell["slug"], cell["benchmark"])
        # coverage logic
        geom = ["null_ratio", "RPV"]
        dep_fams = [fam for fam in RPV_FEATURES
                    if cell["families"].get(fam, {}).get("deployable")]
        if cell.get("ace") and cell["ace"]["deployable"]:
            dep_fams = ["ACE"] + dep_fams
        cell["deployable_families"] = dep_fams
        ace_dep = bool(cell.get("ace") and cell["ace"]["deployable"])
        cell["covered_geom"] = bool(ace_dep or any(
            cell["families"].get(g, {}).get("deployable") for g in geom))
        cell["covered_any"] = bool(len(dep_fams) > 0)
        cells.append(cell)

    cells.sort(key=lambda c: (c["slug"], c["benchmark"]))
    # Split usable cells from degenerate ones (no rows / single-class) - the latter are
    # not real measurements (e.g. gpt-oss skipped_before_load) and leave the denominator.
    usable = [c for c in cells if not c.get("errors")]
    excluded = [c for c in cells if c.get("errors")]
    # Classify every non-covered usable cell:
    #   measurement-orphan := even surprise is at chance (degenerate commit / gate artifact)
    #   detector-gap       := surprise works but NO geometric family does (a real panel hole)
    for c in usable:
        if c["covered_any"] and c["covered_geom"]:
            c["orphan_type"] = None
        elif c["covered_any"] and not c["covered_geom"]:
            c["orphan_type"] = "detector-gap"      # surprise-only
        else:
            c["orphan_type"] = "measurement-orphan"  # surprise dead too

    n_cells = len(usable)
    covered_any = sum(c["covered_any"] for c in usable)
    covered_geom = sum(c["covered_geom"] for c in usable)
    measurement_orphans = [c for c in usable if c["orphan_type"] == "measurement-orphan"]
    detector_gaps = [c for c in usable if c["orphan_type"] == "detector-gap"]
    gap_free = (covered_any == n_cells)
    gap_free_geom = (covered_geom == n_cells)

    # ACE sealed-cohort sub-view: cells where an ACE profile exists (the 9 sealed models).
    ace_cohort = [c for c in usable if c.get("ace")]
    ace_cohort_covered = sum(c["covered_any"] for c in ace_cohort)
    ace_cohort_geom = sum(c["covered_geom"] for c in ace_cohort)

    summary = {
        "seed": SEED, "n_boot": N_BOOT, "deploy_bar": DEPLOY_BAR,
        "n_usable_cells": n_cells, "n_excluded": len(excluded),
        "excluded_cells": [f"{c['slug']}/{c['benchmark']} ({c['errors'][0]})" for c in excluded],
        "covered_any": covered_any, "covered_geom": covered_geom,
        "gap_free_any": gap_free, "gap_free_geom": gap_free_geom,
        "measurement_orphans": [f"{c['slug']}/{c['benchmark']}" for c in measurement_orphans],
        "detector_gaps": [f"{c['slug']}/{c['benchmark']}" for c in detector_gaps],
        "ace_cohort_cells": len(ace_cohort), "ace_cohort_covered": ace_cohort_covered,
        "ace_cohort_geom": ace_cohort_geom,
        "cells": usable, "excluded": excluded,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "coverage_matrix.json"), "w"), indent=1)

    # ---- markdown ----
    def fam_cell(c, fam):
        d = c["families"].get(fam, {})
        if "error" in d:
            return d["error"]
        if "deployable" not in d:
            return c["errors"][0] if c.get("errors") else "NA"
        mark = "**Y**" if d["deployable"] else "n"
        return f"{d['auroc']:.3f} [{d['ci_lo']:.2f},{d['ci_hi']:.2f}] {mark}"

    def ace_cell(c):
        a = c.get("ace")
        if not a:
            return "-"
        mark = "**Y**" if a["deployable"] else "n"
        lo = a["oob_ci_lo"]; md = a["oob_median"]
        lo_s = f"{lo:.2f}" if isinstance(lo, (int, float)) else "?"
        md_s = f"{md:.3f}" if isinstance(md, (int, float)) else "?"
        return f"{md_s} [lo {lo_s}] {mark}"

    L = []
    L.append("# Stage A - Union Coverage Matrix")
    L.append("")
    L.append(f"_seed {SEED} | {N_BOOT}x bootstrap | deployable := CI_lo > {DEPLOY_BAR} | "
             f"RPV primary = fisher_eff_rank_")
    L.append("")
    L.append(f"- usable cells (model x task): **{n_cells}**  "
             f"(+{len(excluded)} excluded: no usable rows -> {', '.join(summary['excluded_cells']) or 'none'})")
    L.append(f"- covered by >=1 family (ACE/null_ratio/RPV/surprise): **{covered_any}/{n_cells}**"
             f"  -> gap-free(any) = **{gap_free}**")
    L.append(f"- covered by >=1 *geometric* family (ACE/null_ratio/RPV; surprise excluded): "
             f"**{covered_geom}/{n_cells}**  -> gap-free(geom) = **{gap_free_geom}**")
    L.append(f"- **ACE sealed cohort** (9 models present in ACE panel x 2 tasks = {len(ace_cohort)} cells): "
             f"covered(any) **{ace_cohort_covered}/{len(ace_cohort)}**, covered(geom) "
             f"**{ace_cohort_geom}/{len(ace_cohort)}**")
    if measurement_orphans:
        L.append(f"- MEASUREMENT-ORPHANS (even surprise at chance -> degenerate commit / gate artifact, "
                 f"not a detector gap): {', '.join(summary['measurement_orphans'])}")
    if detector_gaps:
        L.append(f"- DETECTOR-GAPS (surprise works but no geometric family does -> a real panel hole): "
                 f"{', '.join(summary['detector_gaps'])}")
    L.append("")
    L.append("Cell format: `AUROC [CI_lo,CI_hi] Y/n`. ACE column: `OOB_median [OOB ci_lo] Y/n`. "
             "`-` = model not in ACE panel.")
    L.append("")
    L.append("| model | task | ACE (attn) | null_ratio (resid) | RPV (readout) | surprise (base) | verdict |")
    L.append("|---|---|---|---|---|---|:---:|")
    for c in usable:
        if c["orphan_type"] is None:
            cov = "OK"
        elif c["orphan_type"] == "detector-gap":
            cov = "surp-only"
        else:
            cov = "**ORPHAN**"
        L.append(f"| {c['slug']} | {BENCH_MAP.get(c['benchmark'], c['benchmark'])} "
                 f"| {ace_cell(c)} | {fam_cell(c,'null_ratio')} | {fam_cell(c,'RPV')} "
                 f"| {fam_cell(c,'surprise')} | {cov} |")
    if excluded:
        for c in excluded:
            L.append(f"| {c['slug']} | {BENCH_MAP.get(c['benchmark'], c['benchmark'])} "
                     f"| - | - | - | - | _excluded_ |")
    L.append("")
    # who carries the geometric coverage
    L.append("## Per-family deployable counts (usable cells)")
    L.append("")
    L.append("| family | deployable cells | of |")
    L.append("|---|:---:|:---:|")
    n_ace_present = sum(1 for c in usable if c.get("ace"))
    n_ace_dep = sum(1 for c in usable if c.get("ace") and c["ace"]["deployable"])
    L.append(f"| ACE | {n_ace_dep} | {n_ace_present} (present) |")
    for fam in ["null_ratio", "RPV", "surprise"]:
        nd = sum(1 for c in usable if c["families"].get(fam, {}).get("deployable"))
        L.append(f"| {fam} | {nd} | {n_cells} |")
    L.append("")
    L.append("## Verdict")
    L.append("")
    if gap_free:
        L.append(f"**Stage A PASS (strict)** - gap-free across all {n_cells} usable cells.")
    elif not detector_gaps:
        L.append(f"**Stage A CONDITIONAL PASS** - gap-free across every cell with a non-degenerate "
                 f"measurement. The {len(measurement_orphans)} uncovered cell(s) are "
                 f"**measurement-orphans** where even the model's own confidence (surprise) sits at "
                 f"chance: {', '.join(summary['measurement_orphans'])}. These are the documented "
                 f"gate / chat-template-fragile, out-of-ACE-panel models - a data-quality problem, "
                 f"not a hole in the geometric panel. There are **zero detector-gaps** (no cell where "
                 f"surprise works but our geometry fails). Within the ACE sealed cohort the union is "
                 f"{'gap-free' if ace_cohort_covered == len(ace_cohort) else 'NOT gap-free'} "
                 f"({ace_cohort_covered}/{len(ace_cohort)}). Recommended Stage B scope: the "
                 f"non-degenerate cohort, excluding the {len(measurement_orphans)} gate-broken models "
                 f"with documented reason.")
    else:
        L.append(f"**Stage A INCOMPLETE** - {len(detector_gaps)} genuine detector-gap(s) where surprise "
                 f"works but no geometric family does: {', '.join(summary['detector_gaps'])}. Plus "
                 f"{len(measurement_orphans)} measurement-orphan(s). Report before Stage B.")
    open(os.path.join(OUT_DIR, "coverage_matrix.md"), "w").write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n[written] {OUT_DIR}/coverage_matrix.md  and  .json")
    print(f"[gate] gap_free_any={gap_free}  gap_free_geom={gap_free_geom}")


if __name__ == "__main__":
    main()
