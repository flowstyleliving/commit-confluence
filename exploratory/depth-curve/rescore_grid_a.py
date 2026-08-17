"""Grid-A rescore under the grid-B CANDIDATE endpoint definitions (cross-fitted E5/E6).

PURPOSE — pre-freeze calibration, per the Codex round-2 audit (2026-08-17): the grid-B
bars must be set against grid A rescored with the EXACT new estimators, because the
registered grid-A rates (dip 8/8, CLIFF 7/8) belong to the old, more permissive
statistics. This script does NOT touch RESULTS.json / the registered grid-A verdicts.

Run ONCE after the code audit (single look):
  python rescore_grid_a.py --npz-dir npz/depth_curve
Writes RESCORE_GRID_A.json + RESCORE_GRID_A.md next to this file. Refuses to run if
either output already exists (single-look guard); writes are atomic (tmp + replace);
NO observed statistic is printed or persisted until the entire analysis completes.

Endpoint definitions implemented (Codex round-2 spec + round-3 code-audit fixes):

E5 (cross-fitted terminal-dip contrast), per cell:
  - Fixed stratified 5-fold map over the 200 rows (20 pos / 20 neg held out per fold),
    IDENTICAL across models within a task (labels verified identical).
  - Per fold, on the 4 training folds ONLY: fit each block's direction (raw AUROC >= 0.5
    -> +1 else -1), compute the sign-free training curve, compute a training-rows-only
    shuffled-label envelope (NPERM_INNER perms of the ACTUAL training labels, q97.5 per
    block), apply the qualifying rule (train sign-free AUROC >= 0.65 AND > envelope),
    select the training peak.
  - Any fold without a qualifying training peak => the CELL FAILS E5.
  - Held-out contrast per fold: directed AUROC(peak) - directed AUROC(final block),
    both blocks using their TRAINING-locked directions (never refit on held-out).
  - Delta_c = mean of the 5 held-out contrasts. Cell success: Delta_c >= 0.05.

E6 (cross-fitted directional cliff), per cell (grid-B gatekeeping: confirmatory only
if E5 confirms; computed here for calibration):
  - Per fold: j* = argmax_j [A_train(j+1) - A_train(j)] over ceil(0.5N) <= j < peak_f.
    Empty window or no qualifying peak in any fold => CELL FAILS E6.
  - Held-out directional jump J_f = auc_ho(j*+1) - auc_ho(j*), training-locked
    directions. J_cf = mean over folds.
  - Cross-fitted rise R_f = auc_ho(peak_f) - median over baseline blocks (closed
    interval [0.4N, 0.6N], the registered E2 baseline) of auc_ho(l). R_cf = mean.
  - Cell success: J_cf >= 0.15 AND J_cf > cell's q95 full-procedure permutation null
    AND R_cf > 0 AND J_cf >= 0.5 * R_cf.   (R_cf > 0 added per round-3 MAJOR-7.)

Permutation grid null: NPERM_OUTER WITHIN-FOLD label permutations per task (labels
permuted among the rows of each frozen fold, preserving the 20/20 held-out and 80/80
training design — round-3 MAJOR-1), the SAME permutation applied to all models within
the task, tasks independent; the ENTIRE cross-fit re-run per permutation. Per-task AND
pooled success-count nulls with (1+k)/(B+1) p-values for both E5 and E6 (MAJOR-2/6).

Bootstrap (reported, never gating): NBOOT resamples, rows redrawn with replacement
within (fold x label) strata, IDENTICAL row resamples across models within a task;
entire training selection re-run per resample. Per-cell CONDITIONAL-on-defined CIs
(with undefined fractions), and fixed-denominator aggregate CIs computed only over
replicates where ALL designated cells are defined (round-3 MAJOR-8).

Implementation decisions (D1-D6 audited round 3; D7-D9 added by the round-3 fixes):
  D1 (amended). The row->fold map is FROZEN from the real labels; outer permutations
      act WITHIN each frozen fold, synchronized across models (blessed structure,
      corrected scope per MAJOR-1).
  D2. Grid statistic = success COUNT per task; pooled = elementwise sum of the two
      independent task count arrays paired by permutation index.
  D3. Undefined cell statistics under permutation -> -inf for null quantiles;
      undefined fractions reported per cell.
  D4. The inner envelope applies frozen per-(task, fold) ROW-ORDER draws to the
      CURRENT training labels, so it always permutes the actual training multiset.
  D5. Directions are fit on training raw AUROC; held-out uses the locked direction
      (a_ho if d=+1 else 1-a_ho); no sign-free max() on held-out rows.
  D6. Rank matrices per (cell, fold) are precomputed and reused for the permutation
      null (ranks are label-free); the bootstrap resamples rows and recomputes ranks.
  D7. Permutation-world E6 success uses the cell's observed-null q95 as a plug-in
      fixed threshold (J_p > q95) alongside the fixed rules (J_p >= 0.15, R_p > 0,
      J_p >= 0.5*R_p), enabling a coherent E6 success-count null.
  D8. Monte Carlo conventions: p = (1 + #{null >= obs}) / (B + 1); quantiles of null
      arrays via the empirical order statistic (method="higher"), finite-safe over -inf.
  D9. Aggregate bootstrap CIs use ONLY replicates where every designated cell is
      defined (fixed denominator); the joint undefined fraction is reported.

Frozen run sizes (NOT CLI-mutable — round-3 MAJOR-4): RS_SEED=20260817, K_FOLDS=5,
NPERM_INNER=200, NPERM_OUTER=2000, NBOOT=1000, QUAL_AUROC=0.65, ENVELOPE_Q=97.5,
E5_DIP=0.05, E6_J=0.15, E6_RISE_FRAC=0.5, E6_NULL_Q=95 (empirical, method="higher").
"""
import argparse
import hashlib
import json
import os

import numpy as np

from depth_score import (
    load_cell, EXPECT_N_ROWS, QUAL_AUROC, ENVELOPE_Q, PRIMARY,
    QWEN_TRIO, CONTROL, TASKS, _closed_interval_blocks,
)

RS_SEED = 20260817
K_FOLDS = 5
NPERM_INNER = 200
NPERM_OUTER = 2000
NBOOT = 1000
E5_DIP = 0.05
E6_J = 0.15
E6_RISE_FRAC = 0.5
E6_NULL_Q = 95.0

SLUGS = QWEN_TRIO + [CONTROL]


def _rng(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(RS_SEED + int(h[:12], 16) % (2 ** 31))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mc_p(null_vals, obs):
    """Monte Carlo p with the (1+k)/(B+1) convention (D8). obs must be finite."""
    null_vals = np.asarray(null_vals, dtype=np.float64)
    return float((1 + int((null_vals >= obs).sum())) / (len(null_vals) + 1))


def emp_q(null_vals, q):
    """Empirical order-statistic quantile, finite-safe over -inf (D8)."""
    return float(np.percentile(np.asarray(null_vals, dtype=np.float64), q,
                               method="higher"))


def rank_columns(x):
    """Average ranks per column of x [n, L] (ties averaged), vectorized argsort."""
    n, L = x.shape
    order = np.argsort(x, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.repeat(np.arange(1, n + 1, dtype=np.float64)[:, None], L, axis=1)
    np.put_along_axis(ranks, order, rows, axis=0)
    for li in range(L):  # average ties (rare for continuous metrics; exact anyway)
        col = x[:, li]
        uniq, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
        if len(uniq) != n:
            sums = np.bincount(inv, weights=ranks[:, li])
            ranks[:, li] = (sums / cnt)[inv]
    return ranks


def raw_auroc_from_ranks(ranks, y01):
    """ranks [n, L], y01 [n] in {0,1} -> raw AUROC per block [L] = P(pos > neg)."""
    assert set(np.unique(y01).tolist()) <= {0, 1}, "labels must be binary"
    n1 = int(y01.sum())
    n0 = len(y01) - n1
    assert n1 > 0 and n0 > 0, "degenerate labels"
    return (ranks[y01 == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def raw_auroc_perm_batch(ranks, pos_indicators):
    """pos_indicators [P, n] boolean, every row the same positive count.
    Returns [P, L] raw AUROCs (rank-sum identity; ranks are label-free)."""
    counts = pos_indicators.sum(axis=1)
    assert (counts == counts[0]).all(), "unequal positive counts across permutation rows"
    n1 = int(counts[0])
    n0 = pos_indicators.shape[1] - n1
    assert n1 > 0 and n0 > 0, "degenerate permutation labels"
    # NOTE: on arm64/Accelerate this matmul can emit spurious divide/overflow
    # RuntimeWarnings (known numpy FP-flag artifact); output verified bit-exact
    # against the loop form. The assert below enforces actual finiteness.
    sums = pos_indicators.astype(np.float64) @ ranks
    out = (sums - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    assert np.isfinite(out).all(), "non-finite AUROC from permutation batch"
    return out


def make_folds(y, rng):
    """Stratified K-fold row->fold assignment [n], frozen from the real labels."""
    fold_of = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        for f, chunk in enumerate(np.array_split(idx, K_FOLDS)):
            fold_of[chunk] = f
    return fold_of


def within_fold_permutation(y, fold_of, rng):
    """Permute labels among the rows of each frozen fold (D1 amended, MAJOR-1).
    Preserves each fold's label multiset (20/20), hence 80/80 training counts."""
    yp = y.copy()
    for f in range(K_FOLDS):
        rows = np.flatnonzero(fold_of == f)
        yp[rows] = y[rows[rng.permutation(len(rows))]]
    return yp


def crossfit_cell(prim, y, fold_of, inner_perm_rows, rank_cache=None):
    """One full cross-fit of the primary matrix prim [n, L] under labels y.

    inner_perm_rows: list of K arrays [NPERM_INNER, n_tr] of ROW-ORDER permutations of
    that fold's training rows; envelope indicators = (y_tr[perm_rows] == 1) (D4).
    rank_cache: optional list of K (R_tr, R_ho) pairs — valid only when prim's rows
    are unchanged (permutation null), never for bootstrap (D6).
    """
    n, L = prim.shape
    folds = []
    for f in range(K_FOLDS):
        tr = fold_of != f
        ho = ~tr
        y_tr, y_ho = y[tr], y[ho]
        if y_tr.min() == y_tr.max() or y_ho.min() == y_ho.max():
            return {"ok": False, "why": f"degenerate labels in fold {f}"}
        if rank_cache is not None:
            R_tr, R_ho = rank_cache[f]
        else:
            R_tr, R_ho = rank_columns(prim[tr]), rank_columns(prim[ho])
        a_tr = raw_auroc_from_ranks(R_tr, y_tr)
        dirs = np.where(a_tr >= 0.5, 1, -1)
        A_tr = np.maximum(a_tr, 1.0 - a_tr)

        env_ind = (y_tr[inner_perm_rows[f]] == 1)
        env = raw_auroc_perm_batch(R_tr, env_ind)
        env_sf = np.maximum(env, 1.0 - env)
        env_q = np.percentile(env_sf, ENVELOPE_Q, axis=0)

        qual = (A_tr >= QUAL_AUROC) & (A_tr > env_q)
        if not qual.any():
            return {"ok": False, "why": f"no qualifying training peak in fold {f}"}
        peak = int(np.argmax(np.where(qual, A_tr, -np.inf)))

        a_ho = raw_auroc_from_ranks(R_ho, y_ho)
        auc_ho = np.where(dirs == 1, a_ho, 1.0 - a_ho)  # training-locked directions

        folds.append({"peak": peak, "A_tr": A_tr, "auc_ho": auc_ho})
    return {"ok": True, "folds": folds}


def e5_from_crossfit(cf, L):
    if not cf["ok"]:
        return {"defined": False, "why": cf["why"], "delta": None, "success": False}
    contrasts = [fd["auc_ho"][fd["peak"]] - fd["auc_ho"][L - 1] for fd in cf["folds"]]
    delta = float(np.mean(contrasts))
    return {"defined": True, "delta": delta, "fold_contrasts": [float(c) for c in contrasts],
            "fold_peaks": [fd["peak"] for fd in cf["folds"]],
            "success": bool(delta >= E5_DIP)}


def e6_from_crossfit(cf, L):
    if not cf["ok"]:
        return {"defined": False, "why": cf["why"], "J": None, "R": None}
    lo = int(np.ceil(0.5 * L))
    blo, bhi = _closed_interval_blocks(L, 0.4, 0.6)
    Js, Rs, jstars = [], [], []
    for fd in cf["folds"]:
        peak = fd["peak"]
        if peak < lo + 1:
            return {"defined": False, "why": f"empty E6 window (peak {peak} < {lo + 1})",
                    "J": None, "R": None}
        j_candidates = np.arange(lo, peak)  # j in [ceil(0.5N), peak)
        jumps_tr = fd["A_tr"][j_candidates + 1] - fd["A_tr"][j_candidates]
        jstar = int(j_candidates[np.argmax(jumps_tr)])
        J = float(fd["auc_ho"][jstar + 1] - fd["auc_ho"][jstar])
        R = float(fd["auc_ho"][peak] - np.median(fd["auc_ho"][blo:bhi]))
        Js.append(J); Rs.append(R); jstars.append(jstar)
    return {"defined": True, "J": float(np.mean(Js)), "R": float(np.mean(Rs)),
            "fold_J": Js, "fold_R": Rs, "fold_jstar": jstars}


def e6_success_fixed(J, R, q95):
    """E6 success under the fixed rules + plug-in q95 threshold (D7, MAJOR-7)."""
    return bool(J >= E6_J and J > q95 and R > 0 and J >= E6_RISE_FRAC * R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    jpath = os.path.join(args.out_dir, "RESCORE_GRID_A.json")
    mpath = os.path.join(args.out_dir, "RESCORE_GRID_A.md")
    for p in (jpath, mpath):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING TO RUN: {p} already exists — this rescore is "
                             f"single-look; delete deliberately if a rerun is intended.")

    # ---- load all 8 cells (registered loader, full validation), verify shared labels
    prim_by, y_by_task, L_by, npz_sha = {}, {}, {}, {}
    for task in TASKS:
        for slug in SLUGS:
            scores, y, metrics, meta = load_cell(args.npz_dir, task, slug)
            prim_by[(task, slug)] = scores[:, :, metrics.index(PRIMARY)]
            L_by[(task, slug)] = scores.shape[1]
            npz_sha[f"{task}/{slug}"] = _sha256_file(
                os.path.join(args.npz_dir, task, f"{slug}.depth.npz"))
            if task in y_by_task:
                assert np.array_equal(y_by_task[task], y), f"labels differ: {task}/{slug}"
            else:
                y_by_task[task] = y
    for task in TASKS:
        y = y_by_task[task]
        assert int(y.sum()) == 100 and len(y) == EXPECT_N_ROWS, f"{task}: not 100/100"

    # ---- frozen per-task design: fold map, inner row-order draws, outer perms, boots
    fold_of, inner_rows, boot_idx = {}, {}, {}
    outer_rngs = {}
    for task in TASKS:
        y = y_by_task[task]
        fold_of[task] = make_folds(y, _rng(task, "folds"))
        per_fold = []
        for f in range(K_FOLDS):
            n_tr = int((fold_of[task] != f).sum())
            g = _rng(task, "inner-env", f)
            per_fold.append(np.stack([g.permutation(n_tr) for _ in range(NPERM_INNER)]))
        inner_rows[task] = per_fold
        outer_rngs[task] = _rng(task, "outer-perm-withinfold")
        g = _rng(task, "boot")
        rows = []
        for _ in range(NBOOT):
            take = np.empty(EXPECT_N_ROWS, dtype=np.int64)
            pos = 0
            for f in range(K_FOLDS):
                for cls in (0, 1):
                    stratum = np.flatnonzero((fold_of[task] == f) & (y == cls))
                    take[pos:pos + len(stratum)] = stratum[g.integers(0, len(stratum),
                                                                      size=len(stratum))]
                    pos += len(stratum)
            rows.append(take)
        boot_idx[task] = np.stack(rows)

    # ---- rank caches per (cell, fold) for the permutation null (D6)
    rank_cache = {}
    for task in TASKS:
        for slug in SLUGS:
            prim = prim_by[(task, slug)]
            cache = []
            for f in range(K_FOLDS):
                tr = fold_of[task] != f
                cache.append((rank_columns(prim[tr]), rank_columns(prim[~tr])))
            rank_cache[(task, slug)] = cache

    # ---- observed statistics (computed SILENTLY — no value printed until the end;
    #      round-3 MAJOR-3 single-look guard)
    cells = {}
    for task in TASKS:
        for slug in SLUGS:
            L = L_by[(task, slug)]
            cf = crossfit_cell(prim_by[(task, slug)], y_by_task[task], fold_of[task],
                               inner_rows[task], rank_cache=rank_cache[(task, slug)])
            cells[(task, slug)] = {"L": L, "e5": e5_from_crossfit(cf, L),
                                   "e6": e6_from_crossfit(cf, L)}
            print(f"observed computed: {task}/{slug}")

    # ---- synchronized WITHIN-FOLD permutation null (full procedure per permutation)
    null_delta = {(t, s): np.full(NPERM_OUTER, -np.inf) for t in TASKS for s in SLUGS}
    null_J = {(t, s): np.full(NPERM_OUTER, -np.inf) for t in TASKS for s in SLUGS}
    null_R = {(t, s): np.full(NPERM_OUTER, np.nan) for t in TASKS for s in SLUGS}
    for task in TASKS:
        y = y_by_task[task]
        for p in range(NPERM_OUTER):
            yp = within_fold_permutation(y, fold_of[task], outer_rngs[task])
            for slug in SLUGS:
                L = L_by[(task, slug)]
                cf = crossfit_cell(prim_by[(task, slug)], yp, fold_of[task],
                                   inner_rows[task], rank_cache=rank_cache[(task, slug)])
                e5p = e5_from_crossfit(cf, L)
                e6p = e6_from_crossfit(cf, L)
                if e5p["defined"]:
                    null_delta[(task, slug)][p] = e5p["delta"]
                if e6p["defined"]:
                    null_J[(task, slug)][p] = e6p["J"]
                    null_R[(task, slug)][p] = e6p["R"]
            if (p + 1) % 200 == 0:
                print(f"perm {task}: {p + 1}/{NPERM_OUTER}")

    # per-cell null-derived quantities
    for task in TASKS:
        for slug in SLUGS:
            c = cells[(task, slug)]
            nd, nj, nr = (null_delta[(task, slug)], null_J[(task, slug)],
                          null_R[(task, slug)])
            c["e5"]["perm_p"] = (mc_p(nd, c["e5"]["delta"]) if c["e5"]["defined"]
                                 else None)
            c["e5"]["null_undefined_frac"] = float(np.isneginf(nd).mean())
            q95 = emp_q(nj, E6_NULL_Q)
            c["e6"]["null_q95"] = q95
            c["e6"]["null_undefined_frac"] = float(np.isneginf(nj).mean())
            c["e6"]["perm_p_J"] = (mc_p(nj, c["e6"]["J"]) if c["e6"]["defined"]
                                   else None)
            c["e6"]["success"] = (e6_success_fixed(c["e6"]["J"], c["e6"]["R"], q95)
                                  if c["e6"]["defined"] else False)

    # success-count nulls: per task and pooled, for E5 and E6 (MAJOR-2)
    grid = {}
    task_counts_e5, task_counts_e6 = {}, {}
    for task in TASKS:
        cnt5 = np.zeros(NPERM_OUTER, dtype=np.int64)
        cnt6 = np.zeros(NPERM_OUTER, dtype=np.int64)
        for slug in SLUGS:
            nd = null_delta[(task, slug)]
            cnt5 += (nd >= E5_DIP).astype(np.int64)
            nj, nr = null_J[(task, slug)], null_R[(task, slug)]
            q95 = cells[(task, slug)]["e6"]["null_q95"]
            nr_safe = np.nan_to_num(nr, nan=-1.0)  # undefined R -> fails R>0
            ok6 = ((nj >= E6_J) & (nj > q95) & (nr_safe > 0)
                   & (nj >= E6_RISE_FRAC * nr_safe))
            cnt6 += ok6.astype(np.int64)
        task_counts_e5[task], task_counts_e6[task] = cnt5, cnt6
        obs5 = sum(1 for s in SLUGS if cells[(task, s)]["e5"]["success"])
        obs6 = sum(1 for s in SLUGS if cells[(task, s)]["e6"]["success"])
        grid[task] = {
            "e5_success_count_observed": obs5,
            "p_grid_e5_count": mc_p(cnt5, obs5),
            "null_e5_count_mean": float(cnt5.mean()),
            "e6_success_count_observed": obs6,
            "p_grid_e6_count": mc_p(cnt6, obs6),
            "null_e6_count_mean": float(cnt6.mean()),
        }
    pooled5 = int(sum(grid[t]["e5_success_count_observed"] for t in TASKS))
    pooled6 = int(sum(grid[t]["e6_success_count_observed"] for t in TASKS))
    pooled_null5 = task_counts_e5[TASKS[0]] + task_counts_e5[TASKS[1]]
    pooled_null6 = task_counts_e6[TASKS[0]] + task_counts_e6[TASKS[1]]
    grid["_POOLED"] = {
        "e5_success_observed_of_8": pooled5,
        "p_grid_e5_pooled": mc_p(pooled_null5, pooled5),
        "e6_success_observed_of_8": pooled6,
        "p_grid_e6_pooled": mc_p(pooled_null6, pooled6),
    }

    # ---- bootstrap (reported): conditional per-cell CIs + fixed-denominator aggregates
    boot_delta = {(t, s): np.full(NBOOT, np.nan) for t in TASKS for s in SLUGS}
    for task in TASKS:
        y = y_by_task[task]
        for b in range(NBOOT):
            take = boot_idx[task][b]
            yb = y[take]
            fb = fold_of[task][take]  # strata resampling preserves fold counts
            for slug in SLUGS:
                L = L_by[(task, slug)]
                cf = crossfit_cell(prim_by[(task, slug)][take], yb, fb,
                                   inner_rows[task])
                e5b = e5_from_crossfit(cf, L)
                if e5b["defined"]:
                    boot_delta[(task, slug)][b] = e5b["delta"]
            if (b + 1) % 200 == 0:
                print(f"boot {task}: {b + 1}/{NBOOT}")
    for task in TASKS:
        for slug in SLUGS:
            v = boot_delta[(task, slug)]
            fin = v[~np.isnan(v)]
            cells[(task, slug)]["e5"]["boot_ci_5_95_conditional"] = (
                [float(np.percentile(fin, 5)), float(np.percentile(fin, 95))]
                if len(fin) else None)
            cells[(task, slug)]["e5"]["boot_undefined_frac"] = float(np.isnan(v).mean())
        stack = np.stack([boot_delta[(task, s)] for s in SLUGS])  # [4, NBOOT]
        all_def = ~np.isnan(stack).any(axis=0)
        agg = stack[:, all_def].mean(axis=0)
        grid[task]["aggregate_delta_boot_ci_5_95_alldef"] = (
            [float(np.percentile(agg, 5)), float(np.percentile(agg, 95))]
            if all_def.any() else None)
        grid[task]["aggregate_joint_undefined_frac"] = float((~all_def).mean())
    stack8 = np.stack([boot_delta[(t, s)] for t in TASKS for s in SLUGS])  # [8, NBOOT]
    all_def8 = ~np.isnan(stack8).any(axis=0)  # tasks paired by replicate index
    agg8 = stack8[:, all_def8].mean(axis=0)
    grid["_POOLED"]["aggregate8_delta_boot_ci_5_95_alldef"] = (
        [float(np.percentile(agg8, 5)), float(np.percentile(agg8, 95))]
        if all_def8.any() else None)
    grid["_POOLED"]["aggregate8_joint_undefined_frac"] = float((~all_def8).mean())

    # ---- JSON-safety (round-4 finding): a q95 over an all-undefined null is -inf,
    # which default json.dump would emit as non-standard -Infinity. Encode as null +
    # explicit flag; computation above already consumed the numeric value.
    for (t, s) in list(cells.keys()):
        c = cells[(t, s)]
        q = c["e6"]["null_q95"]
        c["e6"]["null_q95_is_neginf"] = bool(np.isneginf(q))
        c["e6"]["null_q95"] = float(q) if np.isfinite(q) else None

    # ---- write outputs (atomic; nothing observed was printed before this point)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    out = {
        "banner": "GRID-A RESCORE under grid-B CANDIDATE endpoints — pre-freeze "
                  "calibration artifact. Registered grid-A results are unchanged.",
        "config": {"rs_seed": RS_SEED, "k_folds": K_FOLDS, "nperm_inner": NPERM_INNER,
                   "nperm_outer": NPERM_OUTER, "nboot": NBOOT,
                   "qual_auroc": QUAL_AUROC, "envelope_q": ENVELOPE_Q,
                   "e5_dip": E5_DIP, "e6_j": E6_J, "e6_rise_frac": E6_RISE_FRAC,
                   "e6_null_q": E6_NULL_Q, "primary": PRIMARY,
                   "decisions": "D1(amended within-fold perms), D2, D3, D4, D5, D6, "
                                "D7(plug-in q95 for perm E6), D8((1+k)/(B+1), "
                                "method=higher), D9(all-defined aggregates)"},
        "provenance": {
            "npz_sha256": npz_sha,
            "rescorer_sha256": _sha256_file(os.path.join(src_dir, "rescore_grid_a.py")),
            "depth_score_sha256": _sha256_file(os.path.join(src_dir, "depth_score.py")),
            "numpy_version": np.__version__,
            "fold_map_sha256": {t: hashlib.sha256(fold_of[t].tobytes()).hexdigest()
                                for t in TASKS},
            "fold_map": {t: fold_of[t].tolist() for t in TASKS},
        },
        "cells": {f"{t}/{s}": c for (t, s), c in cells.items()},
        "grid": grid,
    }
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)  # any stray non-finite fails loudly
    os.replace(tmp, jpath)

    lines = ["# Grid-A rescore — cross-fitted E5/E6 (candidate grid-B endpoints)", "",
             out["banner"], "",
             "| task | model | E5 Δ_cf | E5 ok | perm p | boot CI (cond.) | E6 J_cf | E6 R_cf | E6 null q95 | E6 ok |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for slug in SLUGS:
            c = cells[(task, slug)]
            e5, e6 = c["e5"], c["e6"]
            d = f"{e5['delta']:.4f}" if e5["defined"] else f"UNDEF ({e5['why']})"
            j = f"{e6['J']:.4f}" if e6["defined"] else f"UNDEF ({e6.get('why', '')})"
            r = f"{e6['R']:.4f}" if e6["defined"] else "—"
            ci = e5.get("boot_ci_5_95_conditional")
            q95s = ("-inf" if e6.get("null_q95_is_neginf")
                    else round(e6["null_q95"], 4))
            lines.append(
                f"| {task} | {slug} | {d} | {'Y' if e5['success'] else 'N'} | "
                f"{e5.get('perm_p')} | {ci} | {j} | {r} | "
                f"{q95s} | {'Y' if e6['success'] else 'N'} |")
    for task in TASKS:
        g = grid[task]
        lines.append("")
        lines.append(f"**{task}**: E5 {g['e5_success_count_observed']}/4 "
                     f"(p={g['p_grid_e5_count']}), E6 {g['e6_success_count_observed']}/4 "
                     f"(p={g['p_grid_e6_count']}), aggregate Δ CI "
                     f"{g['aggregate_delta_boot_ci_5_95_alldef']} "
                     f"(joint undef {g['aggregate_joint_undefined_frac']})")
    gp = grid["_POOLED"]
    lines += ["", f"**Pooled**: E5 {gp['e5_success_observed_of_8']}/8 "
                  f"(p={gp['p_grid_e5_pooled']}), E6 {gp['e6_success_observed_of_8']}/8 "
                  f"(p={gp['p_grid_e6_pooled']}), 8-cell aggregate Δ CI "
                  f"{gp['aggregate8_delta_boot_ci_5_95_alldef']} "
                  f"(joint undef {gp['aggregate8_joint_undefined_frac']})", ""]
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, mpath)
    print(f"wrote {jpath}\nwrote {mpath}")
    print(f"POOLED: E5 {gp['e5_success_observed_of_8']}/8 (p={gp['p_grid_e5_pooled']}), "
          f"E6 {gp['e6_success_observed_of_8']}/8 (p={gp['p_grid_e6_pooled']})")


if __name__ == "__main__":
    main()
