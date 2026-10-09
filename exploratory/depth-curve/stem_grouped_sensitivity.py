"""Stem-grouped sensitivity rescore of the depth-curve E5/E6 endpoints — UNREGISTERED.

QUESTION. HaluEval-QA rows are stem-paired: 100 knowledge/question stems, each present
once with a hallucinated answer (label 1) and once with a correct one (label 0). Five
ANLI R1 premises also appear twice, each time with one entailed and one contradicted
hypothesis. The frozen fold builder (rescore_grid_a.make_folds) stratifies by label
only, and the registered permutation null and bootstrap treat rows as exchangeable,
so most stems have their two rows in different folds. Does keeping each stem inside
one fold — and permuting / resampling at the stem level — change the reported
cross-fitted terminal-dip (E5) numbers?

SCOPE. Descriptive sensitivity analysis on banked artifacts. It CANNOT upgrade,
downgrade or amend any registered verdict (grid-A rescore, grid-B §5). The registered
outputs (RESCORE_GRID_A.*, GRID_B_RESULTS.*) are read, never written. The frozen
statistical machinery (rescore_grid_a.py, sha256-pinned below) is imported UNMODIFIED.

ENUMERATED BEFORE ANY OUTPUT WAS INSPECTED
  Cells: the 17 evaluable cells — grid A 8 (Qwen2.5 7B/32B/72B + Llama-3.3-70B x 2
    tasks), grid B 9 (status-ok cells; the 3 aborted cells have no data and stay
    failures in every grid-B count). Grids are NEVER pooled.
  Instrument: final_js_no_bos only (the registered E5/E6 primary).
  Groups: HaluEval = meta.source_idx; ANLI = exact premise string. Asserted: every
    group has size 1 or 2, and every size-2 group holds one label-1 and one label-0 row.
  Part A — reproduction gate. The registered fold map must reproduce the banked
    fold_map_sha256 and every banked observed Delta_cf to < 1e-12, else ABORT.
    Also reported: how many stems the registered fold map splits across folds.
  Part B — fold-draw distributions (isolates the fold-map effect). M_DRAWS row-
    stratified maps (the registered builder, fresh seeds) vs M_DRAWS group-
    stratified maps; BOTH arms use the registered row-level inner envelope, so the
    fold map is the only difference. Per cell: mean/sd of observed Delta_cf over
    defined draws, defined fraction, success fraction (defined AND Delta >= 0.05),
    group-minus-row mean difference with Monte Carlo SE, and the percentile of the
    banked registered Delta_cf within the row-arm distribution.
  Part C — fully grouped inference at ONE designated group-stratified fold map
    (_rng(task, "stem-grouped-designated-folds")): group-respecting inner envelope,
    group-respecting synchronized outer null (NPERM_OUTER), stem-cluster bootstrap
    (NBOOT). Group-respecting = each pair's two labels swap with probability 1/2,
    and singleton labels are shuffled at the SAME scope as the registered design:
    across the whole training set for the inner envelope (registered D4 shuffles all
    n_tr training rows), and within each fold for the outer null (registered D1).
    The swap null tests conditional exchangeability of each pair's two rows, not
    merely Delta_cf = 0. ANLI has one pair per fold, so its pair stratum has no
    bootstrap variability: ANLI intervals are conditional on those five premises
    (5 of 200 rows). Reported per
    cell: Delta_cf, success, perm p, conditional 5/95 bootstrap CI; E6 J/R/null
    q95/success (descriptive only). Per grid: success counts, per-task and pooled
    p_grid, all-defined aggregate CIs. Grid B additionally: the frozen §5 decision
    rule applied COUNTERFACTUALLY (12-cell denominator, aborted cells = failures),
    labelled non-verdict.
  Not covered: E7, E8, E1'', the seven-metric / v-norm follow-ons.

Run (single look; refuses if outputs exist):
  ../../.venv/bin/python stem_grouped_sensitivity.py
Writes STEM_GROUPED_SENSITIVITY.json + .md next to this file.
"""
import argparse
import hashlib
import json
import os
import re
import time

import numpy as np

from depth_score import load_cell, QWEN_TRIO, CONTROL
from rescore_grid_a import (
    K_FOLDS, NPERM_INNER, NPERM_OUTER, NBOOT, E5_DIP, E6_NULL_Q, E6_J, E6_RISE_FRAC,
    _rng, _sha256_file, mc_p, emp_q, rank_columns,
    make_folds, within_fold_permutation, crossfit_cell,
    e5_from_crossfit, e6_from_crossfit, e6_success_fixed,
)
from score_grid_b import (
    RESCORE_PIN_SHA256, PRIMARY, TASKS, EXPECT_N_ROWS, LOCAL_DATA, GRID_B, FAMILIES,
    load_task_labels, load_cell_b,
)

M_DRAWS = 200
REPRO_TOL = 1e-12
_HERE = os.path.dirname(os.path.abspath(__file__))
GRID_A_SLUGS = QWEN_TRIO + [CONTROL]
GRID_A_DIR = os.path.join(_HERE, "npz", "depth_curve")
GRID_B_DIR = os.path.join(_HERE, "npz", "depth_grid_b")


# ── groups ──────────────────────────────────────────────────────────────────────
def load_groups(task, y):
    """Group id per row from the frozen data file. Returns int array [n]."""
    keys = []
    with open(LOCAL_DATA[task], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if task == "halueval_qa":
                keys.append(("src", int(r["meta"]["source_idx"])))
            else:
                m = re.search(r"Premise: (.*?)\nHypothesis:", r["prompt"], re.S)
                if m is None:
                    raise SystemExit(f"{task}: premise not parseable")
                keys.append(("premise", m.group(1)))
    uniq = {}
    gid = np.array([uniq.setdefault(k, len(uniq)) for k in keys], dtype=np.int64)
    if len(gid) != len(y):
        raise SystemExit(f"{task}: group/label length mismatch")
    for g in np.unique(gid):
        rows = np.flatnonzero(gid == g)
        if len(rows) > 2:
            raise SystemExit(f"{task}: group of size {len(rows)}")
        if len(rows) == 2 and sorted(y[rows].tolist()) != [0, 1]:
            raise SystemExit(f"{task}: size-2 group not one-pos/one-neg: {rows}")
    return gid


def units_by_type(gid, y):
    """Partition rows into units: 'pair' -> [pos_row, neg_row]; 'pos1'/'neg1' -> [row]."""
    out = {"pair": [], "pos1": [], "neg1": []}
    for g in np.unique(gid):
        rows = np.flatnonzero(gid == g)
        if len(rows) == 2:
            pos = int(rows[y[rows] == 1][0]); neg = int(rows[y[rows] == 0][0])
            out["pair"].append([pos, neg])
        else:
            out["pos1" if y[rows[0]] == 1 else "neg1"].append([int(rows[0])])
    return out


def make_group_folds(units, n, rng):
    """Group-stratified K-fold: each unit type is dealt evenly across folds, so every
    fold keeps the 20/20 label balance and no pair is ever split."""
    fold_of = np.full(n, -1, dtype=np.int64)
    for typ in ("pair", "pos1", "neg1"):
        us = units[typ]
        order = rng.permutation(len(us))
        for f, chunk in enumerate(np.array_split(order, K_FOLDS)):
            for u in chunk:
                fold_of[us[u]] = f
    assert (fold_of >= 0).all()
    return fold_of


def stems_split(fold_of, units):
    return int(sum(fold_of[a] != fold_of[b] for a, b in units["pair"]))


def row_inner_design(fold_of, rng_parts):
    """The registered row-level inner-envelope design for an arbitrary fold map."""
    per_fold = []
    for f in range(K_FOLDS):
        n_tr = int((fold_of != f).sum())
        g = _rng(*rng_parts, f)
        per_fold.append(np.stack([g.permutation(n_tr) for _ in range(NPERM_INNER)]))
    return per_fold


# ── canonical layout for Part C (observed and bootstrap share one slot layout) ──────
def canonical_layout(fold_of, units):
    """Row order: fold 0..K-1; within a fold, pair units as [pos, neg], then pos
    singletons, then neg singletons. Returns (canon row index [n], slot spec per fold)."""
    canon, spec = [], []
    for f in range(K_FOLDS):
        pairs = sorted([u for u in units["pair"] if fold_of[u[0]] == f])
        pos1 = sorted([u[0] for u in units["pos1"] if fold_of[u[0]] == f])
        neg1 = sorted([u[0] for u in units["neg1"] if fold_of[u[0]] == f])
        start = len(canon)
        for p, q in pairs:
            canon += [p, q]
        canon += pos1 + neg1
        spec.append({"start": start, "n_pairs": len(pairs), "n_pos1": len(pos1),
                     "n_neg1": len(neg1)})
    return np.asarray(canon, dtype=np.int64), spec


def group_perm_for_layout(pair_starts, singleton_pos, n, rng):
    """Row-order permutation (applied as y[perm]) that swaps each pair with prob 1/2
    and permutes the given singleton positions among themselves."""
    perm = np.arange(n)
    swap = rng.random(len(pair_starts)) < 0.5
    for s, sw in zip(pair_starts, swap):
        if sw:
            perm[s], perm[s + 1] = s + 1, s
    if len(singleton_pos):
        sp = np.asarray(singleton_pos)
        perm[sp] = sp[rng.permutation(len(sp))]
    return perm


def layout_positions(spec, fold_mask_fn):
    """Positions (in the sub-array selected by fold_mask_fn) of pair starts and of
    singletons, walking the canonical layout in order."""
    pair_starts, singles, pos = [], [], 0
    for f, s in enumerate(spec):
        if not fold_mask_fn(f):
            continue
        for _ in range(s["n_pairs"]):
            pair_starts.append(pos); pos += 2
        for _ in range(s["n_pos1"] + s["n_neg1"]):
            singles.append(pos); pos += 1
    return pair_starts, singles, pos


def group_inner_design(spec, task):
    """Group-respecting inner envelope per held-out fold f (training = folds != f).
    Singletons are shuffled across ALL training folds — the same scope as the
    registered D4 envelope, which permutes all n_tr training rows."""
    per_fold = []
    for f in range(K_FOLDS):
        ps, ss, n_tr = layout_positions(spec, lambda ff: ff != f)
        g = _rng(task, "stem-grouped-inner-env", f)
        per_fold.append(np.stack([group_perm_for_layout(ps, ss, n_tr, g)
                                  for _ in range(NPERM_INNER)]))
    return per_fold


def group_outer_perm(spec, n, rng):
    """Group-respecting outer permutation: pairs swap; singletons permute WITHIN fold."""
    perm = np.arange(n)
    for s in spec:
        base = s["start"]
        ps = [base + 2 * i for i in range(s["n_pairs"])]
        ss = list(range(base + 2 * s["n_pairs"], base + 2 * s["n_pairs"]
                        + s["n_pos1"] + s["n_neg1"]))
        sub = group_perm_for_layout(ps, ss, n, rng)
        moved = ps + [p + 1 for p in ps] + ss
        perm[moved] = sub[moved]
    return perm


def cluster_boot_take(spec, rng):
    """Stem-cluster bootstrap in canonical coordinates: within each fold, resample
    pair units with replacement among that fold's pairs, and singletons within their
    label stratum. The slot layout (hence the label vector) is preserved exactly."""
    take = []
    for s in spec:
        base = s["start"]
        np_, n1, n0 = s["n_pairs"], s["n_pos1"], s["n_neg1"]
        if np_:
            for k in rng.integers(0, np_, size=np_):
                take += [base + 2 * k, base + 2 * k + 1]
        p0 = base + 2 * np_
        if n1:
            take += (p0 + rng.integers(0, n1, size=n1)).tolist()
        if n0:
            take += (p0 + n1 + rng.integers(0, n0, size=n0)).tolist()
    return np.asarray(take, dtype=np.int64)


# ── loading ───────────────────────────────────────────────────────────────────────
def load_all():
    y_by_task = {t: load_task_labels(t) for t in TASKS}
    cells = {}  # (grid, task, slug) -> (prim [n, L], L)
    for task in TASKS:
        for slug in GRID_A_SLUGS:
            scores, y, metrics, _meta = load_cell(GRID_A_DIR, task, slug)
            if not np.array_equal(y.astype(np.int64), y_by_task[task]):
                raise SystemExit(f"grid-A labels != frozen data: {task}/{slug}")
            cells[("A", task, slug)] = (scores[:, :, metrics.index(PRIMARY)],
                                        scores.shape[1])
        for slug in GRID_B:
            spath = os.path.join(GRID_B_DIR, task, f"{slug}.status.json")
            if json.load(open(spath)).get("status") != "ok":
                continue
            prim, L, _meta, _p = load_cell_b(GRID_B_DIR, task, slug, y_by_task[task])
            cells[("B", task, slug)] = (prim, L)
    return y_by_task, cells


def banked_deltas():
    a = json.load(open(os.path.join(_HERE, "RESCORE_GRID_A.json")))
    b = json.load(open(os.path.join(_HERE, "GRID_B_RESULTS.json")))
    out = {}
    for grid, d in (("A", a), ("B", b)):
        for key, c in d["cells"].items():
            task, slug = key.split("/", 1)
            out[(grid, task, slug)] = c["e5"]
    return out, {"A": a["provenance"]["fold_map_sha256"],
                 "B": b["provenance"]["fold_map_sha256"]}


def summarize(v):
    v = np.asarray(v, dtype=np.float64)
    d = v[~np.isnan(v)]
    return d


# ── main ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=_HERE)
    args = ap.parse_args()
    t0 = time.time()

    live = _sha256_file(os.path.join(_HERE, "rescore_grid_a.py"))
    if live != RESCORE_PIN_SHA256:
        raise SystemExit(f"STAT MACHINERY DRIFT: {live} != {RESCORE_PIN_SHA256}")
    jpath = os.path.join(args.out_dir, "STEM_GROUPED_SENSITIVITY.json")
    mpath = os.path.join(args.out_dir, "STEM_GROUPED_SENSITIVITY.md")
    for p in (jpath, mpath):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING TO RUN: {p} exists — single look.")

    y_by_task, cells = load_all()
    gid = {t: load_groups(t, y_by_task[t]) for t in TASKS}
    units = {t: units_by_type(gid[t], y_by_task[t]) for t in TASKS}
    keys = sorted(cells)
    print(f"loaded {len(keys)} cells; units: "
          + ", ".join(f"{t}: {len(units[t]['pair'])} pairs / "
                      f"{len(units[t]['pos1'])}+{len(units[t]['neg1'])} singletons"
                      for t in TASKS), flush=True)

    # ── Part A: reproduction gate (silent on outcomes; abort on any mismatch) ──────
    banked, banked_sha = banked_deltas()
    reg_fold, reg_inner = {}, {}
    for task in TASKS:
        reg_fold[task] = make_folds(y_by_task[task], _rng(task, "folds"))
        sha = hashlib.sha256(reg_fold[task].tobytes()).hexdigest()
        for grid in ("A", "B"):
            if sha != banked_sha[grid][task]:
                raise SystemExit(f"PART A FAIL: fold map sha {task} grid {grid}")
        reg_inner[task] = row_inner_design(reg_fold[task], (task, "inner-env"))
    reg_delta = {}
    for k in keys:
        grid, task, slug = k
        prim, L = cells[k]
        e5 = e5_from_crossfit(crossfit_cell(prim, y_by_task[task], reg_fold[task],
                                            reg_inner[task]), L)
        bk = banked[k]
        if e5["defined"] != bk["defined"]:
            raise SystemExit(f"PART A FAIL: defined mismatch {k}")
        if e5["defined"] and abs(e5["delta"] - bk["delta"]) >= REPRO_TOL:
            raise SystemExit(f"PART A FAIL: delta mismatch {k}")
        reg_delta[k] = e5["delta"] if e5["defined"] else None
    split_reg = {t: stems_split(reg_fold[t], units[t]) for t in TASKS}
    print(f"Part A passed: {len(keys)} cells reproduce the banked Delta_cf", flush=True)

    # ── Part B: fold-draw distributions ─────────────────────────────────────────────
    draws = {(arm, k): np.full(M_DRAWS, np.nan) for arm in ("row", "group") for k in keys}
    split_row_draws = {t: [] for t in TASKS}
    for task in TASKS:
        y = y_by_task[task]
        tk = [k for k in keys if k[1] == task]
        for m in range(M_DRAWS):
            maps = {"row": make_folds(y, _rng(task, "sens-rowfold", m)),
                    "group": make_group_folds(units[task], len(y),
                                              _rng(task, "sens-groupfold", m))}
            split_row_draws[task].append(stems_split(maps["row"], units[task]))
            assert stems_split(maps["group"], units[task]) == 0
            for arm, fo in maps.items():
                inner = row_inner_design(fo, (task, f"sens-inner-{arm}", m))
                for k in tk:
                    prim, L = cells[k]
                    e5 = e5_from_crossfit(crossfit_cell(prim, y, fo, inner), L)
                    if e5["defined"]:
                        draws[(arm, k)][m] = e5["delta"]
            if (m + 1) % 50 == 0:
                print(f"Part B {task}: {m + 1}/{M_DRAWS} draws "
                      f"({time.time() - t0:.0f}s)", flush=True)
    part_b = {}
    for k in keys:
        r, g = draws[("row", k)], draws[("group", k)]
        rd, gd = summarize(r), summarize(g)
        rec = {
            "row_defined_frac": float(len(rd) / M_DRAWS),
            "group_defined_frac": float(len(gd) / M_DRAWS),
            "row_success_frac": float(np.nansum(r >= E5_DIP) / M_DRAWS),
            "group_success_frac": float(np.nansum(g >= E5_DIP) / M_DRAWS),
            "row_mean": float(rd.mean()) if len(rd) else None,
            "group_mean": float(gd.mean()) if len(gd) else None,
            "row_sd": float(rd.std(ddof=1)) if len(rd) > 1 else None,
            "group_sd": float(gd.std(ddof=1)) if len(gd) > 1 else None,
            "registered_delta": reg_delta[k],
        }
        if len(rd) > 1 and len(gd) > 1:
            rec["group_minus_row"] = float(gd.mean() - rd.mean())
            rec["diff_mc_se"] = float(np.sqrt(gd.var(ddof=1) / len(gd)
                                              + rd.var(ddof=1) / len(rd)))
        else:
            rec["group_minus_row"] = rec["diff_mc_se"] = None
        if reg_delta[k] is not None and len(rd):
            rec["registered_pctile_in_row_arm"] = float((rd <= reg_delta[k]).mean() * 100)
        part_b["/".join(k)] = rec
    part_b_grid = {}
    for grid in ("A", "B"):
        diffs = [part_b["/".join(k)]["group_minus_row"] for k in keys
                 if k[0] == grid and part_b["/".join(k)]["group_minus_row"] is not None]
        part_b_grid[grid] = {
            "n_cells": len(diffs),
            "median_group_minus_row": float(np.median(diffs)) if diffs else None,
            "mean_group_minus_row": float(np.mean(diffs)) if diffs else None,
            "n_negative": int(sum(d < 0 for d in diffs)),
        }

    # ── Part C: fully grouped inference at the designated grouped fold map ──────────
    des_fold, canon, spec, inner_g, y_c = {}, {}, {}, {}, {}
    for task in TASKS:
        y = y_by_task[task]
        des_fold[task] = make_group_folds(units[task], len(y),
                                          _rng(task, "stem-grouped-designated-folds"))
        canon[task], spec[task] = canonical_layout(des_fold[task], units[task])
        assert sorted(canon[task].tolist()) == list(range(len(y)))
        y_c[task] = y[canon[task]]
        inner_g[task] = group_inner_design(spec[task], task)
    fold_c = {t: des_fold[t][canon[t]] for t in TASKS}
    prim_c = {k: cells[k][0][canon[k[1]]] for k in keys}

    rank_cache = {}
    for k in keys:
        fo = fold_c[k[1]]
        rank_cache[k] = [(rank_columns(prim_c[k][fo != f]),
                          rank_columns(prim_c[k][fo == f])) for f in range(K_FOLDS)]
    obs = {}
    for k in keys:
        grid, task, slug = k
        L = cells[k][1]
        cf = crossfit_cell(prim_c[k], y_c[task], fold_c[task], inner_g[task],
                           rank_cache=rank_cache[k])
        obs[k] = {"L": L, "e5": e5_from_crossfit(cf, L), "e6": e6_from_crossfit(cf, L)}

    null_d = {k: np.full(NPERM_OUTER, -np.inf) for k in keys}
    null_J = {k: np.full(NPERM_OUTER, -np.inf) for k in keys}
    null_R = {k: np.full(NPERM_OUTER, np.nan) for k in keys}
    for task in TASKS:
        g = _rng(task, "stem-grouped-outer-perm")
        n = len(y_c[task])
        tk = [k for k in keys if k[1] == task]
        for p in range(NPERM_OUTER):
            yp = y_c[task][group_outer_perm(spec[task], n, g)]
            for k in tk:
                L = cells[k][1]
                cf = crossfit_cell(prim_c[k], yp, fold_c[task], inner_g[task],
                                   rank_cache=rank_cache[k])
                e5p, e6p = e5_from_crossfit(cf, L), e6_from_crossfit(cf, L)
                if e5p["defined"]:
                    null_d[k][p] = e5p["delta"]
                if e6p["defined"]:
                    null_J[k][p], null_R[k][p] = e6p["J"], e6p["R"]
            if (p + 1) % 500 == 0:
                print(f"Part C perm {task}: {p + 1}/{NPERM_OUTER} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    for k in keys:
        c = obs[k]
        c["e5"]["perm_p"] = mc_p(null_d[k], c["e5"]["delta"]) if c["e5"]["defined"] else None
        c["e5"]["null_undefined_frac"] = float(np.isneginf(null_d[k]).mean())
        q95 = emp_q(null_J[k], E6_NULL_Q)
        c["e6"]["_q95_raw"] = q95
        c["e6"]["success"] = (e6_success_fixed(c["e6"]["J"], c["e6"]["R"], q95)
                              if c["e6"]["defined"] else False)

    boot_d = {k: np.full(NBOOT, np.nan) for k in keys}
    for task in TASKS:
        g = _rng(task, "stem-grouped-boot")
        tk = [k for k in keys if k[1] == task]
        for b in range(NBOOT):
            take = cluster_boot_take(spec[task], g)
            assert np.array_equal(y_c[task][take], y_c[task])
            assert np.array_equal(fold_c[task][take], fold_c[task])
            for k in tk:
                L = cells[k][1]
                e5b = e5_from_crossfit(crossfit_cell(prim_c[k][take], y_c[task],
                                                     fold_c[task], inner_g[task]), L)
                if e5b["defined"]:
                    boot_d[k][b] = e5b["delta"]
            if (b + 1) % 250 == 0:
                print(f"Part C boot {task}: {b + 1}/{NBOOT} ({time.time() - t0:.0f}s)",
                      flush=True)
    for k in keys:
        v = boot_d[k]
        fin = v[~np.isnan(v)]
        obs[k]["e5"]["boot_ci_5_95_conditional"] = (
            [float(np.percentile(fin, 5)), float(np.percentile(fin, 95))]
            if len(fin) else None)
        obs[k]["e5"]["boot_undefined_frac"] = float(np.isnan(v).mean())

    # per-grid counts, p_grid (evaluable cells), aggregates — grids never pooled
    part_c_grid = {}
    for grid in ("A", "B"):
        gk = [k for k in keys if k[0] == grid]
        rec = {"per_task": {}}
        pooled_null5 = np.zeros(NPERM_OUTER, dtype=np.int64)
        pooled_null6 = np.zeros(NPERM_OUTER, dtype=np.int64)
        for task in TASKS:
            tk = [k for k in gk if k[1] == task]
            cnt5 = sum((null_d[k] >= E5_DIP).astype(np.int64) for k in tk)
            cnt6 = np.zeros(NPERM_OUTER, dtype=np.int64)
            for k in tk:
                nj, nr = null_J[k], np.nan_to_num(null_R[k], nan=-1.0)
                q95 = obs[k]["e6"]["_q95_raw"]
                cnt6 += ((nj >= E6_J) & (nj > q95) & (nr > 0)
                         & (nj >= E6_RISE_FRAC * nr)).astype(np.int64)
            o5 = sum(obs[k]["e5"]["success"] for k in tk)
            o6 = sum(obs[k]["e6"]["success"] for k in tk)
            pooled_null5 += cnt5
            pooled_null6 += cnt6
            stack = np.stack([boot_d[k] for k in tk])
            alldef = ~np.isnan(stack).any(axis=0)
            agg = stack[:, alldef].mean(axis=0)
            rec["per_task"][task] = {
                "e5_success": int(o5), "evaluable": len(tk),
                "p_grid_e5": mc_p(cnt5, o5),
                "e6_success": int(o6), "p_grid_e6": mc_p(cnt6, o6),
                "aggregate_delta_ci_5_95_alldef": (
                    [float(np.percentile(agg, 5)), float(np.percentile(agg, 95))]
                    if alldef.any() else None),
                "aggregate_joint_undefined_frac": float((~alldef).mean())}
        o5 = sum(obs[k]["e5"]["success"] for k in gk)
        o6 = sum(obs[k]["e6"]["success"] for k in gk)
        rec["e5_success_evaluable"] = f"{o5}/{len(gk)}"
        rec["e6_success_evaluable"] = f"{o6}/{len(gk)}"
        rec["p_grid_e5_pooled"] = mc_p(pooled_null5, o5)
        rec["p_grid_e6_pooled"] = mc_p(pooled_null6, o6)
        stack = np.stack([boot_d[k] for k in gk])
        alldef = ~np.isnan(stack).any(axis=0)
        agg = stack[:, alldef].mean(axis=0)
        rec["aggregate_delta_ci_5_95_alldef"] = (
            [float(np.percentile(agg, 5)), float(np.percentile(agg, 95))]
            if alldef.any() else None)
        rec["aggregate_joint_undefined_frac"] = float((~alldef).mean())
        if grid == "B":
            fam = {f: sum(obs[("B", t, s)]["e5"]["success"] for t in TASKS for s in m
                          if ("B", t, s) in obs) for f, m in FAMILIES.items()}
            guards = {"count_ge_10": o5 >= 10,
                      "families_ge_3of4": all(v >= 3 for v in fam.values()),
                      "p_grid_lt_0.05": rec["p_grid_e5_pooled"] < 0.05}
            rec["counterfactual_s5_rule_NON_VERDICT"] = {
                "count_of_12": int(o5), "per_family": fam, "guards": guards,
                "decision": ("CONFIRM" if all(guards.values())
                             else "FALSIFY" if o5 <= 7 else "WEAKEN")}
        part_c_grid[grid] = rec

    for k in keys:
        q = obs[k]["e6"].pop("_q95_raw")
        obs[k]["e6"]["null_q95_is_neginf"] = bool(np.isneginf(q))
        obs[k]["e6"]["null_q95"] = float(q) if np.isfinite(q) else None
        obs[k]["e6"]["null_undefined_frac"] = float(np.isneginf(null_J[k]).mean())
        obs[k]["e5"]["registered_delta"] = reg_delta[k]
        obs[k]["e5"]["registered_perm_p"] = banked[k].get("perm_p")
        obs[k]["e5"]["registered_boot_ci"] = banked[k].get("boot_ci_5_95_conditional")
        obs[k]["e5"]["registered_success"] = banked[k].get("success")

    out = {
        "banner": "UNREGISTERED stem-grouped sensitivity rescore. Cannot move any "
                  "registered verdict. Grids A and B are never pooled.",
        "config": {"m_draws": M_DRAWS, "k_folds": K_FOLDS, "nperm_inner": NPERM_INNER,
                   "nperm_outer": NPERM_OUTER, "nboot": NBOOT, "e5_dip": E5_DIP,
                   "primary": PRIMARY, "part_a_tol": REPRO_TOL},
        "provenance": {
            "script_sha256": _sha256_file(os.path.abspath(__file__)),
            "rescore_machinery_sha256": live,
            "numpy_version": np.__version__,
            "designated_group_fold_map_sha256": {
                t: hashlib.sha256(des_fold[t].tobytes()).hexdigest() for t in TASKS},
            "runtime_s": round(time.time() - t0, 1)},
        "groups": {t: {"pairs": len(units[t]["pair"]),
                       "pos_singletons": len(units[t]["pos1"]),
                       "neg_singletons": len(units[t]["neg1"]),
                       "pairs_split_by_registered_fold_map": split_reg[t],
                       "pairs_split_row_draws_median": float(np.median(split_row_draws[t]))}
                   for t in TASKS},
        "part_b_fold_draws": {"cells": part_b, "by_grid": part_b_grid},
        "part_c_grouped_inference": {"cells": {"/".join(k): obs[k] for k in keys},
                                     "by_grid": part_c_grid},
    }
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, jpath)

    L_ = ["# Stem-grouped sensitivity rescore (UNREGISTERED)", "", out["banner"], "",
          "## Groups", ""]
    for t in TASKS:
        gg = out["groups"][t]
        L_.append(f"- {t}: {gg['pairs']} pairs, {gg['pos_singletons']}+"
                  f"{gg['neg_singletons']} singletons; registered fold map splits "
                  f"{gg['pairs_split_by_registered_fold_map']} pairs")
    L_ += ["", "## Part B — fold-draw distributions (row-stratified vs group-stratified)",
           "", "| grid | task | model | registered Δ | row mean (sd) | group mean (sd) | "
           "group−row (MC se) | success row / group | reg. pctile in row arm |",
           "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"
    for k in keys:
        r = part_b["/".join(k)]
        L_.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(r['registered_delta'])} | "
                  f"{fmt(r['row_mean'])} ({fmt(r['row_sd'])}) | "
                  f"{fmt(r['group_mean'])} ({fmt(r['group_sd'])}) | "
                  f"{fmt(r['group_minus_row'])} ({fmt(r['diff_mc_se'])}) | "
                  f"{r['row_success_frac']:.2f} / {r['group_success_frac']:.2f} | "
                  f"{fmt(r.get('registered_pctile_in_row_arm'))} |")
    for grid in ("A", "B"):
        L_.append(f"\nGrid {grid}: {json.dumps(part_b_grid[grid])}")
    L_ += ["", "## Part C — fully grouped inference (designated grouped fold map)", "",
           "| grid | task | model | Δ reg → grouped | ok reg → grouped | perm p reg → "
           "grouped | boot CI reg → grouped | E6 J / R / ok |",
           "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        e5, e6 = obs[k]["e5"], obs[k]["e6"]
        L_.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(e5['registered_delta'])} → "
                  f"{fmt(e5['delta'])} | {e5['registered_success']} → {e5['success']} | "
                  f"{e5['registered_perm_p']} → {e5.get('perm_p')} | "
                  f"{e5['registered_boot_ci']} → {e5.get('boot_ci_5_95_conditional')} | "
                  f"{fmt(e6.get('J'))} / {fmt(e6.get('R'))} / {e6['success']} |")
    for grid in ("A", "B"):
        L_.append(f"\nGrid {grid}: {json.dumps(part_c_grid[grid])}")
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(L_) + "\n")
    os.replace(tmp, mpath)
    print(f"wrote {jpath}\nwrote {mpath}\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
