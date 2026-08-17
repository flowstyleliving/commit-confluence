"""Grid-B confirmatory scorer — PRE_REGISTRATION_EXPANSION.md §4–§6, run ONCE.

Statistical machinery imported UNMODIFIED from the calibration implementation
(rescore_grid_a.py — sha256 verified against the freeze pin): stratified fixed
5-fold cross-fit, training-only directions/qualification/peak, held-out contrasts,
within-fold synchronized permutations, (1+k)/(B+1) p-values, method="higher"
quantiles, D1–D9. This file adds the grid-B specifics; round-5 audit fixes:

  * labels come from the LOCAL FROZEN DATA FILES (sha256-verified, `label` field),
    not from artifacts — so fold designs and verdicts exist even when a task has
    zero evaluable cells (MAJOR-7); npz labels are cross-validated against them;
  * frozen per-slug HF revisions validated against artifact provenance (MAJOR-6);
  * EVERY loader/validation exception becomes a registered cell failure (MAJOR-6);
  * terminal-status enforcement (12 statuses or refuse; aborted/invalid = failure);
  * §5 decisions with exact precedence; evaluable-only sensitivity recounts and
    leave-Medium-out recounts reported explicitly (MINOR-4);
  * registered aggregates: per-task and pooled all-defined bootstrap CIs over the
    FIXED designated set = evaluable cells (fixed before the bootstrap, excluded
    cells listed; no moving denominator) (MAJOR-8);
  * E7 cross-task peak distances WITH shared-resample bootstrap intervals
    (fold-peak medians captured per replicate) (MAJOR-8);
  * E8 with the FULL qualification rule (training AUROC >= 0.65 AND > the fold's
    training-rows-only envelope), recomputed descriptively from the pinned
    primitives with the same frozen fold/inner designs; reports qualifying-block
    sets in [0.4N, 0.9N] per fold, counts, majority stats, and the median training
    curve as context (MAJOR-9);
  * E1'' panel with model/family-clustered summary (MINOR-5);
  * provenance: npz + status-file + extractor + prereg + machinery sha256s
    (MINOR-1); no bare asserts on the failure paths (MINOR-6);
  * single-look guard, atomic writes, strict JSON (allow_nan=False).

Permutation-null denominator (documented): null success counts run over the
EVALUABLE cells only — a cell with no data contributes 0 to both observed and
null counts — while CONFIRM/WEAKEN/FALSIFY decisions always use the 12-cell
denominator with failures counted as failures.

Run: .venv/bin/python score_grid_b.py --npz-dir npz/depth_grid_b
"""
import argparse
import hashlib
import json
import os

import numpy as np

from depth_score import QUAL_AUROC, ENVELOPE_Q
from rescore_grid_a import (
    RS_SEED, K_FOLDS, NPERM_INNER, NPERM_OUTER, NBOOT,
    E5_DIP, E6_J, E6_RISE_FRAC, E6_NULL_Q,
    _rng, _sha256_file, mc_p, emp_q, rank_columns,
    raw_auroc_from_ranks, raw_auroc_perm_batch,
    make_folds, within_fold_permutation, crossfit_cell,
    e5_from_crossfit, e6_from_crossfit, e6_success_fixed,
)

RESCORE_PIN_SHA256 = "522cd5301f4f783faaec13f2c90a58f68c6247f2d0dd5b9f5b86b522ca81a958"

PRIMARY = "final_js_no_bos"
EXPECT_METRICS = ["final_js", "final_js_no_bos", "final_js_kv_groups", "final_bos_mass"]
EXPECT_SCHEMA = "furnace-depth-curve/1.1-gridB"
EXPECT_N_ROWS = 200
TASKS = ["anli_r1", "halueval_qa"]
FROZEN_DATA_SHA256 = {
    "anli_r1": "57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b",
    "halueval_qa": "a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0",
}

# slug -> (expected layers, expected precision label, family, FROZEN HF revision)
GRID_B = {
    "Llama-3.1-8B-Instruct": (32, "nf4", "llama31",
                              "0e9e39f249a16976918f6564b8830bc894c89659"),
    "Llama-3.1-70B-Instruct": (80, "nf4", "llama31",
                               "1605565b47bb9346c5515c34102e054115b4f98b"),
    "Mistral-Small-3.2-24B-Instruct-2506": (40, "nf4", "mistral",
                                            "95a6d26c4bfb886c58daf9d3f7332c857cb27b43"),
    "Mistral-Medium-3.5-128B": (88, "fp8origin-dequant-bf16", "mistral",
                                "22b2b868a15677cfa6061277ed2f653d1349a9ab"),
    "gemma-3-12b-it": (48, "nf4", "gemma",
                       "96b6f1eccf38110c56df3a15bffe176da04bfd80"),
    "gemma-3-27b-it": (62, "nf4", "gemma",
                       "005ad3404e59d6023443cb575daa05336842228a"),
}
SLUGS = list(GRID_B.keys())
FAMILIES = {"llama31": [s for s, v in GRID_B.items() if v[2] == "llama31"],
            "mistral": [s for s, v in GRID_B.items() if v[2] == "mistral"],
            "gemma": [s for s, v in GRID_B.items() if v[2] == "gemma"]}
MEDIUM_SLUG = "Mistral-Medium-3.5-128B"
GRID_A_SLUGS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct",
                "Qwen2.5-72B-Instruct", "Llama-3.3-70B-Instruct"]

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = {t: os.path.join(_HERE, "data", f"{t}_n200.jsonl") for t in TASKS}


def load_task_labels(task):
    """Labels from the LOCAL frozen data file (sha256-verified) — independent of
    any artifact, so a fully-aborted task still has a fold design + verdict."""
    path = LOCAL_DATA[task]
    if not os.path.exists(path):
        raise SystemExit(f"missing local frozen data file: {path}")
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FROZEN_DATA_SHA256[task]:
        raise SystemExit(f"LOCAL DATA HASH MISMATCH {task}: {sha}")
    y = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            y.append(int(json.loads(line)["label"]))
    y = np.asarray(y, dtype=np.int64)
    if len(y) != EXPECT_N_ROWS or int(y.sum()) != 100:
        raise SystemExit(f"{task}: labels not 100/100 x 200 (n={len(y)}, n1={int(y.sum())})")
    return y


def load_cell_b(npz_dir, task, slug, y_task):
    path = os.path.join(npz_dir, task, f"{slug}.depth.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"npz missing: {path}")
    z = np.load(path, allow_pickle=False)
    metrics = json.loads(str(z["metrics"]))
    meta = json.loads(str(z["meta"]))
    scores = z["scores"]
    y = z["labels"].astype(np.int64)
    exp_layers, exp_prec, _fam, exp_rev = GRID_B[slug]

    def req(cond, msg):
        if not cond:
            raise ValueError(f"{path}: {msg}")

    req(meta.get("schema") == EXPECT_SCHEMA, f"schema {meta.get('schema')!r}")
    req(str(meta.get("model", "")).endswith(slug), f"model {meta.get('model')!r}")
    req(meta.get("task") == task, f"task {meta.get('task')!r}")
    req(meta.get("precision") == exp_prec,
        f"precision {meta.get('precision')!r} != expected {exp_prec!r}")
    req(meta.get("data_sha256") == FROZEN_DATA_SHA256[task], "data sha mismatch")
    req(int(meta.get("n_layers", -1)) == exp_layers,
        f"n_layers {meta.get('n_layers')} != {exp_layers}")
    req(meta.get("revision_pinned") == exp_rev,
        f"revision_pinned {meta.get('revision_pinned')!r} != frozen {exp_rev!r}")
    req(meta.get("provenance", {}).get("hf_model_revision") == exp_rev,
        "loaded HF revision != frozen revision")
    req(metrics == EXPECT_METRICS, f"metrics {metrics}")
    req(scores.shape == (EXPECT_N_ROWS, exp_layers, len(EXPECT_METRICS)),
        f"scores shape {scores.shape}")
    req(y.shape == (EXPECT_N_ROWS,), "labels shape")
    req(np.array_equal(y, y_task), "npz labels != frozen data labels")
    req(np.array_equal(z["sample_idx"], np.arange(EXPECT_N_ROWS)), "sample_idx")
    gate = meta.get("gate") or {}
    req(bool(gate.get("GATE_cos_ok")) and bool(gate.get("GATE_yes_no_ok")), "gates")
    req(float(meta.get("yes_no_commit_rate", 0.0)) >= 0.5, "yes/no rate")
    req(np.isfinite(scores).all(), "non-finite scores")
    req(meta.get("capture_mode") == "full_eager_retention", "capture mode")
    return scores[:, :, metrics.index(PRIMARY)], exp_layers, meta, path


def fold_qual_masks(prim, y, fold_of, inner_rows):
    """Descriptive helper (E8, MAJOR-9): per fold, the FULL qualification rule —
    training sign-free AUROC >= QUAL_AUROC AND > the training-rows-only envelope —
    rebuilt from the pinned primitives with the same frozen designs. Returns
    (list of qual masks [L] per fold, list of A_tr per fold)."""
    quals, curves = [], []
    for f in range(K_FOLDS):
        tr = fold_of != f
        y_tr = y[tr]
        R_tr = rank_columns(prim[tr])
        a_tr = raw_auroc_from_ranks(R_tr, y_tr)
        A_tr = np.maximum(a_tr, 1.0 - a_tr)
        env_ind = (y_tr[inner_rows[f]] == 1)
        env = raw_auroc_perm_batch(R_tr, env_ind)
        env_sf = np.maximum(env, 1.0 - env)
        env_q = np.percentile(env_sf, ENVELOPE_Q, axis=0)
        quals.append((A_tr >= QUAL_AUROC) & (A_tr > env_q))
        curves.append(A_tr)
    return quals, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--grid-a-dir", default=os.path.join(_HERE, "npz", "depth_curve"))
    ap.add_argument("--out-dir", default=_HERE)
    args = ap.parse_args()

    live_pin = _sha256_file(os.path.join(_HERE, "rescore_grid_a.py"))
    if live_pin != RESCORE_PIN_SHA256:
        raise SystemExit(f"STAT MACHINERY DRIFT: {live_pin} != {RESCORE_PIN_SHA256}")

    jpath = os.path.join(args.out_dir, "GRID_B_RESULTS.json")
    mpath = os.path.join(args.out_dir, "GRID_B_RESULTS.md")
    for p in (jpath, mpath):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING TO RUN: {p} exists — single look.")

    # ---- labels from frozen data (task designs exist unconditionally)
    y_by_task = {t: load_task_labels(t) for t in TASKS}

    # Fail-closed grid-A label identity (round-6 M7 residual): the E1'' panel and
    # E8 context claim the SAME fold designs as the calibration; any present
    # grid-A npz whose labels differ from the frozen data is a hard stop, checked
    # BEFORE any confirmatory computation (absence of grid-A npzs is fine — the
    # descriptive panel is then omitted, which is not a mismatch).
    for task in TASKS:
        for slug in GRID_A_SLUGS:
            apath = os.path.join(args.grid_a_dir, task, f"{slug}.depth.npz")
            if os.path.exists(apath):
                za = np.load(apath, allow_pickle=False)
                if not np.array_equal(za["labels"].astype(np.int64), y_by_task[task]):
                    raise SystemExit(f"GRID-A LABEL MISMATCH vs frozen data: {apath}")

    # ---- terminal-status enforcement + load (EVERY exception -> registered failure)
    failures, prim_by, L_by = {}, {}, {}
    status_sha, npz_sha = {}, {}
    for task in TASKS:
        for slug in SLUGS:
            spath = os.path.join(args.npz_dir, task, f"{slug}.status.json")
            if not os.path.exists(spath):
                raise SystemExit(f"NOT ALL CELLS TERMINAL: missing {spath} — the "
                                 f"scorer runs once after all 12 cells terminate.")
            status_sha[f"{task}/{slug}"] = _sha256_file(spath)
            # hash ANY existing npz — including ones that later fail validation —
            # so failure-determining artifacts are pinned too (round-6 MINOR-1)
            npath0 = os.path.join(args.npz_dir, task, f"{slug}.depth.npz")
            if os.path.exists(npath0):
                npz_sha[f"{task}/{slug}"] = _sha256_file(npath0)
            try:
                status = json.load(open(spath))
            except Exception as e:  # noqa: BLE001
                failures[(task, slug)] = f"unreadable status: {type(e).__name__}: {e}"
                continue
            if status.get("status") != "ok":
                failures[(task, slug)] = f"aborted: {str(status.get('reason', '?'))[:300]}"
                continue
            try:
                prim, L, _meta, _npath = load_cell_b(args.npz_dir, task, slug,
                                                     y_by_task[task])
            except Exception as e:  # noqa: BLE001  (round-5 MAJOR-6: broad)
                failures[(task, slug)] = f"invalid artifact: {type(e).__name__}: {e}"
                continue
            prim_by[(task, slug)] = prim
            L_by[(task, slug)] = L

    evaluable = {t: [s for s in SLUGS if (t, s) in prim_by] for t in TASKS}
    n_evaluable = sum(len(v) for v in evaluable.values())
    print(f"cells: evaluable {n_evaluable}/12, failures {len(failures)}", flush=True)

    # ---- frozen designs (unconditional per task)
    fold_of, inner_rows, outer_rngs, boot_idx = {}, {}, {}, {}
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
                    take[pos:pos + len(stratum)] = stratum[
                        g.integers(0, len(stratum), size=len(stratum))]
                    pos += len(stratum)
            rows.append(take)
        boot_idx[task] = np.stack(rows)

    rank_cache = {}
    for (task, slug), prim in prim_by.items():
        cache = []
        for f in range(K_FOLDS):
            tr = fold_of[task] != f
            cache.append((rank_columns(prim[tr]), rank_columns(prim[~tr])))
        rank_cache[(task, slug)] = cache

    # ---- observed (silent: identifiers only)
    cells = {}
    for task in TASKS:
        for slug in SLUGS:
            if (task, slug) in failures:
                cells[(task, slug)] = {
                    "L": GRID_B[slug][0], "failed": True,
                    "reason": failures[(task, slug)],
                    "e5": {"defined": False, "why": failures[(task, slug)],
                           "delta": None, "success": False},
                    "e6": {"defined": False, "why": failures[(task, slug)],
                           "J": None, "R": None, "success": False}}
                continue
            L = L_by[(task, slug)]
            cf = crossfit_cell(prim_by[(task, slug)], y_by_task[task], fold_of[task],
                               inner_rows[task], rank_cache=rank_cache[(task, slug)])
            cells[(task, slug)] = {"L": L, "failed": False,
                                   "e5": e5_from_crossfit(cf, L),
                                   "e6": e6_from_crossfit(cf, L)}
            print(f"observed computed: {task}/{slug}", flush=True)

    # ---- synchronized within-fold permutation null (evaluable cells)
    null_delta = {(t, s): np.full(NPERM_OUTER, -np.inf)
                  for t in TASKS for s in evaluable[t]}
    null_J = {(t, s): np.full(NPERM_OUTER, -np.inf)
              for t in TASKS for s in evaluable[t]}
    null_R = {(t, s): np.full(NPERM_OUTER, np.nan)
              for t in TASKS for s in evaluable[t]}
    for task in TASKS:
        y = y_by_task[task]
        for p in range(NPERM_OUTER):
            yp = within_fold_permutation(y, fold_of[task], outer_rngs[task])
            for slug in evaluable[task]:
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
                print(f"perm {task}: {p + 1}/{NPERM_OUTER}", flush=True)

    for task in TASKS:
        for slug in evaluable[task]:
            c = cells[(task, slug)]
            nd, nj = null_delta[(task, slug)], null_J[(task, slug)]
            c["e5"]["perm_p"] = (mc_p(nd, c["e5"]["delta"])
                                 if c["e5"]["defined"] else None)
            c["e5"]["null_undefined_frac"] = float(np.isneginf(nd).mean())
            q95 = emp_q(nj, E6_NULL_Q)
            c["e6"]["_q95_raw"] = q95
            c["e6"]["null_undefined_frac"] = float(np.isneginf(nj).mean())
            c["e6"]["perm_p_J"] = (mc_p(nj, c["e6"]["J"])
                                   if c["e6"]["defined"] else None)
            c["e6"]["success"] = (e6_success_fixed(c["e6"]["J"], c["e6"]["R"], q95)
                                  if c["e6"]["defined"] else False)

    # ---- counts, guards, decisions (§5)
    def ok(endpoint, t, s):
        return bool(cells[(t, s)][endpoint].get("success"))

    obs5 = sum(ok("e5", t, s) for t in TASKS for s in SLUGS)
    obs6 = sum(ok("e6", t, s) for t in TASKS for s in SLUGS)
    fam5 = {f: sum(ok("e5", t, s) for t in TASKS for s in m)
            for f, m in FAMILIES.items()}
    fam6 = {f: sum(ok("e6", t, s) for t in TASKS for s in m)
            for f, m in FAMILIES.items()}
    task5 = {t: sum(ok("e5", t, s) for s in SLUGS) for t in TASKS}
    task6 = {t: sum(ok("e6", t, s) for s in SLUGS) for t in TASKS}

    task_counts5, task_counts6 = {}, {}
    for task in TASKS:
        cnt5 = np.zeros(NPERM_OUTER, dtype=np.int64)
        cnt6 = np.zeros(NPERM_OUTER, dtype=np.int64)
        for slug in evaluable[task]:
            nd = null_delta[(task, slug)]
            cnt5 += (nd >= E5_DIP).astype(np.int64)
            nj, nr = null_J[(task, slug)], null_R[(task, slug)]
            q95 = cells[(task, slug)]["e6"]["_q95_raw"]
            nr_safe = np.nan_to_num(nr, nan=-1.0)
            ok6 = ((nj >= E6_J) & (nj > q95) & (nr_safe > 0)
                   & (nj >= E6_RISE_FRAC * nr_safe))
            cnt6 += ok6.astype(np.int64)
        task_counts5[task], task_counts6[task] = cnt5, cnt6
    p_grid5 = mc_p(task_counts5[TASKS[0]] + task_counts5[TASKS[1]], obs5)
    p_grid6 = mc_p(task_counts6[TASKS[0]] + task_counts6[TASKS[1]], obs6)

    e5_guards = {"count_ge_10": obs5 >= 10,
                 "families_ge_3of4": all(v >= 3 for v in fam5.values()),
                 "p_grid_lt_0.05": p_grid5 < 0.05}
    if all(e5_guards.values()):
        e5_decision = "CONFIRM"
    elif obs5 <= 7:
        e5_decision = "FALSIFY"
    else:
        e5_decision = "WEAKEN"

    if e5_decision != "CONFIRM":
        e6_decision, e6_guards = "NOT TESTED — gate closed", None
    else:
        e6_guards = {"count_ge_9": obs6 >= 9,
                     "tasks_ge_4of6": all(v >= 4 for v in task6.values()),
                     "families_ge_2of4": all(v >= 2 for v in fam6.values()),
                     "p_grid_lt_0.05": p_grid6 < 0.05}
        if all(e6_guards.values()):
            e6_decision = "CONFIRM"
        elif obs6 <= 7:
            e6_decision = "FALSIFY"
        else:
            e6_decision = "WEAKEN"

    sensitivity = {
        "leave_medium_out": {
            "e5_of_10": sum(ok("e5", t, s) for t in TASKS for s in SLUGS
                            if s != MEDIUM_SLUG),
            "e6_of_10": sum(ok("e6", t, s) for t in TASKS for s in SLUGS
                            if s != MEDIUM_SLUG)},
        "evaluable_only": {
            "e5": f"{sum(ok('e5', t, s) for t in TASKS for s in evaluable[t])}/{n_evaluable}",
            "e6": f"{sum(ok('e6', t, s) for t in TASKS for s in evaluable[t])}/{n_evaluable}"},
    }

    # ---- bootstrap: per-cell CIs + fixed-designated-set aggregates + E7 peaks
    boot_delta = {(t, s): np.full(NBOOT, np.nan) for t in TASKS for s in evaluable[t]}
    boot_peakmed = {(t, s): np.full(NBOOT, np.nan) for t in TASKS for s in evaluable[t]}
    for task in TASKS:
        y = y_by_task[task]
        for b in range(NBOOT):
            take = boot_idx[task][b]
            yb = y[take]
            fb = fold_of[task][take]
            for slug in evaluable[task]:
                L = L_by[(task, slug)]
                cf = crossfit_cell(prim_by[(task, slug)][take], yb, fb,
                                   inner_rows[task])
                e5b = e5_from_crossfit(cf, L)
                if e5b["defined"]:
                    boot_delta[(task, slug)][b] = e5b["delta"]
                    boot_peakmed[(task, slug)][b] = float(np.median(e5b["fold_peaks"]))
            if (b + 1) % 200 == 0:
                print(f"boot {task}: {b + 1}/{NBOOT}", flush=True)
    for task in TASKS:
        for slug in evaluable[task]:
            v = boot_delta[(task, slug)]
            fin = v[~np.isnan(v)]
            cells[(task, slug)]["e5"]["boot_ci_5_95_conditional"] = (
                [float(np.percentile(fin, 5)), float(np.percentile(fin, 95))]
                if len(fin) else None)
            cells[(task, slug)]["e5"]["boot_undefined_frac"] = float(np.isnan(v).mean())

    aggregates = {}
    for task in TASKS:
        des = evaluable[task]  # fixed designated set, decided before the bootstrap
        if des:
            stack = np.stack([boot_delta[(task, s)] for s in des])
            all_def = ~np.isnan(stack).any(axis=0)
            aggregates[task] = {
                "designated_cells": des,
                "excluded_failed_cells": [s for s in SLUGS if s not in des],
                "ci_5_95_alldef": ([float(np.percentile(stack[:, all_def].mean(axis=0), 5)),
                                    float(np.percentile(stack[:, all_def].mean(axis=0), 95))]
                                   if all_def.any() else None),
                "joint_undefined_frac": float((~all_def).mean())}
        else:
            aggregates[task] = {"designated_cells": [], "ci_5_95_alldef": None,
                                "excluded_failed_cells": SLUGS,
                                "joint_undefined_frac": 1.0}
    des_all = [(t, s) for t in TASKS for s in evaluable[t]]
    if des_all:
        stack = np.stack([boot_delta[k] for k in des_all])
        all_def = ~np.isnan(stack).any(axis=0)
        aggregates["pooled"] = {
            "designated_cells": [f"{t}/{s}" for t, s in des_all],
            "ci_5_95_alldef": ([float(np.percentile(stack[:, all_def].mean(axis=0), 5)),
                                float(np.percentile(stack[:, all_def].mean(axis=0), 95))]
                               if all_def.any() else None),
            "joint_undefined_frac": float((~all_def).mean())}
    else:
        aggregates["pooled"] = {"designated_cells": [], "ci_5_95_alldef": None,
                                "joint_undefined_frac": 1.0}

    # ---- E7: cross-task peak distance with shared-resample bootstrap intervals
    def peak_cf_obs(t, s):
        c = cells[(t, s)]
        if c.get("failed") or not c["e5"]["defined"]:
            return None
        return float(np.median(c["e5"]["fold_peaks"]))

    e7 = {}
    for slug in SLUGS:
        pa, ph = peak_cf_obs("anli_r1", slug), peak_cf_obs("halueval_qa", slug)
        if pa is None or ph is None:
            e7[slug] = {"defined": False,
                        "why": "one or both cells lack a defined cross-fit",
                        "distance_blocks": None, "distance_frac": None,
                        "distance_boot_ci_5_95_blocks": None,
                        "distance_boot_ci_5_95_frac": None,
                        "boot_pair_undefined_frac": 1.0}
            continue
        N = GRID_B[slug][0]
        entry = {"defined": True, "peak_anli": pa, "peak_halueval": ph,
                 "distance_blocks": abs(pa - ph), "distance_frac": abs(pa - ph) / N,
                 # always-emit fields (round-6 M8 residual: silence is not a value)
                 "distance_boot_ci_5_95_blocks": None,
                 "distance_boot_ci_5_95_frac": None,
                 "boot_pair_undefined_frac": 1.0}
        ka, kh = ("anli_r1", slug), ("halueval_qa", slug)
        if ka in boot_peakmed and kh in boot_peakmed:
            da = boot_peakmed[ka]; dh = boot_peakmed[kh]
            both = ~(np.isnan(da) | np.isnan(dh))  # replicate-index pairing
            entry["boot_pair_undefined_frac"] = float((~both).mean())
            if both.any():
                dist = np.abs(da[both] - dh[both])
                lo5, hi95 = float(np.percentile(dist, 5)), float(np.percentile(dist, 95))
                entry["distance_boot_ci_5_95_blocks"] = [lo5, hi95]
                entry["distance_boot_ci_5_95_frac"] = [lo5 / N, hi95 / N]
        e7[slug] = entry

    # ---- E8: full qualification rule, per-fold sets, context curves
    e8 = {}
    for slug in ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]:
        for task in TASKS:
            key = f"{task}/{slug}"
            if (task, slug) not in prim_by:
                e8[key] = {"defined": False, "why": "cell not evaluable"}
                continue
            prim = prim_by[(task, slug)]
            y = y_by_task[task]
            N = L_by[(task, slug)]
            quals, curves = fold_qual_masks(prim, y, fold_of[task], inner_rows[task])
            lo, hi = int(np.ceil(0.4 * N)), int(np.floor(0.9 * N))
            sets, counts = [], []
            for q in quals:
                blocks = [int(b) for b in range(lo, hi + 1) if q[b]]
                sets.append(blocks)
                counts.append(len(blocks))
            e8[key] = {"defined": True, "window": [lo, hi],
                       "qualifying_blocks_per_fold": sets,
                       "counts_per_fold": counts,
                       "majority_ge_10": int(np.median(counts)) >= 10,
                       "majority_lt_5": int(np.median(counts)) < 5,
                       "median_train_curve": np.median(np.stack(curves), axis=0)
                       .round(4).tolist()}
    # Registered context: the banked Llama-3.3-70B curves beside the 3.1 cells
    # (round-6 M9 residual). Recomputed from grid-A npzs with the same machinery;
    # labeled grid-A context, never pooled. Absence is reported, not silent.
    for task in TASKS:
        key = f"{task}/Llama-3.3-70B-Instruct(grid-A context)"
        apath = os.path.join(args.grid_a_dir, task, "Llama-3.3-70B-Instruct.depth.npz")
        if not os.path.exists(apath):
            e8[key] = {"defined": False, "why": "grid-A npz not present"}
            continue
        try:
            from depth_score import load_cell as _lca
            scores_a, ya, metrics_a, _m = _lca(args.grid_a_dir, task,
                                               "Llama-3.3-70B-Instruct")
            prim_a = scores_a[:, :, metrics_a.index(PRIMARY)]
            N = prim_a.shape[1]
            quals, curves = fold_qual_masks(prim_a, ya, fold_of[task],
                                            inner_rows[task])
            lo, hi = int(np.ceil(0.4 * N)), int(np.floor(0.9 * N))
            sets = [[int(b) for b in range(lo, hi + 1) if q[b]] for q in quals]
            e8[key] = {"defined": True, "window": [lo, hi], "grid": "A (context)",
                       "qualifying_blocks_per_fold": sets,
                       "counts_per_fold": [len(s) for s in sets],
                       "median_train_curve": np.median(np.stack(curves), axis=0)
                       .round(4).tolist()}
        except Exception as e:  # noqa: BLE001
            e8[key] = {"defined": False, "why": f"{type(e).__name__}: {e}"}

    # ---- E1'': panel + family-clustered summary
    e1pp = {"grid_b": {}, "grid_a_separate": {}, "family_summary": {}}
    fam_fracs = {}
    for slug in SLUGS:
        for task in TASKS:
            pk = peak_cf_obs(task, slug)
            if pk is not None:
                N = GRID_B[slug][0]
                e1pp["grid_b"][f"{task}/{slug}"] = {"N": N, "peak_cf": pk,
                                                    "frac": round(pk / N, 4)}
                fam_fracs.setdefault(GRID_B[slug][2], []).append(pk / N)
    # Grid-A panel points: built TRANSACTIONALLY (round-7) — family fractions
    # merge only if the whole grid-A pass completes; on any exception nothing
    # merges and the error is reported, so no unlabeled partial family summary.
    ga_points, ga_fracs = {}, {}
    try:
        from depth_score import load_cell as load_cell_a
        for task in TASKS:
            for slug in GRID_A_SLUGS:
                scores, ya, metrics_a, _m = load_cell_a(args.grid_a_dir, task, slug)
                prim = scores[:, :, metrics_a.index(PRIMARY)]
                cf = crossfit_cell(prim, ya, fold_of[task], inner_rows[task])
                if cf["ok"]:
                    pk = float(np.median([fd["peak"] for fd in cf["folds"]]))
                    ga_points[f"{task}/{slug}"] = {
                        "N": prim.shape[1], "peak_cf": pk,
                        "frac": round(pk / prim.shape[1], 4)}
                    fam_a = "qwen(A)" if slug.startswith("Qwen") else "llama33(A)"
                    ga_fracs.setdefault(fam_a, []).append(pk / prim.shape[1])
        e1pp["grid_a_separate"] = ga_points
        for fam_a, fr in ga_fracs.items():
            fam_fracs[fam_a] = fr
    except Exception as e:  # noqa: BLE001
        e1pp["grid_a_separate"] = {"error": f"{type(e).__name__}: {e}"}
    # family-clustered spread over ALL families, grid-labeled — grid-A families
    # are separate clusters, never numerically pooled with grid B (round-6 MINOR-5)
    for fam, fr in fam_fracs.items():
        e1pp["family_summary"][fam] = {"n": len(fr), "mean_frac": float(np.mean(fr)),
                                       "sd_frac": float(np.std(fr, ddof=1))
                                       if len(fr) > 1 else None}

    predictions = {
        "P6_e5_confirms": e5_decision == "CONFIRM",
        "P7_e6_ge_9_if_tested": (None if e6_guards is None else obs6 >= 9),
        "P8_llama70b_anli_band_majority_ge_10": e8.get(
            "anli_r1/Llama-3.1-70B-Instruct", {}).get("majority_ge_10"),
        "P9_llama8b_anli_band_majority_lt_5": e8.get(
            "anli_r1/Llama-3.1-8B-Instruct", {}).get("majority_lt_5"),
        "P10_no_obvious_placement_rule": "see E1'' panel (descriptive)",
    }

    # ---- sanitize + outputs (atomic; first exposure of observed values)
    for (t, s) in list(cells.keys()):
        c = cells[(t, s)]
        q = c["e6"].pop("_q95_raw", None)
        if q is not None:
            c["e6"]["null_q95_is_neginf"] = bool(np.isneginf(q))
            c["e6"]["null_q95"] = float(q) if np.isfinite(q) else None

    out = {
        "banner": "GRID-B REGISTERED RESULTS — PRE_REGISTRATION_EXPANSION.md; "
                  "single look; failures counted per §5.",
        "config": {"rs_seed": RS_SEED, "k_folds": K_FOLDS,
                   "nperm_inner": NPERM_INNER, "nperm_outer": NPERM_OUTER,
                   "nboot": NBOOT, "e5_dip": E5_DIP, "e6_j": E6_J,
                   "e6_rise_frac": E6_RISE_FRAC, "e6_null_q": E6_NULL_Q,
                   "primary": PRIMARY},
        "provenance": {
            "rescore_machinery_sha256": live_pin,
            "scorer_sha256": _sha256_file(os.path.abspath(__file__)),
            "extractor_sha256": _sha256_file(os.path.join(_HERE, "modal_depth_b.py")),
            "prereg_sha256": _sha256_file(os.path.join(_HERE, "PRE_REGISTRATION_EXPANSION.md")),
            "numpy_version": np.__version__,
            "fold_map_sha256": {t: hashlib.sha256(fold_of[t].tobytes()).hexdigest()
                                for t in TASKS},
            "npz_sha256": npz_sha, "status_sha256": status_sha,
        },
        "cell_failures": {f"{t}/{s}": r for (t, s), r in failures.items()},
        "n_evaluable_of_12": n_evaluable,
        "E5": {"decision": e5_decision, "count_of_12": obs5, "per_task": task5,
               "per_family": fam5, "p_grid_pooled": p_grid5, "guards": e5_guards},
        "E6": {"decision": e6_decision, "count_of_12": obs6, "per_task": task6,
               "per_family": fam6, "p_grid_pooled": p_grid6, "guards": e6_guards},
        "sensitivity": sensitivity,
        "bootstrap_aggregates": aggregates,
        "E7_peak_distance": e7,
        "E8_llama_context": e8,
        "E1pp_panel": e1pp,
        "predictions_as_registered": predictions,
        "cells": {f"{t}/{s}": c for (t, s), c in cells.items()},
    }
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, jpath)

    lines = ["# Grid-B registered results (single look)", "", out["banner"], "",
             f"**E5: {e5_decision}** — {obs5}/12 (families {fam5}, tasks {task5}, "
             f"p_grid {p_grid5:.4g}; sensitivity {sensitivity})", "",
             f"**E6: {e6_decision}** — {obs6}/12 (families {fam6}, tasks {task6}, "
             f"p_grid {p_grid6:.4g})", "",
             "| task | model | E5 Δ_cf | E5 | E6 J_cf | E6 R_cf | E6 | note |",
             "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for slug in SLUGS:
            c = cells[(task, slug)]
            e5c, e6c = c["e5"], c["e6"]
            d = f"{e5c['delta']:.4f}" if e5c.get("defined") else "UNDEF"
            j = f"{e6c['J']:.4f}" if e6c.get("defined") else "UNDEF"
            r = f"{e6c['R']:.4f}" if e6c.get("defined") else "—"
            note = c.get("reason", "") or (e5c.get("why", "") if not e5c.get("defined") else "")
            lines.append(f"| {task} | {slug} | {d} | {'Y' if e5c.get('success') else 'N'} | "
                         f"{j} | {r} | {'Y' if e6c.get('success') else 'N'} | {note[:80]} |")
    lines += ["", f"Aggregates: {json.dumps(aggregates, default=str)[:900]}", "",
              f"Predictions: {json.dumps(predictions, default=str)}", "",
              f"E7: {json.dumps(e7, default=str)[:1200]}", ""]
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, mpath)
    print(f"wrote {jpath}\nwrote {mpath}")
    print(f"E5: {e5_decision} {obs5}/12 (p={p_grid5:.4g}) | "
          f"E6: {e6_decision} {obs6}/12 (p={p_grid6:.4g})")


if __name__ == "__main__":
    main()
