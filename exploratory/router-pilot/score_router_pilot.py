"""Local single-look scorer. Verification: not run by Codex.

No observed quantity is printed or published until all analysis is complete.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from make_freeze import (HERE, ROOT, TASKS, DATA_SHA, atomic, check_freeze,
                         json_bytes, require, sha, utc)

ATT = ("final_js", "final_js_no_bos", "final_js_kv_groups", "final_bos_mass",
       "final_v_norm_max", "final_v_norm_lastq_weighted")
ATT_SELECT = ("final_js", "final_js_no_bos", "final_bos_mass",
              "final_v_norm_max", "final_v_norm_lastq_weighted")
ROUTER = ("router_entropy_norm", "router_margin", "router_top8_mass_precast")
OUTPUT = ("out_p_max", "out_entropy", "out_yesno_logodds")
COMPARATORS = ("att_best", "out_best", "prompt_len")
NBOOT, NPERM = 1000, 200
LIMITS = ("Prospective descriptive single-model spending pilot; labels are dataset labels, "
          "not model errors. Final prompt state is the estimand; YES/NO compliance is descriptive. "
          "The best comparator is an upward-selected estimate. Intervals are conditional on "
          "the frozen folds and pointwise, without simultaneous depth coverage. n=200 cannot "
          "establish a reliable three-point advantage. ANLI's one pair per fold has no pair "
          "bootstrap variability. The permutation null assumes conditional exchangeability "
          "within pairs. No pooling, cross-model, placement-law, or conditional incremental claim. "
          "sink_draw is a new, unregistered descriptive diagnostic outside the comparator set and "
          "the spending rule.")


def load_machinery():
    # Called only after freeze integrity validation; never edit/copy these primitives.
    global np, _rng, rank_columns, raw_auroc_from_ranks
    global load_groups, units_by_type, make_group_folds, canonical_layout
    global cluster_boot_take, group_outer_perm
    import numpy as np
    require(np.__version__ == "2.2.6", "local numpy must equal image pin 2.2.6")
    sys.path.insert(0, str(ROOT / "exploratory/depth-curve"))
    from rescore_grid_a import _rng, rank_columns, raw_auroc_from_ranks
    from stem_grouped_sensitivity import (load_groups, units_by_type, make_group_folds,
        canonical_layout, cluster_boot_take, group_outer_perm)


def auroc(x, y):
    require(set(np.unique(y).tolist()) == {0, 1}, "degenerate replicate")
    result = raw_auroc_from_ranks(rank_columns(x), y)
    require(np.isfinite(result).all(), "non-finite AUROC")
    return result


def spearman(a, b):
    ra, rb = rank_columns(a[:, None])[:, 0], rank_columns(b[:, None])[:, 0]
    if np.var(ra) == 0 or np.var(rb) == 0:
        return {"value": None, "reason": "undefined: constant input"}
    value = float(np.corrcoef(ra, rb)[0, 1])
    return {"value": value if np.isfinite(value) else None,
            "reason": None if np.isfinite(value) else "undefined: non-finite correlation"}


def descriptive(operation):
    """A diagnostic error/non-finite value must not consume the single look without a decision."""
    def finite_json(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return {"value": None, "reason": "undefined: non-finite result"}
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        return value
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = finite_json(operation())
        json_bytes(value)
        return value
    except Exception as exc:
        return {"value": None, "reason": f"unavailable: {type(exc).__name__}: {exc}"}


def selection_dispersion(picks):
    counts = Counter((p["metric"], p["block"]) for p in picks)
    probs = np.asarray(list(counts.values()), dtype=float) / 5
    blocks = [p["block"] for p in picks if p["block"] is not None]
    return {"unique_columns": len(counts),
            "counts": [{"metric": m, "block": b, "count": n} for (m, b), n in counts.items()],
            "selection_entropy_nats": float(-(probs * np.log(probs)).sum()),
            "block_range": [min(blocks), max(blocks)] if blocks else None,
            "block_sd": float(np.std(blocks)) if blocks else None}


def crossfit_details(folds, columns, family):
    selections = {name: [] for name in family}
    curves = {name: [] for name in ATT_SELECT + ROUTER + ("sink_draw",)}
    for f, train, direction, directed_test in folds:
        for arm, indexes in family.items():
            j = int(indexes[np.argmax(train[indexes])])
            metric, block = columns[j]
            selections[arm].append({"fold": f, "metric": metric, "block": block,
                                    "direction": int(direction[j]),
                                    "train_signfree_auroc": float(train[j]),
                                    "heldout_auroc": float(directed_test[j])})
        for j, metric in enumerate(ATT_SELECT + ROUTER):
            curves[metric].append(train[j * 16:(j + 1) * 16].tolist())
        curves["sink_draw"].append(train[132:148].tolist())
    dispersion = {arm: descriptive(lambda: selection_dispersion(picks))
                  for arm, picks in selections.items()}
    return {"selections": selections, "selection_dispersion": dispersion,
            "training_signfree_curves_by_fold": curves}


def crossfit(x, y, fold, columns, details=False):
    """Metric-major columns, block ascending; argmax implements frozen first tie."""
    heldout = {name: [] for name in ("att_best", "router_best", "out_best", "prompt_len",
                                     "sink_best") + OUTPUT}
    detail_folds = []
    # sink_best (columns 132-147) is descriptive only: never in COMPARATORS or the decision.
    family = {"att_best": np.arange(80), "router_best": np.arange(80, 128),
              "out_best": np.arange(128, 131), "sink_best": np.arange(132, 148)}
    for f in range(5):
        training, test = fold != f, fold == f
        a = auroc(x[training], y[training])
        direction = np.where(a >= .5, 1, -1)
        train = np.where(direction == 1, a, 1 - a)
        raw_test = auroc(x[test], y[test])
        directed_test = np.where(direction == 1, raw_test, 1 - raw_test)
        for arm, indexes in family.items():
            j = int(indexes[np.argmax(train[indexes])])
            heldout[arm].append(float(directed_test[j]))
        heldout["prompt_len"].append(float(directed_test[131]))
        for j, metric in enumerate(OUTPUT, start=128):
            heldout[metric].append(float(directed_test[j]))
        if details:
            detail_folds.append((f, train, direction, directed_test))
    means = {arm: float(np.mean(values)) for arm, values in heldout.items()}
    best = max(COMPARATORS, key=lambda arm: means[arm])
    means["best_comparator"] = means[best]
    means["router_minus_comparator"] = means["router_best"] - means[best]
    if not details:
        return means
    detail = descriptive(lambda: crossfit_details(detail_folds, columns, family))
    if "reason" in detail:
        detail = {"selections": {}, "selection_dispersion": detail,
                  "training_signfree_curves_by_fold": detail}
    return {"arms": means, "best_comparator_arm": best, "fold_aurocs": heldout, **detail}


def design(task):
    path = ROOT / "exploratory/depth-curve/data" / f"{task}_n200.jsonl"
    require(sha(path) == DATA_SHA[task], "frozen local data hash")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    require(len(y) == 200 and np.count_nonzero(y == 1) == np.count_nonzero(y == 0) == 100,
            "label balance")
    gid = load_groups(task, y)
    units = units_by_type(gid, y)
    require(len(units["pair"]) == (5 if task == "anli_r1" else 100), "pair count")
    require(len(units["pos1"]) == len(units["neg1"]) == (95 if task == "anli_r1" else 0),
            "singleton count")
    fold = make_group_folds(units, len(y), _rng(task, "router-pilot-folds-v1"))
    for f in range(5):
        require(np.bincount(y[fold == f], minlength=2).tolist() == [20, 20], "fold balance")
    require(all(fold[a] == fold[b] for a, b in units["pair"]), "split stem")
    canon, spec = canonical_layout(fold, units)
    return y, fold, canon, spec


def load_cell(directory, task, status, freeze):
    path = directory / f"{task}.router.npz"
    require(sha(path) == status["npz_sha256"], "NPZ/status hash mismatch")
    require(status["task"] == task and status["freeze_entry"] == freeze["entries"][-1], "status provenance")
    manifest = json.loads((directory / f"{task}.manifest.json").read_text())
    require(manifest["data_sha256"] == DATA_SHA[task] and manifest["task"] == task, "manifest data")
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    required = {"attention", "sink_draw", "router", "output", "expert_ids", "expert_weights",
                "prompt_len", "argmax_id", "yes_no", "metadata"}
    require(set(data) == required, "NPZ keys")
    meta = json.loads(str(data.pop("metadata")))
    require(meta["schema"] == "router-pilot-npz/1" and meta["task"] == task, "NPZ schema/task")
    require(meta["model"] == "allenai/OLMoE-1B-7B-0924-Instruct"
            and meta["revision"] == "7f1c97f440f06ce36705e4f2b843edb5925f4498", "model revision")
    require(meta["data_sha256"] == DATA_SHA[task] and meta["freeze_entry"] == freeze["entries"][-1],
            "NPZ source freeze")
    require(meta["manifest_sha256"] == freeze["entries"][-1]["manifests"][f"{task}.manifest.json"],
            "NPZ manifest")
    for key, names in (("attention", ATT), ("router", ROUTER), ("output", OUTPUT)):
        shape = (200, len(names)) if key == "output" else (200, 16, len(names))
        require(tuple(meta[key + "_metrics"]) == names and data[key].shape == shape, "metric axes")
        require(np.isfinite(data[key]).all(), "non-finite banked feature")
    require(meta["sink_metric"] == "sink_draw" and data["sink_draw"].shape == (200, 16)
            and np.isfinite(data["sink_draw"]).all(), "sink_draw axis")
    for key in ("argmax_id", "yes_no", "prompt_len"):
        require(data[key].shape == (200,), "row scalar shape")
    require(np.array_equal(data["prompt_len"], [r["length"] for r in manifest["rows"]]), "row order/length")
    ids, weights = data["expert_ids"], data["expert_weights"]
    require(ids.shape == weights.shape == (200, 16, 8), "routing provenance shape")
    require(np.issubdtype(ids.dtype, np.integer) and ((ids >= 0) & (ids < 64)).all(), "expert IDs")
    require((np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all(), "duplicate experts")
    require(np.isfinite(weights).all() and (weights >= 0).all() and (weights <= 1).all(), "expert weights")
    require(np.allclose(weights.sum(-1), data["router"][:, :, 2], atol=1e-6, rtol=0), "top8 mass")
    sets = manifest["yes_no_token_ids"]
    require(np.array_equal(data["yes_no"], np.isin(data["argmax_id"], sets["YES"] + sets["NO"])),
            "compliance provenance")
    return data, meta, manifest


def analyze(task, data, manifest):
    y, fold, canon, spec = design(task)
    x = np.concatenate([*(data["attention"][:, :, ATT.index(m)] for m in ATT_SELECT),
                        *(data["router"][:, :, j] for j in range(3)),
                        data["output"], data["prompt_len"][:, None], data["sink_draw"]], axis=1)
    columns = [(m, b) for m in ATT_SELECT + ROUTER for b in range(16)]
    columns += [(m, None) for m in OUTPUT + ("prompt_len",)]
    columns += [("sink_draw", b) for b in range(16)]
    require(x.shape == (200, 148), "80/48/3/1/16 family widths")
    xc, yc, fc = x[canon], y[canon], fold[canon]
    observed = crossfit(xc, yc, fc, columns, details=True)
    boot = {key: [] for key in observed["arms"]}
    excluded = 0
    rng = _rng(task, "router-pilot-bootstrap-v1")
    for _ in range(NBOOT):
        take = cluster_boot_take(spec, rng)
        require(np.array_equal(yc[take], yc) and np.array_equal(fc[take], fc), "bootstrap strata")
        if any(len(np.unique(yc[take][fc == f])) < 2 or
               len(np.unique(yc[take][fc != f])) < 2 for f in range(5)):
            excluded += 1
            continue
        result = crossfit(xc[take], yc[take], fc, columns)
        for key in boot:
            boot[key].append(result[key])
    intervals = {key: np.percentile(values, [5, 95], method="linear").tolist() if values else None
                 for key, values in boot.items()}
    null = {key: [] for key in ("router_best", "att_best", "router_minus_comparator")}
    rng = _rng(task, "router-pilot-null-v1")
    for _ in range(NPERM):
        perm = group_outer_perm(spec, len(yc), rng)
        require(np.array_equal(fc[perm], fc), "null crossed folds")
        result = crossfit(xc, yc[perm], fc, columns)
        for key in null:
            null[key].append(result[key])
    sets = manifest["yes_no_token_ids"]
    diagnostic = {"yes_no_count": descriptive(lambda: int(data["yes_no"].sum())),
                  "yes_no_fraction": descriptive(lambda: float(data["yes_no"].mean())),
                  "yes_count": descriptive(lambda: int(np.isin(data["argmax_id"], sets["YES"]).sum())),
                  "no_count": descriptive(lambda: int(np.isin(data["argmax_id"], sets["NO"]).sum())),
                  # Finite inputs can still overflow on subtraction; publish null/reason.
                  "kv_groups_minus_js_max_abs": descriptive(lambda: float(np.max(np.abs(
                      data["attention"][:, :, 2] - data["attention"][:, :, 0])))),
                  # Is sink_draw more than a reweighted bos_mass? Rank correlation per block.
                  "sink_draw_vs_bos_mass_spearman_by_block": [
                      descriptive(lambda: spearman(data["sink_draw"][:, b],
                          data["attention"][:, b, ATT.index("final_bos_mass")]))
                      for b in range(16)]}
    return {"status": "ok", **observed, "diagnostics": diagnostic,
            "fold_map_original_row_order": fold.tolist(),
            "fold_map_sha256": hashlib.sha256(fold.astype("<i8").tobytes()).hexdigest(),
            "canonical_row_order": canon.tolist(), "cluster_spec": spec,
            "bootstrap": {"attempted": NBOOT, "excluded_degenerate": excluded,
                          "included": NBOOT - excluded, "ci_5_95": intervals,
                          "draws": boot}, "permutation_null": {"n": NPERM, "draws": null}}


def markdown(result):
    lines = ["# Router pilot results", "", "Execution verification: not run by Codex.", "",
             f"Spending decision: **{result['decision']}**. {result['decision_reason']}", "", LIMITS, ""]
    for task, cell in result["tasks"].items():
        lines += [f"## {task}", ""]
        if cell["status"] != "ok":
            lines += [f"NOT-RUN: {cell['reason']}", ""]
            continue
        lines += ["| Arm | Mean held-out AUROC / difference | Conditional 5/95 CI |",
                  "|---|---:|---|"]
        for arm, value in cell["arms"].items():
            ci = cell["bootstrap"]["ci_5_95"][arm]
            ci_text = "undefined" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines += [f"| {arm} | {value:.4f} | {ci_text} |"]
        lines += ["", f"Best comparator: {cell['best_comparator_arm']}.", "",
                  f"Bootstrap excluded degenerate replicates: {cell['bootstrap']['excluded_degenerate']}/{NBOOT}.",
                  "", "Per-fold selections:", ""]
        for arm, picks in cell["selections"].items():
            formatted = "; ".join(f"fold {p['fold']}: {p['metric']} block {p['block']} sign {p['direction']:+d}" for p in picks)
            lines += [f"- {arm}: {formatted}"]
        lines += ["", "Descriptive diagnostics:", "", "```json",
                  json.dumps(cell["diagnostics"], indent=2, allow_nan=False), "```", "",
                  "Full training curves, selection dispersion, bootstrap draws and null distributions are in the JSON.", ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE)
    args = ap.parse_args()
    directory = args.results_dir.resolve()
    require(directory.is_relative_to(HERE), "results must remain inside router-pilot")
    jpath, mpath = HERE / "ROUTER_PILOT_RESULTS.json", HERE / "ROUTER_PILOT_RESULTS.md"
    require(not jpath.exists() and not mpath.exists(), "single look: results already exist")
    freeze = json.loads((HERE / "FREEZE.json").read_text())
    stage = freeze["entries"][-1]["stage"]
    check_freeze(freeze, stage, directory if stage == 2 else None)
    statuses = {t: json.loads((directory / f"{t}.status.json").read_text()) for t in TASKS}
    require(all(s["status"] in ("ok", "NOT-RUN") for s in statuses.values()), "both terminal statuses required")
    require(stage == 2 or all(s["status"] == "NOT-RUN" for s in statuses.values()), "success requires FREEZE-2")
    atomic(HERE / "SCORE_STARTED.json", json_bytes({"utc": utc(), "freeze_entry": freeze["entries"][-1]}))
    load_machinery()
    result = {"schema": "router-pilot-results/1", "utc": utc(), "limits": LIMITS,
              "execution_verification": "not run by Codex", "tasks": {},
              "provenance": {"freeze": freeze, "numpy": np.__version__,
                             "status_sha256": {t: sha(directory / f"{t}.status.json") for t in TASKS},
                             "npz_sha256": {}}}
    for task in TASKS:
        status = statuses[task]
        require(status["task"] == task, "terminal task mismatch")
        if status["status"] == "NOT-RUN":
            require(status["freeze_entry"] in freeze["entries"], "NOT-RUN freeze mismatch")
            result["tasks"][task] = {"status": "NOT-RUN", "reason": status["reason"]}
            continue
        data, meta, manifest = load_cell(directory, task, status, freeze)
        result["tasks"][task] = analyze(task, data, manifest)
        result["tasks"][task]["extraction_provenance"] = meta
        result["provenance"]["npz_sha256"][task] = sha(directory / f"{task}.router.npz")
    complete = all(c["status"] == "ok" for c in result["tasks"].values())
    passed = complete and all(c["arms"]["router_best"] >= .65 and
                             c["arms"]["router_minus_comparator"] >= .03 for c in result["tasks"].values())
    result["decision"] = "PROCEED" if passed else "STOP"
    result["decision_reason"] = ("Both task thresholds met; further MoE compute may be proposed." if passed else
                                 "Incomplete pilot: at least one cell NOT-RUN." if not complete else
                                 "At least one task misses a registered threshold.")
    jbytes, mbytes = json_bytes(result), markdown(result).encode()
    jpending, mpending = HERE / ".results-complete.json", HERE / ".results-complete.md"
    atomic(jpending, jbytes)
    atomic(mpending, mbytes)
    os.replace(jpending, jpath)
    os.replace(mpending, mpath)
    print("Completed: ROUTER_PILOT_RESULTS.json and ROUTER_PILOT_RESULTS.md")


if __name__ == "__main__":
    main()
