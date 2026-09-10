#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline test for the CC paper's 29-signal panel claim.

The script is read-only. It loads the *already banked* engineered panel
matrices and the corresponding jsonl data, then compares the paper's
nested out-of-bag select-one-signal procedure against simpler models.

All methods are evaluated under the same group-level bootstrap. The group
is `stem_id`, so paired prompts never straddle a split. This is not a
naive CV comparison.
"""

import os
import re
import sys
import json
import argparse
import hashlib
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


REPORTED_TASK_MEDIANS = {
    "triviaqa_paired_rep": 0.9095,
    "halueval_qa": 0.8751,
    "anli_r1_rep": 0.8410,
    "halueval_summarization": 0.7481,
    "halueval_dialogue": 0.7564,
    "anli_r2": 0.6531,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def as_int_labels(arr):
    """Convert labels to int64 regardless of bool/string representation."""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    try:
        return np.asarray(arr, dtype=int)
    except Exception:
        return np.asarray([int(v) for v in arr], dtype=int)


def safe_auc(y, scores):
    """Return AUROC, or 0.5 when AUROC is undefined."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        return 0.5
    if len(np.unique(scores)) < 2:
        return 0.5
    try:
        return roc_auc_score(y, scores)
    except Exception:
        return 0.5


def clean_1d_train(scores):
    """Impute non-finite values with training median."""
    scores = np.asarray(scores, dtype=float)
    med = np.nanmedian(scores)
    if not np.isfinite(med):
        med = 0.0
    cleaned = np.where(np.isfinite(scores), scores, med)
    return cleaned, med


def clean_1d_with_med(scores, med):
    scores = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(scores), scores, med)


def clean_matrix_train_test(X_train, X_test):
    """Impute column-wise using training medians."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    meds = np.nanmedian(X_train, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)
    X_tr = np.where(np.isfinite(X_train), X_train, meds)
    X_te = np.where(np.isfinite(X_test), X_test, meds)
    return X_tr, X_te, meds


def parse_panel(panel_data):
    """Normalize the npz panel field to a list of [step, family, signal_name]."""
    if isinstance(panel_data, str):
        try:
            panel_data = json.loads(panel_data)
        except Exception:
            return []
    elif isinstance(panel_data, np.ndarray) and panel_data.ndim == 0:
        return parse_panel(panel_data.item())

    entries = []
    for item in panel_data:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                item = None
        if item is None:
            entries.append(None)
        else:
            entries.append(list(item))
    return entries


def find_matrix_files(profiles_dir):
    """Return list of .matrix.npz files under profiles_dir."""
    files = []
    for root, _, fnames in os.walk(profiles_dir):
        for fname in fnames:
            if fname.endswith(".matrix.npz"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def find_data_files(data_dir):
    """Map task name -> jsonl path for filenames matching *_seed*_n*.jsonl."""
    mapping = {}
    if not os.path.isdir(data_dir):
        return mapping
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.endswith(".jsonl"):
                m = re.match(r"^(.*)_seed\d+_n\d+\.jsonl$", fname)
                if m:
                    task = m.group(1)
                    mapping[task] = os.path.join(root, fname)
    return mapping


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    print(f"Warning: skipping bad json line in {path}: {exc}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def generate_group_bootstrap(groups, y, n_bootstrap, rng, max_attempts_per_sample=50):
    """Bootstrap on stem_id groups. Returns train/oob row indices with duplicates."""
    groups = np.asarray(groups)
    y = np.asarray(y)

    # Integer group codes are much faster than string keys.
    group_codes = pd.factorize(groups)[0]
    unique_codes = np.unique(group_codes)
    n_groups = len(unique_codes)

    group_to_rows = {code: np.where(group_codes == code)[0] for code in unique_codes}

    resamples = []
    attempts = 0
    max_total_attempts = n_bootstrap * max_attempts_per_sample + 1000

    while len(resamples) < n_bootstrap and attempts < max_total_attempts:
        attempts += 1
        sampled_codes = rng.choice(unique_codes, size=n_groups, replace=True)

        counts = {code: 0 for code in unique_codes}
        for code in sampled_codes:
            counts[code] += 1

        train_idx = []
        oob_idx = []

        for code in unique_codes:
            rows = group_to_rows[code]
            if counts[code] > 0:
                for _ in range(counts[code]):
                    train_idx.extend(rows)
            else:
                oob_idx.extend(rows)

        if not train_idx or not oob_idx:
            continue

        train_idx = np.asarray(train_idx, dtype=int)
        oob_idx = np.asarray(oob_idx, dtype=int)

        if len(np.unique(y[train_idx])) < 2:
            continue
        if len(np.unique(y[oob_idx])) < 2:
            continue
        if len(oob_idx) < 10:
            continue

        resamples.append({"train": train_idx, "oob": oob_idx})

    if len(resamples) < n_bootstrap:
        print(
            f"Warning: requested {n_bootstrap} bootstrap resamples, got {len(resamples)} valid ones."
        )
    return resamples


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def run_selector(X, y, resamples, mask=None):
    """
    Paper-like selector: within each bootstrap, pick the single best column
    by training AUROC, sign-lock, and score OOB with that column.
    """
    n_cols = X.shape[1]
    if mask is None:
        mask = np.ones(n_cols, dtype=bool)

    out = []
    for res in resamples:
        tr = res["train"]
        oob = res["oob"]

        best_j = -1
        best_sign = 1.0
        best_signed = -1.0

        for j in range(n_cols):
            if not mask[j]:
                continue
            tr_scores, _ = clean_1d_train(X[tr, j])
            if len(np.unique(tr_scores)) < 2:
                continue
            try:
                auc = roc_auc_score(y[tr], tr_scores)
            except Exception:
                continue
            signed = max(auc, 1.0 - auc)
            if signed > best_signed:
                best_signed = signed
                best_j = j
                best_sign = 1.0 if auc >= 0.5 else -1.0

        if best_j < 0:
            out.append(0.5)
            continue

        tr_scores, med = clean_1d_train(X[tr, best_j])
        oob_scores = clean_1d_with_med(X[oob, best_j], med) * best_sign
        out.append(safe_auc(y[oob], oob_scores))

    return np.asarray(out, dtype=float)


def run_all27_lr(X, y, resamples):
    """Multivariate logistic regression on all 27 banked panel signals."""
    out = []
    for res in resamples:
        tr = res["train"]
        oob = res["oob"]
        X_tr, X_te, _ = clean_matrix_train_test(X[tr], X[oob])
        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(X_tr_s, y[tr])
            proba = clf.predict_proba(X_te_s)[:, 1]
            out.append(safe_auc(y[oob], proba))
        except Exception:
            out.append(0.5)

    return np.asarray(out, dtype=float)


def select_fixed_column_from_deployments(deployments):
    """Pool rows from deployments and select one column by sign-locked AUROC."""
    if not deployments:
        return None

    pool_X = np.concatenate([d["X"] for d in deployments], axis=0)
    pool_y = np.concatenate([d["y"] for d in deployments], axis=0)

    n_cols = pool_X.shape[1]
    best_j = -1
    best_sign = 1.0
    best_signed = -1.0

    for j in range(n_cols):
        scores, _ = clean_1d_train(pool_X[:, j])
        if len(np.unique(scores)) < 2:
            continue
        try:
            auc = roc_auc_score(pool_y, scores)
        except Exception:
            continue
        signed = max(auc, 1.0 - auc)
        if signed > best_signed:
            best_signed = signed
            best_j = j
            best_sign = 1.0 if auc >= 0.5 else -1.0

    if best_j < 0:
        return None
    return best_j, best_sign


def run_fixed_column(X, y, resamples, col, sign):
    """Evaluate a fixed column/sign under the same group bootstrap."""
    out = []
    for res in resamples:
        tr = res["train"]
        oob = res["oob"]
        _, med = clean_1d_train(X[tr, col])
        oob_scores = clean_1d_with_med(X[oob, col], med) * sign
        out.append(safe_auc(y[oob], oob_scores))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_deployments(matrix_mapping, data_map):
    deployments = []
    for (task, model), npz_path in sorted(matrix_mapping.items()):
        if task not in data_map:
            print(f"Warning: no data jsonl for task {task}; skipping {model}")
            continue

        jsonl_path = data_map[task]
        df = load_jsonl(jsonl_path)

        required_cols = {"label", "stem_id"}
        if not required_cols.issubset(df.columns):
            print(f"Warning: {jsonl_path} missing {required_cols - set(df.columns)}; skipping")
            continue

        try:
            npz = np.load(npz_path, allow_pickle=True)
            X = npz["score_matrix"].astype(float)
            y_npz = as_int_labels(npz["labels"])
            sample_idx = np.asarray(npz["sample_idx"]).flatten().astype(int)
            panel_data = npz["panel"] if "panel" in npz else None
        except Exception as exc:
            print(f"Error loading {npz_path}: {exc}")
            continue

        if not (len(X) == len(y_npz) == len(sample_idx)):
            print(f"Warning: shape mismatch in {npz_path}; skipping")
            continue

        if np.any(sample_idx < 0) or np.any(sample_idx >= len(df)):
            print(f"Error: sample_idx out of range for {npz_path}; skipping")
            continue

        df_sel = df.iloc[sample_idx].reset_index(drop=True)
        labels_df = as_int_labels(df_sel["label"])

        # Verify alignment rather than assuming it.
        if not np.array_equal(labels_df, y_npz):
            # Fall back to sequential order and retry.
            if len(df) == len(X) and np.array_equal(as_int_labels(df["label"]), y_npz):
                df_sel = df.iloc[: len(X)].reset_index(drop=True)
                print(
                    f"Warning: sample_idx alignment failed for {npz_path}; "
                    "using sequential jsonl order because labels match."
                )
            else:
                print(
                    f"Error: label mismatch between {npz_path} and {jsonl_path}; "
                    "refusing to use potentially misaligned data."
                )
                continue

        groups = df_sel["stem_id"].to_numpy()
        if pd.isna(groups).any():
            print(f"Error: missing stem_id for {npz_path}; skipping")
            continue

        panel_entries = parse_panel(panel_data) if panel_data is not None else []
        if len(panel_entries) != X.shape[1]:
            print(
                f"Warning: panel length {len(panel_entries)} != column count {X.shape[1]} "
                f"for {npz_path}; family masks may be incomplete."
            )
            panel_entries = [None] * X.shape[1]

        deployments.append(
            {
                "task": task,
                "model": model,
                "X": X,
                "y": y_npz,
                "groups": groups,
                "panel_entries": panel_entries,
                "npz_path": npz_path,
            }
        )

    return deployments


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def percentile_ci(arr, lower=2.5, upper=97.5):
    if arr is None or len(arr) == 0:
        return float("nan"), float("nan")
    return np.percentile(arr, [lower, upper])


def method_row_str(method_row, paper_row):
    pooled_median = method_row["pooled_median"]
    lo, hi = method_row["pooled_ci"]
    if paper_row is not None and method_row["method"] != "paper_selector":
        diff = method_row["pooled"] - paper_row["pooled"]
        med_diff = np.median(diff)
        dlo, dhi = percentile_ci(diff)
        beat = np.mean(diff > 0)
        return (
            f"{method_row['method']:<20s} "
            f"{method_row['median_across_models']:.4f} "
            f"[{method_row['min']:.4f},{method_row['max']:.4f}] "
            f"{pooled_median:.4f} [{lo:.4f},{hi:.4f}] "
            f"diff vs panel: {med_diff:+.4f} [{dlo:+.4f},{dhi:+.4f}] "
            f"beat={beat * 100:.0f}%"
        )
    else:
        return (
            f"{method_row['method']:<20s} "
            f"{method_row['median_across_models']:.4f} "
            f"[{method_row['min']:.4f},{method_row['max']:.4f}] "
            f"{pooled_median:.4f} [{lo:.4f},{hi:.4f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="Baseline test for the panel-selector claim.")
    parser.add_argument("--data-dir", required=True, help="Root directory containing data_bench/")
    parser.add_argument("--profiles-dir", required=True, help="Root directory containing profiles_bench/ and/or profiles_ext/")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pre-commit interpretation
    # ------------------------------------------------------------------
    print("=" * 100)
    print("PRE-COMMITTED INTERPRETATION")
    print("=" * 100)
    print(
        "The panel's select-one-signal machinery is considered NOT justified for a task if\n"
        "any simpler comparator built from the same banked data is not reliably worse than\n"
        "the paper selector.\n"
    )
    print(
        "Operationally: for comparator C we compute paired differences C - paper_selector\n"
        "over identical bootstrap resamples. If the 95% percentile interval of that\n"
        "difference includes 0 or any positive value (i.e., upper bound >= 0), then the\n"
        "simpler comparator is 'not reliably worse' and the panel machinery is NOT\n"
        "justified for that task. The panel machinery is justified only if it is reliably\n"
        "better than ALL simpler comparators: the (C - paper_selector) interval lies\n"
        "entirely below 0.\n"
    )
    print(
        "Comparators:\n"
        "  all27_lr            logistic regression on all 27 banked signals\n"
        "  attention_selector  select-one-signal restricted to Attention family\n"
        "  non_attention_sel   select-one-signal restricted to non-Attention families\n"
        "  confidence_selector select-one-signal restricted to confidence family (if any)\n"
        "  fixed_loo_model     one fixed column chosen from other models of the same task\n"
        "  fixed_loto          one fixed column chosen from all other tasks\n"
    )
    print("=" * 100)
    print()

    # ------------------------------------------------------------------
    # Locate files
    # ------------------------------------------------------------------
    data_map = find_data_files(args.data_dir)
    matrix_files = find_matrix_files(args.profiles_dir)

    if not matrix_files:
        print("No .matrix.npz files found under profiles dir.")
        sys.exit(1)

    # De-duplicate if both profiles_bench and profiles_ext contain same (task, model).
    matrix_mapping = {}
    for path in matrix_files:
        task = os.path.basename(os.path.dirname(path))
        model = os.path.basename(path)[: -len(".matrix.npz")]
        key = (task, model)
        if key not in matrix_mapping:
            matrix_mapping[key] = path
        else:
            # Prefer profiles_bench if both exist.
            if "profiles_bench" in path and "profiles_bench" not in matrix_mapping[key]:
                matrix_mapping[key] = path

    deployments = load_deployments(matrix_mapping, data_map)

    if not deployments:
        print("No valid deployments loaded.")
        sys.exit(1)

    tasks = sorted({d["task"] for d in deployments})
    models_by_task = {t: sorted({d["model"] for d in deployments if d["task"] == t}) for t in tasks}

    print(f"Bootstrap resamples per deployment: {args.n_bootstrap}")
    print(f"Number of valid deployments: {len(deployments)}")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    # ------------------------------------------------------------------
    # Precompute leave-one-task-out fixed columns
    # ------------------------------------------------------------------
    fixed_loto_info = {}
    for task in tasks:
        other_deployments = [d for d in deployments if d["task"] != task]
        fixed_loto_info[task] = select_fixed_column_from_deployments(other_deployments)

    # ------------------------------------------------------------------
    # Evaluate every deployment
    # ------------------------------------------------------------------
    results = {}

    for dep in deployments:
        task = dep["task"]
        model = dep["model"]
        X = dep["X"]
        y = dep["y"]
        groups = dep["groups"]
        panel_entries = dep["panel_entries"]

        # Stable, order-independent seed.
        digest = hashlib.md5(f"{task}:{model}".encode("utf-8")).hexdigest()[:8]
        rng_seed = (args.seed + int(digest, 16)) % (2**32)
        rng = np.random.default_rng(rng_seed)

        resamples = generate_group_bootstrap(groups, y, args.n_bootstrap, rng)

        # Family masks.
        families = []
        for e in panel_entries:
            if e is not None and len(e) > 1:
                families.append(str(e[1]))
            else:
                families.append("unknown")
        families = np.asarray(families, dtype=object)

        attention_mask = np.asarray([f.lower() == "attention" for f in families], dtype=bool)
        non_attention_mask = ~attention_mask
        confidence_mask = np.asarray(["confidence" in f.lower() for f in families], dtype=bool)

        aucs = {}
        aucs["paper_selector"] = run_selector(X, y, resamples, mask=None)
        aucs["all27_lr"] = run_all27_lr(X, y, resamples)

        if attention_mask.any():
            aucs["attention_selector"] = run_selector(X, y, resamples, mask=attention_mask)
        if non_attention_mask.any():
            aucs["non_attention_sel"] = run_selector(X, y, resamples, mask=non_attention_mask)
        if confidence_mask.any():
            aucs["confidence_selector"] = run_selector(X, y, resamples, mask=confidence_mask)

        # Fixed column chosen from other models in the same task.
        other_models = [d for d in deployments if d["task"] == task and d["model"] != model]
        fixed_loo = select_fixed_column_from_deployments(other_models)
        if fixed_loo is not None:
            col, sign = fixed_loo
            aucs["fixed_loo_model"] = run_fixed_column(X, y, resamples, col, sign)

        # Fixed column chosen from all other tasks.
        if len(tasks) > 1 and fixed_loto_info.get(task) is not None:
            col, sign = fixed_loto_info[task]
            aucs["fixed_loto"] = run_fixed_column(X, y, resamples, col, sign)

        results[(task, model)] = aucs

    # ------------------------------------------------------------------
    # Per-task report
    # ------------------------------------------------------------------
    method_order = [
        "paper_selector",
        "all27_lr",
        "attention_selector",
        "non_attention_sel",
        "confidence_selector",
        "fixed_loo_model",
        "fixed_loto",
    ]

    summary_counts = {m: {"paper_better": 0, "no_diff": 0, "comparator_better": 0} for m in method_order[1:]}

    for task in tasks:
        models = models_by_task[task]

        print("=" * 100)
        reported_str = (
            f"{REPORTED_TASK_MEDIANS[task]:.4f}"
            if task in REPORTED_TASK_MEDIANS
            else "not reported"
        )
        print(f"Task: {task}")
        print(f"Reported panel OOB median: {reported_str}")
        print(f"Deployments: {', '.join(models)}")
        print("=" * 100)

        print(
            f"{'Method':<20s} {'Median':>7s} {'[min,max]':>18s} "
            f"{'Pooled median [95% CI]':>27s} {'Diff vs panel [95% CI]':>35s} {'Beat':>5s}"
        )

        method_rows = {}
        for method in method_order:
            arrays = []
            for model in models:
                arr = results.get((task, model), {}).get(method)
                if arr is not None and len(arr) > 0:
                    arrays.append(arr)
            if not arrays:
                continue

            pooled = np.concatenate(arrays)
            model_medians = [np.median(a) for a in arrays]
            method_rows[method] = {
                "method": method,
                "model_medians": model_medians,
                "median_across_models": np.median(model_medians),
                "min": np.min(model_medians),
                "max": np.max(model_medians),
                "pooled_median": np.median(pooled),
                "pooled_ci": percentile_ci(pooled),
                "pooled": pooled,
            }

        paper_row = method_rows.get("paper_selector")
        for method in method_order:
            row = method_rows.get(method)
            if row is None:
                continue
            print(method_row_str(row, paper_row))

        # Per-deployment medians.
        print("\nPer-deployment medians:")
        header_cols = ["Deployment"] + [m for m in method_order if m in method_rows]
        print("  " + "  ".join(f"{h:>16s}" for h in header_cols))
        for model in models:
            vals = []
            for method in method_order:
                if method not in method_rows:
                    continue
                arr = results.get((task, model), {}).get(method)
                if arr is not None and len(arr) > 0:
                    vals.append(f"{np.median(arr):.4f}")
                else:
                    vals.append("-")
            line = [f"{model:>16s}"]
            for v in vals:
                line.append(f"{v:>16s}")
            print("  " + "  ".join(line))
        print()

        # Task-level decisions for summary.
        if paper_row is not None:
            for method in method_order[1:]:
                row = method_rows.get(method)
                if row is None:
                    continue
                diff = row["pooled"] - paper_row["pooled"]
                dlo, dhi = percentile_ci(diff)
                if dhi < 0:
                    summary_counts[method]["paper_better"] += 1
                elif dlo > 0:
                    summary_counts[method]["comparator_better"] += 1
                else:
                    summary_counts[method]["no_diff"] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        "For each comparator, number of tasks where paper_selector is reliably better, "
        "where there is no reliable difference, and where the comparator is reliably better."
    )
    print(f"{'Comparator':<20s} {'Paper better':>12s} {'No diff':>10s} {'Comparator better':>18s}")
    for method in method_order[1:]:
        counts = summary_counts.get(method)
        if counts is None:
            continue
        total_valid = counts["paper_better"] + counts["no_diff"] + counts["comparator_better"]
        print(
            f"{method:<20s} {counts['paper_better']:>12d} {counts['no_diff']:>10d} "
            f"{counts['comparator_better']:>18d}   (valid tasks: {total_valid})"
        )

    print()
    print("Interpretation: if any 'No diff' or 'Comparator better' count is nonzero for")
    print("a comparator, the panel selector is not justified against that comparator in")
    print("those tasks. It is justified only where it is reliably better than all comparators.")
    print()


if __name__ == "__main__":
    main()
