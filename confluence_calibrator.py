#!/usr/bin/env python3
"""
confluence_calibrator - the unified commit-moment dispatcher.

Composes (does NOT reimplement) the sealed nested-OOB selection from
t0-morphology-furnace/pri_calibrator.py over a MERGED candidate panel:

    ACE attention cells  (attention pass)        - W_u-free routing morphology
    null_ratio           (readout pass, 1 source)- v3 residual-motion detector
    RPV stats            (readout pass)           - fisher_eff_rank + spectral_entropy + logvol
    surprise, p_max      (readout pass)           - confidence base

The sealed selector is IMPORTED read-only; we never edit the frozen ACE/T0 core.
Sign-lock + OOB honesty are inherited verbatim from `_nested_bootstrap_oob_auroc`.

Run with the t0 venv (has sklearn/mlx):
    /Users/msrk/Documents/t0-morphology-furnace/.venv/bin/python confluence_calibrator.py ...
"""
from __future__ import annotations
import json, os, sys
from typing import Any, Dict, List, Tuple
import numpy as np

T0_REPO = "/Users/msrk/Documents/t0-morphology-furnace"
if T0_REPO not in sys.path:
    sys.path.insert(0, T0_REPO)

# --- sealed machinery, imported read-only (R2) ---
import pri_calibrator as SEAL  # noqa: E402

PanelCell = Tuple[int, str, str]

# Readout-pass families. null_ratio is sourced HERE only (R3) - never also from the
# attention/residual pass - so the metric has exactly one definition in the merge.
READOUT_PANEL: List[PanelCell] = [
    (0, "Readout", "fisher_eff_rank"),       # RPV primary
    (0, "Readout", "spectral_entropy"),      # RPV secondary
    (0, "Readout", "neg_shadow_logvol_r1"),  # RPV secondary
    (0, "Readout", "null_ratio_post_rank1"), # v3 residual-motion detector
    (0, "Readout", "surprise"),              # confidence base
    (0, "Readout", "p_max"),                 # confidence base
]
READOUT_KEYS = {c: c[2] for c in READOUT_PANEL}  # cell -> row dict key

GEOMETRIC_FAMILIES = {"fisher_eff_rank", "spectral_entropy", "neg_shadow_logvol_r1",
                      "null_ratio_post_rank1"}  # ACE attention cells are geometric too
CONFIDENCE_KEYS = {"surprise", "p_max"}


# ──────────────────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_readout_matrix(rpv_json_path: str) -> Dict[str, Any]:
    """Per-sample readout features + labels from an existing RPV comprehensive run.
    Drops rows with any non-finite feature so the merged matrix is clean."""
    d = json.load(open(rpv_json_path))
    rows = d.get("rows", [])
    keys = [READOUT_KEYS[c] for c in READOUT_PANEL]
    X, y, idx = [], [], []
    for r in rows:
        vals = [r.get(k) for k in keys]
        if any(v is None for v in vals):
            continue
        fv = [float(v) for v in vals]
        if not all(np.isfinite(fv)):
            continue
        X.append(fv); y.append(int(r["label"])); idx.append(int(r.get("sample_idx", len(idx))))
    return {
        "model": d.get("model"), "slug": (d.get("model") or "").split("/")[-1],
        "benchmark": d.get("benchmark"), "data_path": d.get("data_path"),
        "data_hash": d.get("diagnostics", {}).get("data_hash_sha256"),
        "score_matrix": np.array(X, dtype=np.float64), "labels": np.array(y, dtype=np.int64),
        "sample_idx": np.array(idx, dtype=np.int64), "panel": list(READOUT_PANEL),
        "n": len(y),
    }


# ──────────────────────────────────────────────────────────────────────────────
# selection (wraps the sealed nested-OOB selector)
# ──────────────────────────────────────────────────────────────────────────────
def run_selection(score_matrix: np.ndarray, labels: np.ndarray, panel: List[PanelCell],
                  *, n_bootstrap: int = 2000, seed: int = 20260610,
                  restrict_keys: set | None = None) -> Dict[str, Any]:
    """Run the SEALED nested-OOB selector over (a column-subset of) the panel.
    restrict_keys: keep only cells whose detail (c[2]) is in this set (e.g. drop
    confidence cells for the geometric-only endpoint)."""
    if restrict_keys is not None:
        cols = [j for j, c in enumerate(panel) if c[2] in restrict_keys]
        sm = score_matrix[:, cols]
        pn = [panel[j] for j in cols]
    else:
        sm, pn = score_matrix, list(panel)

    oob = SEAL._nested_bootstrap_oob_auroc(sm, labels, pn, n_bootstrap, seed)

    # full-sample marginal per cell (reference only; deployability uses OOB)
    marg = {}
    for j, c in enumerate(pn):
        auc, sign, _ = SEAL._score_candidate(sm[:, j], labels)
        marg[SEAL._cell_label(c)] = {"auroc": None if not np.isfinite(auc) else round(float(auc), 4),
                                     "sign": int(sign)}
    # selected winner = most-frequent in-bag winner (the dispatcher's honest pick)
    wc = oob.get("winner_counts", {})
    winner = max(wc, key=wc.get) if wc else None
    ci_lo = oob.get("oob_auroc_ci_lo")
    return {
        "winner": winner,
        "oob_auroc_median": oob.get("oob_auroc_median"),
        "oob_auroc_ci_lo": ci_lo, "oob_auroc_ci_hi": oob.get("oob_auroc_ci_hi"),
        "winner_stability": oob.get("winner_stability"),
        "winner_counts": wc, "oob_n_bootstrap_used": oob.get("oob_n_bootstrap_used"),
        "n_cells": len(pn), "deployable": bool(ci_lo is not None and np.isfinite(ci_lo) and ci_lo > 0.50),
        "full_sample_marginals": marg,
    }


def calibrate_cell(loaded: Dict[str, Any], *, n_bootstrap: int = 2000,
                   seed: int = 20260610) -> Dict[str, Any]:
    """Produce both endpoints for one (model, task) cell from a loaded score matrix."""
    sm, y, pn = loaded["score_matrix"], loaded["labels"], loaded["panel"]
    full = run_selection(sm, y, pn, n_bootstrap=n_bootstrap, seed=seed)              # PRIMARY (incl. confidence)
    geom_keys = {c[2] for c in pn if c[2] not in CONFIDENCE_KEYS}
    geom = run_selection(sm, y, pn, n_bootstrap=n_bootstrap, seed=seed,
                         restrict_keys=geom_keys)                                     # SECONDARY (geometric-only)
    return {
        "schema_version": "confluence/0.1",
        "model": loaded.get("model"), "slug": loaded.get("slug"),
        "benchmark": loaded.get("benchmark"), "n": loaded["n"],
        "data_hash": loaded.get("data_hash"),
        "primary_full_panel": full,
        "secondary_geometric_only": geom,
        "provenance": {
            "seed": seed, "n_bootstrap": n_bootstrap,
            "sealed_module": "t0-morphology-furnace/pri_calibrator.py",
            "sealed_selector": "_nested_bootstrap_oob_auroc (imported, not modified)",
            "panel_keys": [c[2] for c in pn],
        },
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="confluence calibrator (readout-only path / cross-check)")
    p.add_argument("--rpv-json", required=True, help="existing RPV comprehensive output json")
    p.add_argument("--out", default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260610)
    a = p.parse_args()
    loaded = load_readout_matrix(a.rpv_json)
    prof = calibrate_cell(loaded, n_bootstrap=a.n_bootstrap, seed=a.seed)
    s = json.dumps(prof, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)
