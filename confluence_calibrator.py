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
def _rows_to_readout(rows: List[Dict[str, Any]]):
    """Parse per-sample readout rows -> (X, y, sample_idx), dropping any non-finite row."""
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
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), np.array(idx, dtype=np.int64))


def load_readout_matrix(rpv_json_path: str) -> Dict[str, Any]:
    """Per-sample readout features from an EXISTING RPV comprehensive run (sealed-seed reuse)."""
    d = json.load(open(rpv_json_path))
    X, y, idx = _rows_to_readout(d.get("rows", []))
    return {
        "model": d.get("model"), "slug": (d.get("model") or "").split("/")[-1],
        "benchmark": d.get("benchmark"), "data_path": d.get("data_path"),
        "data_hash": d.get("diagnostics", {}).get("data_hash_sha256"),
        "score_matrix": X, "labels": y, "sample_idx": idx, "panel": list(READOUT_PANEL), "n": len(y),
    }


def collect_readout_matrix_fresh(model_id: str, benchmark: str, data_path: str, *,
                                 seed: int, limit: int | None = None) -> Dict[str, Any]:
    """FRESH readout pass (RPV + null_ratio + surprise + p_max) at the gen_step=1 commit instant,
    computed by importing the sealed-grade shadow-ambiguity compute (trace_pair_features) - NOT by
    reusing existing rows. This is the readout arm for the fresh-seed seal."""
    from pathlib import Path
    SHADOW = os.path.join(T0_REPO, "exploratory/shadow-ambiguity")
    if SHADOW not in sys.path:
        sys.path.insert(0, SHADOW)
    import comprehensive_run as CR
    fr = CR.trace_pair_features(model_id, benchmark, Path(data_path),
                                limit=(limit or 0), max_new_tokens=1,
                                k_support=CR.K_SUPPORT_DEFAULT, seed=seed)
    X, y, idx = _rows_to_readout(fr.rows)
    dh = (fr.diagnostics or {}).get("data_hash_sha256")
    return {
        "model": model_id, "slug": model_id.split("/")[-1], "benchmark": benchmark,
        "data_path": str(data_path), "data_hash": dh,
        "score_matrix": X, "labels": y, "sample_idx": idx, "panel": list(READOUT_PANEL),
        "n": len(y), "drops": fr.drops,
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


# ──────────────────────────────────────────────────────────────────────────────
# ACE attention pass - replicates the sealed collection loop by IMPORT (R2: no edits
# to pri_calibrator). Identical code path -> per-sample ACE scores must reproduce the
# sealed profile's AUROC for the same (model, data); that is the wiring correctness gate.
# ──────────────────────────────────────────────────────────────────────────────
def collect_ace_matrix(model_slug: str, jsonl_path: str, *, seed: int,
                       max_new_tokens: int = 8, limit: int | None = None,
                       panel: List[PanelCell] | None = None) -> Dict[str, Any]:
    # S1 fix: default to the SEALED ACE instrument (21-cell, with v-norms) so the unified
    # panel can actually select the sealed per-model winners (some are v_norm cells).
    from diagnose_inter_head_disagreement import (
        _find_layers, _target_layer_map, attention_capture, attention_capture_with_values)
    panel = list(panel or SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS)
    capture_v_norms = SEAL._requires_v_norm_capture(panel)
    state = SEAL.load_calibration_state(model_slug, layer_name="final", seed=seed)
    prompts, labels, data_hash = SEAL._load_calibration_jsonl(jsonl_path)
    if limit is not None:
        prompts, labels = prompts[:limit], labels[:limit]
    decoder_layers = _find_layers(state.model)
    target_map = _target_layer_map(len(decoder_layers))
    n_kv: Dict[str, int] = {}
    for tag, idx in target_map.items():
        k = getattr(decoder_layers[idx].self_attn, "n_kv_heads", None)
        if k is None:
            k = getattr(decoder_layers[idx].self_attn, "n_heads", None)
        if k is not None:
            n_kv[tag] = int(k)
    n = len(prompts)
    sm = np.full((n, len(panel)), np.nan, dtype=np.float64)
    for i, prompt in enumerate(prompts):
        v_caps_snap = None
        if capture_v_norms:
            with attention_capture_with_values(decoder_layers, target_map) as (caps, v_caps):
                trace = SEAL._trace_one_prompt(state.model, state.tokenizer, state.projection,
                                               state.layer_indices, prompt, state.prompt_strategy,
                                               max_new_tokens)
                sample_caps = {tag: list(caps[tag]) for tag in caps}
                v_caps_snap = {tag: list(v_caps[tag]) for tag in v_caps}
        else:
            with attention_capture(decoder_layers, target_map) as caps:
                trace = SEAL._trace_one_prompt(state.model, state.tokenizer, state.projection,
                                               state.layer_indices, prompt, state.prompt_strategy,
                                               max_new_tokens)
                sample_caps = {tag: list(caps[tag]) for tag in caps}
        per_cell = SEAL._compute_panel_scores_for_sample(
            state.pri_computer, trace, state.layer_name, panel, alpha=1.0,
            attention_captures=sample_caps, attention_n_kv_heads=n_kv,
            attention_v_norm_captures=v_caps_snap)
        for j, cell in enumerate(panel):
            v = per_cell.get(cell)
            if v is not None:
                sm[i, j] = float(v)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[ace] {model_slug.split('/')[-1]} {i+1}/{n}", flush=True)
    return {"score_matrix": sm, "labels": np.asarray(labels, dtype=np.int64),
            "panel": panel, "data_hash": data_hash, "slug": model_slug.split("/")[-1],
            "sample_idx": np.arange(n, dtype=np.int64), "n": n}


def merge_and_calibrate(ace: Dict[str, Any], readout: Dict[str, Any], *,
                        n_bootstrap: int = 2000, seed: int = 20260610) -> Dict[str, Any]:
    """Merge the ACE attention sub-matrix with the readout sub-matrix (R3: one source per
    family), aligning by sample_idx (robust to the readout run dropping non-finite rows),
    with a HARD label-alignment assert on the intersection, then run the dual-endpoint
    dispatcher over the full merged panel."""
    ia = {int(s): k for k, s in enumerate(ace["sample_idx"])}
    ib = {int(s): k for k, s in enumerate(readout["sample_idx"])}
    common = sorted(set(ia) & set(ib))
    if len(common) < 4:
        raise AssertionError(f"too few aligned samples: |common|={len(common)}")
    ra = [ia[s] for s in common]; rb = [ib[s] for s in common]
    ya = ace["labels"][ra]; yb = readout["labels"][rb]
    if not np.array_equal(ya, yb):
        raise AssertionError(f"label misalignment on {len(common)} shared sample_idx: "
                             f"{int((ya != yb).sum())} disagree")
    M = np.hstack([ace["score_matrix"][ra], readout["score_matrix"][rb]])
    panel = list(ace["panel"]) + list(readout["panel"])
    ya_dropped = len(ace["labels"]) - len(common)
    full = run_selection(M, ya, panel, n_bootstrap=n_bootstrap, seed=seed)
    geom_keys = {c[2] for c in panel if c[2] not in CONFIDENCE_KEYS}
    geom = run_selection(M, ya, panel, n_bootstrap=n_bootstrap, seed=seed, restrict_keys=geom_keys)
    return {
        "schema_version": "confluence/0.1-unified",
        "slug": ace.get("slug"), "n": len(ya), "n_aligned": len(common),
        "n_dropped_unaligned": ya_dropped, "n_cells_total": len(panel),
        "n_ace_cells": len(ace["panel"]), "n_readout_cells": len(readout["panel"]),
        "ace_data_hash": ace.get("data_hash"), "readout_data_hash": readout.get("data_hash"),
        "primary_full_panel": full, "secondary_geometric_only": geom,
        "provenance": {"seed": seed, "n_bootstrap": n_bootstrap,
                       "sealed_selector": "_nested_bootstrap_oob_auroc (imported, not modified)"},
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
