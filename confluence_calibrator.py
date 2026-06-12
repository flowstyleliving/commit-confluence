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
    "$CONFLUENCE_T0_REPO"/.venv/bin/python confluence_calibrator.py ...
"""
from __future__ import annotations
import json, os, sys
from typing import Any, Dict, List, Tuple
import numpy as np

# Path to the sealed ACE/T0 dependency repo. Override with $CONFLUENCE_T0_REPO; defaults to a
# sibling under the home dir so the committed source carries no absolute username path.
T0_REPO = os.environ.get("CONFLUENCE_T0_REPO",
                         os.path.expanduser("~/Documents/t0-morphology-furnace"))
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
# E4: full fusion contains surprise -> excluded from the geometric endpoint alongside confidence
NON_GEOMETRIC_KEYS = CONFIDENCE_KEYS | {"fusion_rank_mean_full"}
FUSION_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "stage_b", "fusion_signs.json")


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
        # C6: sample_idx is REQUIRED (KeyError if absent). The old `len(idx)` fallback would
        # silently compact indices and misalign with the ACE pass exactly when rows drop.
        X.append(fv); y.append(int(r["label"])); idx.append(int(r["sample_idx"]))
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
        # C5: the OOB CI aggregates resamples whose in-bag winners may DIFFER; `winner` is the
        # modal in-bag pick. `deployable` therefore certifies the selection PROCEDURE on this
        # panel, not the named cell in isolation. winner_marginal = that cell's own full-sample
        # marginal, for reference only.
        "ci_semantics": "procedure-level OOB CI (in-bag winners vary across resamples); winner = modal in-bag pick",
        "winner_marginal": marg.get(winner),
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
    geom_keys = {c[2] for c in pn if c[2] not in NON_GEOMETRIC_KEYS}
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
# E4 fusion cells - pre-registered cross-locus candidates (spec: stage_b/fusion_signs.json,
# frozen from SEALED-ERA artifacts only; see stage_b/build_fusion_spec.py). The fusion
# columns are PRECOMPUTED before the bootstrap, so orientations must be a priori - never
# fit on the data being calibrated.
# ──────────────────────────────────────────────────────────────────────────────
def _rank01(v: np.ndarray) -> np.ndarray:
    """NaN-aware rank transform to (0,1): (avg_rank - 0.5)/n_finite; NaN stays NaN."""
    from scipy.stats import rankdata
    out = np.full(v.shape, np.nan, dtype=np.float64)
    f = np.isfinite(v)
    n = int(f.sum())
    if n == 0:
        return out
    out[f] = (rankdata(v[f], method="average") - 0.5) / n
    return out


def append_fusion_columns(M: np.ndarray, panel: List[PanelCell], slug: str, benchmark: str,
                          spec: Dict[str, Any] | None = None):
    """Append the two pre-registered fusion cells. Component orientations locked per
    (model, task) from sealed-era artifacts; missing entries -> cohort-modal fallback.
    Fusion = mean of oriented component ranks; NaN if ANY component non-finite."""
    if spec is None:
        spec = json.load(open(FUSION_SPEC_PATH))
    key = f"{slug}|{benchmark}"
    bylabel = {c[2]: j for j, c in enumerate(panel)}
    ace_detail = spec["ace_component"]["column_name"].split("::")[-1]
    ace_sign = spec["ace_component"]["per_cell_sign"].get(key,
               spec["ace_component"]["modal_sign_fallback"])
    ro_sign = spec["readout_per_cell_sign"].get(key, {})
    comp_cols: Dict[str, Tuple[str, int]] = {"ACE_modal": (ace_detail, int(ace_sign))}
    for comp in spec["readout_components"]:
        s = ro_sign.get(comp) or spec["readout_modal_sign_fallback"][comp]  # 0/missing -> modal
        comp_cols[comp] = (comp, int(s))
    ranked, used = {}, {}
    for name, (detail, sgn) in comp_cols.items():
        j = bylabel.get(detail)
        if j is None:
            raise KeyError(f"fusion component '{detail}' not in merged panel")
        ranked[name] = _rank01(M[:, j] * float(sgn))
        used[name] = {"column": detail, "sign": sgn,
                      "source": "per_cell" if (name == "ACE_modal" and key in spec["ace_component"]["per_cell_sign"])
                                or (name != "ACE_modal" and bool(ro_sign.get(name))) else "modal_fallback"}
    cols, pan = [], list(panel)
    for fname, fdef in sorted(spec["fusion_cells"].items()):
        cols.append(np.vstack([ranked[c] for c in fdef["components"]]).mean(axis=0))
        pan.append((0, "Fusion", fname))
    return np.hstack([M, np.column_stack(cols)]), pan, {"key": key, "components": used}


# ──────────────────────────────────────────────────────────────────────────────
# C2 controls + provenance
# ──────────────────────────────────────────────────────────────────────────────
def shuffled_label_control(M: np.ndarray, y: np.ndarray, panel: List[PanelCell], *,
                           n_bootstrap: int, seed: int, restrict_keys: set | None = None,
                           k_perms: int = 3) -> Dict[str, Any]:
    """Pre-registered per-cell control: re-run the FULL nested-OOB selection on permuted
    labels. A single permutation's 95% CI excludes 0.5 upward ~2.5% of the time by chance,
    so the FLAG requires >=2 of k_perms permutations with CI_lo > 0.50 (amendment A3)."""
    perms = []
    for k in range(k_perms):
        rng = np.random.RandomState(seed + 90210 + k)
        yp = y[rng.permutation(len(y))]
        r = run_selection(M, yp, panel, n_bootstrap=n_bootstrap, seed=seed,
                          restrict_keys=restrict_keys)
        lo = r.get("oob_auroc_ci_lo")
        perms.append({"oob_auroc_median": r.get("oob_auroc_median"), "oob_auroc_ci_lo": lo,
                      "oob_auroc_ci_hi": r.get("oob_auroc_ci_hi"),
                      "excludes_null_upward": bool(lo is not None and np.isfinite(lo) and lo > 0.50)})
    n_excl = sum(p["excludes_null_upward"] for p in perms)
    return {"k_perms": k_perms, "n_excluding_null_upward": n_excl,
            "pass": bool(n_excl < 2), "perms": perms}


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def module_hashes() -> Dict[str, str]:
    """Drift hashes for every imported/composed module + the frozen fusion spec (pre-reg
    Controls: recorded per profile; must match the cross-check-era artifacts). Prefers the
    ACTUALLY-IMPORTED module's __file__ (sys.modules) over the expected path - hashing a
    different file than the one executed would be a provenance bug."""
    expected = {
        "pri_calibrator": os.path.join(T0_REPO, "pri_calibrator.py"),
        "comprehensive_run": os.path.join(T0_REPO, "exploratory/shadow-ambiguity/comprehensive_run.py"),
        "diagnose_inter_head_disagreement": os.path.join(T0_REPO, "scripts/diagnose_inter_head_disagreement.py"),
        # M5: the readout/ACE hot path also executes the sealed runtime + IO/MLX plugins +
        # model adapters. Hashing only the top-level modules let unrecorded runtime code drift
        # while the drift hashes still matched. Cover every module the forward path touches.
        "pri_runtime": os.path.join(T0_REPO, "pri_runtime.py"),
        "pri_v2_io_plugins": os.path.join(T0_REPO, "pri_v2_io_plugins.py"),
        "pri_v2_mlx_pipeline": os.path.join(T0_REPO, "pri_v2_mlx_pipeline.py"),
        "model_adapters": os.path.join(T0_REPO, "model_adapters.py"),
        # M5-fix-2: comprehensive_run imports the RPV statistic fns (fisher_eff_rank,
        # fisher_spectral_entropy, shadow_logvol_post_rank) from test_shadow_ambiguity - that
        # module IS executed on the readout path, so its drift must be recorded too.
        "test_shadow_ambiguity": os.path.join(T0_REPO, "exploratory/shadow-ambiguity/test_shadow_ambiguity.py"),
    }
    out = {}
    for mod, exp in expected.items():
        m = sys.modules.get(mod)
        p = os.path.abspath(getattr(m, "__file__", None) or exp)
        if os.path.exists(p):
            out[f"{mod}.py"] = _sha256_file(p)
    out["confluence_calibrator.py"] = _sha256_file(os.path.abspath(__file__))
    if os.path.exists(FUSION_SPEC_PATH):
        out["fusion_signs.json"] = _sha256_file(FUSION_SPEC_PATH)
    return out


def model_snapshot_sha(model_id: str):
    """HF-cache snapshot revision ACTUALLY resolved for the local weights (provenance; no
    network). M5: a bare `from_pretrained`/`mlx_load` with no explicit revision resolves
    `refs/main`, so the honest provenance is that commit - not the lexicographically-first of
    however many snapshot dirs happen to be cached. We read refs/main and confirm its snapshot
    dir exists; the raw dir listing is kept only as a fallback + a multi-snapshot tripwire."""
    repo = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
    snap_dir = os.path.join(repo, "snapshots")
    try:
        cached = sorted(os.listdir(snap_dir))
    except OSError:
        return None
    resolved = None
    ref_main = os.path.join(repo, "refs", "main")
    if os.path.exists(ref_main):
        with open(ref_main) as f:
            rev = f.read().strip()
        if rev and os.path.isdir(os.path.join(snap_dir, rev)):
            resolved = rev
    return {
        "resolved_revision": resolved,            # refs/main -> the snapshot a default load uses
        "resolved_source": "refs/main" if resolved else "unresolved",
        "cached_snapshots": cached,               # tripwire: >1 means an ambiguous local cache
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


def merge_matrices(ace: Dict[str, Any], readout: Dict[str, Any], *,
                   max_dropped: int | None = None) -> Dict[str, Any]:
    """Merge the ACE attention sub-matrix with the readout sub-matrix (R3: one source per
    family), aligning by sample_idx (robust to the readout run dropping non-finite rows),
    with a HARD label-alignment assert on the intersection. Returns the merged matrix dict
    (also what the harness persists to .npz for the pre-registered E1-E3 analyses).

    M4: the readout pass drops rows with non-finite features, so |common| can be < the planned
    n - silently certifying a "registered n=200" cell on a survivor subset that may exclude
    exactly the hard examples. `max_dropped` caps the allowed shrink: a registered run passes
    max_dropped=0 (every planned sample must score) and any drop raises rather than calibrating
    on fewer rows. Previews/smokes pass None (lenient)."""
    ia = {int(s): k for k, s in enumerate(ace["sample_idx"])}
    ib = {int(s): k for k, s in enumerate(readout["sample_idx"])}
    common = sorted(set(ia) & set(ib))
    if len(common) < 4:
        raise AssertionError(f"too few aligned samples: |common|={len(common)}")
    n_dropped = len(ace["labels"]) - len(common)
    if max_dropped is not None and n_dropped > max_dropped:
        raise AssertionError(
            f"sample-denominator shrink: {n_dropped} of {len(ace['labels'])} planned samples "
            f"dropped (readout non-finite or unaligned), exceeds max_dropped={max_dropped}. A "
            f"registered cell must calibrate on every planned sample; do NOT certify on a subset.")
    ra = [ia[s] for s in common]; rb = [ib[s] for s in common]
    ya = ace["labels"][ra]; yb = readout["labels"][rb]
    if not np.array_equal(ya, yb):
        raise AssertionError(f"label misalignment on {len(common)} shared sample_idx: "
                             f"{int((ya != yb).sum())} disagree")
    M = np.hstack([ace["score_matrix"][ra], readout["score_matrix"][rb]])
    panel = list(ace["panel"]) + list(readout["panel"])
    return {
        "score_matrix": M, "labels": ya, "panel": panel,
        "sample_idx": np.array(common, dtype=np.int64),
        "slug": ace.get("slug"), "n": len(ya), "n_aligned": len(common),
        "n_dropped_unaligned": len(ace["labels"]) - len(common),
        "n_ace_cells": len(ace["panel"]), "n_readout_cells": len(readout["panel"]),
        "ace_data_hash": ace.get("data_hash"), "readout_data_hash": readout.get("data_hash"),
    }


def calibrate_merged(mm: Dict[str, Any], *, n_bootstrap: int = 2000, seed: int = 20260610,
                     model_id: str | None = None, benchmark: str | None = None,
                     with_fusion: bool = True, with_controls: bool = True) -> Dict[str, Any]:
    """Dual-endpoint dispatcher over a merged matrix + pre-registered fusion cells (E4) +
    per-cell shuffled-label controls (A3) + drift/provenance hashes (C2)."""
    M, ya, panel = mm["score_matrix"], mm["labels"], list(mm["panel"])
    fusion_meta = None
    if with_fusion:
        M, panel, fusion_meta = append_fusion_columns(
            M, panel, mm.get("slug") or "", benchmark or "")
    full = run_selection(M, ya, panel, n_bootstrap=n_bootstrap, seed=seed)
    geom_keys = {c[2] for c in panel if c[2] not in NON_GEOMETRIC_KEYS}
    geom = run_selection(M, ya, panel, n_bootstrap=n_bootstrap, seed=seed, restrict_keys=geom_keys)
    controls = None
    if with_controls:
        controls = {
            "shuffled_label_full": shuffled_label_control(
                M, ya, panel, n_bootstrap=n_bootstrap, seed=seed),
            "shuffled_label_geometric": shuffled_label_control(
                M, ya, panel, n_bootstrap=n_bootstrap, seed=seed, restrict_keys=geom_keys),
        }
        controls["pass"] = bool(controls["shuffled_label_full"]["pass"]
                                and controls["shuffled_label_geometric"]["pass"])
    return {
        "schema_version": "confluence/0.2-unified",
        "model": model_id, "slug": mm.get("slug"), "benchmark": benchmark,
        "n": mm["n"], "n_aligned": mm["n_aligned"],
        "n_dropped_unaligned": mm["n_dropped_unaligned"], "n_cells_total": len(panel),
        "n_ace_cells": mm["n_ace_cells"], "n_readout_cells": mm["n_readout_cells"],
        "n_fusion_cells": len(panel) - mm["n_ace_cells"] - mm["n_readout_cells"],
        "ace_data_hash": mm.get("ace_data_hash"), "readout_data_hash": mm.get("readout_data_hash"),
        "primary_full_panel": full, "secondary_geometric_only": geom,
        "controls": controls, "fusion": fusion_meta,
        "provenance": {"seed": seed, "n_bootstrap": n_bootstrap,
                       "sealed_selector": "_nested_bootstrap_oob_auroc (imported, not modified)",
                       "module_hashes": module_hashes(),
                       "model_snapshot_sha": model_snapshot_sha(model_id) if model_id else None},
    }


def merge_and_calibrate(ace: Dict[str, Any], readout: Dict[str, Any], *,
                        n_bootstrap: int = 2000, seed: int = 20260610,
                        model_id: str | None = None, benchmark: str | None = None,
                        with_fusion: bool = True, with_controls: bool = True) -> Dict[str, Any]:
    """Back-compatible wrapper: merge, then calibrate (fusion + controls on by default)."""
    return calibrate_merged(merge_matrices(ace, readout), n_bootstrap=n_bootstrap, seed=seed,
                            model_id=model_id, benchmark=benchmark,
                            with_fusion=with_fusion, with_controls=with_controls)


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
