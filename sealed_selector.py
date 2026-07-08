"""sealed_selector - vendored, read-only copy of the sealed nested-OOB selector.

Why this file exists
--------------------
The registered analyses in this repository (the endpoint verdicts and the descriptive
E1/E2/E3 probes) only need the sealed SELECTION machinery - scoring, sign-locking, and
the nested out-of-bag bootstrap - applied to the published `stage_b/profiles/**.matrix.npz`
score matrices. That machinery lives in the sealed dependency repo `t0-morphology-furnace`
(private, pending the companion papers). This module vendors exactly those functions so
that `stage_b/analyze_universality.py` and `stage_b/verify_endpoints.py` run from this
repository alone, with no models and no private dependencies.

Provenance (byte-level)
-----------------------
The function bodies below are verbatim copies (only the surrounding module and import
glue differ) from the sealed dependency repo at the state that produced the registered
run:

    pri_calibrator.py
        sha256 78c4f098295fe600cc4f6f1a14cc7b496ac93d8d70dd65743c1504eb20931101
        -> _cell_label, _score_candidate, _nested_bootstrap_oob_auroc,
           ATTENTION_FAMILY, DERIVED_COMPOSITE_FAMILY, PanelCell
    scripts/analyze_adaptive_step.py
        sha256 2dffb7f03fa4876e1c4098402a9c322773d4cd246fdf66d4ec8ac71a1d79cb69
        -> auroc_signed

The first sha256 equals `provenance.module_hashes["pri_calibrator.py"]` recorded inside
every registered profile under `stage_b/profiles/`, so the vendored selector is the exact
code that produced the sealed verdict. When the sealed repo is importable (point
`$CONFLUENCE_T0_REPO` at it), `confluence_calibrator` prefers the original import and this
module is unused.

Scope guard
-----------
Only selection/analysis is vendored. Extraction (model forwards: ACE attention capture,
readout traces) is NOT - regenerating a matrix from scratch still requires the sealed
repo. Accessing any extraction-side symbol through this module raises AttributeError
with that explanation instead of an anonymous one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

PanelCell = Tuple[int, str, str]

ATTENTION_FAMILY = "Attention"
DERIVED_COMPOSITE_FAMILY = "Composite"


def auroc_signed(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, int]:
    """Direction-agnostic AUROC: max(auc, 1-auc). Returns (auc, sign in {-1, +1}).
    NaN scores are dropped. NaN auc if fewer than 4 finite samples or fewer than
    2 distinct labels."""
    finite = np.isfinite(scores)
    if finite.sum() < 4 or len(np.unique(labels[finite])) < 2:
        return float("nan"), 0
    auc = roc_auc_score(labels[finite], scores[finite])
    return float(max(auc, 1 - auc)), 1 if auc >= 0.5 else -1


def _cell_label(cell: PanelCell) -> str:
    """Human-readable cell name for reports + provenance."""
    step, fam, label = cell
    if fam == "scalar":
        return f"{label} @ step {step}"
    if fam == DERIVED_COMPOSITE_FAMILY:
        return f"composite[{label}] @ step {step}"
    if fam == ATTENTION_FAMILY:
        return f"attention[{label}] @ step {step}"
    return f"{fam} {label} @ step {step}"


def _score_candidate(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, int, int]:
    """Return (auroc, sign, n_evaluated). Sign is locked from THIS data —
    that's the whole point of calibration. Drops NaN scores; if fewer than
    4 samples with both labels survive, returns (nan, 0, n_finite)."""
    finite = np.isfinite(scores)
    n_eval = int(finite.sum())
    s = scores[finite]
    y = labels[finite]
    if n_eval < 4 or len(np.unique(y)) < 2:
        return float("nan"), 0, n_eval
    auc, sign = auroc_signed(y, s)
    return float(auc), int(sign), n_eval


def _nested_bootstrap_oob_auroc(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    panel: List[PanelCell],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Nested bootstrap: at each round, resample the calibration set with
    replacement to form an in-bag set; re-run the WHOLE cell selection
    (best cell + sign-lock) on the in-bag samples; then evaluate the
    selected (cell, sign) on the out-of-bag samples. The OOB AUROC
    distribution is the honest deployment estimate — it accounts for the
    selection bias that contaminates in-sample stats.

    Also tracks how often each cell is selected across rounds; if one cell
    wins ≪ 100% of resamples, the selection is noisy at this n and the
    profile gets a `winner_unstable` warning.

    Returns a dict with the OOB summary stats. If no round produced a
    valid OOB AUROC (degenerate small-n case), returns a dict full of
    NaNs.
    """
    from sklearn.metrics import roc_auc_score

    n, n_cells = score_matrix.shape
    rng = np.random.RandomState(seed + 1)  # +1 so it doesn't clash with _bootstrap_auroc's stream
    oob_aurocs: List[float] = []
    winner_counts: Dict[int, int] = {j: 0 for j in range(n_cells)}

    for _ in range(n_bootstrap):
        in_bag = rng.randint(0, n, size=n)
        in_bag_set = set(in_bag.tolist())
        oob = np.array([i for i in range(n) if i not in in_bag_set], dtype=np.int64)
        if len(oob) < 4 or len(np.unique(labels[oob])) < 2:
            continue

        # Re-run cell selection inside this resample.
        best_j = -1
        best_distance = -1.0
        best_sign = 0
        for j in range(n_cells):
            s_in = score_matrix[in_bag, j]
            y_in = labels[in_bag]
            auc, sign, _ = _score_candidate(s_in, y_in)
            if np.isfinite(auc):
                d = abs(auc - 0.5)
                if d > best_distance:
                    best_distance = d
                    best_j = j
                    best_sign = sign
        if best_j < 0:
            continue
        winner_counts[best_j] += 1

        # Evaluate the in-bag-selected cell on OOB with the in-bag-locked sign.
        s_oob = score_matrix[oob, best_j] * best_sign
        y_oob = labels[oob]
        finite = np.isfinite(s_oob)
        if finite.sum() < 4 or len(np.unique(y_oob[finite])) < 2:
            continue
        oob_aurocs.append(float(roc_auc_score(y_oob[finite], s_oob[finite])))

    total_winners = sum(winner_counts.values())
    if total_winners > 0:
        max_count = max(winner_counts.values())
        winner_stability = max_count / total_winners
    else:
        winner_stability = float("nan")
    winner_counts_labeled = {
        _cell_label(panel[j]): c for j, c in winner_counts.items() if c > 0
    }
    if not oob_aurocs:
        return {
            "oob_auroc_median": float("nan"),
            "oob_auroc_ci_lo": float("nan"),
            "oob_auroc_ci_hi": float("nan"),
            "oob_n_bootstrap_used": 0,
            "winner_stability": float(winner_stability) if np.isfinite(winner_stability) else float("nan"),
            "winner_counts": winner_counts_labeled,
        }
    arr = np.array(oob_aurocs)
    return {
        "oob_auroc_median": float(np.median(arr)),
        "oob_auroc_ci_lo": float(np.percentile(arr, 2.5)),
        "oob_auroc_ci_hi": float(np.percentile(arr, 97.5)),
        "oob_n_bootstrap_used": int(len(oob_aurocs)),
        "winner_stability": float(winner_stability),
        "winner_counts": winner_counts_labeled,
    }


def __getattr__(name: str):
    raise AttributeError(
        f"sealed_selector has no '{name}': only the selection/analysis machinery is vendored. "
        "Extraction (model forwards) requires the sealed dependency repo - point "
        "$CONFLUENCE_T0_REPO at t0-morphology-furnace and re-run."
    )
