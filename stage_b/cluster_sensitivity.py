#!/usr/bin/env python3
"""Descriptive stem-cluster sensitivity for the sealed TriviaQA result.

This report is additive and non-gating.  It does not modify the sealed selector, the
registered endpoint, or the published 18/20 verdict.  It reuses the sealed candidate scoring
and in-bag-selection/OOB-evaluation order through confluence_calibrator's registered cluster
analogue, changing only the exchangeable unit from row to question stem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import confluence_calibrator as CC


DEFAULT_PROFILES = ROOT / "stage_b/profiles/triviaqa_paired"
DEFAULT_DATA = ROOT / "stage_b/data/triviaqa_paired_seed20260612_n200.jsonl"
DEFAULT_JSON = ROOT / "stage_b/cluster_sensitivity.json"
DEFAULT_MARKDOWN = ROOT / "stage_b/cluster_sensitivity.md"
SEALED_SEED = 20260612
SEALED_NBOOT = 2000


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stem_id(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    value = row.get("stem_id", meta.get("stem_id", meta.get("question_id")))
    if value is None:
        raise ValueError("TriviaQA row has no stem_id or meta.question_id")
    return str(value)


def endpoint_view(result: dict[str, Any]) -> dict[str, Any]:
    lo = result.get("oob_auroc_ci_lo")
    return {
        "winner": result.get("winner"),
        "oob_auroc_median": result.get("oob_auroc_median"),
        "oob_auroc_ci_lo": lo,
        "oob_auroc_ci_hi": result.get("oob_auroc_ci_hi"),
        "oob_n_bootstrap_used": result.get("oob_n_bootstrap_used"),
        "winner_stability": result.get("winner_stability"),
        "descriptive_deployable_ci_lo_gt_0_50": bool(
            lo is not None and np.isfinite(lo) and lo > 0.50),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sealed TriviaQA stem-cluster sensitivity",
        "",
        "> Descriptive and non-gating. This does not alter the sealed 18/20 verdict.",
        "",
        "| Model | Full CI low | Full > .50 | Geometric CI low | Geometric > .50 |",
        "|---|---:|:---:|---:|:---:|",
    ]
    for row in report["cells"]:
        full, geom = row["primary_full_panel"], row["secondary_geometric_only"]
        lines.append(
            f"| {row['slug']} | {full['oob_auroc_ci_lo']:.4f} | "
            f"{'yes' if full['descriptive_deployable_ci_lo_gt_0_50'] else 'no'} | "
            f"{geom['oob_auroc_ci_lo']:.4f} | "
            f"{'yes' if geom['descriptive_deployable_ci_lo_gt_0_50'] else 'no'} |")
    lines.extend([
        "",
        ("Exchangeable unit: TriviaQA question stem. Candidate selection and OOB scoring "
         "otherwise follow the sealed procedure."),
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seed", type=int, default=SEALED_SEED)
    parser.add_argument("--nboot", type=int, default=SEALED_NBOOT)
    args = parser.parse_args()

    rows = load_rows(args.data)
    source_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    source_stems = np.asarray([stem_id(row) for row in rows], dtype=np.str_)
    if len(set(source_stems.tolist())) * 2 != len(source_stems):
        raise ValueError("sealed TriviaQA source is not exactly two rows per stem")

    cells = []
    for matrix_path in sorted(args.profiles_dir.glob("*.matrix.npz")):
        slug = matrix_path.name.removesuffix(".matrix.npz")
        profile_path = matrix_path.with_name(f"{slug}.profile.json")
        profile = json.loads(profile_path.read_text())
        provenance = profile.get("provenance") or {}
        if provenance.get("seed") != args.seed or provenance.get("n_bootstrap") != args.nboot:
            raise ValueError(f"{slug}: sealed seed/bootstrap provenance mismatch")

        matrix = np.load(matrix_path, allow_pickle=False)
        sample_idx = matrix["sample_idx"].astype(np.int64)
        labels = matrix["labels"].astype(np.int64)
        if not np.array_equal(labels, source_labels[sample_idx]):
            raise ValueError(f"{slug}: matrix labels do not align to published source rows")
        stems = source_stems[sample_idx]
        panel = [tuple(cell) for cell in json.loads(str(matrix["panel"]))]
        scores, panel, fusion = CC.append_fusion_columns(
            matrix["score_matrix"], panel, slug, "triviaqa_paired")
        geometric_keys = {cell[2] for cell in panel
                          if cell[2] not in CC.NON_GEOMETRIC_KEYS}
        full = CC.run_selection(
            scores, labels, panel, n_bootstrap=args.nboot, seed=args.seed,
            stem_ids=stems, bootstrap_unit="cluster")
        geometric = CC.run_selection(
            scores, labels, panel, n_bootstrap=args.nboot, seed=args.seed,
            restrict_keys=geometric_keys, stem_ids=stems, bootstrap_unit="cluster")
        cells.append({
            "slug": slug,
            "n_rows": int(len(labels)),
            "n_stems": int(len(set(stems.tolist()))),
            "fusion": fusion,
            "primary_full_panel": endpoint_view(full),
            "secondary_geometric_only": endpoint_view(geometric),
        })

    if len(cells) != 10:
        raise ValueError(f"expected 10 published TriviaQA cells, found {len(cells)}")
    report = {
        "schema_version": "sealed-cluster-sensitivity/1.0",
        "status": "DESCRIPTIVE-NON-GATING",
        "alters_sealed_verdict": False,
        "sealed_verdict_preserved": "18/20",
        "task": "triviaqa_paired",
        "bootstrap_unit": "question_stem",
        "seed": args.seed,
        "n_bootstrap": args.nboot,
        "method": ("sealed candidate scoring and in-bag selection/OOB evaluation order; "
                   "exchangeable unit changed from row to question stem"),
        "cells": cells,
    }
    args.out_json.write_text(json.dumps(report, indent=1) + "\n")
    args.out_markdown.write_text(render_markdown(report))
    print(f"report -> {args.out_json}")
    print(f"table  -> {args.out_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
