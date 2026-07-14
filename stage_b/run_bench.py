#!/usr/bin/env python3
"""BENCH v1.2 Phase-3/4 launcher and Phase-1 extension-manifest writer.

Authored by Codex under a write/audit-only constraint. Execution belongs to Claude Code
or MK after the binding phase gates in PRE_REGISTRATION_BENCH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import string
import subprocess
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC
import check_fresh_data as GATE


SPEC_VERSION = "bench/1.3"
SEED = 20260711
NBOOT = 2000
FROZEN_FUSION_SHA256 = "92b5468bd241b517dd2d5cf70ad28556157424deb54b5f85f88af0305ff35372"
PHASE0_CALIBRATOR_SHA256 = "f55279162eb15a3806a0698e1540e898a13f70dbdc9e99820a969bb2bf563bbb"
HALU_COMMIT = "b7253db3cdaa0ab2c382f92b26b390109174f77e"
ANLI_FINGERPRINT = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
TRIVIA_FINGERPRINT = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
FROZEN_ARROW_HASHES = {
    "anli-train_r1.arrow": "b32df9e1ee446fa9d34c6996f788dbce7fbbe9ec682d0672cb340837904ee40a",
    "anli-dev_r2.arrow": "6ff4c3bac8b0ae917cf89dd73cf9966107d5888232d8e423ecde8da8555486fd",
    "anli-test_r2.arrow": "d63398b51f5c29f92b251b1f5b54c9a1a5c9772a1b2a7ed96a047cee0221e655",
    "trivia_qa-validation.arrow": "8e95a5f9ce34a037cc3dd0d2e544961a20470cb6c415f6ab48a1e115ed5a7a90",
}

COHORT = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
]
SLUGS = [model.split("/")[-1] for model in COHORT]
TASKS = {
    "halueval_qa": {"family": "A", "confirmatory": True, "grouped": True,
                    "expected_n": 1000, "canonical_fusion": "halueval_qa"},
    "anli_r1_rep": {"family": "B", "confirmatory": True, "grouped": False,
                    "expected_n": 1000, "canonical_fusion": "anli_r1"},
    "triviaqa_paired_rep": {"family": "B", "confirmatory": True, "grouped": True,
                            "expected_n": 1000, "canonical_fusion": "triviaqa_paired"},
    "anli_r2": {"family": "C", "confirmatory": False, "grouped": False,
                "expected_n": 1000, "canonical_fusion": "anli_r2"},
    "halueval_dialogue": {"family": "C", "confirmatory": False, "grouped": True,
                          "expected_n": None, "canonical_fusion": "halueval_dialogue"},
    "halueval_summarization": {"family": "C", "confirmatory": False, "grouped": True,
                               "expected_n": None,
                               "canonical_fusion": "halueval_summarization"},
}

ROOT = Path(__file__).resolve().parents[1]
STAGE_B = Path(__file__).resolve().parent
T0_REPO = Path(CC.T0_REPO)
SEALED_REFERENCES = [
    T0_REPO / "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl",
    T0_REPO / "experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl",
    STAGE_B / "data/anli_R1_seed20260612_n200.jsonl",
    STAGE_B / "data/triviaqa_paired_seed20260612_n200.jsonl",
]
MANIFEST_FILES = [
    STAGE_B / "run_bench.py", STAGE_B / "generate_bench_data.py",
    STAGE_B / "check_fresh_data.py", STAGE_B / "analyze_universality.py",
    ROOT / "confluence_calibrator.py", STAGE_B / "fusion_signs.json",
    STAGE_B / "PRE_REGISTRATION_BENCH.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_frozen_arrow(recorded_path: Path, expected_sha256: str) -> Path:
    """Resolve a frozen HF Arrow artifact by bytes, not by the builder machine's path (A3)."""
    roots = []
    configured = os.environ.get("CONFLUENCE_HF_CACHE")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(Path.home() / ".cache/huggingface/datasets")

    candidates: List[Path] = []
    for root in roots:
        if root.is_file() and root.name == recorded_path.name:
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(sorted(root.rglob(recorded_path.name)))
    candidates.append(recorded_path.expanduser())

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and sha256_file(candidate) == expected_sha256:
            return candidate.resolve()
    raise FileNotFoundError(
        f"frozen Arrow artifact {recorded_path.name} unavailable; set CONFLUENCE_HF_CACHE "
        f"to a Hugging Face cache containing sha256={expected_sha256}")


def resolve_exclusion_reference(recorded_path: Path,
                                expected_by_name: Dict[str, Tuple[Path, str]]) -> Tuple[Path, str]:
    """Resolve a provenance path to the vendored/current file and verify its frozen bytes."""
    expected = expected_by_name.get(recorded_path.name)
    if expected is None:
        raise ValueError(f"unregistered exclusion reference: {recorded_path.name}")
    current_path, expected_sha256 = expected
    candidates = [recorded_path.expanduser(), current_path]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_sha256:
            return candidate.resolve(), expected_sha256
    raise FileNotFoundError(
        f"exclusion reference {recorded_path.name} unavailable/mismatched; "
        f"expected sha256={expected_sha256} (vendored path: {current_path})")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def stem_vector_sha256(stem_ids: Sequence[Any]) -> str:
    return sha256_json([str(stem) for stem in stem_ids])


def runtime_versions() -> Dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for dist, key in (("mlx-lm", "mlx_lm"), ("mlx", "mlx"), ("numpy", "numpy")):
        try:
            versions[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_task_args(values: Sequence[str]) -> Dict[str, Path]:
    tasks = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--task must be KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        if key not in TASKS:
            raise ValueError(f"unregistered BENCH task: {key}")
        if key in tasks:
            raise ValueError(f"duplicate task: {key}")
        tasks[key] = Path(path)
    # TASKS literal order is the binding Phase-4 order: confirmatory A/B before C.
    return {key: tasks[key] for key in TASKS if key in tasks}


def emit_extension_manifest(out_dir: Path) -> Path:
    for path in MANIFEST_FILES:
        if not path.exists():
            raise FileNotFoundError(f"manifest file missing: {path}")
    if sha256_file(STAGE_B / "fusion_signs.json") != FROZEN_FUSION_SHA256:
        raise RuntimeError("frozen fusion_signs.json hash drift")
    baseline = subprocess.run(
        ["git", "show", "HEAD:confluence_calibrator.py"],
        cwd=ROOT, check=True, capture_output=True).stdout
    if hashlib.sha256(baseline).hexdigest() != PHASE0_CALIBRATOR_SHA256:
        raise RuntimeError(
            "HEAD confluence_calibrator.py is not the frozen Phase-0 baseline; "
            "cannot construct an honest pre/post diff")
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--unified=0", "--", "confluence_calibrator.py"],
        cwd=ROOT, check=True, text=True, capture_output=True).stdout
    numstat = subprocess.run(
        ["git", "diff", "HEAD", "--numstat", "--", "confluence_calibrator.py"],
        cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    if not diff or not numstat:
        raise RuntimeError("Phase-1 calibrator diff is empty; extension baseline is not ready")
    runtime = runtime_versions()
    manifest = {
        "schema_version": "bench-extension-manifest/1.2",
        "spec_version": SPEC_VERSION, "seed": SEED,
        "phase0_calibrator_sha256": PHASE0_CALIBRATOR_SHA256,
        "extension_calibrator_sha256": sha256_file(ROOT / "confluence_calibrator.py"),
        "fusion_signs_sha256": FROZEN_FUSION_SHA256,
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in MANIFEST_FILES},
        "runtime_versions": runtime,
        "runtime_limitation": "complete seal-time runtime tuple was not committed",
        "calibrator_diff": {"numstat": numstat, "sha256": hashlib.sha256(diff.encode()).hexdigest(),
                            "text": diff},
        "executor_attestations": {
            "diff_scope_reviewed": False,
            "ten_model_frozen_row_parity_completed": False,
            "ten_model_frozen_row_parity_all_passed": None,
            "note": "Executor must set only after performing the Phase-1 checks; not run by Codex.",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "EXTENSION_MANIFEST.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def attest_phase1(out_dir: Path, *, diff_reviewed: bool) -> Path:
    manifest_path = out_dir / "EXTENSION_MANIFEST.json"
    parity_path = out_dir / "PHASE1_PARITY.json"
    manifest = json.load(manifest_path.open())
    parity = json.load(parity_path.open())
    if not diff_reviewed:
        raise RuntimeError("--diff-reviewed is required after human/static diff inspection")
    results = parity.get("results") or {}
    if set(results) != set(SLUGS) or any("pass" not in result for result in results.values()):
        raise RuntimeError("ten-model frozen-row parity sentinel is incomplete")
    for rel, expected in (manifest.get("files") or {}).items():
        path = ROOT / rel
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"extension manifest drift before attestation: {rel}")
    if manifest.get("runtime_versions") != runtime_versions():
        raise RuntimeError("runtime tuple drift before Phase-1 attestation")
    failed_slugs = sorted(slug for slug, result in results.items() if not result["pass"])
    manifest["executor_attestations"] = {
        "diff_scope_reviewed": True,
        "ten_model_frozen_row_parity_completed": True,
        "ten_model_frozen_row_parity_all_passed": not failed_slugs,
        "feature_version_delta_slugs": failed_slugs,
        "parity_report": str(parity_path.resolve()),
        "parity_report_sha256": sha256_file(parity_path),
        "attested_by": "executor (Claude Code / MK), not Codex",
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path


def validate_static_hashes(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.load(manifest_path.open())
    if manifest.get("spec_version") != SPEC_VERSION:
        raise RuntimeError("extension manifest spec mismatch")
    attest = manifest.get("executor_attestations") or {}
    if (not attest.get("diff_scope_reviewed")
            or not attest.get("ten_model_frozen_row_parity_completed")):
        raise RuntimeError(
            "Phase-1 executor attestations incomplete: diff review and ten-model parity "
            "sentinel must be completed before runs")
    parity_path = Path(attest.get("parity_report") or "")
    if (not parity_path.is_file()
            or sha256_file(parity_path) != attest.get("parity_report_sha256")):
        raise RuntimeError("Phase-1 parity report missing or hash-drifted")
    parity = json.load(parity_path.open())
    results = parity.get("results") or {}
    failed_slugs = sorted(slug for slug, result in results.items() if not result.get("pass"))
    if (set(results) != set(SLUGS)
            or attest.get("feature_version_delta_slugs") != failed_slugs
            or attest.get("ten_model_frozen_row_parity_all_passed") != (not failed_slugs)):
        raise RuntimeError("Phase-1 parity disposition is incomplete or inconsistent")
    for rel, expected in (manifest.get("files") or {}).items():
        path = ROOT / rel
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"extension manifest drift: {rel}")
    if manifest.get("runtime_versions") != runtime_versions():
        raise RuntimeError("BENCH runtime tuple drift from extension manifest")
    current = CC.module_hashes()
    sealed = json.load((STAGE_B / "profiles/SUMMARY.json").open())["module_hashes"]
    t0_keys = set(sealed) - {"confluence_calibrator.py", "fusion_signs.json"}
    mismatches = {key: (sealed.get(key), current.get(key)) for key in sorted(t0_keys)
                  if sealed.get(key) != current.get(key)}
    if mismatches:
        raise RuntimeError(f"T0-subset hash drift: {mismatches}")
    if current.get("fusion_signs.json") != FROZEN_FUSION_SHA256:
        raise RuntimeError("fusion_signs.json drift")
    return manifest


def model_snapshot_mismatches(models: Sequence[str]) -> Dict[str, Any]:
    mismatches = {}
    for model in models:
        slug = model.split("/")[-1]
        sealed_path = STAGE_B / "profiles/anli_r1" / f"{slug}.profile.json"
        sealed = json.load(sealed_path.open())
        expected = ((sealed.get("provenance") or {}).get("model_snapshot_sha") or {}).get(
            "resolved_revision")
        current = (CC.model_snapshot_sha(model) or {}).get("resolved_revision")
        if expected != current:
            mismatches[slug] = {"expected": expected, "current": current}
    return mismatches


def run_parity_sentinel(out_dir: Path) -> Dict[str, Any]:
    """Executor-only Phase-1 sentinel; authored here but never run by Codex."""
    parity_seed = 20260612
    source = STAGE_B / "data/anli_R1_seed20260612_n200.jsonl"
    first = load_jsonl(source)[0]
    first["stem_id"] = "parity:sample_idx=0"
    parity_data = out_dir / "_phase1/anli_20260612_sample0.jsonl"
    write_jsonl(parity_data, [first])
    snapshot_failures = model_snapshot_mismatches(COHORT)
    results = {}
    for model in COHORT:
        slug = model.split("/")[-1]
        if slug in snapshot_failures:
            results[slug] = {"pass": False, "reason": "model resolved_revision drift",
                             "snapshot": snapshot_failures[slug]}
            continue
        ace = CC.collect_ace_matrix(
            model, str(parity_data), seed=parity_seed, max_new_tokens=1,
            require_stem_ids=True)
        readout = CC.collect_readout_matrix_fresh(
            model, "anli_r1", str(parity_data), seed=parity_seed,
            require_stem_ids=True)
        if (not np.array_equal(ace["sample_idx"], readout["sample_idx"])
                or not np.array_equal(ace["labels"], readout["labels"])
                or not np.array_equal(ace["stem_ids"], readout["stem_ids"])):
            results[slug] = {"pass": False, "reason": "parity extraction arms misaligned"}
            continue
        actual = np.hstack([ace["score_matrix"], readout["score_matrix"]])[0]
        reference_path = STAGE_B / "profiles/anli_r1" / f"{slug}.matrix.npz"
        reference = np.load(reference_path, allow_pickle=False)
        matches = np.flatnonzero(reference["sample_idx"] == 0)
        if len(matches) != 1:
            results[slug] = {"pass": False, "reason": "reference sample_idx=0 missing/duplicate"}
            continue
        expected = np.asarray(reference["score_matrix"][matches[0]], dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        finite = np.isfinite(expected)
        passed = bool(expected.shape == actual.shape
                      and expected[finite].tobytes() == actual[finite].tobytes())
        results[slug] = {
            "pass": passed, "n_reference_finite": int(finite.sum()),
            "n_exact": int(np.sum(expected[finite] == actual[finite])) if passed else int(
                np.sum(expected[finite] == actual[finite]) if expected.shape == actual.shape else 0),
            "reference_matrix": str(reference_path),
        }
    report = {"schema_version": "bench-parity/1.2", "seed": parity_seed, "sample_idx": 0,
              "source": str(source), "results": results,
              "pass": bool(len(results) == 10 and all(row["pass"] for row in results.values()))}
    report_path = out_dir / "PHASE1_PARITY.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=1)
    return report


def run_phase1_unit_checks() -> Dict[str, Any]:
    """Executor-only synthetic checks for the new statistical/plumbing contracts."""
    labels = np.asarray([0, 1] * 20, dtype=np.int64)
    scores = np.column_stack([
        np.linspace(-1.0, 1.0, len(labels)),
        np.asarray([0.1, 0.9] * 20, dtype=np.float64),
    ])
    panel = [(0, "Readout", "synthetic_a"), (0, "Readout", "synthetic_b")]
    unique_stems = np.asarray([f"row:{i}" for i in range(len(labels))])
    row = CC.run_selection(scores, labels, panel, n_bootstrap=100, seed=SEED)
    unique_cluster = CC.run_selection(
        scores, labels, panel, n_bootstrap=100, seed=SEED,
        stem_ids=unique_stems, bootstrap_unit="cluster")
    comparable_keys = set(row) - {"bootstrap_unit"}
    if any(row[key] != unique_cluster[key] for key in comparable_keys):
        raise AssertionError("unique-stem cluster path did not reduce to sealed row path")
    paired_stems = np.asarray([f"pair:{i // 2}" for i in range(len(labels))])
    cluster = CC.run_selection(
        scores, labels, panel, n_bootstrap=100, seed=SEED,
        stem_ids=paired_stems, bootstrap_unit="cluster")
    control = CC.shuffled_label_control(
        scores, labels, panel, n_bootstrap=100, seed=SEED,
        stem_ids=paired_stems, bootstrap_unit="cluster")
    if control["permutation"] != "within_stem_label_swap":
        raise AssertionError("paired control did not preserve the stem design")
    dummy = np.column_stack([
        scores[:, 0], scores[:, 1], scores[:, 0] * 0.5,
        scores[:, 1] * 0.5, scores[:, 0] * 0.2])
    fusion_panel = [(0, "Attention", "last_minus_1_js_no_bos"),
                    (0, "Readout", "null_ratio_post_rank1"),
                    (0, "Readout", "fisher_eff_rank"),
                    (0, "Readout", "surprise"), (0, "Readout", "p_max")]
    _, _, fusion = CC.append_fusion_columns(
        dummy, fusion_panel, "Qwen2.5-7B-Instruct-4bit", "anli_r1_rep")
    if fusion["canonical_benchmark"] != "anli_r1":
        raise AssertionError("replication fusion alias did not resolve")
    if normalize_commitment(" \nYES!!!") != "yes" or normalize_commitment(" No. ") != "no":
        raise AssertionError("commitment normalization contract failed")
    if (not is_canonical_commitment("y") or not is_canonical_commitment("no")
            or is_canonical_commitment("") or is_canonical_commitment("to")
            or is_canonical_commitment("the")):
        raise AssertionError("bench/1.3 commitment prefix contract failed")
    return {"pass": True, "row_unique_cluster_equal": True,
            "cluster_n_groups": cluster["n_groups"],
            "control_permutation": control["permutation"],
            "fusion_key": fusion["key"]}


def _local_tokenizer(model_id: str):
    from mlx_lm.utils import hf_repo_to_path, load_tokenizer
    return load_tokenizer(hf_repo_to_path(model_id))


def normalize_commitment(text: str) -> str:
    value = str(text).lstrip().casefold().rstrip()
    while value and (value[-1] in string.punctuation
                     or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1].rstrip()
    return value


def is_canonical_commitment(normalized: str) -> bool:
    # bench/1.3 Amendment A1: subword tokenizers (e.g. Mistral "YES" -> "Y"+"ES")
    # emit the answer commit as a strict prefix of the canonical form. A non-empty
    # normalized prefix of "yes" or "no" is a canonical commit; the prefix sets of
    # the two forms do not overlap ("y"/"ye" vs "n"), so no ambiguity is introduced.
    # Whitespace-only ("") and non-prefix tokens (" To" -> "to") still fail.
    return bool(normalized) and ("yes".startswith(normalized) or "no".startswith(normalized))


def commitment_audit(model_id: str, token_ids: Sequence[int]) -> Dict[str, Any]:
    tokenizer = _local_tokenizer(model_id)
    rows = []
    for token_id in token_ids:
        decoded = "" if int(token_id) < 0 else tokenizer.decode([int(token_id)])
        normalized = normalize_commitment(decoded)
        rows.append({"token_id": int(token_id), "decoded": decoded,
                     "normalized": normalized,
                     "canonical": is_canonical_commitment(normalized)})
    n_canonical = sum(row["canonical"] for row in rows)
    return {"n": len(rows), "n_canonical": n_canonical,
            "n_noncanonical": len(rows) - n_canonical, "rows": rows}


def reference_panel() -> List[Tuple[int, str, str]]:
    path = STAGE_B / "profiles/anli_r1/Qwen2.5-7B-Instruct-4bit.matrix.npz"
    data = np.load(path, allow_pickle=False)
    return [tuple(cell) for cell in json.loads(str(data["panel"]))]


def panel_sha256(panel: Sequence[Tuple[int, str, str]]) -> str:
    return sha256_json([list(cell) for cell in panel])


def anli_r2_split_descriptives(profile: Dict[str, Any], mm: Dict[str, Any],
                               data_path: Path) -> Dict[str, Any]:
    """Apply each pooled selected cell/sign to dev_r2 and test_r2 separately."""
    from sklearn.metrics import roc_auc_score

    M, panel, _ = CC.append_fusion_columns(
        mm["score_matrix"], list(mm["panel"]), mm.get("slug") or "", "anli_r2",
        canonical_benchmark="anli_r2")
    labels = [CC.SEAL._cell_label(cell) for cell in panel]
    source_rows = load_jsonl(data_path)
    splits = np.asarray([
        (source_rows[int(idx)].get("meta") or {}).get("source_split")
        for idx in mm["sample_idx"]
    ])
    out = {}
    for unit, endpoints in profile["endpoints_by_unit"].items():
        out[unit] = {}
        for endpoint_name, endpoint in endpoints.items():
            winner = endpoint.get("winner")
            marginal = (endpoint.get("full_sample_marginals") or {}).get(winner) or {}
            sign = int(marginal.get("sign") or 0)
            split_aurocs = {}
            for split in ("dev_r2", "test_r2"):
                mask = splits == split
                if winner not in labels or sign == 0 or mask.sum() < 4:
                    split_aurocs[split] = None
                    continue
                scores = M[mask, labels.index(winner)] * sign
                y = np.asarray(mm["labels"])[mask]
                finite = np.isfinite(scores)
                split_aurocs[split] = (
                    float(roc_auc_score(y[finite], scores[finite]))
                    if finite.sum() >= 4 and len(np.unique(y[finite])) == 2 else None)
            out[unit][endpoint_name] = {
                "pooled_winner": winner, "pooled_fitted_sign": sign,
                "split_aurocs": split_aurocs,
            }
    return {
        "estimator": "pooled selected cell and pooled full-sample sign, fixed within split",
        "by_bootstrap_unit_and_endpoint": out,
    }


def run_cell(model: str, task: str, data_path: Path, out_dir: Path, *,
             nboot: int, strict: bool, manifest_sha: str,
             feature_version_deltas: set[str] | None = None) -> Dict[str, Any]:
    slug = model.split("/")[-1]
    ace = CC.collect_ace_matrix(
        model, str(data_path), seed=SEED, max_new_tokens=1,
        require_stem_ids=True)
    readout = CC.collect_readout_matrix_fresh(
        model, task, str(data_path), seed=SEED, require_stem_ids=True)
    mm = CC.merge_matrices(ace, readout, max_dropped=0 if strict else None)
    if mm.get("stem_ids") is None:
        raise AssertionError("BENCH merge lost stem_ids")
    if mm.get("gen_token_ids") is None:
        raise AssertionError("BENCH merge lost committed token ids")
    if list(mm["panel"]) != reference_panel():
        expected, got = set(reference_panel()), set(mm["panel"])
        raise AssertionError(
            f"panel mismatch MISSING={sorted(expected-got)} EXTRA={sorted(got-expected)}")

    audit = commitment_audit(model, mm["gen_token_ids"])
    profile = CC.calibrate_merged(
        mm, n_bootstrap=nboot, seed=SEED, model_id=model, benchmark=task,
        bootstrap_units=("row", "cluster"),
        canonical_fusion_benchmark=TASKS[task]["canonical_fusion"])
    profile["spec_version"] = SPEC_VERSION
    profile["data_path"] = str(data_path)
    profile["data_file_sha256"] = sha256_file(data_path)
    profile["commitment_audit"] = audit
    profile["terminal_status"] = (
        "OK" if audit["n_noncanonical"] == 0 else "COMMITMENT-FAIL")
    unit_status = {}
    for unit, controls in profile["controls_by_unit"].items():
        unit_status[unit] = (
            "OK" if controls["shuffled_label_geometric"]["pass"]
            else f"CONTROL-FAIL[{unit}]")
    profile["endpoint_status_by_unit"] = unit_status
    if strict and task == "anli_r2":
        profile["anli_r2_split_descriptives"] = anli_r2_split_descriptives(
            profile, mm, data_path)

    stems = np.asarray(mm["stem_ids"]).astype(str)
    psha = panel_sha256(mm["panel"])
    ssha = stem_vector_sha256(stems.tolist())
    profile["provenance"].update({
        "spec_version": SPEC_VERSION, "extension_manifest_sha256": manifest_sha,
        "panel_sha256": psha, "ordered_stem_id_sha256": ssha,
        "canonical_fusion_benchmark": TASKS[task]["canonical_fusion"],
        "control_seeds": [SEED + 90210 + k for k in range(3)],
        "controls_by_unit_sha256": sha256_json(profile["controls_by_unit"]),
        "endpoint_status_by_unit": unit_status,
        "commitment_status": profile["terminal_status"],
        "commitment_tally": {
            "n": audit["n"], "n_canonical": audit["n_canonical"],
            "n_noncanonical": audit["n_noncanonical"],
        },
        "commitment_audit_sha256": sha256_json(audit),
        "feature_extraction_comparability": (
            "feature-version-delta: Phase-1 byte-parity sentinel mismatch"
            if slug in (feature_version_deltas or set())
            else "Phase-1 frozen-row byte parity passed"),
    })
    if "anli_r2_split_descriptives" in profile:
        profile["provenance"]["anli_r2_split_descriptives_sha256"] = sha256_json(
            profile["anli_r2_split_descriptives"])
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    npz_path = task_dir / f"{slug}.matrix.npz"
    meta = {"model": model, "benchmark": task, "data_path": str(data_path),
            "seed": SEED, "spec_version": SPEC_VERSION, "panel_sha256": psha,
            "ordered_stem_id_sha256": ssha, "extension_manifest_sha256": manifest_sha,
            "canonical_fusion_benchmark": TASKS[task]["canonical_fusion"]}
    np.savez_compressed(
        npz_path, score_matrix=mm["score_matrix"], labels=mm["labels"],
        sample_idx=mm["sample_idx"], stem_ids=stems,
        gen_token_ids=np.asarray(mm["gen_token_ids"], dtype=np.int64),
        panel=json.dumps(mm["panel"]), meta=json.dumps(meta))
    with (task_dir / f"{slug}.profile.json").open("w") as f:
        json.dump(profile, f, indent=1)
    return profile


def select_smoke_rows(rows: Sequence[Dict[str, Any]], grouped: bool) -> List[Dict[str, Any]]:
    def length(row):
        return int((row.get("meta") or {}).get("max_wrapped_tokens", 0))
    if grouped:
        by_stem = defaultdict(list)
        for row in rows:
            by_stem[str(row["stem_id"])].append(row)
        ordered = sorted(by_stem, key=lambda stem: max(length(row) for row in by_stem[stem]))
        if len(ordered) < 8:
            raise ValueError("grouped smoke requires at least 8 stems")
        chosen = ordered[:4] + [stem for stem in ordered[-4:] if stem not in ordered[:4]]
        if len(chosen) != 8:
            raise ValueError("short/long smoke strata overlap")
        selected = [row for stem in chosen for row in by_stem[stem]]
    else:
        selected = []
        for label in (0, 1):
            candidates = sorted((row for row in rows if int(row["label"]) == label), key=length)
            if len(candidates) < 8:
                raise ValueError(f"ungrouped smoke label {label} has fewer than 8 rows")
            chosen = candidates[:4] + [row for row in candidates[-4:] if row not in candidates[:4]]
            if len(chosen) != 8:
                raise ValueError("short/long smoke strata overlap")
            selected.extend(chosen)
    if len(selected) != 16:
        raise AssertionError(f"stratified smoke must contain 16 rows, got {len(selected)}")
    return selected


def _control_structure_valid(profile: Dict[str, Any]) -> bool:
    expected_seeds = [SEED + 90210 + k for k in range(3)]
    for unit in ("row", "cluster"):
        controls = (profile.get("controls_by_unit") or {}).get(unit) or {}
        for key in ("shuffled_label_full", "shuffled_label_geometric"):
            perms = (controls.get(key) or {}).get("perms") or []
            if [item.get("seed") for item in perms] != expected_seeds:
                return False
    return True


def validate_resumed_profile(profile: Dict[str, Any], model: str, task: str,
                             data_path: Path, profile_path: Path, npz_path: Path,
                             manifest_sha: str, nboot: int,
                             feature_version_deltas: set[str] | None = None) -> None:
    mismatches = []
    prov = profile.get("provenance") or {}
    if profile.get("spec_version") != SPEC_VERSION:
        mismatches.append("spec_version")
    if profile.get("model") != model or profile.get("benchmark") != task:
        mismatches.append("model/task")
    if profile.get("data_file_sha256") != sha256_file(data_path):
        mismatches.append("data_file_sha256")
    if prov.get("seed") != SEED or prov.get("n_bootstrap") != nboot:
        mismatches.append("seed/n_bootstrap")
    expected_n = len(load_jsonl(data_path))
    if profile.get("n_aligned") != expected_n or profile.get("n_dropped_unaligned") != 0:
        mismatches.append("strict sample denominator")
    if prov.get("extension_manifest_sha256") != manifest_sha:
        mismatches.append("extension_manifest_sha256")
    if prov.get("canonical_fusion_benchmark") != TASKS[task]["canonical_fusion"]:
        mismatches.append("canonical_fusion_benchmark")
    if (profile.get("fusion") or {}).get("canonical_benchmark") != TASKS[task]["canonical_fusion"]:
        mismatches.append("fusion canonical key/fallback")
    if prov.get("module_hashes") != CC.module_hashes():
        mismatches.append("module_hashes")
    if prov.get("model_snapshot_sha") != CC.model_snapshot_sha(model):
        mismatches.append("model_snapshot_sha")
    if set((profile.get("endpoints_by_unit") or {})) != {"row", "cluster"}:
        mismatches.append("required row/cluster endpoints")
    if prov.get("bootstrap_units") != ["row", "cluster"]:
        mismatches.append("bootstrap_units")
    if not _control_structure_valid(profile):
        mismatches.append("control seeds/results")
    if prov.get("controls_by_unit_sha256") != sha256_json(profile.get("controls_by_unit")):
        mismatches.append("control results digest")
    expected_unit_status = {
        unit: ("OK" if controls["shuffled_label_geometric"]["pass"]
               else f"CONTROL-FAIL[{unit}]")
        for unit, controls in (profile.get("controls_by_unit") or {}).items()
    }
    if (profile.get("endpoint_status_by_unit") != expected_unit_status
            or prov.get("endpoint_status_by_unit") != expected_unit_status):
        mismatches.append("control endpoint disposition")
    expected_comparability = (
        "feature-version-delta: Phase-1 byte-parity sentinel mismatch"
        if model.split("/")[-1] in (feature_version_deltas or set())
        else "Phase-1 frozen-row byte parity passed")
    if prov.get("feature_extraction_comparability") != expected_comparability:
        mismatches.append("feature parity disposition")
    if task == "anli_r2" and (
            not profile.get("anli_r2_split_descriptives")
            or prov.get("anli_r2_split_descriptives_sha256") != sha256_json(
                profile.get("anli_r2_split_descriptives"))):
        mismatches.append("ANLI R2 split descriptives")
    if not npz_path.exists():
        mismatches.append("matrix missing")
    else:
        data = np.load(npz_path, allow_pickle=False)
        panel = [tuple(cell) for cell in json.loads(str(data["panel"]))]
        stems = data["stem_ids"].astype(str)
        if len(data["labels"]) != expected_n or len(stems) != expected_n:
            mismatches.append("matrix sample denominator")
        if prov.get("panel_sha256") != panel_sha256(panel):
            mismatches.append("panel_sha256")
        if prov.get("ordered_stem_id_sha256") != stem_vector_sha256(stems.tolist()):
            mismatches.append("ordered_stem_id_sha256")
        fresh_audit = commitment_audit(model, data["gen_token_ids"])
        if profile.get("commitment_audit") != fresh_audit:
            mismatches.append("commitment tally/results")
        expected_commitment = (
            "OK" if fresh_audit["n_noncanonical"] == 0 else "COMMITMENT-FAIL")
        expected_tally = {
            "n": fresh_audit["n"], "n_canonical": fresh_audit["n_canonical"],
            "n_noncanonical": fresh_audit["n_noncanonical"],
        }
        if (profile.get("terminal_status") != expected_commitment
                or prov.get("commitment_status") != expected_commitment
                or prov.get("commitment_tally") != expected_tally
                or prov.get("commitment_audit_sha256") != sha256_json(fresh_audit)):
            mismatches.append("commitment status/provenance")
    if mismatches:
        raise RuntimeError(
            f"terminal ERROR: stale resume {profile_path}: {', '.join(mismatches)}; recompute")


def run_smokes(tasks: Dict[str, Path], models: Sequence[str], out_dir: Path,
               manifest_sha: str, nboot: int, aborted_tasks: set | None = None,
               aborted_models: set | None = None,
               feature_version_deltas: set[str] | None = None) -> Dict[str, Any]:
    results = {}
    for task in sorted(aborted_tasks or set()):
        for model in models:
            results[f"{model.split('/')[-1]}/{task}"] = {
                "status": "ABORT", "reason": "registered data gate failed"}
    for task in tasks:
        if task in (aborted_tasks or set()):
            continue
        for model in models:
            if model.split("/")[-1] in (aborted_models or set()):
                results[f"{model.split('/')[-1]}/{task}"] = {
                    "status": "ABORT", "reason": "model resolved_revision drift"}
    for task, data_path in tasks.items():
        if task in (aborted_tasks or set()):
            continue
        smoke_path = out_dir / "_smokes" / f"{task}.jsonl"
        write_jsonl(smoke_path, select_smoke_rows(load_jsonl(data_path), TASKS[task]["grouped"]))
        for model in models:
            tag = f"{model.split('/')[-1]}/{task}"
            if model.split("/")[-1] in (aborted_models or set()):
                continue
            try:
                profile = run_cell(
                    model, task, smoke_path, out_dir / "_smoke_profiles",
                    nboot=nboot, strict=False, manifest_sha=manifest_sha,
                    feature_version_deltas=feature_version_deltas)
                audit = profile["commitment_audit"]
                controls_pass = all(
                    controls["pass"] for controls in profile["controls_by_unit"].values())
                passed = bool(profile.get("n_aligned") == 16
                              and audit["n_canonical"] >= 15 and controls_pass)
                results[tag] = {"status": "PASS" if passed else "BEHAVIORAL-FAIL",
                                "n_aligned": profile.get("n_aligned"),
                                "commitment": audit, "controls_pass": controls_pass}
            except Exception as exc:
                results[tag] = {"status": "BEHAVIORAL-FAIL", "error": str(exc)}
                traceback.print_exc()
    summary = {"spec_version": SPEC_VERSION, "seed": SEED,
               "extension_manifest_sha256": manifest_sha,
               "task_data_sha256": {task: sha256_file(path) for task, path in tasks.items()},
               "models": [model.split("/")[-1] for model in models],
               "results": results}
    with (out_dir / "SMOKE_SUMMARY.json").open("w") as f:
        json.dump(summary, f, indent=1)
    return summary


def _geom_result(profile: Dict[str, Any], unit: str):
    return profile["endpoints_by_unit"][unit]["secondary_geometric_only"]


def _endpoint_value(cell: Dict[str, Any], unit: str,
                    systematic_commitment_fail: set) -> bool | None:
    if cell["task"] in systematic_commitment_fail:
        return False
    status = cell.get("terminal_status")
    if status in {"BEHAVIORAL-FAIL", "COMMITMENT-FAIL", "ABORT"}:
        return False
    if status in {"UNRUN", "ERROR", None}:
        return None
    profile = cell.get("profile") or {}
    if (profile.get("endpoint_status_by_unit") or {}).get(unit) != "OK":
        return False
    return bool(_geom_result(profile, unit).get("deployable"))


def score_endpoints(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    commitment_failures = Counter(
        cell["task"] for cell in cells if cell.get("terminal_status") == "COMMITMENT-FAIL")
    systematic = {task for task, count in commitment_failures.items() if count >= 3}
    by_tag = {(cell["slug"], cell["task"]): cell for cell in cells}

    def planned_value(slug, task, unit):
        cell = by_tag.get((slug, task), {"task": task, "terminal_status": "UNRUN"})
        return _endpoint_value(cell, unit, systematic)

    a1_values = [planned_value(slug, "halueval_qa", "cluster") for slug in SLUGS]
    procedural = [planned_value(slug, task, "row") for slug in SLUGS
                  for task in ("anli_r1_rep", "triviaqa_paired_rep")]
    valid = ([planned_value(slug, "anli_r1_rep", "row") for slug in SLUGS]
             + [planned_value(slug, "triviaqa_paired_rep", "cluster") for slug in SLUGS])
    b2_slugs = ["gemma-3-4b-it-4bit", "Llama-3.1-8B-Instruct-4bit"]
    b2 = {slug: planned_value(slug, "anli_r1_rep", "row") for slug in b2_slugs}

    def verdict(values, bar):
        if any(value is None for value in values):
            return {"n_pass": sum(value is True for value in values), "bar": bar,
                    "denominator": len(values), "pass": None, "blocked_by_unrun": True}
        return {"n_pass": sum(values), "bar": bar, "denominator": len(values),
                "pass": bool(sum(values) >= bar), "blocked_by_unrun": False}
    return {"A1": verdict(a1_values, 8), "B1_procedural": verdict(procedural, 17),
            "B1_valid": verdict(valid, 17),
            "B2_orphan_probe": {
                "task": "anli_r1_rep", "bootstrap_unit": "row",
                "geometric_deployable": b2,
            },
            "systematic_commitment_fail_tasks": sorted(systematic)}


def run_strict(tasks: Dict[str, Path], models: Sequence[str], out_dir: Path,
               manifest_sha: str, nboot: int, resume: bool,
               aborted_tasks: set | None = None,
               aborted_models: set | None = None,
               feature_version_deltas: set[str] | None = None) -> Dict[str, Any]:
    smoke_path = out_dir / "SMOKE_SUMMARY.json"
    if not smoke_path.exists():
        raise RuntimeError("Phase 3 smoke summary missing; strict phase is blocked")
    smoke = json.load(smoke_path.open())
    expected_data_hashes = {task: sha256_file(path) for task, path in tasks.items()}
    if (smoke.get("spec_version") != SPEC_VERSION
            or smoke.get("extension_manifest_sha256") != manifest_sha
            or smoke.get("task_data_sha256") != expected_data_hashes):
        raise RuntimeError("smoke provenance drift; Phase 3 must be rerun")
    cells = []
    for task, data_path in tasks.items():
        for model in models:
            slug, tag = model.split("/")[-1], f"{model.split('/')[-1]}/{task}"
            if task in (aborted_tasks or set()):
                cells.append({"slug": slug, "task": task, "terminal_status": "ABORT",
                              "reason": "registered data gate failed"})
                continue
            if slug in (aborted_models or set()):
                cells.append({"slug": slug, "task": task, "terminal_status": "ABORT",
                              "reason": "model resolved_revision drift"})
                continue
            smoke_result = smoke.get("results", {}).get(tag)
            if smoke_result is None:
                cells.append({"slug": slug, "task": task, "terminal_status": "UNRUN",
                              "reason": "required smoke was not run"})
                continue
            if smoke_result.get("status") == "ABORT":
                cells.append({"slug": slug, "task": task, "terminal_status": "ABORT",
                              "reason": smoke_result.get("reason")})
                continue
            if smoke_result.get("status") != "PASS":
                cells.append({"slug": slug, "task": task,
                              "terminal_status": "BEHAVIORAL-FAIL"})
                continue
            profile_path = out_dir / task / f"{slug}.profile.json"
            npz_path = out_dir / task / f"{slug}.matrix.npz"
            try:
                if resume and profile_path.exists():
                    profile = json.load(profile_path.open())
                    validate_resumed_profile(
                        profile, model, task, data_path, profile_path, npz_path,
                        manifest_sha, nboot, feature_version_deltas)
                else:
                    profile = run_cell(
                        model, task, data_path, out_dir, nboot=nboot,
                        strict=True, manifest_sha=manifest_sha,
                        feature_version_deltas=feature_version_deltas)
                cells.append({"slug": slug, "task": task,
                              "terminal_status": profile["terminal_status"],
                              "endpoint_status_by_unit": profile["endpoint_status_by_unit"],
                              "profile": profile})
            except Exception as exc:
                traceback.print_exc()
                cells.append({"slug": slug, "task": task,
                              "terminal_status": "ERROR", "error": str(exc)})
    summary = {"spec_version": SPEC_VERSION, "seed": SEED, "n_bootstrap": nboot,
               "cells": [{k: v for k, v in cell.items() if k != "profile"} for cell in cells],
               "endpoints": score_endpoints(cells)}
    with (out_dir / "SUMMARY.json").open("w") as f:
        json.dump(summary, f, indent=1)
    return summary


def gate_tasks(tasks: Dict[str, Path]) -> Dict[str, Any]:
    reports = {}
    for task, path in tasks.items():
        expected = TASKS[task]["expected_n"]
        if expected is None:
            expected = len(load_jsonl(path))
        reports[task] = GATE.run_bench_gate(
            str(path), [str(p) for p in SEALED_REFERENCES], task, expected,
            length_cap=2048)
        try:
            data_manifest = validate_data_manifest(task, path)
            reports[task]["data_manifest"] = data_manifest
        except Exception as exc:
            reports[task]["hard_failures"].append(f"data-manifest: {exc}")
            reports[task]["pass"] = False
    return reports


def validate_data_manifest(task: str, data_path: Path) -> Dict[str, Any]:
    path = data_path.with_suffix(".manifest.json")
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.load(path.open())
    if manifest.get("schema_version") != "bench-data/1.2" or manifest.get("task") != task:
        raise ValueError("schema/task mismatch")
    if manifest.get("preview"):
        raise ValueError("preview data cannot enter a registered cell")
    if manifest.get("seed") != SEED:
        raise ValueError("seed mismatch")
    if manifest.get("data_sha256") != sha256_file(data_path):
        raise ValueError("data sha256 mismatch")
    if manifest.get("rng_seed") != SEED:
        raise ValueError("sampling RNG seed mismatch")
    rows = load_jsonl(data_path)
    if int(manifest.get("n_rows", -1)) != len(rows):
        raise ValueError("manifest row count mismatch")
    label_counts = {
        str(label): sum(int(row.get("label", -1)) == label for row in rows)
        for label in (0, 1)
    }
    if manifest.get("selected_label_counts") != label_counts:
        raise ValueError("manifest selected-label counts mismatch")
    effective_n = len({str(row.get("stem_id")) for row in rows})
    if manifest.get("effective_n_stems") != effective_n:
        raise ValueError("manifest effective stem n mismatch")
    if any(not path.is_file() for path in SEALED_REFERENCES):
        missing = [str(path) for path in SEALED_REFERENCES if not path.is_file()]
        raise FileNotFoundError(
            "sealed exclusion references unavailable; set CONFLUENCE_T0_REPO to the "
            f"byte-pinned extraction core (missing: {missing})")
    expected_by_name = {
        path.name: (path.resolve(), sha256_file(path)) for path in SEALED_REFERENCES
    }
    if len(expected_by_name) != len(SEALED_REFERENCES):
        raise ValueError("sealed exclusion reference basenames are not unique")
    recorded_references = [Path(value) for value in manifest.get("exclusion_references", [])]
    resolved_references = [
        resolve_exclusion_reference(recorded, expected_by_name)
        for recorded in recorded_references
    ]
    resolved_hashes = Counter(digest for _, digest in resolved_references)
    expected_hashes = Counter(digest for _, digest in expected_by_name.values())
    if resolved_hashes != expected_hashes:
        raise ValueError(
            "enumerated exclusion-reference content union mismatch; "
            f"expected sha256 multiset={dict(expected_hashes)}")
    resolved_sources = []
    if task.startswith("halueval_"):
        if manifest.get("source_commit") != HALU_COMMIT or not manifest.get("raw_sha256"):
            raise ValueError("HaluEval commit/raw hash mismatch")
        raw_path = Path(manifest.get("raw_file") or "")
        if (HALU_COMMIT not in raw_path.parts or not raw_path.exists()
                or sha256_file(raw_path) != manifest.get("raw_sha256")):
            raise ValueError("HaluEval pinned raw bytes unavailable/mismatched")
        if manifest.get("rng") != "numpy.random.RandomState":
            raise ValueError("HaluEval RNG class mismatch")
    elif task in {"anli_r1_rep", "anli_r2"}:
        if manifest.get("artifact_fingerprint") != ANLI_FINGERPRINT:
            raise ValueError("ANLI fingerprint mismatch")
        sources = manifest.get("source_files") or {}
        expected_names = ({"train_r1"} if task == "anli_r1_rep" else {"dev_r2", "test_r2"})
        if set(sources) != expected_names:
            raise ValueError("ANLI source split set mismatch")
        for source in sources.values():
            recorded_path = Path(source["path"])
            name = recorded_path.name
            expected_sha = FROZEN_ARROW_HASHES.get(name)
            if source.get("sha256") != expected_sha or expected_sha is None:
                raise ValueError(f"ANLI source hash mismatch: {name}")
            source_path = resolve_frozen_arrow(recorded_path, expected_sha)
            resolved_sources.append(str(source_path))
        if manifest.get("rng") != "numpy.random.RandomState":
            raise ValueError("ANLI RNG class mismatch")
    else:
        if manifest.get("artifact_fingerprint") != TRIVIA_FINGERPRINT:
            raise ValueError("TriviaQA fingerprint mismatch")
        recorded_path = Path(manifest.get("source_file") or "")
        name = recorded_path.name
        expected_sha = FROZEN_ARROW_HASHES.get(name)
        if manifest.get("source_sha256") != expected_sha or expected_sha is None:
            raise ValueError("TriviaQA source hash mismatch")
        source_path = resolve_frozen_arrow(recorded_path, expected_sha)
        resolved_sources.append(str(source_path))
        if manifest.get("rng") != "random.Random":
            raise ValueError("TriviaQA RNG class mismatch")
    return {"path": str(path), "sha256": sha256_file(path),
            "source_verified": True,
            "resolved_source_files": resolved_sources,
            "resolved_exclusion_references": [str(path) for path, _ in resolved_references]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", action="append", default=[], help="registered KEY=JSONL")
    ap.add_argument("--out-dir", default="stage_b/profiles_bench")
    ap.add_argument("--phase", choices=["smoke", "strict"])
    ap.add_argument("--models", help="comma-separated slug substrings (smoke/debug only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--emit-manifest", action="store_true")
    ap.add_argument("--parity-sentinel", action="store_true",
                    help="executor-only ten-model frozen-row byte parity")
    ap.add_argument("--phase1-unit-checks", action="store_true",
                    help="executor-only synthetic BENCH plumbing checks")
    ap.add_argument("--attest-phase1", action="store_true")
    ap.add_argument("--diff-reviewed", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    if a.emit_manifest:
        path = emit_extension_manifest(out_dir)
        print(path)
        return
    if a.parity_sentinel:
        print(json.dumps(run_parity_sentinel(out_dir), indent=1))
        return
    if a.phase1_unit_checks:
        print(json.dumps(run_phase1_unit_checks(), indent=1))
        return
    if a.attest_phase1:
        print(attest_phase1(out_dir, diff_reviewed=a.diff_reviewed))
        return
    if a.phase is None:
        raise ValueError("--phase is required unless --emit-manifest is used")
    if a.nboot != NBOOT:
        raise ValueError("BENCH registered nboot is exactly 2000")
    tasks = parse_task_args(a.task)
    if not tasks:
        raise ValueError("at least one --task KEY=PATH is required")
    family_c = {task for task in tasks if TASKS[task]["family"] == "C"}
    required_confirmatory = {
        task for task, spec in TASKS.items() if spec["confirmatory"]}
    if family_c and not required_confirmatory.issubset(tasks):
        raise ValueError(
            "Phase-4 order requires all family-A/B task arguments whenever family C is run")
    manifest_path = out_dir / "EXTENSION_MANIFEST.json"
    manifest = validate_static_hashes(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    feature_version_deltas = set(
        (manifest.get("executor_attestations") or {}).get(
            "feature_version_delta_slugs") or [])
    gate_reports = gate_tasks(tasks)
    failed = {task: report["hard_failures"] for task, report in gate_reports.items()
              if not report["pass"]}
    if failed:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "GATE_FAILURES.json").open("w") as f:
            json.dump({"spec_version": SPEC_VERSION, "failures": failed,
                       "reports": gate_reports}, f, indent=1)
    aborted_tasks = set(failed)
    models = [model for model in COHORT
              if not a.models or any(part in model for part in a.models.split(","))]
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_failures = model_snapshot_mismatches(models)
    aborted_models = set(snapshot_failures)
    if snapshot_failures:
        with (out_dir / "MODEL_SNAPSHOT_ABORTS.json").open("w") as f:
            json.dump(snapshot_failures, f, indent=1)
    if a.phase == "smoke":
        summary = run_smokes(
            tasks, models, out_dir, manifest_sha, a.nboot,
            aborted_tasks, aborted_models, feature_version_deltas)
    else:
        if a.models:
            raise ValueError("strict registered phase forbids --models cohort filtering")
        summary = run_strict(
            tasks, models, out_dir, manifest_sha, a.nboot, a.resume,
            aborted_tasks, aborted_models, feature_version_deltas)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
