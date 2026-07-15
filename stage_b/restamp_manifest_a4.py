#!/usr/bin/env python3
"""Append BENCH Amendment A4 and machine-re-stamp its frozen files.

Executor-only. Codex authored this script but did not run it. The script verifies the exact
post-A2/A3 manifest state, preserves every prior amendment and restamp record, adds the new
dependency-free spec leaf, and deliberately leaves SMOKE_SUMMARY.json stale/blocking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "stage_b/profiles_bench/EXTENSION_MANIFEST.json"
CALIBRATOR_SHA256 = "c79009a3adaf57c6ad38d2400c3c705e12d00ab18d917399b59d56b23d513e9a"
POST_A2_FILES = {
    "confluence_calibrator.py": CALIBRATOR_SHA256,
    "stage_b/PRE_REGISTRATION_BENCH.md":
        "69293dbe64e3e8ceadf1fef108eb69a691ef8f0af3f315d28fea864fbf3e65c7",
    "stage_b/analyze_universality.py":
        "3a7cddbb71a6466175eefc520d6c45705b4bf724392fd443b05db76dadabe2fc",
    "stage_b/check_fresh_data.py":
        "333b26369401bccb4fc7cbea5d7c97e1595089d4f0e586d178f337de83d2e917",
    "stage_b/fusion_signs.json":
        "92b5468bd241b517dd2d5cf70ad28556157424deb54b5f85f88af0305ff35372",
    "stage_b/generate_bench_data.py":
        "f24b5b3171725b6b19b8691371fb8b36182b0b506b54d63020d003ce7bdb21c9",
    "stage_b/run_bench.py":
        "f670c1e08ab44a6176807a312fdf2680b1188d2daff06f7c46bfddaa73a7ba57",
}
CHANGED_EXISTING = (
    "stage_b/PRE_REGISTRATION_BENCH.md",
    "stage_b/analyze_universality.py",
    "stage_b/run_bench.py",
)
NEW_FILE = "stage_b/bench_spec.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", required=True, help="human/executor identity")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    amendments = manifest.get("amendments") or []
    amendment_ids = [record.get("id") for record in amendments]
    if "A4" in amendment_ids:
        raise RuntimeError("A4 is already present; refusing a second restamp")
    if amendment_ids != ["A1", "A2", "A3"]:
        raise RuntimeError(
            f"expected exact post-A2/A3 amendment order [A1, A2, A3], got {amendment_ids}")
    if manifest.get("files") != POST_A2_FILES:
        raise RuntimeError("manifest files{} is not the exact registered post-A2/A3 state")
    if manifest.get("extension_calibrator_sha256") != CALIBRATOR_SHA256:
        raise RuntimeError("registered extension calibrator hash drift")
    if NEW_FILE in manifest["files"]:
        raise RuntimeError(f"{NEW_FILE} is already frozen; refusing to overwrite its provenance")

    for relative, expected in POST_A2_FILES.items():
        if relative in CHANGED_EXISTING:
            continue
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"unchanged frozen file drift: {relative}")
    for relative in (*CHANGED_EXISTING, NEW_FILE):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    changed_files = {
        relative: {
            "old": POST_A2_FILES[relative],
            "new": sha256(ROOT / relative),
        }
        for relative in CHANGED_EXISTING
    }
    changed_files[NEW_FILE] = {
        "old": None,
        "new": sha256(ROOT / NEW_FILE),
        "status": "NEW frozen files{} entry",
    }
    amendments.append({
        "id": "A4",
        "date": "2026-07-14",
        "amended_by": args.executor,
        "prereg_section": "9",
        "reason": (
            "replace the runner/analyzer spec-version mirror with one dependency-free leaf; "
            "filed before any completed Phase-4 cell"
        ),
        "spec_version": {"old": "bench/1.3", "new": "bench/1.3"},
        "changed_files": changed_files,
        "unchanged": (
            "calibrator, fusion_signs, builders, gate, frozen Phase-2 manifest bytes, bars, "
            "denominators, endpoints, estimators, data, controls, attestations, parity report"
        ),
        "smoke_provenance_status": (
            "STALE-BLOCKING: executor must complete the post-A4 Phase-3 smoke provenance "
            "re-audit; SMOKE_SUMMARY.json was not rewritten by this script"
        ),
        "bundled_e3_regeneration": (
            "executor-owned after restamp/smoke re-audit; not run by Codex"
        ),
        "verification": "not run by Codex; executor verification required",
    })

    for relative, hashes in changed_files.items():
        manifest["files"][relative] = hashes["new"]
    prior_restamp = manifest.get("restamp_provenance")
    manifest["restamp_provenance"] = {
        "script": "stage_b/restamp_manifest_a4.py",
        "script_sha256": sha256(Path(__file__)),
        "executed_by": args.executor,
        "codex_execution_status": "not run by Codex",
        "smoke_summary_rewritten": False,
        "previous": prior_restamp,
    }

    temporary = MANIFEST_PATH.with_suffix(".json.a4.tmp")
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(temporary, MANIFEST_PATH)
    print(f"re-stamped -> {MANIFEST_PATH}")
    print("Phase 4 remains blocked pending the post-A4 smoke provenance re-audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
