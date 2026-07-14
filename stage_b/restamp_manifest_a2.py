#!/usr/bin/env python3
"""Machine-produce the A2/A3 EXTENSION_MANIFEST re-stamp.

Codex authored this script but did not execute it.  The executor must review the patch, run
the required verification, and invoke this script with an explicit attesting identity.  The
script intentionally does not rewrite SMOKE_SUMMARY.json; strict Phase 4 must remain blocked
until the post-A3 smoke provenance re-audit is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "stage_b/profiles_bench/EXTENSION_MANIFEST.json"
FILES = {
    "prereg": ROOT / "stage_b/PRE_REGISTRATION_BENCH.md",
    "analysis": ROOT / "stage_b/analyze_universality.py",
    "runner": ROOT / "stage_b/run_bench.py",
    "calibrator": ROOT / "confluence_calibrator.py",
}
OLD = {
    "stage_b/PRE_REGISTRATION_BENCH.md":
        "bfb3ba2c38271c3221505746c5f49aaca78f8c6708dd0c37b615f5a97c81b286",
    "stage_b/analyze_universality.py":
        "d85cacf891f1e5c574e1aa53b55e486e284d9db57e8af917d370d934a7f3aef2",
    "stage_b/run_bench.py":
        "70b15c9f030cc32c3421b50c6668baa37328b7c57b73ec3e24bfe609e89fcb56",
}
FROZEN_CALIBRATOR_SHA256 = (
    "c79009a3adaf57c6ad38d2400c3c705e12d00ab18d917399b59d56b23d513e9a")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def changed(old_key: str, new_path: Path) -> dict[str, str]:
    return {"old": OLD[old_key], "new": sha256(new_path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executor", required=True,
        help="identity that reviewed, executed, and owns this provenance mutation")
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    amendments = manifest.get("amendments")
    require(isinstance(amendments, list), "manifest amendments must be a list")
    require({item.get("id") for item in amendments} == {"A1"},
            "expected exactly the pre-A2 A1 amendment history")
    for path, expected in OLD.items():
        require((manifest.get("files") or {}).get(path) == expected,
                f"manifest's pre-amendment hash drifted: {path}")
    require(sha256(FILES["calibrator"]) == FROZEN_CALIBRATOR_SHA256,
            "confluence_calibrator.py changed; A2/A3 scope exceeded")
    require(manifest.get("extension_calibrator_sha256") == FROZEN_CALIBRATOR_SHA256,
            "manifest calibrator attestation drifted")

    prereg_change = changed("stage_b/PRE_REGISTRATION_BENCH.md", FILES["prereg"])
    analysis_change = changed("stage_b/analyze_universality.py", FILES["analysis"])
    runner_change = changed("stage_b/run_bench.py", FILES["runner"])
    require(analysis_change["new"] != analysis_change["old"],
            "analysis did not change for A2")
    require(runner_change["new"] != runner_change["old"],
            "runner did not change for A3")
    require(prereg_change["new"] != prereg_change["old"],
            "pre-registration did not receive A2/A3 entries")

    a1 = amendments[0]
    require(a1.get("unchanged") ==
            "calibrator, fusion_signs, builders, gate, analysis, attestations, parity report",
            "A1 original unchanged wording drifted")
    a1["unchanged_correction"] = {
        "filed_by": "A2",
        "date": "2026-07-14",
        "original_wording_retained": True,
        "correction": ("analysis was byte-unchanged but incompatible with A1's bench/1.3 "
                       "outputs; A2 restores the registered analysis path"),
    }

    amendments.extend([
        {
            "id": "A2",
            "date": "2026-07-14",
            "amended_by": args.executor,
            "prereg_section": "9",
            "reason": ("restore strict bench/1.3 A2 input gates and make descriptive E3 "
                       "paired-stem-aware; filed before any completed Phase-4 cell"),
            "spec_version": {"old": "bench/1.3", "new": "bench/1.3"},
            "estimator_output_schema": {
                "old": "bench-a2/1.2", "new": "bench-a2/1.2"},
            "changed_files": {
                "stage_b/PRE_REGISTRATION_BENCH.md": prereg_change,
                "stage_b/analyze_universality.py": analysis_change,
            },
            "unchanged": ("calibrator, run_bench execution semantics, fusion_signs, builders, "
                          "gate, bars, denominators, endpoints, data, attestations, parity report"),
            "verification": "not run by Codex; executor verification required",
        },
        {
            "id": "A3",
            "date": "2026-07-14",
            "amended_by": args.executor,
            "prereg_section": "9",
            "reason": ("resolve frozen Arrow and exclusion inputs by registered content hash "
                       "without rewriting Phase-2 manifests"),
            "spec_version": {"old": "bench/1.3", "new": "bench/1.3"},
            "changed_files": {
                "stage_b/PRE_REGISTRATION_BENCH.md": prereg_change,
                "stage_b/run_bench.py": runner_change,
            },
            "unchanged": ("calibrator, fusion_signs, builders, gate, frozen Phase-2 manifest "
                          "bytes, bars, denominators, endpoints, data, attestations, parity report"),
            "smoke_provenance_status": (
                "STALE-BLOCKING: executor must complete post-A3 Phase-3 provenance re-audit; "
                "SMOKE_SUMMARY.json was not rewritten by this script"),
            "verification": "not run by Codex; executor verification required",
        },
    ])

    manifest["files"]["stage_b/PRE_REGISTRATION_BENCH.md"] = prereg_change["new"]
    manifest["files"]["stage_b/analyze_universality.py"] = analysis_change["new"]
    manifest["files"]["stage_b/run_bench.py"] = runner_change["new"]
    manifest["restamp_provenance"] = {
        "script": "stage_b/restamp_manifest_a2.py",
        "script_sha256": sha256(Path(__file__)),
        "executed_by": args.executor,
        "codex_execution_status": "not run by Codex",
        "smoke_summary_rewritten": False,
    }

    temporary = MANIFEST_PATH.with_suffix(".json.a2-a3.tmp")
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(temporary, MANIFEST_PATH)
    print(f"re-stamped -> {MANIFEST_PATH}")
    print("Phase 4 remains blocked pending the post-A3 smoke provenance re-audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
