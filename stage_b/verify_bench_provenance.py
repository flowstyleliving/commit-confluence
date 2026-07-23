#!/usr/bin/env python3
"""Verify the BENCH strict Phase-4 provenance sidecar against the files on disk.

This checks REPOSITORY COHERENCE: that every file the sidecar declares exists and
matches its recorded sha256, that counts and matrix<->profile pairing hold, and that
the run identity (spec/seed/nboot) in the sidecars agrees. It does NOT recompute
endpoints, does NOT validate model/data semantics, and does NOT prove an
execution-time timestamp for the original run.

Exit codes:
  0  every declared file exists and matches; counts and pairing hold
  1  a declared file is missing, a digest mismatches, or a count/pairing check fails
  2  unsupported schema, malformed manifest, or an unsafe path (absolute / '..' / dup)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

SUPPORTED_SCHEMAS = {"bench-provenance/1.1"}
EXPECTED_SPEC = "bench/1.3"
EXPECTED_SEED = 20260711
EXPECTED_NBOOT = 2000


def _fail(code: int, msg: str) -> "tuple[int, str]":
    return code, msg


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(rel: str) -> bool:
    if not rel or os.path.isabs(rel):
        return False
    parts = rel.replace("\\", "/").split("/")
    return ".." not in parts and "" not in parts


def verify(manifest_path: str, root: str):
    problems: list[str] = []
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return _fail(2, f"malformed manifest: {exc}")

    schema = manifest.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        return _fail(2, f"unsupported schema: {schema!r} (want one of {sorted(SUPPORTED_SCHEMAS)})")

    # Collect every declared (relative-path -> digest) pair from the hash maps.
    declared: dict[str, str] = {}
    seen: set[str] = set()
    map_fields = ["matrix_sha256", "profile_sha256", "sidecar_sha256", "lifecycle_sha256"]
    for field in map_fields:
        for rel, digest in (manifest.get(field) or {}).items():
            if not _safe_rel(rel):
                return _fail(2, f"unsafe path in {field}: {rel!r}")
            if rel in seen:
                return _fail(2, f"duplicate declared path: {rel!r}")
            seen.add(rel)
            declared[rel] = digest
    # Top-level singleton hashes.
    for rel_field, name in [("summary_sha256", "SUMMARY.json"),
                            ("extension_manifest_sha256", "EXTENSION_MANIFEST.json")]:
        digest = manifest.get(rel_field)
        if digest and name not in declared:
            declared[name] = digest

    # Hash check.
    for rel, digest in sorted(declared.items()):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            problems.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"MISMATCH: {rel}\n  declared {digest}\n  actual   {actual}")

    # Count checks.
    exp = manifest.get("expected_counts") or {}
    n_matrices = len(manifest.get("matrix_sha256") or {})
    n_profiles = len(manifest.get("profile_sha256") or {})
    if exp.get("matrices") not in (None, n_matrices):
        problems.append(f"COUNT: matrices declared {exp['matrices']} but hash map has {n_matrices}")
    if exp.get("profiles") not in (None, n_profiles):
        problems.append(f"COUNT: profiles declared {exp['profiles']} but hash map has {n_profiles}")

    # Matrix<->profile pairing: every matrix has a sibling profile and vice versa.
    mats = {rel[: -len(".matrix.npz")] for rel in (manifest.get("matrix_sha256") or {})
            if rel.endswith(".matrix.npz")}
    profs = {rel[: -len(".profile.json")] for rel in (manifest.get("profile_sha256") or {})
             if rel.endswith(".profile.json")}
    for stem in sorted(mats - profs):
        problems.append(f"PAIRING: matrix without profile: {stem}")
    for stem in sorted(profs - mats):
        problems.append(f"PAIRING: profile without matrix: {stem}")

    # Run-identity cross-check against SUMMARY.json (bytes already hash-checked above).
    summary_path = os.path.join(root, "SUMMARY.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as fh:
                summ = json.load(fh)
            if summ.get("spec_version") != EXPECTED_SPEC:
                problems.append(f"IDENTITY: SUMMARY spec_version {summ.get('spec_version')!r} != {EXPECTED_SPEC!r}")
            if summ.get("seed") != EXPECTED_SEED:
                problems.append(f"IDENTITY: SUMMARY seed {summ.get('seed')} != {EXPECTED_SEED}")
            if summ.get("n_bootstrap") != EXPECTED_NBOOT:
                problems.append(f"IDENTITY: SUMMARY n_bootstrap {summ.get('n_bootstrap')} != {EXPECTED_NBOOT}")
        except ValueError as exc:
            problems.append(f"IDENTITY: SUMMARY.json unparseable: {exc}")

    if problems:
        return _fail(1, "\n".join(problems))
    return 0, (f"OK: {len(declared)} files verified "
               f"({n_matrices} matrices + {n_profiles} profiles + sidecars/lifecycle); "
               f"counts and pairing hold; run identity {EXPECTED_SPEC}/{EXPECTED_SEED}/{EXPECTED_NBOOT}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--json", action="store_true", help="emit a JSON verdict")
    args = ap.parse_args()

    code, msg = verify(args.manifest, args.root)
    if args.json:
        print(json.dumps({"exit_code": code,
                          "ok": code == 0,
                          "detail": msg}, indent=1))
    else:
        print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main())
