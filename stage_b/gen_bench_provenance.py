#!/usr/bin/env python3
"""Regenerate the BENCH strict Phase-4 provenance sidecar (schema 1.1).

Upgrades 1.0 -> 1.1: adds profile-JSON hashes, sidecar hashes (SMOKE_SUMMARY,
A2_REGISTERED, SUMMARY, EXTENSION_MANIFEST), official lifecycle log/exit hashes,
expected counts + matrix<->profile pairing, and a known_analysis_defects entry.
This does NOT modify SUMMARY.json or any registered artifact. It records a
post-run checksum inventory that gives repository coherence, NOT a trusted
execution-time signature.
"""
import hashlib
import json
import os

ROOT = "/Users/msrk/Documents/commit-confluence"
BENCH = os.path.join(ROOT, "stage_b", "profiles_bench")
TASKS = ["anli_r1_rep", "anli_r2", "halueval_dialogue",
         "halueval_qa", "halueval_summarization", "triviaqa_paired_rep"]

# Files that are part of the published lifecycle record. Kept explicit so the
# inventory never silently drifts to include transient PIDs or duplicate logs.
SIDECAR_FILES = ["SUMMARY.json", "EXTENSION_MANIFEST.json",
                 "SMOKE_SUMMARY.json", "A2_REGISTERED.json"]
LIFECYCLE_FILES = ["strict_phase4_a4.log",
                   "smoke_phase3_a4.log", "smoke_phase3_a4.exit",
                   "a2_analyze.log",
                   "resume_reattest_2026-07-22.log",
                   "resume_reattest_2026-07-22.exit",
                   "GATE_FAILURES.2026-07-14.superseded.json"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_map(names):
    out = {}
    for name in names:
        p = os.path.join(BENCH, name)
        if os.path.exists(p):
            out[name] = sha256_file(p)
    return out


summary = json.load(open(os.path.join(BENCH, "SUMMARY.json")))

matrices, profiles, pairing = {}, {}, {}
for task in TASKS:
    task_dir = os.path.join(BENCH, task)
    npzs = sorted(n for n in os.listdir(task_dir) if n.endswith(".matrix.npz"))
    for name in npzs:
        slug = name[: -len(".matrix.npz")]
        rel_m = f"{task}/{name}"
        rel_p = f"{task}/{slug}.profile.json"
        matrices[rel_m] = sha256_file(os.path.join(task_dir, name))
        profiles[rel_p] = sha256_file(os.path.join(task_dir, f"{slug}.profile.json"))
        pairing.setdefault(task, []).append(slug)

report = {
    "schema_version": "bench-provenance/1.1",
    "kind": "sidecar checksum manifest (independently checkable by verify_bench_provenance.py; NOT an execution-time signature)",
    "applies_to": "stage_b/profiles_bench/ (BENCH strict Phase-4, seed 20260711)",
    "why_sidecar": (
        "run_bench.py writes SUMMARY.json without an extension_manifest_sha256 "
        "field, and run_bench.py itself is frozen as a harness hash in "
        "EXTENSION_MANIFEST.json (enforced at run_bench.py:299) rather than in "
        "the per-profile module_hashes(). The binding is therefore recorded "
        "out-of-band rather than by editing registered code after the run. "
        "Future spec revisions should emit both fields inline."
    ),
    "not_an_execution_signature": (
        "These hashes were assembled AFTER the run. They prove that the files "
        "currently in this directory are mutually consistent and match what was "
        "published; they do NOT and cannot prove a trusted execution-time "
        "timestamp for the 2026-07-15 launch. Git commit + tag is the anchor."
    ),
    "verifier": "stage_b/verify_bench_provenance.py",
    "verify_commands": [
        "python3 stage_b/verify_bench_provenance.py --manifest stage_b/profiles_bench/PROVENANCE.json --root stage_b/profiles_bench",
        "jq -r '.matrix_sha256 | to_entries[] | \"\\(.value)  \\(.key)\"' PROVENANCE.json | shasum -a 256 -c -",
        "jq -r '.profile_sha256 | to_entries[] | \"\\(.value)  \\(.key)\"' PROVENANCE.json | shasum -a 256 -c -",
    ],
    "run": {
        "spec_version": summary.get("spec_version"),
        "seed": summary.get("seed"),
        "n_bootstrap": summary.get("n_bootstrap"),
        "phase": "strict Phase-4 (confirmatory)",
    },
    "expected_counts": {
        "matrices": 53,
        "profiles": 53,
        "cells_total": 60,
        "cells_disposition": {"OK": 46, "BEHAVIORAL_FAIL": 7, "COMMITMENT_FAIL": 7},
        "note": "53 = 60 - 7 behavioral fails; behavioral-fail cells have no profile/matrix.",
    },
    "matrix_profile_pairing": pairing,
    "extension_manifest_sha256": sha256_file(os.path.join(BENCH, "EXTENSION_MANIFEST.json")),
    "summary_sha256": sha256_file(os.path.join(BENCH, "SUMMARY.json")),
    "sidecar_sha256": hash_map(SIDECAR_FILES),
    "lifecycle_sha256": hash_map(LIFECYCLE_FILES),
    "matrix_sha256": matrices,
    "profile_sha256": profiles,
    "n_matrices": len(matrices),
    "n_profiles": len(profiles),
    "known_analysis_defects": [
        {
            "id": "e3-fabricated-row-id-fallback",
            "where": "stage_b/analyze_universality.py:66 (_recover_stems returns None) -> ~:343 fabricates unique row IDs -> :355 uniqueness assertion validates the fabrication",
            "impact": (
                "A task DECLARED grouped whose stem metadata is unrecoverable "
                "silently falls back to row subsampling. It does NOT affect this "
                "run: every grouped BENCH task (TriviaQA, HaluEval x3) entered the "
                "'stem' path; only genuinely ungrouped ANLI used 'row'."
            ),
            "disposition": "future-version defect; analyzer is hash-frozen and already executed for A2 — NOT patched post hoc.",
        }
    ],
    "reattestation_2026_07_22": {
        "command": "./confluence resume-bench",
        "artifacts": ["resume_reattest_2026-07-22.exit", "resume_reattest_2026-07-22.log"],
        "exit_status": 0,
        "exit_status_source": "captured from the process; not hand-written",
        "original_a4_exit_status": (
            "NOT CAPTURED. The registered detached Phase-4 launch (2026-07-15) was "
            "specified to write stage_b/profiles_bench/strict_phase4_a4.exit; that "
            "file was never written and its status is unrecoverable. The 2026-07-22 "
            "re-attestation is a SEPARATE later event and is deliberately filed "
            "under its own name so it cannot be mistaken for the original."
        ),
        "summary_reproduced_byte_identical": True,
        "what_the_resume_did": (
            "Reconstructed all 60 cell dispositions: validated 53 stored profiles "
            "via validate_resumed_profile(), and reused 7 stored smoke records for "
            "the behavioral-fail cells (those short-circuit before any profile/matrix "
            "check). No model forward ran. The resulting SUMMARY.json contained no "
            "ERROR cells (46 OK / 7 BEHAVIORAL-FAIL / 7 COMMITMENT-FAIL) and was "
            "byte-identical to the pre-resume file."
        ),
        "what_it_did_NOT_do": (
            "Exit status 0 alone is not evidence of validation: run_bench.py catches "
            "validation exceptions and converts them to ERROR cells while still "
            "exiting 0 - the absence of ERROR cells is the actual evidence. "
            "validate_resumed_profile() checks matrix existence, array lengths, panel "
            "digest, stem digest and commitment tokens; it does NOT hash the NPZ, "
            "compare labels against source values, or recompute endpoint statistics. "
            "This is structural/provenance validation, not full matrix or endpoint "
            "revalidation."
        ),
    },
}

out_path = os.path.join(BENCH, "PROVENANCE.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=1)
    f.write("\n")

print(f"wrote {out_path}")
print(f"  matrices bound  = {report['n_matrices']}")
print(f"  profiles bound  = {report['n_profiles']}")
print(f"  sidecars bound  = {len(report['sidecar_sha256'])}")
print(f"  lifecycle bound = {len(report['lifecycle_sha256'])}")
