# CODEX WORK ORDER — Pin the A3 exclusion-reference digests (O4 finding L1/L2)

**Filed:** 2026-08-25 · **Author:** Claude Code (executor) · **Assignee:** Codex (authoring only)
**Status:** NON-BLOCKING for published results — BLOCKING for any future registered BENCH run.
**Source:** O4 fresh-eyes audit, findings L1 + L2 (`stage_b/O4_AUDIT_REPORT.md`).
**Policy:** Codex is write/audit-only. Author the patch and the amendment prose. Run nothing —
no tests, no harness, no Python execution, no `run_bench.py`. Mark every verification claim
"not run by Codex"; Claude Code executes and verifies.

---

## The defect

`stage_b/run_bench.py` derives the "expected" sha256 of each sealed exclusion reference **by
hashing the very file it is about to check**:

```python
# run_bench.py:950-952, inside validate_data_manifest()
expected_by_name = {
    path.name: (path.resolve(), sha256_file(path)) for path in SEALED_REFERENCES
}
```

`resolve_exclusion_reference()` (run_bench.py:130-143) then tries `[recorded_path, current_path]`
and returns the first whose digest equals `expected_sha256`. But `current_path` **is** the file
that produced `expected_sha256`, so the second candidate always matches and the function cannot
raise. The multiset comparison that follows (run_bench.py:959-964) therefore compares digests that
were all drawn from `expected_by_name` — it reduces to **a basename-multiset check**, not a content
check.

### Demonstrated, not inferred

Claude Code lifted `sha256_file` and `resolve_exclusion_reference` verbatim from `run_bench.py`
and exercised them against a copy of a sealed reference truncated to zero rows:

```
real       rows= 200  sha=d1a3aed5e86af05c   gate -> PASS
tampered   rows=   0  sha=e3b0c44298fc1c14   gate -> PASS
control (recorded basename not registered)   gate -> FAIL: ValueError
```

`e3b0c442...` is the sha256 of the empty string. The control confirms the surviving check is the
basename lookup at run_bench.py:134.

### Why it matters

`exclusion_union()` (generate_bench_data.py:120-136) reads these files' **contents** to build the
prompt-hash and question-id sets that fresh benchmark data must not intersect. A substituted or
emptied sealed reference yields a smaller or empty exclusion set, so genuine leakage between the
sealed corpus and fresh bench data would pass with `pass=true` and no warning.

### The exposure is asymmetric — fix the right half

Claude Code checked whether each sealed digest is pinned anywhere else in the tree:

| sealed reference | digest (16) | other committed files pinning it |
|---|---|---|
| `stage_b/data/anli_R1_seed20260612_n200.jsonl` | `57ad341f2c29c886` | **33** |
| `stage_b/data/triviaqa_paired_seed20260612_n200.jsonl` | `7e0a2e2e049c83d5` | **17** |
| `vendor/t0_core/.../anli_R1_seed20260526_n200.jsonl` | `d1a3aed5e86af05c` | **0** |
| `vendor/t0_core/.../triviaqa_paired_seed20260526_n100.jsonl` | `f2f870a7e2feb2c7` | **0** |

Substituting either `stage_b/data/` file would be caught by `data_gate_*.json`, the `profiles_ext/`
profiles, and the `stage_b/audit/runs/` records. The two `vendor/t0_core/` files have **no second
check anywhere in this repository**, and they are reached through the `CONFLUENCE_T0_REPO`
environment variable — that is the live hole.

**Scope discipline:** no published number is affected. The committed files are the correct ones
(digests above verified against the 33/17 pinned copies). This is a weakened *guarantee*, not a
corrupted *result*, and the amendment prose must say so in those terms.

---

## W1 — Pin the four digests as constants

### Required change

Introduce a module-level constant in `stage_b/run_bench.py` adjacent to `SEALED_REFERENCES`,
mapping basename to its frozen digest, and use it as the source of `expected_sha256`:

```python
# Frozen 2026-08-25 (O4 L1/L2). These digests are the registration's record of the sealed
# exclusion corpus. They are NOT recomputed from the files under test -- that is the defect
# this constant exists to close. Changing a value here is an Amendments entry, never an edit.
SEALED_REFERENCE_SHA256 = {
    "anli_R1_seed20260526_n200.jsonl":          "d1a3aed5...",  # full 64 hex, see W3
    "triviaqa_paired_seed20260526_n100.jsonl":  "f2f870a7...",
    "anli_R1_seed20260612_n200.jsonl":          "57ad341f...",
    "triviaqa_paired_seed20260612_n200.jsonl":  "7e0a2e2e...",
}
```

Then in `validate_data_manifest()`:

```python
expected_by_name = {
    path.name: (path.resolve(), SEALED_REFERENCE_SHA256[path.name])
    for path in SEALED_REFERENCES
}
```

### Behavioural requirements

1. **Fail closed on an unregistered basename.** If a `SEALED_REFERENCES` entry has no key in
   `SEALED_REFERENCE_SHA256`, raise — do not fall back to hashing.
2. **Fail closed on content mismatch.** `resolve_exclusion_reference` must now genuinely be able to
   raise `FileNotFoundError` when neither candidate matches the pinned digest. Do not soften its
   message; add the offending file's actual digest to it so a mismatch is diagnosable.
3. **Do not change the multiset comparison.** With the digests pinned it becomes a real content
   check for free. Leave run_bench.py:959-964 alone.
4. **No new environment variable, no override flag, no `--skip` path.** A gate with a bypass is the
   defect in a different costume.

---

## W2 — Disclose the manifest-era bump (MANDATORY, do not skip)

`stage_b/run_bench.py` is itself listed in `MANIFEST_FILES` (run_bench.py:84-92), so its digest
feeds `extension_manifest_sha256`, which is written into every profile's provenance and **checked**
at run_bench.py:661-662.

Claude Code measured the blast radius: **117 committed artifacts under `stage_b/profiles_bench/`
carry a recorded `extension_manifest_sha256`, across two distinct values already** — the manifest
era has moved once before in this lane, so there is precedent, but this patch creates a **third
era** and every existing profile will mismatch a re-validation run under the new code.

### Required

- A new dated entry under `## 9. Amendments` in `stage_b/PRE_REGISTRATION_BENCH.md` that states:
  the defect, that it was found by the O4 audit, that **no registered result changes**, that the
  patch bumps `extension_manifest_sha256` to a third era, and that profiles written under eras one
  and two remain valid as written and are not to be re-validated against the new manifest.
- The amendment must be **forward-only**: it governs any future registered run. It must not claim
  to rescue, re-bless, or re-attest anything already scored. BENCH strict Phase-4 closed
  2026-07-22 and stays closed.
- Do **not** edit any existing profile, matrix, `SUMMARY.json`, or scored artifact to carry the new
  hash. Rewriting provenance to match new code is the precise failure this registration exists to
  prevent.

---

## W3 — Deliverables

1. A patch against `stage_b/run_bench.py` implementing W1. **Leave the four digest values as
   `PLACEHOLDER_<basename>` literals** — Claude Code will substitute the verified 64-hex digests
   and re-verify them against the pinned copies before commit. Codex must not transcribe hashes it
   cannot compute.
2. The `## 9. Amendments` entry for W2, as prose ready to paste.
3. A short note appended to `stage_b/O4_AUDIT_REPORT.md` under findings L1/L2 recording that a
   patch was authored, with this file's name — findings stay open until MK closes them.
4. A statement of what W1 does **not** fix, in your own reading of the code. At minimum address:
   whether `MANIFEST_FILES` has the same self-referential shape, and whether `module_hashes()` and
   `model_snapshot_sha()` (run_bench.py:667-670) are pinned or recomputed.

---

## Non-negotiables

- **Run nothing.** No `python`, no `pytest`, no `run_bench.py`, no import of project modules. Every
  verification line reads "not run by Codex".
- **Touch no frozen artifact**: nothing under `profiles_bench/`, no `*.exit`, no scored JSON, no
  committed matrix, no `SUMMARY.json`.
- **No `--skip-gate`-shaped escape hatch**, and no widening of a check to make a case pass.
- Exact verification commands for the executor, to be included in your handoff and marked not run
  by you: recompute each of the four digests with `sha256sum`; confirm the patched
  `resolve_exclusion_reference` raises on a zero-row copy; confirm the unpatched basename control
  still fails; confirm no file under `profiles_bench/` changed (`git status --short`).
