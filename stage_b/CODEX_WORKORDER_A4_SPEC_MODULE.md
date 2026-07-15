# Work order — Amendment A4: single source of truth for `SPEC_VERSION`

**Status:** SUPERSEDED by `CODEX_WORKORDER_A4_AND_PHASE4_READINESS.md`; Codex authoring complete,
executor restamp/verification pending. Filed 2026-07-14, after A2/A3 landed.
**Lane:** Codex authors (write/audit only, runs nothing). Claude Code / MK executes and verifies.
**Blocking:** BENCH Phase 4 is already blocked pending the post-A3 smoke provenance re-audit.
A4 should land *before* that re-audit so the two re-stamps collapse into one.

---

## Why

Amendment A2 fixed the symptom. It did not close the channel.

The original A2 bug: A1 bumped the spec to `bench/1.3` in `run_bench.py`, while
`analyze_universality.py` kept hard-gating on `"bench/1.2"`. Nothing caught it, and the
registered analysis path silently became unreachable. The file was hash-frozen in
`EXTENSION_MANIFEST.json` under an A1 record that attested `"unchanged": "…analysis…"` —
literally true, consequentially false.

What A2 shipped is a **mirror**, not a single source of truth:

| file | constant | value |
|---|---|---|
| `stage_b/run_bench.py:31` | `SPEC_VERSION` | `"bench/1.3"` |
| `stage_b/analyze_universality.py:50` | `ACCEPTED_SPEC_VERSION` | `"bench/1.3"` |

These are two independent string literals. Nothing in code ties them together — only a
comment and the convention that the manifest re-stamps both files together. **That is the
exact failure channel that produced A2.** A future bump to `bench/1.4` that touches only
the runner reproduces the bug, and the manifest will happily record the analyzer's
unchanged hash while the analysis path is dead.

Codex's reason for mirroring is legitimate and should be preserved: importing `run_bench`
from the analyzer would couple a matrix-only, NumPy-only analysis path to the runner's MLX
runtime imports. The fix is not "import the runner" — it is to extract the constant.

## What to build

A new leaf module with no dependencies:

```python
# stage_b/bench_spec.py
SPEC_VERSION = "bench/1.3"
```

Then:

- `run_bench.py` — replace the literal at `:31` with `from bench_spec import SPEC_VERSION`.
- `analyze_universality.py` — replace `ACCEPTED_SPEC_VERSION = "bench/1.3"` at `:50` with
  an import of the same constant. Keep the local name if it reads better at the call sites;
  bind it to the imported value, do not restate the string.
- Both call sites keep **strict equality**. Do not widen into a prefix/regex/`startswith`
  matcher — the whole point of A2 was that the gate is exact.

The module must stay import-safe from the analyzer's environment: **no MLX, no torch, no
`confluence_calibrator`, no transitive runtime imports.** A bare constant, nothing else.

## Amendment discipline — this is A4, not cleanup

This changes the bytes of `run_bench.py` (load-bearing, hash-frozen) and
`analyze_universality.py`, and adds `stage_b/bench_spec.py` to the frozen file set. It is a
provenance mutation and must be filed as **Amendment A4** in `PRE_REGISTRATION_BENCH.md` §9
(append-only) and re-stamped into `EXTENSION_MANIFEST.json`. Do not slip it in as a tidy-up
commit.

**Timing note — the ideal window is gone.** The original review note recommended landing
this *before* the executor ran `restamp_manifest_a2.py`, so A4 would ride along with the
A2/A3 re-stamp. That did not happen: the re-stamp was executed and committed as `bf1933f`
on 2026-07-14. Consequences for whoever picks this up:

1. `restamp_manifest_a2.py` **cannot be re-run.** It guards on
   `{amendment ids} == {"A1"}` (`:66-67`) and the manifest now carries `A1, A2, A3`. It will
   raise. This is correct behaviour — it is an idempotency guard, not a bug. Do not weaken it.
2. A4 therefore needs its **own** machine-produced re-stamp script (`restamp_manifest_a4.py`),
   built on the same pattern: verify the pre-amendment hashes still match, append the A4
   record, update `files{}`, record `restamp_provenance`, and refuse to run twice.
3. The new script must also add `stage_b/bench_spec.py` to `files{}` — a *new* frozen entry,
   not a re-stamp of an existing one.
4. `SMOKE_SUMMARY.json` still attests the manifest's own sha256, and `run_bench.py:842`
   enforces it. A4's re-stamp moves that hash again. This does **not** add a new blocker —
   Phase 4 is already hard-blocked by A3 pending the smoke provenance re-audit. It does mean
   **A4 should land before the re-audit**, so the re-audit runs once against a manifest that
   is final, rather than twice.

## Non-negotiables

- Do not edit `sealed_selector.py`, `confluence_calibrator.py`, or the frozen Phase-2 data
  manifests under `stage_b/data_bench/`.
- Do not change bars, denominators, endpoints, estimators, or data.
- Do not widen the version gates.
- Do not backdate amendments or rewrite A1/A2/A3 records. A4 is append-only.
- Codex runs nothing. Write the exact verification commands and mark them
  "not run by Codex".

## Executor verification (for whoever runs it)

- All four frozen surfaces byte-unchanged.
- `./confluence python -m py_compile` on every touched file.
- `./confluence python -c "import sys; sys.path.insert(0,'stage_b'); import bench_spec, analyze_universality"`
  — proves the analyzer still imports without an MLX environment.
- Grep proves exactly one `"bench/1.x"` literal remains in `stage_b/*.py` (in `bench_spec.py`).
- `./confluence doctor` → READY.
- `./confluence verify` → **geometric 18/20 PASS, full panel 18/20 FAIL**. The registered
  verdict must be unperturbed; A4 is a refactor of a constant, not a change of estimand. If
  either endpoint moves, stop.
