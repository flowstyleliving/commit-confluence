# Work order — Amendment A4 + Phase-4 launch readiness (bundled)

**Status:** AUTHORED BY CODEX — executor verification/restamp/smoke/E3/Phase 4 pending. Filed
2026-07-14. Tracks O1 (+ enables O2, O3) in `OPEN_ITEMS.md`.
**Codex:** authors everything below (write/audit only — runs nothing: no smokes, no forwards, no
calibration, no `restamp` execution, no agent spawns). Marks every verification "not run by Codex".
**Executor (Claude Code / MK):** runs the restamp, the smoke re-audit, and the strict Phase-4 seal
on this Mac mini (MLX, cached weights). The executor spawns/runs the job — never Codex.

**Supply precondition — CLEARED 2026-07-14.** `halueval_qa` is built and gate-clean: 1000 rows /
500 stems / 500–500 balanced, 0 over the 2048-token cap, 0 sealed overlap, no warnings. The
confirmatory-supply-abort risk (pre-reg §539) does not fire. Phase 4 is a go on supply; what
remains is mechanical.

---

## W1 — Amendment A4: single `SPEC_VERSION` source of truth  (this is O1)

Full rationale in `CODEX_WORKORDER_A4_SPEC_MODULE.md`; summary: `run_bench.SPEC_VERSION` and
`analyze_universality.ACCEPTED_SPEC_VERSION` are two independent `"bench/1.3"` literals — the exact
channel that caused the A2 bug.

- Create `stage_b/bench_spec.py` — a bare leaf module, no dependencies:
  `SPEC_VERSION = "bench/1.3"`. **No MLX, no torch, no calibrator, no transitive runtime imports**
  (it must import from the analyzer's NumPy-only environment).
- `run_bench.py:31` → `from bench_spec import SPEC_VERSION`.
- `analyze_universality.py:50` → import the same constant (keep a local alias if it reads better;
  bind it, don't restate the string).
- Both call sites keep **strict equality**. Do not introduce `startswith`/prefix/regex.
- File as **Amendment A4** in `PRE_REGISTRATION_BENCH.md` §9 (append-only).
- Author `stage_b/restamp_manifest_a4.py` (own script — `restamp_manifest_a2.py` correctly refuses
  to re-run). It must: verify the current post-A2 manifest hashes still match; append the A4
  record; re-stamp `run_bench.py` and `analyze_universality.py`; **add `stage_b/bench_spec.py` as a
  NEW `files{}` entry**; write `restamp_provenance`; refuse to run twice.

## W2 — Regenerate the stale registered E3 artifact  (this is O3)

`stage_b/universality.json` (committed `bc6e2be`) was computed with the pre-A2 row-split draw and
now disagrees with the code. **Executor** regenerates it via `./confluence analyze` after A4 lands,
so the registered descriptive artifacts move exactly once. Codex: add a one-line note to the A4
prereg entry that the regen is bundled here, and confirm (statically) that `analyze` writes
`subsample_unit` into every E3 record. Expected result is already known and does NOT gate anything:
15/18 deployable deployments reach ≥0.8 at 150 labels, ANLI bit-identical (see
`wiki/results/e3-stem-aware-2026-07-14.md`). If the regen deviates from that, STOP — the refactor
moved something it shouldn't have.

## W3 — Phase-4 launch readiness review (static)

Do NOT run Phase 4. Statically confirm the harness is launch-ready and the guards are real, so the
executor's multi-hour run can't silently corrupt the registered cohort:

1. **Confirmatory denominator = exactly 30 cells** (families A+B: `halueval_qa`, `anli_r1_rep`,
   `triviaqa_paired_rep` × 10 models). Confirm `run_bench.py` enforces this and that an errored /
   incomplete cell counts as NOT deployable (no LOMO escape).
2. **Stem-cluster gate is the confirmatory selector** for grouped/QA cells (pre-reg §365-366) —
   confirm the gate actually uses `stem_id` end-to-end and never silently falls back to row
   bootstrap. This is the same failure class as the A2 stem bug; verify it cannot recur here.
3. **Confirmatory-supply abort** (§539) and **parity/exclusion predicates** fire on violation, not
   warn. Confirm `--resume` re-validates each existing profile (seed / nboot / model / task /
   data-sha256 / module-hashes) and turns drift into an error → FAIL.
4. **The A3 content-hash resolution** (Arrow + exclusion refs) is on the Phase-4 path, so the run
   is portable and `CONFLUENCE_HF_CACHE` is honored.
5. Produce a **RUN COMMAND BLOCK** the executor pastes verbatim: the exact `./confluence` seal
   invocation (seed, nboot, task files, out-dir, `--resume` policy), plus the post-run
   `./confluence analyze` and the A2 `--bench-a2` scoring call. Include the smoke-re-audit command
   the executor must run FIRST (O2) to unblock `run_bench.py:842`.

## Ordering the executor will follow (Codex documents it; does not run it)

1. Land W1 (A4 code + prereg + `restamp_manifest_a4.py`).  2. Executor runs `restamp_manifest_a4.py`.
3. Executor runs the **smoke re-audit** (O2) → re-stamps `SMOKE_SUMMARY.json` against the final
   manifest.  4. Executor regenerates `universality.json` (W2).  5. Executor runs strict Phase 4
   (W3 command block), detached, streaming to a run log + `wiki/log.md`.  6. `./confluence analyze`
   + A2 scoring on the completed cohort.

## Non-negotiables
- Do not edit `sealed_selector.py`, `confluence_calibrator.py`, or the frozen Phase-2 data
  manifests under `stage_b/data_bench/`.
- Do not widen any version/spec gate. Do not backdate or rewrite A1/A2/A3 records.
- Do not change confirmatory bars, denominators, endpoints, estimators, or data.
- Codex runs nothing; every verification is an executor command marked "not run by Codex".

## Executor verification (post-run, before believing the verdict)
- All four frozen surfaces byte-unchanged.
- Exactly one `"bench/1.x"` literal survives in `stage_b/*.py` (in `bench_spec.py`).
- `./confluence verify` on the SEALED cohort still returns geometric 18/20 PASS, full 18/20 FAIL
  (A4 is a constant refactor; the sealed verdict must not move).
- Phase-4 SUMMARY: 30/30 confirmatory cells scored, 0 drops, controls pass, per-cell provenance
  intact.

---

## POST-RUN — Codex review prompt (hand this to Codex AFTER the executor completes Phase 4)

> Codex, adversarially review the completed BENCH Phase-4 confirmatory run in
> `/Users/msrk/Documents/commit-confluence`. **Write/audit only — run nothing** (no smokes,
> forwards, calibration, or re-stamps); for any check needing execution, write the exact command
> and mark it "not run by Codex". Do not edit `sealed_selector.py`, `confluence_calibrator.py`, or
> the frozen Phase-2 data manifests.
>
> Verify, in order:
> 1. **Provenance chain is intact end-to-end.** The A4 re-stamp appended A4 to
>    `EXTENSION_MANIFEST.json`, added `bench_spec.py` to `files{}`, and re-stamped the two touched
>    files; `SMOKE_SUMMARY.json`'s `extension_manifest_sha256` now matches the final manifest (the
>    O2 re-audit). Confirm no amendment record was backdated or rewritten, and A1's original
>    `unchanged` wording is still retained with its correction.
> 2. **The confirmatory verdict is honestly derived.** 30/30 cells scored, 0 drops; the
>    stem-cluster bootstrap (not row bootstrap) is the gate for `halueval_qa` and every grouped
>    cell; the planned-denominator rule holds (errored/incomplete = NOT deployable, no LOMO
>    escape); shuffled-label controls pass. Recompute the PASS/FAIL against the pre-registered A1
>    task bar and A2 fusion-LOMO floor from the emitted matrices and confirm the reported verdict
>    matches — do not take SUMMARY.json's word for it.
> 3. **No pooling violation.** Confirmatory MLX cells are not compared against the non-byte-
>    comparable torch/mlx-vlm extension cells as if equivalent.
> 4. **The sealed 18/20 is unperturbed.** A4 was a constant refactor; confirm the sealed cohort
>    verdict (geometric 18/20 PASS, full 18/20 FAIL) is byte-unchanged and that Phase 4 added a
>    *separate* confirmatory section rather than mutating the sealed one.
> 5. **The paper's claims still match the artifacts.** Any `halueval_qa` / label-budget / clustering
>    statement in `wiki/paper/cc-draft.tex` is consistent with what actually ran; flag every
>    overclaim.
>
> Deliver a severity-ranked findings list; separate provenance defects, verdict-derivation defects,
> and paper-vs-artifact mismatches. Propose fixes as patches or follow-on work orders — do not apply
> them.
