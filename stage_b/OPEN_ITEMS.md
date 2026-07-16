# commit-confluence — OPEN ITEMS (single source of truth)

This is the one place open threads live, so no session has to remember which handoff
file to reopen. Newest concerns first. When an item closes, move it to CLOSED at the
bottom with the commit that closed it. Every item names its own detail file.

Last touched: 2026-07-14.

---

## OPEN

### O1 — Amendment A4 + Phase-4 launch readiness  ⟶ the path to the confirmatory run
- **What:** A4 = leaf module `stage_b/bench_spec.py` (single `SPEC_VERSION` source) imported by both
  `run_bench.py` and `analyze_universality.py`, closing the two-independent-literals channel that
  caused the A2 bug. Bundled with Phase-4 launch readiness (denominator/stem-gate/abort guards +
  a paste-able run command block) and the O3 regen.
- **Detail / work order:** `stage_b/CODEX_WORKORDER_A4_AND_PHASE4_READINESS.md` (bundled; supersedes
  the standalone `CODEX_WORKORDER_A4_SPEC_MODULE.md`, which stays as the A4 rationale). Includes a
  **post-run Codex review prompt** to be handed back after the executor completes Phase 4.
- **Constraints:** A4 is an Amendment (changes a load-bearing hash, adds a frozen file); needs its
  OWN `restamp_manifest_a4.py` — `restamp_manifest_a2.py` correctly refuses to re-run
  (`{ids}=={"A1"}` guard, verified). Land A4 **before** the O2 smoke re-audit so it runs once.
- **Supply precondition — CLEARED 2026-07-14:** `halueval_qa` built + gate-clean (1000 rows / 500
  stems / 500–500 balanced, 0 over 2048-token cap, 0 sealed overlap, no warnings). The earlier
  "~457 stems" worry was a mis-read — 457 is a *token length*; QA is 100% admissible. The
  confirmatory-supply-abort risk does NOT fire. What remains is mechanical, not scientific.
- **Owner:** Codex authoring complete; executor (Claude Code / MK) runs. See
  `stage_b/CODEX_A4_PHASE4_READINESS_HANDOFF.md`. EXECUTOR STEPS PENDING.

### O2 — Post-A4 smoke provenance re-audit  ⟶ hard-blocks BENCH Phase 4
- **What:** the A2/A3 and pending A4 manifest re-stamps invalidate
  `SMOKE_SUMMARY.json`'s attested `extension_manifest_sha256`. `run_bench.py:842` enforces that
  equality, so strict Phase 4 hard-fails until the 60/60 Phase-3 smokes are re-verified against
  the final manifest and `SMOKE_SUMMARY.json` is re-stamped. This is deliberate fail-closed
  behaviour, not a bug.
- **Detail:** `stage_b/CODEX_A4_PHASE4_READINESS_HANDOFF.md` (exact smoke command), the A2/A3
  handoff, and the A3 record in `stage_b/profiles_bench/EXTENSION_MANIFEST.json`.
- **Constraints:** do AFTER O1. Executor-run (re-scores smokes); Codex cannot run it.
- **Owner:** executor. NOT STARTED. **BENCH Phase 4 stays blocked until this closes.**

### O3 — Committed `stage_b/universality.json` is stale vs the A2 code
- **What:** the registered E3 descriptive artifact committed at `bc6e2be` was computed with the
  pre-A2 **row-splitting** draw (0 `subsample_unit` fields). The code now draws whole stems on
  TriviaQA. The file a reviewer reads therefore disagrees with the code that generates it.
- **Impact — LOW, and quantified:** re-running E3 under the stem-aware draw does **not** move
  the paper's headline. At 150 labels, **14/18** deployments reach ≥0.8 deployability on BOTH
  endpoints — identical to the old file (`bc6e2be` is also 14/18). ANLI is bit-identical (0/30
  numbers moved; it takes the preserved legacy row path). Fresh numbers saved at
  `<scratchpad>/universality_postA2.json`; regen is bit-identical to them (deterministic).
- **REGENERATED 2026-07-15 (A4 executor):** `stage_b/universality.json` re-run via
  `./confluence analyze --profiles-dir stage_b/profiles`. Verified: ANLI 0/30 numbers moved
  (only provenance fields `label_budget`/`subsample_unit` added), 14/18 @150 both draws,
  deterministic. **Two handoff assertion bugs found (not data drift):** (a) ANLI dict-equality
  compares the newly-added provenance fields ⇒ unpassable by construction; (b) magic number `15`
  never matched any file — the true count is 14/18. Corrected the "15/18" miscount in the results
  page. Regenerated artifact validated; commit decision pending user checkpoint.
- **Detail:** `wiki/results/e3-stem-aware-2026-07-14.md` (vault).
- **Owner:** executor. REGEN DONE + VALIDATED; awaiting commit alongside strict Phase-4 checkpoint.

### O4 — Scoped adversarial audit of the reviewer packet + A2/A3 logic
- **What:** Codex authored A2/A3, so it has not been adversarially reviewed by fresh eyes — only
  executionally verified (endpoints reproduce, E3 draws stems). Audit scope: (a) the
  reviewer-facing packet (README construct box, 5-tier status table, claim→artifact map,
  LICENSE/CITATION, byte-comparability tiering, vendored-extractor provenance incl. the
  `confluence_calibrator` divergence), and (b) the A2/A3 *logic* — can the sha256-multiset
  compare in `resolve_exclusion_reference` be satisfied by a wrong-but-colliding set; does
  `_e3_subsample`'s paired-stem assertion have an odd-budget / unbalanced-stem edge; can
  `matrix_stem_ids` silently return `None` on a grounded task and fall back to row sampling.
- **Explicitly OUT of scope:** the manifest / hash-chain and smoke provenance — those move under
  O1+O2 and get ONE clean audit after A4. Auditing them now is wasted.
- **Detail / work order:** `stage_b/CODEX_WORKORDER_AUDIT_REVIEWER_PACKET.md`
- **Owner:** fresh reviewer (NOT Codex — it wrote the code). NOT STARTED.

---

## CLOSED

- **A2 + A3** — restore the registered analysis path (strict `bench/1.3` gates); paired-stem-aware
  E3; portable content-hash Arrow/exclusion gates. Executor-verified, registered verdict
  unperturbed (geometric 18/20 PASS, full 18/20 FAIL). Commit `bf1933f` (branch `review-readiness`).
- **Review-readiness pass** — standalone runner rescued, real GPU extractor vendored, honest
  scope box + 5-tier status table + claim→artifact map, LICENSE/CITATION corrected. Commit
  `106ffd1` (branch `review-readiness`).
