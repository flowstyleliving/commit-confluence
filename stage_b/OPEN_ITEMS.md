# commit-confluence — OPEN ITEMS (single source of truth)

This is the one place open threads live, so no session has to remember which handoff
file to reopen. Newest concerns first. When an item closes, move it to CLOSED at the
bottom with the commit that closed it. Every item names its own detail file.

Last touched: 2026-07-14.

---

## OPEN

### O1 — Amendment A4: single source of truth for `SPEC_VERSION`  ⟶ blocks a clean smoke re-audit
- **What:** `run_bench.SPEC_VERSION` and `analyze_universality.ACCEPTED_SPEC_VERSION` are two
  independent `"bench/1.3"` literals with no code guard. This is the exact channel that caused
  the A2 bug. Fix = leaf module `stage_b/bench_spec.py` imported by both.
- **Detail / work order:** `stage_b/CODEX_WORKORDER_A4_SPEC_MODULE.md`
- **Constraints:** filed as Amendment A4 (changes a load-bearing hash, adds a frozen file).
  Needs its OWN re-stamp script — `restamp_manifest_a2.py` correctly refuses to re-run
  (`{ids}=={"A1"}` guard, verified). Land A4 **before** the post-A3 smoke re-audit (O2) so that
  re-audit runs once against a final manifest.
- **Owner:** Codex authors, executor (Claude Code / MK) runs. NOT STARTED.

### O2 — Post-A3 smoke provenance re-audit  ⟶ hard-blocks BENCH Phase 4
- **What:** the A2/A3 (and forthcoming A4) manifest re-stamps invalidate
  `SMOKE_SUMMARY.json`'s attested `extension_manifest_sha256`. `run_bench.py:842` enforces that
  equality, so strict Phase 4 hard-fails until the 60/60 Phase-3 smokes are re-verified against
  the final manifest and `SMOKE_SUMMARY.json` is re-stamped. This is deliberate fail-closed
  behaviour, not a bug.
- **Detail:** `stage_b/CODEX_A2_AMENDMENT_HANDOFF.md` (smoke_provenance_status), and the A3
  record in `stage_b/profiles_bench/EXTENSION_MANIFEST.json`.
- **Constraints:** do AFTER O1. Executor-run (re-scores smokes); Codex cannot run it.
- **Owner:** executor. NOT STARTED. **BENCH Phase 4 stays blocked until this closes.**

### O3 — Committed `stage_b/universality.json` is stale vs the A2 code
- **What:** the registered E3 descriptive artifact committed at `bc6e2be` was computed with the
  pre-A2 **row-splitting** draw (0 `subsample_unit` fields). The code now draws whole stems on
  TriviaQA. The file a reviewer reads therefore disagrees with the code that generates it.
- **Impact — LOW, and quantified:** re-running E3 under the stem-aware draw does **not** move
  the paper's headline. At 150 labels, **15/18** deployments reach ≥0.8 deployability on BOTH
  endpoints — identical to the old file. ANLI is bit-identical (0/30 numbers moved; it takes the
  preserved legacy row path). Only 8 TriviaQA cells moved, all at n=50 (one at n=100), mostly
  *down* by ≤0.2 — the expected direction, since the old draw let a stem's correct row calibrate
  against its own wrong twin. Fresh numbers are saved at
  `<scratchpad>/universality_postA2.json` and transcribed into the results page (see detail).
- **Decision needed:** regenerate and re-commit `universality.json` on the branch (it is
  descriptive / non-gating), OR bundle the regen into the A4 execution so the registered
  artifacts move once. Recommend **bundle with A4** — one coherent re-stamp.
- **Detail:** `wiki/results/e3-stem-aware-2026-07-14.md` (vault).
- **Owner:** executor. NOT STARTED.

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
