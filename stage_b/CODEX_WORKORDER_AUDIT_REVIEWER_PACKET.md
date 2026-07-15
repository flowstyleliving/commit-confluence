# Work order — Scoped adversarial audit: reviewer packet + A2/A3 logic

**Status:** OPEN — not started. Filed 2026-07-14. Tracked as O4 in `stage_b/OPEN_ITEMS.md`.
**Reviewer:** FRESH EYES — explicitly **not** Codex. Codex authored A2/A3; it reviewing its own
patch is self-review. A human, Claude Code, or a different model runs this.
**Why now:** the reviewer packet is stable and already public on `origin/review-readiness`. If the
scope framing or tier boundaries still overclaim anywhere, better we find it than the PM does.

---

## In scope

### Part A — the reviewer-facing packet (stable; audit now)
Read as a skeptical program manager who will try to catch an overclaim:

1. **`README.md`** — the "What this study measures / does not" construct box, the 5-tier status
   table, and the claim→artifact map. Question every claim: is each row's byte-comparability tier
   correct; does any sentence imply spontaneous free-generation hallucination detection (the study
   is candidate-answer correctness readout at verification time); is the A2 blocker still
   disclosed honestly now that A2 has landed (it has — README may need a status refresh).
2. **`LICENSE` / `CITATION.cff`** — ANLI is CC BY-**NC**; confirm nothing in the repo implies a
   commercial grant over ANLI-derived prompts. Confirm the HaluEval MIT notice is intact. Confirm
   CITATION no longer carries the invented "hallucination monitor" title.
3. **Vendored-extractor provenance** — `modal/PROVENANCE.md` and `modal/seal/`: the load-bearing
   claim is that Modal ran `confluence_calibrator.py @ 6142217f…` while repo HEAD carries
   `c79009a3…` (BENCH-amended, +268 lines), so `modal/seal/` pins the version that RAN. Verify
   that story is internally consistent and that no doc silently implies the GPU numbers reproduce
   against HEAD.
4. **Byte-comparability guardrail** — confirm no page pools sealed / mlx-vlm / Modal-torch cells
   as if comparable.

### Part B — the A2/A3 logic (authored by Codex; never adversarially read)
Executional verification already passed (endpoints reproduce; E3 draws whole stems). What is
UNVERIFIED is the logic under adversarial conditions. Probe at least:

1. **`resolve_exclusion_reference` / `resolve_frozen_arrow`** (`stage_b/run_bench.py`) — the gate
   now compares a **sha256 multiset**. Can it pass on a wrong-but-satisfying set? e.g. two
   references with swapped identities but the same hash multiset; a candidate resolved from
   `CONFLUENCE_HF_CACHE` that hash-matches but is the wrong split. Is "multiset equality" strictly
   stronger-or-equal to the old per-slot equality it replaced, or did portability weaken it?
2. **`_e3_subsample`** (`stage_b/analyze_universality.py`) — the paired-stem path asserts every
   stem has exactly 2 rows with labels `{0,1}` and requires an even budget. Edge cases: odd label
   budget; a stem with 1 or 3 rows; a task that is grouped but not perfectly paired; a budget that
   equals the stem count. Does it raise (good) or silently mis-draw (bad)?
3. **`matrix_stem_ids`** — recovers stems from `meta.question_id` when `stem_ids` is absent. Can it
   return `None` on a task that IS grounded/paired, silently sending it down the row path (the
   very bug A2 fixed)? What if `sample_idx` is present but source rows lack `question_id`? Confirm
   the label-agreement check (`labels disagree with source rows`) actually fires on a mismatch.
4. **The strict-equality gates** — confirm both `ACCEPTED_SPEC_VERSION` sites reject `"bench/1.2"`
   AND anything that is not exactly `"bench/1.3"` (no `startswith`, no prefix match crept in).

## Out of scope — do NOT audit now
- The manifest / hash-chain integrity and `SMOKE_SUMMARY.json` provenance. These move under
  Amendment A4 (O1) and the smoke re-audit (O2). Auditing them before A4 lands wastes the pass —
  everything checked gets invalidated by the next re-stamp. They get ONE clean audit after A4.

## Deliverable
A findings list ranked by severity, each with a concrete failure scenario (inputs → wrong
output), separating **reviewer-packet overclaims** from **A2/A3 logic defects**. For anything that
needs a code change, write the exact fix as a patch or a follow-on work order — do not apply it as
part of the audit. If a check requires running code, write the command and mark it "not run by
auditor" per the write/audit-only rule.
