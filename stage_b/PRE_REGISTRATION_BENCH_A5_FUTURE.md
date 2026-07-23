STATUS: PROPOSED — NOT FILED / NOT ACTIVE. Motivated by the CLOSED BENCH 20260711 run; may govern ONLY a fresh future run after user sign-off + document hash + commit BEFORE any new outcome data is inspected. July 22 remains B1 = 7/20; this amendment does NOT reinterpret it.

# BENCH A5 future-run amendment outline

1. **Status.** This is a future-only proposal. It has no effect on the CLOSED BENCH
   20260711 run, its registrations, its profiles, or its reported endpoints. No outcome
   from that run may be rescored under the rules below, and no historical §4 or §8.1
   disposition is softened.

2. **Disclosed motivation: the B1 §4×§8.1 commitment-gate cascade.** In the July 22
   result, §4's zero-error full-cell commitment audit and §8.1's task-level rule compose
   as follows: any task with **>=3 COMMITMENT-FAIL cells has ALL cells zeroed**, including
   cells whose terminal status is `OK`. That registered cascade, rather than a geometric
   signal failure, produced B1 = 7/20. This observed consequence motivates prospective
   redesign only; B1 must never be framed as a signal negative.

3. **Fresh-run identity.** Before execution, a separate controlling registration must
   enumerate and freeze: a new seed that is neither `20260711` nor any earlier pilot or
   registered seed; byte hashes for every newly drawn data file and every source
   artifact; the complete ordered model cohort and immutable model revisions; the exact
   task keys, sources, splits, sample sizes, stem rules, and fixed denominators; and the
   complete software/hardware stack, prompt-strategy revisions, tokenizer revisions,
   harness hashes, dependency versions, and bootstrap count. No CLOSED BENCH matrix,
   generated answer, label-derived statistic, or task draw may be reused as new outcome
   data.

4. **Exact acceptable-answer grammar and template bytes.** Every task-specific prompt
   body must end with the following UTF-8/ASCII byte string, with `\n` denoting byte
   `0a`, no trailing newline, and no leading or trailing spaces beyond those shown:

   ```text
   Judge only whether the candidate is faithful to the provided reference. Do not answer the underlying question or continue the dialogue. Reply with exactly YES if faithful or NO if not faithful.\nAnswer:
   ```

   The complete generated-answer grammar is `ANSWER := "YES" | "NO"`; the only canonical
   answer bytes are `59 45 53` and `4e 4f`. Case-folding, punctuation stripping, prose,
   an answered trivia question, and synonyms are not part of the grammar. Before a fresh
   run, the registration must print every fully expanded task template as bytes and bind
   its SHA-256; a hash mismatch aborts before model execution.

5. **Per-cell commitment-error budget.** Let `N` be the predeclared planned number of
   audited full-cell rows, including rows that later become unusable for any other reason.
   Only §6 single-token format blips are budget-eligible. A cell passes this portion of
   the commitment audit iff `E_blip / N <= 0.01`, equivalently
   `E_blip <= floor(0.01 * N)`. The denominator is never reduced to completed, finite,
   parseable, or class-balanced rows, and equality at 0.01 passes. Any non-blip grammar
   violation is handled categorically under §6 and cannot be absorbed by this budget.

6. **Predefined blip-versus-behavior classification.** A rescuable format blip is exactly
   one generated token whose decoded bytes are either `20 28` (a lone `" ("`) or `0a`
   (a lone `"\n"`). It may be the entire generated response or the sole leading token
   immediately before canonical `YES` or `NO`; no substantive content may intervene.
   Such a row increments `E_blip`; it is accepted only through the §5 cell budget and its
   raw bytes remain reported. Any other departure from the grammar is a real behavioral
   miss. In particular, a model answering the trivia question instead of judging
   faithfulness is a **REAL behavioral miss and is NEVER rescued**, regardless of
   frequency, apparent correctness, or downstream geometry. Template continuation,
   reasoning, a substantive prefix, and any other absent canonical decision are likewise
   real behavioral misses.

7. **Exact systematic-abort rule under the new budget.** First classify every audited row
   under §§4–6. A cell is `COMMITMENT-FAIL` if it contains at least one real behavioral
   miss or if `E_blip / N > 0.01`. After every planned cell on a task has a terminal
   commitment disposition, if **>=3 planned cohort cells on that task are
   `COMMITMENT-FAIL`**, declare the task behaviorally infeasible and score **all planned
   cells for that task as failures** in every affected fixed denominator, including cells
   otherwise marked `OK`. Fewer than three failing cells do not trigger task-wide
   zeroing; each failing cell still contributes zero. There is no post-outcome waiver,
   relabeling, or template retuning.

8. **Endpoint bars and fixed denominators.** Unless a separately signed future
   registration replaces the entire endpoint family before outcome inspection, the bars
   remain A1 >=8/10, A2 strict AUROC >0.55 on >=8/10 planned holdouts, and each B1 endpoint
   >=17/20. Denominators remain exactly 10, 10, and 20. Missing, aborted, control-failed,
   commitment-failed, and task-zeroed planned cells or holdouts contribute zero; they are
   never removed from a denominator. Equality passes the count bars, while A2 equality at
   AUROC 0.55 fails. An interrupted but nonterminal cell blocks endpoint scoring.

9. **Smoke/full-cell audit relationship.** Smokes use the identical prompt bytes, output
   grammar, and row classifier, but are diagnostic and contribute no endpoint values and
   no rows to a full-cell denominator. A 16-row smoke passes the commitment screen only
   with zero real behavioral misses and at most one eligible format blip; every blip is
   disclosed. Passing smoke never exempts, samples, or imputes the full audit. Every row
   of every full cell is audited independently under §§5–7, and smoke outputs are never
   resumed into the full cell.

10. **Controls and failure dispositions.** The future registration must freeze control
    units, seeds, comparison operators, failure thresholds, and endpoint-specific control
    mapping before outcomes. Source/hash/model drift aborts the affected scope; a real
    behavioral miss or excess eligible-blip rate yields `COMMITMENT-FAIL`; a frozen null
    control failure yields `CONTROL-FAIL[unit]` only for its mapped endpoint; a metric or
    resource failure yields an explicit terminal failure rather than row dropping. Raw
    generations, token ids/bytes, classifications, counts, controls, and all task-cascade
    effects are retained and reported. No failure may be converted to success by manual
    adjudication after labels or metrics are inspected.

11. **Historical result restatement.** **July 22 remains B1=7/20.** It remains the
    registered result of the CLOSED BENCH 20260711 rules, is not recomputed under this
    proposal, and is reported as a commitment-gate cascade over intact observed geometry,
    never as evidence that the geometric signal disappeared.

12. **Future seal procedure.** The user must first approve a completed version containing
    every run identity, byte string, hash, bar, denominator, control, and failure rule.
    Then compute and record this document's SHA-256 and the hashes of all bound artifacts,
    and commit the document plus hash manifest **BEFORE any new outcome data is inspected
    or any executor launches the new run**. Any edit after that commit requires a new
    append-only, user-approved amendment and a new hash-and-commit cycle before execution.
