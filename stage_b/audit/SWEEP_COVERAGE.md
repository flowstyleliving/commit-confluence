# Stage-B sealed-cell reproducibility sweep — coverage report

**Status: sweep ran to completion. No FAIL, no ERROR, no halt.**
All 20 cells of the Stage-B cohort carry a PASS-form verdict with `worst_abs_diff = 0.0`
on every comparison. Rail 4 (a FAIL or ERROR halts the sweep) never fired.

- Report generated: 2026-08-25
- Repo: `/Users/msrk/Documents/commit-confluence`
- Verifier: `stage_b/audit/verify_cell.py` (sha256 `a62bab8aa57950d13640baff83af1564eb5fcf1f529bd1f0fd24416ca881bf63`), run-record schema `cc-audit-run-record/3`
- This document is derived from the artifacts on disk (profiles, matrices, the row draw, and the run records), not from prose.

---

## 1. What the verifier actually checks, in plain words

`verify_cell.py` takes one sealed cell — one model on one task — and asks a single
question: **if we run the model again today on a handful of the same rows, do we get back
exactly the numbers that are stored in the committed matrix?** Not "close to." Exactly —
bit-for-bit identical float64 values.

To make that question meaningful it first proves it is looking at the right things, then
it runs the model for real, then it compares. Twenty-one checks, all of which must pass:

**Before touching the model — is this the right data, the right matrix, the right weights?**
1. The input `.jsonl` on disk hashes to the sha256 the profile recorded. (Nobody swapped the data.)
2. The matrix's own metadata — model id, benchmark, seed, ACE and readout data hashes — matches the profile.
3. The stored panel is exactly 21 Attention columns + 6 Readout columns, with no duplicated column identity.
4. The stored matrix is the expected shape and dtype (`(200, 27)`, float64).
5. `sample_idx` is a complete permutation of `0..n-1` — every row accounted for, none dropped or repeated.
6. The labels stored in the matrix agree with the labels in the `.jsonl` at every row, not just the sampled ones.
7. The row indices requested on the command line are in range, distinct, and nonnegative.
8. The model's resolved HuggingFace snapshot revision matches the one the profile recorded. (This check is always fatal — see limitations, it is a *pointer* comparison, not a hash of the weight shards.)
9. The subset lines handed to the fresh extraction are byte-identical to the corresponding source rows.

**Then it actually loads the model and runs real forward passes.**
10–12. The freshly extracted Attention panel has the same 27-column identity, in the same order, with the right shape/dtype, and every value is finite.
13. The number of ACE comparisons is exactly 84 — 4 distinct rows × 21 Attention columns. (A silently shortened comparison cannot masquerade as a pass.)
14. **Every one of those 84 values is byte-identical to the stored matrix.**
15. The readout re-extraction's `sample_idx` is an exact prefix with no dropped samples. (Readout rows are prefix-only *by design*: the readout extractor derives its per-row RNG from the run row index, `seed + 100_003 + i`, so only a prefix is RNG-aligned with the sealed full run. Scattered rows are valid only for the deterministic attention capture.)
16–18. The fresh Readout panel matches in identity, order, shape, dtype, and finiteness.
19. The number of readout comparisons is exactly 24 — 4 rows × 6 Readout columns.
20. **Every one of those 24 values is byte-identical to the stored matrix.**
21. Every module hash matches what the profile recorded, *or* the drift is waived at a specific pinned hash.

The equality criterion recorded in every run record is verbatim:
`"float64 byte equality; source arrays asserted float64 and finite before comparison"`.

**The verifier is winner-agnostic.** It re-extracts and compares *all 27 stored columns*
regardless of which cell won that model's panel. Fusion columns are derived downstream and
are never stored in the `.npz` (the shape is asserted `(n, 27)`), so a fusion-winner cell
verifies exactly like any other. See §5.

**Verdict vocabulary.** `verify_cell.py` emits bare `PASS` only when all checks pass **and**
the drift list is empty; `PASS_WITH_ACKNOWLEDGED_CODE_DRIFT` when all checks pass but a
waived drift exists; `FAIL` (exit 1) on any failed check; `ERROR` (exit 2) on an exception.
**Every record in this sweep reads `PASS_WITH_ACKNOWLEDGED_CODE_DRIFT`. Zero records read
bare `PASS`** — one module, `confluence_calibrator.py`, has drifted from its profile-recorded
hash on every cell, and is waived at the single pinned hash `c79009a3adaf57c6`. That waiver
was used as written on every run; it was never widened, renamed, or added to.

---

## 2. The fixed row draw and its seed

Stated up front so the protocol is auditable rather than taken on trust.

| Property | Value |
|---|---|
| Draw file | `stage_b/audit/SWEEP_ROW_DRAW.json` |
| Draw file sha256 | `266f330f790b177fc1071be4e487bc8fa11e9354a840ecbe8379821c92c831ec` |
| Schema | `cc-sweep-row-draw/1` |
| Drawing code | `stage_b/audit/draw_sweep_rows.py`, embedded verbatim in the JSON as `drawing_code`, sha256 `e8d59e1b2b29c311a8b78f7fa51d782ac3b8a1668780474134776bc3e20900fd` |
| **Seed** | **20260823** |
| RNG | `numpy.random.default_rng(20260823).choice(n, size=4, replace=False)`, numpy 2.0.2 |
| Rows per cell | 4 ACE rows; readout prefix 4 |
| Cell order | task ascending, then model slug ascending; **one shared Generator consumed in that order** |
| n_cells | 20 |

**Drawn and written to disk BEFORE any cell of this sweep ran.** The draw-file sha256 was
re-derived at run time by each cell and matched on every one.

Because a **single** Generator is consumed across all cells in a canonical order, cell *k*'s
draw depends on the entire prefix. Re-rolling any one cell would change every later cell —
which makes silent cherry-picking structurally detectable rather than merely forbidden.
Re-running the drawing script reproduced a byte-identical JSON (same sha256).

Every drawn index is in `[0, 199]`, distinct within its cell, and nonnegative — satisfying
the verifier's own "distinct nonnegative" and "requested rows in range" assertions.

### Rows as drawn

| # | Task | Model slug | ACE rows |
|---|---|---|---|
| 1 | anli_r1 | Llama-3.1-8B-Instruct-4bit | 8, 35, 170, 3 |
| 2 | anli_r1 | Llama-3.2-3B-Instruct-4bit | 74, 185, 191, 92 |
| 3 | anli_r1 | Mistral-7B-Instruct-v0.3-4bit | 58, 139, 190, 165 |
| 4 | anli_r1 | Mistral-Nemo-Instruct-2407-4bit | 148, 128, 193, 192 |
| 5 | anli_r1 | Phi-3.5-mini-instruct-4bit | 73, 181, 52, 83 |
| 6 | anli_r1 | Phi-4-mini-instruct-4bit | 22, 161, 48, 129 |
| 7 | anli_r1 | Qwen2.5-7B-Instruct-4bit | 36, 171, 3, 62 — **drawn but not executed** (already audited) |
| 8 | anli_r1 | Qwen3-1.7B-4bit | 95, 127, 32, 158 |
| 9 | anli_r1 | Qwen3-8B-4bit | 29, 171, 7, 99 |
| 10 | anli_r1 | gemma-3-4b-it-4bit | 35, 111, 26, 175 |
| 11 | triviaqa_paired | Llama-3.1-8B-Instruct-4bit | 104, 11, 186, 102 |
| 12 | triviaqa_paired | Llama-3.2-3B-Instruct-4bit | 117, 116, 189, 31 — **drawn but not executed** (already audited) |
| 13 | triviaqa_paired | Mistral-7B-Instruct-v0.3-4bit | 107, 154, 51, 42 |
| 14 | triviaqa_paired | Mistral-Nemo-Instruct-2407-4bit | 24, 113, 197, 58 |
| 15 | triviaqa_paired | Phi-3.5-mini-instruct-4bit | 50, 118, 80, 156 |
| 16 | triviaqa_paired | Phi-4-mini-instruct-4bit | 161, 164, 73, 20 |
| 17 | triviaqa_paired | Qwen2.5-7B-Instruct-4bit | 145, 45, 198, 156 |
| 18 | triviaqa_paired | Qwen3-1.7B-4bit | 47, 129, 29, 146 |
| 19 | triviaqa_paired | Qwen3-8B-4bit | 160, 63, 52, 62 |
| 20 | triviaqa_paired | gemma-3-4b-it-4bit | 24, 118, 92, 64 |

**The two already-audited cells were still drawn**, so the draw covers the full cohort and is
not conditioned on which cells had already been done. They are flagged `already_audited: true`
in the JSON, and their **historical** rows are recorded alongside as
`historical_ace_rows_used` — `anli_r1/Qwen2.5-7B = [168, 21, 0, 100]`,
`triviaqa_paired/Llama-3.2-3B = [7, 42, 123, 199]` — so no one can quietly substitute one
draw for the other. **Their newly drawn rows were never executed**; their coverage in §4 rests
on the 2026-08-13 historical runs and their historical rows. That is stated, not smoothed over.

---

## 3. The cohort

Derived from the committed profiles on disk: `stage_b/profiles/{anli_r1,triviaqa_paired}/*.profile.json`.
Exactly **10 profile + matrix pairs per task = 20 cells**. Every cell has both a
`.profile.json` and a `.matrix.npz`. All 20 report `n_aligned = 200`; both data `.jsonl`
files are 200 lines with sha256 matching the profile stamp.

The 10 model slugs are identical in both tasks: Llama-3.1-8B-Instruct-4bit,
Llama-3.2-3B-Instruct-4bit, Mistral-7B-Instruct-v0.3-4bit, Mistral-Nemo-Instruct-2407-4bit,
Phi-3.5-mini-instruct-4bit, Phi-4-mini-instruct-4bit, Qwen2.5-7B-Instruct-4bit,
Qwen3-1.7B-4bit, Qwen3-8B-4bit, gemma-3-4b-it-4bit.

Deployability recomputed from disk confirms the recorded **18/20** on the geometric-only
endpoint (`secondary_geometric_only.deployable`). The two non-deployable cells are
`anli_r1/Llama-3.1-8B-Instruct-4bit` (CI-lo 0.479) and `anli_r1/gemma-3-4b-it-4bit`
(CI-lo 0.403), matching the recorded orphans.

**Non-deployable does not mean unauditable.** `verify_cell.py` checks numerical reproduction
of the stored matrix and is indifferent to deployability. **All 20 cells are in scope and all
20 were verified.**

---

## 4. Coverage table

Verdict, `worst_abs_diff`, and record path for every cell in the cohort. Paths are relative to
`/Users/msrk/Documents/commit-confluence/`. Every record shows ACE `n_compared = 84`,
readout `n_compared = 24`, `n_mismatch = 0` on both.

| # | Task | Model | Verdict | ACE worst_abs_diff | Readout worst_abs_diff | Run record |
|---|---|---|---|---|---|---|
| 1 | anli_r1 | Llama-3.1-8B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Llama-3.1-8B-Instruct-4bit__anli_r1__20260824T023935Z.json` |
| 2 | anli_r1 | Llama-3.2-3B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Llama-3.2-3B-Instruct-4bit__anli_r1__20260825T173430Z.json` |
| 3 | anli_r1 | Mistral-7B-Instruct-v0.3-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Mistral-7B-Instruct-v0.3-4bit__anli_r1__20260825T173720Z.json` |
| 4 | anli_r1 | Mistral-Nemo-Instruct-2407-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Mistral-Nemo-Instruct-2407-4bit__anli_r1__20260825T174200Z.json` |
| 5 | anli_r1 | Phi-3.5-mini-instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Phi-3.5-mini-instruct-4bit__anli_r1__20260825T174927Z.json` |
| 6 | anli_r1 | Phi-4-mini-instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Phi-4-mini-instruct-4bit__anli_r1__20260825T175137Z.json` |
| 7 | anli_r1 | Qwen2.5-7B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen2.5-7B-Instruct-4bit__anli_r1__20260813T012728Z.json` *(historical, rows 168,21,0,100)* |
| 8 | anli_r1 | Qwen3-1.7B-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen3-1.7B-4bit__anli_r1__20260825T175442Z.json` |
| 9 | anli_r1 | Qwen3-8B-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen3-8B-4bit__anli_r1__20260825T175608Z.json` |
| 10 | anli_r1 | gemma-3-4b-it-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/gemma-3-4b-it-4bit__anli_r1__20260825T180122Z.json` |
| 11 | triviaqa_paired | Llama-3.1-8B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Llama-3.1-8B-Instruct-4bit__triviaqa_paired__20260825T180431Z.json` |
| 12 | triviaqa_paired | Llama-3.2-3B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Llama-3.2-3B-Instruct-4bit__triviaqa_paired__20260813T013138Z.json` *(historical, rows 7,42,123,199)* |
| 13 | triviaqa_paired | Mistral-7B-Instruct-v0.3-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Mistral-7B-Instruct-v0.3-4bit__triviaqa_paired__20260825T180940Z.json` |
| 14 | triviaqa_paired | Mistral-Nemo-Instruct-2407-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Mistral-Nemo-Instruct-2407-4bit__triviaqa_paired__20260825T181412Z.json` |
| 15 | triviaqa_paired | Phi-3.5-mini-instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Phi-3.5-mini-instruct-4bit__triviaqa_paired__20260825T182113Z.json` |
| 16 | triviaqa_paired | Phi-4-mini-instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Phi-4-mini-instruct-4bit__triviaqa_paired__20260825T182301Z.json` |
| 17 | triviaqa_paired | Qwen2.5-7B-Instruct-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen2.5-7B-Instruct-4bit__triviaqa_paired__20260825T182551Z.json` |
| 18 | triviaqa_paired | Qwen3-1.7B-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen3-1.7B-4bit__triviaqa_paired__20260825T183026Z.json` |
| 19 | triviaqa_paired | Qwen3-8B-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/Qwen3-8B-4bit__triviaqa_paired__20260825T183154Z.json` |
| 20 | triviaqa_paired | gemma-3-4b-it-4bit | PASS_WITH_ACKNOWLEDGED_CODE_DRIFT | 0.0 | 0.0 | `stage_b/audit/runs/gemma-3-4b-it-4bit__triviaqa_paired__20260825T183700Z.json` |

### Count

> **20 of 20 cells verified byte-identical.**
> 20 / 20 PASS-form verdicts (all `PASS_WITH_ACKNOWLEDGED_CODE_DRIFT`; 0 bare `PASS`).
> 0 FAIL. 0 ERROR. 0 halts.
> 20 × (84 ACE + 24 readout) = **2,160 individual float64 comparisons, 0 mismatches, worst absolute difference 0.0 across the entire sweep.**
>
> All **20 executed on the pre-fixed draw**. The report as first written recorded 18 executed
> and 2 carried from the 2026-08-13 historical audit on their historical rows; the executor
> then re-ran those two cells on their drawn rows — `anli_r1/Qwen2.5-7B-Instruct-4bit`
> (`36,171,3,62`) and `triviaqa_paired/Llama-3.2-3B-Instruct-4bit` (`117,116,189,31`) — both
> `PASS_WITH_ACKNOWLEDGED_CODE_DRIFT`, worst absolute difference 0.0. Records
> `Qwen2.5-7B-Instruct-4bit__anli_r1__20260825T184600Z.json` and
> `Llama-3.2-3B-Instruct-4bit__triviaqa_paired__20260825T185005Z.json`. The historical records
> remain on disk and were not deleted.

### Duplicate records on disk (bookkeeping, not a discrepancy)

`stage_b/audit/runs/` holds 25 JSON records for 20 cells. The extras are repeat runs with
**identical row arguments and identical results** — not re-picks:

- `anli_r1/Llama-3.2-3B-Instruct-4bit` — two records (`20260824T024505Z`, `20260825T173430Z`), both rows `74,185,191,92`, both `worst_abs_diff 0.0`. The table cites the later one.
- `anli_r1/Qwen2.5-7B-Instruct-4bit` — three records dated 2026-08-13, all rows `168,21,0,100`.
- `triviaqa_paired/Llama-3.2-3B-Instruct-4bit` — three records dated 2026-08-13, all rows `7,42,123,199`.

---

## 5. Untraceable-by-design cells — COVERAGE INFORMATION, NOT FAILURES

Per rail 5. **Nothing in this section is a failure, and none of it reduces the 20/20 count.**

Three cells have a **fusion** winner. Fusion columns (`fusion_rank_mean_geom`,
`fusion_rank_mean_full`) are *derived* from the stored panel and are never stored in the
`.npz`. Two different tools are therefore affected differently:

- **`verify_cell.py` — unaffected.** It is winner-agnostic: it re-extracts and compares all
  27 stored columns (21 Attention + 6 Readout) regardless of winner. **All three fusion-winner
  cells verified normally, byte-identical, and are counted in the 20/20.**
- **`trace_cc_example.py` — affected.** It resolves the winner column arithmetically and
  **fails closed** on a fusion winner. What remains out of reach for these three cells is only
  the *independent arithmetic hand-trace of the winner value* — not its reproduction.

| Task | Model | Endpoint(s) with a fusion winner | Winner cell | Verified by `verify_cell.py`? |
|---|---|---|---|---|
| anli_r1 | Phi-4-mini-instruct-4bit | geometric **and** primary | `Fusion fusion_rank_mean_geom @ step 0` | Yes — byte-identical, 84 + 24, 0.0 |
| triviaqa_paired | Qwen3-8B-4bit | geometric **and** primary | `Fusion fusion_rank_mean_geom @ step 0` | Yes — byte-identical, 84 + 24, 0.0 |
| triviaqa_paired | gemma-3-4b-it-4bit | primary only (geometric winner is `attention[final_js_kv_groups] @ step 0`) | `Fusion fusion_rank_mean_full @ step 0` | Yes — byte-identical, 84 + 24, 0.0 |

The remaining 17 cells have single-column Attention or Readout winners and are hand-traceable
in principle as well as verified in fact.

*Note for the record:* the per-cell narrative for `triviaqa_paired/gemma-3-4b-it-4bit` phrased
this as "the fusion winner value is untraceable-by-design at this level", while the narratives
for the other two fusion cells phrased it as "not untraceable-by-design". Both describe the
same underlying situation; the table above is the reconciled statement. The numbers are
identical either way.

---

## 6. Limitations — what this sweep does **not** establish

Read this section before citing the 20/20.

**It verifies re-extraction reproducibility of committed matrices. That is all it verifies.**

1. **Not an independent reimplementation.** The fresh forwards run through the *same* vendored
   extraction code (`CONFLUENCE_T0_REPO=vendor/t0_core`) that produced the sealed matrices. A
   systematic error in that code would reproduce perfectly and pass. Byte-identity proves
   *determinism and non-tampering*, not *correctness*.
2. **It does not revalidate any statistic.** Bootstrap CIs, out-of-bag selection, deployability
   thresholds, winner selection, sign locking, fusion rank-mean construction, and every scored
   endpoint are untouched by this sweep. Nothing here confirms or disconfirms a single
   registered number about detector performance. Deployability figures in §3 are *recomputed
   from stored fields*, not re-derived from data.
3. **Sampling is thin by design.** 4 of 200 rows per cell = **2% of ACE rows**; 84 of 5,400
   stored ACE values and 24 of 1,200 stored Readout values per cell. Rows outside the draw were
   never re-extracted. A defect confined to un-drawn rows would not be caught.
4. **Readout coverage is prefix-only, and structurally so.** The readout extractor derives
   per-row RNG from the run row index (`seed + 100_003 + i`), so only a prefix is RNG-aligned
   with the sealed run. Readout reproducibility is therefore demonstrated only for rows 0–3.
5. **The weight check is a pointer comparison, not a hash.** It compares the resolved
   HuggingFace snapshot *revision* under the default cache layout. Weight shards are not hashed,
   and a non-default HF cache location could diverge from what MLX actually loads.
6. **Bootstrapping caveat on provenance.** `module_hashes()` is itself supplied by
   `confluence_calibrator.py` — the one module whose drift is waived. The provenance check is
   not fully independent of the code it checks.
7. **Every verdict carries acknowledged code drift.** No cell produced a bare `PASS`.
   `confluence_calibrator.py` differs from its profile-recorded hash
   (`6142217f7608dc7c…` recorded vs `c79009a3adaf57c6…` current) on all 20 cells, waived at the
   pinned hash. The waiver forgives *one specific known state of one file*; unlisted or unpinned
   drift stays fatal. The waiver was never widened and no waiver was added.
8. **Dependencies are recorded, not hashed.** Python 3.9.6, numpy 2.0.2, mlx 0.29.3,
   mlx_lm 0.29.1, macOS-15.6.1-arm64 are recorded per run; the installed libraries are not
   hashed.
9. **No cryptographic attestation.** Run records are plain JSON written by the tool itself. A
   run killed externally leaves no record at all — an absent record is not evidence of absence
   of a failure.
10. **Tamper-evidence gap in git.** At the time every cell ran, `SWEEP_ROW_DRAW.json` and
    `draw_sweep_rows.py` were **untracked** — the recommended pre-first-cell commit of the draw
    was not performed. The draw is tamper-evident on disk (sha256, and the chained-RNG
    structure) but **not yet in git history**. Every run record notes this in its
    `dirty_paths`. Committing the draw remains outstanding.
11. **Exit codes were not observed.** The operator's shell used `${PIPESTATUS[0]}` under zsh
    (which uses lowercase `$pipestatus`, 1-indexed), so the echoed exit code came back empty on
    every cell. Verdicts in this report are read from the tool's printed `VERDICT` line and from
    the persisted record's `verdict` field — not inferred from an exit status. `verify_cell.py`
    returns 0 for both PASS forms, 1 for FAIL, 2 for ERROR.
12. ~~Two cells were not re-run.~~ **CLOSED 2026-08-25.** Both were subsequently executed on
    their drawn rows and passed; see the coverage note above. Every cell in the table now rests
    on a run against the pre-fixed draw. The superseded caveat is struck rather than deleted so
    the correction is visible.
13. **Environments differ between the historical and current runs.** The 2026-08-13 records and
    the 2026-08-24/25 records were produced at different repo HEADs. Nothing was pooled beyond
    the verdict itself.

**What it does establish, stated precisely:** for 20 of 20 sealed Stage-B cells, on
pre-registered rows fixed before any cell ran, re-running the real model forwards today under
checked data / matrix / label / snapshot provenance reproduces the committed matrix values
**bit-for-bit**, with one known and hash-pinned module drift.

---

## 7. Halt status

**No halt.** Rail 4 states that a FAIL or ERROR verdict halts the sweep and is a project-level
event. No cell returned FAIL or ERROR. The sweep ran to completion across the full cohort.

## 8. Discipline attestation

- Read-only with respect to every registered number. No profile, matrix, scored result,
  `PRE_REGISTRATION*` file, or sealed/frozen path was read-modified or written.
- Rows were read verbatim from `SWEEP_ROW_DRAW.json` on every cell; no row was re-picked,
  adjusted, or retried after a result.
- The code-drift waiver was used exactly as pinned (`confluence_calibrator.py=c79009a3adaf57c6`)
  on every cell. No waiver was widened to a bare module name and none was added.
- The only files created are the verifier's own run records under `stage_b/audit/runs/`, the
  row-draw artifacts, and this report — all inside `stage_b/audit/`.
- No push. No vault write.
