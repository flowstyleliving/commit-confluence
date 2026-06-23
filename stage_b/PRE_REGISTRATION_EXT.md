# Stage B — Out-of-Sample Extension Pre-Registration (gemma scale/generation/family probe)

**Status:** REGISTERED / FROZEN 2026-06-18 — the predictions below were fixed BEFORE the strict
(n=200, nboot=2000) run launched. Adapter + panel comparability were validated first by a limit-8
`is_preview` smoke on `gemma-3-12b-it`: model loaded (48 layers, lm_head V=262208), ACE 8/8 +
readout 8/8 usable, controls passed, and the produced matrix `panel` is **byte-identical** to the
seal's 27-signal matrix (`MISSING=[]`, `EXTRA=[]`). The smoke is lenient (`n_aligned=8`) and
produced NO registered metric (cannot be `--resume`d into a cell per M3-fix-2). Created 2026-06-18.

**Relationship to the sealed run:** this is an OUT-OF-SAMPLE EXTENSION. It does NOT enter, alter,
or re-open the registered seal (`prereg-seal-20260612`, 18/20 geometric PASS / 18/20 full FAIL).
The sealed 20 cells and `stage_b/profiles/` are untouched; extension cells are written to
`stage_b/profiles_ext/` via the same `run_cell`. No "X/20" bar applies — each new cell is
individually deployable-or-not, and the value is the orphan decomposition below.

## Motivation

The seal's defining honest-negative is the **`gemma-3-4b/anli` orphan**: the one cell that failed
BOTH endpoints (confidence is not its backstop). Its cause is confounded across three factors —
**scale** (4B), **generation** (Gemma 3), and **family** (Gemma). This extension de-confounds them
by adding models that move one factor at a time, on the same data, seed, panel, and selector.

## Cohort extension

Same two tasks (ANLI R1, TriviaQA paired), same fresh data, same seed.

| Model | Moves vs `gemma-3-4b` | Adapter | mlx-lm | Phase |
|---|---|---|---|---|
| `mlx-community/gemma-3-12b-it-4bit` | **scale** (4B→12B, gen-3 held) | existing GemmaAdapter | 0.29.1 (seal venv) | 1 — clean |
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | **family** control (non-gemma, 12–14B class) | existing Qwen adapter | 0.29.1 (seal venv) | 1 — clean |
| `mlx-community/gemma-4-12B-it-qat-4bit` | **generation** (gen-3→4, 12B held) | NEW `gemma4_unified` (sealed-core edit) | needs upgrade (parallel venv) | 2 — GATED |

Phase 1 runs under the **identical** module set and mlx-lm as the seal → `module_hashes` match →
fully comparable. Phase 2 requires (a) a gemma4-capable mlx-lm in a PARALLEL venv (the seal venv is
not upgraded) and (b) a sealed-core adapter + re-derived manual attention recompute; it is run only
on explicit user authorization and reported with a documented version delta (NOT byte-comparable).

## Comparability protocol (Phase 1, registered)

- `run_cell(model, task, data_path, seed=20260612, nboot=2000, limit=None, strict=True)` — the
  seal's own function; ACE live + readout `collect_readout_matrix_fresh` + merge + sealed
  `_nested_bootstrap_oob_auroc`. No reimplementation.
- Data files (identical examples to the seal):
  - `stage_b/data/anli_R1_seed20260612_n200.jsonl`
  - `stage_b/data/triviaqa_paired_seed20260612_n200.jsonl`
- Strict run: `max_dropped=0`, `n_aligned == 200` per cell (M4), incomplete cell ⇒ NOT deployable.
- Output: `stage_b/profiles_ext/<task>/<slug>.profile.json` + `<slug>.matrix.npz`.
- Provenance recorded per profile (M5): module hashes, model snapshot SHA, data sha256. Phase-1
  module hashes MUST match the sealed run's; any mismatch invalidates comparability and is reported.

## Deployability criterion

Per cell, unchanged from the seal: **geometric-only** (ACE + null_ratio + RPV) and **full-panel**
(+ confidence + 2 fusion) nested-OOB selected-cell **OOB AUROC 95% CI lower bound > 0.50**.

## Registered predictions (frozen before any strict metric)

Confidence stated explicitly; recorded so the result cannot be retrofitted.

- **`gemma-3-12b-it` / ANLI R1** — geometric deployable: **LEAN YES (~60%)**. Rationale: gen-3 NLI
  legibility should improve substantially 4B→12B; the orphan is *plausibly* partly a small-model
  effect. Predicted winner family if it passes: ACE attention or RPV.
- **`gemma-3-12b-it` / TriviaQA** — deployable: **YES (~90%)** (the 4B already passed TriviaQA).
  Winner: ACE or RPV.
- **`Qwen2.5-14B-Instruct` / ANLI R1** — deployable: **YES (~85%)** (Qwen2.5-7B passed both).
  Role: family control. Winner: ACE or fusion.
- **`Qwen2.5-14B-Instruct` / TriviaQA** — deployable: **YES (~90%)**. Winner: ACE/RPV/fusion.
- **`gemma-4-12B-it` / ANLI R1** (Phase 2, conditional) — deployable: **GENUINELY OPEN (~50%)**.
- **`gemma-4-12B-it` / TriviaQA** (Phase 2, conditional) — deployable: **LEAN YES (~75%)**.

## Pre-registered interpretation of the ANLI orphan (the science)

Decision table on ANLI-R1 *geometric* deployability, registered in advance:

| `g3-4b` | `g3-12b` | `Qwen2.5-14b` | `g4-12b` | Registered reading |
|---|---|---|---|---|
| FAIL | **PASS** | (any) | (any) | Orphan was a **scale/small-model artifact** (gen-3). |
| FAIL | **FAIL** | PASS | (any) | **Gemma-family NLI blind spot at 12B scale** — not generic scale. Strengthens "no universal detector." |
| FAIL | FAIL | (any) | **FAIL** | Blind spot **survives scale AND generation** — strongest honest-negative. |
| FAIL | FAIL | (any) | **PASS** | **Generation** (gen-4 arch) resolves it. |

Qwen2.5-14B is the discriminator between "12B-scale fixes it" and "gemma-family-specific."

## Controls (inherited from the seal, re-asserted per cell)

- Shuffled-label control, **K=3** permutations, full nested-OOB each; cell FLAGGED if ≥2/3 permuted
  CIs have CI_lo > 0.50 (does not flip deployability; reported).
- Drift hashes on all imported sealed modules recorded per profile (Phase 1: must equal seal's).
- Rotation-invariance (RPV) inherited by code identity (same compute path sha256).

## Falsification / honesty guards

- A predicted PASS that lands FAIL (or vice-versa) is reported as-is; predictions above are the
  registered record. No prediction gates publication of the cell.
- If a Phase-1 cell's module hashes drift from the seal, the cell is reported as NOT byte-comparable.
- If `gemma-3-12b/anli` passes only on the FULL panel (confidence) but fails geometric, that is
  reported distinctly (confidence-backstopped scale recovery), not as a geometric win.
- gemma-4 (Phase 2) numbers carry a mandatory version-delta caveat and are never pooled with the
  byte-comparable Phase-1 / sealed cells.

## Not in scope

No change to the sealed endpoints, the 18/20 verdict, the paper's headline cohort ("Ten Language
Models"), or `stage_b/profiles/`. Extension results land in a clearly-labeled post-seal subsection.
