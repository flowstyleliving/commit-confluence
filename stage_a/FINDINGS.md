# Stage A - Findings (2026-06-10)

Read-only union coverage over sealed artifacts (no new compute). Full matrix:
[`out/coverage_matrix.md`](out/coverage_matrix.md). Machine copy: `out/coverage_matrix.json`.

## Headline

| view | covered (any family) | covered (geometric family) |
|---|:---:|:---:|
| all usable cells (13 models x 2 tasks) | **23 / 26** | 22 / 26 |
| **ACE sealed cohort** (9 models x 2 tasks) | **18 / 18** | 17 / 18 |

Deployable := standalone-AUROC CI lower bound > 0.50. RPV primary = `fisher_eff_rank`.
null_ratio / RPV / surprise CIs are 2000x sign-locked bootstrap from the RPV run's own
per-sample rows (same data, same cells); ACE uses its sealed nested-OOB CI lower bound.

## The structure of the misses

Strict gap-free over all 26 cells is **not** met. But the four uncovered cells split cleanly
into two kinds, and only one kind is a real problem for the panel:

### 1 genuine detector-gap (surprise works, our geometry does not)
- **`gemma-3-4b-it / anli`** - surprise 0.621 (deployable); ACE 0.656 *lo 0.49* (near-miss),
  null_ratio 0.516, RPV 0.558 (both chance). This model already **failed ACE's E_A1 seal**
  (one of the 2/9 originals), so the gap is consistent with prior knowledge, not a surprise.
  The *same model on TriviaQA* is fully covered (ACE 0.861, null 0.845, RPV 0.803): it is
  task-fragile on ANLI, not dead. On this cell, plain confidence is the backstop.

### 3 measurement-orphans (even surprise is at chance -> degenerate commit)
- **`DeepSeek-R1-Distill-Qwen-7B / triviaqa`**, **`dolphin-2.9.3-mistral-nemo-12b / anli`**,
  **`gemma-3-1b / anli`**. On each, the model's own next-token confidence sits at chance
  (CI_lo < 0.5), i.e. the commit is not cleanly measured here. All three are **out-of-ACE-panel**
  and are exactly the models with documented behavioral-gate / chat-template breakage
  (dolphin ChatML-vs-`[INST]`, gemma-3-1b CoT-overflow, DeepSeek reasoning-CoT). This is a
  data-quality problem to fix in a separate gate/parser lane, not a hole in the geometry.

## What the panel buys you (where surprise alone is weak)

The columns are not redundant - the geometry earns its place precisely where confidence is blind:
- **Mistral-7B / anli**: surprise 0.525 (chance) -> ACE 0.784, null_ratio 0.779, RPV 0.770.
- **Llama-3.2-3B / triviaqa**: surprise 0.514 (chance) -> ACE 0.830, null_ratio 0.712, RPV 0.702.
- **Qwen3-8B / anli** (the null_ratio collapse case): null_ratio 0.517 (chance) -> RPV 0.851,
  ACE 0.823. This is the backstop relationship the architecture was designed around, confirmed.

## Verdict and gate

- Strict "gap-free over all 26 cells": **NOT met** (1 detector-gap + 3 measurement-orphans).
- "Gap-free over the ACE sealed cohort, any usable signal": **met (18/18)**.
- "Zero detector-gaps outside a known ACE-seal-failure model": **met** (the only detector-gap is
  gemma-3-4b/anli, itself an E_A1 failure, and it is surprise-backstopped).

**Recommendation:** Stage B is justified, scoped to the non-degenerate deployable cohort, with
`gemma-3-4b/anli` carried as a documented surprise-backstopped gap and the 3 gate-broken models
excluded with reason (routed to the empirical-parser / chat-template lane). The naive
all-13-model union is explicitly **not** claimed gap-free.
