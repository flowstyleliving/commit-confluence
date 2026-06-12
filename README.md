# commit-confluence

A calibrated **commit-moment monitor** that unifies three in-house geometric signals
for hallucination-risk readout at (or before) the first generated token.

> ✅ **Status: registered run COMPLETE (seed 20260612).** Results below. The run was executed
> from tag [`prereg-seal-20260612`](https://github.com/flowstyleliving/commit-confluence/releases/tag/prereg-seal-20260612)
> so the executed code is byte-identical to the pre-registration (`stage_b/PRE_REGISTRATION.md`,
> Amendments v1–v5). The pre-registration and gated fresh data were committed *before* the run.
>
> **Reproducibility.** The per-cell score matrices (`stage_b/profiles/*/*.matrix.npz`) are
> published, so the descriptive analyses (E1/E2/E3 via `stage_b/analyze_universality.py`) are
> **independently reproducible from this repo alone** — no models or private dependencies needed.
> The *forward pass* that produced those matrices imports sealed modules from a separate dependency
> repo (`t0-morphology-furnace`: `pri_calibrator`, `comprehensive_run`,
> `diagnose_inter_head_disagreement`, the RPV statistics in `test_shadow_ambiguity`), a frozen
> research core **not yet public** (pending the paper); point `$CONFLUENCE_T0_REPO` at it to
> regenerate matrices locally.

## Results (registered run, seed 20260612 — 10 models × {ANLI R1, TriviaQA paired}, n=200)

Clean run: 20/20 cells computed, zero errors, all shuffled-label controls passed, registered (not preview). Two pre-registered endpoints (lead with the geometric one):

- **Geometric-only dispatcher — PASS, 18/20** (bar ≥17). A confidence-free panel (ACE attention + null_ratio + RPV) under the honest nested-OOB selector is deployable (OOB CI lower bound > 0.50) on 18 of 20 cohort cells. The registered geometric claim **holds**.
- **Full-panel (incl. confidence + fusion) — FAIL, 18/20** (bar ≥19). The strict product claim allowed ≤1 non-deployable cell and predicted exactly one (`gemma-3-4b/anli`); a second appeared, so it misses by one → the strict claim is **falsified** (the honest, registered outcome).

**Both endpoints fail the identical two ANLI cells** (`gemma-3-4b/anli`, predicted; `Llama-3.1-8B/anli`, the one model with no prior ACE seal). Confidence and fusion rescued neither — coverage is 18/20 *with or without* confidence, so those two are genuine epistemic blind spots no panel cell covers (TriviaQA 10/10, ANLI 8/10).

**No universal best signal:** the deployable cells are won by **12 distinct cells** — ACE attention dominant, RPV (fisher_eff_rank / spectral_entropy / neg_shadow) covering 4 cells where attention does not, and the pre-registered cross-locus fusion cell winning 2 outright. Corroboration *with* complementarity.

### Descriptive analyses (pre-registered, non-gating — `stage_b/universality_*.json`)

- **E1 — partial universality (first positive in the program).** Pooling 9 models to pick one fixed cell and testing on the held-out 10th, the cross-locus **fusion** cell clears the pre-registered ≥8/10 bar on both tasks (ANLI 9/10, TriviaQA 10/10 holdouts at AUROC > 0.55). No universal *champion*, but a universal **above-chance floor** — aggregation buys cross-model robustness. (Bar is 0.55 ≈ "beats chance"; strength is heterogeneous.)
- **E2 — task transfer.** Applying a model's per-task winner across tasks: median transfer AUROC **0.67**, above-floor on **85%** of transfers. Per-*model* calibration is a decent cross-task proxy.
- **E3 — label-efficiency** *(preview: reduced repeats=3, nboot=200; registered nboot=1000 pending).* Mean fraction of cells deployable climbs **0.51 (n=50) → 0.71 (n=100) → 0.83 (n=150) → 0.90 (n=200)**. The knee is ~n=100; standing up a new (model, task) costs **~150–200 labels** — affordable, not thousands.

The thesis, refined by these: *no universal best signal, but a fixed aggregate gives a universal above-chance floor; per-model calibration transfers across tasks ~85% of the time; full strength still needs per-deployment calibration at ~150–200 labels.*

## Thesis

Every signal that has survived falsification in this research program is a
*curvature / spread reading of a categorical distribution somewhere on the
commitment pathway*. Three independent research lines walked into the same room:

| Stream | Signal | Organ | Timing | Needs |
|---|---|---|---|---|
| attention | **ACE** panel (js, bos_mass, v-norm, …) | attention routing | t=0, pre-generation | nothing (W_u-free, single pass) |
| residual  | **null_ratio** (v3) | residual-stream motion Δh | gen_step ≈ 1 | Δh + W_u |
| readout   | **RPV** (fisher_eff_rank) | readout geometry of the state | any t, Δh-free | W_u only |
| base      | **surprise / p_max** | the output distribution | every token | logits |

They *converge* — they do not subsume one another. The monitor treats them as a
**panel of specialists with one honest dispatcher**: a per-(model, exact deployment
distribution) `CalibrationProfile` (nested-OOB CIs, sign-lock, drift hashes,
deployability rails) picks the deployable cell without oracle knowledge.

## Honest constraint set (what the verdicts forbid)

1. No universal *champion* cell — per-(model, distribution) selection yields 12 distinct winners.
   (But E1 finds a universal above-chance *floor* in the fusion aggregate — see Results.)
2. Stacking gains are small — the signals overlap (corroborate), they don't add orthogonally.
3. But the overlap has holes, and the holes are covered (e.g. null_ratio dies on
   Qwen3-8B where RPV is alive).
4. The streams fire at different *times* — ACE exists before generation; null_ratio cannot.

## Plan

- **Stage A — union coverage matrix** (`stage_a/`): zero-new-compute read over sealed
  artifacts. For every (model, task) cell, which families are OOB-deployable, and is
  the union gap-free (≥1 deployable family everywhere)?
- **Stage B — sealed unified-panel run** (gated on Stage A): fresh-seed pre-reg where the
  calibrator fits all families in one pass; registered claim = at least one panel cell is
  deployable on every model, and the dispatcher selects it without oracle knowledge.

## Source artifacts (read-only inputs; not vendored)

- ACE sealed profiles — `t0-morphology-furnace/experiments/t0-sealed/2026-05-26/profiles/`
- RPV comprehensive run — `t0-morphology-furnace/exploratory/shadow-ambiguity/comprehensive_outputs/`
- v3 null_ratio — carried inside the RPV run as `null_ratio_post_rank1` (same cells, same data)

This repo holds the *integration layer* only. It does not re-run or vendor the source
experiments; it reads their sealed outputs and composes them. The dependency-repo root is
configurable via the `CONFLUENCE_T0_REPO` environment variable (defaults to
`~/Documents/t0-morphology-furnace`); no absolute machine paths are committed.
