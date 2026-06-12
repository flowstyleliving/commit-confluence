# commit-confluence

A calibrated **commit-moment monitor** that unifies three in-house geometric signals
for hallucination-risk readout at (or before) the first generated token.

> ⚠️ **Status: pre-registration snapshot — not standalone-runnable.**
> This repository is published to **timestamp the pre-registration** (`stage_b/PRE_REGISTRATION.md`,
> Amendments v1–v5) and its gated fresh data **before** the registered sealed run is executed —
> that is the point of a pre-registration. The registered seal is run from a tagged commit of this
> repo so the public record and the executed code are byte-identical.
>
> The integration layer here **imports sealed modules from a separate dependency repo**
> (`t0-morphology-furnace`: `pri_calibrator`, `comprehensive_run`, `diagnose_inter_head_disagreement`,
> the RPV statistic functions in `test_shadow_ambiguity`). Those modules are part of a frozen
> research core that is **not yet public** (pending the paper), so a fresh clone will not run as-is.
> Point `$CONFLUENCE_T0_REPO` at that repo to run locally. The code, the registration, and the
> verification record are all here and fully readable; reproduction is available on request and
> will be complete when the sealed core is published.

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

1. No universal cell — selection is per-(model, distribution).
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
