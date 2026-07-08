# commit-confluence

A calibrated **commit-moment monitor** that unifies three in-house geometric signals
for hallucination-risk readout at (or before) the first generated token.

**Companion paper:** *Decoder LLM Hallucination: No Universal Detector, but a Universal
Floor — A Pre-Registered Study of Commit-Moment Hallucination Monitoring Across Ten
Language Models* (M. S. R. Kitti, Furnace Research, June 2026). This repository is the
paper's reproducibility artifact: the pre-registration, the gated fresh data, the
registered per-deployment score matrices and profiles, and the analysis code. Citation
metadata in [`CITATION.cff`](CITATION.cff); code MIT, artifacts CC BY 4.0 ([`LICENSE`](LICENSE)).

> ✅ **Status: registered run COMPLETE (seed 20260612).** Results below. The run was executed
> from tag [`prereg-seal-20260612`](https://github.com/flowstyleliving/commit-confluence/releases/tag/prereg-seal-20260612)
> so the executed code is byte-identical to the pre-registration (`stage_b/PRE_REGISTRATION.md`,
> Amendments v1–v5). The pre-registration and gated fresh data were committed *before* the run.
>
> **Reproducibility.** The per-deployment score matrices (`stage_b/profiles/*/*.matrix.npz`) are
> published, and the sealed selection machinery is vendored (`sealed_selector.py`, byte-identical
> to the sealed module — provenance sha256 in its docstring matches the `module_hashes` recorded
> in every registered profile). Both registered endpoint verdicts
> (`stage_b/verify_endpoints.py`) and the descriptive analyses E1/E2/E3
> (`stage_b/analyze_universality.py`) are therefore **reproducible from this repo alone** — no
> models, no private dependencies (see *Reproduce the registered results* below).
> The *forward pass* that produced the matrices imports sealed extraction modules from a separate
> dependency repo (`t0-morphology-furnace`: `pri_calibrator`, `comprehensive_run`,
> `diagnose_inter_head_disagreement`, the RPV statistics in `test_shadow_ambiguity`), a frozen
> research core **not yet public** (pending the companion reports); point `$CONFLUENCE_T0_REPO`
> at it to regenerate matrices from scratch.

## Reproduce the registered results (no models needed)

```bash
pip install -r requirements-analysis.txt   # numpy + scipy + scikit-learn

# Both pre-registered endpoint verdicts, re-derived from the published matrices at the
# registered settings (seed 20260612, nboot 2000) and compared byte-exactly against the
# committed profiles. Prints the 18/20 PASS / 18/20-vs-19 FAIL tallies. (~minutes; add
# --nboot 200 for a quick pass.)
python stage_b/verify_endpoints.py

# The pre-registered descriptive analyses E1 (LOMO universality) / E2 (task transfer) /
# E3 (label efficiency). Registered E3 settings are --repeats 10 --nboot-labeleff 1000.
python stage_b/analyze_universality.py --profiles-dir stage_b/profiles --out /tmp/universality.json

# The post-seal extension cells (scale/family + the non-byte-comparable gemma-4 axis):
python stage_b/verify_endpoints.py --profiles-dir stage_b/profiles_ext
```

E1/E2 are deterministic given the matrices and reproduce `stage_b/universality.json`
identically; E3 and the endpoint verification are exactly reproducible at the registered
seed/bootstrap settings.

## Results (registered run, seed 20260612 — 10 models × {ANLI R1, TriviaQA paired}, n=200)

**Terminology:** a *deployment* = one (model, task) pairing (20 total); a *signal* = one candidate detector in the 29-entry panel (e.g. `attention[final_bos_mass] @ step 0`). The honest selector picks one signal per deployment.

Clean run: 20/20 deployments computed, zero errors, all shuffled-label controls passed, registered (not preview). Two pre-registered endpoints (lead with the geometric one):

- **Geometric-only dispatcher — PASS, 18/20** (bar ≥17). A confidence-free panel (ACE attention + PRI + RPV) under the honest nested-OOB selector is deployable (OOB CI lower bound > 0.50) in 18 of 20 deployments. The registered geometric claim **holds**.
- **Full-panel (incl. confidence + fusion) — FAIL, 18/20** (bar ≥19). The strict product claim allowed ≤1 non-deployable deployment and predicted exactly one (`gemma-3-4b/anli`); a second appeared, so it misses by one → the strict claim is **falsified** (the honest, registered outcome).

**Both endpoints fail the identical two ANLI deployments** (`gemma-3-4b/anli`, predicted; `Llama-3.1-8B/anli`, the one model with no prior ACE seal). Confidence and fusion rescued neither — coverage is 18/20 *with or without* confidence, so those two are genuine epistemic blind spots no panel signal covers (TriviaQA 10/10, ANLI 8/10).

**No universal best signal:** the 18 deployable deployments are won by **12 distinct signals** — ACE attention dominant, RPV (fisher_eff_rank / spectral_entropy / neg_shadow) winning 4 deployments where attention does not, and the pre-registered cross-locus fusion signal winning 2 outright. Corroboration *with* complementarity.

### Descriptive analyses (pre-registered, non-gating — `stage_b/universality.json`)

- **E1 — partial universality (first positive in the program).** Pooling 9 models to pick one fixed signal and testing on the held-out 10th, the cross-locus **fusion** signal clears the pre-registered ≥8/10 bar on both tasks (ANLI 9/10, TriviaQA 10/10 holdouts at AUROC > 0.55). No universal *champion*, but a universal **above-chance floor** — aggregation buys cross-model robustness. (Bar is 0.55 ≈ "beats chance"; strength is heterogeneous.)
- **E2 — task transfer.** Applying a model's per-task winner across tasks: median transfer AUROC **0.67**, above-floor on **85%** of transfers. Per-*model* calibration is a decent cross-task proxy.
- **E3 — label-efficiency** (registered: repeats=10, nboot=1000). Mean fraction of deployments deployable climbs **0.45 (n=50) → 0.67 (n=100) → 0.79 (n=150) → 0.90 (n=200)** (geometric; full-panel tracks 0.01–0.04 higher). The knee is ~n=100; n=50 is below a coin flip; standing up a new deployment costs **~150–200 labels** — affordable, not thousands.

The thesis, refined by these: *no universal best signal, but a fixed aggregate gives a universal above-chance floor; per-model calibration transfers across tasks ~85% of the time; full strength still needs per-deployment calibration at ~150–200 labels.*

## Post-seal extensions (do NOT enter or alter the sealed 18/20)

The paper's extension section asks whether the two sealed ANLI orphans are permanent blind
spots or capacity artifacts. Artifacts for the local runs are published here:

- **Scale + family axis (byte-comparable to the seal).** Pre-registered before any metric
  (`stage_b/PRE_REGISTRATION_EXT.md`, run via `stage_b/run_ext.py`; same data, seed, panel,
  selector, and module hashes as the seal). `gemma-3-12b-it` and `Qwen2.5-14B-Instruct`,
  both tasks, n=200: **4/4 deployable** (geometric OOB CI-lo — gemma-3-12b: ANLI 0.709,
  TriviaQA 0.929; Qwen2.5-14B: ANLI 0.766, TriviaQA 0.597). The sealed `gemma-3-4b/anli`
  orphan (0.403 FAIL) is recovered by scale; the Qwen-14B control rules out a generic
  12–14B effect. Matrices + profiles: `stage_b/profiles_ext/`.
- **Generation axis (`gemma-4-12B`, NOT byte-comparable).** The `gemma4_unified`
  architecture is unsupported by the sealed MLX stack, so extraction uses a reimplemented
  loader + attention recompute (`stage_b/gemma4_full_extract.py`, validated to o_proj
  cosine 1.0; build spec in `stage_b/GEMMA4_BUILD_SPEC.md`), scored by the same calibrator.
  **2/2 deployable** (ANLI 0.691, TriviaQA 0.751), both winners the cross-locus fusion
  signal. The orphan does not reappear a generation later. Matrices:
  `stage_b/profiles_ext/*/gemma-4-12B-it_FIXED.matrix.npz` (reported as standalone,
  never pooled with byte-comparable cells).
- **GPU / torch panel (30B–70B, NOT byte-comparable).** `modal/` holds the PyTorch
  extraction app used for the larger-model cells discussed in the paper (Qwen2.5-32B/72B,
  Llama-3.3-70B locus dissociation, and the precision-ladder deconfound), plus the exact
  uploaded data and MLX reference matrices used for its cross-implementation validation.
  See `modal/README.md` for the comparability caveats.

## Thesis

Every signal that has survived falsification in this research program is a
*curvature / spread reading of a categorical distribution somewhere on the
commitment pathway*. Three independent research lines walked into the same room:

| Stream | Signal | Organ | Timing | Needs |
|---|---|---|---|---|
| attention | **ACE** panel (js, bos_mass, v-norm, …) | attention routing | t=0, pre-generation | nothing (W_u-free, single pass) |
| residual  | **PRI** (v3 `null_ratio`) | residual-stream motion Δh | gen_step ≈ 1 | Δh + W_u |
| readout   | **RPV** (fisher_eff_rank) | readout geometry of the state | any t, Δh-free | W_u only |
| base      | **surprise / p_max** | the output distribution | every token | logits |

They *converge* — they do not subsume one another. The monitor treats them as a
**panel of specialists with one honest dispatcher**: a per-(model, exact deployment
distribution) `CalibrationProfile` (nested-OOB CIs, sign-lock, drift hashes,
deployability rails) picks the deployable signal without oracle knowledge.

## Honest constraint set (what the verdicts forbid)

1. No universal *champion* signal — per-(model, distribution) selection yields 12 distinct winners.
   (But E1 finds a universal above-chance *floor* in the fusion aggregate — see Results.)
2. Stacking gains are small — the signals overlap (corroborate), they don't add orthogonally.
3. But the overlap has holes, and the holes are covered (e.g. PRI dies on
   Qwen3-8B where RPV is alive).
4. The streams fire at different *times* — ACE exists before generation; PRI cannot.

## Plan

- **Stage A — union coverage matrix** (`stage_a/`): zero-new-compute read over sealed
  artifacts. For every (model, task) deployment, which families are OOB-deployable, and is
  the union gap-free (≥1 deployable family everywhere)?
- **Stage B — sealed unified-panel run** (gated on Stage A): fresh-seed pre-reg where the
  calibrator fits all families in one pass; registered claim = at least one panel signal is
  deployable in every deployment, and the dispatcher selects it without oracle knowledge.

## Source artifacts (read-only inputs; not vendored)

- ACE sealed profiles — `t0-morphology-furnace/experiments/t0-sealed/2026-05-26/profiles/`
- RPV comprehensive run — `t0-morphology-furnace/exploratory/shadow-ambiguity/comprehensive_outputs/`
- PRI (v3) — carried inside the RPV run as `null_ratio_post_rank1` (same deployments, same data)

This repo holds the *integration layer* only. It does not re-run or vendor the source
experiments; it reads their sealed outputs and composes them. (The single vendored
exception is the selection machinery in `sealed_selector.py` — a byte-identical,
hash-stamped copy kept so the published matrices are analyzable without the private
repo; see the Reproducibility note above.) The dependency-repo root is configurable via
the `CONFLUENCE_T0_REPO` environment variable (defaults to
`~/Documents/t0-morphology-furnace`); no absolute machine paths are committed.
