# commit-confluence

A pre-registered study of whether four families of **commit-moment** internal signals can be
unified into one calibrated detector, and whether that detector is **universal or must be
calibrated per deployment**.

**Companion paper:** *Decoder LLM Hallucination: No Universal Detector, but a Universal Floor —
A Pre-Registered Study of Commit-Moment Hallucination Monitoring Across Ten Language Models*
(M. S. R. Kitti, Furnace Research, June 2026). This repository is the paper's reproducibility
artifact: the pre-registration, the gated fresh data, the registered per-deployment score matrices
and profiles, and the analysis code. Citation metadata in [`CITATION.cff`](CITATION.cff);
code MIT, artifacts CC BY 4.0 ([`LICENSE`](LICENSE)).

---

## ⚠️ What this study measures — and what it does not

**Read this before citing any number below.** The construct is narrower than the word
"hallucination" suggests, and the paper is explicit about it (§Cohort: *"the hallucination analog
being the contradiction / wrong-answer class"*). Stating it plainly here:

**What the models are actually asked to do.** Every sealed task hands the model a **supplied
candidate** and asks for a YES/NO judgment at a single commit token:

- **ANLI R1** — a premise and a hypothesis; label *entailment* vs *contradiction*.
- **TriviaQA (paired)** — a question and a **candidate answer**; judge it *correct* vs *wrong*.

The detector then reads the model's internal geometry at the moment it commits to that YES/NO
token. So this is **candidate-answer correctness readout at verification time** — a discriminative
judgment about text placed in front of the model.

**It is therefore NOT** a detector of spontaneous hallucination in open-ended free generation. No
result here shows that these signals flag a model inventing a false citation mid-paragraph. That is
a plausible adjacent hypothesis; it is not what was tested.

**Scope of "universal."** The word is used in the paper's registered, narrow sense — *does one fixed
signal beat chance on a held-out model within this cohort* — and the cohort is:

| | |
|---|---|
| Models | **10**, open-weight, **1.7B–8B**, all **4-bit quantized**, MLX on Apple silicon |
| Families | Llama 3.1/3.2, Mistral 7B/Nemo, Phi 3.5/4, Qwen 2.5/3, Gemma 3 |
| Tasks | **2** (ANLI R1, TriviaQA paired) |
| n | 200 per deployment |
| Floor bar | AUROC > **0.55** — i.e. "beats chance," a deliberately modest bar |

No claim in this repository extends beyond that cohort and protocol. The larger-model evidence
(30B–70B) comes from a **different framework** and is explicitly non-byte-comparable — see below.

---

## Status at a glance

Five evidence tiers live in this repo. They have **different epistemic standing** and are never
pooled. This table is the map.

| Tier | What | Status | Byte-comparable to seal? | Artifacts |
|---|---|---|---|---|
| 🔒 **Sealed** | Registered run, 10 models × 2 tasks, seed 20260612 | **COMPLETE.** Geometric **18/20 PASS** (bar ≥17); full panel **18/20 FAIL** (bar ≥19) | — (is the seal) | `stage_b/profiles/`, `stage_b/PRE_REGISTRATION.md` |
| 📈 **Extension — scale/family** | `gemma-3-12b`, `Qwen2.5-14B` | **COMPLETE.** 4/4 deployable | ✅ yes | `stage_b/profiles_ext/`, `PRE_REGISTRATION_EXT.md` |
| 🧬 **Extension — generation** | `gemma-4-12B` | **COMPLETE.** 2/2 deployable | ❌ no (reimplemented extractor) | `stage_b/profiles_ext/*/gemma-4-12B-it_FIXED.matrix.npz` |
| ☁️ **GPU / torch panel** | Qwen 32B/72B, Llama-3.3-70B, precision ladder | **COMPLETE, exploratory.** Not a registered benchmark expansion | ❌ no (torch + bitsandbytes, NVIDIA) | `modal/` — see [`modal/PROVENANCE.md`](modal/PROVENANCE.md) |
| 🧪 **CC extension — BENCH** | 6 new tasks (ANLI R2, HaluEval ×3, replications) | **COMPLETE.** Strict Phase 4 closed 2026-07-22 (seed 20260711, nboot 2000, 53 profiles). **A1 PASS 10/10; A2 FAIL 6/10 ⇒ registered A1∧A2 conjunction NOT satisfied.** B1 7/20 via a pre-registered commitment cascade | ✅ intended | `stage_b/PRE_REGISTRATION_BENCH.md`, `stage_b/profiles_bench/` |

### 🧪 CC extension — BENCH: the registered Phase-4 verdict

The BENCH extension of commit-confluence ("CC extension") asked whether the sealed floor reaches
**HaluEval-QA**, under two endpoints that had to hold **together**:

| Endpoint | Question | Result |
|---|---|---|
| **A1** | Calibrated on its own labels, is each model deployable on halueval_qa? | **PASS 10/10** (bar ≥8); weakest cluster CI-lo 0.6705 |
| **A2** | Freeze cell `fusion_rank_mean_geom` + fit **one** sign on 9 models, apply blind to the 10th | **FAIL 6/10** (bar ≥8), `aborted=false` |

**The conjunction is not satisfied.** The floor extends to HaluEval-QA in per-model-calibration form;
the run did **not** support a cohort-wide **fixed-orientation** detector. (A2 did transfer on six of
ten holdouts — it missed the registered ≥8 bar, and the four misses are inversions, not near-misses.)

**A2 fails by sign inversion, not by absence of signal.** The four misses land far *below* 0.5
(Mistral-7B 0.174, Mistral-Nemo 0.206, Qwen2.5-7B 0.276, Phi-3.5 0.394): each independently selects
`+1` in its own calibration while all six passers select `−1`. Verified from the raw matrices — mean
fused rank faithful/hallucinated mirrors exactly (Llama-3.2-3B 0.62/0.38, high = faithful; Mistral-7B
0.37/0.63, high = hallucinated). Reversal is *not* a rescue for A2: knowing to reverse requires the
holdout's labels, which is precisely what A1 may use and A2 may not.

Restate this carefully. It establishes **no universal *orientation*** — and, with eight distinct A1
winners across ten models, no universal *best* cell. It does **not** show that no common informative
cell exists: `fusion_rank_mean_geom` with a per-model sign clears 0.55 on **all ten**. A2 rejects the
compound *"fixed cell + fixed sign"* deployment, not cell identity.

### ⚠️ B1 reads 7/20 — a pre-registered gate cascade, not a geometric collapse

Do not propagate B1 as a signal negative. `_endpoint_value` (`stage_b/run_bench.py:784`) zeroes every
cell of any task carrying ≥3 COMMITMENT-FAIL cells (§4 zero error budget × §8.1 systematic abort), so
all ten `triviaqa_paired_rep` cells — **including seven whose terminal status is OK** — are forced
False. The layers separate cleanly:

| Layer | Count |
|---|---|
| Raw geometry deployable | **18/18** (stem-cluster OOB CI-lo 0.6760–0.9804) |
| Pre-cascade admissibility | 14/20 |
| Post-§8.1 cascade (registered endpoint) | **7/20** |

Triggers are rare (Llama-3.1-8B 1/1000, Qwen3-1.7B 1/1000, gemma-3-4b 12/1000 — gemma sometimes
answers the trivia question instead of judging faithfulness). The `anli_r1_rep` behavioral fails are
the signatures Amendment A1 explicitly declined to rescue, and were pre-disclosed. Per §8 amendment
discipline the rule **cannot** be softened retroactively without a new registration. An A5-style
amendment (per-cell commitment error budget, widened acceptable-answer template, blip-vs-behavior
split) is **proposed, not filed**, and must be pre-registered blind for a future run.

### ✅ Resolved blockers (previously disclosed here)

- **A2 spec gate** — the estimator hardcoded `spec_version == "bench/1.2"` while Amendment A1 had
  bumped the spec to `bench/1.3`, so A2 could not read a single Phase-4 profile. Now imported from
  `bench_spec` (`stage_b/analyze_universality.py:35`); A2 executed against the strict summary.
- **E3 stem splitting** — label-efficiency subsampling split paired stems. `_e3_subsample` is now
  stem-aware for every matrix currently in this repo. ⚠️ Not a total fix: see the caveat below — a
  grouped task whose stem metadata is unrecoverable still falls back to row subsampling *silently*.
- **A1 manifest record** — the `EXTENSION_MANIFEST.json` attestation of `"unchanged": "…analysis…"`
  was literally true and consequentially false (leaving the file unedited is exactly what broke A2).
  The amendment corrected that record visibly rather than erasing it.

---

## Claim → artifact map

Every load-bearing claim, and the exact thing to open to check it.

| Claim | Verify with | From repo alone? |
|---|---|---|
| Geometric dispatcher deployable **18/20** (bar ≥17) → **PASS** | `python stage_b/verify_endpoints.py` | ✅ no models, no GPU |
| Full panel **18/20** (bar ≥19) → **FAIL**; strict claim falsified | same command | ✅ |
| Both endpoints fail the **same two** ANLI cells ⇒ confidence is not the backstop | same command | ✅ |
| **No universal champion** — 12 distinct winners across 18 deployable cells | `stage_b/profiles/*/*.json` (winner per cell) | ✅ |
| **E1** universal above-chance floor (fusion; ANLI 9/10, TriviaQA 10/10) | `python stage_b/analyze_universality.py` | ✅ deterministic |
| **E2** task transfer, median AUROC 0.67, 85% above floor | same command | ✅ deterministic |
| **E3** label cost **≥150** examples (largest budget measured) | same command | ⚠️ yes, but **see caveat** |
| Executed code byte-identical to the pre-registration | tag [`prereg-seal-20260612`](https://github.com/flowstyleliving/commit-confluence/releases/tag/prereg-seal-20260612); `module_hashes` in every profile | ✅ |
| Orphans close at scale; family-dependent signal locus | `modal/` + [`modal/PROVENANCE.md`](modal/PROVENANCE.md) | ❌ needs Modal account + GPU spend |
| **CC extension A1** deployable **10/10** on halueval_qa | `stage_b/profiles_bench/SUMMARY.json` → `endpoints.A1` | ✅ |
| **CC extension A2** transfer **6/10** (bar ≥8) → **FAIL**; conjunction not satisfied | `python stage_b/analyze_universality.py --profiles-dir stage_b/profiles_bench --bench-a2`; `A2_REGISTERED.json` | ✅ deterministic |
| A2's four misses are **sign inversions** (AUROC ≪ 0.5), not noise | per-holdout AUROC + `fitted_sign` in `A2_REGISTERED.json` | ✅ |
| **B1 7/20** is a §4×§8.1 gate cascade; raw geometry is 18/18 | `stage_b/run_bench.py:784` (`_endpoint_value`) vs per-cell CI-lo in the profiles | ✅ |
| Phase-4 tree matches its published checksums (53 matrices + 53 profiles + sidecars + lifecycle logs) | `python stage_b/verify_bench_provenance.py --manifest stage_b/profiles_bench/PROVENANCE.json --root stage_b/profiles_bench` (also in CI). Verifies **repository coherence, not an execution-time signature** | ✅ |

### Caveats on the above

- **TriviaQA inference is row-bootstrapped, not stem-clustered.** `sealed_selector.py:121` resamples
  rows independently, but TriviaQA is 100 question stems × 2 correlated rows, so the sealed TriviaQA
  CIs are **anti-conservative**. `PRE_REGISTRATION_BENCH.md` §366 already registers the stem-cluster
  bootstrap as the correct gate and declares the row-bootstrap result historical. An ad-hoc clustered
  re-check still returns 10/10 deployable, so the result appears robust — but a **registered**
  clustered sensitivity is owed, and is in flight.
- **E3 stem splitting — fixed in effect, not fixed by construction.** E3 previously subsampled rows,
  not stems, so on TriviaQA it split paired stems and optimistically biased the label-efficiency
  curve behind the "~150–200 labels" figure. `_e3_subsample` (`analyze_universality.py:341`) is now
  stem-aware and returns its grouping mode, and every grouped matrix in this repo does enter the stem
  path. **The residual defect:** when stem metadata cannot be recovered (`analyze_universality.py:66`
  returns `None`), the function fabricates unique row IDs — so the "uniqueness assertion" that fences
  the legacy path is then validating *its own fabrication*, and a grouped dataset with missing
  metadata re-enters row subsampling silently. The analyzer is hash-frozen and has already executed
  for A2, so this is recorded as a **future-version defect**, not patched post hoc. The sealed-era
  published figure was produced under the old path — treat the **sealed** number as provisional.

---

## Reproduce the registered results (no models needed)

```bash
pip install -r requirements-analysis.txt

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

# CC extension (BENCH) — the registered A2 transfer endpoint on halueval_qa. Reads the existing
# Phase-4 profiles and performs the pre-registered nine-model sign fit per holdout, never using
# holdout labels. Prints 6/10, bar 8, pass=False.
python stage_b/analyze_universality.py --profiles-dir stage_b/profiles_bench --bench-a2

# CC extension — reconstruct all 60 Phase-4 cell dispositions and rewrite SUMMARY.json. Validates
# the 53 stored profiles structurally (arrays, panel/stem digests, commitment tokens — it does NOT
# hash the NPZ or recompute endpoints) and reuses the 7 stored smoke records, so no model forward
# runs. On 2026-07-22 this produced no ERROR cells and reproduced SUMMARY.json byte-identically —
# see PROVENANCE.json and resume_reattest_2026-07-22.{exit,log}.
./confluence resume-bench
```

E1/E2 are deterministic given the matrices and reproduce `stage_b/universality.json` identically;
E3 and the endpoint verification are exactly reproducible at the registered seed/bootstrap settings.

## Standalone local setup (including fresh model forwards)

Fresh extraction requires Apple silicon/macOS and locally available MLX model snapshots. Setup
creates a repo-local virtual environment at the versions recorded by the BENCH parity attestation:

```bash
./confluence setup
./confluence doctor

# Analysis-only entrypoints
./confluence verify
./confluence analyze --profiles-dir stage_b/profiles --out /tmp/universality.json

# Fresh extraction entrypoints (pass the same arguments documented by each harness)
./confluence seal --help
./confluence bench --help
```

`CONFLUENCE_PYTHON=/path/to/python` selects an existing environment. The launcher sets
`CONFLUENCE_T0_REPO` to `vendor/t0_core` by default; overriding it is an explicit opt-in to a
different extraction core and may invalidate provenance guards. `setup` defaults to macOS's
`/usr/bin/python3` because the BENCH run is provenance-pinned to Python 3.9.6; set
`CONFLUENCE_SETUP_PYTHON` if that interpreter lives elsewhere. The non-byte-comparable Gemma-4
extension retains its separate newer `stage_b/setup_gemma4.sh` environment.

> **BENCH reproduction is now mostly portable.** Amendment A3 replaced absolute-path validation with
> **content resolution**: exclusion references are matched as a sha256 multiset and frozen Arrow
> sources are located by bytes (`resolve_frozen_arrow`, `stage_b/run_bench.py:101`; gate body
> `:950-1002`), with `CONFLUENCE_HF_CACHE` overriding the HF cache root. What remains is that the
> registered **bytes must be locally present** — and the HaluEval pinned raw file is still checked at
> its recorded path, so that one task is not yet machine-portable. The **sealed** analysis path above
> is unaffected and *is* portable.

## Results (registered run, seed 20260612 — 10 models × {ANLI R1, TriviaQA paired}, n=200)

**Terminology:** a *deployment* = one (model, task) pairing (20 total); a *signal* = one candidate
detector in the 29-entry panel (e.g. `attention[final_bos_mass] @ step 0`). The honest selector picks
one signal per deployment.

Clean run: 20/20 deployments computed, zero errors, all shuffled-label controls passed, registered
(not preview). Two pre-registered endpoints:

- **Geometric-only dispatcher — PASS, 18/20** (bar ≥17). A confidence-free panel (ACE attention +
  PRI + RPV) under the honest nested-OOB selector is deployable (OOB CI lower bound > 0.50) in 18 of
  20 deployments. The registered geometric claim **holds**.
- **Full-panel (incl. confidence + fusion) — FAIL, 18/20** (bar ≥19). The strict product claim
  allowed ≤1 non-deployable deployment and predicted exactly one (`gemma-3-4b/anli`); a second
  appeared, so it misses by one → the strict claim is **falsified** (the honest, registered outcome).

**Both endpoints fail the identical two ANLI deployments** (`gemma-3-4b/anli`, predicted;
`Llama-3.1-8B/anli`, the one model with no prior ACE seal). Confidence and fusion rescued neither —
coverage is 18/20 *with or without* confidence, so those two are genuine epistemic blind spots no
panel signal covers (TriviaQA 10/10, ANLI 8/10).

**No universal best signal:** the 18 deployable deployments are won by **12 distinct signals** — ACE
attention dominant, RPV (fisher_eff_rank / spectral_entropy / neg_shadow) winning 4 deployments where
attention does not, and the pre-registered cross-locus fusion signal winning 2 outright. Corroboration
*with* complementarity.

### Descriptive analyses (pre-registered, non-gating — `stage_b/universality.json`)

- **E1 — partial universality (first positive in the program).** Pooling 9 models to pick one fixed
  signal and testing on the held-out 10th, the cross-locus **fusion** signal clears the pre-registered
  ≥8/10 bar on both tasks (ANLI 9/10, TriviaQA 10/10 holdouts at AUROC > 0.55). No universal
  *champion*, but a universal **above-chance floor** — aggregation buys cross-model robustness.
- **E2 — task transfer.** Applying a model's per-task winner across tasks: median transfer AUROC
  **0.67**, above-floor on **85%** of transfers. Per-*model* calibration is a decent cross-task proxy.
- **E3 — label-efficiency** (registered: repeats=10, nboot=1000). Mean fraction of deployments
  deployable across the 20 sealed cells climbs **0.44 (n=50) → 0.66 (n=100) → 0.79 (n=150)**
  (geometric-only; full-panel 0.46 → 0.71 → 0.81). The curve is still climbing at the largest budget
  ever computed, so the earlier "~150–200 labels" headline **extrapolates past the data**: no `n=200`
  point exists in `stage_b/universality.json`, and `label_efficiency` defaults to `(50, 100, 150)`
  (`analyze_universality.py:374`). Read n=150 as a lower bound, not a knee.
  ⚠️ **Provisional** — these sealed-era figures were computed under the pre-amendment row-subsampling
  path; see the stem-splitting caveat above. The CC extension runs the stem-aware path.

The thesis, refined by these: *no universal best signal, but a fixed aggregate gives a universal
above-chance floor; per-model calibration transfers across tasks ~85% of the time; full strength still
needs per-deployment calibration at **≥150** labels (the largest budget measured; the curve had not
yet flattened).*

## Post-seal extensions (do NOT enter or alter the sealed 18/20)

The paper's extension section asks whether the two sealed ANLI orphans are permanent blind spots or
capacity artifacts.

- **Scale + family axis (byte-comparable to the seal).** Pre-registered before any metric
  (`stage_b/PRE_REGISTRATION_EXT.md`, run via `stage_b/run_ext.py`; same data, seed, panel, selector,
  and module hashes as the seal). `gemma-3-12b-it` and `Qwen2.5-14B-Instruct`, both tasks, n=200:
  **4/4 deployable** (geometric OOB CI-lo — gemma-3-12b: ANLI 0.709, TriviaQA 0.929; Qwen2.5-14B:
  ANLI 0.766, TriviaQA 0.597). The sealed `gemma-3-4b/anli` orphan (0.403 FAIL) is recovered by scale;
  the Qwen-14B control rules out a generic 12–14B effect. Matrices + profiles: `stage_b/profiles_ext/`.
- **Generation axis (`gemma-4-12B`, NOT byte-comparable).** The `gemma4_unified` architecture is
  unsupported by the sealed MLX stack, so extraction uses a reimplemented loader + attention recompute
  (`stage_b/gemma4_full_extract.py`, validated to o_proj cosine 1.0; build spec in
  `stage_b/GEMMA4_BUILD_SPEC.md`), scored by the same calibrator. **2/2 deployable** (ANLI 0.691,
  TriviaQA 0.751), both winners the cross-locus fusion signal. The orphan does not reappear a
  generation later.
- **GPU / torch panel (30B–70B, NOT byte-comparable).** `modal/` holds the **actual** PyTorch extractor
  that produced the larger-model cells discussed in the paper (Qwen2.5-32B/72B, Llama-3.3-70B locus
  dissociation, precision-ladder deconfound), together with the exact seal modules it mounted and the
  uploaded data / MLX reference matrices used for cross-implementation validation.
  ⚠️ **The calibrator vendored under `modal/seal/` is deliberately NOT the one at repo HEAD** — the GPU
  cells predate the BENCH amendment to `confluence_calibrator.py`. Repointing that mount at HEAD would
  silently fail to reproduce the published numbers. Read [`modal/PROVENANCE.md`](modal/PROVENANCE.md)
  before running or citing anything from this path. Running it needs a Modal account and GPU spend;
  **nothing in the registered analysis path depends on it.**

## Thesis

Every signal that has survived falsification in this research program is a *curvature / spread reading
of a categorical distribution somewhere on the commitment pathway*. Three independent research lines
walked into the same room:

| Stream | Signal | Organ | Timing | Needs |
|---|---|---|---|---|
| attention | **ACE** panel (js, bos_mass, v-norm, …) | attention routing | t=0, pre-generation | nothing (W_u-free, single pass) |
| residual  | **PRI** (v3 `null_ratio`) | residual-stream motion Δh | gen_step ≈ 1 | Δh + W_u |
| readout   | **RPV** (fisher_eff_rank) | readout geometry of the state | any t, Δh-free | W_u only |
| base      | **surprise / p_max** | the output distribution | every token | logits |

They *converge* — they do not subsume one another. The monitor treats them as a **panel of specialists
with one honest dispatcher**: a per-(model, exact deployment distribution) `CalibrationProfile`
(nested-OOB CIs, sign-lock, drift hashes, deployability rails) picks the deployable signal without
oracle knowledge.

## Limitations

Stated plainly, because a reviewer will find them anyway.

1. **Construct.** Candidate-answer correctness readout, not free-generation hallucination detection
   (see the scope box above).
2. **Cohort.** Ten models, 1.7B–8B, all 4-bit, one framework, two tasks. "Universal" means *within
   this cohort*.
3. **A modest bar.** The universality floor is AUROC > 0.55. It is a floor, not a performance claim;
   held-out strength spans 0.54–0.95.
4. **Two genuine blind spots.** The sealed orphans mean some (model, task, stack) deployments have
   **no** certified commit-moment signal under this panel. The protocol was built to be able to return
   that negative, and it did.
5. **Clustered inference** is owed on the *sealed* TriviaQA cells, and the sealed label-cost figure
   remains provisional — it was computed under row subsampling, and its upper bound was never
   measured at all. The CC extension gates A1 on cluster-geometric deployability and subsamples
   stem-aware, so the fix exists — it has not been back-applied to the seal.
6. **The large-model evidence is out-of-stack.** Different framework, different numerics, never pooled
   with the seal, not reproducible without GPU spend.
7. **No universal champion, and stacking gains are small.** The signals corroborate more than they add.
8. **The CC extension returned a split verdict.** BENCH strict Phase 4 is complete, and its registered
   A1∧A2 conjunction **failed**: per-model calibration reaches HaluEval-QA (A1 10/10), but a frozen
   cell + frozen sign does not transfer (A2 6/10), because orientation itself is model-dependent.
   Separately, B1's 7/20 is a pre-registered commitment-gate cascade over intact geometry (18/18
   deployable), not a signal collapse — and the rule was kept rather than softened after the fact.

## Historical source artifacts

- ACE sealed profiles — `t0-morphology-furnace/experiments/t0-sealed/2026-05-26/profiles/`
- RPV comprehensive run — `t0-morphology-furnace/exploratory/shadow-ambiguity/comprehensive_outputs/`
- PRI (v3) — carried inside the RPV run as `null_ratio_post_rank1` (same deployments, same data)

This repo does not vendor the source experiments or their full output trees; it reads their sealed
outputs and composes them. It does vendor the exact selection machinery (`sealed_selector.py`) and the
minimal fresh-extraction import closure (`vendor/t0_core/`). `./confluence` selects the vendored core;
direct script invocation can instead set `CONFLUENCE_T0_REPO` explicitly.
