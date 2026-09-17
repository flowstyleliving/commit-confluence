# RPV pass-one overlay pre-registration

Draft authored 2026-09-16. No execution or empirical acceptance is claimed.
This is a new measurement registration, not a revision of H1: H1 remains NO-GO.
Authority is the 2026-09-12 capture workorder, Amendments 1, 2 and 3 in full,
and the five adjudicated rulings in this task. The original prompt for
`task-mu4ndj88-ah5luj` was not recovered from the local records searched; this
specification is reconstructed from those authorities.

## Scope, frozen choices, and handoff

Codex authors only these two Python files and this document. Claude performs
future smoke execution and A1′–A4 verification. MK authorizes the full run.
Neither script is executed as part of authoring. No sealed/archive source or
`controls/` may be modified. Runtime JSON outputs use exclusive creation inside
`exploratory/rpv-pass1/`, outside `controls/`; imports disable bytecode writes.
The scripts must run as standalone single-threaded processes because their
observation hooks temporarily replace module attributes in memory.

The frozen metric-temperature grid is **T = [0.5, 0.7, 1.0, 1.4, 2.0]**,
**frozen by MK on 2026-09-16**. It is log-symmetric about 1.0 and bounded by a
measured degeneracy test: at T=0 the distribution collapses to one token, the
centered Fisher matrix `diag(p) - p p^T` becomes exactly zero, and every statistic
returns a constant for every row; above about T=2 the statistic saturates into the
K=512 truncation (effective support 295/512 at T=4) and measures the instrument
rather than the model. The authoring draft proposed [0.5, 0.75, 1.0, 1.5, 2.0]
as an unjustified assumption; that is superseded. T=1.0 is primary and the only temperature eligible
for confirmatory comparison. Every other value is descriptive, reported together
in grid order. Selecting or promoting the best-performing T is forbidden.

Two independent reasons require freezing the grid before the run: prevent
post-hoc temperature selection, and bank the full-vocabulary logsumexp at each
registered T while logits remain available. In general `lse(z/T) != lse(z)/T`.
A new T cannot recover confidence from a T=1 normalizer.

This is **metric temperature**: reweight the Fisher metric using softmax(z/T)
with fixed greedy decoding, committed token, prompts, and labels. Sampling at T
is out of scope; it would change tokens and labels and require a separate run.
Metric replay does require eigendecompositions and loading weights; zero extra
model forward passes does not imply zero CPU/GPU time or zero memory cost.

**Roster dependency — RESOLVED 2026-09-17.** The authoring draft noted that the
panel roster was unspecified and required a content-hashed manifest before freeze.
That manifest is `panel/PANEL_MANIFEST.json`, frozen 2026-09-16 **before the first
cell ran**. It derives the roster from the banked pass-two artifacts rather than a
hand-typed list, and records per cell the model ID, local weight snapshot, benchmark
JSONL and its SHA-256, row limit and seed, plus reference and overlay code hashes.
The registered panel is **26 cells / 13 models / 3,900 rows** -- not the 28/14/4,200
quoted in the workorder: `gpt-oss-20b` banks no rows (harness skip, too heavy for the
local MLX cache), so it contributes nothing and is excluded by the data, not by choice.
No cell was added or removed on the basis of any outcome.
The CLI takes one explicit cell and performs no automatic model discovery.
Defaults are a five-row smoke, seed 20260611, one greedy generated token.
The preserved smoke cell is Llama-3.2-3B-Instruct-4bit / anli_r1 / limit 5.
The missing roster is a registration dependency, not permission to invent cells.

## Measurement and comparator set

Pass one uses the last prompt position, the distribution selecting the committed
answer. Readout statistics use `prefix_probs[-1]`, passed to the support selector with
no renormalization -- exactly as the reference passes its pass-two `p_t`.
(An earlier draft claimed the reference applies an additional normalization
to its readout. It does not; corrected by steward audit 2026-09-16.) Each last-prefix late-block hidden
state goes through the same final RMSNorm and projection logit lens as the
reference. The aggregate is the arithmetic mean of readout plus the last
ceil(N/4) blocks, each with equal weight; no missing-block averaging is allowed.
Support is the same top-512 selector (or the full vocabulary if smaller).
The sealed spectrum/statistic functions are imported without changes:
`fisher_eff_rank` (primary), `neg_shadow_logvol_r1`, `spectral_entropy` (secondary).
Active eigenvalues satisfy lambda > 1e-12 * lambda_max; the inherited log-volume
rank-one removal and epsilon are unchanged. Raw log-volume is diagnostic only.

The complete comparator set is:

- Original readout `surprise`, taken unchanged from the committed-token trace.
- Original `null_ratio_post_rank1` (the workorder's null_ratio).
- Original pass-two RPV, with primary and secondary statistics reported separately.

**Null ratio spans both passes:** its h_prev is from pass one, while p_t and h_t
are from pass two. It cannot be moved into pass one alone. It stays at its
existing pass-two definition in both arms and is a comparator, never a treatment.
The overlay stores the inherited pass-two row unchanged, including this field.

At greedy readout T=1, p_max and exp(-surprise) are **equal to within an additive
1e-10 floor**, not exactly: runtime surprise is `-log(p + 1e-10)`.
Numerical paths add measurable rounding differences. p_max remains excluded as a
separate comparator because it double-counts confidence; p_max(T) remains a
confidence diagnostic, not an admitted comparator. Original surprise is not
silently replaced by temperature-specific surprise. At a late block, the actual
committed token need not be its argmax; its probability must not be called p_max.

Pass-one and pass-two measurements are reported side by side on identical paired
usable rows, never pooled as observations. Preserve all pass-two-only rows,
drop counters, pass-one failures, and pass-one-only indices for eligibility audit.
Do not infer a passing gate from an empty/degenerate cell or an unexplained drop.

## Registered prediction and analysis boundary

Predict stronger coupling of pass-one RPV to original surprise than for pass-two
RPV, because pass one contains the answer's own probability. The decisive
brittleness gate is the bootstrap upper CI for correlation with surprise >= 0.75.
Apply the gate to primary aggregate fisher_eff_rank in every frozen cell; a
failure in any cell prevents a general claim. Report both secondary correlations
without rescuing a failed primary by selection.

Where the workorder is silent, use the inherited Pearson correlation and paired
row bootstrap with 4,000 replicates, seed 20260611, percentile 95% intervals.
**REGISTERED 2026-09-17:** these were authoring assumptions and are now frozen
choices. They may not be varied after seeing any result; any change requires a new
registration, not an amendment. Compare the two positions with the
same resampled row indices and report the paired difference in correlations.
Undefined correlations/CIs are inconclusive, never passing. Report each cell;
no pooling positions or selecting cells. The gate refers to the signed Pearson
coefficient as written; no unregistered absolute-correlation substitution.

Capture and temperature scripts perform no label-performance selection or
confirmatory analysis. A separate frozen analysis plan is required before
adding predictive performance claims, model fitting, or multiplicity decisions
beyond these correlations. Do not invoke the old harness's broad `analyze_rows`
routine unfiltered: it includes p_max comparator sets excluded here.

## Bank and replay contract

Each paired row contains sample_idx, label, raw `prompt_sha256` (SHA-256 of UTF-8
prompt text returned by the unchanged JSONL loader), `wrapped_prompt_sha256`,
the unchanged `pass_two` row, complete prefix source bank, source-level and
aggregate statistics for both computation paths, and the pinned layer window.
Artifact metadata hashes dataset, three authoring files, reference spectrum and
runtime code, and all local safetensors/config JSON files. Model-family tokenizer
settings are retained even when loading by a local path. Offline replay rejects
weight/config or reference-hash mismatch. A changed environment still requires
numerical A4 verification; hashes alone do not prove reproducibility.

For every source, bank ordered support_idx as int32, support_logits rounded to
float32, and full_vocab_lse as float64 at **every** frozen T. Bank max_logit and
the actual committed_token_logit as well: the latter can be outside a late
block's support and is necessary for its full-vocabulary surprise. These types
are established before JSON serialization and reinstated on replay. JSON is a
portable authoring choice, not a packed binary bank: on-disk size exceeds the
workorder's approximate 134 MB packed-array budget. No projection rows, full
vocabulary vectors, or hidden states are serialized.

**Projection geometry is re-derived offline:** load the same model weights,
construct its projection, and call `projection.get_rows(support_idx)`.
This is weight loading and selected-row dequantization, **not a forward pass**.
Do not bank W_s: its estimated 211 GB panel cost is unnecessary. The offline tool
never calls the model, trace_sample, generation, or projection.project.
It may load the full model object to use the inherited projection abstraction.

Replay normalizes support logits at T and calls fc_full_spectrum, which itself
normalizes support probabilities. Thus the three statistics require no full-vocab
normalizer. p_max(T) uses exp(max_logit/T - banked_lse[T]); token surprise uses
-log(exp(committed_token_logit/T - banked_lse[T]) + 1e-10).

Exact positive-temperature ranking is invariant, but numerical ties at the
support cutoff need care. Preserve the original T=1 support order and set at all
T, including tied logits; never reselect support offline. Capture rejects a
probability-selected support that excludes a strictly larger logit due to
rounding/underflow. Such failures require investigation, not a silent selector
change. This fixes geometry consistently across temperature, including ties.

## Acceptance and numerical resolution

**A1′:** the original `trace_pair_features` executes with observation hooks only.
`pass_two_rows` retains all inherited fields. With `--a1-reference`, compare all
rows recursively to the committed-script smoke artifact; floating-point hex
representations must agree, including signed zero. Report mismatches explicitly.
No supplied reference means pending, not passed. The prior 325-value control
establishes committed-code equivalence to banked artifacts; it does not by itself
prove this new overlay. The future executor must also demonstrate that a copied
reference perturbed by one ULP is detected, outside `controls/`, and match model,
benchmark, input hash, seed and limits. Do not alter the preserved control.

**A2:** each prefix trace must contain every block in the pinned late window.
The code records row failures without changing original pass-two eligibility;
acceptance requires investigating every such failure per model. Partial source
aggregates are forbidden.

**A3:** freeze this document, scripts, exact panel manifest, and resolution
justification before the authorized full run and before outcome analysis.
No data or measured resolution have been produced during authoring.

**A4:** the run writes `offline_t1` through the exact function used in offline
replay. Offline replay compares reconstructed T=1 to banked offline_t1, like
with like. Separately report measured maximum absolute discrepancies for every
source/statistic and aggregate between inline_t1 and offline_t1. Also report
full_softmax_t1 versus offline_t1: direct source inspection shows the readout
uses float32 safe_softmax, whereas the logit-lens inline path uses float64
softmax_np. This is an additional numerical path beyond the workorder's
full-vocabulary-versus-support-only distinction. Neither is assumed identical.

The code reports the minimum positive between-row separation for each
source/statistic and a candidate tolerance equal to one tenth of that separation.
This conservative margin is frozen independently of observed discrepancy: two
errors below it cannot reverse the smallest measured distinct separation.
A constant/single-row source has no measurable separation and cannot pass.
The executor must justify, on a designated smoke calibration sample before label
analysis, why that measured separation is the smallest scientifically meaningful
difference; document the measured separation, maximum error, ratio, and any
threshold-crossing sensitivity. Use the frozen calibrated tolerances on the full
run; full-run candidate values are audit diagnostics, not replacement thresholds.
The report intentionally marks acceptance pending until that justification exists.
Never enlarge a tolerance to make failures pass. Active-set discontinuities may
produce nontrivial differences requiring a failed/inconclusive A4, not rounding
away. Bit-identity is not the A4 requirement. Repeat this resolution comparison
for offline_t1 versus fresh offline replay.

## Future invocation interface (not executed)

Run `rpv_pass1_run.py` with explicit `--model-id`, `--model-path`, `--benchmark`,
`--data`, `--limit`, `--seed`, `--a1-reference` for smoke, and `--out` naming a new
JSON artifact inside this overlay directory. Then run `rpv_pass1_temperature.py`
with `--bank`, the identical `--model-path`, and a distinct `--out`. All grid
values are emitted together. Full-run authorization remains with MK; authoring
these interfaces does not imply smoke, A1′–A4, or the full run has happened.


---

# FROZEN — 2026-09-17

**Status: FROZEN. A3 satisfied.** Label-linked analysis may now proceed under this document and no other.
Any change from here requires a **new** registration, not an edit.

## What is frozen

- The measurement (pass one at the last prompt position), the pinned aggregate, the top-512 support.
- The temperature grid **T = {0.5, 0.7, 1.0, 1.4, 2.0}**, with **T = 1.0 the sole confirmatory value**.
- The comparator set: original `surprise`, original `null_ratio_post_rank1`, original pass-two RPV.
  `p_max` is **excluded**; `p_max(T)` is a diagnostic, never an admitted comparator.
- The registered prediction: pass-one RPV couples to `surprise` **more** tightly than pass-two RPV does.
- The decisive brittleness gate: bootstrap **upper** CI of the correlation with `surprise` **>= 0.75**,
  applied to primary aggregate `fisher_eff_rank` in **every** cell; a failure in any cell prevents a general
  claim, and a failed primary may not be rescued by a secondary.
- Analysis parameters: Pearson, paired row bootstrap, 4,000 replicates, seed 20260611, 95% percentile
  intervals, shared resampled indices across positions.
- The roster: `panel/PANEL_MANIFEST.json`, 26 cells / 13 models / 3,900 rows.

## ⚠️ Disclosure — the data was captured before this freeze

The 3,900 rows were captured on **2026-09-16**; this document is frozen on **2026-09-17**. That ordering is a
genuine weakness and is stated rather than hidden. What limits it:

- 🔒 **The capture is label-blind by construction.** `rpv_pass1_run.py` performs no label-linked computation;
  it carries the inherited `label` field through untouched and never reads it.
- 👁️ **What the steward has seen:** per-cell feature distributions (pass-one vs pass-two `fisher_eff_rank`
  medians and ranges) and the A4 numerical diagnostics.
- 🚫 **What the steward has NOT computed, at all:** any relationship between any feature and any label, and
  **any correlation with `surprise`** -- which is the registered endpoint and the brittleness gate's input.
  The gate is therefore still genuinely blind at the moment of freezing.
- ⛔ **Consequently the roster could not have been outcome-selected:** it was fixed by a manifest written
  before the first cell ran, and no outcome existed to select on.

A reader who regards pre-data freezing as the only acceptable standard should treat everything downstream as
**descriptive**. Nothing here can revise **H1**, which was registered on pass-two features; the two positions
are reported side by side and are **never pooled**.

## Frozen-document identity

The SHA-256 of this file as frozen is recorded in `PREREG_FREEZE.txt` alongside the manifest hash, so any
later edit is detectable.
