# PRE-REGISTRATION — Depth-grid expansion (grid B): held-out-model confirmation on two fixed benchmarks

**Frozen at commit time (this file's introducing commit = the freeze).** Drafted
2026-08-17 after: the registered grid-A run (`PRE_REGISTRATION.md`, results @
`768ea7e`), a four-round adversarial Codex gpt-5.6 dialogue (plan RED → spec →
code YELLOW → verified), and the grid-A rescore under the exact estimators below
(`RESCORE_GRID_A.json` @ `60d4507`: E5 8/8, Δ_cf 0.111–0.416; E6 7/8, sole miss
structural). Grid A is **discovery**; grid B is **prospective held-out-model
confirmation on ANLI R1 and HaluEval-QA** — never task-general. Llama-3.1 cells
are family-seen/model-unseen (Llama-3.3-70B was in discovery).

## 1. Grid (12 core cells; 405B stretch OUTSIDE all denominators)

| # | model (slug) | HF id @ pinned revision | wrapper | N layers / heads / KV | precision | GPU |
|---|---|---|---|---|---|---|
| 1 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` | LlamaForCausalLM | 32 / 32 / 8 | nf4 | 1×A100-80 |
| 2 | Llama-3.1-70B-Instruct | `meta-llama/Llama-3.1-70B-Instruct` @ `1605565b47bb9346c5515c34102e054115b4f98b` | LlamaForCausalLM | 80 / 64 / 8 | nf4 | 1×A100-80 |
| 3 | Mistral-Small-3.2-24B | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` @ `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` | Mistral3ForConditionalGeneration | 40 / 32 / 8 | nf4 (decoder) | 1×A100-80 |
| 4 | Mistral-Medium-3.5-128B | `mistralai/Mistral-Medium-3.5-128B` @ `22b2b868a15677cfa6061277ed2f653d1349a9ab` | Mistral3ForConditionalGeneration | 88 / 96 / 8 | **FP8-origin → deterministic dequant → BF16 compute** | 4×A100-80 (fallback 2×H200, same BF16 compute) |
| 5 | gemma-3-12b-it | `google/gemma-3-12b-it` @ `96b6f1eccf38110c56df3a15bffe176da04bfd80` | Gemma3 (text tower) | 48 / 16 / 8 | nf4 | 1×A100-80 |
| 6 | gemma-3-27b-it | `google/gemma-3-27b-it` @ `005ad3404e59d6023443cb575daa05336842228a` | Gemma3 (text tower) | 62 / 32 / 16 | nf4 | 1×A100-80 |
| — | Llama-3.1-405B-Instruct (STRETCH, descriptive-only) | `meta-llama/Llama-3.1-405B-Instruct` @ `be673f326cab4cd22ccfef76109faf68e41aa5f1` | LlamaForCausalLM | 126 / 128 / 8 | nf4 | 8×A100-80 default; 4× only via §7 smoke gate | 

× tasks {anli_r1, halueval_qa} ⇒ **12 core cells**. The 405B pair runs only on
MK's explicit go, is reported descriptively, and enters **no** confirmatory count.

**No-mirror rule:** if a pinned revision becomes unavailable, the cell is reported
as NOT-RUN; substituting a mirror or a different revision requires a new dated
amendment naming it as a different registered model. Expected layer counts are
verified at load; mismatch ⇒ hard abort of that cell.

## 2. Data (frozen, identical to grid A)

Same two frozen n=200 jsonl files, sha256-enforced in-extractor:
- anli_r1 `57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b`
- halueval_qa `a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0`
Both are 100/100 by class; rows 0..199; labels identical across models within a task.

## 3. Instrument (ONE capture mode; identical to grid A except the registry deltas)

Per prompt: one t=0 forward, eager attention, **full retention of all blocks'
attention maps** (no streaming variant anywhere), last-row read per block, the four
sealed metric cells computed per block by the sealed kernel
(`_compute_attention_score`), primary `final_js_no_bos`. `DEPTH_MAX_TOKENS = 900`.
Chat serialization via each model's native template through one shared **decoder
descriptor** that resolves, for wrapper checkpoints, the text tower
(`language_model`) for layers, o_proj hooks, head/KV counts, and provenance alike.
**Mistral tokenization rule (both Mistral cells):** the AutoTokenizer chat-template
ids must MATCH the `mistral-common` encoding of the same single-user-turn request
on all 200 rows × both tasks at smoke — any mismatch aborts the cell (two
implementations must agree before the template is trusted). Prompts are plain text
single-turn (no reasoning-mode system prompt is added for Medium 3.5; the template
is used as shipped). BOS: position 0 must be exactly one BOS token per model
(verified at smoke); `*_no_bos` semantics unchanged.

**Terminal-status rule:** every attempted cell writes `<slug>.status.json`
(`ok` | `aborted` + reason) on the volume; the scorer refuses to run unless all 12
core cells have terminal statuses, and `aborted` cells enter §5 as failures.
**Terminal states are immutable:** the extractor refuses to run any (model, task)
whose status file or output npz already exists — a rerun that would rescue an
`aborted` cell requires a new dated amendment, never a silent relaunch. (A cell
killed by infrastructure BEFORE writing any terminal status may be relaunched —
only terminal states bind.)

**Manifest pinning:** smoke writes per-(model, task) prompt-token manifests; the
freeze commit embeds each manifest file's sha256 in the extractor
(`FROZEN_MANIFEST_SHA256`; sentinel values until then make extraction structurally
impossible). Extraction validates the volume manifest's bytes against the frozen
hash AND its identity fields (model, revision, task, schema, 200 rows) AND
re-derives all 200 rows live — three independent fail-closed checks. Post-freeze
smoke cannot replace a frozen manifest.

**Medium method freeze:** the dequant method AND GPU shape smoke selects are
frozen into the extractor (`FROZEN_MEDIUM_DEQUANT_METHOD`,
`FROZEN_MEDIUM_GPU`) at the freeze commit; extraction refuses to run the Medium
cells while either is unset, aborts if the live loader used any other method, and
aborts if launched on any other GPU shape. All models: CPU/disk offload is
rejected (full-GPU residency asserted); **decoder quantizer classes are ENFORCED**
on the descriptor-resolved decoder blocks (every 2-D-weight leaf module is
bitsandbytes `Linear4bit` for nf4 modes; plain bf16 `nn.Linear` for the Medium
dequant mode — vision towers, deliberately unquantized, are outside the walk);
actual GPU names and full decoder-linear/parameter-dtype histograms are recorded
per cell. Gate cosines are enforced on RAW values, never rounded ones.

**Recovered-status rule:** a complete (atomically-written) npz with NO status file
is the kill-in-write-window state; the launcher recovers the `ok` status WITHOUT
re-extraction (the scorer fully re-validates every npz regardless). This is
bookkeeping recovery, not a rescue; it can never resurrect an `aborted` cell.

**Zero-evaluable-task rule:** the scorer derives labels and fold designs from the
LOCAL frozen data files (sha256-verified), not from artifacts, so the §5 verdict
is issued even if one or both tasks have zero evaluable cells (all such cells
count as failures; npz labels are cross-validated against the frozen data).

**Gates (all identical to grid A; any row failure aborts the cell):** o_proj
reconstruction cos ≥ 0.999 (rows 0–1), YES/NO commit ≥ 0.5, zero dropped rows,
frozen-data sha enforcement, finite scores. Gate-aborted cells are **counted as
confirmatory failures** (§5) and reported with their gate evidence.

**Precision:** nf4 (bitsandbytes, lm_head/embed/vision-tower unquantized) for all
cells except Medium 3.5, which is registered as *"FP8-origin weights,
deterministically dequantized; BF16 compute"* — NOT a BF16 reference checkpoint;
the Qwen nf4↔bf16 precision-ladder invariance does not cover it; both Medium cells
carry the flag, and §5 registers a leave-Medium-out sensitivity summary.

**Prospective pinning:** the Modal image pins exact package versions (recorded in
the extractor source and image digest); npz meta records model revision (must equal
§1), wrapper class, device map summary, capture mode, actual quantizer module
types, attention dtype, per-GPU peak memory, library versions, and prompt-token
sha256 per row. At smoke (gates-only), per-model prompt-token hashes over all 200
rows are written to a manifest; the confirmatory extraction fails closed if its
prompt tokens do not reproduce that manifest.

## 4. Statistics (frozen by reference; scorer runs ONCE)

The confirmatory machinery is **exactly** the grid-A rescore implementation —
`rescore_grid_a.py` @ commit `60d4507` (sha256 recorded in the scorer) — with the
model list generalized: stratified fixed 5-fold per task (same RS_SEED=20260817 ⇒
**identical fold maps to the calibration**), training-only directions +
qualification (training-rows-only envelope, NPERM_INNER=200, q97.5, AUROC ≥ 0.65)
+ peak selection; held-out contrasts with locked directions; within-fold
synchronized label permutations (NPERM_OUTER=2000) shared across all six models
per task; (1+k)/(B+1) p-values; method="higher" quantiles; stratified bootstrap
(NBOOT=1000) with all-defined aggregates; decisions D1–D9 as documented in that
file. The grid-B scorer (`score_grid_b.py`) imports these functions unmodified;
its only additions are the §5 denominators/guards/decisions, the §6 descriptive
outputs, the single-look guard (refuses to run if outputs exist; atomic writes; no
observed value surfaced before completion), and provenance.

## 5. Confirmatory endpoints (denominator = the 12 core cells, always)

Missing, gate-aborted, or no-qualifying-peak cells count as **failures**; an
evaluable-cells-only sensitivity recount is reported alongside (never a verdict).
The three grid-B families are Llama-3.1 {8B,70B}×2, Mistral×2 models, Gemma×2 —
4 cells each.

- **E5 (PRIMARY) — cross-fitted terminal dip.** Cell success: Δ_cf ≥ 0.05 with all
  five folds qualifying. Decision:
  - **CONFIRM** = ≥10/12 successes AND ≥3/4 in each of the three families AND
    pooled p_grid < 0.05 (success-count statistic, synchronized within-task
    permutations, tasks independent, pooled by permutation index).
  - **WEAKEN** = 8–9/12, or ≥10/12 with a family guard or p_grid failure.
  - **FALSIFY** = ≤7/12.
- **E6 (SECONDARY, gatekept) — cross-fitted directional cliff.** Tested
  confirmatorily ONLY if E5 = CONFIRM; otherwise **NOT TESTED — gate closed**
  (reported descriptively). Cell success: J_cf ≥ 0.15 AND J_cf > the cell's q95
  full-procedure permutation null (plug-in) AND R_cf > 0 AND J_cf ≥ 0.5·R_cf.
  **Early-peak cells (any fold with training peak < ceil(0.5N)+1) are E6-undefined
  and count as failures** (grid-A precedent: Llama-3.3-70B/halueval). Decision:
  - **CONFIRM** = ≥9/12 AND ≥4/6 per task AND ≥2/4 per family AND p_grid < 0.05.
  - **WEAKEN** = exactly 8/12, or ≥9/12 with any guard failure.
  - **FALSIFY** = ≤7/12.
- **Leave-Medium-out sensitivity (registered, non-verdict):** E5/E6 counts over the
  10 non-Medium cells, reported next to the 12-cell verdicts.

Grid-A discovery rates under these exact estimators (calibration, not part of any
grid-B count): E5 8/8; E6 7/8.

## 6. Descriptive-registered outputs (no verdict vocabulary)

- **E7 — cross-task peak distance.** Per model: peak_cf per cell = median of the 5
  training-fold peaks; report |peak_cf(anli) − peak_cf(halueval)| in blocks and as
  a fraction of N, with bootstrap intervals from the shared resamples. Frozen case
  handling: both cells defined → distance; one/none defined → reported as
  UNDEFINED with the reason; no agreement/disagreement labels.
- **E8 — Llama cells in context.** 3.1-8B and 3.1-70B full curves reported beside
  the banked 3.3-70B curve (same-size version comparison; base-checkpoint identity
  NOT established). Report qualifying-block sets and mid-region [0.4N, 0.9N]
  counts. No "replicates/stable/emergent" language.
- **E1″ — placement panel.** Peak-fraction (ℓ*_cf/N) vs N across all 10 models
  (12 core + 8 grid-A cells plotted separately, never pooled numerically),
  model/family-clustered spread. No law/non-law claims.
- Registered predictions, reported as written, non-gating: **P6** E5 CONFIRMs;
  **P7** E6 reaches ≥9/12 if tested; **P8** Llama-3.1-70B/anli shows ≥10
  qualifying training blocks in [0.4N, 0.9N] (majority of folds); **P9**
  Llama-3.1-8B/anli shows <5 such blocks (guess: band absent at 8B); **P10** the
  E1″ panel yields no obvious transferable placement rule.

## 7. Compute + staging

Staged detached launch (server-side, kill-proof, per-cell logs, `vol.commit()`
banking): stage 1 = 8B/12B/24B/27B (8 cells), stage 2 = 3.1-70B (2 cells), stage
3 = Medium-3.5 (2 cells). **No scorer look until all 12 bank.** The Medium load
path is probed at smoke (gates-only): deterministic dequant to BF16 on 4×A100-80;
if infeasible, the registered fallback is 2×H200 with dequant-at-load and
identical BF16 compute + capture (hardware recorded; still FP8-origin-flagged).
405B (if MK approves): 8×A100-80 default; 4× permitted only if ALL nine frozen
smoke conditions hold (no offload; complete device map; verified Linear4bit nf4 +
skip-list match; per-GPU peak ≤66 GiB alloc / ≤70 GiB reserved / ≥10 GiB headroom;
<1 GiB reserved growth over 3 max-length forwards; BF16 attention dtype + exact
shapes; o_proj cos ≥ 0.999; timing model `load + 200 × p95 × 1.2` inside timeout
and cost cap; ≥12 h timeout); any miss ⇒ 8× or cancel; no 4× retry after
outcome-bearing extraction begins.

## 8. Discipline

**Freeze ordering:** gates-only smoke runs BEFORE the freeze commit (it produces
no outcome-bearing quantity — gates, tokenization checks, prompt manifests, and
the Medium load-path probe only), so mechanical fixes it surfaces land pre-freeze.
The freeze commit then pins this file + `modal_depth_b.py` + `score_grid_b.py` +
the smoke manifests together; **no outcome-bearing extraction starts before that
commit exists**, and no endpoint/threshold/guard may change after it.

Smoke is gates-only (never prints a per-block AUROC). The scorer runs ONCE, after
all 12 core cells reach a terminal status; no grid-B AUROC is inspected before that. Thresholds,
guards, and denominators in this file cannot change after the freeze; misses are
reported as written; any amendment is a new dated section that cannot rescue this
run. Grid-B cells are never pooled numerically with sealed MLX cells or with
grid-A cells; grid-A rates are cited as discovery only. This registration claims
held-out-model confirmation on two fixed benchmarks — nothing task-general, no
causal lineage/post-training claims, no factorial family×scale claims.
