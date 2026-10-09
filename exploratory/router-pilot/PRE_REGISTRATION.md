# PRE-REGISTRATION (DRAFT v2 — NOT FROZEN) — Router-uncertainty depth pilot on one MoE model

**Status: DRAFT v2, revised after the round-1 design audit (Codex `gpt-6-astra`, RED; all 7 MAJOR
and 3 MINOR findings accepted).** Becomes binding at FREEZE-1 (§9).

**Question.** In a mixture-of-experts model, does the router's uncertainty at the **final prompt
position** separate the frozen task labels, at which depth, and how does it compare held-out with
the attention instruments and with output-level confidence read from the same forward pass?

**Estimand (fixed before any smoke).** Discrimination of the dataset labels by features read at the
final prompt position — the position whose next-token distribution is the model's answer. It does
**not** require the model to answer YES/NO. YES/NO compliance is observed behavior, reported
descriptively, and is **not** a gate. (Chosen before smoke so a non-compliant small model cannot
trigger a post-hoc change of estimand.)

**Kind of claim.** Prospective, descriptive, single-model pilot. **No confirmatory endpoint, no
CONFIRM/WEAKEN/FALSIFY vocabulary, no placement-law or cross-model claim, no claim that router
information is conditionally incremental over attention or output confidence.** The only decision
this run makes is a **spending decision** (§7). The labels are dataset labels (answer supported /
premise entails), not the model's own errors.

**Comparability.** Torch on Modal; non-byte-comparable with sealed MLX panels; never pooled with
depth grids A or B (different precision, hardware, model class).

## 1. Model (one cell per task; two cells total)

| field | value |
|---|---|
| HF id | `allenai/OLMoE-1B-7B-0924-Instruct` (ungated; no HF secret used) |
| revision | `7f1c97f440f06ce36705e4f2b843edb5925f4498` (no-mirror rule: unavailable ⇒ NOT-RUN) |
| class | `OlmoeForCausalLM`, `model.eval()`, `torch.no_grad()` |
| layers / query heads / KV heads / head_dim | 16 / 16 / 16 (MHA) / 128 — asserted at load: `head_dim = hidden_size/16 = v_proj.out_features/16` |
| experts / active | 64 / top-8 every block; `norm_topk_prob = false` — asserted |
| `clip_qkv` | must be `None` (else values are clamped after the v_proj hook) — asserted, else abort |
| precision | native bf16 weights and compute; **no quantization anywhere** (parameter dtype histogram recorded; any non-bf16 decoder parameter ⇒ abort) |
| attention | `attn_implementation="eager"`, `output_attentions=True`, `output_router_logits=True` |
| hardware | 1 × NVIDIA A10G (24 GB), Modal personal workspace; full GPU residency asserted |
| environment | independent Modal app and new volume; exact pins: Python 3.12, `transformers==4.57.6` (the source the auditor read), a compatible `torch` CUDA wheel, `accelerate`, `huggingface_hub<1.0` (transformers 4.57.6 requires `>=0.34,<1.0`), `safetensors`, `numpy`; image digest and versions recorded. Import + load verified at smoke. |

## 2. Data (frozen; identical files to the depth grids)

- `anli_r1_n200.jsonl` sha256 `57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b`
- `halueval_qa_n200.jsonl` sha256 `a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0`
- Both 100/100 by label. **Groups:** HaluEval by `meta.source_idx` (100 pairs, one label-1 and one
  label-0 row each); ANLI by exact premise string (5 pairs, 190 singletons). Asserted at load.
  Labels are read only by the scorer.

## 3. Serialization

Render with the native chat template as shipped (`tokenize=False`, `add_generation_prompt=True`,
whitespace preserved), then tokenize the text with `add_special_tokens=False`, **and** cross-check
equality with `apply_chat_template(..., tokenize=True)`. Assert tokenizer BOS = EOS = id 50279 and
that 50279 occurs **only** at index 0. No padding, no truncation; length ≤ 900. All 400 rows are
tokenized and validated **before any GPU allocation**. Smoke writes a per-task prompt-token manifest
(sha256 per row and of the rendered suffix); extraction re-derives every row and fails closed on
mismatch. (Pre-smoke local evidence with the pinned tokenizer: max lengths 243 / 297 tokens; one
50279 per prompt, at index 0.)

## 4. Capture (one t=0 forward per prompt, batch 1, `use_cache=False`)

All features are read at the **final prompt position** `T−1`.

**Attention instruments — 6 metrics × 16 blocks, all mandatory** (any undefined or non-finite
value aborts the cell; the additive v-norm channel's swallowed-failure paths become aborts):
`final_js`, `final_js_no_bos`, `final_js_kv_groups`, `final_bos_mass` (sealed kernel
`_compute_attention_score`, unmodified) and `final_v_norm_max`, `final_v_norm_lastq_weighted`
(the sealed kernel's v-norm reductions, unmodified). Semantics:
`bos_mass` = attention mass on position 0 and `js_no_bos` drops position 0 — neither is "all
end-of-text attention", although BOS and EOS share id 50279.
- **Selectable ATT metrics (5):** `final_js`, `final_js_no_bos`, `final_bos_mass`,
  `final_v_norm_max`, `final_v_norm_lastq_weighted` ⇒ **80 columns**.
- **Recorded, not selectable:** `final_js_kv_groups` — theoretically equivalent to `final_js`
  under MHA but implemented with different EPS ordering; max |difference| reported, never an
  abort.
- **Deprecated, not captured:** `final_v_norm_bos`. It reads only V at position 0, which under
  causal attention depends only on the shared first token, so it is **structurally constant**
  across prompts (banked grid evidence: 3–7 distinct values per block over 200 rows, relative range
  ≤ 3%). Its residual variation is an implementation artifact.

**Sink draw — 1 metric × 16 blocks, descriptive diagnostic only** (replacement for the deprecated
`final_v_norm_bos`; new and unregistered): `sink_draw` = mean over heads `h` of
`A_h[T−1, 0] · ‖V_h[0]‖` — how much the final position draws from the position-0 sink, attention
times the sink's value size (MHA: query head `h` uses KV head `h`). It varies across prompts
through the attention term. **Never selectable for `att_best`, never a comparator, never part of the
§7 rule.**

**Router instruments — 3 metrics × 16 blocks = 48 columns**, from `router_logits[ℓ][T−1]` (64
logits; the final prompt token's post-attention normalized hidden state through `mlp.gate`),
softmax in fp32 → `p`:
- `router_entropy_norm` = `H(p) / log 64`
- `router_margin` = `p(8) − p(9)` (sorted descending; the top-k boundary)
- `router_top8_mass_precast` = sum of the 8 largest fp32 `p` (the module later casts the applied
  weights to bf16)

Provenance only (never scored): selected expert IDs and fp32 weights at `T−1` per block.

**Output instruments — 3 scalars:** `out_p_max` (max next-token probability), `out_entropy`
(full-vocabulary next-token entropy), `out_yesno_logodds` = `log Σ_{t∈YES} p_t − log Σ_{t∈NO} p_t`,
where YES / NO are the frozen token-id sets whose `decode(skip_special_tokens=True).strip().upper()`
is exactly `"YES"` / `"NO"` (sets computed from the pinned tokenizer and recorded in the manifest).
**Length:** `prompt_len` = T (separate comparator, exposes length shortcuts).

Also recorded per row: next-token argmax id and a `yes_no` flag (descriptive compliance).

## 5. Gates (any failure aborts; aborted cells are reported, never rescued)

Smoke (labels never read; never computes an AUROC) on rows 0–9 plus the longest row per task:
- all §1 load assertions; residency; dtype histogram; module coverage (16 attention + 16 MoE
  blocks hooked);
- every hook fires **exactly once per block per forward**; tensor shapes, dtypes and devices as
  expected;
- **o_proj reconstruction** (existing gate) at **every** block on rows 0–1 of each task:
  cosine ≥ 0.999 **and** relative L2 error ≤ 1e-2 (cosine alone misses scale errors);
- **MoE reconstruction** at **every** block on rows 0–1 of each task: from the captured MLP input
  at `T−1`, recompute the MLP output using *our* top-8 selection from `router_logits`, the
  module's own weight cast to bf16 and its experts' own forward; compare with the captured MLP
  output: cosine ≥ 0.999 **and** relative L2 error ≤ 1e-2. This verifies that the router features
  describe the routing actually applied;
- **router-output neutrality:** next-token logits at `T−1` with `output_router_logits` on vs off:
  identical argmax and max |Δlogit| ≤ 1e-3;
- all 6 + 1 + 3 + 3 features finite on every smoke row;
- determinism: three forwards of the longest row give identical top-8 expert IDs at every block and
  feature values within 1e-5;
- memory: peak reserved ≤ 20 GiB, growth < 0.5 GiB over the three repeats;
- timing: `load + 400 × p95_forward × 1.2` fits inside the remaining budget (§8).

Full extraction: 200/200 rows per task, every feature finite. YES/NO compliance reported, not gated.

## 6. Analysis (scorer runs ONCE, after both cells reach a terminal status)

- **Folds:** 5 stem-grouped folds per task, frozen from a declared seed tag (pairs never split;
  20/20 labels per fold). Labels from the local frozen files.
  Use the imported `rescore_grid_a._rng(task, "router-pilot-folds-v1")` and
  `make_group_folds`; freeze the resulting map by this algorithm before observing features.
  Bootstrap/null use the independent tags `router-pilot-bootstrap-v1` /
  `router-pilot-null-v1` with the same imported RNG. Record maps and their hashes.
- **Columns:** ATT 80; ROUTER 48; OUTPUT 3; LENGTH 1; SINK 16 (descriptive).
- **Per fold, training rows only:** orientation per column (raw AUROC ≥ 0.5 → +1 else −1); within
  each of ATT, ROUTER and OUTPUT select the column with the highest training sign-free AUROC
  (ties: declared metric order, then the shallower block). Held-out AUROC uses the locked
  orientation on raw oriented scores.
  Metric tie order is the order listed in §4 within each selectable family; blocks are
  zero-based 0–15. Comparator ties use `att_best`, `out_best`, `prompt_len`, in that order.
- **Arms (held-out AUROC = mean over 5 folds):** `router_best`, `att_best`, `out_best`,
  `prompt_len`. Descriptive arms: each output scalar alone, and `sink_best` (training-only
  selection over the 16 sink-draw columns; outside the comparator maximum).
- **Best comparator** = the highest held-out arm among {`att_best`, `out_best`, `prompt_len`} —
  a maximum over comparators, conservative for the router and upward-selected as an estimate of
  comparator performance.
- **Bootstrap (conditional on the frozen folds):** NBOOT 1000 stem-cluster resamples within the
  original folds — pairs resampled among that fold's pairs, singletons within their label stratum
  — with draws shared across arms; orientation, all three selections and the comparator maximum
  re-run inside every resample. 5/95 CIs by numpy's default (linear) percentile; degenerate
  replicates excluded and counted. ANLI's one pair per fold cannot vary.
- **Null (descriptive):** 200 group-respecting permutations — pairs swap labels with probability
  ½, singletons are permuted **within their fold** — with the full orientation and selection
  re-run per permutation; report the null distributions of `router_best`, `att_best` and the
  router − comparator difference. Assumes conditional exchangeability within pairs.
- **Descriptive:** per-block training sign-free AUROC curves for all 8 selectable instrument
  metrics and `sink_draw`; per-fold selected (metric, block) for each family; selection
  dispersion; YES/NO compliance; `final_js_kv_groups` − `final_js` max difference; per-block
  Spearman correlation of `sink_draw` with `final_bos_mass` across prompts (is the sink draw more
  than a reweighted `bos_mass`?).
- **Limits stated up front:** intervals are pointwise, with no simultaneous guarantee across
  depth; n = 200 (100 HaluEval stems) cannot establish that a three-point advantage is reliable.

## 7. Spending rule (the only decision; not significance, not confirmation)

**PROCEED** (further MoE compute may be proposed) only if, on **both** tasks, `router_best`
held-out AUROC ≥ 0.65 **and** `router_best − best_comparator` ≥ +0.03. Otherwise **STOP**. Either
outcome is reported as written.

## 8. Budget and terminal states

Hard cap **2 GPU-hours cumulative** across every launch — smoke, extraction, retries and reloads —
tracked in a ledger, with 10 minutes reserved for shutdown. A gate abort or reaching the cap stops
the pilot: every cell not yet terminal is written `NOT-RUN` with the reason, and the scorer reports
those cells as not run (no scoring of them). No larger GPU, no prompt change, no rescue. Terminal
statuses are immutable.

The ledger uses a persistent, atomic cross-launch claim and append-only invocation events.
Charge launch-to-return wall time conservatively (including loading), plus shutdown allowance;
reserve the entire allowed invocation duration before allocation. Unclosed reservations are
charged in full, never erased or retried. CPU preflight validates all 400 prompts before the
GPU function is invoked. GPU function timeouts and an internal absolute deadline derive from
the remaining ledger budget, less the 600-second reserve. After a killed controller, the next
entry point reconciles its expired reservation and terminalizes remaining cells without GPU work.
Terminal states are `ok` (complete, hashed NPZ) and `NOT-RUN` (including partial/aborted cells).

## 9. Freeze ordering and discipline

- **FREEZE-1 (before smoke):** `FREEZE.json` records the sha256 of this file, the extractor and the
  scorer, and is logged before smoke launches. Gate definitions and thresholds cannot change after
  it.
- **Smoke** produces no outcome-bearing quantity. A *mechanical* defect it reveals (crash, shape,
  import) may be fixed only by a dated amendment section here plus a new FREEZE entry, before any
  re-smoke, within the budget.
- **FREEZE-2 (before extraction):** adds the sha256 of both smoke prompt manifests.
- The scorer refuses to run twice, writes atomically, and surfaces no observed value before
  completion. Misses are reported as written.

### Round-2 implementation clarifications (2026-10-08; before FREEZE-1)

`FREEZE.json` is an append-only history published by atomic replacement: unique stage entries
`1`, then `2`, each with UTC timestamp and file hashes. Stage 2 preserves stage 1 and adds both
smoke manifests. Imported scoring/sealed sources and the freeze helper are hashed too.
A dated mechanical amendment can add a unique `1-amend-<date-tag>` entry only before stage 2
and before successful smoke or any terminal cell. ~~A GPU invocation failure is a terminal abort
under §8; it cannot be reclassified as a repairable mechanical defect. Mechanical import failures
before GPU reservation may be amended and retried; serialization, data or pinned-download failures
are terminal even on CPU.~~ *(Superseded by the round-3 clarification below.)* No existing stage or terminal status is replaced.
The scorer claims its single look before reading outcomes; interrupted publication is not a rerun.
If either task is NOT-RUN, report STOP (incomplete pilot), scoring only any complete `ok` cell.
Smoke checks label-free stem multiplicities; the scorer alone asserts pair label balance.
Image provenance records the Modal image ID and SHA256 of its frozen build recipe (Modal does
not expose an OCI digest through the inspected public SDK); package versions are recorded at smoke.

### Round-3 clarifications (2026-10-08; before FREEZE-1; steward decision)

**Smoke-stage failures.** Smoke reads no labels and produces no outcome-bearing quantity, so
repairing a crash there leaks nothing; the depth lane likewise landed mechanical fixes before its
freeze. Therefore, before FREEZE-2:

- Every smoke-stage failure writes an immutable `smoke.failed.<freeze-entry>.json`
  (exception type, message, traceback, `gate_failure` flag). A mechanical failure alone does
  not write `NOT-RUN`; a gate failure does, as specified below.
- A **measured-gate failure** — any §5 threshold, the §3 serialization checks, the §2 data and
  stem checks, all §1 hardware/configuration/architecture/dtype/environment facts, a zero
  reconstruction reference, deadline/budget exhaustion, or an unavailable pinned revision —
  raises a distinct `GateFailure`, is **terminal**, writes `NOT-RUN` for both cells, and can never
  be amended. Instrumentation shape/programming errors and import errors remain mechanical.
- Any other smoke-stage exception is a **mechanical defect**. Re-smoke requires a dated amendment
  section in this file and a new FREEZE entry `1-amend-<YYYY-MM-DD-tag>`, inside the cumulative
  2 GPU-hour cap. One smoke attempt per freeze entry. `make_freeze.py` refuses an amendment that
  changes sections 1–8 (`design_sha256`), follows a gate failure, or follows a successful smoke.
- After FREEZE-2 every failure is terminal, unchanged.

An immutable `smoke.attempt.<freeze-entry>.json` is committed before smoke preflight; its presence
refuses another attempt under that entry, even without a failure record. All work after claim
acquisition is protected, including reload, accounting and publication. One failure publisher
handles smoke and extraction; extraction failures terminalize every non-terminal cell even before
GPU reservation. The coordinator releases its owned claim only after durable ledger and result or
failure state. Storage failure retains the claim for expired-claim reconciliation; unresolved
reconciliation storage failures require executor inspection, not another GPU attempt.

The worker returns gate failures in a structured envelope, which the coordinator re-raises as
`GateFailure`. Classification follows both exception-chain links, including Hub repository/revision
errors wrapped by transformers and `GATE:` embedded in Modal deserialization fallback messages.

**Budget accounting.** When the coordinator observes a GPU call's end — success or failure — it
charges launch-to-return time (queue and load included) plus 30 s. Only an unclosed reservation
(killed coordinator) is charged in full, at reconciliation. The interval is measured immediately
around `.remote()`; artifact publication and storage commits add no GPU charge. The post-call
deadline/budget check runs on both success and exception paths, before successful publication;
exhaustion takes terminal precedence over any concurrent mechanical failure.

**Weights.** The CPU coordinator first performs the protected download of the pinned snapshot,
then loads the tokenizer from that local cache with `local_files_only=True`, before any GPU is
reserved. The GPU worker also loads with `local_files_only=True`.

**Descriptive publication.** Per-block Spearman is a value/reason record: zero rank variance in
either input yields JSON `null` and `undefined: constant input`. Diagnostic errors or non-finite
results (including subtraction overflow) are reported as null with a reason and cannot veto the
single-look spending result. This does not change selection, comparator membership or thresholds.

**`v_norm_bos` deprecated; `sink_draw` added (MK decision, 2026-10-08, before FREEZE-1).** The
structural constancy above makes `final_v_norm_bos` uninformative by construction, so it is no longer
captured. `sink_draw` is its descriptive replacement, as defined in §4; it cannot affect the §7
decision.
