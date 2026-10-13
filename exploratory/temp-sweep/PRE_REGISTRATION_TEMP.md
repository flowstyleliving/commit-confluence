# Furnace / PRI commit-confluence temperature-robustness study

**Status: v1.3 draft, NOT frozen. Pending targeted delta-verification and MK sign-off before any run.**

## Revision history

| Version | Date | Status | Change |
|---|---:|---|---|
| v1.0 draft | 2026-08-26 | Superseded draft; never frozen or signed | Initial temperature-study specification. |
| v1.1 draft | 2026-08-26 | Superseded draft; never frozen or signed | Full restructure responding to round-1 adversarial findings D1-D11. |
| v1.2 draft | 2026-08-26 | Superseded draft; never frozen or signed | Full rewrite responding to round-2 findings N1-N9: late-window aggregate loci/T rule, evaluable-family verdict, selector-equivalence route and compute estimate, historical T=1 arithmetic, ANLI gate relabeling, unconditional pilot cells, exact attribution endpoints/settings, and manifest/formula corrections. |
| v1.3 draft | 2026-08-26 | NOT frozen; unsigned | Targeted round-3 corrections N9/M1-M7: exact random primitives, non-historical arithmetic bindings, complete discontinuity/environment/interpretation disclosures, per-source audits, and full compute disclosure. |

This is a documentation/spec-authoring artifact. It records intended work before any temperature extraction, sampling, calibration, test, or inference run. No validity gate or study result is represented as having been evaluated. This draft becomes frozen only through a later revision after adversarial audit, completion of the freeze manifest, and MK sign-off.

The structural template and amendment discipline are inherited from `stage_b/PRE_REGISTRATION_BENCH.md`, SHA-256 `c4b732b3f3df5607287e9d582ec33b8a86d40cd345340c13fb427a466654c974`.

---

## 1. Scientific question, named endpoints, and falsification target

### 1.1 Scientific question

Does the sealed single-token commit-step hallucination detector retain deployability when its internal probability distributions are temperature-rescaled, and—separately—what happens when the commit token is genuinely sampled rather than selected by argmax?

Softmax temperature is a monotone rescaling of logits. For every positive T, `argmax(z/T) = argmax(z)`, including the lowest-token-ID resolution of an exact tie inherited from `numpy.argmax`. This licenses two distinct endpoints:

1. **Endpoint (A), reshape-only — PRIMARY CONFIRMATORY FALSIFICATION ENDPOINT.** The commit remains greedy. Prefix and post-commit probability-dependent metrics are recomputed at the registered analysis temperatures while the token and hidden/attention traces are held fixed. Section 6 specifies the task-dependent identity gates.
2. **Endpoint (B), real-sampling — PILOT.** The commit is sampled from the temperature-scaled prefix distribution. The sampled token, post-commit hidden state/logits, correctness, and readout metrics may vary by draw. Endpoint (B) is descriptive and must never be described using promotion, universality, confirmation, or production-readiness language.

Endpoint (B) is gatekept behind Endpoint (A): pilot inference cannot begin until Endpoint (A) has completed its binding extraction, gate ledger, and single-look scoring, including all FAILED cells in the denominator.

**Interpretation limitation.** The unchanged sealed detector contains no temperature parameter, so under greedy decoding its outputs are T-invariant by construction. Endpoint (A) instead evaluates a counterfactual detector in which the analysis T is applied to the prefix distribution, the post-commit readout distribution, and every pinned late-block logit-lens distribution; the logit-lens distributions are internal detector constructions, not sampling distributions. Accordingly, “deployability” here means robustness of this counterfactual temperature-rescaled detector, not behavior of the deployed-unchanged sealed implementation.

### 1.2 H-TEMP-1: flat-or-monotone single-token temperature response

**Directional prediction.** In this one-token forced-choice regime, commit error rate and detector AUROC should be flat or monotone across T, not U-shaped. The U-shaped curve discussed in code-generation literature is attributed to greedy-decoding repetition-loop degeneracy in multi-token free generation; that mechanism is structurally impossible in a one-token commit.

For Endpoint (A), the greedy commit-outcome vector is fixed across T by construction, so its canonical-NO rate must be flat and is descriptive. The confirmatory falsification family is the full-denominator deployability-AUROC reversal family in section 7. Endpoint (B) reports the analogous sampled correctness-error and two named detector reversals descriptively.

### 1.3 Scope exclusions

Only `anli_r1` and `halueval_qa` enter this study. `triviaqa_paired` is excluded because the BENCH-lineage B1 gate-cascade entanglement and paired-row-bootstrap validity issue would confound this temperature analysis.

The ANLI behavioral threshold is informed by existing same-template BENCH evidence rather than assumed perfect canonicality. In `stage_b/profiles_bench/SMOKE_SUMMARY.json`, `anli_r1_rep` yielded 7/16 canonical commits for Qwen2.5-7B (nine token-2014 `" To"` commits), 12/16 for Phi-3.5-mini (four token-13 newline commits), and 16/16 for Qwen3-8B. The full Qwen3-8B `anli_r1_rep` profile recorded 996/1,000 canonical. These are different prompt draws from the registered `anli_r1` cohort, so they inform the 95% rule in section 6.3 but do not predetermine any TEMP cell.

A poisoned-context / adversarial-hallucination probe comparing contaminated and clean grounding is explicitly OUT OF SCOPE: it is a different mechanistic question with its own comparator design and belongs in a separate future pre-registration.

---

## 2. Frozen-on-sign-off cohort, prompts, seeds, held-fixed machinery, and loci

This is the cohort/seed/machinery block that will become frozen only at sign-off. In v1.3 it is complete but neither frozen nor signed.

### 2.1 Endpoint-(A) cohort

| Order | Exact model ID | Resolved revision |
|---:|---|---|
| 1 | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` |
| 2 | `mlx-community/Llama-3.1-8B-Instruct-4bit` | `90215b22ec18e72f623dde2ea7af4097025160e2` |
| 3 | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `a4b8f870474b0eb527f466a03fbc187830d271f5` |
| 4 | `mlx-community/Mistral-Nemo-Instruct-2407-4bit` | `647ca0751669b21a364c86ccc5df54c4d7e4e91c` |
| 5 | `mlx-community/Phi-3.5-mini-instruct-4bit` | `7b2052fd882fe017300d4d42a4eb06a27b816af4` |
| 6 | `mlx-community/Phi-4-mini-instruct-4bit` | `ac1c269cb4222a4e136a3d09edad301056c1f36a` |
| 7 | `mlx-community/Qwen2.5-7B-Instruct-4bit` | `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed` |
| 8 | `mlx-community/Qwen3-1.7B-4bit` | `3b1b1768f8f8cf8351c712464f906e86c2b8269e` |
| 9 | `mlx-community/Qwen3-8B-4bit` | `545dc4251c05440727734bcd94334791f6ab0192` |
| 10 | `mlx-community/gemma-3-4b-it-4bit` | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |

All ten are the sealed `mlx-community` 4-bit cohort. Model order is normative.

### 2.2 Endpoint-(B) pilot cohort

The pilot cohort is exactly four models, using the revisions above:

1. `mlx-community/Qwen3-8B-4bit` — Qwen structure.
2. `mlx-community/Llama-3.1-8B-Instruct-4bit` — Llama structure.
3. `mlx-community/Phi-3.5-mini-instruct-4bit` — low-literal-decidedness stress model.
4. `mlx-community/Mistral-7B-Instruct-v0.3-4bit` — fourth architecture family.

The Step-0 result reported literal-form coverage `0.185` for Phi-3.5-mini. This motivates its inclusion only; section 5.5 uses wider A1-prefix buckets, so TEMP coverage values are not numerically comparable to `0.185`.

### 2.3 Existing prompt draws, labels, and clustering

Both endpoints reuse the sealed/BENCH prompt draws rather than drawing fresh data. This explicitly trades data freshness for T=0 comparability.

| Task | Exact source | SHA-256 | Rows / inference clusters | Selector seed |
|---|---|---|---:|---:|
| `anli_r1` | `stage_b/data/anli_R1_seed20260612_n200.jsonl` | `57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b` | 200 rows / 200 prompt clusters | `20260612` |
| `halueval_qa` | `stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl` | `460019e12d25c33e9f842f52bb0e3fd70274256290dc7784101ebaa95b2163e7` | 1,000 rows / 500 `stem_id` clusters | `20260711` |

`sample_idx` is zero-based source-row order. `labels` in the historical matrices are copies of source JSONL `label`: `0` means the expected canonical answer is YES, and `1` means the expected canonical answer is NO. They are ground truth and are never treated as a record of what the model committed.

For HaluEval, stems are ordered by first row occurrence. Every resample carries both rows of a stem together. For ANLI, each prompt is its own cluster.

### 2.4 Prompt and commitment contract

`mlx-community/Mistral-Nemo-Instruct-2407-4bit` uses the tokenizer chat template. The other nine models use raw prompt passthrough. `max_new_tokens=1` is a registered pin for both endpoints; no free generation or second generated token is allowed.

For a greedy commit, choose `argmax(z_prefix)`; an exact tie resolves to the lowest token ID. Decode the single token and normalize in this exact order: left-strip whitespace, Unicode case-fold, right-strip whitespace, then repeatedly strip trailing ASCII/Unicode punctuation and right-strip again. A nonempty normalized prefix of `yes` is a canonical-YES commit; a nonempty normalized prefix of `no` is a canonical-NO commit. All other commits are non-canonical.

Changing this normalizer or canonical-prefix rule is a scientific-design change under section 10, not an implementation correction.

### 2.5 Loci: sealed semantics retained

The following loci are load-bearing and registered for freeze at sign-off:

| Cell family | Locus and required trace |
|---|---|
| 21 ACE attention cells | `t=0` / prefix-last attention capture from the prefix trace. |
| `surprise` | Prefix distribution before the committed token is appended, evaluated at the actual greedy/sampled token. |
| `p_max`, `null_ratio_post_rank1` | `gen_step=1` readout after the committed token is appended. `p_max` uses the post-commit readout distribution. Null ratio uses that distribution, final-layer `h_t`, prefix-last `h_prev`, and surprise. |
| `fisher_eff_rank`, `spectral_entropy`, `neg_shadow_logvol_r1` | `gen_step=1` late-window aggregate: arithmetic mean over the readout source and `ceil(B/4)` pinned late-block logit-lens sources. For B blocks, use zero-based blocks `B-ceil(B/4)` through `B-1`. Each lens source uses post-commit hidden `h_l`, `logits_l=W_u·rmsnorm(h_l)`, and its probability distribution. Every source uses top-512 support. No per-layer `h_prev` enters a lens source. |
| NEW `plain_confidence`, NEW `plain_entropy`, decidedness | Prefix distribution at the registered analysis/sampling T. |

Evidence: `vendor/t0_core/pri_runtime.py:985-1013` takes the first token from prefix logits, records prefix surprise, appends the token, and performs the updated-context forward; `pri_runtime.py:1114-1118` stores the post-commit logits/probabilities. `vendor/t0_core/exploratory/shadow-ambiguity/comprehensive_run.py:345-377` consumes readout probability/hidden state; `:404-455` captures pinned late blocks and constructs logit-lens sources; `:465-490` requires every source and persists their mean; and `:528` states that shadow statistics are the mean over late-window blocks plus readout. ACE capture is a separate prefix attention trace at `confluence_calibrator.py:542-560`. The v1.3 extractor may combine instrumentation, but it may not move or simplify a metric while retaining its sealed key.

### 2.6 Temperature grids and random streams

Endpoint (A) positive analysis grid is ordered `{0.5, 0.7, 1.0, 1.5, 2.0}`, plus the greedy `T=0` operating-point alias. `T=0` never means division by zero: after the T=1 parity sentinel is satisfied, the `T=0` column is an exact alias of the fresh `T=1.0` score matrix. Thus the six reported operating points contain five distinct probability distributions, and T=0/T=1.0 coincide by construction.

Endpoint (B) sampling anchors are ordered `{0.7, 1.0, 1.5}` with exactly `K=5` draws per prompt/T. Base seed is `20260826`. Before pilot inference, one `numpy.random.RandomState(20260826)` generates uniforms in nested order: section-2.2 model; task `anli_r1`, then `halueval_qa`; T `0.7, 1.0, 1.5`; ascending `sample_idx`; draw `0..4`. Sampling is inverse CDF over ascending token ID: `token=np.searchsorted(np.cumsum(q_T),u,side="right")`, clipped to `V-1`. Thus if floating-point cumulative mass ends below `u`, the selected ID is exactly `V-1`. The uniform array is written and hashed before pilot model calls.

### 2.7 Held-fixed summary

The held-fixed pins are: exact cohorts/revisions; data paths/hashes; prompt strategies; `max_new_tokens=1`; commitment normalizer; loci in section 2.5; positive T grids and T=0 alias; `K=5`; all seeds; exact 31-key panel/order; selector `nboot=2000`; prompt/stem resampling; label and non-canonical disposition; decidedness rules; fusion orientations; denominator and failure rules; phase order; and module/artifact hashes recorded in the freeze manifest.

Numeric algorithm settings registered for freeze are: `k_support=512` independently for the readout and every late-block source; null-ratio `topk_values=[32]`, `lowrank_values=[32]`, `v3_rank_values=[1]`, `alpha=1.0`; late-window size `ceil(B/4)` with block range `B-ceil(B/4)..B-1`; and aggregate source count exactly `1+ceil(B/4)`. The literal 512/[32]/[32]/[1]/1.0 settings are verified at `comprehensive_run.py:57,371-374`; the late-window rule is verified at `:198-210`.

### 2.8 Artifact isolation

All new artifacts go under fresh `stage_b/profiles_temp/`. They are never pooled with `stage_b/profiles/`, `stage_b/profiles_ext/`, `stage_b/profiles_bench/`, or torch-cloud cells. Existing sealed, EXT, and BENCH files are read-only and never modified.

---

## 3. Independent defect verification, persistence constraint, and provenance

### 3.1 D1 independently verified: labels are ground truth, not commitments

`confluence_calibrator.py:82-105` reads source JSONL `label`; `:121-140` places row `label` into the readout matrix and asserts it equals source metadata; `:160-173` repeats that assertion for fresh extraction and separately persists `gen_token_ids`. Therefore matrix `labels` cannot serve as a commit-identity reference. Direct reading of the Qwen3-8B HaluEval `.npy` members independently found token ID `14080` on 893/1,000 rows, including 393 rows whose source/matrix label is `1`; a reproduction of those commitments plainly cannot equal ground truth row-by-row. v1.0's comparison of fresh canonical outcome to `labels` was semantically invalid and is deleted.

### 3.2 D2/N1 independently verified: post-commit readout and late-window aggregate

The source evidence in section 2.5 confirms three distinct computations: prefix ACE/surprise; final readout `p_max`/null ratio; and the aggregate shadow statistics. `comprehensive_run.py:198-210` fixes the late-window rule; `:404-455` derives a logit-lens source from every pinned block; `:465-490` averages those sources with readout and persists the aggregate; `:491-497` stores readout-only shadow statistics separately, outside the matrix fields; and `:528` names the aggregate locus. Thus `fisher_eff_rank`, `spectral_entropy`, and `neg_shadow_logvol_r1` are not readout-only cells.

Endpoint (A) permits at most one prefix trace plus one post-commit step forward per row, capturing final readout state and all pinned late-block hiddens, with zero additional forwards per T. Endpoint (B) permits one prefix trace and one such post-commit capture per unique sampled token ID within each prompt/T, capped at five. Duplicate sampled token IDs reuse the capture. No cell is moved, simplified, or renamed silently.

### 3.3 N2 independently verified: a total-family verdict would be unreachable in practice

The registered BENCH smoke artifact `stage_b/profiles_bench/SMOKE_SUMMARY.json` (SHA-256 `ab5ca5612893d4902734cac55defbbb6028b8e141176c4042b09d70ed2cb8f9e`) records, on the same ANLI prompt template, Qwen2.5-7B at 7/16 canonical with nine token-2014 `" To"` commits, Phi-3.5-mini at 12/16 canonical with four token-13 newline commits, and Qwen3-8B at 16/16 canonical. The full Qwen3-8B ANLI profile `stage_b/profiles_bench/anli_r1_rep/Qwen3-8B-4bit.profile.json` (SHA-256 `0d026a01f4cb2ee4d3e12ecec793ce10fd9516333bebe219cecf04d1062f6825`) records 996/1,000 canonical. Those source facts support retaining every failure in the 120-cell ledger while evaluating H-TEMP-1 only on a pre-registered minimum complete family, as section 7 specifies.

### 3.4 Verified persistence constraint

Direct inspection of `stage_b/profiles_ext/triviaqa_paired/Qwen2.5-14B-Instruct-4bit.matrix.npz` found exactly `score_matrix (200,27)`, `labels (200,)`, `sample_idx (200,)`, scalar `panel`, and scalar `meta`. No logits or hidden states are stored. Historical ANLI matrices have the same five members. HaluEval BENCH matrices additionally have `stem_ids` and `gen_token_ids`, but no logits or hidden states.

Endpoint (A) therefore requires one fresh extraction sweep per model/task. Each row uses at most one prefix trace plus one appended-token forward capturing the full pinned late window, then emits all five T transformations from those captured logits/tensors. Cost is one extraction sweep, not one sweep per temperature.

### 3.5 Inspected baseline hashes

| Artifact | SHA-256 |
|---|---|
| `confluence_calibrator.py` | `c79009a3adaf57c6ad38d2400c3c705e12d00ab18d917399b59d56b23d513e9a` |
| `sealed_selector.py` | `1883abc7ea6d6ed97b13dde40355e2e3bcea0fed14f0f3eb2eb9e194a3bc51d7` |
| `stage_b/analyze_universality.py` | `000e8d017024e645272e5218601131c97acfd1a4991b9c95836a191ac7374396` |
| `vendor/t0_core/pri_v2_io_plugins.py` | `6c56be1888abc6ad44150583b869bac06366fa70a7d8e13230da3703064eb7b7` |
| `stage_b/run_seal.py` | `214b4580bad5fb3e16f7fa218f3f2387c8db14001e7272cc73ab0c9259c0e4ef` |
| `stage_b/generate_fresh_data.py` | `74e688ace1519a18eceb6f39991a23578ec12b325bfbef9b34beb9681b5f0b6e` |
| `stage_b/run_bench.py` | `4e6ae7dc5fadd1003fb10529e2c26f8a94ec429700d814dc69af5d8121f0dec6` |
| `stage_b/fusion_signs.json` | `92b5468bd241b517dd2d5cf70ad28556157424deb54b5f85f88af0305ff35372` |
| `vendor/t0_core/pri_runtime.py` | `cf56a2607b94666ab092647a764b1b73c7974b38445caa6a4b252a9cdf240784` |
| `vendor/t0_core/exploratory/shadow-ambiguity/comprehensive_run.py` | `f6f5958bae5b035f07f034b9f4f5784b1a86c1ab00ab0f9c9ef4a5584776e8f8` |
| `vendor/t0_core/pri_calibrator.py` | `78c4f098295fe600cc4f6f1a14cc7b496ac93d8d70dd65743c1504eb20931101` |
| `vendor/t0_core/pri_v2_mlx_pipeline.py` | `abb0debd8b7b3f0503418e09a21050fad968d88a6ea10fd4412983c6ba6e4a5c` |
| `vendor/t0_core/model_adapters.py` | `5ce42c7f8de787e6f301c4542ebbb5057bc8b1702758702095f62cc720743137` |
| `vendor/t0_core/scripts/diagnose_inter_head_disagreement.py` | `b996ed923ac3eefeb0550901b2bf8ce5c5e0ae6486c8d41efdc2de9fb9567c99` |
| `vendor/t0_core/exploratory/shadow-ambiguity/test_shadow_ambiguity.py` | `12046a3ff98ddaba207396b943415dd5360f4cb4a942462994889d0aa8771961` |
| `/Users/msrk/Documents/t0-morphology-furnace/scripts/step0_belief_readout.py` | `81e74f9724c03fcb50a88e1c36aae79e7ec76462e4c566c008d470e881fa42b0` |
| `stage_b/profiles_bench/SMOKE_SUMMARY.json` | `ab5ca5612893d4902734cac55defbbb6028b8e141176c4042b09d70ed2cb8f9e` |
| `stage_b/profiles_bench/anli_r1_rep/Qwen3-8B-4bit.profile.json` | `0d026a01f4cb2ee4d3e12ecec793ce10fd9516333bebe219cecf04d1062f6825` |

These are inspected-baseline hashes, not future implementation hashes.

### 3.6 Exact historical reference matrices

| Model basename | `stage_b/profiles/anli_r1/` SHA-256 | `stage_b/profiles_bench/halueval_qa/` SHA-256 |
|---|---|---|
| `Llama-3.1-8B-Instruct-4bit.matrix.npz` | `1c265f9bf01c48076f456542f44e291cc79775cd4ab0f2410c313fd6ecc05782` | `80c6da24a1d166e866c6fe7e39231fb2b30a97a0341ecfc71a0b5a42bca658ce` |
| `Llama-3.2-3B-Instruct-4bit.matrix.npz` | `bbfe19675667f2bda453ea27a84f1f0a5508c18ad1ab3642a11e2affd47bc4c0` | `be6f9f28d8efbf76603afe22f95d4a18abb40e106f7828bd8f7f07d9b33ff391` |
| `Mistral-7B-Instruct-v0.3-4bit.matrix.npz` | `2ee0ae473ccb028519a609fb3651db14b2f4419e7f79c578382ee318635f30ec` | `0bd53934e48aca43bb5a6bff4f72c6b8f47234ae86b2ac9c4edf73fc41103322` |
| `Mistral-Nemo-Instruct-2407-4bit.matrix.npz` | `7f4cfcc5a5cf981bbf1a1c6e1a2895f83af2859c846fa5141447428ae19e8cc9` | `aa54a8077e815d04a69801ec5e2e7c1b89a9b9c40ce98a9bc4847b9e6d3279af` |
| `Phi-3.5-mini-instruct-4bit.matrix.npz` | `50fb54c589807ddce978cbb4a9bd34f3d8f0373e8055355c419af7b9632092bb` | `89a29c2f2ac0b2b7957ea7d2ab2fdf436cd84a15d0c7ac356149021979b2ed66` |
| `Phi-4-mini-instruct-4bit.matrix.npz` | `6f0c16d2d9af017a868bd8e224827caf3b7862c9d33fb3fe92c183f50c98f02a` | `5ee6feb24b60cc44244cc194e0cd38e34c27eca66740bb03025382be2cdee79c` |
| `Qwen2.5-7B-Instruct-4bit.matrix.npz` | `065c5a5a25b1a780ebda4ba5412936db415b1384ebc5c0079e7bd897fafada5c` | `7c1e23f1c3ec9cb2a8a33018eeaae682f4de7a9707ecbf402e1e9c4498f1e3dd` |
| `Qwen3-1.7B-4bit.matrix.npz` | `0e3383f6842320a76d2ca2f6d2c9dc19e044a5110988565f13825f959e12bc2a` | `b4379faad2f1109599652bdd99ef5734c38d981f47f41450a423020dbedb6142` |
| `Qwen3-8B-4bit.matrix.npz` | `14138ecfe38e11ae689539e65b89f0eced46905f5acb8fb55ba86327d957f1c4` | `2e60b521e33b219ddc162b4d6a8672b8357f8fa3a339443890b9d27cfe277b94` |
| `gemma-3-4b-it-4bit.matrix.npz` | `3dbbdb7b8960d852287bc5dfe9a25f01378fce832be9264366c15fd102ca40df` | `b64f941fe6ff4b399ae9bcc27b198f3d3c8218054083ae6031c266528c9cb483` |

### 3.7 Freeze-time implementation manifest

The exact filenames and hashes of the new Endpoint-A extractor/scorer, Endpoint-B sampler/scorer, common schema, smoke/parity driver, decidedness builder, optimized selector if used, selector-equivalence driver, and H-TEMP-1 analysis script are **TBD-at-freeze** because those files do not yet exist. Before any run, a signed revision must record each repository-relative filename and SHA-256, output-schema version/hash, imported module hashes, git commit, environment lock, deterministic smoke-row manifest, tokenizer buckets, pilot uniforms, and the 12-vector selector-equivalence report. For every historical reference profile, copy its recorded module/environment provenance into the freeze manifest next to the fresh environment lock and produce an explicit historical-versus-fresh comparison. No placeholder filename or hash is executable provenance.

---

## 4. Exact 31-cell panel and comparator logic

The final ordered panel has exactly 31 cells. The first 27 exact keys were verified from persisted `panel`; readout keys were also verified in `confluence_calibrator.py:49-56`. Fusion family/order follows `append_fusion_columns` and sorted `fusion_signs.json` keys.

### 4.1 Existing 27 exact keys

1. `(0, "Attention", "final_js")`
2. `(0, "Attention", "final_js_kv_groups")`
3. `(0, "Attention", "final_js_no_bos")`
4. `(0, "Attention", "final_bos_mass")`
5. `(0, "Attention", "final_v_norm_bos")`
6. `(0, "Attention", "final_v_norm_max")`
7. `(0, "Attention", "final_v_norm_lastq_weighted")`
8. `(0, "Attention", "mid_js")`
9. `(0, "Attention", "mid_js_kv_groups")`
10. `(0, "Attention", "mid_js_no_bos")`
11. `(0, "Attention", "mid_bos_mass")`
12. `(0, "Attention", "mid_v_norm_bos")`
13. `(0, "Attention", "mid_v_norm_max")`
14. `(0, "Attention", "mid_v_norm_lastq_weighted")`
15. `(0, "Attention", "last_minus_1_js")`
16. `(0, "Attention", "last_minus_1_js_kv_groups")`
17. `(0, "Attention", "last_minus_1_js_no_bos")`
18. `(0, "Attention", "last_minus_1_bos_mass")`
19. `(0, "Attention", "last_minus_1_v_norm_bos")`
20. `(0, "Attention", "last_minus_1_v_norm_max")`
21. `(0, "Attention", "last_minus_1_v_norm_lastq_weighted")`
22. `(0, "Readout", "fisher_eff_rank")`
23. `(0, "Readout", "spectral_entropy")`
24. `(0, "Readout", "neg_shadow_logvol_r1")`
25. `(0, "Readout", "null_ratio_post_rank1")`
26. `(0, "Readout", "surprise")`
27. `(0, "Readout", "p_max")`

### 4.2 Existing fusion keys and orientation provenance

28. `(0, "Fusion", "fusion_rank_mean_full")`
29. `(0, "Fusion", "fusion_rank_mean_geom")`

| Task/model condition | ACE component orientation | Three readout component orientations |
|---|---|---|
| `anli_r1`, nine models other than Llama-3.1-8B | Per-cell entries in `fusion_signs.json` | Per-cell entries |
| `anli_r1`, Llama-3.1-8B | Modal fallback because its ACE per-cell entry is absent | Per-cell entries are present |
| `halueval_qa`, all ten models | Modal fallback | Modal fallback |

This table is component-exact: the shorthand “Llama-3.1 is modal on ANLI” would be false for its readout components. Fusion ranks are computed independently within each `(model, task, T)` matrix; no ranks are pooled across T.

### 4.3 NEW comparator cells

No `plain_confidence` or `plain_entropy` key exists in the inspected code or matrices. This registration introduces:

30. `(0, "Comparator", "plain_confidence")`
31. `(0, "Comparator", "plain_entropy")`

For float64 logits `z` and positive T, calculate stable softmax without a denominator epsilon:

`q_T(v) = exp((z(v)-max(z))/T) / sum_u exp((z(u)-max(z))/T)`.

At the prefix locus:

`plain_confidence(T) = max_v q_T(v)`

`plain_entropy(T) = -sum_v q_T(v) log(q_T(v))`,

using natural log and `0 log 0 = 0`.

Section 5.1 defines the sole arithmetic compatibility exception: Endpoint-A T=1 derives both comparator values from the historical prefix-probability vector. Endpoint B and every other Endpoint-A T use the float64 formula above.

Under greedy decoding `surprise(T) = -log(plain_confidence(T)+1e-10)` exactly; the folded AUROCs are identical and the earliest-cell tie-break makes `plain_confidence` unselectable in Endpoint A. It is retained only as a sanity column; the registered confidence-vs-geometry contrast is carried by `surprise` and `plain_entropy`.

For a fixed logit vector and T increasing from 1 upward, raw maximum probability is nonincreasing and raw entropy is nondecreasing. No such direction is claimed when comparing a T below 1 to T=1 in the reverse order, and no AUROC direction follows mechanically.

The 31-cell T=0 selector includes two columns absent from the published 29-cell sealed/BENCH selector. Its AUROC is therefore a new-study reference and is not comparable to the published 29-cell seal/BENCH AUROC.

---

## 5. Common arithmetic, selector, decidedness, controls, smokes, and phase order

### 5.1 Arithmetic and tolerance-parity decision

For the 27 existing cells at exactly `T=1.0`, the fresh extractor must execute the byte-historical arithmetic path: prefix/post logits cast to float32; runtime `safe_softmax` with `+1e-12`; prefix surprise with `+1e-10`; post probabilities cast to float64 and renormalized with `+1e-300`; late-block lens probabilities through historical float64 `softmax_np` with `+1e-300`; and the existing downstream aggregate/statistic functions. ACE remains its historical prefix computation. This choice is registered rather than empirically tuning a tolerance.

For Endpoint A at `T in {0.5,0.7,1.5,2.0}`, use the float64 formula in section 4.3. At Endpoint-A T=1, compute both new comparators from the same historical prefix-probability vector used by surprise; this preserves the exact surprise/plain-confidence identity. Endpoint B uses float64 temperature distributions for sampling and every metric at all three anchors, including T=1, because it has no historical-parity column.

On the non-historical branch, `z` is the float64 cast of the identical captured logits; all downstream statistic functions are the identical sealed implementations with unchanged internal constants; `surprise(T)=-log(q_prefix,T(c)+1e-10)`.

On every non-historical branch, every readout and late-block source receives the same T: `q_post,T=softmax(z_post/T)` and `p_l,T=softmax(logits_l/T)`. Each source uses one fixed top-support ID vector at every T. Let `p_ref` be that source's T=1 probability vector, set `k=min(512,len(p_ref))`, and compute the support exactly as historical `_topk_indices`: `np.argpartition(-p_ref,kth=k-1)[:k].astype(np.int32)` (`comprehensive_run.py:177-179`). Persist the returned ID vector and reuse it without reordering at all T. No alternative tie rule is introduced; ties inherit this exact NumPy operation. Require finite/nonnegative probabilities and absolute sum error `<=1e-12` on the float64 paths.

At Endpoint-A T=1, decidedness, `p_yes`, `p_no`, and `control_mass` use the same historical prefix-probability vector used by surprise and the two new comparators.

This creates a deliberate arithmetic discontinuity at T=1 for existing probability-dependent cells, the two new comparators, and decidedness/coverage quantities: T=1 uses the historical prefix/post probability vectors and float32-compatible arithmetic where applicable, while the other analysis temperatures use float64 rescaling. The discontinuity, including its possible effect on trajectories, must appear in every limitations section and cannot be removed after results are visible.

The T=1 historical sentinel is **tolerance parity after historical arithmetic**: after joining on `sample_idx`, each of the 27 fresh existing-cell values must match the historical value with absolute error `<=1e-6`; NaN locations must match exactly; `labels`, `sample_idx`, and ordered historical `panel` must match exactly. New comparators and reconstructed fusion cells have no historical target. Any desire to widen this tolerance requires a new pre-registration version, not an implementation amendment.

### 5.2 Normative selector

Use `sealed_selector._nested_bootstrap_oob_auroc` for ANLI and `confluence_calibrator._cluster_bootstrap_oob_auroc` (`confluence_calibrator.py:181-254`) for HaluEval. `n_bootstrap=2000`. HaluEval stems are ordered by first row occurrence, exactly as `confluence_calibrator.py:186-195` specifies.

Each selector initializes `numpy.random.RandomState(selector_seed+1)`. Each round draws the original number of rows/clusters with replacement; carries multiplicity in-bag; defines OOB as absent units; requires at least four OOB rows and both labels; evaluates all ordered candidate cells in-bag after candidate-local finite filtering; folds AUROC; locks sign; selects maximum distance from 0.5 with earliest-panel resolution of exact ties; and evaluates the selected signed cell OOB. Report attempted rounds 2,000, usable rounds, median, 2.5/97.5 percentiles, winner counts, and winner stability. Zero usable rounds makes the cell FAILED.

### 5.3 K=3 shuffled-label controls for Endpoint (A)

For every Endpoint-(A) `(model, task, operating point)`, run exactly three controls using `confluence_calibrator.shuffled_label_control` on the full 31-cell panel, same row/stem bootstrap unit, `n_bootstrap=2000`, and control seeds `selector_seed + 90210 + k` for `k=0,1,2`.

ANLI uses global row-label permutation. HaluEval uses independent within-stem swaps of its paired `{0,1}` labels, preserving stem structure. Record each null OOB interval. A control flag is raised when at least two of three lower bounds exceed 0.50. A control flag makes that Endpoint-(A) cell FAILED; it remains in the denominator. Endpoint (B) has no shuffled-label gate because all its 31 per-cell AUROCs are explicitly pilot/descriptive and its registered reversals make no confirmatory claim.

### 5.4 Exact 16-row smoke procedure

Build one deterministic 16-row list per task by taking the eight smallest `sample_idx` rows with source `label=0` and the eight smallest with `label=1`, then sorting by `sample_idx`. These are two distinct row lists, each applied to ten models. The manifest contains 20 model/task entries referencing the appropriate one of the two row-list hashes; it does not imply 20 distinct row selections. A smoke is separate from, and never resumable into, a study cell.

A smoke is satisfactory only if: the pinned model revision and tokenizer load; all 16 rows execute within the section-2.5 pass budget; all 31 keys are emitted in section-4 order at T=1; all scores/probabilities meet section-5.1 numeric rules; at least 15/16 commits are canonical; the HaluEval raw-ID smoke rows equal historical `gen_token_ids`; and existing 27 values meet tolerance parity. Otherwise all six Endpoint-(A) cells for that model/task are FAILED inside the 120-cell denominator. Smoke outcomes never replace full-row gates.

### 5.5 Exact decidedness/coverage decomposition

For every model, enumerate the entire vocabulary in ascending token ID and freeze three explicit lists:

1. **YES bucket:** every token whose one-token decode, after section-2.4 normalization, is a nonempty prefix of `yes`.
2. **NO bucket:** every token whose one-token decode, after section-2.4 normalization, is a nonempty prefix of `no`.
3. **Control bundle:** exactly the union of (i) every token whose decoded string is nonempty and consists solely of Unicode White_Space code points; (ii) every ID in tokenizer `all_special_ids`; (iii) every integer ID key in tokenizer `added_tokens_decoder`; and (iv) every valid vocabulary ID obtained from `special_tokens_map_extended` by recursively flattening dictionary values and list/tuple elements, converting each scalar token object/string with `convert_tokens_to_ids`, and retaining IDs in `[0,vocab_size)`. Deduplicate and sort ascending. No data-dependent selection or subset choice is allowed.

Hash each model's token-ID lists and decoded audit table before inference. At each prefix distribution compute `p_yes`, `p_no`, `decidedness=p_yes+p_no`, `control_mass`, `decidedness_floor=5.0*control_mass`, and `eligible=(decidedness>decidedness_floor)`. The inequality is strict.

Report full-denominator `eligible_cov`, medians/IQRs of decidedness and control mass, and eligible-only selector AUROC. `eligible_cov` divides by 200 for ANLI and 1,000 for HaluEval regardless of finite scores. Eligible-only AUROC is a registered decomposition readout but is not part of the H-TEMP-1 confirmatory falsification family.

This uses the BENCH A1 prefix buckets, not Step-0's literal-token buckets (`step0_belief_readout.py:219-242`). Only the factor 5.0 and strict `>` are inherited from Step-0 (`:49`, `:297-316`). TEMP coverage is therefore not directly comparable to the Step-0 Phi-3.5 value `0.185`.

Attribution labels compare exactly T=2.0 against T=1.0, with endpoint difference defined as `value(T=2.0)-value(T=1.0)`. Declining full AUROC with falling coverage but stable eligible-only AUROC is `coverage-driven`; declining eligible-only AUROC with stable coverage is `geometry-needs-sharpness`; both declining is `mixed`; all other patterns are `other`. “Stable” means absolute endpoint difference `<0.01`; these labels are descriptive.

### 5.6 Binding execution order

The order is binding:

1. Complete adversarial audit; resolve section 11; publish signed/frozen revision.
2. Build all implementation artifacts; fill every filename/hash/environment/token-list/smoke/uniform manifest entry. No model call occurs in this phase.
3. If an optimized selector is proposed, execute the 12-vector equivalence suite in section 7.6 and freeze its report. Any mismatch forbids that implementation.
4. Run the non-resumable T=1 parity sentinel on `sample_idx=0` for every model/task. Any defect triggers section 10 before smokes.
5. Run all non-resumable 16-row smokes and issue the 120-cell Endpoint-(A) status ledger entries implied by failures.
6. Run one Endpoint-(A) extraction sweep for every smoke-satisfactory model/task; then complete full gates.
7. Perform a single look at Endpoint-(A) scoring, controls, decomposition, and H-TEMP-1 analysis. No tuning follows this look.
8. Only after step 7 completes, run all 24 Endpoint-(B) cells under section 8.1, then its single descriptive scoring look.
9. Write the complete report and immutable artifact manifest.

---

## 6. Endpoint (A): reshape-only extraction and validity gates

### 6.1 Denominator and pass budget

Endpoint (A) contains exactly `10 models × 2 tasks × 6 reported operating points = 120 cells`. The five distinct distributions are `0.5,0.7,1.0,1.5,2.0`; T=0 aliases T=1.0.

For each row, the maximum model compute is one prefix trace plus one post-commit step forward after the greedy token is appended; that post pass must capture final readout state and the complete pinned late window. Every positive-T metric is derived from those captured logits/tensors with zero additional forward passes per grid point. Exceeding this budget makes all six cells for the model/task FAILED.

The main Endpoint-(A) extraction is exactly `10×(200+1,000)×2=24,000` model forward calls: one prefix and one post-commit forward for each of 12,000 rows, excluding the separately reported parity sentinels and non-resumable smokes.

### 6.2 Exact per-row algorithm

For each model, task, and ascending `sample_idx`:

1. Construct the inherited prompt and obtain the prefix trace with ACE attention capture, prefix-last logits `z_prefix`, and prefix-last hidden state `h_prev`.
2. Set committed token ID `c=argmax(z_prefix)`. Decode and normalize `c` under section 2.4.
3. Append `c` once and perform one updated-context forward, capturing post-commit logits `z_post`, final-layer hidden `h_t`, and post-commit per-layer hiddens `h_l` for every pinned late block `B-ceil(B/4)..B-1`.
4. Compute the 21 ACE cells once from the prefix attention trace; copy them across all positive T.
5. For each T in order, use the historical-T=1 or float64-T-not-1 arithmetic branch in section 5.1 for prefix, post-readout, and every late-block lens source.
6. Compute `surprise(T)` from the prefix distribution and committed token. Compute `p_max` from the post-readout distribution. Compute `null_ratio_post_rank1` from post-readout probability, final `h_t`, `h_prev`, surprise, `alpha=1.0`, `topk_values=[32]`, `lowrank_values=[32]`, and `v3_rank_values=[1]`. For each aggregate source independently—readout plus every pinned late block—use its T-specific distribution and invariant top-512 support to compute `fisher_eff_rank`, `spectral_entropy`, and `neg_shadow_logvol_r1`; require exactly `1+ceil(B/4)` valid sources and take the unweighted arithmetic mean of each statistic. Compute the two new prefix comparators.
7. Emit 29 primitive values per T; after the model/task/T matrix is assembled, reconstruct both fusion columns and order the final 31 columns exactly as section 4.
8. Set `greedy_error=0` only for source `label=0` with canonical YES or source `label=1` with canonical NO; set it to `1` for the opposite canonical answer or a non-canonical commit. This correctness field is not an identity gate.
9. Emit row audit fields: `sample_idx`, source `label`, `stem_id` where applicable, greedy token ID, decoded token, normalized token, canonical disposition, `greedy_error`, prefix/post probability checks, forward counts, and all five score vectors. For every T and every aggregate source—readout plus each pinned late block—also emit source name/block index, exact top-support ID vector and hash, and the source-level `fisher_eff_rank`, `spectral_entropy`, and `neg_shadow_logvol_r1` values before averaging.

No row may be dropped. Missing or non-finite primitives invoke section 10.

### 6.3 Load-bearing commit identity, behavioral canonicality, and row audit

**HaluEval exact-ID gate.** Join fresh and historical HaluEval matrices one-to-one on `sample_idx`. The fresh greedy token ID must equal the historical `gen_token_ids` entry for every row. Missing, duplicate, extra, or unequal IDs make all six cells for that model/task FAILED. Decoded canonical-YES/canonical-NO/non-canonical equality is reported as a secondary diagnostic, not a substitute for raw-ID equality.

**ANLI behavioral canonicality gate.** Historical ANLI matrices contain no `gen_token_ids`, and seal profiles contain no per-row commit record or commit tally. Exact token identity therefore has no direct historical reference. Require at least 190/200 fresh commits (`>=95%`) to be canonical under section 2.4. Retain all non-canonical rows in every score/rate denominator and report token ID/text/disposition. Fewer than 190 canonical commits makes all six ANLI cells for that model FAILED. This is explicitly a behavioral gate, not an identity gate.

**ANLI row audit.** The one greedily computed token ID must be attached unchanged to all five positive-T row records. This catches emission/bookkeeping divergence only; it is not evidence of historical identity and cannot fail under a correctly implemented section-6.2 computation.

**ANLI transitive identity protection and disclosure.** ANLI commit identity is protected transitively by section 6.4 tolerance parity, whose existing `surprise` and downstream cells are functions of the committed token and sealed compute path. Every report/table/caption containing ANLI Endpoint-(A) results must state: “ANLI has no historical per-row token record; it uses T=1 full-panel tolerance parity for transitive identity protection and a separate >=95% behavioral canonicality gate.” No rowwise comparison to source `label` is an identity test.

For both tasks, also require complete one-to-one `sample_idx` coverage and exact source/matrix `labels` equality as data alignment only. This alignment check must never be described as commit agreement.

### 6.4 T=1 tolerance-parity gate and T=0 alias

Join the fresh T=1 existing 27 columns to the exact historical matrix by `sample_idx` and apply section-5.1 tolerance parity. This is the direct compute-parity gate for both tasks and the transitive commit-identity protection for ANLI. Executability assumes that the fresh MLX/Metal/NumPy/module environment can reproduce the historical forwards closely enough to meet all-row `1e-6` score parity and HaluEval exact-ID equality; section 3.7 requires the historical-versus-fresh provenance comparison before model calls. Environment drift is the expected failure mode if this assumption is false. Such a mismatch stops the affected work and invokes section 10; it may not be waived or absorbed by widening tolerance. Reconstruct fusion columns. Once parity, HaluEval raw-ID identity, and ANLI behavioral canonicality as applicable are satisfactory, create T=0 as an exact alias of the complete fresh T=1 31-column matrix. Do not construct a separate T=0 selector matrix from historical 27 columns.

Any parity discrepancy makes all six model/task cells FAILED and requires an amendment before regeneration.

### 6.5 Per-cell registered readouts

Every Endpoint-(A) cell reports:

1. Full-denominator 31-cell deployability AUROC from section 5.2.
2. Winner counts/stability and all 31 one-cell folded AUROCs.
3. `eligible_cov`, decidedness/control summaries, and eligible-only selector AUROC from section 5.5.
4. Canonical-YES, canonical-NO, and non-canonical commit counts; greedy correctness-error rate; and greedy commit-NO rate. Both rates are descriptive and fixed across T.
5. K=3 shuffled-label-control results.
6. Full gate, row-count, finite-count, bootstrap-attempt/usable, and forward-count audit fields.
7. The explicit limitation that Endpoint-A T=1 existing probability-dependent cells, the two new comparators, and decidedness/coverage quantities use historical probability vectors/arithmetic whereas other T values use float64 rescaling.

---

## 7. Endpoint-(A) H-TEMP-1 algorithm, multiplicity, compute budget, and verdict

### 7.1 Registered reversal contrasts

Define an **evaluable pair** as a `(model,task)` pair for which all five distinct positive-T cells are valid after smoke, extraction, identity/behavior/parity, numeric, selector, and control rules. A pair with any FAILED positive-T cell is excluded from the reversal family but remains fully reported as FAILED in the 120-cell ledger. Let E be the evaluable-pair set and require `|E|>=14` of 20 for a study-level verdict.

For each pair in E, let `M(T)` be the point-estimate full-denominator selector median AUROC at positive T. For all three interior anchors `i in {0.7,1.0,1.5}`, calculate

`R_i = (M(T_i)-M(T_{i-1})) * (M(T_{i+1})-M(T_i))`.

There are exactly `3|E|` confirmatory contrasts, between 42 and 60 whenever the minimum family is met. Negative R indicates a reversal. Every evaluable contrast is bootstrapped regardless of whether its observed sequence appears monotone; point-estimate classification never screens a test.

### 7.2 Exact nested outer bootstrap

For each pair in E, initialize the outer stream `rng=RandomState(selector_seed+20001)`. Generate exactly 2,000 outer replicates in replicate-index order. Each replicate draws `rng.randint(0,n_clusters,size=n_clusters)` once and applies that same outer cluster-index multiset to all five positive-T matrices. Each sampled occurrence receives a unique outer-occurrence cluster ID while retaining all rows of its source cluster.

Within every outer replicate and every T, rerun the full 31-cell selector with `n_bootstrap=2000`. Each outer replicate re-initializes the inner selector with `RandomState(selector_seed+1)`. Compute five medians and three reversals. Define `B_common` as replicate indices for which all `3|E|` evaluable contrasts are available. Require `|B_common|>=1900`, exactly 95% of B=2,000. Failure of this bootstrap-completeness bar makes the study-level result inconclusive; no further reduced family is allowed.

### 7.3 One-sided familywise simultaneous decision

For contrast `c`, let point value be `R_c`, bootstrap values `R*_{c,b}` restricted to `b in B_common`, and `s_c` their sample standard deviation computed with `ddof=1`. For `s_c=0`, set all standardized deviations to zero. For each `b in B_common`, compute

`D_b = max_c ((R_c - R*_{c,b}) / s_c)`

over all `3|E|` contrasts with positive `s_c`; zero-variance contrasts contribute zero. Let `q` be the empirical 97.5th percentile of the `|B_common|` values using NumPy's default linear percentile rule. Define simultaneous upper bound `U_c = R_c + q*s_c`; for `s_c=0`, `U_c=R_c`.

This max-statistic construction is the single multiplicity rule across the entire registered evaluable family at one-sided familywise alpha `0.025`. Per-contrast unadjusted percentile results are descriptive only.

### 7.4 Registered study-level output

The single Endpoint-(A) decision output is:

- `H-TEMP-1_FALSIFIED` if `|E|>=14`, `|B_common|>=1900`, and at least one evaluable contrast has `R_c<0` and simultaneous `U_c<0`.
- `H-TEMP-1_NOT_FALSIFIED` if the same family/completeness bars hold and no evaluable contrast meets that rule.
- `H-TEMP-1_INCONCLUSIVE` if `|E|<14` or `|B_common|<1900`.

FAILED pairs never disappear from the 120-cell denominator; they are excluded only from a hypothesis family they cannot numerically enter. The threshold is at least one familywise-adjusted reversal among a pre-specified minimum of 14 evaluable pairs. `NOT_FALSIFIED` is not evidence of universality or proof of monotonicity.

### 7.5 Registered confidence-versus-geometry interpretation

For every T, report one-cell AUROCs for `surprise`, `plain_entropy`, and `fusion_rank_mean_geom`. The registered signal-beyond-confidence pattern is: fusion geometry does not decline by `0.01` or more from T=1 to T=2 while either surprise or entropy AUROC declines by `0.01` or more. The converse pattern is geometry decline without confidence decline. Counts use evaluable pairs E and are descriptive; FAILED pairs are shown separately and counts do not alter section 7.4.

### 7.6 Compute budget disclosure

The upper-bound H-TEMP-1 procedure requires `20 pairs × 2,000 outer replicates × 5 T = 200,000` full selector invocations; at 2,000 inner rounds that is 400 million selector rounds and roughly 12.4 billion in-bag candidate-AUROC evaluations plus OOB evaluations. At the minimum 14-pair family, the corresponding counts are 140,000 invocations, 280 million rounds, and roughly 8.68 billion candidate evaluations.

A numerically optimized or vectorized reimplementation is compliant only if it produces bit-identical selector outputs—median, both CI bounds, winner counts, winner stability, and usable-round count—for identical inputs and seeds. Before inference, demonstrate equivalence on exactly 12 registered vectors: the historical matrices for `{Llama-3.2-3B-Instruct-4bit, Phi-3.5-mini-instruct-4bit, Qwen3-8B-4bit}` crossed with `{anli_r1,halueval_qa}`, each in two variants: (a) the raw persisted 27 columns; and (b) a 31-column audit matrix formed by reconstructing both historical fusion columns, then appending an audit `plain_confidence` column that is an exact copy of historical `p_max` and an audit `plain_entropy` column that is the exact negative of historical `surprise`. These last two audit values test 31-column handling and exact folded-AUROC/tie behavior; they are equivalence fixtures only, not TEMP comparator measurements. Use each task's registered selector seed, row bootstrap for ANLI, and stem bootstrap for HaluEval, all with 2,000 rounds. Hash the complete reference-versus-optimized equivalence report into the freeze manifest. Any mismatch forbids the optimized implementation.

Planning estimate on the named execution host—Mac mini `Mac16,10`, Apple M4 10-core, 16 GB—is 10–25 seconds per literal full-selector invocation, or approximately 23–58 days single-process at the 200,000-invocation upper bound (16–41 days at 140,000). This is an estimate, not a measured benchmark. The freeze manifest records actual hardware and the optimized equivalence result. Reducing registered rounds or changing outputs remains a new-version change; bit-identical optimization and parallel scheduling are allowed.

With adequate independent workers, literal selectors parallelized across model/task pairs are expected to take roughly 3–6 wall-clock days; a successfully equivalence-validated vectorized implementation may reduce scoring to hours. These are planning alternatives, not registered performance bars, and the freeze report must replace them with measured equivalence-suite timing before choosing the execution route.

---

## 8. Endpoint (B): real-sampling PILOT

### 8.1 Denominator and launch condition

Endpoint (B) has exactly `4 models × 2 tasks × 3 T anchors = 24 cells`. Each prompt/T has five draws. Endpoint (B) remains PILOT and descriptive. If Endpoint (A) cannot reach its scoring step because of unresolved global machinery, all 24 pilot cells are reported `NOT RUN`; none is dropped. If Endpoint A does reach its scoring step, every one of the 24 pilot cells runs regardless of the corresponding model/task's Endpoint-A cell statuses.

Endpoint (B) executes 14,400 prefix forwards and at most 72,000 unique-token post-commit forwards, for a maximum of `86,400` model forward calls; duplicate sampled token IDs reduce only the post-commit count and must be reported.

### 8.2 Exact trace, sampling, and post-commit algorithm

For each model/task/T/prompt in section-2 order:

1. Run one prefix trace with ACE capture, prefix logits `z_prefix`, and prefix-last `h_prev`.
2. Compute float64 `q_prefix,T`; consume the five pre-generated uniforms; sample five token IDs by inverse CDF.
3. For each unique sampled token ID, append that token and perform one post-commit step forward to obtain `z_post`, final `h_t`, and every pinned late-block `h_l`. Reuse that complete capture for duplicate draws. The budget is one prefix trace plus at most five post-commit forwards per prompt/T; no second generated token is allowed.
4. For each draw, normalize the sampled token. Set canonical disposition to YES, NO, or non-canonical.
5. Set `sample_error=0` only when source `label=0` and disposition is canonical YES, or source `label=1` and disposition is canonical NO. Set `sample_error=1` for the opposite canonical answer or any non-canonical token. No draw is dropped.
6. Set descriptive `strict_non_yes=0` only for canonical YES and `1` for canonical NO or non-canonical. This is an answer-bias readout, not a hallucination/error rate.
7. Compute the 21 ACE cells from the shared prefix trace. Compute `plain_confidence` and `plain_entropy` from shared float64 `q_prefix,T`, and draw-specific `surprise` from the sampled token's prefix probability. Compute `p_max` and null ratio from that token's post-readout source, using the fixed `[32]`, `[32]`, `[1]`, and alpha settings. For `fisher_eff_rank`, `spectral_entropy`, and `neg_shadow_logvol_r1`, compute the same `1+ceil(B/4)` aggregate as Endpoint A: rescale post-readout and every late-block lens logits by the sampling T, use each source's invariant top-512 support, compute source statistics, and take their unweighted mean. Thus all six sealed readout cells are draw-dependent in principle; duplicate token draws share identical values. Reconstruct both fusion cells after row/draw primitives are assembled.
8. Emit sample/stem IDs, source label, T, draw, uniform, sampled token ID/text/disposition, `sample_error`, `strict_non_yes`, post-trace reuse ID, forward counts, and all 31 scores.

### 8.3 Pilot summaries and exactly named detector readouts

The inference unit is never an individual draw. For ANLI, a unit is one prompt with five draws. For HaluEval, a bootstrap unit is one `stem_id` carrying both labeled rows and all ten draws.

Report `sample_error_rate` as the unweighted mean of prompt/stem unit error means; this is the registered H-TEMP-1 behavioral quantity. Also report descriptive non-canonical rate and strict-non-YES answer-bias rate.

Compute per-cell AUROC for all 31 cells descriptively. For ANLI, average each score across five draws within prompt, then compare to source label. For HaluEval, average each score across five draws within each labeled row; retain both rows and source labels while resampling their common stem. AUROC is direction-folded with `sealed_selector.auroc_signed` on the evaluated sample. No 31-cell selector is invoked for Endpoint B.

The only detector cells registered for section-8.4 reversal analysis are exactly:

1. `(0, "Fusion", "fusion_rank_mean_geom")`
2. `(0, "Readout", "surprise")`

No other cell's reversal is registered.

### 8.4 Exact joint-T pilot bootstrap

Index the eight `(model,task)` pairs in section-2.2 model order and task order ANLI then HaluEval. For `pair_index=0..7`, initialize `rng=RandomState(20260826+40001+pair_index)` and generate exactly 2,000 joint replicates in replicate-index order. Each replicate draws `rng.randint(0,n_clusters,size=n_clusters)` once and applies that same cluster-index multiset at all three T anchors. Recompute within the replicate at each T: sample error rate and folded per-cell AUROC for the two registered cells.

For each registered quantity M, compute

`R_1.0 = (M(1.0)-M(0.7)) * (M(1.5)-M(1.0))`.

Report point R, bootstrap median, 2.5/97.5 percentiles, attempted 2,000, and usable count. Per-cell T-specific point estimates and intervals for all 31 cells remain descriptive. No multiplicity-adjusted or confirmatory pilot verdict is permitted.

---

## 9. Denominators, status ledger, and abort rules

### 9.1 Frozen denominators

Every `(model,task,T)` cell remains in its frozen denominator. Endpoint A always reports 120 statuses. Endpoint B reports 24 statuses if launched, or 24 `NOT RUN` statuses if its launch condition is unmet. A smoke, gate, control, extraction, schema, numeric, bootstrap, or artifact failure is a FAILED cell inside the denominator, never silently dropped.

This explicitly closes the adversarial-review escape hatch in which “at least 8/10” silently becomes “8/9” after a failure. No aggregate promotion bar exists.

### 9.2 Rows, draws, and finite handling

No source row or pilot draw may be removed. Candidate-local non-finite filtering inside inherited selector machinery is recorded but never changes the full data, coverage, or status denominator. Report full N, finite N per cell, attempted/usable bootstrap rounds, and every failure reason. Non-canonical pilot draws are errors, not abstentions.

### 9.3 Stop conditions

Stop the affected model/task and file an amendment before regeneration when any of these occurs:

1. Model/data/module/environment/manifest hash or revision mismatch.
2. Panel key/order mismatch.
3. Section-6.3 HaluEval identity, ANLI behavioral-canonicality, or row-audit defect; or section-6.4 parity defect.
4. Probability/score numeric failure or missing row/draw.
5. Endpoint-A forward budget exceeds one prefix plus one post-commit pass per row, or any T-specific extra forward occurs.
6. Endpoint-B forward budget exceeds one prefix plus five unique-token post passes per prompt/T, or duplicate-token reuse is inconsistent.
7. Smoke criterion, shuffled-label control criterion, or bootstrap usability criterion is unsatisfied.
8. Pilot uniforms, inverse-CDF results, cluster order, outer occurrence IDs, or seeds are not exactly reproducible.

All registered T cells for a stopped model/task become FAILED for that endpoint. Independent model/tasks may continue only if the defect cannot contaminate them and the amendment states why.

---

## 10. Amendment and versioning rule

Implementation-only corrections require a dated amendment filed before regeneration. It must identify the defect; affected files/hashes/models/tasks/T/rows/draws/artifacts; whether outputs were visible; exact correction; and regeneration scope. Superseded artifacts remain preserved.

Any change to grids, cohort, tasks, prompt draws, seeds, labels, correctness/non-canonical disposition, panel keys/order/formulas, loci, late-window/source aggregation, T=1 arithmetic branch, numeric supports/ranks, forward budgets, selector/control/smoke/equivalence algorithms, bootstrap or inference units, `N_min=14`, multiplicity/verdict rules, bars, validity gates, endpoint logic, pilot launch rule, decidedness/coverage definitions, YES/NO buckets, control-bundle rule, commitment normalizer, canonical-prefix rule, fusion orientations, or phase order voids the current phase and requires a new pre-registration version.

No gate, exclusion, bar, or interpretive exception may be added after outputs are visible. Post-hoc diagnostics must be labeled post hoc and cannot replace registered readouts.

---

## 11. OPEN ITEMS before freeze

These are deterministic implementation-finalization tasks, not undecided science:

1. **Implementation/hash manifest.** Create and hash the Endpoint-A extractor/scorer, Endpoint-B sampler/scorer, common schema, smoke/parity driver, decidedness builder, optimized selector/equivalence driver if used, and H-TEMP-1 analysis script; record git/environment/imported-module hashes and the historical-versus-fresh provenance comparison required by section 3.7.
2. **Deterministic manifests and equivalence.** Materialize and hash the two smoke row lists/20 references, per-model YES/NO/control token lists and decoded audits, pilot uniform array, and the 12-vector selector-equivalence report if optimization is used.
3. **Targeted delta-verification and sign-off.** Verify the v1.3 N9/M1-M7 edits and obtain MK sign-off before any model call.

No cell name, model/data path, grid, locus, gate, statistical readout, denominator, or failure disposition remains scientifically open in this draft.

---

## 12. Closing adversarial self-audit

| Defect/risk | v1.3 disposition | Freeze evidence required |
|---|---|---|
| N1: aggregate shadow cells misdescribed | Readout plus every pinned late-block logit lens is captured, T-rescaled, top-512 scored, and mean-aggregated for A and B. | Per-source row audit and T=1 parity. |
| N2: verdict unreachable through totality/canonicality | ANLI behavioral bar is 95%; FAILED pairs stay in ledger but family requires at least 14 valid pairs. Known same-template rates are disclosed. | Evaluable-pair ledger and `N_min` check. |
| N3: literal compute infeasible/optimization ambiguous | Bit-identical optimized selector allowed only after 12 registered equivalence vectors; named-host wall-clock range disclosed. | Hashed equivalence and hardware report. |
| N4: unmeasured tolerance gamble | Endpoint-A T=1 existing cells use the historical arithmetic path; tolerance remains a sentinel, not a float-path bet. | Per-cell parity-error table. |
| N5: ANLI behavioral/bookkeeping checks mislabeled identity | 95% rule is behavioral; token replication is an audit; T=1 parity is transitive identity protection. | Separate status fields and required disclosure. |
| N6: pilot eligibility undefined | All 24 pilot cells run after the global Endpoint-A scoring launch condition, regardless of pair status. | Complete 24-cell ledger. |
| N7: attribution endpoints unnamed | Difference is exactly T=2.0 minus T=1.0. | Attribution table with both values. |
| N8: numeric settings absent | 512 support, `[32]`, `[32]`, `[1]`, alpha 1.0, and `ceil(B/4)` rule are literal pins. | Extractor config/schema audit. |
| N9: typo/manifest implication | The `M(T_{i-1})` delimiter typo is corrected in v1.3; two task row lists feed 20 smoke references. | Formula lint and manifest hashes. |
| M1: random primitives underdetermined | Inverse-CDF side/clipping, outer `rng.randint` calls/order, and `ddof=1` are literal pins. | Seeded primitive fixtures in the analysis audit. |
| M2: non-historical branch underbound | Identical captured logits are cast to float64; sealed functions/constants and the surprise formula are mandatory. | Function/hash and row-formula audit. |
| M3: incomplete discontinuity disclosure | Existing cells, new comparators, and decidedness/coverage all disclose and bind the T=1 historical branch. | Limitations and coverage-vector audit. |
| M4: historical-environment assumption implicit | Historical and fresh provenance are compared before calls; drift is a named parity-failure mode governed by section 10. | Hashed provenance comparison and parity ledger. |
| M5: deployability over-reading | Endpoint A is explicitly counterfactual and rescales all prefix/readout/logit-lens distributions; unchanged deployment is T-invariant. | Limitations text retained in report. |
| M6: per-source audit absent | Every readout/block source emits support identity and three pre-aggregation statistics. | Per-source row audit and T=1 parity. |
| M7: compute disclosure incomplete | A/B forward counts and literal-parallel/optimized selector timing alternatives are explicit. | Forward-count ledger and measured equivalence timing. |
| D1-D11 retained corrections | Exact panel, loci, raw Halu ID, pilot cells/bootstrap, controls, decidedness, fusion provenance, and amendment rules remain. | Full regression audit. |
| Denominator escape | FAILED/NOT RUN cells remain in 120/24 ledgers; only evaluability for a numeric hypothesis family is conditional. | Complete status ledger. |
| Premature frozen claim | Header is v1.3 draft, NOT frozen and unsigned. | Later signed revision only. |
| Sealed artifact contamination | New outputs isolated under `stage_b/profiles_temp/`; sealed artifacts read-only. | Before/after file manifest. |
| Poisoned-context scope leak | Explicitly out of scope. | Separate future pre-registration if pursued. |

End of v1.3 draft.
