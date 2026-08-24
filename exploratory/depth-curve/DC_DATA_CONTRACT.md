# DC figure suite — data contract

_Established 2026-08-23 by a read-only pass over the banked grid-A/grid-B artifacts. Recovered from a workflow journal after the build phase was cut short by a spend limit. Read this BEFORE writing any DC figure code._

## python

/Users/msrk/Documents/commit-confluence/.venv/bin/python — VERIFIED by execution: numpy 2.0.2, matplotlib 3.9.4 (both import cleanly). This is the same numpy version recorded in GRID_B_RESULTS.json provenance.numpy_version ("2.0.2"), so arithmetic here is consistent with the scoring environment. No gap.

## npz_inventory

ROOT: /Users/msrk/Documents/commit-confluence/exploratory/depth-curve/npz/

=== THE REPORTED CLAIM IS FALSE AS STATED — READ THIS FIRST ===
"Grid A banks all four metric curves per cell while grid B banks only a reduced set" is WRONG at the npz level. BOTH grids bank the identical array set, all four metrics, every block, every row. The reduction is real but lives in the SCORED JSON, not the npz: RESULTS.json banks per-block AUROC curves for all 4 metrics + the shuffled envelope for every grid-A cell; GRID_B_RESULTS.json banks NO curve of any kind for 8 of its 9 evaluable cells (only cross-fitted E5/E6 scalars), plus one cross-fitted median TRAINING curve for the 4 Llama-3.1 cells inside E8_llama_context. Downstream agents must be told this precisely or FIG1 will be mis-specced.

=== SHARED SCHEMA (identical in BOTH grids) ===
Every *.depth.npz contains exactly 8 arrays:
  scores        (200, N, 4)  float64   [row, block, metric]; metric order FROZEN
  labels        (200,)       int64     binary; every cell exactly 100/100
  sample_idx    (200,)       int64     identity 0..199 (scorer asserts this)
  gen_token_ids (200,)       int64
  commit_p      (200,)       float64
  yes_no        (200,)       int64
  metrics       ()           <U71      JSON str: ["final_js","final_js_no_bos","final_js_kv_groups","final_bos_mass"]
  meta          ()           <U…       JSON str (below)
metric index map: 0=final_js, 1=final_js_no_bos (PRIMARY), 2=final_js_kv_groups, 3=final_bos_mass.
Blocks are 0-indexed layers 0..N-1; N == meta.n_layers. NO NaN/inf anywhere: all 17 npz have nan-fraction 0.0000 on all 4 metric columns (verified). "Block" and "decoder layer" are the same object in this lane.

=== GRID A — npz/depth_curve/<task>/<slug>.depth.npz  (schema "furnace-depth-curve/1.0") ===
 anli_r1/Qwen2.5-7B-Instruct        (200,28,4)  N=28  nf4  yes_no_rate 0.970
 anli_r1/Qwen2.5-32B-Instruct       (200,64,4)  N=64  nf4  1.000
 anli_r1/Qwen2.5-72B-Instruct       (200,80,4)  N=80  nf4  0.985
 anli_r1/Llama-3.3-70B-Instruct     (200,80,4)  N=80  nf4  0.950
 halueval_qa/Qwen2.5-7B-Instruct    (200,28,4)  N=28  nf4  1.000
 halueval_qa/Qwen2.5-32B-Instruct   (200,64,4)  N=64  nf4  1.000
 halueval_qa/Qwen2.5-72B-Instruct   (200,80,4)  N=80  nf4  1.000
 halueval_qa/Llama-3.3-70B-Instruct (200,80,4)  N=80  nf4  0.995
 8/8 present. Sidecar: <slug>.gates.json. meta keys: schema, model, task, precision, n_layers, n_heads, n_kv_heads, n_rows, metrics, backend("modal-torch"), comparable(false), data_path, data_sha256, data_hash_loader, yes_no_commit_rate, gate{rows[],GATE_cos_ok,GATE_yes_no_ok}, prereg, note, provenance{extractor_code_commit c56b7e1, hf_model_revision, torch, transformers, bitsandbytes, numpy, seal hashes, depth_max_tokens}.

=== GRID B — npz/depth_grid_b/<task>/<slug>.depth.npz  (schema "furnace-depth-curve/1.1-gridB") ===
 anli_r1/Llama-3.1-8B-Instruct                     (200,32,4)  N=32  nf4                      0.970
 anli_r1/Llama-3.1-70B-Instruct                    (200,80,4)  N=80  nf4                      0.995
 anli_r1/Mistral-Medium-3.5-128B                   (200,88,4)  N=88  fp8origin-dequant-bf16   0.895   <-- FP8-ORIGIN, FLAG IN EVERY PANEL
 anli_r1/gemma-3-12b-it                            (200,48,4)  N=48  nf4                      1.000
 halueval_qa/Llama-3.1-8B-Instruct                 (200,32,4)  N=32  nf4                      1.000
 halueval_qa/Llama-3.1-70B-Instruct                (200,80,4)  N=80  nf4                      1.000
 halueval_qa/Mistral-Small-3.2-24B-Instruct-2506   (200,40,4)  N=40  nf4                      1.000
 halueval_qa/Mistral-Medium-3.5-128B               (200,88,4)  N=88  fp8origin-dequant-bf16   1.000   <-- FP8-ORIGIN
 halueval_qa/gemma-3-12b-it                        (200,48,4)  N=48  nf4                      1.000
 9 of 12 core cells have an npz. THREE HAVE NO NPZ AT ALL (status.json only, no .depth.npz, no .gates.json):
   anli_r1/Mistral-Small-3.2-24B-Instruct-2506 — "missing smoke prompt manifest … run smoke first" (OPERATIONAL abort)
   anli_r1/gemma-3-27b-it     — "row 0 block 3 cell (0,'Attention','final_js_no_bos'): bad score None"  <-- js_no_bos INSTRUMENT-DOMAIN boundary
   halueval_qa/gemma-3-27b-it — same js_no_bos None abort
 gemma-3-27b-it's N=62 is known only from GRID_B_RESULTS.json cells[…].L (registry expectation), never from an npz.
 Grid-B meta ADDS (vs grid A): model_key, revision_pinned, capture_mode, attn_dtypes_seen, load_notes, wrapper_class, device_param_counts, cuda_peak_alloc_bytes, prompt_manifest_file_sha256, prompt_row_hashes. Grid B is strictly RICHER in metadata, never poorer in data.
 Sidecars: <slug>.gates.json AND <slug>.status.json {status ok|aborted, reason, model_key, model_id, revision, task, schema, gpu_label, result{…}}.

=== NO 405B ANYWHERE ===
No 405B npz, no 405B key, zero literal "405B" substrings in RESULTS.json / GRID_B_RESULTS.json / RESCORE_GRID_A.json (grep-verified, count 0 in all three). score_405b.py exists in the lane but produced nothing. Rail 7 satisfied by absence.

## scored_json_fields

Three scored JSONs, three DIFFERENT estimators. Never mix them on one axis without saying which.

===== A. RESULTS.json — REGISTERED GRID-A (PRE_REGISTRATION.md; in-sample sign-free; seed 20260816, NBOOT 1000, NPERM 200) =====
top: prereg, seed, nboot, nperm, primary_metric("final_js_no_bos"), E1_per_task, E1_OVERALL("UNDECIDED"), predictions, cells
cells key "<task>/<slug>", 8 entries. Per cell:
  .model .task .n_layers (=N) .n_rows
  .curves_signfree_auroc        dict metric-name -> list[N] float   <== THE ONLY BANKED PER-BLOCK CURVES IN THE LANE
  .envelope_q97_5_primary       list[N] float   <== shuffled-label envelope, PRIMARY METRIC ONLY
  .lstar                        int   in-sample qualifying peak block (A>=0.65 AND A>env)
  .lstar_boot_median            float
  .lstar_boot_ci                [p5,p95]  <== the ONLY per-cell peak-location CI banked anywhere
  .boot_noqual_frac
  .peak_auroc                   float = A[lstar]
  .N_minus_lstar_median, .lstar_frac_median   (bootstrap-median based, NOT lstar/N)
  .E2 {baseline_mid, rise (R), max_adjacent_jump (J), shape CLIFF|GRADUAL|MIXED}   <== FIG5 SOURCE
  .E3_early_blocks  list[int]
  .E4_terminal_dip  BOOLEAN ONLY — no magnitude, no CI
  .mid_region_median  (median A over closed [0.4N,0.6N])
  .max_auroc_any_block

===== B. RESCORE_GRID_A.json — GRID A UNDER THE GRID-B CROSS-FITTED ESTIMATORS (rescore_grid_a.py; RS_SEED 20260817, K=5, NPERM_INNER 200, NPERM_OUTER 2000, NBOOT 1000) =====
Banner: "pre-freeze calibration artifact. Registered grid-A results are unchanged." Captions using it must say cross-fitted rescore, never registered grid-A E4/E2.
top: banner, config, provenance{npz_sha256, rescorer_sha256, depth_score_sha256, numpy_version, fold_map_sha256, fold_map}, cells(8), grid
per cell:
  .L
  .e5 {defined, delta (Δ_cf = DIP MAGNITUDE), fold_contrasts[5], fold_peaks[5], success, perm_p, null_undefined_frac,
       boot_ci_5_95_conditional [lo,hi]  <== THE DIP CI, boot_undefined_frac}
  .e6 {defined, J (J_cf), R (R_cf), fold_J[5], fold_R[5], fold_jstar[5], null_q95, null_q95_is_neginf, null_undefined_frac, perm_p_J, success}
grid: per-task + "_POOLED" success counts, p_grid, aggregate_delta_boot_ci_5_95_alldef. Grid-A rates under these estimators: E5 8/8, E6 7/8.

===== C. GRID_B_RESULTS.json — REGISTERED GRID-B SINGLE LOOK (PRE_REGISTRATION_EXPANSION.md; machinery imported unmodified from B) =====
top: banner, config, provenance{rescore_machinery_sha256, scorer_sha256, extractor_sha256, prereg_sha256, numpy_version, fold_map_sha256, npz_sha256, status_sha256}, cell_failures, n_evaluable_of_12(=9), E5, E6, sensitivity, bootstrap_aggregates, E7_peak_distance, E8_llama_context, E1pp_panel, predictions_as_registered, cells
  .E5 {decision "WEAKEN", count_of_12 8, per_task{anli_r1 4, halueval_qa 4}, per_family{llama31 4, mistral 2, gemma 2},
       p_grid_pooled 0.0004997501249375312, guards{count_ge_10 false, families_ge_3of4 false, "p_grid_lt_0.05" true}}
  .E6 {decision "NOT TESTED — gate closed", count_of_12 2, per_task{1,1}, per_family{llama31 0, mistral 1, gemma 1}, p_grid_pooled 0.0004997…, guards null}
  .sensitivity {leave_medium_out{e5_of_10 6, e6_of_10 1}, evaluable_only{e5 "8/9", e6 "2/9"}}
  .bootstrap_aggregates {anli_r1|halueval_qa|pooled} each {designated_cells[], excluded_failed_cells[], ci_5_95_alldef[lo,hi], joint_undefined_frac}
       pooled ci_5_95_alldef = [0.12276666666666666, 0.2012333333333333]  (the "[0.123, 0.201]" pooled dip CI)
  .cells (12 keys incl. 3 aborted). Defined: {L, failed:false, e5{same shape as B}, e6{…}}.
       Aborted: {L, failed:true, reason, e5{defined:false, why, delta:null, success:false}, e6{defined:false, why, J:null, R:null, success:false}} — NO fold arrays, NO CI.
  .E7_peak_distance  per MODEL (6): {defined, peak_anli, peak_halueval, distance_blocks, distance_frac, distance_boot_ci_5_95_blocks, distance_boot_ci_5_95_frac, boot_pair_undefined_frac} or {defined:false, why, nulls}
  .E8_llama_context  6 keys (4 Llama-3.1 grid-B cells + 2 "…(grid-A context)" Llama-3.3 cells):
       {defined, window [0.4N,0.9N], qualifying_blocks_per_fold[5][…], counts_per_fold[5], majority_ge_10, majority_lt_5, median_train_curve list[N]}
       <== the ONLY per-block curve banked for grid B; CROSS-FITTED MEDIAN TRAINING curve (directed, training rows only) — a DIFFERENT statistic from RESULTS.json's in-sample sign-free curve.
  .E1pp_panel {grid_b{9 cells: N, peak_cf, frac}, grid_a_separate{8 cells: N, peak_cf, frac}, family_summary{llama31, mistral, gemma, "qwen(A)", "llama33(A)" -> n, mean_frac, sd_frac}}
       peak_cf = median of the 5 training-fold peaks. NO CI on any peak_cf.
  .predictions_as_registered {P6_e5_confirms false, P7_e6_ge_9_if_tested null, P8 true, P9 true, P10 "see E1'' panel (descriptive)"}
  .cell_failures  3 entries, verbatim abort reasons.

===== FROZEN BARS FOR T2 (PRE_REGISTRATION_EXPANSION.md §5/§6; denominator ALWAYS 12) =====
 E5 CONFIRM = >=10/12 AND >=3/4 in each of 3 families AND p_grid<0.05 ; WEAKEN = 8-9/12, or >=10/12 with a family/p failure ; FALSIFY = <=7/12.  ACTUAL 8/12, families 4/2/2, p 0.0005 -> WEAKEN.
 E6 gatekept on E5==CONFIRM. CONFIRM = >=9/12 AND >=4/6 per task AND >=2/4 per family AND p_grid<0.05 ; WEAKEN = exactly 8/12 or >=9/12 with guard failure ; FALSIFY = <=7/12.  ACTUAL gate closed -> NOT TESTED (descriptive 2/12).
 E7/E8/E1'' descriptive-registered, "no verdict vocabulary", no bars.
 Grid-A (PRE_REGISTRATION.md): E1 ABSOLUTE if span median(N-l*)<=2 AND span median(l*/N)>=0.03; RELATIVE if span median(l*/N)<=0.015 AND span median(N-l*)>=3; else UNDECIDED. E2 CLIFF if J>=0.15 AND J>=0.5R; GRADUAL if J<=0.08; else MIXED. E4 fires if l*<=N-2 AND A[N-1]<=A[l*]-0.05.

===== TASK 5 — VERIFICATION OF THE FIVE QUOTED NUMBERS: ALL CONFIRMED, ONE WITH A CAVEAT. NOTHING IS WRONG. =====
 * 0.897 peak at block 48 of 80 — CONFIRMED. RESULTS.json cells["anli_r1/Llama-3.3-70B-Instruct"]: .lstar=48, .peak_auroc=0.8975, .n_layers=80. Only note is rounding (0.8975 -> 0.897 or 0.898; vault and result page both use 0.897 — pick one and hold it across all seven artifacts).
 * Rungs 40 / 78 / 79 of 80 — CONFIRMED, and verified EMPIRICALLY rather than by trusting the prereg text. PRE_REGISTRATION.md line 9 defines the panel rungs as mid=N//2, N-2, N-1 => 0-indexed 40/78/79 at N=80. I then cross-checked the ACE panel's stored attention marginals against the depth curve at exactly those indices, four cells x four metrics: every value agrees to 4 dp, diff 0.0000. E.g. anli/Llama-3.3-70B panel attention[mid_js] 0.6612 == depth[40] 0.6612; attention[last_minus_1_js_kv_groups] 0.6232 == depth[78] 0.6232; attention[final_bos_mass] 0.7035 == depth[79] 0.7035. Same exact agreement for Qwen2.5-72B (40/78/79), Qwen2.5-32B (32/62/63), Qwen2.5-7B (14/26/27). Mapping certain; this also independently validates that the depth extractor reproduces the sealed panel kernel.
 * 0.816 panel readout winner — CONFIRMED as 0.8156, with a caveat (see gaps item 4). Source is OUTSIDE this lane: /Users/msrk/Documents/furnace-guard/artifacts/modal_profiles_ext/profiles_ext/anli_r1/Llama-3.3-70B-Instruct.profile.json, .primary_full_panel.winner = "Readout neg_shadow_logvol_r1 @ step 0", .winner_marginal = {auroc 0.8156, sign -1}. Top of that model's 29-cell panel (next: Fusion fusion_rank_mean_geom 0.8097, then Readout null_ratio_post_rank1 0.7798). Deployed OOB median for the same winner is 0.7954 [0.6995, 0.8728], winner_stability 0.511.
 * 44/80 blocks over envelope, envelope ceiling 0.615 — CONFIRMED by counting the stored arrays (not recomputing): exactly 44 blocks have A > env; envelope max 0.6145.
 * E5 WEAKEN 8/12 and pooled dip CI [0.123, 0.201] — CONFIRMED: E5.count_of_12 8, decision "WEAKEN", guards count_ge_10 false / families_ge_3of4 false, p_grid_pooled 0.0004997501249375312; bootstrap_aggregates.pooled.ci_5_95_alldef [0.12276666666666666, 0.2012333333333333].

## grid_a_vs_b_difference

STORAGE (npz): IDENTICAL. Both grids bank scores (200, N, 4) over all four metrics, every block, every row, zero NaN — verified array-by-array. The claim that grid B banks a reduced metric set is FALSE. Grid B's meta is a SUPERSET of grid A's (adds revision_pinned, prompt_row_hashes, capture_mode, device_param_counts, cuda_peak_alloc_bytes, wrapper_class, attn_dtypes_seen, load_notes, prompt_manifest_file_sha256) and adds a status.json sidecar. Grid B's only true data deficit is that 3 of 12 core cells never produced an npz at all.

SCORED-JSON COVERAGE: this is where the real asymmetry lives, and it is severe.
  Banked for grid A (RESULTS.json), per cell: 4 full per-block sign-free AUROC curves, the 97.5pct shuffled-label envelope, l*, l* bootstrap CI, peak AUROC value, E2 {J, R, baseline, shape}, mid_region_median, E3 early blocks, E4 boolean, max_auroc_any_block.
  Banked for grid B (GRID_B_RESULTS.json), per cell: L, failed/reason, e5 {Δ_cf, 5 fold contrasts, 5 fold peaks, success, perm_p, bootstrap CI}, e6 {J_cf, R_cf, 5 fold J/R/j*, null q95, perm_p_J, success}. THAT IS ALL.
  NOT banked for grid B, for ANY cell: sign-free AUROC curve, shuffled-label envelope, peak AUROC VALUE, mid-region median, E2 rise-shape classification, per-cell peak-location bootstrap CI, E3 early-block set, max AUROC.
  Partial exception: E8_llama_context.median_train_curve gives a length-N per-block curve for exactly 4 grid-B cells (Llama-3.1-8B and 70B x both tasks) plus 2 Llama-3.3 grid-A context cells — 6 curves total — and it is the cross-fitted median TRAINING curve, a different statistic from the full-sample in-sample sign-free curve. Mistral-Small, Mistral-Medium and gemma-3-12b have NO curve of any kind.

THREE DISTINCT PEAK ESTIMATORS ARE IN PLAY — the single largest trap in this contract:
  (1) RESULTS.json .lstar — in-sample sign-free argmax over N blocks, grid A only. Has a bootstrap CI (.lstar_boot_ci).
  (2) RESULTS.json .lstar_frac_median — bootstrap-MEDIAN l* divided by N. This is what RESULTS.md's "l*/N" column prints, and it is NOT lstar/N. Llama-3.3-70B/anli: lstar=48 so lstar/N=0.600, but lstar_frac_median=0.8625. The registered result page already footnotes the discrepancy.
  (3) E1pp_panel .peak_cf — median of the 5 cross-fitted TRAINING-fold peaks; the only estimator computed identically on BOTH grids, hence the only one legal for a shared FIG4 axis. It has NO CI on either grid. It differs from lstar: e.g. anli/Qwen2.5-32B lstar=56 but peak_cf=57.
  Same story for the dip: registered grid-A E4 is a BOOLEAN; grid-B E5 is a MAGNITUDE Δ_cf with a CI. The only like-for-like magnitude+CI pairing across both grids is RESCORE_GRID_A.json.e5 (grid A) vs GRID_B_RESULTS.json.cells[].e5 (grid B) — identical machinery (score_grid_b.py imports rescore_grid_a.py functions unmodified, same RS_SEED 20260817, same fold maps, prereg §4 says "exactly the grid-A rescore implementation") — but grid A's is explicitly a PRE-FREEZE CALIBRATION artifact, not the registered grid-A verdict.

WHAT MUST BE RECOMPUTED FOR THE PLANNED SUITE (i.e. what does not exist):
  - grid-B per-block sign-free AUROC curves (FIG1, 5 of 9 cells; 4 Llama cells could instead use the E8 training curve at the cost of mixing statistics)
  - grid-B shuffled-label envelopes (FIG1 shading) — NOT a pure re-render: 200 fresh label permutations under an RNG seed convention (_stable_seed(task, slug, "env"), SEED 20260816) never registered for grid-B slugs; a new Monte-Carlo statistic on confirmatory cells
  - grid-B peak AUROC values, mid-medians, E2 shapes (T1)
  - any bootstrap CI on peak_cf, either grid (FIG4 as specced)

## derivable

FIG2 (MONEY FIGURE) — Llama-3.3-70B / anli_r1. FULLY AVAILABLE, no derivation risk.
  curve: RESULTS.json cells["anli_r1/Llama-3.3-70B-Instruct"].curves_signfree_auroc["final_js_no_bos"], list[80] — DIRECT.
  envelope shading: .envelope_q97_5_primary, list[80], max 0.6145 — DIRECT. 44 of 80 blocks sit above it (counted from stored arrays; matches the registered "44/80").
  peak annotation: .lstar = 48, .peak_auroc = 0.8975 — DIRECT.
  rung markers at blocks 40, 78, 79 — DIRECT (N=80; mid=N//2=40, last_minus_1=N-2=78, final=N-1=79, 0-indexed). Curve values there: A[40]=0.6196, A[78]=0.5712, A[79]=0.5512 — all three far below the peak, which is the whole point of the figure.
  0.816 reference line — AVAILABLE but EXTERNAL to the lane (exact value 0.8156); see gaps item 4 for the two mandatory caption obligations.
  Optional and fully available: the ACE panel's own attention marginals at those three rungs, from the profiles_ext profile JSON, if you want the rungs drawn as actual panel points rather than bare vertical rules.

FIG1 — full grid, all cells, both tasks, envelope shaded, grids distinguished. SPLIT VERDICT.
  Grid A (8 cells): FULLY AVAILABLE — 8 curves + 8 envelopes direct from RESULTS.json. Render hollow per rail 5.
  Grid B (9 evaluable cells): MISSING. No curves, no envelopes banked. 4 of 9 (Llama-3.1-8B/70B x 2 tasks) can be drawn from E8_llama_context.median_train_curve, but that is a cross-fitted median TRAINING curve — if used it needs its own panel row and its own caption sentence, never grid A's line style. The remaining 5 (Mistral-Small/halueval, Mistral-Medium x2, gemma-3-12b x2) have nothing. The 3 aborted cells must render as explicitly undefined panels carrying their verbatim abort reason.
  => As specced, FIG1 is NOT buildable from banked values. Two honest options: (a) grid-A-only figure plus a grid-B companion built from E8 for the 4 Llama cells; (b) escalate to MK for permission to compute grid-B sign-free curves + envelopes — new statistical work on confirmatory cells, not rendering.

FIG3 — terminal-dip forest, magnitude + CI, grid A hollow / grid B filled. AVAILABLE, with one mandatory caption obligation.
  Grid B (12 rows): GRID_B_RESULTS.json cells[k].e5.delta and .e5.boot_ci_5_95_conditional — DIRECT for the 9 evaluable; 3 aborted rows render undefined (delta null, ci null, verbatim reason).
    anli/Llama-3.1-8B 0.1405 [0.0188, 0.2851]; anli/Llama-3.1-70B 0.1405 [0.04997, 0.18805]; anli/Mistral-Medium 0.2930 [0.1480, 0.4651]; anli/gemma-3-12b 0.2180 [0.09298, 0.36253]; halueval/Llama-3.1-8B 0.1060 [-0.03063, 0.16758]; halueval/Llama-3.1-70B 0.2010 [0.09650, 0.26353]; halueval/Mistral-Small 0.0045 [-0.06258, 0.07202] (the one true miss, success=false); halueval/Mistral-Medium 0.1510 [-0.01053, 0.21955]; halueval/gemma-3-12b 0.3055 [0.19595, 0.45253].
    pooled band: [0.12277, 0.20123].
  Grid A (8 rows): RESCORE_GRID_A.json cells[k].e5.delta + .e5.boot_ci_5_95_conditional — same estimator, like-for-like. Δ_cf: anli 7B 0.2420 / 32B 0.1460 / 72B 0.2115 / Llama-3.3 0.4160; halueval 7B 0.2020 / 32B 0.2610 / 72B 0.2640 / Llama-3.3 0.1105.
  MANDATORY: the grid-A rows are the PRE-FREEZE CALIBRATION rescore, not the registered grid-A E4 verdict (which is a boolean, 8/8 fired, no magnitude, no CI). Caption must say so. Rail 3 also applies to the grid-B side: WEAKEN 8/12 against a frozen >=10/12 plus family bars that were missed — do not round up, soften, or re-derive a friendlier bar.
  Derivable alternative, offer only as annotation: registered grid-A dip magnitude = peak_auroc - curve[N-1] = 0.2350 / 0.1803 / 0.2414 / 0.3463 (anli 7B/32B/72B/Llama-3.3) and 0.1897 / 0.2589 / 0.2556 / 0.1500 (halueval). Pure arithmetic on stored values, and exactly the quantity E4 thresholds at 0.05 — but NO CI, so it cannot carry a forest plot.

FIG4 — peak fraction vs N scatter with bootstrap CIs. POINTS AVAILABLE, CIs MISSING.
  Points: E1pp_panel.grid_b (9 cells) and E1pp_panel.grid_a_separate (8 cells), each {N, peak_cf, frac} — DIRECT, and the ONLY estimator computed identically on both grids. Prereg §6 already mandates "plotted separately, never pooled numerically", coinciding exactly with rail 5.
  CIs: NOT AVAILABLE on either grid under this estimator. Grid B has no per-cell peak bootstrap at all (E7's CIs are on cross-task DISTANCE, a different quantity). Grid A's .lstar_boot_ci belongs to estimator (1), not peak_cf — attaching it to a peak_cf point silently splices two estimators; do not.
  Available substitutes that are still already-scored values: E1pp_panel.family_summary mean_frac/sd_frac (llama31 0.408+/-0.254, mistral 0.878+/-0.133, gemma 0.635+/-0.250, qwen(A) 0.849+/-0.058, llama33(A) 0.463+/-0.194), and the 5 raw fold peaks per cell in cells[k].e5.fold_peaks, shown as a fold spread.
  Rail 9: label this "no rule established", never "non-law proven". P10 is registered as descriptive with no law/non-law claim permitted.

FIG5 — cliff onset, rise-in-one-block vs total rise, DISCOVERY ONLY, grid A. AVAILABLE, one structural trap.
  Registered grid-A E2: RESULTS.json cells[k].E2.max_adjacent_jump (J) vs .E2.rise (R), plus .E2.baseline_mid and .E2.shape — DIRECT for all 8.
    (J, R, shape): anli 7B (0.1830, 0.2876, CLIFF); 32B (0.2445, 0.3200, CLIFF); 72B (0.2325, 0.2984, CLIFF); Llama-3.3 (0.2665, 0.2484, CLIFF); halueval 7B (0.2322, 0.3251, CLIFF); 32B (0.3507, 0.2924, CLIFF); 72B (0.2351, 0.3010, CLIFF); Llama-3.3 (0.0000, 0.1412, GRADUAL).
  TRAP: halueval_qa/Llama-3.3-70B's J = 0.0000 is STRUCTURAL, not measured. Its l*=26 sits below ceil(0.5*80)=40, so the search segment A[40:27] is empty and depth_score.py assigns J=0.0 via its `if len(seg)>=2 else 0.0` fallback. Plotting it as a genuine zero-jump point is a factual error — render as an undefined/empty-window marker. Same structural condition makes that cell E6-undefined in the cross-fit.
  Cross-fitted alternative: RESCORE_GRID_A.json cells[k].e6 J_cf / R_cf (7/8 succeed; the sole miss is the same empty-window case).
  Rail 4 is absolute: E6 was NEVER TESTED — gatekept behind an E5 CONFIRM that did not happen. Every cliff caption must carry DISCOVERY ONLY.

T1 — verdict table (peak block, CI, peak value, mid-median, E2, E4/dip per cell). GRID A COMPLETE, GRID B STRUCTURALLY INCOMPLETE.
  Grid A, all columns DIRECT from RESULTS.json: lstar, lstar_boot_ci, peak_auroc, mid_region_median, E2.shape, E4_terminal_dip (+ Δ_cf and CI from RESCORE_GRID_A.json for a magnitude column).
    Full grid-A row set (N, l*, CI, peak, mid-med, E2, dip-bool): anli 7B (28, 22, [22,25], 0.8336, 0.5460, CLIFF, Y); anli 32B (64, 56, [56,62], 0.8655, 0.5455, CLIFF, Y); anli 72B (80, 66, [66,77], 0.8376, 0.5392, CLIFF, Y); anli Llama-3.3 (80, 48, [48,74], 0.8975, 0.6491, CLIFF, Y); halueval 7B (28, 26, [26,26], 0.8713, 0.5462, CLIFF, Y); halueval 32B (64, 56, [35,61], 0.9106, 0.6182, CLIFF, Y); halueval 72B (80, 63, [63,63], 0.8961, 0.5951, CLIFF, Y); halueval Llama-3.3 (80, 26, [26,48], 0.8621, 0.7209, GRADUAL, Y).
  Grid B: peak block = E1pp peak_cf (or median of e5.fold_peaks) DIRECT; dip = e5.delta + CI DIRECT; E5 outcome DIRECT.
    peak-location CI MISSING. peak AUROC VALUE MISSING. mid-median MISSING. E2 rise shape MISSING (E2 was never defined for grid B; nearest registered analogue is E6 J_cf/R_cf, a different statistic and NOT TESTED).
  => T1 must be two tables, or one table with grid-B cells showing explicit em-dashes in four columns plus a footnote saying those quantities were never scored for grid B. Do not silently blank them; do not backfill by computing.

T2 — registered endpoint ledger (E5/E6/E7/E8/E1'' with frozen bars and actual outcomes). FULLY AVAILABLE.
  Bars: PRE_REGISTRATION_EXPANSION.md §5/§6 (quoted verbatim above). Outcomes: GRID_B_RESULTS.json E5/E6/sensitivity/E7_peak_distance/E8_llama_context/E1pp_panel/predictions_as_registered. Grid-A calibration rates (E5 8/8, E6 7/8) from RESCORE_GRID_A.json.grid._POOLED; the prereg itself calls them "calibration, not part of any grid-B count".
  E1'' has no bar by construction ("No law/non-law claims"). Include the leave-Medium-out sensitivity (E5 6/10, E6 1/10) and the evaluable-only recount (8/9, 2/9) — both registered, both explicitly non-verdict. Include all three cell_failures verbatim; the two gemma-3-27b rows are the js_no_bos instrument-domain boundary (rail 11) and must read as undefined, never zero, never dropped.
  P8/P9 both HIT: E8 counts_per_fold for anli/Llama-3.1-70B = [24,25,24,23,22] (majority_ge_10 true) vs anli/Llama-3.1-8B = [4,3,6,4,5] (majority_lt_5 true); grid-A Llama-3.3-70B context = [29,27,24,28,27] anli / [30,31,25,28,30] halueval.

## gaps

Ordered by how badly each blocks the suite.

1. BLOCKING — FIG1 grid-B curves and envelopes DO NOT EXIST. GRID_B_RESULTS.json banks no per-block AUROC curve and no shuffled-label envelope for any cell. Five of nine evaluable grid-B cells (Mistral-Small/halueval, Mistral-Medium x2, gemma-3-12b x2) have no curve anywhere. The other four (Llama-3.1-8B/70B x 2 tasks) exist only as E8_llama_context.median_train_curve, a cross-fitted median TRAINING curve, not the in-sample sign-free full-sample curve grid A's panels show. Producing a matching grid-B curve means running signfree_auroc_matrix over the grid-B npz; producing the shading means 200 fresh label permutations under a seed convention (_stable_seed(task, slug, "env"), SEED 20260816) never registered for grid-B slugs. The curve half is arguably pure scorer-defined arithmetic; the ENVELOPE half is NOT — it is a new Monte-Carlo statistic on confirmatory cells. Per rail 1 I am reporting both as a gap rather than assuming. NEEDS MK DECISION.

2. BLOCKING AS SPECCED — FIG4 has no bootstrap CIs. peak_cf carries no interval on either grid. Grid B has no per-cell peak-location bootstrap at all; E7's intervals are on cross-task peak DISTANCE, a different quantity. Grid A's .lstar_boot_ci belongs to the in-sample argmax estimator, not peak_cf, so splicing it onto peak_cf points would silently mix estimators. Either drop the CIs and show the five per-fold peaks (cells[k].e5.fold_peaks) as an honest spread, or respec the figure.

3. BLOCKING FOR FOUR T1 COLUMNS — grid B has no peak AUROC value, no mid-region median, no E2 rise-shape, no peak-location CI. None were ever scored for grid B. Backfilling any of them is new computation on confirmatory cells.

4. EXTERNAL, NOT MISSING BUT NOT IN THIS LANE — the 0.816 reference line. It is not in the depth-curve lane at all. Source: /Users/msrk/Documents/furnace-guard/artifacts/modal_profiles_ext/profiles_ext/anli_r1/Llama-3.3-70B-Instruct.profile.json, .primary_full_panel.winner_marginal.auroc = 0.8156. Two caption obligations: (a) 0.8156 is the IN-SAMPLE full-panel marginal; the deployed OOB median for that same winner is 0.7954 [0.6995, 0.8728] with winner_stability 0.511 — using in-sample 0.8156 is the correct like-for-like against the depth curve's in-sample 0.8975, but the choice must be stated or a reader will think 0.897 beats a bootstrapped bound. (b) It is a READOUT cell read at the commit step — a different instrument from the per-layer ATTENTION curve it is drawn across; FIG2 is a cross-instrument comparison and must say so. Also worth noting: 0.8156 is a full-PANEL winner (29 cells, readout + attention + fusion), so "the peak beats the panel readout winner" is the accurate phrasing, not "beats the panel".

5. RENDERING HAZARD, not missing data — halueval_qa/Llama-3.3-70B's E2.max_adjacent_jump = 0.0000 is a structural empty-window fallback (l*=26 < ceil(0.5N)=40, so the jump-search segment is empty and depth_score.py returns 0.0), NOT a measured absence of a jump. FIG5 must not plot it as a real zero.

6. RENDERING HAZARD — the three grid-B aborted cells are two different failure classes and must not be conflated. gemma-3-27b x2 = sealed js_no_bos returns None at row 0 block 3 under total BOS-sink collapse (rail 11: render as explicitly undefined with the instrument-domain reason, never zero, never dropped). anli/Mistral-Small-3.2 = operational abort, a missing smoke prompt manifest, nothing to do with the instrument. All three count as FAILURES in the frozen 12-cell denominator regardless. gemma-3-27b's N=62 is known only from the registry field cells[].L, never from an npz.

7. NAMING PRECISION — RESULTS.md's "l*/N" column is lstar_frac_median (bootstrap-median-based), NOT lstar/N. For Llama-3.3-70B/anli it prints 0.863 while the point fraction is 48/80 = 0.600, and E1pp_panel uses peak_cf/N = 0.6. Anyone building FIG4 from the .md twin instead of the .json will get it wrong.

8. ROUNDING — the quoted 0.897 is exactly 0.8975. Under round-half-up that is 0.898. Recommend printing 0.897 (matching the vault and the registered result page) or the full 0.8975, but pick one and hold it across all seven artifacts.

NO OTHER GAPS. All five quoted numbers check out (four exactly, 0.816 with the caveat above) — see the verification section at the end of scored_json_fields. Rails 6 and 7 need no action: every cell in this lane is torch/Modal, comparable=false, so nothing here can be pooled with the sealed MLX panels, and there is no 405B field anywhere to ignore.
