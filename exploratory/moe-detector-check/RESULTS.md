# MoE commit-moment detector check — results (2026-10-12 → 2026-10-13)

Exploratory instrument check. Not a registered result, and not byte-comparable with the sealed MLX
4-bit panels (torch bf16, chat-template prompts). Never pool these cells with sealed cells.

**Question.** Does the unchanged commit-moment detector run cleanly on mixture-of-experts (MoE)
models, and does it clear the stated target there? The detector is fixed: layers N//2, N-2 and N-1;
21 ACE cells and 6 readout cells; the D0/D1 split; the sealed selector with 2,000 bootstraps and
seed 20260612.

**Target (stated by MK).** Median out-of-bag (OOB) AUROC ≥ 0.55 **and** a 95% CI lower bound
> 0.5. Task: ANLI R1, the fixed 200-row file `stage_b/data/anli_R1_seed20260612_n200.jsonl`.

## Results

| Model | Type | Where | Best cell (primary, full panel) | OOB AUROC [95% CI] | Target | Winner stability |
|---|---|---|---|---|---|---|
| Granite-3.1-1B-A400M-instruct | MoE, 32 experts | local CPU | `mid_bos_mass` | 0.633 [0.428, 0.739] | **miss** (CI lo ≤ 0.5) | 0.76 |
| Qwen3-1.7B | dense control | local CPU | `final_js_no_bos` | 0.605 [0.443, 0.710] | **miss** (CI lo ≤ 0.5) | 0.45 |
| Qwen1.5-MoE-A2.7B-Chat | MoE, 60 experts + shared | Modal A100-80GB | `final_v_norm_lastq_weighted` | **0.710 [0.602, 0.799]** | **meets** | 0.37 |
| Qwen3-30B-A3B | MoE, 128 experts | Modal A100-80GB | — | **not scored** | — | — |
| OLMoE-1B-7B-0924-Instruct | MoE, 64 experts | — | — | **not run: not compatible** | — | — |

The secondary (geometric-only) endpoint picks the same winner in every scored model: Granite 0.634
[0.445, 0.739], Qwen3-1.7B 0.605 [0.447, 0.710], Qwen1.5-MoE 0.715 [0.621, 0.804]. Shuffled-label
controls pass in all three scored models: 0 of 3 permutations exclude chance on either endpoint.
Each scored run extracted 200/200 rows with 0 unavailable and 0 non-finite cells (any one would have
aborted the run) and committed YES or NO on 200/200 rows. Qwen1.5-MoE answers 147/200 correctly.

Qwen1.5-MoE's winning cell is chosen in only 37% of bootstrap resamples. The AUROC interval is
procedure-level (it already includes that selection noise), but "which cell carries the signal" is
not stable for this model.

**Qwen3-30B-A3B stopped at the preregistered behaviour gate, as the protocol requires.** With
`enable_thinking=False` (rendered prompt verified to end in the empty think block), its first
token on validation rows 0 and 1 is `Answer`, not YES or NO. The model does not commit to a label at
the measured position, so the cell cannot be scored. Every numerical check passed (below). No retry
and no prompt change were made. Changing the prompt or the checkpoint (for example the
non-thinking `Qwen3-30B-A3B-Instruct-2507`) would be a new decision and needs an amendment.

## Instrument checks (worst case over every validation and extraction forward)

| Model | Layer (index) | Forwards | Reconstruction cosine ≥ | Reconstruction rel. L2 ≤ | Row-sum error ≤ | Future-token leak | Final-norm rel. L2 ≤ | Logit rel. L2 ≤ | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Granite-3.1-1B-A400M | mid (12) / N-2 (22) / final (23) | 404 | 0.9999992 | 0.00125 | 0.0033 | 0 | 0.0037 | 0.0099 | all |
| Qwen3-1.7B | mid (14) / N-2 (26) / final (27) | 404 | 0.9999996 | 0.00083 | 0.0032 | 0 | 0.0032 | 0.0049 | all |
| Qwen1.5-MoE-A2.7B-Chat | mid (12) / N-2 (22) / final (23) | 404 | 0.9999988 | 0.00150 | 0.0033 | 0 | 0.0039 | 0.0091 | all |
| Qwen3-30B-A3B | mid (24) / N-2 (46) / final (47) | 4 (validation only) | 0.9999954 | 0.00443 | 0.0030 | 0 | 0.0025 | 0.0029 | all |

Bounds are rounded outward from the exact extrema (`summarize_runs.py` prints them), so each
displayed bound holds literally.

- **Reconstruction:** `o_proj(A · repeat_kv(V))` against the native attention output, direction
  **and** magnitude.
- **Logit parity:** `W_u @ sealed_RMSNorm(h, eps=1e-6)` against the model's own logits. Granite
  uses `W_u / 6`, because its config sets `logits_scaling = 6`. The Qwen models have no logit
  scaling, so they use bare `W_u`.
- **Row sums and causal leakage:** checked over every query row, not only the last.
- **Thresholds (bf16):** cosine ≥ 0.999; relative L2 ≤ 2% (reconstruction, logits) and ≤ 1% (final
  norm); row-sum error ≤ 0.01; future-token probability ≤ 1e-7.
- **Argmax agreement** between the detector's logits and the native logits was 96.8% (Granite),
  100% (Qwen3-1.7B), 98.8% (Qwen1.5-MoE) and 100% (Qwen3-30B validation). Every disagreeing
  forward still passed parity (worst logit error 0.99%, under the 2% tolerance), which suggests bf16
  near-ties. Top-token margins were not recorded, so near-ties are likely but not shown. The
  committed token always comes from the native logits.

## Not run

- **OLMoE-1B-7B-0924-Instruct is not compatible with the preregistered readout** (MK decision
  2026-10-13). Its pinned config (`7f1c97f`) sets no epsilon, so Transformers 5.19.0 uses the OLMoE
  default `rms_norm_eps = 1e-5` (`configuration_olmoe.py:72`). The sealed readout fixes the final
  RMSNorm at `eps = 1e-6` (`modal/seal/pri_runtime.py:1174`), so running it would change the readout
  instrument. The wrapper registry marks it `not_compatible` and refuses every launch path. $0 spent.
- Hybrid and linear-attention models (Qwen3.5, Nemotron), MLA models (DeepSeek) and attention-sink
  models (gpt-oss) are out of scope. They fail the detector's structural requirements or change ACE
  semantics.

## Modal spend (hard cap $10 for the whole task)

Measured = container wall clock × list price (A100-80GB $0.000694/s, CPU $0.0000131/core/s,
memory $0.00000222/GiB/s, dated 2026-10-10). Billed = Modal's per-app billing report; see
`modal_spend_ledger.json` and the billing note below.

| Run | GPU | Container wall (s) | Ceiling (USD) | Measured (USD) | Outcome |
|---|---|---:|---:|---:|---|
| Download Qwen1.5-MoE (28.6 GB) | none (4 CPU) | 69.2 | 0.200 | 0.005 | complete |
| Smoke Qwen1.5-MoE, n=12 | A100-80GB | 833.0 | 0.799 | 0.725 | budget watchdog stopped calibration; all checks passed |
| Full Qwen1.5-MoE, n=200 | A100-80GB | 1,318.5 | 2.400 | 1.147 | complete |
| Download Qwen3-30B-A3B (61.1 GB) | none (4 CPU) | 211.2 | 0.250 | 0.015 | complete |
| Full Qwen3-30B-A3B, n=200 | A100-80GB | 75.7 | 3.299 | 0.066 | behaviour gate failed; not scored |
| **Total** | | | 6.949 | 1.957 | |

**Billed by Modal: $2.026 in total** (per-app hourly billing report for 16:00–17:00 UTC,
reconciled into the ledger by `reconcile`): downloads $0.023 and $0.051, smoke $0.728, full
Qwen1.5-MoE $1.155, full Qwen3-30B-A3B $0.069. The local command apps that only host the
entrypoints billed $0. Downloads bill above their measured cost because image build and container
startup fall outside the function's stopwatch.

GPU time: 0.619 A100-80GB hours (2,227.2 s, measured container wall clock). After the runs, the
cached weights (89.7 GB) were deleted from the `moe-detector-cache` volume to stop storage charges:
`modal volume rm -r moe-detector-cache /hub` reported success, and a listing of `/` afterwards
showed only `downloads` and `results`. Run artifacts remain under `/results`. A repeat must re-download first (~$0.02, ~5 min). The ledger still
records the downloads as complete, so a `run` without a fresh download fails fast at "config.json is
not cached".

## Failed attempt, kept

The smoke run (`runs/Qwen1.5-MoE-A2.7B-Chat/modal-20261013T161005547953Z-n12/`) confirmed the
wrapper end to end. Payload hashes matched; validation passed; 12/12 rows extracted with 0
non-finite cells; artifacts were returned and the watchdog fired as designed. It then ran out of
its $0.80 budget during the sealed calibration's shuffled-label controls. Its report records
`KeyboardInterrupt` from the watchdog, not a detector fault.

Phase timings come from Modal's timestamped container logs (`modal app logs <app id> --timestamps`;
app IDs are in the ledger), not from the reports, which record only total wall time. Smoke: a cold
28.6 GB weight load took 7 min 33 s (about 63 MB/s from the volume). Full Qwen1.5-MoE run: the warm
load took 16 s, the 200 rows took 139 s (0.7 s per row), and the sealed calibration, which is
CPU-bound, took 18 min 47 s while the GPU sat idle.

## Files

- `local_detector.py` — the detector adapter. The CPU results were produced by sha256
  `5b5775da93c3…` (commit `60944ad`). The CUDA edit adds the device option, device-aware
  reconstruction tensors, the Qwen3-family `enable_thinking=False` fix (`qwen3_moe` was previously
  missed, so thinking would have stayed on) and package provenance. No detector logic changed.
- `modal_detector.py` — Modal 1.2.6 wrapper with the spend guard (`plan`, `download`, `smoke`,
  `run`, `reconcile`). The image pins torch 2.14.1+cu126 and Transformers 5.19.0. Checkpoint and
  tokenizer share the pinned snapshot SHA. Runs use bf16, one A100-80GB and eager attention.
- `modal_spend_ledger.json` — the per-run spend ledger (ceilings, measured, billed).
- `summarize_runs.py` — builds the tables above from `report.json` files.
- `runs/<model>/…/report.json`, `matrix.npz`, `stdout.log`, `modal_run.json` — raw artifacts.
