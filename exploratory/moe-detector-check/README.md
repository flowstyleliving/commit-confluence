# Local commit-moment detector check

Runs the real torch `_Capture` / `_forward`, D0 prompt-only ACE, and D1
prompt-plus-argmax-token PRI/readout on one CPU or CUDA GPU, followed by the unchanged sealed
selector. This is an exploratory architecture/instrument check, not a reproduction
of MLX 4-bit sealed results. The local CLI does not invoke Modal.

From this directory, in an existing environment with torch, **Transformers==5.19.0**
(enforced to match the audit), accelerate, huggingface_hub, NumPy, pandas,
scikit-learn, scipy, safetensors and tokenizer dependencies available:

```sh
python3 local_detector.py --model-id ibm-granite/granite-3.1-1b-a400m-instruct --revision main --device cpu --precision bf16 --task anli_r1 --n 200 --max-prompt-tokens 512
python3 local_detector.py --model-id Qwen/Qwen3-1.7B --revision main --device cpu --precision bf16 --task anli_r1 --n 200 --max-prompt-tokens 512
python3 local_detector.py --model-id Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --device cuda --precision bf16 --task anli_r1 --n 200 --max-prompt-tokens 512 --out runs/Qwen3-30B-A3B/cuda-check
```

Run sequentially after the downloads finish. Loading is offline from the default
Hugging Face cache (or the existing HF cache environment settings). The cached
revision resolves once to a snapshot SHA; both tokenizer and weights load from
that snapshot, and the report records the SHA and installed library versions.
For a repeat, pass that SHA to `--revision` and a fresh directory using
`--out runs/<model-slug>/<run-name>`. Existing results are never overwritten.
`--precision fp32` is supported but uses more memory. `--device cuda` loads the
entire model onto `cuda:0`; unavailable CUDA, insufficient single-GPU memory, or
parameters/buffers on another device fail without offload or multi-GPU fallback.
FP32 matmul TF32 is explicitly disabled; bf16 settings are unchanged. GPU name,
memory, CUDA runtime, TF32/bf16 reduction settings, and installed package versions
are recorded in provenance. Both `qwen3` and `qwen3_moe` receive
`enable_thinking=False`; each rendered template must end with
`<think>\n\n</think>\n\n` or the run fails. The report records row 0's rendered
prompt. `qwen2_moe` receives no thinking kwarg; there is no separate thinking flag.

The input is the existing
`../../stage_b/data/anli_R1_seed20260612_n200.jsonl`: 200 rows, label 0 means
YES expected, label 1 means NO expected. `--n` selects the first N rows and
records their indices/hash; calibration needs at least two of each label.
Chat templates are tokenized directly, unpadded, without duplicated special
tokens. Overlength prompts are rejected, never truncated. Any length rejection
aborts the run before model forwards, rather than calibrating a smaller sample.
The extra D1 token must also fit the native context limit.

Estimated bf16 process peaks at batch one / 512 prompt tokens are **3–5 GiB for
Granite** and **5–7 GiB for Qwen3**; these are audit estimates, not measurements.
The real forward retains eager attention for every layer, so attention memory
grows quadratically with prompt length. NumPy readout/SVD and CPU bf16 can be
slow. Neither elapsed time nor memory use has been verified on this Mac.

`report.json` and `matrix.npz` go under `runs/<model-slug>/`. The report includes
every imported module's `__file__` (also printed; stubs are explicit), source
hashes, snapshot SHA, versions, device/dtype, per-layer validation, norm/logit
gaps, row/cell failure counts, wall time and process peak RSS. The matrix has
the 21 sealed ACE cells plus six readout cells in reference-artifact order.
The selector also computes the two frozen fusion candidates, both endpoints'
procedure-level nested-OOB AUROC intervals, and shuffled-label controls, using
2,000 bootstraps and seed 20260612. Fusion and controls remain enabled.

Before extraction, the first two rows must pass **D0 and D1** reconstruction
on all three original targets (`N//2`, `N-2`, `N-1`), and D0 must commit YES/NO.
The same numerical checks guard every extraction forward. Thresholds:

| Check | bf16 | fp32 |
|---|---:|---:|
| Reconstruction and logit cosine, minimum | 0.999 | 0.99999 |
| Reconstruction relative L2 error, maximum | 0.02 | 0.0001 |
| Attention row-sum absolute error, maximum | 0.01 | 0.00001 |
| Future-token probability, maximum | 0.0000001 | 0.0000001 |
| Final-norm relative L2 error, maximum | 0.01 | 0.00001 |
| Logit relative L2 error, maximum | 0.02 | 0.0001 |

All attention probabilities must be finite and nonnegative; row sums and causal
leakage are checked across **all query rows**, not only the last row. Relative
error uses the native vector's L2 norm as denominator (floor `1e-12`). Maximum
absolute and RMS gaps are also reported. Required architecture paths and the
sealed RMSNorm epsilon (`1e-6`) are enforced. Unknown norms, missing projections,
or failed reconstruction/parity cause a clear failure, not replacement layers.

Granite's native logits divide by `config.logits_scaling` (6 for this checkpoint).
The adapter divides its fp32 output weight by the same value **before all sealed
readout computations**. It compares `(W_u/6) @ sealed_RMSNorm(h)` to native logits
and separately reports the uncorrected bare-weight gap. This changes the local
instrument relative to the old Modal bare-weight path and is recorded explicitly.
Neither sealed scoring code nor model weights are edited.

Unavailable/non-finite ACE or readout cells are counted, named and fatal. The
sealed ACE scorer itself maps non-finite results to `None`; those are recorded
as unavailable with `sealed_may_have_masked_nonfinite=true`, not misclassified
as observed finite values. Recovering the original failure reason is outside
this adapter's post-check boundary. The
adapter stops at the first failed row; counts describe observed failures, with
unattempted rows reported separately. It never substitutes zero, drops a bad row
into an otherwise successful calibration, or calibrates a partial matrix. An
extraction/validation failure saves completed finite rows as a marked partial
matrix; failures before capture setup produce only a report. The full-run YES/NO
commit rate must also reach the sealed 50% gate. A successful run means the
instrument checks and extraction/calibration completed. Predictive evidence
requires examining the OOB intervals and shuffled-label controls; an instrument
pass alone does not establish predictive validity or deployment readiness.

Modal isolation uses an AST allowlist of original, undecorated definitions from
`../../modal/modal_app.py`; its imports, decorators and deployment/volume body
never execute. The existing MLX import shim is reused, with both `SEAL_REMOTE`
and `CONFLUENCE_T0_REPO` explicitly pointing to `../../modal/seal`; actual sealed
module locations are checked. No files in those source directories are changed.

## Modal GPU runs

Use the installed Modal client **1.2.6**, from this directory:

```sh
modal run modal_detector.py::plan
modal run modal_detector.py::download --model qwen15
modal run modal_detector.py::download --model qwen3
modal run modal_detector.py::smoke
modal run modal_detector.py::run --model qwen15
modal run modal_detector.py::run --model qwen3
modal run modal_detector.py::reconcile
```

Downloads run on CPU into the new `moe-detector-cache` volume and are pinned to
the registry SHAs. A complete snapshot returns quickly on repeat. GPU runs require
a completed matching download in the local ledger. Each invocation uses a fresh
detector subprocess on one A100-80GB with eager attention, bf16, batch one,
unpadded/untruncated prompts capped at 512 tokens, and the same ANLI data and
sealed selector. Smoke uses the first 12 rows and still runs all 2,000 bootstraps;
full runs use 200. No detector failures are retried automatically.

The wrapper pins the supplied Python 3.12 image packages, preserves the repository
payload layout, hashes every payload file, and compares GPU-container hashes to
the reserved local hashes. It saves `report.json`, optional `matrix.npz`,
`stdout.log`, and `modal_run.json` in a fresh
`runs/<model-slug>/modal-<UTC timestamp>-n<n>/`. Report, matrix, and stdout also
persist at `/cache/results/<model-slug>/<run-name>/` on the new volume.
`modal_run.json` records the image recipe, payload hashes, GPU/driver information,
container elapsed time, exit code, and spend record. If killed before the adapter
can report, the wrapper writes an explicitly identified failure report.

The local `modal_spend_ledger.json` sits outside ignored `runs/`. A file lock
serializes wrapper commands; the ledger reserves the full ceiling before starting
the worker app. The **$10 task cap** uses billed amounts when reconciled, otherwise
measured cost, retaining the full ceiling for launches with no confirmed end.
An uncertain call keeps its reservation and requires audit; it is never silently
treated as free. Keep this ledger across repeats and use this checkout for every
launch. `--share <USD>` on download/smoke/run changes that run's share, subject to
the same cap. Timeouts are derived by rounding share/rate down, with these defaults:

| Run | Share | Timeout (s) | Ceiling / initial estimate (USD) |
|---|---:|---:|---:|
| Download Qwen1.5 | 0.20 | 2850 | 0.19995600 |
| Download Qwen3-30B | 0.25 | 3563 | 0.24998008 |
| Smoke Qwen1.5, n=12 | 0.80 | 919 | 0.79938296 |
| Full Qwen1.5, n=200 | 2.40 | 2759 | 2.39988856 |
| Full Qwen3-30B, n=200 | 3.30 | 3793 | 3.29930312 |

These are **list-price estimates dated 2026-10-10**: A100-80GB $0.000694/s,
CPU $0.0000131/core/s, and memory $0.00000222/GiB/s. Downloads request 4 CPU/8 GiB;
detection requests 8 CPU/32 GiB. Initial estimates equal ceilings because no speed
has been measured. The watchdog sends SIGINT 90 seconds before the GPU timeout,
then SIGKILL after another 60 seconds; Modal's timeout is the final backstop.
The stopwatch includes model loading, extraction, calibration, and artifact
publication within the worker invocation. Image builds, container startup/teardown,
and idle billing can differ from this compute estimate; the guard is not a provider
billing limit. `reconcile` uses hourly per-app billing and leaves partial/absent
hours pending; rerun it after the run's final full hour is reported. It prints
ceiling/measured/billed totals. `plan` and `reconcile` launch no workers or image builds.

**OLMoE-1B-7B is not compatible** (MK decision 2026-10-13): native final RMSNorm
epsilon is `1e-5`, while the preregistered readout requires `1e-6`. Its registry
status is `not_compatible`; all launch paths refuse it. Only the two specified
Qwen checkpoints are runnable.

Execution risks for Claude Code's audit: Qwen3's roughly 61 GB of bf16 weights
leave limited A100-80GB headroom for loading temporaries and retained eager
attentions; eager MoE execution can be slow. NumPy readout/SVD and the sealed
calibration remain CPU-bound while the GPU is allocated. The watchdog may therefore
stop even a numerically valid run. No thresholds, layer targets, panel cells,
calibration logic, or prompt-length policy have been relaxed to fit the budget.

**verification: not run by Codex**. Only source reading and writing were performed;
no code, tests, Python, installs, models, git or Modal were run.
