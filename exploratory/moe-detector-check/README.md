# Local commit-moment detector check

Runs the real torch `_Capture` / `_forward`, D0 prompt-only ACE, and D1
prompt-plus-argmax-token PRI/readout on one CPU, followed by the unchanged sealed
selector. This is an exploratory architecture/instrument check, not a reproduction
of MLX 4-bit sealed results. No Modal or CUDA is used.

From this directory, in an existing environment with torch, **Transformers==5.19.0**
(enforced to match the audit), accelerate, huggingface_hub, NumPy, pandas,
scikit-learn, scipy, safetensors and tokenizer dependencies available:

```sh
python3 local_detector.py --model-id ibm-granite/granite-3.1-1b-a400m-instruct --revision main --device cpu --precision bf16 --task anli_r1 --n 200 --max-prompt-tokens 512
python3 local_detector.py --model-id Qwen/Qwen3-1.7B --revision main --device cpu --precision bf16 --task anli_r1 --n 200 --max-prompt-tokens 512
```

Run sequentially after the downloads finish. Loading is offline from the default
Hugging Face cache (or the existing HF cache environment settings). The cached
revision resolves once to a snapshot SHA; both tokenizer and weights load from
that snapshot, and the report records the SHA and installed library versions.
For a repeat, pass that SHA to `--revision` and a fresh directory using
`--out runs/<model-slug>/<run-name>`. Existing results are never overwritten.
`--precision fp32` is supported but uses more memory. Qwen3 always receives
`enable_thinking=False`; there is no separate thinking flag.

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

**verification: not run by Codex**. Only source reading and writing were performed;
no code, tests, Python, installs, models, git or Modal were run.
