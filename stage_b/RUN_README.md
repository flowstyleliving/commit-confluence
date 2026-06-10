# Stage B - launch instructions

The harness `run_seal.py` is complete and smoked end-to-end (ACE pass + fresh readout pass +
sample_idx merge + dual-endpoint nested-OOB selection + profile emit + endpoint eval). What remains
before the **registered fresh-seed seal** is: (1) the Codex adversarial sign-off, and (2) **fresh
data files** (the one external prerequisite).

## The fresh-data prerequisite

The ACE seal and the RPV run BOTH used the sealed `seed 20260526` ANLI/TriviaQA samples. Reusing
them is a *preview*, not a registered seal (example-level reuse can flatter the selection). A true
fresh seal needs NEW examples at a fresh seed:

- **ANLI R1**, n=200, fresh seed, same prompt template as the sealed run.
- **TriviaQA paired**, n≈100, fresh seed. Documented build params (from the sealed manifest):
  `hf_dataset=trivia_qa`, `hf_config=rc.wikipedia`, `split=validation`, `shuffled=true`,
  paired correct/wrong, label 0=correct(YES) / 1=wrong(NO),
  template: `"Instruction: ... Question: {question}\nProposed answer: {answer}\nAnswer:"`.

Produce two jsonl files (same schema as `t0-morphology-furnace/experiments/t0-sealed/2026-05-26/data/*.jsonl`:
one `{"prompt"/"text", "label"}` per line) at a NEW seed (e.g. 20260612). The HF data builder lives in
the source program; regenerate there, then point the harness at the files.

## Launch (one command, after Codex sign-off + fresh data)

```bash
/Users/msrk/Documents/t0-morphology-furnace/.venv/bin/python stage_b/run_seal.py \
  --seed 20260612 \
  --anli    /path/to/anli_R1_seed20260612_n200.jsonl \
  --triviaqa /path/to/triviaqa_paired_seed20260612_n100.jsonl \
  --out-dir stage_b/profiles --nboot 2000
```

Emits `stage_b/profiles/<task>/<model>.profile.json` (one CalibrationProfile per cell) and
`stage_b/profiles/SUMMARY.json` with the two endpoints:
- **PRIMARY** (full panel incl. confidence): deployable on >= 19/20 cells.
- **SECONDARY** (geometric-only): deployable on >= 17/20 cells.

## Preview mode (no fresh data, sealed-seed reuse)

To preview the dispatcher on the sealed-seed data NOW (not a registered result):

```bash
.../python stage_b/run_seal.py --seed 20260610 --allow-sealed-data --out-dir stage_b/preview_profiles
```

`SUMMARY.json` will carry `"is_preview": true`. Useful as a thesis-level sanity check; the registered
seal still requires fresh data.

## Compute notes

- 10 models x 2 tasks. Each cell loads the model twice (ACE collection + readout `trace_pair_features`)
  - a fusion optimization is possible later but not needed for the seal.
- All 10 cohort models are cached locally. Full run is multi-hour; run detached and watch SUMMARY.json.
- `--limit N` runs a fast smoke on the first N samples (statistically meaningless; plumbing only).
- `--models Qwen3-8B,Mistral` filters to a subset by substring.
