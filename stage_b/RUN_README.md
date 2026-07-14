# Stage B - launch instructions

> 📜 **HISTORICAL — this page is the pre-launch checklist, kept for provenance.**
> Both prerequisites below were met and **the registered seal RAN on 2026-06-11**
> (seed 20260612, tag `prereg-seal-20260612`). The verdict — geometric **18/20 PASS**,
> full panel **18/20 FAIL** against a ≥19 bar — is in the top-level [`README.md`](../README.md).
> Nothing here is outstanding. Read this only to see what the gate *was*; read the root
> README for what is true now.

The harness `run_seal.py` is complete and smoked end-to-end (ACE pass + fresh readout pass +
sample_idx merge + fusion cells + dual-endpoint nested-OOB selection + shuffled-label controls +
matrix persistence + profile emit + planned-cohort endpoint eval). What remained before the
**registered fresh-seed seal** was: (1) the Codex adversarial sign-off, and (2) **fresh data
files that pass the data gate** (the one external prerequisite). Both were completed.

## The fresh-data prerequisite (amendments A1 + A5)

The ACE seal and the RPV run BOTH used the sealed `seed 20260526` ANLI/TriviaQA samples. Reusing
them is a *preview*, not a registered seal - and **fresh seed != fresh examples**: the fresh draw
must be DISJOINT from the sealed examples or the out-of-sample claim dilutes. Registered counts:

- **ANLI R1**, 200 rows, fresh seed, same prompt template as the sealed run.
- **TriviaQA paired**, **100 pairs = 200 rows** (A1; was 50 pairs in the sealed era), fresh seed.
  Documented build params (from the sealed manifest): `hf_dataset=trivia_qa`,
  `hf_config=rc.wikipedia`, `split=validation`, `shuffled=true`, paired correct/wrong,
  label 0=correct(YES) / 1=wrong(NO),
  template: `"Instruction: ... Question: {question}\nProposed answer: {answer}\nAnswer:"`.

Produce two jsonl files (same schema as the sealed `*.jsonl`: one `{"prompt"/"text", "label"}`
per line) at a NEW seed (e.g. 20260612), **excluding any example whose normalized prompt appears
in the sealed files**. Then gate them:

```bash
./confluence python stage_b/check_fresh_data.py --task anli \
  --fresh /path/anli_R1_seed20260612_n200.jsonl \
  --sealed vendor/t0_core/experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl \
  --out stage_b/data_gate_anli.json
./confluence python stage_b/check_fresh_data.py --task triviaqa \
  --fresh /path/triviaqa_paired_seed20260612_n200.jsonl \
  --sealed vendor/t0_core/experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl \
  --out stage_b/data_gate_triviaqa.json
```

Both must print PASS (schema, n, balance, zero intra-dups, **zero overlap**) before launch.

## Launch (one command, after Codex sign-off + data gate PASS)

```bash
./confluence seal \
  --seed 20260612 \
  --anli    /path/to/anli_R1_seed20260612_n200.jsonl \
  --triviaqa /path/to/triviaqa_paired_seed20260612_n200.jsonl \
  --out-dir stage_b/profiles --nboot 2000
```

Emits per cell: `stage_b/profiles/<task>/<slug>.profile.json` (CalibrationProfile: dual
endpoints, shuffled-label controls, fusion meta, module hashes + model snapshot SHA) and
`<slug>.matrix.npz` (the merged 27-cell score matrix - input to the E1-E3 analyses), plus
`stage_b/profiles/SUMMARY.json` with the two endpoints evaluated over the **planned 20-cell
cohort** (A4: errored cells count as NOT deployable; `incomplete` flagged):
- **PRIMARY** (full 29-cell panel incl. confidence + fusion): deployable on >= 19/20.
- **SECONDARY** (geometric-only): deployable on >= 17/20.

Guards: sealed-CONTENT files (sha256) and pilot seeds {20260512, 20260526, 20260610, 20260611}
are refused without `--allow-sealed-data`, which forces `is_preview` (C3). On a strict run the
harness ALSO runs the fresh-data gate in-process for every task file and refuses to launch on
any hard failure (M1, A5), and requires a zero-drop sample denominator (M4: every planned
sample must score, else the cell errors). `--resume` skips cells whose profile already exists,
but **only resume the exact same registered command** (same seed, nboot, data files, code):
a resumed profile is provenance-validated (seed / nboot / model / task / data-sha256 /
module-hashes / matrix presence) and any drift turns that cell into an error → forces FAIL,
so a stale/preview/old-code profile can never be folded into the registered cohort (M3).

## After the run - pre-registered descriptive analyses (A7, zero new forwards)

```bash
./confluence analyze --profiles-dir stage_b/profiles \
  --out stage_b/universality.json
```

E1 LOMO universality probe (+ sign-stability audit), E2 task-transfer matrix, E3
label-efficiency curve. All descriptive, none gate the seal.

## Preview mode (no fresh data, sealed-seed reuse)

```bash
./confluence seal --seed 20260610 --allow-sealed-data --out-dir stage_b/preview_profiles
```

`SUMMARY.json` carries `"is_preview": true`. Useful as a thesis-level sanity check; the
registered seal still requires gated fresh data.

## Compute notes

- 10 models x 2 tasks. Each cell loads the model twice (ACE collection + readout
  `trace_pair_features`) - a fusion optimization is possible later but not needed for the seal.
- Selection cost per cell is CPU-trivial vs the forwards: 2 endpoints + 2x3 shuffled-label
  permutations = 8 nested-OOB runs (~tens of seconds).
- All 10 cohort models are cached locally. Full run is multi-hour; run detached, use `--resume`
  after any crash, and watch SUMMARY.json.
- `--limit N` runs a fast smoke on the first N samples (statistically meaningless; plumbing only).
- `--models Qwen3-8B,Mistral` filters to a subset by substring.
