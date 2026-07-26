# attention-kv-tension

**Status:** `[PILOT RUN — NO-PROMOTE]`
**Lane:** ACE follow-up / `W_u`-free attention morphology
**Pre-registration:** `PRE_REGISTRATION_DRAFT.md` (written 2026-06-09, before the pilot)
**Relocated here from `t0-morphology-furnace` on 2026-07-25** — see `../README.md`

## What this lane asked

Does the ACE attention signal live in disagreement among **query heads**, or in
disagreement among the shared **key-value groups** those query heads read from?
Existing ACE measures each end (`js`, `js_kv_groups`); this lane added the missing
decomposition — tension *within* each shared KV group versus *between* groups.

## What happened

Stage 0 (sealed-profile audit) and Stage 1 (implementation contracts) completed as
written in the pre-registration. **Stage 3, the 5-model pilot, also completed** —
ANLI R1, n=200, `n_bootstrap=1000`, t=0 commit locus, finished 2026-06-08 22:47 PDT.
Its profiles are in `pilot_outputs/2026-06-09/anli_r1_t0_kv_tension_run02/`.

The pilot was then never scored against its own bars, and the whole lane sat
uncommitted for six weeks. It was scored on 2026-07-25 during the relocation.

## Verdict — does not clear the promotion bar

Reconstructed from the five `*.profile.json` files; absolute-orientation AUROC
(`max(a, 1-a)`), matching the convention used in the Stage-0 audit.

| Model | best KV cell | AUROC | vs best **routing** cell | vs best **any existing ACE** cell | OOB CI-lo |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | `js_within_kv_groups` | 0.8479 | +0.0075 | +0.0075 | 0.7382 |
| Mistral-7B | `js_kv_tension_ratio` | 0.8065 | +0.0195 | +0.0195 | 0.6931 |
| Qwen2.5-7B | `js_kv_tension_ratio` | 0.7535 | **+0.0486** | **−0.0261** | 0.6474 |
| Phi-4-mini | `js_within_kv_groups` | 0.7374 | **+0.0614** | +0.0219 | 0.5806 |
| gemma-3-4b | `js_kv_tension_ratio` | 0.6379 | −0.0521 | −0.0521 | 0.4960 |

"routing" = `js`, `js_kv_groups`, `js_no_bos`. "any existing ACE" additionally
includes `bos_mass`.

The registered promotion bar had three limbs. **Not one of them is satisfied outright:**

1. **≥ +0.03 over the best existing ACE routing comparator on ≥ 2/5 models.**
   Against *routing only*: **2/5** (Phi-4-mini +0.0614, Qwen2.5-7B +0.0486) — met.
   Against *any existing ACE cell*: **0/5** — not met. The pre-registration says
   "routing comparator," so the narrow reading passes; but see limb 3.
2. **OOB-clean, no severe coverage warning.** 4/5 clear `CI_lo > 0.50`
   (gemma-3-4b at 0.4960 does not). However **4/5 models fire `winner_unstable`** —
   including *both* models that carry the numeric win (Phi-4-mini 0.63, Qwen3-8B 0.59,
   Mistral 0.60, gemma 0.36). At n=200 the selected cell is explicitly noise-driven.
3. **Shuffled-label control flat.** **Never run.** No control appears in any pilot
   profile. This limb is unverified, so the bar cannot be satisfied as written.

And the pre-registered *falsification* clause — "all apparent wins collapse to BOS/sink
artifacts" — is partially triggered: on **Qwen2.5-7B the selected winner is
`final_bos_mass`**, a plain sink-mass cell that beats the best KV-tension cell by
0.0261. The lane's largest apparent routing win is on a model where sink mass wins outright.

**Call: NO-PROMOTE, not cleanly falsified.** The honest reading is that the
decomposition is *warm on GQA models and worthless on gemma*, that its two best models
are exactly the two whose winners are least stable, and that the comparator set was
under-specified in the pre-registration in a way that decides the verdict. That last
point is the reusable lesson: **"best existing comparator" must enumerate the cells.**

## If this lane is ever resumed

It needs a fresh pre-registration, not an amendment to the draft here. That
registration must, at minimum:

- enumerate the comparator cells explicitly (settle `bos_mass` in or out *before* running);
- include the shuffled-label control as a launch blocker, not a follow-up;
- raise n or add a stability floor, since `winner_unstable` fired on 4/5 at n=200;
- state up front whether gemma-3-4b's negative result is a scope boundary or a failure.

## Open build task — the implementation is not live here

The metric implementation exists only as a diff against sealed t0 files:

```
t0-patch/kv-tension-against-t0-7c2fcb7.patch
```

It adds `ATTENTION_METRICS_KV_TENSION` + three panels to `pri_calibrator.py`, the
JS-radius math to `scripts/diagnose_inter_head_disagreement.py`, and five contract
tests to `tests/test_attention_cells.py`. **It is not applied.** `t0-morphology-furnace`
is sealed, and its working tree must stay byte-identical to `t0-ace-sealed-2026-05-26`.

To make this runnable again, port it to an additive overlay in this repo, following the
pattern `confluence_calibrator.py` already uses for `READOUT_PANEL`: import the sealed
calibrator read-only, declare the KV-tension metric family locally, and compose the
panel here. The five contract tests port with it. Do **not** apply the patch to t0.
