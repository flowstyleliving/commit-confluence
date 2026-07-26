# V-Norm Attention Audit

**Status:** `[RESOLVED - LAST-QUERY V-NORM NO-PROMOTE]`  
**Date:** 2026-06-09

This note records the V-norm handle that was considered before the
head/KV-tension lane.

Question: do ACE's existing value-vector norm cells add signal beyond
routing-only attention cells?

Audited cells:

- `v_norm_bos`
- `v_norm_max`
- `v_norm_lastq_weighted`

Data: the 18 sealed ACE profiles under:

- `experiments/t0-sealed/2026-05-26/profiles/anli/`
- `experiments/t0-sealed/2026-05-26/profiles/triviaqa/`

Result:

| Quantity | Value |
|---|---:|
| profiles audited | 18 |
| mean(best V-norm AUROC - best non-V AUROC) | -0.0436 |
| profiles with delta >= +0.03 | 0 |
| profiles with delta >= +0.02 | 0 |
| selected V-norm winners | 3/18 |

Selected V-norm winners were all tiny:

- Phi-4 ANLI: `mid_v_norm_lastq_weighted`, +0.0081
- Qwen2.5 ANLI: `final_v_norm_lastq_weighted`, +0.0107
- Qwen2.5 TriviaQA: `final_v_norm_lastq_weighted`, +0.0176

Verdict: do not spend fresh compute on `v_norm_lastq_weighted` as a standalone
follow-up. If value-payload is revisited, use a separate pre-reg for
column-sum V-weighting (`sink_top1_vw`, `sink_topk_sum_vw`), not an ACE
last-query rerun.
