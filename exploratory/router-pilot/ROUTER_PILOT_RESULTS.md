# Router pilot results

Execution verification: not run by Codex.

Spending decision: **STOP**. At least one task misses a registered threshold.

Prospective descriptive single-model spending pilot; labels are dataset labels, not model errors. Final prompt state is the estimand; YES/NO compliance is descriptive. The best comparator is an upward-selected estimate. Intervals are conditional on the frozen folds and pointwise, without simultaneous depth coverage. n=200 cannot establish a reliable three-point advantage. ANLI's one pair per fold has no pair bootstrap variability. The permutation null assumes conditional exchangeability within pairs. No pooling, cross-model, placement-law, or conditional incremental claim. sink_draw is a new, unregistered descriptive diagnostic outside the comparator set and the spending rule.

## anli_r1

| Arm | Mean held-out AUROC / difference | Conditional 5/95 CI |
|---|---:|---|
| att_best | 0.6140 | [0.4082, 0.6715] |
| router_best | 0.5290 | [0.4129, 0.6675] |
| out_best | 0.5775 | [0.4100, 0.6475] |
| prompt_len | 0.4665 | [0.3987, 0.5870] |
| sink_best | 0.5465 | [0.3800, 0.6551] |
| out_p_max | 0.4505 | [0.3840, 0.5825] |
| out_entropy | 0.4420 | [0.3790, 0.5765] |
| out_yesno_logodds | 0.5775 | [0.4149, 0.6475] |
| best_comparator | 0.6140 | [0.4939, 0.6755] |
| router_minus_comparator | -0.0850 | [-0.2130, 0.1022] |

Best comparator: att_best.

Bootstrap excluded degenerate replicates: 0/1000.

Per-fold selections:

- att_best: fold 0: final_js block 4 sign +1; fold 1: final_js block 4 sign +1; fold 2: final_js block 4 sign +1; fold 3: final_js block 4 sign +1; fold 4: final_js block 4 sign +1
- router_best: fold 0: router_margin block 6 sign +1; fold 1: router_entropy_norm block 0 sign -1; fold 2: router_margin block 5 sign +1; fold 3: router_margin block 6 sign +1; fold 4: router_margin block 6 sign +1
- out_best: fold 0: out_yesno_logodds block None sign -1; fold 1: out_yesno_logodds block None sign -1; fold 2: out_yesno_logodds block None sign -1; fold 3: out_yesno_logodds block None sign -1; fold 4: out_yesno_logodds block None sign -1
- sink_best: fold 0: sink_draw block 14 sign +1; fold 1: sink_draw block 5 sign +1; fold 2: sink_draw block 11 sign +1; fold 3: sink_draw block 14 sign +1; fold 4: sink_draw block 14 sign +1

Descriptive diagnostics:

```json
{
  "yes_no_count": 194,
  "yes_no_fraction": 0.97,
  "yes_count": 100,
  "no_count": 94,
  "kv_groups_minus_js_max_abs": 1.0169642905566434e-13,
  "sink_draw_vs_bos_mass_spearman_by_block": [
    {
      "value": 0.9944213605340134,
      "reason": null
    },
    {
      "value": 0.9351083777094429,
      "reason": null
    },
    {
      "value": 0.9788944723618093,
      "reason": null
    },
    {
      "value": 0.9884242106052653,
      "reason": null
    },
    {
      "value": 0.9829640741018527,
      "reason": null
    },
    {
      "value": 0.9895057376434412,
      "reason": null
    },
    {
      "value": 0.9825335633390836,
      "reason": null
    },
    {
      "value": 0.9903757593939849,
      "reason": null
    },
    {
      "value": 0.9923543088577216,
      "reason": null
    },
    {
      "value": 0.990132753318833,
      "reason": null
    },
    {
      "value": 0.9608790219755495,
      "reason": null
    },
    {
      "value": 0.9935843396084904,
      "reason": null
    },
    {
      "value": 0.9724038100952526,
      "reason": null
    },
    {
      "value": 0.9728793219830496,
      "reason": null
    },
    {
      "value": 0.9795199879997003,
      "reason": null
    },
    {
      "value": 0.9894562364059103,
      "reason": null
    }
  ]
}
```

Full training curves, selection dispersion, bootstrap draws and null distributions are in the JSON.

## halueval_qa

| Arm | Mean held-out AUROC / difference | Conditional 5/95 CI |
|---|---:|---|
| att_best | 0.9040 | [0.8369, 0.9340] |
| router_best | 0.8315 | [0.7910, 0.8760] |
| out_best | 0.7730 | [0.7215, 0.8245] |
| prompt_len | 0.6022 | [0.6057, 0.6425] |
| sink_best | 0.8010 | [0.7550, 0.8480] |
| out_p_max | 0.6080 | [0.4605, 0.6695] |
| out_entropy | 0.5960 | [0.4509, 0.6575] |
| out_yesno_logodds | 0.7730 | [0.7265, 0.8245] |
| best_comparator | 0.9040 | [0.8375, 0.9340] |
| router_minus_comparator | -0.0725 | [-0.1150, -0.0005] |

Best comparator: att_best.

Bootstrap excluded degenerate replicates: 0/1000.

Per-fold selections:

- att_best: fold 0: final_js block 9 sign +1; fold 1: final_js block 9 sign +1; fold 2: final_js block 9 sign +1; fold 3: final_js block 9 sign +1; fold 4: final_js block 9 sign +1
- router_best: fold 0: router_entropy_norm block 3 sign +1; fold 1: router_entropy_norm block 3 sign +1; fold 2: router_entropy_norm block 3 sign +1; fold 3: router_entropy_norm block 3 sign +1; fold 4: router_entropy_norm block 3 sign +1
- out_best: fold 0: out_yesno_logodds block None sign -1; fold 1: out_yesno_logodds block None sign -1; fold 2: out_yesno_logodds block None sign -1; fold 3: out_yesno_logodds block None sign -1; fold 4: out_yesno_logodds block None sign -1
- sink_best: fold 0: sink_draw block 12 sign -1; fold 1: sink_draw block 12 sign -1; fold 2: sink_draw block 12 sign -1; fold 3: sink_draw block 12 sign -1; fold 4: sink_draw block 12 sign -1

Descriptive diagnostics:

```json
{
  "yes_no_count": 200,
  "yes_no_fraction": 1.0,
  "yes_count": 123,
  "no_count": 77,
  "kv_groups_minus_js_max_abs": 1.616484723854228e-13,
  "sink_draw_vs_bos_mass_spearman_by_block": [
    {
      "value": 0.9984924623115581,
      "reason": null
    },
    {
      "value": 0.9772384309607742,
      "reason": null
    },
    {
      "value": 0.9753183829595741,
      "reason": null
    },
    {
      "value": 0.9850896272406812,
      "reason": null
    },
    {
      "value": 0.9709517737943449,
      "reason": null
    },
    {
      "value": 0.9919822995574891,
      "reason": null
    },
    {
      "value": 0.9871191779794497,
      "reason": null
    },
    {
      "value": 0.9888847221180531,
      "reason": null
    },
    {
      "value": 0.9888247206180156,
      "reason": null
    },
    {
      "value": 0.9811565289132229,
      "reason": null
    },
    {
      "value": 0.960507012675317,
      "reason": null
    },
    {
      "value": 0.9933623340583516,
      "reason": null
    },
    {
      "value": 0.9890437260931525,
      "reason": null
    },
    {
      "value": 0.9685127128178205,
      "reason": null
    },
    {
      "value": 0.9858696467411687,
      "reason": null
    },
    {
      "value": 0.9540883522088053,
      "reason": null
    }
  ]
}
```

Full training curves, selection dispersion, bootstrap draws and null distributions are in the JSON.

