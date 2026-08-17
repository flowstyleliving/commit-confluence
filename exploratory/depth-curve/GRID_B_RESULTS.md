# Grid-B registered results (single look)

GRID-B REGISTERED RESULTS — PRE_REGISTRATION_EXPANSION.md; single look; failures counted per §5.

**E5: WEAKEN** — 8/12 (families {'llama31': 4, 'mistral': 2, 'gemma': 2}, tasks {'anli_r1': 4, 'halueval_qa': 4}, p_grid 0.0004998; sensitivity {'leave_medium_out': {'e5_of_10': 6, 'e6_of_10': 1}, 'evaluable_only': {'e5': '8/9', 'e6': '2/9'}})

**E6: NOT TESTED — gate closed** — 2/12 (families {'llama31': 0, 'mistral': 1, 'gemma': 1}, tasks {'anli_r1': 1, 'halueval_qa': 1}, p_grid 0.0004998)

| task | model | E5 Δ_cf | E5 | E6 J_cf | E6 R_cf | E6 | note |
|---|---|---|---|---|---|---|---|
| anli_r1 | Llama-3.1-8B-Instruct | 0.1405 | Y | 0.0790 | 0.0545 | N |  |
| anli_r1 | Llama-3.1-70B-Instruct | 0.1405 | Y | 0.1300 | 0.1935 | N |  |
| anli_r1 | Mistral-Small-3.2-24B-Instruct-2506 | UNDEF | N | UNDEF | — | N | aborted: RuntimeError: missing smoke prompt manifest: /models/depth_grid_b/manif |
| anli_r1 | Mistral-Medium-3.5-128B | 0.2930 | Y | 0.1090 | 0.2615 | N |  |
| anli_r1 | gemma-3-12b-it | 0.2180 | Y | 0.3000 | 0.1860 | Y |  |
| anli_r1 | gemma-3-27b-it | UNDEF | N | UNDEF | — | N | aborted: RuntimeError: row 0 block 3 cell (0, 'Attention', 'final_js_no_bos'): b |
| halueval_qa | Llama-3.1-8B-Instruct | 0.1060 | Y | UNDEF | — | N |  |
| halueval_qa | Llama-3.1-70B-Instruct | 0.2010 | Y | UNDEF | — | N |  |
| halueval_qa | Mistral-Small-3.2-24B-Instruct-2506 | 0.0045 | N | UNDEF | — | N |  |
| halueval_qa | Mistral-Medium-3.5-128B | 0.1510 | Y | 0.2185 | 0.2180 | Y |  |
| halueval_qa | gemma-3-12b-it | 0.3055 | Y | UNDEF | — | N |  |
| halueval_qa | gemma-3-27b-it | UNDEF | N | UNDEF | — | N | aborted: RuntimeError: row 0 block 3 cell (0, 'Attention', 'final_js_no_bos'): b |

Aggregates: {"anli_r1": {"designated_cells": ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Mistral-Medium-3.5-128B", "gemma-3-12b-it"], "excluded_failed_cells": ["Mistral-Small-3.2-24B-Instruct-2506", "gemma-3-27b-it"], "ci_5_95_alldef": [0.131225, 0.25915], "joint_undefined_frac": 0.003}, "halueval_qa": {"designated_cells": ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Mistral-Small-3.2-24B-Instruct-2506", "Mistral-Medium-3.5-128B", "gemma-3-12b-it"], "excluded_failed_cells": ["gemma-3-27b-it"], "ci_5_95_alldef": [0.08739000000000001, 0.18730499999999997], "joint_undefined_frac": 0.0}, "pooled": {"designated_cells": ["anli_r1/Llama-3.1-8B-Instruct", "anli_r1/Llama-3.1-70B-Instruct", "anli_r1/Mistral-Medium-3.5-128B", "anli_r1/gemma-3-12b-it", "halueval_qa/Llama-3.1-8B-Instruct", "halueval_qa/Llama-3.1-70B-Instruct", "halueval_qa/Mistral-Small-3.2-24B-Instruct-2506", "halueval_qa/Mistr

Predictions: {"P6_e5_confirms": false, "P7_e6_ge_9_if_tested": null, "P8_llama70b_anli_band_majority_ge_10": true, "P9_llama8b_anli_band_majority_lt_5": true, "P10_no_obvious_placement_rule": "see E1'' panel (descriptive)"}

E7: {"Llama-3.1-8B-Instruct": {"defined": true, "peak_anli": 18.0, "peak_halueval": 1.0, "distance_blocks": 17.0, "distance_frac": 0.53125, "distance_boot_ci_5_95_blocks": [2.0, 17.0], "distance_boot_ci_5_95_frac": [0.0625, 0.53125], "boot_pair_undefined_frac": 0.003}, "Llama-3.1-70B-Instruct": {"defined": true, "peak_anli": 45.0, "peak_halueval": 38.0, "distance_blocks": 7.0, "distance_frac": 0.0875, "distance_boot_ci_5_95_blocks": [0.0, 39.0], "distance_boot_ci_5_95_frac": [0.0, 0.4875], "boot_pair_undefined_frac": 0.0}, "Mistral-Small-3.2-24B-Instruct-2506": {"defined": false, "why": "one or both cells lack a defined cross-fit", "distance_blocks": null, "distance_frac": null, "distance_boot_ci_5_95_blocks": null, "distance_boot_ci_5_95_frac": null, "boot_pair_undefined_frac": 1.0}, "Mistral-Medium-3.5-128B": {"defined": true, "peak_anli": 83.0, "peak_halueval": 85.0, "distance_blocks": 2.0, "distance_frac": 0.022727272727272728, "distance_boot_ci_5_95_blocks": [0.0, 48.0], "distance_boot_ci_5_95_frac": [0.0, 0.5454545454545454], "boot_pair_undefined_frac": 0.0}, "gemma-3-12b-it": {"defined": true, "peak_anli": 39.0, "peak_halueval": 22.0, "distance_blocks": 17.0, "distance_frac": 0.
