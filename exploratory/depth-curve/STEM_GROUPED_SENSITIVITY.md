# Stem-grouped sensitivity rescore (UNREGISTERED)

UNREGISTERED stem-grouped sensitivity rescore. Cannot move any registered verdict. Grids A and B are never pooled.

## Groups

- anli_r1: 5 pairs, 95+95 singletons; registered fold map splits 4 pairs
- halueval_qa: 100 pairs, 0+0 singletons; registered fold map splits 74 pairs

## Part B — fold-draw distributions (row-stratified vs group-stratified)

| grid | task | model | registered Δ | row mean (sd) | group mean (sd) | group−row (MC se) | success row / group | reg. pctile in row arm |
|---|---|---|---|---|---|---|---|---|
| A | anli_r1 | Llama-3.3-70B-Instruct | 0.4160 | 0.3238 (0.0233) | 0.3261 (0.0237) | 0.0023 (0.0023) | 1.00 / 1.00 | 97.5000 |
| A | anli_r1 | Qwen2.5-32B-Instruct | 0.1460 | 0.1496 (0.0130) | 0.1510 (0.0127) | 0.0014 (0.0013) | 1.00 / 1.00 | 35.0000 |
| A | anli_r1 | Qwen2.5-72B-Instruct | 0.2115 | 0.2123 (0.0174) | 0.2118 (0.0163) | -0.0005 (0.0017) | 1.00 / 1.00 | 43.0000 |
| A | anli_r1 | Qwen2.5-7B-Instruct | 0.2420 | 0.2231 (0.0101) | 0.2221 (0.0106) | -0.0010 (0.0010) | 1.00 / 1.00 | 97.5000 |
| A | halueval_qa | Llama-3.3-70B-Instruct | 0.1105 | 0.1133 (0.0233) | 0.1228 (0.0214) | 0.0094 (0.0022) | 0.98 / 1.00 | 43.5000 |
| A | halueval_qa | Qwen2.5-32B-Instruct | 0.2610 | 0.2585 (0.0085) | 0.2585 (0.0083) | 0.0000 (0.0008) | 1.00 / 1.00 | 63.5000 |
| A | halueval_qa | Qwen2.5-72B-Instruct | 0.2640 | 0.2567 (0.0069) | 0.2588 (0.0068) | 0.0021 (0.0007) | 1.00 / 1.00 | 86.0000 |
| A | halueval_qa | Qwen2.5-7B-Instruct | 0.2020 | 0.1898 (0.0076) | 0.1897 (0.0076) | -0.0000 (0.0008) | 1.00 / 1.00 | 95.5000 |
| B | anli_r1 | Llama-3.1-70B-Instruct | 0.1405 | 0.1355 (0.0072) | 0.1351 (0.0077) | -0.0004 (0.0007) | 1.00 / 1.00 | 78.0000 |
| B | anli_r1 | Llama-3.1-8B-Instruct | 0.1405 | 0.1371 (0.0193) | 0.1387 (0.0188) | 0.0016 (0.0019) | 1.00 / 1.00 | 50.5000 |
| B | anli_r1 | Mistral-Medium-3.5-128B | 0.2930 | 0.2874 (0.0121) | 0.2888 (0.0112) | 0.0014 (0.0012) | 1.00 / 1.00 | 70.0000 |
| B | anli_r1 | gemma-3-12b-it | 0.2180 | 0.2079 (0.0384) | 0.2040 (0.0365) | -0.0040 (0.0037) | 1.00 / 1.00 | 71.0000 |
| B | halueval_qa | Llama-3.1-70B-Instruct | 0.2010 | 0.1952 (0.0149) | 0.1979 (0.0140) | 0.0027 (0.0014) | 1.00 / 1.00 | 60.0000 |
| B | halueval_qa | Llama-3.1-8B-Instruct | 0.1060 | 0.0986 (0.0139) | 0.0988 (0.0089) | 0.0002 (0.0012) | 0.97 / 0.99 | 75.5000 |
| B | halueval_qa | Mistral-Medium-3.5-128B | 0.1510 | 0.1495 (0.0165) | 0.1525 (0.0082) | 0.0030 (0.0013) | 1.00 / 1.00 | 41.5000 |
| B | halueval_qa | Mistral-Small-3.2-24B-Instruct-2506 | 0.0045 | 0.0072 (0.0124) | 0.0080 (0.0109) | 0.0008 (0.0012) | 0.00 / 0.00 | 38.5000 |
| B | halueval_qa | gemma-3-12b-it | 0.3055 | 0.2984 (0.0158) | 0.2974 (0.0142) | -0.0011 (0.0015) | 1.00 / 1.00 | 56.0000 |

Grid A: {"n_cells": 8, "median_group_minus_row": 0.0007075000000000137, "mean_group_minus_row": 0.0017315625000000074, "n_negative": 3}

Grid B: {"n_cells": 9, "median_group_minus_row": 0.0007699999999999964, "mean_group_minus_row": 0.00046861111111109907, "n_negative": 3}

## Part C — fully grouped inference (designated grouped fold map)

| grid | task | model | Δ reg → grouped | ok reg → grouped | perm p reg → grouped | boot CI reg → grouped | E6 J / R / ok |
|---|---|---|---|---|---|---|---|
| A | anli_r1 | Llama-3.3-70B-Instruct | 0.4160 → 0.3225 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.23145, 0.541] → [0.22597499999999998, 0.47907500000000003] | 0.2700 / 0.2000 / True |
| A | anli_r1 | Qwen2.5-32B-Instruct | 0.1460 → 0.1565 | True → True | 0.0009995002498750624 → 0.0004997501249375312 | [0.08400000000000003, 0.2425] → [0.07550000000000001, 0.23899999999999996] | 0.2455 / 0.3145 / True |
| A | anli_r1 | Qwen2.5-72B-Instruct | 0.2115 → 0.2155 | True → True | 0.0009995002498750624 → 0.0009995002498750624 | [0.10400000000000002, 0.3235499999999999] → [0.11900000000000002, 0.328525] | 0.3240 / 0.2955 / True |
| A | anli_r1 | Qwen2.5-7B-Instruct | 0.2420 → 0.2085 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.14997500000000002, 0.34554999999999997] → [0.134975, 0.3200499999999999] | 0.1905 / 0.2675 / True |
| A | halueval_qa | Llama-3.3-70B-Instruct | 0.1105 → 0.1215 | True → True | 0.0009995002498750624 → 0.0004997501249375312 | [0.02150000000000001, 0.20152499999999998] → [0.03550000000000002, 0.197] | — / — / False |
| A | halueval_qa | Qwen2.5-32B-Instruct | 0.2610 → 0.2650 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.14250000000000002, 0.33199999999999996] → [0.17095000000000007, 0.321025] | 0.3520 / 0.3090 / True |
| A | halueval_qa | Qwen2.5-72B-Instruct | 0.2640 → 0.2610 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.15645, 0.3340749999999999] → [0.20145000000000002, 0.325525] | 0.3085 / 0.3070 / True |
| A | halueval_qa | Qwen2.5-7B-Instruct | 0.2020 → 0.1895 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.12250000000000001, 0.259525] → [0.11694999999999997, 0.244525] | 0.2350 / 0.3150 / True |
| B | anli_r1 | Llama-3.1-70B-Instruct | 0.1405 → 0.1320 | True → True | 0.0024987506246876563 → 0.001999000499750125 | [0.04997499999999998, 0.18804999999999997] → [0.058974999999999965, 0.17957499999999996] | 0.1190 / 0.1820 / False |
| B | anli_r1 | Llama-3.1-8B-Instruct | 0.1405 → 0.1765 | True → True | 0.0009995002498750624 → 0.0004997501249375312 | [0.018800000000000004, 0.28509999999999996] → [0.023925000000000002, 0.292] | 0.0695 / 0.0780 / False |
| B | anli_r1 | Mistral-Medium-3.5-128B | 0.2930 → 0.2870 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.14800000000000005, 0.46509999999999996] → [0.16250000000000003, 0.430525] | 0.1735 / 0.2420 / True |
| B | anli_r1 | gemma-3-12b-it | 0.2180 → 0.1635 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.092975, 0.36252500000000004] → [0.09747499999999996, 0.382525] | — / — / False |
| B | halueval_qa | Llama-3.1-70B-Instruct | 0.2010 → 0.2160 | True → True | 0.0014992503748125937 → 0.0004997501249375312 | [0.09650000000000003, 0.26352500000000006] → [0.15047500000000003, 0.2655] | — / — / False |
| B | halueval_qa | Llama-3.1-8B-Instruct | 0.1060 → 0.0575 | True → True | 0.0009995002498750624 → 0.0009995002498750624 | [-0.030624999999999965, 0.16757499999999997] → [-0.016525, 0.142] | — / — / False |
| B | halueval_qa | Mistral-Medium-3.5-128B | 0.1510 → 0.1425 | True → True | 0.001999000499750125 → 0.0004997501249375312 | [-0.010525, 0.21955] → [0.05250000000000001, 0.20149999999999996] | 0.2430 / 0.2125 / True |
| B | halueval_qa | Mistral-Small-3.2-24B-Instruct-2506 | 0.0045 → 0.0120 | False → False | 0.0004997501249375312 → 0.0004997501249375312 | [-0.06257500000000002, 0.07202499999999996] → [-0.045549999999999966, 0.07302499999999998] | — / — / False |
| B | halueval_qa | gemma-3-12b-it | 0.3055 → 0.3045 | True → True | 0.0004997501249375312 → 0.0004997501249375312 | [0.19595, 0.452525] → [0.22000000000000003, 0.46799999999999997] | — / — / False |

Grid A: {"per_task": {"anli_r1": {"e5_success": 4, "evaluable": 4, "p_grid_e5": 0.0004997501249375312, "e6_success": 4, "p_grid_e6": 0.0004997501249375312, "aggregate_delta_ci_5_95_alldef": [0.18236875000000002, 0.28788125], "aggregate_joint_undefined_frac": 0.0}, "halueval_qa": {"e5_success": 4, "evaluable": 4, "p_grid_e5": 0.0004997501249375312, "e6_success": 3, "p_grid_e6": 0.0004997501249375312, "aggregate_delta_ci_5_95_alldef": [0.16511875, 0.2368875], "aggregate_joint_undefined_frac": 0.0}}, "e5_success_evaluable": "8/8", "e6_success_evaluable": "7/8", "p_grid_e5_pooled": 0.0004997501249375312, "p_grid_e6_pooled": 0.0004997501249375312, "aggregate_delta_ci_5_95_alldef": [0.188309375, 0.25115], "aggregate_joint_undefined_frac": 0.0}

Grid B: {"per_task": {"anli_r1": {"e5_success": 4, "evaluable": 4, "p_grid_e5": 0.0004997501249375312, "e6_success": 1, "p_grid_e6": 0.0009995002498750624, "aggregate_delta_ci_5_95_alldef": [0.1363375, 0.25876875], "aggregate_joint_undefined_frac": 0.002}, "halueval_qa": {"e5_success": 4, "evaluable": 5, "p_grid_e5": 0.0004997501249375312, "e6_success": 1, "p_grid_e6": 0.0009995002498750624, "aggregate_delta_ci_5_95_alldef": [0.11099000000000002, 0.18580500000000003], "aggregate_joint_undefined_frac": 0.0}}, "e5_success_evaluable": "8/9", "e6_success_evaluable": "2/9", "p_grid_e5_pooled": 0.0004997501249375312, "p_grid_e6_pooled": 0.0004997501249375312, "aggregate_delta_ci_5_95_alldef": [0.13437222222222223, 0.20354166666666668], "aggregate_joint_undefined_frac": 0.002, "counterfactual_s5_rule_NON_VERDICT": {"count_of_12": 8, "per_family": {"llama31": 4, "mistral": 2, "gemma": 2}, "guards": {"count_ge_10": false, "families_ge_3of4": false, "p_grid_lt_0.05": true}, "decision": "WEAKEN"}}
