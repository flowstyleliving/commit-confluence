# Grid-A rescore — cross-fitted E5/E6 (candidate grid-B endpoints)

GRID-A RESCORE under grid-B CANDIDATE endpoints — pre-freeze calibration artifact. Registered grid-A results are unchanged.

| task | model | E5 Δ_cf | E5 ok | perm p | boot CI (cond.) | E6 J_cf | E6 R_cf | E6 null q95 | E6 ok |
|---|---|---|---|---|---|---|---|---|---|
| anli_r1 | Qwen2.5-7B-Instruct | 0.2420 | Y | 0.0004997501249375312 | [0.14997500000000002, 0.34554999999999997] | 0.1800 | 0.2995 | -inf | Y |
| anli_r1 | Qwen2.5-32B-Instruct | 0.1460 | Y | 0.0009995002498750624 | [0.08400000000000003, 0.2425] | 0.2440 | 0.2995 | -inf | Y |
| anli_r1 | Qwen2.5-72B-Instruct | 0.2115 | Y | 0.0009995002498750624 | [0.10400000000000002, 0.3235499999999999] | 0.2340 | 0.2955 | -inf | Y |
| anli_r1 | Llama-3.3-70B-Instruct | 0.4160 | Y | 0.0004997501249375312 | [0.23145, 0.541] | 0.2060 | 0.2060 | -inf | Y |
| halueval_qa | Qwen2.5-7B-Instruct | 0.2020 | Y | 0.0004997501249375312 | [0.12250000000000001, 0.259525] | 0.3095 | 0.3320 | -inf | Y |
| halueval_qa | Qwen2.5-32B-Instruct | 0.2610 | Y | 0.0004997501249375312 | [0.14250000000000002, 0.33199999999999996] | 0.3425 | 0.2895 | -inf | Y |
| halueval_qa | Qwen2.5-72B-Instruct | 0.2640 | Y | 0.0004997501249375312 | [0.15645, 0.3340749999999999] | 0.2450 | 0.3085 | -inf | Y |
| halueval_qa | Llama-3.3-70B-Instruct | 0.1105 | Y | 0.0009995002498750624 | [0.02150000000000001, 0.20152499999999998] | UNDEF (empty E6 window (peak 26 < 41)) | — | -inf | N |

**anli_r1**: E5 4/4 (p=0.0004997501249375312), E6 4/4 (p=0.0004997501249375312), aggregate Δ CI [0.19211874999999998, 0.30901875] (joint undef 0.0)

**halueval_qa**: E5 4/4 (p=0.0004997501249375312), E6 3/4 (p=0.0004997501249375312), aggregate Δ CI [0.15409374999999997, 0.24691249999999998] (joint undef 0.0)

**Pooled**: E5 8/8 (p=0.0004997501249375312), E6 7/8 (p=0.0004997501249375312), 8-cell aggregate Δ CI [0.18930625, 0.26088749999999994] (joint undef 0.0)
