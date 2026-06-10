# S2 - ACE reproduction validation (Qwen2.5-7B-Instruct-4bit / anli)

Full n=200, max_new_tokens=1, seed 20260512 (sealed provenance).

- max abs AUROC diff over 21 cells: **0.00000** (tol 0.01)
- winner `final_v_norm_lastq_weighted` reproduces: **True**
- data hash match: **True**
- **S2 PASS**

| cell | mine | sealed | diff | winner |
|---|---:|---:|---:|:--:|
| final_bos_mass | 0.7796 | 0.7796000000000001 | 0.0 |  |
| final_js | 0.6033 | 0.6033 | 0.0 |  |
| final_js_kv_groups | 0.5901 | 0.5901000000000001 | 0.0 |  |
| final_js_no_bos | 0.7049 | 0.7049000000000001 | 0.0 |  |
| final_v_norm_bos | 0.5 | 0.5 | 0.0 |  |
| final_v_norm_lastq_weighted | 0.7903 | 0.7903 | 0.0 | WIN |
| final_v_norm_max | 0.5134 | 0.5134 | 0.0 |  |
| last_minus_1_bos_mass | 0.5214 | 0.5214 | 0.0 |  |
| last_minus_1_js | 0.6281 | 0.6281 | 0.0 |  |
| last_minus_1_js_kv_groups | 0.6956 | 0.6956 | 0.0 |  |
| last_minus_1_js_no_bos | 0.6379 | 0.6379 | 0.0 |  |
| last_minus_1_v_norm_bos | 0.5 | 0.5 | 0.0 |  |
| last_minus_1_v_norm_lastq_weighted | 0.5238 | 0.5238 | 0.0 |  |
| last_minus_1_v_norm_max | 0.5026 | 0.5025999999999999 | 0.0 |  |
| mid_bos_mass | 0.5118 | 0.5118 | 0.0 |  |
| mid_js | 0.5473 | 0.5473 | 0.0 |  |
| mid_js_kv_groups | 0.5698 | 0.5698 | 0.0 |  |
| mid_js_no_bos | 0.5235 | 0.5235000000000001 | 0.0 |  |
| mid_v_norm_bos | 0.5 | 0.5 | 0.0 |  |
| mid_v_norm_lastq_weighted | 0.508 | 0.508 | 0.0 |  |
| mid_v_norm_max | 0.5551 | 0.5550999999999999 | 0.0 |  |
