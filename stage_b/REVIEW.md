# Stage B pre-reg - adversarial review (2026-06-10)

Critical pass over `PRE_REGISTRATION_DRAFT.md` before building. Findings folded into
`PRE_REGISTRATION.md` (v2). Each is tagged where it lands.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| R1 | "one forward pass" was overclaimed - the families live in two codebases sharing the t=0 locus. | medium | Relax to "common t=0 commit locus, jointly selected; single fused pass is a production optimization." The coverage claim depends on joint *selection*, not runtime. |
| R2 | Build must not edit `pri_calibrator.py` - it is the frozen sealed ACE/T0 core (CLAUDE.md: live work only in `exploratory/` or another repo). | **high** | `confluence_calibrator.py` lives in THIS repo and **imports** `_nested_bootstrap_oob_auroc` / `_score_candidate` / `_cell_label` read-only. Zero edits to t0. |
| R3 | `null_ratio` is computable in BOTH the calibrator (`T0_RESIDUAL_PANEL`) and the shadow runner; mixing the two would double-count / mismatch the metric. | **high** | Single source per family: `{null_ratio, RPV, surprise, p_max}` from the readout pass only; ACE from the attention pass only. Hard label-alignment assert on merge. |
| R4 | Merged panel is ~27 cells (21 ACE + 5 residual + 3 readout + 2 conf -> drop non-finite), more selection multiplicity than ACE's 21. | medium | Do NOT prune to the sealed ACE winner (injects oracle knowledge). Nested-OOB prices multiplicity in via OOB scoring; `winner_unstable` is expected + non-fatal; deployability judged ONLY on OOB CI_lo. Pre-registered. |
| R5 | The "gap" framing was confused: if `surprise` is a panel cell, `gemma-3-4b/anli` is covered by surprise and there is no gap. | **high (clarifying)** | Register TWO claims: PRIMARY full-panel incl. confidence (product) >= 19/20; SECONDARY geometric-only (science) >= 17/20 with `gemma-3-4b/anli` documented. Falling back to confidence when geometry is blind is a feature. |
| R6 | The RPV gauntlet previously caught a degraded-base inflation (+0.13 -> +0.044 fair-base). Could a similar selection-bias leak inflate Stage B? | medium | Stage B's endpoint is COVERAGE (is a deployable cell selected) not INCREMENT-over-base, so the degraded-base trap is off-path; still pre-register the shuffled-label control (selected-cell OOB CI must include 0.5) + the >0.05 fresh-vs-Stage-A drop falsifier to catch any leak. |
| R7 | Sign-fitting risk (test-fold sign selection inflates AUROC near 0.5). | low (already handled) | Inherited for free by reusing the sealed `_nested_bootstrap_oob_auroc`: sign is locked on in-bag, evaluated on OOB. Do NOT reimplement - import. |

## Net

The build is a *composition*, not new statistics: assemble a merged per-sample score matrix from two
co-located collection passes and hand it to the sealed nested-OOB selector. The one genuinely new
correctness surface is the **merge/alignment** (R3), which gets a hard assert. The one genuinely new
*claim* surface is the **dual product/geometric endpoint** (R5). Everything statistical is inherited.
