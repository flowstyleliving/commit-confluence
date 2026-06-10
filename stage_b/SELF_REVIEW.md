# Interim self-adversarial review (2026-06-10)

**Not a substitute for the Codex review** (Codex quota-blocked until 2026-06-11 13:55; the
adversarial prompt is queued to agent `aa5a98bd894eb899b`). This is me attacking my own work so
the seal is not launched on an un-reviewed build. Findings I would expect Codex to also raise.

| # | Finding | Severity | Status |
|---|---|---|---|
| S1 | **ACE panel divergence (12 vs 21).** `collect_ace_matrix` uses `ATTENTION_PANEL_T0` = 12 routing-only cells. The SEALED ACE instrument used the 21-cell `ATTENTION_PANEL_T0_WITH_V_NORMS`; several sealed per-model winners ARE v-norm cells (Qwen2.5 `final_v_norm_lastq_weighted`, Phi-4 `mid_v_norm_lastq_weighted`). So my dispatcher's ACE arm is NOT the sealed instrument and cannot even select the sealed winners. | **BLOCKER (must fix before seal)** | switch to `ATTENTION_PANEL_T0_WITH_V_NORMS` |
| S2 | **ACE-matches-sealed never validated.** Only a 6-sample plumbing smoke was run. The real wiring gate - run `collect_ace_matrix` on the full n=200 for one model and confirm a sealed winner cell's AUROC reproduces the sealed profile - is NOT done. Until then "ACE wired correctly" is asserted, not shown. | **MAJOR (must fix before seal)** | run full-200 reproduction on >=1 model |
| S3 | **"Selector verified exact" overstates.** The cross-check reproduces `_score_candidate`'s full-sample MARGINAL to 4dp - that proves I read the right columns, not that the nested-OOB OOB numbers are independently correct. The OOB CIs come from the imported sealed function (trusted because sealed), but are not independently reproduced here. | MAJOR (wording) | soften BUILD_RESULT/README claim to "inputs verified exact; selection logic inherited from sealed code" |
| S4 | **Stage A sign-locking can inflate near 0.5.** `coverage_matrix.py` locks `surprise`'s sign from the full sample (no fold_signs), then bootstraps - the CI is centered on a sign chosen with all the data. `surprise` has a canonical orientation (higher->riskier); it should be hard-locked `+1`, not fit. Likely doesn't change Stage A conclusions (orphans stay dead, the gemma gap stays) but is a real optimism leak. | MINOR | hard-lock surprise sign; re-run; confirm conclusions stable |
| S5 | **Multiplicity at n=200 over ~22-27 cells.** winner_stability will be low; the "deployable" verdict must rest on OOB CI_lo, never winner identity. Not yet empirically stress-tested on a wide unified panel. | MAJOR (verify) | empirically watch winner_stability + OOB CI_lo on the real merged panel during the run; pre-registered as expected/non-fatal but should be shown |
| S6 | **Dual-endpoint could launder a weak result.** A PRIMARY "product" claim satisfiable by surprise-fallback can hide a weak geometric arm. Mitigated by the SECONDARY geometric-only endpoint being separately registered + reported, but the framing must lead with the geometric result, not the product number. | MINOR | report geometric-only FIRST in results |
| S7 | **measurement-orphan framing is convenient.** "surprise also dead -> out of scope" is defensible (degenerate commit measurement) but excludes the hard models; hold with humility, do not claim universality. | NIT | already documented; keep humility |
| S8 | gpt-oss exclusion legit (RPV run itself skipped them, no rows). | OK | no action |

## The single most likely way this is fooling itself
S1 + S2 together: the ACE arm in the unified panel is a **routing-only 12-cell approximation that was
never validated to reproduce the sealed ACE instrument**. If ACE is silently weaker than sealed, the
dispatcher leans harder on null_ratio/RPV and the "convergence/coverage" story is partly an artifact
of a degraded ACE arm. Both must be fixed before the fresh seal: use the 21-cell with-v-norms panel
AND prove a sealed winner reproduces on full n=200.

## Interim verdict: GO-WITH-FIXES (hold the seal)
Must-fix before launching the fresh-seed sealed run: **S1** (21-cell panel) and **S2** (full-200
ACE-reproduction on >=1 model). Should-fix: S3 (wording), S5 (show multiplicity behaviour). Then
let Codex corroborate (or extend) before the irreversible seal.

## Resolution status (2026-06-10, post-S2)

- **S1 - RESOLVED.** `collect_ace_matrix` now uses the sealed 21-cell `ATTENTION_PANEL_T0_WITH_V_NORMS`
  + the v-norm capture path. Unified panel = 27 cells.
- **S2 - RESOLVED (byte-exact).** `stage_b/validate_ace.py` on Qwen2.5-7B/anli full n=200
  (max_new_tokens=1, sealed provenance): all 21 ACE cell AUROCs reproduce the sealed profile to 4 dp,
  **max abs diff 0.0000**; winner `final_v_norm_lastq_weighted` reproduces (0.7903); data hash matches.
  The ACE arm is the sealed instrument, not an approximation. See `VALIDATE_ACE.md`.
- **S3 - RESOLVED (wording).** Precise statement: the two INPUT marginals (ACE cells + readout cells)
  are now verified exact against their sealed sources; the nested-OOB SELECTION logic is *inherited*
  from the sealed `_nested_bootstrap_oob_auroc` (imported, not independently re-derived), so its OOB
  CIs are trusted-because-sealed, not separately reproduced here.
- **S4 - CONFIRMED NON-IMPACTFUL.** Stage A surprise AUROCs are all >= 0.50 (min 0.514), so the
  full-sample-fit sign was +1 everywhere = the canonical orientation; hard-locking would change nothing.
  Loose in code, null in effect. (Will hard-lock in the Stage B confidence cells regardless.)
- **S5 - DEFERRED to the run.** winner_stability + OOB CI_lo behaviour on the real 27-cell merged panel
  to be reported per cell during the fresh seal (pre-registered as expected/non-fatal).
- **S6 - reporting convention.** Lead results with the geometric-only endpoint, not the product number.

**Must-fixes (S1, S2) are both resolved.** Remaining are should-fix/reporting. Seal still HELD for the
independent Codex pass (quota resets 2026-06-11 13:55).
