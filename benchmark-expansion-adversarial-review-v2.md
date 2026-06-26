# Second Adversarial Review - Benchmark Expansion Proposal v2

Proposal reviewed: `wiki/paper/benchmark-expansion-proposal-v2.md`

Original review: `wiki/paper/benchmark-expansion-adversarial-review.md`

Commit-locus reference: `wiki/references/commit-locus.md`

Code spot-checked:

- `confluence_calibrator.py`
- `stage_b/run_seal.py`
- `stage_b/analyze_universality.py`
- `stage_b/check_fresh_data.py`
- `stage_b/generate_fresh_data.py`

## Bottom Line

**VERDICT: FIX** - v2 is materially better than v1 and resolves most of the original overclaiming in prose. It is **not ready to execute**. The remaining blockers are not cosmetic: the primary LOMO endpoint is still a post-hoc max over cells, several "open decisions" are actually pre-registration inputs, P3 is called primary without a real falsification/inclusion rule, and the group-correlation fix keeps the inflated row-bootstrap as the registered gate.

Do not run 50+ cells from this document as-is. Freeze a corrected `PRE_REGISTRATION_BENCH.md` first.

## Must-Fix Audit

- **TRAP** - Must-fix #1, byte-comparability, is only partially resolved. The main prose is much more honest: v2 splits T0 extraction from the commit-confluence harness and says new cells are a separately registered post-seal extension (v2:32-60). But the audit table overstates the fix: it says launcher, gate, builders, transfer, and pre-reg "are recorded in `module_hashes`" (v2:17), while v2 itself later admits `run_seal.py`, `check_fresh_data.py`, `generate_fresh_data.py`, and `analyze_universality.py` are **not** in `module_hashes()` (v2:42-49). The phrase "separately-hashed extension" is therefore still too broad (v2:51-58). It is separately hashed only for the hot compute modules plus `confluence_calibrator.py`/`fusion_signs.json`, not for the whole harness that defines the run.

- **SAFE** - Must-fix #2, commit locus, is resolved. v2 now states ACE is t=0/prefix-last and PRI/RPV/Confidence are gen_step=1 (v2:62-70), and it correctly warns that P3 predicts eventual correctness from the gen_step=1 state rather than detecting hallucination at token 1 (v2:71-73). This matches `commit-locus.md`.

- **SAFE** - Must-fix #3, "JSONL is enough," is resolved at the proposal level. v2 replaces that claim with a seven-clause prompt/extraction contract covering one-token cue, prompt wrapping, length limits, no-drop policy, first-token sanity, smoke gates, and common intersection sets (v2:75-87). The substance is right. Execution still requires implementing those gates before any strict cell, but the proposal no longer hides the contract.

- **TRAP** - Must-fix #4, preregistration bars, is not fully resolved. v2 adds bars and a falsification frame (v2:200-227), but the paper-level endpoint is not as fixed as it sounds. v2 describes `fixed_cell_max_survival` as "one fixed cell" (v2:211), yet the current code computes a full landscape and then reports the cell with the maximum survival after seeing all holdouts. That is better than per-holdout cherry-picking, but still a max over the 29-cell panel. v2 also flags max-over-cells multiplicity as merely descriptive (v2:226-227) while making a max-over-cells statistic the primary title-level endpoint (v2:211, v2:216-218). If the universal-floor claim rests on `fusion_rank_mean_geom`, freeze that named cell. If it rests on "best fixed cell found by LOMO," register it as a selection procedure with its multiplicity cost.

- **TRAP** - Must-fix #5, P3, is scientifically reframed but not execution-ready. v2 correctly renames the target as predicting eventual short-answer exact-match correctness (v2:165-173). But it leaves key gates underspecified: the first-token audit is only on a "representative subset of models" (v2:169), the normalizer and label-noise audit have no pre-declared accept/reject thresholds (v2:170-171), and "drop a model if its base rate makes AUROC unstable" has no numerical rule (v2:173). Worse, §6 says P3 is "primary too" (v2:209) while §6.4 gives no P3 falsification bar, no inclusion threshold, and no rule for how P3 affects the headline if it fails (v2:220-221).

- **TRAP** - Must-fix #6, group correlation, is only half fixed. v2 adds stem caps, effective-n reporting, and a cluster bootstrap (v2:182-196). But it explicitly keeps the row-bootstrap CI as the registered per-cell gate "for comparability with the seal" (v2:196). That preserves the exact inflation the fix was supposed to address. For newly grouped tasks, a cell that passes row-CI and fails stem-CI should not be called confirmatory deployable. Either make cluster-CI the confirmatory gate for grouped tasks or downgrade grouped-task deployability claims.

- **SAFE** - Must-fix #7, FEVER NEI/evidence joining, is resolved as a scoped scientific decision. v2 admits the join is non-trivial (v2:157-159), registers primary FEVER as SUPPORTS vs REFUTES on gold evidence (v2:160-162), and explicitly says this does not cover NEI/insufficient evidence (v2:162-163). That is narrower, but honest. The FEVER source/license choice remains a pre-execution blocker, not a scientific phrasing blocker (v2:100, v2:261).

- **UNKNOWN** - Must-fix #8, licenses and exact artifacts, is resolved as a hard gate but not as an executable plan. v2 correctly marks HaluEval as blocked on a license mismatch (v2:97, v2:106-111) and flags the FEVER gold-evidence GPL issue (v2:100, v2:108). However, it marks raw `fever/fever` as simply OK under CC BY-SA 3.0 (v2:99), while the actual artifact choice is still open (v2:261). No confirmatory task set can be frozen until HaluEval and FEVER source decisions are closed.

- **SAFE** - Must-fix #9, SimpleQA/RAGTruth comparison, is resolved. v2 no longer pretends HaluEval/TruthfulQA/FEVER are the obvious complete set; it compares SimpleQA, RAGTruth, and FActScore and gives protocol-fit reasons for inclusion/defer/exclusion (v2:115-141). The SimpleQA base-rate pilot must stay exploratory unless its inclusion rule is frozen before any feature matrix (v2:134).

- **SAFE** - Must-fix #10, paper-language isolation from the seal, is resolved. v2 explicitly preserves the sealed 18/20 and two-task headline (v2:231-235), hedges unsupported-answer language until P3 earns it (v2:236), fixes the glossary locus error (v2:237), and narrows confidence/universality claims if new tasks contradict them (v2:238-239).

## New Traps Introduced By v2

- **TRAP** - The §0 audit table now creates its own false claim. It says the changed launcher/gate/builders/transfer/pre-reg are "recorded in `module_hashes`" (v2:17). They are not, by v2's own later description (v2:49). This must be fixed because §0 is the reviewer-facing proof table; a contradiction there will be caught.

- **TRAP** - The universal-floor endpoint is mislabeled as fixed. `fixed_cell_max_survival` is a post-run maximum over cells, not a pre-named fixed cell (v2:211, v2:216). If the intended claim is "fusion_rank_mean_geom is the universal floor," the primary endpoint must be the frozen fusion cell. If the intended claim is "some fixed cell can be selected by a registered LOMO procedure," then the endpoint must say so and pay the 29-cell selection/multiplicity cost.

- **TRAP** - The confirmatory denominator can silently mutate. The primary set is "whose licenses clear §2" (v2:209), while HaluEval is blocked (v2:97), FEVER source is open (v2:261), and the effort envelope itself is open (v2:258). That means M_new and M_pass are not actually frozen. Pre-registration needs an ordered fallback rule: what happens if HaluEval-Summ is excluded, if FEVER gold is not usable, or if the minimal trio is chosen?

- **TRAP** - P3 is called primary but has no headline contract. v2 says P3 is primary too (v2:209), then isolates it to a separate track with endpoints but no pass/fail consequence (v2:220-221). That invites selective narration: if P3 works, it becomes the "real hallucination" depth result; if it fails, it becomes a caveated side track. Freeze that consequence now.

- **TRAP** - The cluster bootstrap is descriptive even though it diagnoses the known dependence violation (v2:196). For 50+ grouped cells, this will not hold water: a confirmatory expansion cannot knowingly certify row-independent CIs while relegating the dependence-corrected CI to a caveat.

- **UNKNOWN** - The "limit-8 smoke gate" may be too weak for prompt-format validity. v2 requires ACE/readout 8/8 usable and meaningful first token (v2:84), but a prompt family can pass 8 rows and still fail on rare FEVER/HaluEval length, refusal, or formatting cases. The strict cell's no-drop policy will catch hard failures (v2:82), but not semantic failures where the first token is parseable but not the intended YES/NO judgment. The smoke needs stratified examples across length/subset/label/stem type, not just first 8.

## Open Decisions That Are Actually Prerequisites

- **TRAP** - Effort envelope is a prerequisite, not a preference. Full primary set versus minimal trio changes the denominator, the M_pass rule, and the paper claim (v2:258). It must be frozen before builders, gates, or predictions are written.

- **TRAP** - M_pass is a prerequisite. v2 says >=4/5 is proposed but MK can set it (v2:217, v2:259). That is exactly the kind of decision that cannot remain open when the run starts.

- **SAFE** - FEVER NEI is not a prerequisite if it remains exploratory. Binary-only FEVER is cleanest and already scoped (v2:160-163, v2:260). If the NEI-collapsed cell is included anywhere near the headline, then it becomes a prerequisite.

- **TRAP** - FEVER source is a prerequisite. `copenlu` GPL clearance versus self-join determines the data artifact, evidence text, grouping behavior, and license status (v2:100, v2:157-159, v2:261).

- **TRAP** - HaluEval license/source is a prerequisite. v2 marks it blocked and recommends RUCAIBox MIT source (v2:97, v2:106-108, v2:262). It cannot be in a confirmatory primary set until this is closed.

- **UNKNOWN** - SimpleQA is not a prerequisite if excluded. If a pilot can add it conditionally, the pilot must be explicitly exploratory and cannot modify M_pass or the primary denominator (v2:134, v2:263).

- **TRAP** - P3 balance policy is a prerequisite if P3 is primary. Shared questions with floating base rates versus per-model 0.50 subsampling changes the estimand and LOMO semantics (v2:173, v2:264). Freeze it before generation.

## Does §6 Hold Water For 50+ Cells?

**TRAP** - Not yet.

The good news: v2 has the right instincts. It separates confirmatory from exploratory (v2:204-207), defines a 5-task shared-label primary set (v2:208-211), registers per-task deployability and LOMO bars (v2:213-218), isolates P3 (v2:220-221), and demotes the max-over-landscape view (v2:226-227).

The failure is that the actual primary paper-level endpoint is still not cleanly pre-registered. `fixed_cell_max_survival` is not a frozen named signal; it is a best fixed cell selected from the landscape after evaluation (v2:211, v2:216). Across 5 tasks and 29 cells, that is not fatal if registered as a selection procedure, but v2 currently sells it as "one fixed cell" and says multiplicity is not primary (v2:226-227). Those two statements conflict.

The second failure is conditionality. A 50-cell plan cannot have unresolved task inclusion, HaluEval source, FEVER source, M_pass, and P3 balance decisions sitting in §9 (v2:256-265). Those choices define the denominator and the estimand. They are not administrative.

The third failure is dependence. If every primary task is stem-paired or stem-correlated (v2:186-195), row-bootstrap deployability should not remain the confirmatory gate (v2:196). At 50 cells, this will look like knowingly choosing the narrower CI.

## Byte-Comparability Phrasing

**TRAP** - The §1.1 phrasing is much more honest than v1, but still not quite honest enough.

Acceptable: v2 says T0 extraction can remain byte-identical and the commit-confluence harness changes (v2:32-52). It also specifies a T0-subset hash rule, extension-baseline hashes, model snapshot equality, and runtime stack matching (v2:54-60).

Still overstated: v2 says the extension is "separately-hashed" (v2:52) after admitting that launcher/gate/builders/transfer edits are not included in `module_hashes()` (v2:49). Also §0 says those same scripts are recorded in `module_hashes` (v2:17), which is false.

Use this instead:

> The T0 extraction modules can remain byte-identical to the seal and will be checked by a T0-subset hash comparison. The hot confluence compute path (`confluence_calibrator.py` and `fusion_signs.json`) is frozen under a new extension-baseline hash. The launcher, data builders, gate, analysis scripts, and pre-registration are changed extension harness code; they must be versioned and archived with the extension, but they are not part of the seal's `module_hashes()` record.

If you want to keep "separately hashed," add explicit hashes for `run_seal.py`, `check_fresh_data.py`, `generate_fresh_data.py`, `analyze_universality.py`, and `PRE_REGISTRATION_BENCH.md` to the extension manifest.

## Required Fixes Before Execution

1. Fix §0 and §1.1 so hash coverage is exact. Either hash the full extension harness or stop saying the harness is separately hashed (v2:17, v2:49-58).

2. Replace the primary LOMO endpoint with one of two explicit choices: a frozen named cell such as `fusion_rank_mean_geom`, or a registered best-fixed-cell selection procedure with multiplicity acknowledged (v2:211, v2:216, v2:226-227).

3. Freeze the primary task set and fallback denominator before any builder/smoke/feature matrix. Close HaluEval source, FEVER source, effort envelope, and M_pass (v2:209, v2:217, v2:256-265).

4. Make P3's primary status real or downgrade it. Define base-rate thresholds, label-noise thresholds, first-token audit scope, model-drop rules, and the headline consequence of P3 failure (v2:168-173, v2:209, v2:220-221, v2:264).

5. For grouped tasks, make the stem-cluster bootstrap confirmatory or explicitly stop calling row-bootstrap deployability confirmatory on grouped datasets (v2:193-196).

6. Strengthen the smoke gate from "limit-8" to stratified smoke coverage for each new task family: long/short, label 0/1, subset type, and FEVER evidence complexity where applicable (v2:81-85).

7. Do not freeze FEVER until the raw-vs-joined source and license status are decided and the evidence grouping policy is specified (v2:99-100, v2:157-159, v2:261).

8. Keep SimpleQA pilot strictly exploratory unless its inclusion rule and denominator effect are frozen in advance (v2:134, v2:263).

## Final Verdict

**FIX.** v2 is no longer scientifically reckless, but it is still preregistration-incomplete. The most important remaining flaw is the claimed "fixed" universal-floor endpoint: as currently described, it is still selected from the post-run cell landscape. Fix that, freeze the §9 prerequisites, and make P3/cluster-bootstrap rules explicit before execution.
