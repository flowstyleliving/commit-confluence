# Adversarial Review — Benchmark Expansion Proposal

Proposal reviewed: `/Users/msrk/Documents/the_GOAT/wiki/paper/benchmark-expansion-proposal.md`

Code checked:

- `/Users/msrk/Documents/commit-confluence/confluence_calibrator.py`
- `/Users/msrk/Documents/commit-confluence/stage_b/run_seal.py`
- `/Users/msrk/Documents/commit-confluence/stage_b/analyze_universality.py`
- `/Users/msrk/Documents/commit-confluence/stage_b/check_fresh_data.py`
- `/Users/msrk/Documents/commit-confluence/stage_b/generate_fresh_data.py`
- `/Users/msrk/Documents/t0-morphology-furnace/pri_calibrator.py`
- `/Users/msrk/Documents/t0-morphology-furnace/pri_v2_io_plugins.py`
- `/Users/msrk/Documents/t0-morphology-furnace/exploratory/shadow-ambiguity/comprehensive_run.py`
- `/Users/msrk/Documents/the_GOAT/wiki/paper/cc-draft.tex`

External factual checks used:

- HaluEval repo: https://github.com/RUCAIBox/HaluEval
- HaluEval HF mirror: https://huggingface.co/datasets/pminervini/HaluEval
- TruthfulQA HF: https://huggingface.co/datasets/truthfulqa/truthful_qa
- TruthfulQA paper: https://arxiv.org/abs/2109.07958
- FEVER paper: https://arxiv.org/abs/1803.05355
- FEVER HF: https://huggingface.co/datasets/fever/fever
- RAGTruth paper: https://arxiv.org/abs/2401.00396
- SimpleQA paper: https://arxiv.org/abs/2411.04368

## Byte-Comparability And Code Path

- **SAFE** — The narrow claim that the readout feature extractor accepts arbitrary prompt/label JSONL is correct. The proposal says `_load_calibration_jsonl(data_path)` accepts any `{"prompt": str, "label": 0|1}` JSONL (proposal lines 41-44). Actual code confirms `pri_calibrator._load_calibration_jsonl` only reads `row["prompt"]`, casts `row["label"]` to int, rejects labels outside `{0,1}`, and ignores all metadata (`pri_calibrator.py:516-539`).

- **SAFE** — The narrow claim that `benchmark` is not used for task-specific feature branching inside `trace_pair_features` is correct. The proposal says the benchmark string is only print/inventory (proposal lines 43-45). Actual `trace_pair_features` loads prompts and labels first, then uses `benchmark` only in the inventory print at `comprehensive_run.py:321`; the feature loop is prompt/label driven (`comprehensive_run.py:273-380`).

- **SAFE** — The proposal correctly flags that `discover_benchmarks()` has hardcoded known benchmark names but is not used by the confluence `run_cell` path. `comprehensive_run._benchmark_name` explicitly recognizes `anli_r1/r2/r3` and `triviaqa_paired` (`comprehensive_run.py:133-150`), while `confluence_calibrator.collect_readout_matrix_fresh` passes an explicit path into `CR.trace_pair_features` (`confluence_calibrator.py:89-101`).

- **TRAP** — "The pipeline is benchmark-agnostic at the extraction layer" is too broad as written (proposal line 11; expanded at lines 32-52). The feature function is mostly benchmark-agnostic, but the *registered run path* is not. `run_seal.py` only accepts `--anli` and `--triviaqa` (`run_seal.py:120-136`), builds `tasks` with exactly `anli_r1` and `triviaqa_paired` (`run_seal.py:134-136`), maps gates with `gate_task = {"anli_r1": "anli", "triviaqa_paired": "triviaqa"}` (`run_seal.py:165-172`), and only knows sealed refs for those two tasks (`run_seal.py:40-44`). New benchmarks require launcher and gate edits before they are runnable as registered cells. The proposal admits some harness edits at lines 93 and 181-183, but that contradicts the "drop-in" tone in lines 11 and 52.

- **TRAP** — "No edits to the frozen `t0-morphology-furnace` core" is plausible, but "module_hashes match" is not guaranteed for a benchmark expansion (proposal line 89). `module_hashes()` includes `confluence_calibrator.py` and `fusion_signs.json` in addition to sealed T0 modules (`confluence_calibrator.py:265-295`). Any necessary change to `confluence_calibrator.py` or fusion spec behavior for new tasks changes the byte hash recorded in extension profiles, even if `t0-morphology-furnace` is untouched. The honest phrasing should be: T0 extraction can remain byte-identical; the commit-confluence harness will not.

- **TRAP** — The proposal says all four signal families are read at "commit instant `gen_step=1`" (proposal line 46) and calls P3 extraction at `gen_step=1` (proposal line 125). The current unified panel is mixed-locus: ACE is t=0/prefix-last attention (`run_seal.py:5`; `confluence_calibrator.py:330-381`; `pri_calibrator.py:592-606`), while the readout pass uses generated step 1 (`confluence_calibrator.py:89-101`; `comprehensive_run.py:321-380`). The paper draft also describes winning glossary entries as read at "commit position (step 0)" (`cc-draft.tex:232-234`). This is not just prose: if P3 is sold as "before the answer is shown," the review must explain exactly which columns are pre-token and which columns are first-token/post-token.

- **UNKNOWN** — The proposal assumes P1/P2/P4/P5 are "fully byte-comparable to the seal" (proposal lines 22, 95). This is only true if the new run uses the same model snapshots, same prompt strategy code, same fusion spec semantics, same seed discipline, same no-drop strictness, and same package/runtime versions. Current profiles record model snapshots and module hashes (`confluence_calibrator.py:298-322`, `run_seal.py:76-116`), but the proposal does not specify a pass/fail rule for hash drift across 50+ new cells.

## Format Contract

- **SAFE** — The proposal correctly states that a new benchmark cannot be arbitrary text classification; it must end in a cue where the model commits to a one-token answer (proposal line 63). The extraction captures only one generated token for both ACE/readout under `run_cell(... max_new_tokens=1 ...)` (`run_seal.py:53-58`).

- **TRAP** — The proposal's "new benchmarks just need `{"prompt","label"}` JSONL" framing is dangerously incomplete (proposal lines 43, 52, 56-63). The loader accepts only prompt/label, but the *scientific contract* also includes: one-token commitment, stable prompt wrapping, no model-specific empty/CoT outputs, no hidden truncation, no systematic first-token whitespace/newline artifacts, and no prompt family that changes the semantic locus. `pri_v2_io_plugins.py` uses raw prompts for most listed models but chat templates for some models (`pri_v2_io_plugins.py:207-259`). A prompt that works raw may behave differently under `apply_chat_template`, and vice versa.

- **TRAP** — There is no behavioral gate for new prompt formats. Existing builders use very specific ANLI and TriviaQA templates (`generate_fresh_data.py:48-64`), and the prompt strategy dict explicitly says some models emit "empty / CoT-overflow / garbled output" without the right wrapper (`pri_v2_io_plugins.py:219-244`). New HaluEval/FEVER/TruthfulQA prompts need smoke tests proving every model produces a meaningful first token under the actual prompt strategy. The proposal mentions length drops but not output-format failures (proposal lines 109, 184).

- **TRAP** — The current fresh-data gate is not generic. It only accepts `--task {anli,triviaqa}` (`check_fresh_data.py:68-73`), has TriviaQA-specific exact 0.50 balance and question-id logic (`check_fresh_data.py:105-156`), and requires a sealed reference file (`check_fresh_data.py:70-72`, `115-124`). The proposal says add a generic path (proposal lines 93, 183), but its "brand-new benchmark cannot be contaminated by the 20260526 seal by construction" claim (proposal line 183) is incomplete: contamination can still occur through reused TriviaQA/HotpotQA/CNN-DM/FEVER examples, prompt templates, source stems, or benchmark overlap with existing TriviaQA questions. "Not in the sealed file" is not the only freshness condition.

- **UNKNOWN** — Max context length is not audited in the proposal. Long HaluEval summarization documents and FEVER evidence may exceed the practical prompt length of smaller local MLX models. The strict merge aborts on any dropped row (`merge_matrices` with `max_dropped=0`, `confluence_calibrator.py:384-421`; `run_seal.py:56-58`), but the proposal does not specify per-model token length prefilters, a common intersection set, or what happens when one model cannot score enough examples.

## Benchmark Selection

- **SAFE** — HaluEval is a legitimate hallucination benchmark and the proposal's broad description is mostly correct. The official repo says it has 35K examples, including 30K task-specific examples across QA, dialogue, and summarization plus 5K general user-query examples; each task-specific file has 10K samples and fields for right and hallucinated outputs. It is also MIT licensed in the official repo. This supports proposal lines 101-107.

- **TRAP** — The proposal cites HF `pminervini/HaluEval` and says "MIT in the source repo" (proposal line 106). The HF mirror currently displays `apache-2.0`, while the official RUCAIBox repo displays MIT. That mismatch must be resolved before redistribution or pinning. Do not write "license verified" until the exact artifact source is chosen.

- **TRAP** — HaluEval is not automatically "source-faithfulness" in the clean RAG sense claimed at proposal lines 103-105. The hallucinated outputs are generated by ChatGPT via a sampling/filtering pipeline, not naturally generated by the subject models. The official repo also says evaluation samples either ground-truth or hallucinated output and asks the model to recognize hallucination. This is a good Framing A recognition benchmark, but it does not test the subject model's own hallucination and may let detectors key on ChatGPT-generated text artifacts.

- **SAFE** — TruthfulQA has 817 questions and is designed around imitative falsehoods/misconceptions, matching proposal lines 111-117. The HF dataset is Apache-2.0 and supports multiple-choice / generation tasks.

- **TRAP** — TruthfulQA-MC is not a strong enough "generation-based hallucination" substitute and is only a weak breadth add. The proposal correctly does not make it P3, but overstates its strategic value at proposal lines 117-118. MC rows measure recognition/ranking of candidate answers. They do not show the model was about to generate the false answer. They also create heavy stem correlation if expanded into `(question, candidate-answer)` rows.

- **SAFE** — FEVER is a real, recognizable fact-verification benchmark. The original paper describes 185,445 Wikipedia-derived claims labeled Supported, Refuted, or NotEnoughInfo, with evidence recorded for Supported/Refuted examples. This supports proposal lines 134-140 at a high level.

- **TRAP** — FEVER integration is understated. The proposal says "claim + gold evidence -> SUPPORTED/REFUTED" and "joining claims with gold evidence (solved by the `fever_gold_evidence` variant)" (proposal lines 134-142). The HF `fever/fever` record exposes evidence page/sentence identifiers, not necessarily a clean joined multi-sentence evidence text; v1 examples are single evidence rows, while v2 validation includes NEI examples with no evidence. Multi-evidence claims, duplicate evidence rows, page text lookup, sentence ordering, and evidence-set grouping must be specified. A sloppy join will either leak only one sentence from a multi-hop evidence set or duplicate claims.

- **TRAP** — Dropping FEVER NEI weakens the hallucination analogy. Proposal line 136 says drop NEI. But unsupported/baseless claims are exactly a key hallucination class. SUPPORTED vs REFUTED tests contradiction against evidence; it does not test "not enough evidence" unsupportedness. If the paper says "unsupported answers," FEVER without NEI is narrower than advertised.

- **TRAP** — ANLI R2/R3 is not "benchmark expansion" in the same sense. The proposal admits it adds no new failure mode (proposal lines 144-150). It should be framed as robustness only, not counted toward "5-7 benchmarks" if the venue concern is hallucination/factuality breadth.

- **UNKNOWN** — The proposal does not compare against stronger or more directly relevant alternatives. For Framing B, SimpleQA is designed for short fact-seeking answers with single indisputable answers and labels correct/incorrect/not attempted; it is a closer generation factuality candidate than TriviaQA-gen. For grounded/source-faithfulness, RAGTruth has naturally generated RAG responses with manual hallucination annotations at case and word level, and is closer to real deployment than HaluEval's ChatGPT-filtered hallucinations. These may be harder to fit the single-token protocol, but the proposal must justify excluding them instead of acting as though HaluEval/TruthfulQA/FEVER are the obvious complete set.

- **TRAP** — MMLU, GSM8K, and HellaSwag should not be added as "hallucination" benchmarks. If the author is tempted to list them for reviewer familiarity, that waters down the paper. MMLU/HellaSwag are multiple-choice capability/commonsense tests; GSM8K is mathematical reasoning. They can be negative-control or task-diversity appendices, but they do not directly test unsupported factual generation. The missing-benchmark fix should prefer factuality/grounding datasets, not generic capability leaderboards.

## Label Balance And Effective Sample Size

- **SAFE** — For HaluEval-style right/hallucinated pairs, exact 50/50 row balance is achievable by emitting both the right and hallucinated output per source item (proposal lines 103-105, 186).

- **TRAP** — "Framing A (paired) is exactly 0.50 for free" is false as a general statement (proposal line 186). It is true for HaluEval if both sides are emitted, and true for the existing TriviaQA paired builder. It is not inherently true for TruthfulQA MC or FEVER unless the builder performs explicit class-balanced sampling. The proposal also lumps FEVER into "easy" balance despite class skew and the need to know the exact evidence variant.

- **TRAP** — Row balance is not the same as independent evidence. TruthfulQA expanded MC rows share the same question stem; HaluEval paired rows share the same context; TriviaQA paired rows already share a question; FEVER may duplicate claims across evidence rows if joined incorrectly. Current bootstrap/selection treats rows as independent samples. If a benchmark emits multiple rows per stem, the proposal needs group-aware sampling or at least a grouped sensitivity analysis. Otherwise `n=200` is inflated.

- **UNKNOWN** — TruthfulQA at `n=200` can be row-balanced, but whether it can be *stem-balanced* and diverse depends on the expansion policy. If each question contributes one true and one false candidate, 100 question stems are enough. If MC2 contributes many true/false references per question, naive sampling can overrepresent a small number of stems. The proposal does not specify stem caps.

- **UNKNOWN** — FEVER can almost certainly supply 100 SUPPORTS and 100 REFUTES examples from the full dataset, but the exact joined gold-evidence variant may not preserve enough clean, short, single-evidence examples after length filtering and duplicate removal. This must be measured before pre-registration.

## P3 / Generation-Based Claim

- **SAFE** — The proposal correctly identifies the central reviewer objection: existing ANLI and TriviaQA paired tasks judge a supplied statement/answer, not the model's own generated hallucination (proposal lines 70-81). The current paper draft indeed claims deployment-time monitoring of unsupported answers before the answer is shown (`cc-draft.tex:72-79`), while the current cohort is ANLI plus paired TriviaQA judgment (`cc-draft.tex:173-176`).

- **TRAP** — TriviaQA-generation is not automatically a valid hallucination benchmark. Exact-match failure on closed-book QA conflates hallucination, ignorance, abstention, spelling/formatting mismatch, partial answer, alternative valid alias, and ambiguous reference. Label `1 = no alias match` is "not exact-match correct," not necessarily "hallucination." The proposal calls label 1 a hallucination (proposal lines 123-127); that is an overclaim.

- **TRAP** — The proposal says "detected at the first committed answer token" and "before the answer is shown" (proposal lines 123-127). But generation labels are assigned after `max_new_tokens≈16-32` (proposal line 125). The first token may be whitespace, newline, "The", an article, a refusal preamble, or a formatting token. A later exact-match failure does not prove the first token already committed to a hallucinated answer. The paper must call this "predicting eventual short-answer correctness from the commit state," not "catching hallucination at token 1," unless a manual audit confirms first-token answer commitment.

- **TRAP** — The proposal says "reuse the alias logic already in `generate_fresh_data.py::_all_aliases`" (proposal line 125), but that function only lowercases/strips aliases for wrong-answer collision avoidance (`generate_fresh_data.py:151-154`); it is not a complete answer grader. It does not normalize punctuation, articles, Unicode, dates, parentheticals, aliases across Wikidata, partial names, or multi-answer lists. It will create noisy labels.

- **TRAP** — P3 breaks the existing shared-label assumptions more than the proposal admits. The proposal notes per-model labels and LOMO semantics shift (proposal lines 128-132), but it understates downstream consequences: transfer across models is now comparing different label functions; label-efficiency subsampling is per-model; class balance differs per model; and "same examples" no longer means same target. A single N-task transfer matrix mixing P3 with shared-label tasks will be hard to interpret.

- **TRAP** — Offline generation must use the exact same prompt strategy and model snapshot as extraction. Current extraction applies `state.prompt_strategy` for ACE (`confluence_calibrator.py:358-367`) and `io_plugins.get_prompt_strategy` for readout (`comprehensive_run.py:299, 326-339`). If the offline generator uses raw prompts while extraction uses a chat template for Mistral-Nemo, labels can correspond to different first-token distributions. The proposal says pin decoding/seed (proposal line 132) but must also pin prompt wrapping, tokenizer, model snapshot, max context behavior, stop criteria, and generated-text parser.

- **UNKNOWN** — P3 balance may fail for strong or weak models. The proposal says keep shared questions and let balance float within +/-0.10 (proposal line 131), but some models may be far outside 40/60 on TriviaQA-gen. If so, the current gate would fail or AUROC CIs become unstable. Need a pilot that only reports base rates and label-noise audits before registering bars.

## Pre-Registration Feasibility

- **SAFE** — The proposal is right that the original `19/20` and `17/20` bars do not transfer to an enlarged denominator (proposal line 91). `run_seal.py` hardcodes bars only when `n_planned == 20`; otherwise it scales to 95% and 85% (`run_seal.py:232-234`), which would be inappropriate for a new benchmark-expansion claim unless re-registered.

- **SAFE** — The proposal correctly says a new profiles directory is needed and the sealed 20 deployments should not be reopened (proposal lines 89-95). The paper draft already has a pattern for that (`cc-draft.tex:297-303`).

- **TRAP** — "Each new (model, benchmark) cell individually deployable-or-not" is acceptable for a small diagnostic model-extension like `PRE_REGISTRATION_EXT.md`, but it is too weak for a benchmark expansion whose purpose is to strengthen a paper-level claim (proposal lines 91-92, 165-170). With 50+ cells, no cohort-level success rule invites cherry-picking: any failures can be narrated as "honest orphan classes" and any passes as breadth. That waters down pre-registration unless the author freezes: primary benchmarks, primary endpoint, per-task success/failure rule, how many failed tasks falsify the expansion, and how P3 model-specific labels are analyzed.

- **TRAP** — The proposal says "freeze predictions before the metric" (proposal line 92) but does not list actual predictions or bars for the proposed benchmark tasks. The EXT precedent froze per-cell qualitative predictions because it tested a specific orphan hypothesis; a benchmark expansion needs stronger preregistration than LEAN-YES/NO per cell. It needs a priori claims like "geometric endpoint deployable on at least X/Y cells for HaluEval summarization" and "fixed fusion LOMO clears >=8/10 on at least K of M new shared-label tasks."

- **UNKNOWN** — The proposal does not define whether the benchmark expansion is confirmatory or exploratory. If confirmatory, every data-builder, filter, prompt template, length policy, label policy, model inclusion/exclusion rule, and task-level bar must be frozen before any feature matrices are drawn. If exploratory, it can still be valuable but should not be used to upgrade the sealed paper's headline claims without clear labeling.

## Paper Claim Compatibility

- **SAFE** — The expansion does not need to alter the sealed "18/20" claim if results are isolated as a post-seal extension. The paper draft already has a pattern for that (`cc-draft.tex:297-303`).

- **TRAP** — The current paper already uses language that the existing data only partially supports. It says a reliable signal would flag unsupported answers before the answer is shown (`cc-draft.tex:72-79`), but the actual cohort is ANLI entailment/contradiction plus paired TriviaQA correct-vs-wrong judgment (`cc-draft.tex:173-176`). The proposal correctly identifies P3 as the only cell that makes this literal (proposal line 201), but until P3 is implemented and audited, the paper should hedge "unsupported answers" as "unsupported/wrong-answer analogs under judgment prompts."

- **TRAP** — Adding HaluEval/TruthfulQA/FEVER could weaken "confidence is not the backstop" if confidence wins or rescues many new cells. The proposal frames TruthfulQA as the strongest test of confidence failing (proposal lines 113-118), but that is an empirical hope, not a safe claim. If confidence dominates TruthfulQA or FEVER, the existing paper's "confidence rescues nothing" becomes "confidence rescued nothing on two tasks, but may matter elsewhere."

- **TRAP** — Adding many new task columns can weaken the "universal above-chance floor" if fusion fails on summarization/dialogue/generation. The proposal says failure would be "equally publishable" (proposal line 107), but it would absolutely narrow the existing claim. The paper currently says one fixed aggregate clears a floor on both tasks (`cc-draft.tex:256-264`). A failed HaluEval-Summarization or P3 result would require changing the claim to "floor on original two tasks, not universal across hallucination regimes."

- **TRAP** — The title "No Universal Detector, but a Universal Floor" becomes riskier with more benchmarks. Right now "universal" means across ten models and two tasks. If the expansion adds five heterogeneous tasks and the floor fails on any, the title overstates. The preregistration must define what "universal" now ranges over.

- **UNKNOWN** — Existing figures and descriptive analyses assume two tasks in places. E1 is per task and E3 is per cell, but E2 explicitly skips if not exactly two tasks (`analyze_universality.py:135-139`). The proposal says this is a 20-line edit (proposal line 68). That estimate may be optimistic because transfer semantics are different for P3 model-specific labels and for tasks with different prompt formats.

## Missing Or Mishandled Alternatives

- **TRAP** — The proposal's "If you can only do three: HaluEval + TruthfulQA + TriviaQA-gen" (proposal line 175) is not obviously optimal. It leaves out a modern short-form factuality benchmark designed for generation grading (SimpleQA) and a real RAG/source-faithfulness corpus (RAGTruth). If the goal is to answer "real hallucination detection," HaluEval + SimpleQA or NQ-open generation + RAGTruth may be stronger than TruthfulQA-MC + TriviaQA-gen.

- **TRAP** — FActScore is dismissed too quickly (proposal lines 154-157). It does not fit the current single-token protocol, so deferring it is practical. But dismissing it as captured by TriviaQA-gen is scientifically wrong: FActScore tests long-form atomic factual precision; TriviaQA-gen tests short-answer exact match. They are not substitutes.

- **UNKNOWN** — The proposal does not decide whether to include "not attempted"/abstention as its own class. SimpleQA's correct/incorrect/not-attempted distinction would be especially useful because an abstention is not a hallucination. The current binary label contract cannot express that without either dropping abstentions or incorrectly grouping them with hallucinations.

- **UNKNOWN** — No plan is given for domain diversity. FEVER/HaluEval/TriviaQA all draw heavily from Wikipedia-adjacent or popular web knowledge sources; TruthfulQA is misconception-heavy. If reviewers ask for medical/legal/scientific factuality, SciFact or domain fact-verification alternatives may be more targeted than adding ANLI R2/R3.

## Verdict

**FIX** — The proposal has a valid core but is not ready to execute or send back as a clean plan.

Must fix before proceeding:

1. Rewrite the byte-comparability claim: extraction is mostly benchmark-agnostic; the launcher, gate, data builders, transfer analysis, and preregistration are not.
2. Correct the commit-locus language: ACE is t=0/prefix-last; readout is gen_step=1. Do not claim all four families are read at gen_step=1.
3. Replace "JSONL prompt/label is enough" with a full prompt contract: one-token cue, model-specific wrapping, length limits, no-drop policy, first-token sanity checks, and per-model smoke gates.
4. Define real preregistration bars for the benchmark expansion. "Each cell deployable-or-not" is too weak for 50+ cells.
5. Rework P3. Call it "predicting generated short-answer correctness" unless manual audit proves non-exact matches are hallucinations and first-token state really precedes answer commitment. Implement robust answer normalization and audit label noise.
6. Specify group-aware sampling or grouped sensitivity analyses for paired/stem-correlated datasets.
7. Resolve FEVER evidence joining and decide whether NEI is included. Dropping NEI narrows the unsupported-answer claim.
8. Verify dataset licenses, exact HF/source revisions, and local availability before freezing anything.
9. Compare against SimpleQA and RAGTruth explicitly. Either include them, or justify why they do not fit the single-token protocol better than TriviaQA-gen/HaluEval.
10. Update paper language so any expansion cannot silently rewrite the sealed 18/20 claims. New results must be a labeled post-seal benchmark extension, not an altered seal.

