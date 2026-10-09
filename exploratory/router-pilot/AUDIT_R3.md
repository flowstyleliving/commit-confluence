# Round-3 static audit and fixes

Date: 2026-10-08. Changes are confined to `exploratory/router-pilot/`.
Verification: **not run by Codex**. No Python, runtime code, tests, Modal, pip, imports or builds
were run. The original audit is reproduced verbatim below; its line references describe the
pre-fix files. The remediation references describe the files after this edit.

## Original round-3 audit (verbatim)

Static audit only: no runtime code, tests, builds, or file modifications.

- **MAJOR — Measured failures remain amendable.** [modal_router_pilot.py:178](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/modal_router_pilot.py:178), also lines 305–338: zero reconstruction reference, hardware requirements, eager/no-quantization configuration, head dimensions, and MoE dimensions still use `require()`. These are registered measured/load assertions, yet `smoke_failure()` classifies their failures as mechanical. Likewise, deadline exhaustion at line 163 is mechanical, and the exception path skips the post-call budget check at line 624. **Fix:** use `gate()` for registered factual/numerical assertions; terminalize budget exhaustion explicitly. Keep actual instrumentation shape/programming errors mechanically repairable.

- **MAJOR — Missing pinned revision can bypass terminal classification.** [modal_router_pilot.py:111](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/modal_router_pilot.py:111), lines 553–568: tokenizer loading precedes the protected `snapshot_download()`. Transformers wraps repository/revision failures in `OSError` with the Hub exception as its cause; this becomes an amendable mechanical failure without reaching the new classifier. **Fix:** perform the protected snapshot download first, load the tokenizer locally, and inspect exception causes where necessary. The Hub 0.36 [signature and error imports](https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.0/src/huggingface_hub/_snapshot_download.py) themselves are valid.

- **MAJOR — Coordinator exception coverage leaves lifecycle holes.** [modal_router_pilot.py:498](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/modal_router_pilot.py:498), lines 577–633: claim acquisition precedes the protected `try`; reload/start-event failures can strand it. Later failures publishing the freeze snapshot or updating the reservation occur outside both smoke handlers. The outer handler ignores smoke entirely and terminalizes extraction only when `reserved=True`. Consequently, some smoke failures leave no immutable failure record and permit the same freeze entry to be attempted again; some pre-reservation extraction failures leave cells nonterminal. **Fix:** protect all work after successful acquisition, persist an immutable per-entry attempt marker before preflight, and centralize failure classification/publication. Release the owned claim only after durable accounting/failure state; preserve an explicit reconciliation path when storage fails.

- **MAJOR — Descriptive correlation can prevent the spending result from being published.** [score_router_pilot.py:55](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/score_router_pilot.py:55), lines 214–216 and 291: constant finite `sink_draw` or `final_bos_mass` produces `NaN` from `corrcoef`. Such columns are permitted by extraction. `json_bytes(...allow_nan=False)` then fails after the irreversible single-look claim, leaving no published decision. **Fix:** detect zero rank variance and report JSON `null` with an “undefined: constant input” reason. A descriptive diagnostic must not veto publication.

- **MINOR — Modal fallback classification is not reliable.** [make_freeze.py:37](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/make_freeze.py:37): ordinary Modal 1.5.1 exception deserialization re-raises `GateFailure` correctly. However, its deserialization-error fallback prepends “Could not deserialize remote exception…” and embeds the remote traceback; `startswith("GATE:")` misses that case. **Fix:** return a structured gate-failure envelope from the worker, or explicitly handle the SDK fallback and exception chain without treating an ambiguous remote failure as amendable.

- **MINOR — S2 charges beyond launch-to-return.** [modal_router_pilot.py:621](/Users/msrk/Documents/commit-confluence/exploratory/router-pilot/modal_router_pilot.py:621): elapsed time includes subsequent artifact publication, volume commits, and failure handling. Slow storage therefore consumes GPU budget after the call returned. **Fix:** capture the return/exception timestamp immediately around `.remote()` and charge that interval plus 30 seconds. Full charging of genuinely unclosed reservations is otherwise retained.

The metric implementation checks out statically: six ATT metrics; `(16,T)` v-norm captures passed in the sealed kernel’s expected list format; correct MHA head correspondence; and `sink_draw = mean_h(A[h,-1,0] × ||V[h,0]||)`. NPZ/scorer axes place `sink_best` at columns 132–147, outside `COMPARATORS` and the threshold calculation. The Spearman issue above is a publication dependency, not numerical selection leakage.

No additional tensor-layout or deterministic-operation incompatibility was identified. PyTorch 2.8 documents deterministic CUDA `index_add`, and the required [cuBLAS workspace setting](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html) is present. Worker volume reload occurs before model files are opened. Hub [`token=False`](https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.0/src/huggingface_hub/utils/_headers.py) is valid. Smoke/coordinator remain label-blind, with no outcome-score printing found. S1’s withdrawal is justified by README’s Python 3.12 environment.

**NO-GO for FREEZE-1 + smoke until the MAJOR findings are corrected.**

## Changes per finding

1. **MAJOR — Measured failures remain amendable.** `modal_router_pilot.py:163` makes the
   worker deadline a gate; `modal_router_pilot.py:175` gates zero reconstruction reference.
   `modal_router_pilot.py:303` gates GPU count, the A10G name (no generic A10 match), memory,
   eager/no-quantization configuration, layer/eval, heads, MoE dimensions and environment pins.
   Actual dtype/device facts are gated at `modal_router_pilot.py:167`; instrumentation shapes
   there and at `modal_router_pilot.py:247` remain `require()` failures. Imports still propagate
   mechanically. Pre-call exhaustion gates are at `modal_router_pilot.py:562` and
   `modal_router_pilot.py:604`; `modal_router_pilot.py:623` checks the deadline and accumulated
   budget on both return and exception paths, with terminal exhaustion taking precedence before
   success publication. `modal_router_pilot.py:480` publishes the terminal state.

2. **MAJOR — Missing pinned revision can bypass terminal classification.** The protected
   pinned `snapshot_download()` now precedes all tokenizer loading at
   `modal_router_pilot.py:584`; `modal_router_pilot.py:111` loads the tokenizer locally only.
   `make_freeze.py:37` walks both `__cause__` and `__context__` with cycle protection;
   `make_freeze.py:49` recognizes Hub repository/revision-not-found classes, including their
   subclasses, anywhere in that chain without adding an import dependency to failure handling.
   `modal_router_pilot.py:480` publishes the gate failure and returns a `GateFailure` for the
   coordinator to re-raise, including transformers' `OSError` wrappers.

3. **MAJOR — Coordinator exception coverage leaves lifecycle holes.**
   `modal_router_pilot.py:547` protects reload/start accounting and all subsequent owned work.
   `modal_router_pilot.py:550` commits an immutable per-entry attempt marker before preflight;
   marker or prior failure existence refuses a second attempt. `modal_router_pilot.py:480`
   centralizes classification/publication for smoke, extraction, release failure and expired-claim
   reconciliation. Smoke failures record immutable `smoke.failed.<entry>.json`; gate failures
   terminalize both smoke cells; extraction failures terminalize remaining cells regardless of
   reservation. `modal_router_pilot.py:514` retains a sticky storage-failure flag.
   `modal_router_pilot.py:657` requires durable failure/accounting state and no unclosed
   reservation before release; `modal_router_pilot.py:667` checks claim ownership and routes
   release failures through the same publisher. `modal_router_pilot.py:524` stores original
   freeze/kind with the claim; expired-claim recovery publishes failures/terminal states and
   closes unclosed reservations at full charge without another GPU launch. README documents
   recovery after storage failure, including manual inspection if reconciliation itself fails.

4. **MAJOR — Descriptive correlation can prevent publication.** `score_router_pilot.py:55`
   checks each input's rank variance and emits a value/reason record, with JSON null and
   `undefined: constant input` on constancy. `score_router_pilot.py:64` contains diagnostic
   exceptions, recursively replaces non-finite diagnostic results with null/reason, and checks
   strict JSON serialization inside the diagnostic guard. Fold selection descriptions, curves
   and dispersion are computed under that guard (`score_router_pilot.py:96`,
   `score_router_pilot.py:145`). Compliance, Spearman and the KV/JS diagnostic are guarded at
   `score_router_pilot.py:247`; even finite operands can overflow during the latter subtraction.
   The registered selections and decision are unchanged. Static review of other NaN paths:
   AUROCs are finite-checked (`score_router_pilot.py:48`); five nonempty folds give bounded arm
   means/differences (`score_router_pilot.py:116`); bootstrap intervals return null for no draws
   and otherwise operate on bounded finite values (`score_router_pilot.py:236`); permutation
   draws use the same checked crossfit. Empty block summaries already use null and are now
   guarded as well. Worker metadata is strict-JSON serialized before NPZ publication, and the
   scorer checks the NPZ hash. No other NaN path to final strict JSON publication was identified
   for artifacts produced by this implementation. Corrupt artifacts, core-analysis failures or
   storage failures are not silently converted into successful scientific results.

5. **MINOR — Modal fallback classification is not reliable.** `modal_router_pilot.py:356`
   returns a structured type/message/traceback envelope for gate failures; ordinary mechanical
   exceptions still propagate. `modal_router_pilot.py:630` re-raises that envelope as
   `GateFailure`. `make_freeze.py:49` also detects `GATE:` anywhere in every exception-chain
   message, including Modal's embedded remote traceback fallback.

6. **MINOR — S2 charges beyond launch-to-return.** `modal_router_pilot.py:612` prepares the
   remote callable before starting a monotonic clock. `modal_router_pilot.py:614` through
   `modal_router_pilot.py:622` measures only the call interval on return or exception, plus
   30 seconds. The end event is committed before result publication; even malformed envelopes
   close the observed interval (`modal_router_pilot.py:635`). `modal_router_pilot.py:472` and
   `modal_router_pilot.py:538` retain full charging for genuinely unclosed reservations.

## Preserved scope and static evidence

- Six captured ATT metrics and five selectable ATT metrics; deprecated `final_v_norm_bos`;
  unchanged sink-draw formula and `(16,T)` layout; `sink_best` columns 132–147 remain outside
  `COMPARATORS` and the spending decision. No thresholds, seeds, resampling counts or fold
  algorithms changed. The round-3 mechanical-amendment policy remains in force.
- CPU pinned downloads, local-only worker loading, worker volume reload before model open,
  immutable result/status publication and conservative unclosed-reservation accounting remain.
- PRE_REGISTRATION.md changes are confined to its post-§9 round-3 clarification area. The
  shell-extracted text from the §1 heading through the §9 heading has the same before/after
  SHA256: `fa64c71dffccdac69bb247d75b61ef4f2863bc726312350fdddd10eee1af6356`.
  This is a text comparison, not execution of `design_sha()` or a runtime check.
- README now documents attempt records, gate classification, cache order, exact billing,
  storage reconciliation and descriptive null/reason records. No freeze or execution artifacts
  were created by Codex.

## Executor verification still open

Verification: **not run by Codex**. The six findings are addressed in source; this is not a
runtime acceptance or a claim that smoke has passed. The executor owns the following next steps.

- **Only actual smoke/load can establish:** exact pinned snapshot/tokenizer availability;
  package resolution/imports and image construction; authenticated personal workspace; actual
  A10G name/memory and bf16 residency; eager/no-quantization and head/MoE facts; local-only
  cache loading after volume reload; tokenizer manifests for all 400 rows; hook layouts and
  coverage; all-block reconstruction, neutrality, finite features, deterministic repeats,
  memory growth/peak and measured completion projection under the remaining budget.
- **Executor fault-path checks:** repository/revision errors nested through either exception
  link; remote gate envelope and fallback classification; budget/deadline exhaustion on success
  and exception paths; per-entry attempt refusal; pre-reservation extraction failures; failures
  at reload, ledger writes, freeze publication, claim updates and artifact commits; retained
  claims and expired-claim recovery; exact remote-only charging plus 30 seconds.
- **Executor scorer checks without consuming the real single look:** constant columns in
  either/both Spearman inputs yield null plus the exact reason, ordinary rank correlation remains
  finite, subtraction overflow and diagnostic exceptions cannot block JSON/Markdown publication,
  and diagnostic-only changes leave comparator selection and spending decisions unchanged.

Done evidence is an executor-reviewed smoke report plus the relevant failure-path/scorer check
results; full extraction and the real single look retain the README's freeze ordering.
