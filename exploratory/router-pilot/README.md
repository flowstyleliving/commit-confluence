# Router pilot executor handoff

Codex is write/audit-only. All execution verification below is **not run by Codex**.
The design check is recorded in `AUDIT_R2.md`; round-3 findings, fixes and static review are in
`AUDIT_R3.md`. No runtime, test, import, build or smoke was run by Codex.

Run from `exploratory/router-pilot/` using the executor's authenticated **personal Modal
workspace** and its `main` environment. The independent app is `router-pilot`; it creates
`router-pilot-cache` and `router-pilot-budget-claim` if absent. Do not point this lane at an
existing depth-lane volume. No Hugging Face secret is configured; downloads pass `token=False`.
Personal profile selection is an executor prerequisite, not something Codex verified.

Executor-only environment preparation — **not run by Codex**:

```sh
python3.12 -m venv .executor
.executor/bin/python -m pip install modal==1.5.1 numpy==2.2.6 scipy==1.15.3
.executor/bin/modal profile list
```

Select the existing personal profile in the executor's Modal configuration before continuing.
The GPU image uses Python 3.12, torch 2.8.0+cu126 from the PyTorch CUDA-12.6 index,
transformers 4.57.6, huggingface_hub 0.36.0, accelerate 1.11.0, safetensors 0.6.2,
numpy 2.2.6, tokenizers 0.22.1, pandas 2.3.3, scikit-learn 1.7.2, and scipy 1.15.3.
These pins satisfy the inspected transformers dependency ranges; resolution/load is
**not run by Codex**. The image ID, frozen recipe SHA256, package versions and dtype histogram
are recorded by smoke. The recipe hash is not an OCI digest.

1. FREEZE-1 — **not run by Codex**:

   ```sh
   .executor/bin/python make_freeze.py --stage 1
   ```

2. Smoke — **not run by Codex**:

   ```sh
   .executor/bin/modal run --env main modal_router_pilot.py::smoke
   ```

   The CPU coordinator commits `smoke.attempt.<entry>.json` before preflight, downloads the
   pinned snapshot first, and loads the tokenizer locally (`local_files_only=True`). It then
   hashes the data, decodes only prompts/stem metadata, validates all
   400 serialized prompts, and freezes YES/NO token sets before dispatching an A10G.
   Smoke downloads both prompt manifests and `smoke.report.json` automatically, or available
   attempt/failure records and terminal statuses when it fails. It publishes
   gates, timing and memory diagnostics; it does not publish feature matrices or read labels.

3. Inspect smoke report — **not run by Codex**:

   ```sh
   cat smoke.report.json
   ```

   Continue only if `passed` is true. Inspect all-block cosine/relative-L2 reconstruction,
   neutrality, finite/hook checks, repeat determinism, memory and timing. The script gates
   these numerically; inspection does not authorize a threshold change. Timing includes
   feature capture/scoring and conservatively budgets both extraction reloads.

4. FREEZE-2 — **not run by Codex**:

   ```sh
   .executor/bin/python make_freeze.py --stage 2
   ```

   This appends a stage-2 entry with both manifest hashes, preserving stage 1. It refuses
   source changes, a failed smoke, mismatched manifests, or a repeated stage.

5. Extract ANLI R1 — **not run by Codex**:

   ```sh
   .executor/bin/modal run --env main modal_router_pilot.py::extract --task anli_r1
   ```

6. Extract HaluEval QA — **not run by Codex**:

   ```sh
   .executor/bin/modal run --env main modal_router_pilot.py::extract --task halueval_qa
   ```

7. Download — **not run by Codex**:

   ```sh
   .executor/bin/modal run --env main modal_router_pilot.py::download
   ```

   Downloads the immutable NPZ/status pairs, manifests, report, freeze snapshots and ledger
   events into this directory. Existing identical files are retained; differing bytes refuse.
   No GPU function is invoked by this helper.

8. Score once, after both terminal statuses exist — **not run by Codex**:

   ```sh
   .executor/bin/python score_router_pilot.py
   ```

   Writes `ROUTER_PILOT_RESULTS.json` and `.md`. A persistent `SCORE_STARTED.json` claim
   precedes label/outcome access. Neither file, nor that claim, may be deleted to rerun.
   Both payloads are completely computed before atomic publication. After an interrupted
   two-file publication, inspect `.results-complete.json` / `.results-complete.md` and any
   already published output; do not recompute the analysis.

   Per-block Spearman reports `{ "value": null, "reason": "undefined: constant input" }`
   if either ranked column is constant. Other diagnostic exceptions/non-finite values also
   report null with a reason; they do not block decision publication. The six captured ATT
   metrics, deprecated `final_v_norm_bos`, descriptive `sink_draw`/`sink_best`, comparator set
   and spending rule are unchanged.

## Abort and budget handling

**Launch each extraction exactly once.** After FREEZE-2 every coordinator failure is terminal,
including a refused duplicate launch: re-running `extract` for a task that already has a status
writes NOT-RUN for any cell that is not yet terminal.


The hard cap is 7,200 cumulative GPU seconds, including a 600-second shutdown reserve.
Each launch reserves its full remaining allowance, uses a matching Modal timeout and an
absolute in-process deadline, and records start/end/GPU type. Normal completion is charged
launch-to-return time plus 30 seconds, success or failure. Only an unclosed reservation
(killed coordinator) is charged in full, at reconciliation. The persistent atomic claim prevents overlapping launches
across independent `modal run` processes. Ledger events are append-only JSON files on the volume.
The clock starts immediately before `.remote()` and stops immediately on return/exception;
publication, volume commits and failure handling are outside that interval (S2). Deadline and
post-call budget exhaustion are terminal on both paths, before any successful result is published.

A measured-gate failure (`GateFailure`), missing pinned revision, or cap writes `NOT-RUN` for every
non-terminal cell. Completed `ok` cells remain immutable. If smoke fails on a measured gate, skip
FREEZE-2/extraction: download and score the two NOT-RUN statuses with the existing stage-1 freeze.
If smoke fails mechanically, it writes `smoke.failed.<entry>.json` and no status; see the amendment
path below. If extraction fails, skip
remaining GPU commands and download/score. A partial NPZ under NOT-RUN is never scored.
Registered hardware facts require an A10G name and the recorded 24 GB memory range; eager/no
quantization, architecture/head/MoE dimensions, dtype/environment facts and zero reconstruction
reference use terminal gates. Instrumentation shape/programming errors and imports remain
mechanical during smoke. Worker gate failures use a structured envelope; the coordinator also
recognizes `GATE:` anywhere in a wrapper chain and wrapped Hub repository/revision-not-found errors.

After a killed coordinator or retained claim following storage failure, wait until its persisted claim expires and invoke the same
entry point once to reconcile the reservation and write NOT-RUN statuses; it launches no
GPU work. A live claim only refuses competing launches. Do not erase claims or ledger events.
This recovery is executor-triggered; no background monitor has been scheduled.
A reconciliation storage failure leaves its own claim in place for manual inspection; do
not treat missing terminal files as successful completion or start more GPU work.
All post-acquisition work uses the same failure publisher, including pre-reservation extraction
failures. Release requires durable accounting and result/failure state. An observed GPU end is
recorded before publishing artifacts; an unclosed reservation remains fully charged. The claim
stores its original freeze/kind for reconciliation. If reconciliation itself fails, restore storage
and have the executor inspect the retained active/reconciliation claims and durable ledger before
completing recovery; do not clear them to permit another run. Coordinator-owned NPZ/status
publication and the worker's reload-before-model-open discipline remain in place.

A mechanical (non-gate) smoke failure can use the amendment path; a measured-gate failure cannot.
Record a dated `YYYY-MM-DD-tag` amendment section in `PRE_REGISTRATION.md` (sections 1–8 must stay
byte-identical), fix the code, then, before successful smoke or any terminal cell, run — **not run by
Codex**:

```sh
.executor/bin/python make_freeze.py --stage 1 --amendment YYYY-MM-DD-tag
```

Then re-run smoke (one attempt per freeze entry). An attempt marker alone refuses a second smoke;
an unresolved claim/storage failure must be reconciled before considering an amendment.
No gate-failed cell can be rescued. A changed source file without a new eligible freeze
entry refuses. Freeze hashes also cover the unchanged sealed source and imported depth-lane
helpers, including their local import dependencies.

## Remaining executor assumptions

All verification: **not run by Codex**. In particular: package-wheel availability and imports;
ungated availability of the exact model/tokenizer revision; personal workspace authentication;
Modal image construction, volume atomic rename/fsync and cross-launch claim behavior; bf16
numerical gate tolerances; A10G residency/peak memory; and measured completion within two hours.
The implementation refuses or terminalizes failures instead of changing those requirements.
Smoke must establish actual A10G name/memory, eager/no-quantization configuration, head/MoE
dimensions, local-cache model/tokenizer loading, hook coverage/layout, reconstruction/neutrality,
finite features, deterministic repeats, memory growth and measured timing. Executor fault-path
checks are also still needed for wrapped Hub errors, remote gate envelopes/fallbacks, both budget
exhaustion paths, duplicate attempts, storage failures/reconciliation, and constant-input diagnostic
publication. None of these execution checks was run by Codex.
