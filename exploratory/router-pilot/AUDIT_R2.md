# Round-2 design audit

Date: 2026-10-08. Static read/audit only. Verification: **not run by Codex**.

The revised final-prompt-state estimand, descriptive compliance, MoE reconstruction gate,
bootstrap strata, and diagnostic-only v_norm_bos resolve the stated round-1 concerns.

Remaining MAJOR findings, corrected in PRE_REGISTRATION.md before implementation:

1. **Undeclared analysis choices.** The fold seed and tie order were placeholders. Declared
   task-specific RNG tags, metric-first/block-second ties, comparator ties, and map provenance.
2. **Incomplete lifecycle contract.** Stage-2 freeze extension conflicted with no-overwrite;
   repair/re-smoke conflicted with immutable aborts; killed workers and overlapping launches
   could escape budget accounting. Specified append-only freeze entries, pre-GPU-only mechanical
   amendment eligibility, persistent cross-launch claim, pessimistic reservation reconciliation,
   terminal states, single-look claim, and incomplete-pilot STOP. Pair-label checks belong only
   to the scorer; smoke checks group multiplicities without parsing label fields.

Implementation clarification: the installed public Modal SDK exposes an image ID, not an OCI
digest. The document now explicitly requests that ID plus frozen image-recipe SHA256. This is
not represented as an OCI digest. Exact package pins are in the extractor and frozen with it.

No remaining conceptual MAJOR identified by static reading. This is not runtime acceptance.
Import/load, gates, GPU fit/timing, package resolution, remote storage and numerical execution:
**not run by Codex**. Executor smoke must resolve these assumptions without weakening gates.

Implementation handoff (static source review; execution verification **not run by Codex**):

- Extractor imports the sealed dispatcher and its unmodified v-norm implementations.
  Hook failures propagate; reconstruction hooks are removed before manual module calls.
  MoE reconstruction mirrors the inspected 4.57.6 bf16 cast and expert-index accumulation.
- CPU prompt parsing does not decode label fields. All 400 prompts are checked before
  GPU dispatch; stage-2 manifests are re-derived and checked locally, on the volume and
  again against the shipped worker arguments.
- Persistent cross-launch claims, full failed-call reservations, immutable statuses and
  coordinator-owned NPZ publication prevent late worker writes and ordinary retry rescues.
  After an abruptly killed coordinator, terminalization requires the documented executor
  reconciliation call after claim expiry; there is no scheduled background process.
- Scorer imports the grouped helpers and rank/AUROC primitives, uses metric-major family
  layouts (80/48/3/1), refits signs/selections inside shared bootstrap/null draws, and records
  frozen maps, curves, selections, diagnostic-only channels and incomplete-pilot STOP.
- README contains the ordered executor commands. No freeze, smoke, extraction, installation,
  import check, syntax execution, test, build, or scoring command was run by Codex.
