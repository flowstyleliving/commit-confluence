# Codex handoff — A2/A3 amendment authoring

**Authored:** 2026-07-14  
**Execution status:** not run by Codex

Codex authored the A2 version-gate repair, stem-aware E3 sampling, the additive sealed
TriviaQA cluster-sensitivity reporter, A3 content-addressed manifest resolution, amendment
prose, and the machine-produced manifest re-stamp script.

The manifest has intentionally **not** been re-stamped yet. The executor must review and run:

```bash
python stage_b/restamp_manifest_a2.py --executor "Claude Code / MK"
```

That script does not rewrite `SMOKE_SUMMARY.json`. Once A3 changes the extension-manifest
sha256, Phase 4 must remain blocked until the harness-required Phase-3 provenance re-audit is
completed and documented by the executor.

## Not verified by Codex

- Python parsing, imports, compilation, or unit tests
- shell syntax or launcher execution
- any data-manifest validation or Hugging Face cache resolution
- the A2 estimator against a strict Phase-4 summary (none was available for authoring)
- E3 numerical outputs or the unique-row identity assertion
- clustered sensitivity JSON/Markdown generation or its numerical result
- sha256 values produced by `restamp_manifest_a2.py`
- frozen-file hash preservation, including calibrator and Phase-2 manifests
- Phase-3 smoke provenance re-audit, model forwards, or Phase-4 resume

The pre-existing untracked `profiles_bench/GATE_FAILURES.json`, `SUMMARY.json`, and
`strict_phase4.log` were not edited or treated as registered results.
