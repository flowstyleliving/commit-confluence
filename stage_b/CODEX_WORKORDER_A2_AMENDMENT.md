# CODEX WORK ORDER — Amendment A2, cluster-aware inference, portable manifests

**Filed:** 2026-07-14 · **Author:** Claude Code (executor) · **Assignee:** Codex (authoring only)
**Status:** BLOCKING — W1 must land before BENCH Phase 4 resumes.
**Policy:** Codex is write/audit-only. Author the patches and the amendment prose. Run nothing —
no tests, no harness, no Python execution. Mark every verification claim "not run by Codex";
Claude Code executes and verifies.

---

## W1 — Amendment A2: the registered A2 estimator cannot read Phase-4 output (BLOCKING)

### The defect

Amendment A1 (2026-07-12) bumped `SPEC_VERSION` to `bench/1.3` in `stage_b/run_bench.py:31`
and in `stage_b/PRE_REGISTRATION_BENCH.md`. Every strict Phase-4 profile and the strict
`SUMMARY.json` will therefore be stamped `spec_version: "bench/1.3"`.

The registered A2 estimator hard-rejects anything that is not `bench/1.2`:

- `stage_b/analyze_universality.py:176` — `if profile.get("spec_version") != "bench/1.2": failures[slug] = "missing/mismatched bench profile"`
- `stage_b/analyze_universality.py:342` — `if summary.get("spec_version") != "bench/1.2": raise ValueError("registered A2 requires a bench/1.2 strict summary")`

Consequence: **A2 will reject 100% of Phase-4 profiles and then raise on the summary.** The
registered BENCH v1.2 fixed-cell LOMO estimator can never score the run it was registered for.

### The aggravating factor — a false attestation

`stage_b/profiles_bench/EXTENSION_MANIFEST.json`, amendment A1, records:

```json
"unchanged": "calibrator, fusion_signs, builders, gate, analysis, attestations, parity report"
```

`analyze_universality.py` **is** hash-frozen in the same manifest
(`files["stage_b/analyze_universality.py"] = "d85cacf891f1e5c574e1aa53b55e486e284d9db57e8af917d370d934a7f3aef2"`).
So the attestation is *literally* true — nobody edited the file — and *consequentially false*:
leaving the analysis unchanged is exactly what broke it. A freeze record that certifies
"analysis unchanged" while the analysis is silently dead is worse for a reviewer's trust than
no record at all. A2 must correct this line, not quietly supersede it.

Likewise `PRE_REGISTRATION_BENCH.md` §9 A1's "What does NOT change" clause does not name the
analysis path at all. A2's prose must own that omission explicitly.

### Timing legitimacy

Phase 4 is **paused**; no strict cell has run; no registered metric has been computed. We are
inside the legitimate pre-Phase-4 amendment window described at `PRE_REGISTRATION_BENCH.md:520`.
File A2 now, before resuming. Do not backdate; do not fold into A1.

### Required changes

1. **`stage_b/PRE_REGISTRATION_BENCH.md` §9** — append an **A2** entry (append-only; do not edit
   A1's text, which is frozen history). It must state: the defect; that A1's "what does not
   change" clause omitted the analysis path and the manifest's `unchanged` string was
   consequentially misleading; that the fix is spec-restoring, not spec-altering (it changes no
   bar, denominator, endpoint, estimator, cell set, sign convention, or data); the timing
   (pre-Phase-4, no metric computed); and a disclosure rule for the paper.

2. **`stage_b/analyze_universality.py`** — fix the desync *structurally*, not by editing the
   string literal. A second literal is a second thing to forget. Introduce a single source of
   truth for the accepted spec version and have both gate sites read it. Preferred shape:

   ```python
   # bench/1.3 Amendment A2: single source of truth; a future SPEC_VERSION bump must not
   # silently desync the registered A2 estimator from the profiles it is registered to score.
   ACCEPTED_SPEC_VERSION = "bench/1.3"
   ```

   Gate on `ACCEPTED_SPEC_VERSION` at both `:176` and `:342`. Keep the gate **strict equality** —
   do not widen it to a permissive set, do not accept `bench/1.2`, do not `startswith("bench/")`.
   Phase-4 profiles are all `bench/1.3`; a promiscuous gate trades one blocker for a worse one.

   Consider (and state your reasoning either way in the A2 prose) whether to import
   `SPEC_VERSION` from `run_bench` instead of mirroring it, so the two literally cannot desync.
   If import coupling is undesirable at analysis time, mirror it with an explicit assertion and
   say so.

3. **A2 output schema strings** — `"schema_version": "bench-a2/1.2"` at `:200`, `:216`, `:240`
   is the *estimator output* schema, not the spec version; the two are independent. Decide
   whether it bumps, and **disclose the decision in A2** rather than leaving it ambiguous. My
   read: leave it, because the estimator's output contract is genuinely unchanged — but say that
   out loud in the amendment.

4. **`stage_b/profiles_bench/EXTENSION_MANIFEST.json`** — append an A2 amendment record with the
   `old`/`new` sha256 for `stage_b/analyze_universality.py` and `stage_b/PRE_REGISTRATION_BENCH.md`,
   and update `files[]` to the new hashes. **Correct A1's `unchanged` string in place** — it is a
   factual error in a provenance record, and the correction must be visible: retain the original
   wording alongside a correction note (e.g. an `unchanged_correction` field naming A2), so the
   audit trail shows the error and its repair rather than erasing it. Do not silently rewrite it.

   Author the re-stamp as a **script** (`stage_b/restamp_manifest_a2.py`) that recomputes the
   hashes and writes the manifest, so Claude can execute it and the hashes are machine-produced,
   not hand-typed. Hand-typed sha256s in a provenance record are how the next bug gets in.

5. Load-bearing preservation: `confluence_calibrator.py` must remain `c79009a3…` and
   `stage_b/run_bench.py` must remain `70b15c9f…`. A2 touches **neither**. If your patch changes
   either hash, you have exceeded scope — stop and say so.

---

## W2 — Cluster-aware inference (NON-BLOCKING, but ships with the review packet)

### Context — this is narrower than it looks

`PRE_REGISTRATION_BENCH.md:366` **already registers** the stem-cluster bootstrap as the
confirmatory gate for grouped cells, and already declares the sealed row-bootstrap result
historical. So there is **no re-seal here** and `sealed_selector.py` is frozen — **do not touch
it.** What is missing is (a) a reported sensitivity and (b) one genuine code defect.

### W2a — sealed TriviaQA clustered sensitivity (new, additive, non-gating)

The sealed TriviaQA cells resample rows independently (`sealed_selector.py:121`,
`rng.randint(0, n, size=n)`), but TriviaQA is 100 question stems × 2 correlated rows. The sealed
CIs are therefore anti-conservative for that task. An ad-hoc 200-bootstrap check during review
still produced 10/10 deployable, so the result appears robust — but "appears robust under an
unregistered ad-hoc check" is not a sentence that belongs in a review packet.

Author a **new, standalone, additive** script (`stage_b/cluster_sensitivity.py`) that recomputes
stem-clustered OOB CI_lo for all 10 sealed TriviaQA cells from the published matrices, reusing
the sealed selection logic without modifying `sealed_selector.py`. Output a JSON report plus a
markdown table. Stamp it explicitly **descriptive, non-gating, does not alter the sealed 18/20**.

### W2b — E3 label-efficiency splits paired stems (a real defect)

`analyze_universality.py:293` (`label_efficiency`) subsamples individual rows via `pos`/`neg`
index draws. On TriviaQA that **splits the 100 paired stems**, leaking a stem's twin across the
subsample boundary and optimistically biasing the label-efficiency curve — which is the sole
support for the paper's "$\sim$150–200 labels" cost claim (`cc-draft.tex:292-295`).

Make E3 subsampling **stem-aware for grouped tasks**: draw stems, not rows. For ungrouped tasks
(ANLI) stem = row and the procedure must reduce to the current one *identically* — assert this.
E3 is a descriptive, non-gating analysis, so this is a correction, not an endpoint change; it is
in the same A2 amendment's scope but must be **named separately** in the prose so a reviewer can
see it was disclosed, not slipped in.

---

## W3 — BENCH manifests are not portable (blocks "standalone")

Two gates in `stage_b/run_bench.py` validate registered data manifests by **absolute path**, so
they fail on any machine that is not MK's. The repo is therefore not standalone in the sense a
reviewer needs.

**W3a — Arrow source paths.** `run_bench.py:920` requires `source_path.exists()` at a
machine-absolute `/Users/msrk/.cache/...` Arrow path, then hashes it.

**W3b — exclusion references (this is the important one).** `run_bench.py:899-902`:

```python
references = {str(Path(path).resolve()) for path in manifest.get("exclusion_references", [])}
expected_references = {str(path.resolve()) for path in SEALED_REFERENCES}
if references != expected_references:
    raise ValueError("enumerated exclusion-reference union mismatch")
```

This compares **path strings**. Once the two sealed exclusion JSONLs are vendored under
`vendor/t0_core/`, `SEALED_REFERENCES` resolves there while the frozen manifests still record
`/Users/msrk/Documents/t0-morphology-furnace/...` — so the gate raises.

**This is precisely why the uncommitted standalone WIP rewrote all six frozen Phase-2 manifests**
(a 62k-line re-indent plus a two-leaf `exclusion_references` edit). That fix is wrong three times
over: it mutates frozen provenance bytes without an amendment, it gratuitously changes their
sha256 via re-indentation, and the replacement paths are *still absolute*, so it does not even
achieve portability. I have reverted those manifest edits; they are byte-identical to HEAD.

### Constraint

Phase-2 manifests are **frozen** (`PRE_REGISTRATION_BENCH.md:520`). **Do not rewrite them.** Fix
the gate, not the record.

### Required shape

The path was never the gate — the *content* is. Both W3a and W3b must gate on **content hash**:

- W3b: compare the sha256 of each resolved exclusion-reference file against the sha256 of the
  corresponding `SEALED_REFERENCES` entry. Identity of the exclusion set is a fact about bytes,
  not about where the bytes live. Keep the manifest's recorded absolute path as **provenance** —
  it is a true statement about the machine that drew the data — and stop comparing it.
- W3a: resolve the Arrow file through a configurable root (`$CONFLUENCE_HF_CACHE`, then the
  platform HF-cache default, then the recorded path), and verify against `FROZEN_ARROW_HASHES`.
- Both: fail closed with a message naming the env var to set and the sha256 expected.

This changes `run_bench.py`, whose hash is load-bearing (`70b15c9f…` in `EXTENSION_MANIFEST.json`).
So **W3 is itself an amendment (A3)** and must be filed as one, with the manifest re-stamped. It
touches no bar, endpoint, estimator, or datum — only source *resolution*. Say that explicitly.

If you judge that W3 should wait until after Phase 4 rather than re-stamping a load-bearing hash
mid-flight, **say so and argue it** — that is a legitimate call, and I would rather have your
reasoning than your compliance. Note the tension if you defer: without W3 the repo cannot honestly
be called standalone, so the review packet would have to disclose that BENCH reproduction is
MK-machine-only until Phase 4 closes.

---

## Deliverables

1. Patches for W1, W2a, W2b, W3 (or a reasoned refusal on W3's timing).
2. Amendment prose for §9: **A2** (W1 + W2b) and **A3** (W3, if you take it).
3. `stage_b/restamp_manifest_a2.py` — machine-computed re-stamp, executable by Claude.
4. A short note listing what you did **not** verify (i.e. everything requiring execution).

## Non-negotiables

- Do not edit `sealed_selector.py`, `confluence_calibrator.py`, or any frozen Phase-2 manifest.
- Do not edit A1's §9 prose or silently rewrite its manifest record — **correct it visibly**.
- Do not widen a version gate into a permissive matcher.
- Do not backdate an amendment or fold a new one into an existing entry.
- If a fix requires exceeding this scope, stop and file the reasoning instead of proceeding.
