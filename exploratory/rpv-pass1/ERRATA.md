# Errata and deviations — RPV pass-one lane

Corrections to statements published in this directory, and disclosure of one deviation from the frozen
pre-registration. Raised by an independent adversarial audit (Codex `gpt-6-astra`, 2026-09-21) and verified
against the artifacts before being recorded here.

**Why the source files are not edited in place.** `panel/PANEL_MANIFEST.json` records the SHA-256 of the
scripts that produced the panel, and `rpv_pass1_run.py`, `rpv_pass1_temperature.py` and `run_panel.py` still
match it exactly. Editing them — even to fix a comment — would break the chain tying the code to the banked
results. The frozen pre-registration is likewise left byte-for-byte intact. Corrections therefore live here.
Any corrected implementation will be published as a **new, separately versioned** script, not as an edit.

---

## E1 — The claim that the overlay matches the reference's readout normalization is FALSE

**Where:** `rpv_pass1_run.py:29–34` and `PRE_REGISTRATION_RPV_PASS1.md:60–63`.

**What those files say (incorrect):**

> "an earlier comment here claimed this renormalization matched the reference. It does NOT" …
> "passes `p_t` straight into `_support_spectrum` with no renormalization" …
> "is removed so the inline path treats `p` exactly as the reference does."

And in the pre-registration: *"An earlier draft claimed the reference applies an additional normalization to
its readout. It does not; corrected by steward audit 2026-09-16."*

**The truth.** The sealed reference **does** normalize the readout probability vector. At
`vendor/t0_core/exploratory/shadow-ambiguity/comprehensive_run.py:362`:

```python
p_t = p_t / (float(np.sum(p_t)) + 1e-300)
```

executed before `_support_spectrum` at line 383. The original authoring comment was **correct**; the
"steward audit" correction that removed the step was **wrong**, and the attribution of that correction in the
pre-registration is therefore also wrong.

**Consequence.** This overlay's inline path **omits** a preliminary normalization that the reference performs.
The step cancels in exact arithmetic, because `fc_full_spectrum` renormalizes over the support
(`comprehensive_run.py:162–164`), so no *mathematical* difference follows — but it changes floating-point
rounding, and the inline-versus-offline discrepancies reported by this lane are numerical quantities.

**What is NOT affected.** The banked columns — `support_idx`, `support_logits`, `full_vocab_lse` — are derived
from logits, and top-k selection is scale-invariant, so they are unchanged. The A1′ pass-two equivalence check
compares the reference's own rows and is unchanged.

**Correct description of this implementation:** it omits the reference's preliminary readout normalization.
It does not match the reference on that step, and no claim of numerical equivalence to the reference's readout
path should be made on its behalf.

---

## E2 — Deviation: the A4 acceptance rule was replaced after observing failures

**Disclosed deviation from the frozen pre-registration**, which remains authoritative as written.

`PRE_REGISTRATION_RPV_PASS1.md:192–206` specifies an A4 tolerance of one tenth of the minimum positive
between-row separation, calibrated on a designated smoke sample, with fixed tolerances carried into the full
run, acceptance marked pending until justified — and states: *"Never enlarge a tolerance to make failures pass."*

The implementation does not do that. `rpv_pass1_temperature.py:138–152` computes `max_abs_delta / IQR` per
cell, source and statistic, and emits a pass/fail against `ACCEPT_RATIO = 1e-4`.

**The sequence, stated plainly.** After observing failures under the minimum-gap procedure at n=200, that
procedure was replaced with the IQR-denominated rule. The stated reason — that a minimum-adjacent-gap
denominator shrinks as rows are added, so the bar tightens with sample size for reasons unrelated to
correctness — is a real structural objection, and the measured margins were 2650× at n=5, 5.7× at n=50, and
failing at n=200. **That reasoning does not change the sequence: the rule changed after the failures were
seen.** A reviewer is entitled to treat the replacement as a post-hoc procedure.

**Explicitly:**

- Passing the replacement rule is **not** passing the original rule.
- The original failures remain part of the record and are not withdrawn.
- `1e-4` is an engineering tolerance. It is **not** derived from the sensitivity of the registered endpoint,
  and no such derivation is offered here.
- The replacement does not preserve the original rule's guarantee against reversing the ordering of close
  row values.
- The replacement does not make everything pass: **four cells still contain failing keys** (see E6).
- The code docstring at `rpv_pass1_temperature.py:124–133` already records the change; this entry supplies the
  reconciliation with the frozen document that the docstring does not.

---

## E3 — "A3 satisfied" overstates literal compliance

`PRE_REGISTRATION_RPV_PASS1.md:222` and `PREREG_FREEZE.txt:5` declare A3 satisfied. A3 requires freezing
**before the full run** (`:179–180`), and the document itself discloses that capture occurred 2026-09-16 while
the freeze is dated 2026-09-17 (`:241–247`). The chronology is disclosed, but the compliance claim is stronger
than the facts support.

Relatedly, `panel/PANEL_MANIFEST.json:20` records pre-registration hash `2d636748…` while
`PREREG_FREEZE.txt:3` records `53501734…`. These are **different document versions**: the manifest hashed the
pre-freeze draft. No single document was frozen before capture.

Also inaccurate: `:46–47` states the manifest records row limits and seeds. Manifest cells carry `banked_rows`;
the driver passes `--limit 0` and the capture default seed (`run_panel.py:87–90`).

---

## E4 — "Falsified on all three limbs" overstates the test

`A4_ANOMALY_HYPOTHESIS.md:58` claims falsification on all three limbs. Its own table at `:64` records limb 3
("exact zeros in the inline support") as **not reached**. Two limbs produced failed predictions; the third was
not tested. The hypothesis' own rule (`:44–49`) makes any single failed limb sufficient, so the **rejection
stands**; the count does not.

Two further imprecisions in that document: 87.3 is the float32 **normal/subnormal boundary**, not the point
where exponentials become zero — subnormals are conflated with underflow to zero (`:21–23, 38–40`); and the
claim that "only the span itself" determines underflow neglects normalization (`:27–30`). Neither rescues the
hypothesis at the observed spans.

Finally, `:78–79` says "the pass-one measurement is unaffected." Scope that: the **banked columns** are
unaffected. That is not a demonstration that every downstream numerical result or conclusion is unaffected.

---

## E5 — "Never reads the label" is literally false

`PRE_REGISTRATION_RPV_PASS1.md` states the capture never reads the label. `rpv_pass1_run.py:208` reads
`inherited["label"]` to carry it into the output row, and the inherited loader validates and hashes labels.

**Accurate statement:** the capture performs **no label-dependent selection, fitting or performance analysis**.
The label is carried through as data. That is the property the blindness argument actually needs; the stronger
wording is wrong.

Relatedly, `:248–252` asserts the roster "could not have been outcome-selected." The manifest was frozen before
the first cell ran, which is genuine evidence — but what a steward computed outside this code cannot be
established by the code. Treat the blindness claim as an **attestation**, not a property enforced by the
implementation.

---

## E6 — The four A4-failing cells, stated precisely

Seven failing keys across four cells:

| Cell | Failing keys |
|---|---|
| Phi-4-mini / ANLI | `readout/neg_shadow_logvol_r1`, `block_31/neg_shadow_logvol_r1` |
| Phi-4-mini / TriviaQA | `block_31/neg_shadow_logvol_r1` |
| gemma-3-1b / ANLI | `block_25/spectral_entropy`, `aggregate/spectral_entropy` |
| gemma-3-1b / TriviaQA | `readout/spectral_entropy`, `aggregate/spectral_entropy` |

**All seven are secondary statistics.** None involves `fisher_eff_rank`, the registered primary.

The cause is **unexplained**. Five candidate explanations have been tested and falsified: small IQR,
vocabulary size, peakedness/ill-conditioning, and float32 underflow in the support tail (the last pre-stated
and recorded in `A4_ANOMALY_HYPOTHESIS.md`). Investigation was stopped deliberately rather than continued as
post-hoc variable search.

**What must not be claimed about these cells:**

- That the omitted readout normalization (E1) caused them. Four of the seven keys are **block** sources, which
  that normalization does not touch.
- That restoring the normalization would fix them, or that it could not affect them, absent evidence.
- That they demonstrate a failed fresh offline replay. That is a separate comparison
  (`rpv_pass1_temperature.py:188–193`) which the panel driver does not perform.
- That "unexplained" implies "harmless to every downstream inference."

---

## E7 — "26/26 OK" means processes completed, not A4 accepted

`run_panel.py:95–104` assigns `status="ok"` when capture completes, independently of A4 outcomes; it counts A4
failures separately. `panel/panel.log` and the lane's commit message summarise "26/26 cells OK". That is
correct for **process completion and zero pass-one failures**, and must not be read as "26/26 passed A4" — four
cells did not (E6).

A related implementation gap: a zero IQR yields `passed=None` (`rpv_pass1_temperature.py:143–149`) and the
driver counts only literal `False` (`run_panel.py:99`), so a degenerate source could be absent from failure
counts rather than flagged.

---

## E8 — Minor corrections

- `rpv_pass1_temperature.py:25–27` says measured ratios of ~1e-7 are "four orders below" `1e-4`. It is **three**.
- The frozen grid `{0.5, 0.7, 1.0, 1.4, 2.0}` is described as log-symmetric about 1.0. It is **approximately**
  so: `0.7 × 1.4 = 0.98`.
- The T=0 exclusion argument should be stated as a limit: softmax at T=0 is **undefined** (division by zero),
  and the one-hot limit as T→0⁺ holds only for a **unique** maximum. With tied maxima the Fisher geometry does
  not collapse to zero. The exclusion of T=0 from the grid stands.

---

## Standing scope

Nothing in this lane revises **H1**, which was registered on pass-two features. The two positions are reported
side by side and are never pooled. No label-linked analysis has been published from this lane.

*Recorded 2026-09-21.*
