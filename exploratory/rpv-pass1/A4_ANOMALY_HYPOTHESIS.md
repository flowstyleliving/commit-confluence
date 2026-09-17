# A4 anomaly — pre-stated hypothesis

**Written 2026-09-17, BEFORE looking at any supporting data.** One hypothesis, one test, stop either way.
No second hypothesis may be substituted after seeing the result; a miss is reported as a miss.

## The observation to be explained

4 of 26 panel cells show inline-vs-offline `T=1.0` discrepancies **100–500× the panel norm**:
Phi-4-mini (both tasks, `neg_shadow_logvol_r1`) and gemma-3-1b (both tasks, `spectral_entropy`).
The other 22 cells sit at ~1e-07. Three earlier hypotheses were falsified: small IQR, vocabulary size,
and peakedness/ill-conditioning.

## H: float32 underflow in the readout softmax tail

The two paths differ in **where the softmax is normalized**:

- **Inline** readout uses `trace["prefix_probs"][-1]`, produced by `pri_runtime.safe_softmax`, then takes
  `p[idx]` and lets `fc_full_spectrum` renormalize over the support.
- **Offline** recomputes `softmax_np(z_support / T)` in float64 directly from the banked float32 logits.

Mathematically identical. Numerically not: if `safe_softmax` carries float32, then a support token whose
probability falls below the float32 normal floor (~1.18e-38) is flushed toward zero in the inline path while
the offline path, working in float64 from logits, retains it as a small positive. Zeros versus tiny positives
change `diag(p) − ppᵀ` and therefore the spectrum — and they change the *small* eigenvalues most, which is
exactly what `spectral_entropy` and `neg_shadow_logvol_r1` (a rank-one-removed log-volume) are sensitive to.

**Why the earlier three hypotheses missed it:** the driver is neither vocabulary size nor peakedness alone,
but the **within-support logit span** — how far the 512th-ranked token sits below the top one. A large vocab
makes a wide span *available*; a peaked distribution makes it *likely*; but only the span itself determines
whether `exp()` underflows.

## The pre-stated, falsifiable prediction

Define, per row, the **within-support logit span** `S = max(z_support) − min(z_support)` at the readout.

1. **Primary:** cells failing A4 have a materially larger median `S` than cells passing it, and the panel-wide
   rank correlation between a cell's median `S` and its max A4 delta is **positive and strong (Spearman ρ ≥ 0.6)**.
2. **Mechanistic threshold:** the failing cells' `S` reaches or exceeds **≈87.3** (`−ln` of the float32 minimum
   normal, 1.18e-38), the point at which `exp(z − z_max)` underflows in float32, while passing cells largely
   do not.
3. **Direct confirmation:** in failing cells, the inline readout probability vector contains **exact zeros**
   within the 512-token support; in passing cells it does not.

## What falsifies it

- ρ < 0.6, **or** failing cells' median `S` comfortably below ~87, **or** no exact zeros in the inline support
  of failing cells.
- Any of these means H is wrong. **Report it as wrong and stop** — do not substitute a fresh hypothesis and
  present it as if pre-stated.

## Scope

Diagnostic only. The banked columns (indices, logits, per-temperature logsumexp) are identical under either
path, so nothing here changes the pass-one measurement, and nothing here bears on H1.

---

# RESULT, 2026-09-17 — H is FALSIFIED on all three limbs. Stopping as pre-committed.

| Limb | Bar | Measured | Verdict |
|---|---|---|---|
| 1. Rank correlation | Spearman ρ ≥ 0.60 | **ρ = −0.142** | ❌ fails, and the sign is *negative* |
| 2. Underflow threshold | failing cells reach S ≈ 87.3 | failing median **18.87**, panel max **28.80** | ❌ nowhere near; no float32 underflow is possible |
| 3. Exact zeros in support | present in failing cells | not reached — limb 2 rules the mechanism out a priori | ❌ moot |

Failing cells' median within-support logit span is **18.87** against **17.25** for passing cells — effectively
the same. The strongest single counterexample is **Phi-4-mini**, which has among the *narrowest* spans in the
panel (14.24 / 14.38) and yet fails, while **Qwen3-1.7B** has the second-widest span (25.41) and passes cleanly
at 2.83e-07. At the observed maximum span of 28.80, `exp(−28.80) ≈ 3e-13` — some twenty-five orders above the
float32 normal floor. The proposed mechanism cannot operate at these magnitudes.

**Conclusion: the A4 anomaly is NOT float32 underflow in the readout softmax tail.** That is now the fourth
falsified explanation, alongside small IQR, vocabulary size, and peakedness.

**Stopping here, as pre-committed.** The hypothesis file fixed one test and forbade substituting another after
seeing the result. Continuing to hunt would convert a clean negative into post-hoc fishing — the precise trap
the project's standing framing rules name three times. The anomaly stays `[OPEN]` and **unexplained**, with its
scope unchanged: A4 compares an inline convenience copy against the offline replay path, the banked columns are
identical either way, and the pass-one measurement is unaffected.

Any future attempt should start from a *newly* pre-stated hypothesis in a new file, not from this one.
