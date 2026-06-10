# Stage A - Union Coverage Matrix

_seed 20260610 | 2000x bootstrap | deployable := CI_lo > 0.5 | RPV primary = fisher_eff_rank_

- usable cells (model x task): **26**  (+2 excluded: no usable rows -> gpt-oss-20b-MXFP4-Q4/anli_r1 (no rows), gpt-oss-20b-MXFP4-Q4/triviaqa_paired (no rows))
- covered by >=1 family (ACE/null_ratio/RPV/surprise): **23/26**  -> gap-free(any) = **False**
- covered by >=1 *geometric* family (ACE/null_ratio/RPV; surprise excluded): **22/26**  -> gap-free(geom) = **False**
- **ACE sealed cohort** (9 models present in ACE panel x 2 tasks = 18 cells): covered(any) **18/18**, covered(geom) **17/18**
- MEASUREMENT-ORPHANS (even surprise at chance -> degenerate commit / gate artifact, not a detector gap): DeepSeek-R1-Distill-Qwen-7B-4bit/triviaqa_paired, dolphin-2.9.3-mistral-nemo-12b-4bit/anli_r1, gemma-3-1b-it-4bit/anli_r1
- DETECTOR-GAPS (surprise works but no geometric family does -> a real panel hole): gemma-3-4b-it-4bit/anli_r1

Cell format: `AUROC [CI_lo,CI_hi] Y/n`. ACE column: `OOB_median [OOB ci_lo] Y/n`. `-` = model not in ACE panel.

| model | task | ACE (attn) | null_ratio (resid) | RPV (readout) | surprise (base) | verdict |
|---|---|---|---|---|---|:---:|
| DeepSeek-R1-Distill-Qwen-7B-4bit | anli | - | 0.617 [0.54,0.69] **Y** | 0.530 [0.44,0.61] n | 0.603 [0.52,0.68] **Y** | OK |
| DeepSeek-R1-Distill-Qwen-7B-4bit | triviaqa | - | 0.529 [0.42,0.64] n | 0.574 [0.46,0.68] n | 0.529 [0.42,0.64] n | **ORPHAN** |
| Llama-3.1-8B-Instruct-4bit | anli | - | 0.508 [0.43,0.59] n | 0.674 [0.60,0.74] **Y** | 0.689 [0.61,0.76] **Y** | OK |
| Llama-3.1-8B-Instruct-4bit | triviaqa | - | 0.584 [0.47,0.69] n | 0.876 [0.80,0.94] **Y** | 0.749 [0.65,0.84] **Y** | OK |
| Llama-3.2-3B-Instruct-4bit | anli | 0.597 [lo 0.40] n | 0.630 [0.56,0.71] **Y** | 0.697 [0.62,0.77] **Y** | 0.569 [0.49,0.65] n | OK |
| Llama-3.2-3B-Instruct-4bit | triviaqa | 0.830 [lo 0.65] **Y** | 0.712 [0.60,0.81] **Y** | 0.702 [0.59,0.80] **Y** | 0.514 [0.40,0.63] n | OK |
| Mistral-7B-Instruct-v0.3-4bit | anli | 0.784 [lo 0.65] **Y** | 0.779 [0.71,0.84] **Y** | 0.770 [0.70,0.84] **Y** | 0.525 [0.45,0.60] n | OK |
| Mistral-7B-Instruct-v0.3-4bit | triviaqa | 0.989 [lo 0.97] **Y** | 0.874 [0.78,0.96] **Y** | 0.876 [0.78,0.96] **Y** | 0.786 [0.69,0.87] **Y** | OK |
| Mistral-Nemo-Instruct-2407-4bit | anli | 0.887 [lo 0.82] **Y** | 0.626 [0.55,0.70] **Y** | 0.773 [0.70,0.84] **Y** | 0.685 [0.60,0.76] **Y** | OK |
| Mistral-Nemo-Instruct-2407-4bit | triviaqa | 0.980 [lo 0.93] **Y** | 0.833 [0.74,0.91] **Y** | 0.811 [0.72,0.90] **Y** | 0.664 [0.55,0.77] **Y** | OK |
| Phi-3.5-mini-instruct-4bit | anli | 0.740 [lo 0.60] **Y** | 0.820 [0.76,0.88] **Y** | 0.741 [0.67,0.81] **Y** | 0.663 [0.59,0.74] **Y** | OK |
| Phi-3.5-mini-instruct-4bit | triviaqa | 0.843 [lo 0.71] **Y** | 0.842 [0.75,0.92] **Y** | 0.551 [0.43,0.67] n | 0.537 [0.42,0.65] n | OK |
| Phi-4-mini-instruct-4bit | anli | 0.690 [lo 0.55] **Y** | 0.707 [0.63,0.78] **Y** | 0.669 [0.59,0.74] **Y** | 0.733 [0.66,0.80] **Y** | OK |
| Phi-4-mini-instruct-4bit | triviaqa | 0.933 [lo 0.85] **Y** | 0.721 [0.62,0.81] **Y** | 0.553 [0.43,0.66] n | 0.792 [0.68,0.89] **Y** | OK |
| Qwen2.5-7B-Instruct-4bit | anli | 0.780 [lo 0.66] **Y** | 0.835 [0.78,0.89] **Y** | 0.679 [0.60,0.75] **Y** | 0.586 [0.51,0.66] **Y** | OK |
| Qwen2.5-7B-Instruct-4bit | triviaqa | 0.924 [lo 0.81] **Y** | 0.658 [0.55,0.76] **Y** | 0.896 [0.82,0.95] **Y** | 0.783 [0.68,0.87] **Y** | OK |
| Qwen3-1.7B-4bit | anli | 0.641 [lo 0.50] **Y** | 0.563 [0.48,0.64] n | 0.522 [0.44,0.60] n | 0.733 [0.66,0.80] **Y** | OK |
| Qwen3-1.7B-4bit | triviaqa | 0.680 [lo 0.47] n | 0.621 [0.51,0.73] **Y** | 0.759 [0.65,0.86] **Y** | 0.851 [0.77,0.92] **Y** | OK |
| Qwen3-8B-4bit | anli | 0.823 [lo 0.74] **Y** | 0.517 [0.44,0.60] n | 0.851 [0.79,0.90] **Y** | 0.744 [0.67,0.81] **Y** | OK |
| Qwen3-8B-4bit | triviaqa | 0.778 [lo 0.64] **Y** | 0.811 [0.72,0.90] **Y** | 0.885 [0.81,0.95] **Y** | 0.766 [0.66,0.86] **Y** | OK |
| dolphin-2.9.3-mistral-nemo-12b-4bit | anli | - | 0.519 [0.44,0.60] n | 0.559 [0.48,0.64] n | 0.534 [0.45,0.62] n | **ORPHAN** |
| dolphin-2.9.3-mistral-nemo-12b-4bit | triviaqa | - | 0.633 [0.52,0.74] **Y** | 0.638 [0.53,0.75] **Y** | 0.551 [0.43,0.67] n | OK |
| gemma-3-1b-it-4bit | anli | - | 0.528 [0.44,0.61] n | 0.554 [0.48,0.63] n | 0.538 [0.46,0.62] n | **ORPHAN** |
| gemma-3-1b-it-4bit | triviaqa | - | 0.657 [0.55,0.76] **Y** | 0.528 [0.41,0.65] n | 0.549 [0.44,0.66] n | OK |
| gemma-3-4b-it-4bit | anli | 0.656 [lo 0.49] n | 0.516 [0.43,0.59] n | 0.558 [0.48,0.64] n | 0.621 [0.55,0.70] **Y** | surp-only |
| gemma-3-4b-it-4bit | triviaqa | 0.861 [lo 0.72] **Y** | 0.845 [0.77,0.92] **Y** | 0.803 [0.71,0.89] **Y** | 0.758 [0.66,0.85] **Y** | OK |
| gpt-oss-20b-MXFP4-Q4 | anli | - | - | - | - | _excluded_ |
| gpt-oss-20b-MXFP4-Q4 | triviaqa | - | - | - | - | _excluded_ |

## Per-family deployable counts (usable cells)

| family | deployable cells | of |
|---|:---:|:---:|
| ACE | 15 | 18 (present) |
| null_ratio | 18 | 26 |
| RPV | 17 | 26 |
| surprise | 17 | 26 |

## Verdict

**Stage A INCOMPLETE** - 1 genuine detector-gap(s) where surprise works but no geometric family does: gemma-3-4b-it-4bit/anli_r1. Plus 3 measurement-orphan(s). Report before Stage B.
