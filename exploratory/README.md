# exploratory/ — unsealed morphology lanes

Live, **unsealed** research lanes that compose over the sealed ACE/T0 core rather than
modifying it. Nothing here is registered, and nothing here gates a published claim.

This area follows the same discipline as `confluence_calibrator.py`: the sealed row
selector and the sealed metric definitions are **imported read-only**; new metric
families are declared locally as additive panels. No lane may edit the 13 sealed root
modules in `t0-morphology-furnace`.

| Lane | Status |
|---|---|
| `attention-kv-tension/` | `[PILOT RUN — NO-PROMOTE]` within/between KV-group attention tension. 5-model ANLI R1 pilot completed 2026-06-08; does not clear its own pre-registered promotion bar. |
| `v-norm-attention/` | `[RESOLVED — NO-PROMOTE]` last-query V-norm cells add nothing beyond routing-only ACE cells across the 18 sealed profiles. |

## Provenance — relocated from `t0-morphology-furnace`, 2026-07-25

Both lanes were developed in `t0-morphology-furnace/exploratory/` in June 2026 and sat
**uncommitted** there. They were relocated here so that `t0-morphology-furnace` holds
the sealed archive and nothing else — its working tree is now pristine against
`t0-ace-sealed-2026-05-26`, which is what makes the packaging tag `t0-pkg-v0.1.0`
trustworthy.

The relocation carried the artifacts verbatim (byte-identical copies, verified by
`diff -r`). One thing did **not** come across as live code: the KV-tension metric
implementation was written as an in-place edit to sealed files
(`pri_calibrator.py`, `scripts/diagnose_inter_head_disagreement.py`,
`tests/test_attention_cells.py`). That diff is preserved as provenance at
`attention-kv-tension/t0-patch/kv-tension-against-t0-7c2fcb7.patch` and is
**not applied anywhere**. Re-homing it as a proper additive overlay is an open build
task — see that lane's README.
