# Vendored sealed modules (verbatim copies — DO NOT EDIT; edit upstream then re-vendor)

Imported by `cloud/modal_app.py` so the Modal (torch) extractor reuses the sealed kernels unchanged.
MLX is stubbed at import time (see `_import_seal`); only the model forward is reimplemented in torch.

| file | source repo |
|---|---|
| pri_calibrator.py | t0-morphology-furnace |
| pri_runtime.py | t0-morphology-furnace |
| comprehensive_run.py | t0-morphology-furnace/exploratory/shadow-ambiguity |
| confluence_calibrator.py | commit-confluence |
| pri_v2_io_plugins.py | t0-morphology-furnace |
| diagnose_inter_head_disagreement.py | t0-morphology-furnace/scripts |
| analyze_adaptive_step.py | t0-morphology-furnace/scripts (transitive) |
| test_shadow_ambiguity.py | t0-morphology-furnace/exploratory/shadow-ambiguity (transitive) |
| stage_b/fusion_signs.json | commit-confluence/stage_b |

## sha256 (vendored 2026-06-21)

- analyze_adaptive_step.py: 2dffb7f03fa4876e
- comprehensive_run.py: f6f5958bae5b035f
- confluence_calibrator.py: 6142217f7608dc7c
- diagnose_inter_head_disagreement.py: b996ed923ac3eefe
- pri_calibrator.py: 78c4f098295fe600
- pri_runtime.py: cf56a2607b94666a
- pri_v2_io_plugins.py: 6c56be1888abc6ad
- test_shadow_ambiguity.py: 12046a3ff98ddaba
