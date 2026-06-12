#!/usr/bin/env python3
"""Tiny plumbing smoke: prove collect_ace_matrix (attention pass) runs end-to-end and
merges with the readout matrix. Not a statistical check - just wiring."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC

RPV_DIR = os.path.join(CC.T0_REPO, "exploratory/shadow-ambiguity/comprehensive_outputs")
DATA = os.path.join(CC.T0_REPO, "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl")
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
RPV = RPV_DIR + "/shadow_v2__Llama-3.2-3B-Instruct-4bit__anli_r1__limitall.json"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6
print(f"[smoke] collecting ACE on {MODEL} limit={limit} ...", flush=True)
ace = CC.collect_ace_matrix(MODEL, DATA, seed=20260526, limit=limit)
fin = np.isfinite(ace["score_matrix"]).sum(0).tolist()
print("[smoke] ACE matrix", ace["score_matrix"].shape, "finite/cell:", fin)
print("[smoke] ACE cells:", [c[2] for c in ace["panel"]])
ro = CC.load_readout_matrix(RPV)
print("[smoke] readout n:", ro["n"])
prof = CC.merge_and_calibrate(ace, ro, n_bootstrap=200, seed=20260610)
print("[smoke] merged:", {k: prof[k] for k in ("n", "n_aligned", "n_cells_total",
                                               "n_ace_cells", "n_readout_cells")})
print("[smoke] primary winner:", prof["primary_full_panel"]["winner"])
print("[smoke] geom winner:   ", prof["secondary_geometric_only"]["winner"])
print("[smoke] OK")
