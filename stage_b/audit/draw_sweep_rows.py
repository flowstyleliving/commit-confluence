#!/usr/bin/env python3
"""Fix the ACE row draw for the commit-confluence sealed reproducibility sweep.

Run ONCE, BEFORE any cell is verified. Writes stage_b/audit/SWEEP_ROW_DRAW.json.
Re-running reproduces byte-identical output (pure function of SEED + the cohort).

Draw protocol
-------------
SEED = 20260823 (single global seed, stated in the sweep report).
Cells are ordered canonically: task ascending, then model slug ascending (Python
sort on the exact profile basenames found on disk). One numpy Generator is created
from SEED and consumed in that order, so cell k's draw depends on the whole prefix
-> the draw cannot be re-rolled per cell without changing every later cell.
For each cell: n = number of lines in that cell's data jsonl (asserted == the
profile's n_aligned, which is what verify_cell.py requires), and 4 DISTINCT indices
are drawn uniformly without replacement from range(n). Order as drawn is preserved.
"""
import glob
import hashlib
import json
import os

import numpy as np

SEED = 20260823
K = 4  # ACE rows per cell
READOUT_PREFIX = 4  # fixed by protocol (RNG-aligned prefix), not drawn
CC_ROOT = "/Users/msrk/Documents/commit-confluence"
TASKS = ["anli_r1", "triviaqa_paired"]
OUT = os.path.join(CC_ROOT, "stage_b", "audit", "SWEEP_ROW_DRAW.json")

# already-executed audit cells and the rows those runs actually used (historical,
# NOT re-drawn; recorded so the sweep can never silently re-pick them)
ALREADY = {
    ("anli_r1", "Qwen2.5-7B-Instruct-4bit"): [168, 21, 0, 100],
    ("triviaqa_paired", "Llama-3.2-3B-Instruct-4bit"): [7, 42, 123, 199],
}

cells = []
for task in TASKS:
    for p in sorted(glob.glob(os.path.join(CC_ROOT, "stage_b", "profiles", task, "*.profile.json"))):
        slug = os.path.basename(p)[: -len(".profile.json")]
        prof = json.load(open(p))
        data_path = os.path.join(CC_ROOT, prof["data_path"])
        n = len(open(data_path).read().splitlines())
        assert n == prof["n_aligned"], (task, slug, n, prof["n_aligned"])
        cells.append({"task": task, "model_slug": slug, "model_id": prof["model"],
                      "n_rows": n, "data_path": prof["data_path"],
                      "geom_deployable": prof["secondary_geometric_only"]["deployable"],
                      "geom_winner": prof["secondary_geometric_only"]["winner"],
                      "primary_winner": prof["primary_full_panel"]["winner"]})

rng = np.random.default_rng(SEED)
for c in cells:
    c["ace_rows"] = [int(x) for x in rng.choice(c["n_rows"], size=K, replace=False)]
    c["readout_prefix"] = READOUT_PREFIX
    key = (c["task"], c["model_slug"])
    c["already_audited"] = key in ALREADY
    if key in ALREADY:
        c["historical_ace_rows_used"] = ALREADY[key]

doc = {
    "schema": "cc-sweep-row-draw/1",
    "purpose": "Pre-fixed ACE row indices for the sealed-cell reproducibility sweep. "
               "Drawn BEFORE any cell ran. Re-picking rows after a mismatch is "
               "cherry-picking and is forbidden.",
    "seed": SEED,
    "rows_per_cell": K,
    "readout_prefix": READOUT_PREFIX,
    "cell_order": "task ascending, then model slug ascending; one shared Generator "
                  "consumed in that order",
    "rng": "numpy.random.default_rng(SEED).choice(n, size=4, replace=False), numpy "
           + np.__version__,
    "drawing_code": open(os.path.abspath(__file__)).read(),
    "drawing_code_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
    "n_cells": len(cells),
    "cells": cells,
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=1)
print(f"wrote {OUT}  ({len(cells)} cells, seed {SEED})")
for c in cells:
    flag = "  [ALREADY AUDITED]" if c["already_audited"] else ""
    print(f"  {c['task']:16s} {c['model_slug']:32s} n={c['n_rows']} rows={c['ace_rows']}{flag}")
