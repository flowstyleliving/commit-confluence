#!/usr/bin/env python3
"""Drive the full RPV pass-one panel. Enumerates and freezes the roster BEFORE running.

Steward-authored 2026-09-16 under MK authorization. Writes a content-hashed cell
manifest first, then runs every cell sequentially, continuing past per-cell failures
so one bad cell cannot silently truncate the panel.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rpv_pass1_temperature as offline  # noqa: E402

BANKED = Path("/Users/msrk/Documents/t0-morphology-furnace/exploratory/shadow-ambiguity/comprehensive_outputs")
DATA_ROOT = Path("/Users/msrk/Documents/commit-confluence/vendor/t0_core")
HUB = Path.home() / ".cache/huggingface/hub"
PYTHON = Path("/Users/msrk/Documents/t0-morphology-furnace/.venv/bin/python")
OUT = HERE / "panel"


def snapshot_for(model_id):
    slug = "models--" + model_id.replace("/", "--")
    snaps = sorted((HUB / slug / "snapshots").glob("*"))
    for s in snaps:
        if any(p.suffix == ".safetensors" for p in s.rglob("*")):
            return s
    raise FileNotFoundError(f"No local safetensors snapshot for {model_id}")


def enumerate_cells():
    """The roster is READ from the banked artifacts, never hand-typed."""
    cells = []
    for f in sorted(glob.glob(str(BANKED / "shadow_v2__*.json"))):
        d = json.loads(Path(f).read_text())
        if "rows" not in d or not d["rows"]:
            continue  # gpt-oss-20b: banked artifact carries no rows (harness skip)
        cells.append(dict(model_id=d["model"], benchmark=d["benchmark"],
                          banked_rows=len(d["rows"]), data_path=d["data_path"]))
    return cells


def main():
    OUT.mkdir(exist_ok=True)
    cells = enumerate_cells()
    manifest = dict(
        schema="rpv-pass1-panel-manifest/v1",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        temperature_grid=list(offline.GRID),
        a4_accept_ratio=offline.ACCEPT_RATIO,
        reference_hashes=offline.reference_hashes(),
        overlay_hashes={p.name: offline.sha256(HERE / p.name) for p in
                        [Path("rpv_pass1_run.py"), Path("rpv_pass1_temperature.py"),
                         Path("PRE_REGISTRATION_RPV_PASS1.md"), Path("run_panel.py")]},
        n_cells=len(cells), n_models=len({c["model_id"] for c in cells}),
        total_banked_rows=sum(c["banked_rows"] for c in cells),
        cells=[],
    )
    for c in cells:
        data = DATA_ROOT / c["data_path"]
        snap = snapshot_for(c["model_id"])
        manifest["cells"].append({**c, "data_sha256": offline.sha256(data),
                                  "snapshot": str(snap)})
    mpath = OUT / "PANEL_MANIFEST.json"
    if mpath.exists():
        sys.exit(f"Manifest already frozen at {mpath}; refusing to overwrite.")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] frozen: {len(cells)} cells, {manifest['n_models']} models, "
          f"{manifest['total_banked_rows']} rows -> {mpath}")

    results = []
    t0 = time.time()
    for i, c in enumerate(manifest["cells"], 1):
        tag = c["model_id"].split("/")[-1] + "__" + c["benchmark"]
        out = OUT / f"pass1__{tag}.json"
        if out.exists():
            print(f"[{i}/{len(cells)}] SKIP (exists) {tag}")
            results.append(dict(cell=tag, status="skipped_exists"))
            continue
        cmd = [str(PYTHON), str(HERE / "rpv_pass1_run.py"),
               "--model-id", c["model_id"], "--model-path", c["snapshot"],
               "--benchmark", c["benchmark"], "--data", str(DATA_ROOT / c["data_path"]),
               "--limit", "0", "--out", str(out)]
        start = time.time()
        print(f"[{i}/{len(cells)}] RUN {tag} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - start
        if proc.returncode == 0 and out.exists():
            d = json.loads(out.read_text())
            fails = len(d.get("pass_one_failures", {}))
            a4 = d.get("a4_inline_discrepancy", {})
            a4_fail = sum(1 for v in a4.values() if v.get("passed") is False)
            print(f"      OK {len(d['rows'])} rows, {fails} pass-one failures, "
                  f"{a4_fail}/{len(a4)} A4 keys failing, {dt/60:.1f} min", flush=True)
            results.append(dict(cell=tag, status="ok", rows=len(d["rows"]),
                                pass_one_failures=fails, a4_failing_keys=a4_fail,
                                minutes=round(dt / 60, 2)))
        else:
            err = (proc.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
            print(f"      FAILED rc={proc.returncode}: {err[0][:200]}", flush=True)
            results.append(dict(cell=tag, status="failed", returncode=proc.returncode,
                                error=err[0][:500], minutes=round(dt / 60, 2)))
        (OUT / "PANEL_PROGRESS.json").write_text(json.dumps(
            dict(completed=len(results), of=len(cells),
                 elapsed_min=round((time.time() - t0) / 60, 1), results=results), indent=2) + "\n")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[panel] {ok}/{len(cells)} cells OK in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
