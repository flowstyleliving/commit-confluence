#!/usr/bin/env python3
"""Static standalone-install audit; does not import MLX or project runtime code."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENDORED_T0 = ROOT / "vendor" / "t0_core"

PINNED_HASHES = {
    "pri_calibrator.py": "78c4f098295fe600cc4f6f1a14cc7b496ac93d8d70dd65743c1504eb20931101",
    "exploratory/shadow-ambiguity/comprehensive_run.py": "f6f5958bae5b035f07f034b9f4f5784b1a86c1ab00ab0f9c9ef4a5584776e8f8",
    "scripts/diagnose_inter_head_disagreement.py": "b996ed923ac3eefeb0550901b2bf8ce5c5e0ae6486c8d41efdc2de9fb9567c99",
    "pri_runtime.py": "cf56a2607b94666ab092647a764b1b73c7974b38445caa6a4b252a9cdf240784",
    "pri_v2_io_plugins.py": "6c56be1888abc6ad44150583b869bac06366fa70a7d8e13230da3703064eb7b7",
    "pri_v2_mlx_pipeline.py": "abb0debd8b7b3f0503418e09a21050fad968d88a6ea10fd4412983c6ba6e4a5c",
    "model_adapters.py": "5ce42c7f8de787e6f301c4542ebbb5057bc8b1702758702095f62cc720743137",
    "exploratory/shadow-ambiguity/test_shadow_ambiguity.py": "12046a3ff98ddaba207396b943415dd5360f4cb4a942462994889d0aa8771961",
}

REQUIRED_DISTRIBUTIONS = {
    "numpy": ("numpy", "2.0.2"),
    "scipy": ("scipy", "1.13.1"),
    "scikit-learn": ("sklearn", "1.6.1"),
    "pandas": ("pandas", "2.3.3"),
    "mlx": ("mlx", "0.29.3"),
    "mlx-lm": ("mlx_lm", "0.29.1"),
    "huggingface-hub": ("huggingface_hub", "0.36.2"),
    "transformers": ("transformers", "4.57.6"),
    "matplotlib": ("matplotlib", "3.9.4"),
    "seaborn": ("seaborn", "0.13.2"),
    "datasets": ("datasets", "4.5.0"),
    "pyarrow": ("pyarrow", "21.0.0"),
    "tqdm": ("tqdm", "4.68.1"),
}

SEALED_DATA = (
    "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl",
    "experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    print(f"repo: {ROOT}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"CONFLUENCE_T0_REPO: {os.environ.get('CONFLUENCE_T0_REPO', '<unset>')}")

    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        failures.append("fresh MLX extraction requires Apple silicon/macOS")
    if platform.python_version() != "3.9.6":
        failures.append(
            "BENCH resume is pinned to Python 3.9.6; rerun setup with "
            "CONFLUENCE_SETUP_PYTHON=/path/to/python3.9.6 if needed")

    print("\nvendored extraction core:")
    for relative, expected in PINNED_HASHES.items():
        path = VENDORED_T0 / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        ok = actual == expected
        print(f"  {'OK' if ok else 'FAIL'} {relative}")
        if not ok:
            failures.append(f"hash mismatch: {relative} ({actual})")

    for relative in SEALED_DATA:
        if not (VENDORED_T0 / relative).is_file():
            failures.append(f"missing sealed exclusion data: {relative}")

    dependency_errors = []
    for distribution, (module, expected) in REQUIRED_DISTRIBUTIONS.items():
        if importlib.util.find_spec(module) is None:
            dependency_errors.append(f"{distribution}=MISSING")
            continue
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_errors.append(f"{distribution}=MISSING")
            continue
        if actual != expected:
            dependency_errors.append(f"{distribution}={actual} (expected {expected})")
    print("\npython dependencies:")
    print("  OK" if not dependency_errors else "  DRIFT " + ", ".join(dependency_errors))
    if dependency_errors:
        failures.append("run ./confluence setup")

    if failures:
        print("\nNOT READY")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nREADY: analysis and fresh MLX extraction dependencies are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
