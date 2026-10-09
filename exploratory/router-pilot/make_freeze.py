"""Freeze authoring and shared integrity/I/O helpers. Verification: not run by Codex."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import re
import tempfile

# Modal can auto-mount an import under /root; integrity checks must use the
# explicitly image-shipped tree, not that SDK-dependent module mount location.
_SHIPPED = Path("/payload/exploratory/router-pilot")
HERE = _SHIPPED if (_SHIPPED / "PRE_REGISTRATION.md").is_file() else Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ("anli_r1", "halueval_qa")
DATA_SHA = {
    "anli_r1": "57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b",
    "halueval_qa": "a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class GateFailure(RuntimeError):
    """A measured gate missed its frozen threshold: terminal, never amendable."""


def gate(condition, message):
    if not condition:
        raise GateFailure("GATE: " + message)


def exception_chain(exc):
    """Visit both wrapper links, including cycles and suppressed contexts."""
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


def is_gate_failure(exc):
    for current in exception_chain(exc):
        # Modal's deserialization fallback embeds the remote traceback in its message.
        if isinstance(current, GateFailure) or "GATE:" in str(current):
            return True
        # Avoid importing Hub while classifying an import failure. Inspect base classes
        # too (e.g. a gated-repository error); transformers may wrap these in OSError.
        if any(c.__module__.startswith("huggingface_hub.") and c.__name__ in
               ("RepositoryNotFoundError", "RevisionNotFoundError")
               for c in type(current).__mro__):
            return True
    return False


def design_sha():
    """Sections 1-8 of the pre-registration: amendments may never change them."""
    text = (HERE / "PRE_REGISTRATION.md").read_text()
    start, end = text.index("\n## 1. "), text.index("\n## 9. ")
    return hashlib.sha256(text[start:end].encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic(path, data, replace=False, serialized=False):
    """Exclusive writer claim plus fsync/rename; no dependency on Volume hard links."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = None
    if not serialized:
        claim = path.with_name("." + path.name + ".write-claim")
        claim_fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(claim_fd)
    tmp = None
    try:
        require(replace or not path.exists(), f"refusing overwrite: {path.name}")
        fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
        if claim is not None:
            claim.unlink()


def sources():
    files = [HERE / n for n in ("PRE_REGISTRATION.md", "modal_router_pilot.py",
                               "score_router_pilot.py", "make_freeze.py")]
    files += [ROOT / "exploratory/depth-curve" / n for n in
              ("stem_grouped_sensitivity.py", "rescore_grid_a.py", "depth_score.py", "score_grid_b.py")]
    files += sorted((ROOT / "modal/seal").glob("*.py"))
    require(files[-1].parent == ROOT / "modal/seal", "sealed source directory missing")
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def check_freeze(freeze, stage, manifest_dir=None):
    require(freeze["schema"] == "router-pilot-freeze/1", "freeze schema")
    entries = freeze["entries"]
    require(entries and entries[0]["entry"] == "1", "missing initial freeze")
    require(len({e["entry"] for e in entries}) == len(entries), "duplicate freeze stage")
    current = entries[-1]
    require(freeze["stage"] == current["stage"], "freeze stage header")
    require(current["stage"] == stage, "wrong freeze stage")
    require(current["files"] == sources(), "frozen source drift")
    if stage == 2:
        require(len(entries) >= 2 and entries[-2]["stage"] == 1, "missing FREEZE-1")
        require(current["files"] == entries[-2]["files"], "code changed after smoke")
        require(set(current["manifests"]) == {f"{t}.manifest.json" for t in TASKS},
                "manifest set")
        if manifest_dir is not None:
            for name, digest in current["manifests"].items():
                require(sha(Path(manifest_dir) / name) == digest, "frozen manifest drift")
    return current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--amendment", default="")
    args = ap.parse_args()
    path = HERE / "FREEZE.json"
    lock = HERE / ".freeze-writing"
    # An interrupted writer requires inspection, not silent lock removal.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        freeze = json.loads(path.read_text()) if path.exists() else {
            "schema": "router-pilot-freeze/1", "entries": []}
        entry = str(args.stage)
        if args.amendment:
            require(args.stage == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+", args.amendment),
                    "amendment needs stage 1 and YYYY-MM-DD-tag")
            require(freeze["entries"] and freeze["entries"][-1]["stage"] == 1,
                    "amendment requires existing stage 1")
            require(args.amendment in (HERE / "PRE_REGISTRATION.md").read_text(),
                    "dated amendment must be recorded in pre-registration")
            require(not (HERE / "smoke.report.json").exists()
                    and not list(HERE.glob("*.status.json")),
                    "a successful smoke or a terminal cell cannot be amended")
            require(not any(json.loads(p.read_text()).get("gate_failure")
                            for p in HERE.glob("smoke.failed.*.json")),
                    "a measured-gate failure is terminal")
            require(design_sha() == freeze["entries"][0]["design_sha256"],
                    "an amendment cannot change sections 1-8")
            entry = "1-amend-" + args.amendment
        require(all(e["entry"] != entry for e in freeze["entries"]), "stage already exists")
        rec = {"entry": entry, "stage": args.stage, "utc": utc(), "files": sources(),
               "design_sha256": design_sha()}
        if args.stage == 2:
            check_freeze(freeze, 1)
            report = json.loads((HERE / "smoke.report.json").read_text())
            require(report["passed"] is True, "smoke did not pass")
            require(report["freeze_entry"] == freeze["entries"][-1], "smoke freeze mismatch")
            rec["manifests"] = {f"{t}.manifest.json": sha(HERE / f"{t}.manifest.json") for t in TASKS}
            require(rec["manifests"] == report["manifests"], "smoke manifest mismatch")
        elif not args.amendment:
            require(not freeze["entries"], "stage ordering")
        freeze["entries"].append(rec)
        freeze["stage"], freeze["utc"] = args.stage, rec["utc"]
        atomic(path, json_bytes(freeze), replace=path.exists())
    finally:
        lock.unlink()
    print("Freeze written; no model or analysis was run.")


if __name__ == "__main__":
    main()
