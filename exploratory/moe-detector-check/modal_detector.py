"""Budgeted Modal 1.2.6 transport for the unchanged local detector panel.

Run from this directory: modal run modal_detector.py::plan
No seal imports here: every detector invocation is a fresh Python subprocess.
"""
import contextlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

import modal


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAYLOAD = Path("/payload")
REMOTE_HERE = PAYLOAD / "exploratory/moe-detector-check"
LEDGER = HERE / "modal_spend_ledger.json"
HARD_CAP_USD = 10.00
APP_NAME = "moe-detector-check"
# Modal list-price estimates dated 2026-10-10, not an invoice guarantee.
RATES = {"A100-80GB": Decimal("0.000694"), "cpu": Decimal("0.0000131"),
         "memory_gib": Decimal("0.00000222")}
SHARES = {("download", "qwen15"): 0.20, ("download", "qwen3"): 0.25,
          ("smoke", "qwen15"): 0.80, ("full", "qwen15"): 2.40,
          ("full", "qwen3"): 3.30}


def resources(kind):
    return {"GPU": None, "cpu": 4, "memory": 8} if kind == "download" else {
        "GPU": "A100-80GB", "cpu": 8, "memory": 32}


def rate_for(spec):
    return (RATES.get(spec["GPU"], Decimal(0)) + spec["cpu"] * RATES["cpu"]
            + spec["memory"] * RATES["memory_gib"])


def budget(kind, share):
    share = Decimal(str(share))
    if not share.is_finite() or not 0 < share <= Decimal(str(HARD_CAP_USD)):
        raise ValueError("share must be finite, positive, and at most $10")
    spec = resources(kind)
    rate = rate_for(spec)
    # Round DOWN: the hard timeout, rather than a runtime prediction, defines liability.
    timeout = int((share / rate).to_integral_value(rounding=ROUND_FLOOR))
    if timeout < (1 if kind == "download" else 120) or timeout > 86400:
        raise ValueError("share gives an unusable timeout (GPU needs >=120s; maximum 24h)")
    ceiling = timeout * rate
    assert ceiling <= share
    return {**spec, "share_usd": float(share), "timeout": timeout,
            "rate_usd_per_second": float(rate), "ceiling_usd": float(ceiling),
            # No measured speed yet: the initial estimate conservatively equals the ceiling.
            "estimate_usd": float(ceiling), "estimate_basis": "timeout at list price"}


MODELS = {
    "qwen15": {"model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
               "revision": "ec052fda178e241c7c443468d2fa1db6618996be",
               "model_type": "qwen2_moe", "GPU": "A100-80GB", "status": "compatible",
               "timeout": budget("full", SHARES[("full", "qwen15")])["timeout"]},
    "qwen3": {"model_id": "Qwen/Qwen3-30B-A3B",
              "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
              "model_type": "qwen3_moe", "GPU": "A100-80GB", "status": "compatible",
              "timeout": budget("full", SHARES[("full", "qwen3")])["timeout"]},
    "olmoe": {"model_id": "allenai/OLMoE-1B-7B-0924-Instruct",
              "revision": "7f1c97f440f06ce36705e4f2b843edb5925f4498",
              "GPU": None, "timeout": None, "status": "not_compatible",
              "reason": "MK decision 2026-10-13: native final RMSNorm epsilon 1e-5; "
                        "the preregistered sealed readout requires 1e-6."},
}
TORCH = "torch==2.14.1+cu126"
PINS = ("transformers==5.19.0", "accelerate==1.15.0", "numpy==2.5.3",
        "pandas==3.0.6", "scikit-learn==1.9.1", "scipy==1.18.1",
        "safetensors==0.8.0", "tokenizers==0.23.2", "huggingface_hub==1.33.0",
        "hf-xet==1.6.0", "jinja2==3.1.6")
RECIPE = {"base": "debian_slim", "python": "3.12", "torch": TORCH,
          "torch_index": "https://download.pytorch.org/whl/cu126", "pins": PINS}
PAYLOAD_FILES = ("modal/modal_app.py", "modal/refs_upload/anli_r1.matrix.npz",
                 "stage_b/data/anli_R1_seed20260612_n200.jsonl",
                 "exploratory/moe-detector-check/local_detector.py")
ALLOW = ["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
         "added_tokens.json", "chat_template*", "merges.txt", "vocab*",
         "*.safetensors", "*.safetensors.index.json"]

# Separate apps keep plan/reconcile from building the image or launching workers.
# The local command app has no functions; the worker app owns all billed compute.
app = modal.App(APP_NAME)
workers = modal.App(APP_NAME)
vol = modal.Volume.from_name("moe-detector-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(TORCH, index_url=RECIPE["torch_index"])
         .pip_install(*PINS)
         .env({"HF_HOME": "/cache/hub", "HF_HUB_DISABLE_TELEMETRY": "1",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(REPO / "modal/seal"), "/payload/modal/seal", copy=True,
                        ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc"))
for relative in PAYLOAD_FILES:
    image = image.add_local_file(str(REPO / relative), str(PAYLOAD / relative), copy=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def model_spec(key):
    if key not in MODELS:
        raise ValueError(f"unknown model {key!r}; choose qwen15 or qwen3")
    spec = MODELS[key]
    if spec["status"] != "compatible":
        raise ValueError(f"refusing {key}: {spec['status']}: {spec['reason']}")
    return spec


def slug_for(spec):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec["model_id"].split("/")[-1])


def write_json(path, value, sync_directory=True):
    path = Path(path)
    tmp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if not sync_directory:
        return  # Modal Volume publication uses vol.commit(), not a directory fsync.
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def payload_hashes(root):
    paths = [root / p for p in PAYLOAD_FILES]
    paths += [p for p in (root / "modal/seal").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid amount in spend ledger; manual audit required")
    return result


def spent_so_far(ledger):
    total = Decimal(0)
    for entry in ledger["entries"]:
        if entry["billed_usd"] is not None:
            total += money(entry["billed_usd"])
        else:
            total += max(money(entry["measured_cost_usd"] or 0),
                         money(entry["ceiling_usd"]) if not entry["end_utc"] else Decimal(0))
    return total


@contextlib.contextmanager
def locked_ledger():
    # Serialize local launches and reconciliation. A crash releases flock but leaves the reservation.
    with LEDGER.with_suffix(".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another wrapper command holds the spend ledger; launch nothing") from None
        try:
            if LEDGER.exists():
                ledger = json.loads(LEDGER.read_text())
                if ledger["schema"] != 1 or ledger["hard_cap_usd"] != HARD_CAP_USD:
                    raise ValueError("unrecognized ledger; manual audit required")
                spent_so_far(ledger)  # Fail closed on missing or invalid accounting fields.
            else:
                ledger = {"schema": 1, "hard_cap_usd": HARD_CAP_USD,
                          "task_start_utc": utc(), "entries": []}
            yield ledger
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def download_complete(ledger, key):
    spec = model_spec(key)
    return any(e["kind"] == "download" and e["model"] == spec["model_id"]
               and e["revision"] == spec["revision"] and e["status"] == "complete"
               for e in ledger["entries"])


def snapshot_manifest(snapshot, revision):
    if snapshot.name != revision or snapshot.parent.name != "snapshots":
        raise RuntimeError(f"snapshot does not match pinned SHA: {snapshot}")
    files = {str(p.relative_to(snapshot)): p.stat().st_size
             for p in snapshot.rglob("*") if p.is_file()}
    if not files.get("config.json") or not files.get("tokenizer_config.json"):
        raise RuntimeError("snapshot is missing config/tokenizer config")
    if not (files.get("tokenizer.json") or (files.get("vocab.json") and files.get("merges.txt"))):
        raise RuntimeError("snapshot is missing tokenizer vocabulary")
    weights = [p for p in files if p.endswith(".safetensors")]
    if not weights or any(files[p] == 0 for p in weights):
        raise RuntimeError("snapshot has no complete safetensors weights")
    for index in snapshot.glob("*.safetensors.index.json"):
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(not files.get(p) for p in shards):
            raise RuntimeError("snapshot is missing indexed weight shards")
    return files


@workers.cls(image=image, volumes={"/cache": vol}, cpu=4, memory=8 * 1024,
             timeout=budget("download", SHARES[("download", "qwen15")])["timeout"],
             retries=0, max_containers=1, scaledown_window=2)
class Downloader:
    @modal.method()
    def download(self, key):
        start = time.perf_counter()
        spec = model_spec(key)
        vol.reload()
        marker = Path("/cache/downloads") / (slug_for(spec) + ".json")
        cached = None
        if marker.exists():
            try:
                prior = json.loads(marker.read_text())
                if prior["revision"] == spec["revision"]:
                    files = snapshot_manifest(Path(prior["snapshot"]), spec["revision"])
                    if files == prior["files"]:
                        cached = prior
            except (OSError, ValueError, KeyError, RuntimeError):
                pass  # Incomplete cache is repaired only by this CPU download path.
        if cached is None:
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(spec["model_id"], revision=spec["revision"],
                                             allow_patterns=ALLOW, token=False))
            files = snapshot_manifest(snapshot, spec["revision"])
            cached = {"model": spec["model_id"], "revision": spec["revision"],
                      "snapshot": str(snapshot), "files": files, "bytes": sum(files.values())}
            marker.parent.mkdir(parents=True, exist_ok=True)
            write_json(marker, cached, sync_directory=False)
        vol.commit()
        elapsed = time.perf_counter() - start
        print(f"DOWNLOAD container wall seconds: {elapsed:.3f}", flush=True)
        return {**cached, "elapsed_seconds": elapsed, "exit_code": 0}


@workers.cls(image=image, gpu="A100-80GB", volumes={"/cache": vol}, cpu=8, memory=32 * 1024,
             timeout=budget("smoke", SHARES[("smoke", "qwen15")])["timeout"],
             retries=0, max_containers=1, scaledown_window=2)
class Detector:
    @modal.method()
    def detect(self, key, n, run_name, run_budget_seconds, expected_hashes):
        start = time.perf_counter()
        spec = model_spec(key)
        if n not in (12, 200) or (n == 12 and key != "qwen15"):
            raise ValueError("unsupported detector run")
        if not re.fullmatch(r"modal-\d{8}T\d{12}Z-n" + str(n), run_name):
            raise ValueError("invalid run name")
        slug = slug_for(spec)
        relative = Path("runs") / slug / run_name
        out = REMOTE_HERE / relative
        destination = Path("/cache/results") / slug / run_name
        proc, watchdog = None, None
        stop_watchdog = threading.Event()
        log = []
        gpu_info = None
        exit_code = 1
        vol.reload()
        destination.mkdir(parents=True, exist_ok=False)
        # Persist the attempt before model execution; a duplicate invocation must stop here.
        (destination / "attempt.json").write_text(json.dumps(
            {"model": spec["model_id"], "revision": spec["revision"], "run_name": run_name}))
        vol.commit()
        if out.exists():
            raise RuntimeError("refusing to overwrite remote detector output")
        try:
            if payload_hashes(PAYLOAD) != expected_hashes:
                raise RuntimeError("image payload differs from reserved local payload")
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                capture_output=True, text=True, timeout=10, check=False)
            gpu_info = {"stdout": gpu.stdout, "stderr": gpu.stderr, "exit_code": gpu.returncode}
            if gpu.returncode or len(gpu.stdout.strip().splitlines()) != 2:
                raise RuntimeError("nvidia-smi must report exactly one GPU")
            if time.perf_counter() - start >= run_budget_seconds - 90:
                raise RuntimeError("preflight exhausted the subprocess budget")
            command = [sys.executable, "-u", "local_detector.py", "--model-id", spec["model_id"],
                       "--revision", spec["revision"], "--device", "cuda", "--precision", "bf16",
                       "--task", "anli_r1", "--n", str(n), "--max-prompt-tokens", "512",
                       "--out", str(relative)]
            proc = subprocess.Popen(command, cwd=str(REMOTE_HERE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})

            def watch():
                delay = max(0, start + run_budget_seconds - 90 - time.perf_counter())
                if not stop_watchdog.wait(delay) and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        return
                    if not stop_watchdog.wait(60) and proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

            watchdog = threading.Thread(target=watch, daemon=True)
            watchdog.start()
            with proc.stdout:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.append(line)
            exit_code = proc.wait()
        except (Exception, KeyboardInterrupt):
            message = traceback.format_exc()
            print(message, flush=True)
            log.append(message)
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            stop_watchdog.set()
            if watchdog is not None:
                watchdog.join(timeout=1)
        report_path = out / "report.json"
        if report_path.exists():
            report = report_path.read_bytes()
        else:
            exit_code = exit_code or 1
            report = (json.dumps({"status": "failed", "origin": "modal_wrapper",
                                 "error": "detector did not produce report.json; see stdout.log"},
                                indent=2) + "\n").encode()
        matrix = (out / "matrix.npz").read_bytes() if (out / "matrix.npz").exists() else None
        stdout = "".join(log)
        # Exclusive artifact creation also refuses a duplicate run after a container restart.
        for name, data in (("report.json", report), ("matrix.npz", matrix),
                           ("stdout.log", stdout.encode())):
            if data is not None:
                with (destination / name).open("xb") as handle:
                    handle.write(data)
        vol.commit()
        elapsed = time.perf_counter() - start
        print(f"DETECT container wall seconds: {elapsed:.3f}; exit code: {exit_code}", flush=True)
        return {"report_json": report, "matrix_npz": matrix, "stdout_log": stdout,
                "exit_code": exit_code, "elapsed_seconds": elapsed, "gpu_info": gpu_info,
                "payload_sha256": expected_hashes, "volume_results": str(destination)}


def launch(kind, key, share):
    spec = model_spec(key)  # Includes the permanent OLMoE refusal, before any worker launch.
    allocation = budget(kind, SHARES[(kind, key)] if share is None else share)
    n = 12 if kind == "smoke" else 200
    run_name = "modal-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-n{n}"
    with locked_ledger() as ledger:
        if kind != "download" and not download_complete(ledger, key):
            raise RuntimeError(f"download for {key}@{spec['revision']} is not recorded complete")
        used = spent_so_far(ledger)
        if used + money(allocation["ceiling_usd"]) > money(HARD_CAP_USD):
            raise RuntimeError(f"launch refused: spent/reserved ${used} + ceiling "
                               f"${allocation['ceiling_usd']} exceeds ${HARD_CAP_USD:.2f}")
        out = None
        if kind != "download":
            runs = (HERE / "runs").resolve()
            out = runs / slug_for(spec) / run_name
            if not runs.is_relative_to(HERE) or not out.resolve().is_relative_to(runs):
                raise RuntimeError("runs/ must stay inside this lane")
            out.mkdir(parents=True, exist_ok=False)
        record = {"run_id": uuid.uuid4().hex, "run_name": run_name, "kind": kind,
                  "model_key": key, "model": spec["model_id"], "revision": spec["revision"],
                  **allocation, "memory_unit": "GiB", "launch_utc": utc(), "end_utc": None,
                  "elapsed_seconds": None, "measured_cost_usd": None, "modal_app_id": None,
                  "billed_usd": None, "status": "launching", "image_recipe": RECIPE,
                  "payload_sha256": payload_hashes(REPO), "list_price_date": "2026-10-10"}
        ledger["entries"].append(record)
        write_json(LEDGER, ledger)  # Durable reservation BEFORE image build and .remote().
        try:
            with workers.run():
                record["modal_app_id"] = workers.app_id
                record["modal_image_id"] = image.object_id
                write_json(LEDGER, ledger)
                if kind == "download":
                    result = Downloader.with_options(timeout=allocation["timeout"])().download.remote(key)
                else:
                    result = Detector.with_options(timeout=allocation["timeout"])().detect.remote(
                        key, n, run_name, allocation["timeout"], record["payload_sha256"])
                elapsed = result["elapsed_seconds"]
                if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("invalid container elapsed time; retaining full reservation")
                record.update({"end_utc": utc(), "elapsed_seconds": elapsed,
                               "measured_cost_usd": float(Decimal(str(elapsed)) * rate_for(allocation)),
                               "exit_code": result["exit_code"],
                               "status": "complete" if result["exit_code"] == 0 else "failed"})
                record.update({k: v for k, v in result.items()
                               if k not in ("report_json", "matrix_npz", "stdout_log")})
                write_json(LEDGER, ledger)
                if out is not None:
                    for name, data in (("report.json", result["report_json"]),
                                       ("matrix.npz", result["matrix_npz"]),
                                       ("stdout.log", result["stdout_log"].encode())):
                        if data is not None:
                            with (out / name).open("xb") as handle:
                                handle.write(data)
                    write_json(out / "modal_run.json", record)
            record["app_context_exit_utc"] = utc()
            write_json(LEDGER, ledger)
            if out is not None:
                write_json(out / "modal_run.json", record)
        except BaseException as exc:
            record["modal_app_id"] = record["modal_app_id"] or workers.app_id
            record["app_context_exit_utc"] = utc()
            record["local_exception_utc"] = utc()
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "remote_unknown" if record["end_utc"] is None else "artifact_error"
            # No container result means no end time or invented elapsed: retain the full ceiling.
            write_json(LEDGER, ledger)
            if out is not None:
                write_json(out / "modal_run.json", record)
            raise
        print(json.dumps(record, indent=2))
        if record["exit_code"] != 0:
            raise SystemExit("detector failed; artifacts saved; no retry launched")


@app.local_entrypoint()
def download(model: str, share: float = None):
    """CPU-only pinned snapshot download; --share overrides its dollar ceiling."""
    launch("download", model, share)


@app.local_entrypoint()
def smoke(share: float = None):
    """Qwen1.5, first 12 rows, including the unchanged sealed calibration."""
    launch("smoke", "qwen15", share)


@app.local_entrypoint()
def run(model: str, share: float = None):
    """Full n=200 ANLI run; requires a completed pinned download in the ledger."""
    launch("full", model, share)


@app.local_entrypoint()
def plan():
    """Print the default plan without building the image or launching any workers."""
    print(json.dumps(MODELS, indent=2))
    with locked_ledger() as ledger:
        print("List-price estimates (2026-10-10); memory in GiB; no speed estimate yet.")
        print("kind      model    state       share     timeout     ceiling     estimate")
        for (kind, key), share in SHARES.items():
            spec = model_spec(key)
            done = any(e["kind"] == kind and e["model"] == spec["model_id"]
                       and e["revision"] == spec["revision"] and e["status"] == "complete"
                       for e in ledger["entries"])
            b = budget(kind, share)
            print(f"{kind:9} {key:8} {'complete' if done else 'pending':9} "
                  f"${share:.2f} {b['timeout']:9}s ${b['ceiling_usd']:.8f} ${b['estimate_usd']:.8f}")
        print(f"Spent/reserved: ${spent_so_far(ledger):.8f}; hard task cap: ${HARD_CAP_USD:.2f}")


@app.local_entrypoint()
def reconcile():
    """Attribute completed hourly billing to the one invocation in each worker app."""
    from modal.billing import workspace_billing_report

    with locked_ledger() as ledger:
        if not ledger["entries"]:
            print("No launches recorded; total $0.00")
            return
        cutoff = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = workspace_billing_report(start=datetime.fromisoformat(ledger["task_start_utc"]),
                                        resolution="h")
        groups = {}
        for entry in ledger["entries"]:
            if entry["modal_app_id"]:
                groups.setdefault(entry["modal_app_id"], []).append(entry)
        for app_id, entries in groups.items():
            if len(entries) != 1:
                raise RuntimeError("multiple runs share an app ID; cannot allocate its bill safely")
            entry = entries[0]
            matching = [r for r in rows if r["object_id"] == app_id]
            # Partial hours are excluded. Never replace a reservation with a partial/absent bill.
            stopped = entry.get("app_context_exit_utc")
            if not entry["end_utc"] or not stopped or not matching:
                continue
            final_hour = datetime.fromisoformat(stopped).replace(minute=0, second=0, microsecond=0)
            if final_hour + timedelta(hours=1) > cutoff:
                continue
            if max(r["interval_start"] for r in matching) < final_hour:
                continue
            entry["billed_usd"] = float(sum((money(r["cost"]) for r in matching), Decimal(0)))
            entry["billing_reconciled_utc"] = utc()
            entry["billing_rows"] = [{**r, "cost": str(r["cost"]),
                                      "interval_start": r["interval_start"].isoformat()}
                                     for r in matching]
        write_json(LEDGER, ledger)
        print("run id                            kind/model            ceiling      measured       billed")
        totals = [Decimal(0), Decimal(0), Decimal(0)]
        for e in ledger["entries"]:
            values = [e["ceiling_usd"], e["measured_cost_usd"], e["billed_usd"]]
            display = ["pending" if v is None else f"${v:.8f}" for v in values]
            for i, value in enumerate(values):
                totals[i] += money(value or 0)
            print(f"{e['run_id']}  {e['kind'] + '/' + e['model_key']:18} " + "  ".join(display))
        print("TOTAL (known amounts): " + "  ".join(f"${v:.8f}" for v in totals))
        print(f"Cap accounting, including pending reservations: ${spent_so_far(ledger):.8f}")
        print("Pending bills/reservations stay pending; rerun after the final full hour is reported.")
