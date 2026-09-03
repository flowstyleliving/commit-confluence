"""Per-layer depth-curve extractor (Modal) — exploratory lane, pre-registered.

See PRE_REGISTRATION.md (frozen before any run). Per prompt: ONE t=0 forward on the
chat-templated prompt; for EVERY decoder block read the last-query attention row and
compute the four weight-only attention cells via the SEALED kernel
(pri_calibrator._compute_attention_score), presenting each block's weights under tag
"final". No commit forward, no readout, no calibration — raw per-layer scalars only.

Reuse policy: this file ships the June extractor (modal/modal_app.py) into the image as
a module `cc_modal_app` and calls its _import_seal/_load/_chat_ids/_Capture/_forward
verbatim — single source of truth, no copy drift. The seal dir is mounted at BOTH
/seal (for cc_modal_app's sys.path) and /pkg/seal (in case of eager path validation).

DISCIPLINE (registered): this extractor prints gates/progress ONLY — it never computes
or prints an AUROC. Scoring happens locally in depth_score.py after all cells finish.

COMPARABILITY: torch nf4, NON-byte-comparable; never pool with sealed cells.

Run (one cell):
  modal run exploratory/depth-curve/modal_depth.py --model-id Qwen/Qwen2.5-7B-Instruct --task anli_r1

--out-dir selects the output namespace and DEFAULTS TO THE UNREGISTERED TREE
(`depth_curve_vnorm`). The registered grid-A tree `depth_curve` holds the banked
artifacts behind every depth result since 2026-08-16; targeting it requires passing
it explicitly, and even then the no-overwrite guard refuses an existing npz.

Additive v-norm channel (unregistered, candidate #16): every block's value-vector
norms are captured via v_proj hooks and scored into a SEPARATE `<slug>.vnorm.npz`
sidecar. The registered `<slug>.depth.npz` keeps exactly its original eight arrays.
"""
import os
from pathlib import Path

import modal

APP_NAME = "cc-depth-curve"
VOL_NAME = "model-cache"
MNT = "/models"
SEAL_REMOTE = "/seal"
PKG_REMOTE = "/pkg"
# ── output namespaces ───────────────────────────────────────────────────────────
# `depth_curve` is the REGISTERED grid-A tree. Its eight banked `.depth.npz` files
# are the inputs to every depth result since 2026-08-16 (RESULTS.json,
# RESCORE_GRID_A.json, COLOCATION.json, REDUNDANCY.json, DEPTH_COVERAGE.json) and
# overwriting one would destroy them. Grid A has no terminal-status guard, so the
# protection here is (a) a NEW default namespace and (b) the no-overwrite check in
# depth_extract. Pass --out-dir depth_curve explicitly to target the frozen tree.
OUT_DIR_REGISTERED = "depth_curve"
# Default for new runs: candidate #16 is unregistered descriptive work and writes
# to its own namespace rather than amending the grid-A pre-registration.
OUT_DIR_DEFAULT = "depth_curve_vnorm"
# Memory-safe bound for THIS lane (Codex audit MINOR-6): all blocks' bf16 attention
# weights are materialized, ~n_layers*H*T^2*2 bytes; at 80x64 and T=900 that is ~8.3GiB,
# fine beside nf4 weights on an 80GB card. Observed wrapped prompts are <650 tokens.
DEPTH_MAX_TOKENS = 900

# Frozen data provenance (PRE_REGISTRATION.md) — ENFORCED, not just recorded
# (Codex audit MAJOR-1): the mounted volume file must hash to the registered value.
FROZEN_DATA_SHA256 = {
    "anli_r1": "57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b",
    "halueval_qa": "a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0",
}

# The four weight-only attention cells (sealed panel details, t=0). v-norm cells are
# excluded by design: per-layer v-hooks are unnecessary for the depth question and
# v_norm_captures=None makes the sealed kernel return None for them.
DEPTH_CELL_DETAILS = ("final_js", "final_js_no_bos", "final_js_kv_groups", "final_bos_mass")

# ── ADDITIVE v-norm channel (added 2026-09-01; candidate #16 gating build step) ──
# The registered channel above is UNCHANGED: `scores` keeps shape [n_rows, n_layers, 4],
# `metrics` keeps the frozen 4-list, and the four cells are still computed with
# v_norm_captures=None on the untouched code path. This adds a SEPARATE, PARALLEL
# output channel `v_norm_scores` of shape [n_rows, n_layers, 3], written to its OWN
# `<slug>.vnorm.npz` sidecar, so the third ACE attention instrument finally gets
# per-layer depth coverage.
#
# HARD INVARIANT: nothing in the additive path may raise into the registered path,
# and nothing additive is serialized by the registered `np.savez` call. Every
# failure mode degrades to NaN in `v_norm_scores` + a diagnostic counter; a depth
# cell that would have succeeded before must still succeed, with an identical
# registered array set.
#
# ORDER IS THE SEALED ORDER: pri_calibrator.ATTENTION_METRICS_V_NORMS is
# ("v_norm_bos", "v_norm_max", "v_norm_lastq_weighted"), and the trailing axis of
# `v_norm_scores` follows it exactly. `_run_extract` re-derives this tuple from the
# sealed constant at runtime and disables the channel on any mismatch, so a typo in
# a detail key cannot silently produce an all-NaN column.
# CONSUMERS MUST INDEX BY NAME via the `v_norm_metrics` array, never by position:
# index 0 is v_norm_bos, NOT the headline lastq_weighted metric.
#
# All three reduce the SAME captured (n_kv, T) norms, so adding the two extra
# metrics costs no extra capture and no extra forward — only two more calls to the
# sealed scorer per (row, block). Confirmed against
# diagnose_inter_head_disagreement._mean_v_norm_bos (:491) and _mean_v_norm_max
# (:503): both take `v_norms` alone.
ADDITIVE_DEPTH_CELL_DETAILS = (
    "final_v_norm_bos",
    "final_v_norm_max",
    "final_v_norm_lastq_weighted",
)

# `v_norm_capture_mode` in REGISTERED meta is drawn from this closed set of two
# module-level literals and nothing else. Free-form disable/error text (which is
# runtime-derived, unbounded, and can embed an arbitrary exception message) goes
# ONLY to the sidecar's `additive_capture_diagnostics.reason`. This keeps the
# registered meta's value schema fixed and its serialization risk nil.
_VNORM_MODE_ON = "per_block_v_proj_hook"
_VNORM_MODE_OFF = "disabled"

# ── INERT SENTINELS, allocated ONCE at import ───────────────────────────────────
# Every additive fallback is a REBIND to one of these, never a fresh literal.
# `MemoryError` is an `Exception`, so an unguarded `{}` or `[]` in a failure
# handler can raise from inside the handler and escape — in grid B that would
# permanently burn a claimed cell. Constructing the fallback is the thing that
# fails, so the fallback must already exist. NOTHING may mutate these.
_VNORM_EMPTY_MAP = {}          # read-only stand-in for a per-row capture dict
_VNORM_EMPTY_SEQ = ()          # read-only stand-in for add_cells / handle lists
_VNORM_DIAG_FALLBACK = {"enabled": False, "mode": _VNORM_MODE_OFF,
                        "diagnostics_error": "diagnostics unavailable"}

_HERE = Path(__file__).parent
_MODAL_DIR = (_HERE / ".." / ".." / "modal").resolve()

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0", "transformers>=4.46.0", "accelerate>=0.34.0", "numpy<2",
        "pandas", "scikit-learn>=1.3", "scipy", "sentencepiece", "safetensors",
        "huggingface_hub", "hf_transfer", "datasets>=2.20.0", "bitsandbytes",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{MNT}/hub",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(_MODAL_DIR / "seal"), SEAL_REMOTE, copy=True,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
    .add_local_dir(str(_MODAL_DIR / "seal"), f"{PKG_REMOTE}/seal", copy=True,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
    .add_local_file(str(_MODAL_DIR / "modal_app.py"), f"{PKG_REMOTE}/cc_modal_app.py", copy=True)
)

GPU_CONFIG = "A100-80GB"


def _oproj_cos_gate(BASE, model, tok, model_id, prompt, tags, final_idx):
    """Replicates the June validate() o_proj reconstruction gate for one prompt.
    Returns (cos, commit_is_yes_no, commit_repr)."""
    import numpy as np
    import torch

    cap = BASE._Capture(model, tags, final_idx)
    try:
        ids = BASE._chat_ids(tok, prompt, model_id)
        logits, caps, vcaps, nkv, h = BASE._forward(model, ids, cap, tags)
        p = np.exp(logits - logits.max()); p /= p.sum()
        gid = int(np.argmax(p))
        commit = tok.decode([gid], skip_special_tokens=True).strip().upper()
        fa = model.model.layers[final_idx].self_attn
        oproj_device = fa.o_proj.weight.device
        oproj_dtype = getattr(fa.o_proj, "compute_dtype", None) or fa.o_proj.weight.dtype
        def is_floating_dtype(dtype):
            return isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point()
        if not is_floating_dtype(oproj_dtype):
            oproj_dtype = getattr(model, "dtype", torch.float32)
        if not is_floating_dtype(oproj_dtype):
            oproj_dtype = torch.float32
        w = torch.as_tensor(caps["final"][0], device=oproj_device, dtype=torch.float32)
        vv = torch.as_tensor(cap.values["final"], device=oproj_device, dtype=torch.float32)
        H = w.shape[0]; nrep = H // vv.shape[0]
        if H % vv.shape[0] != 0 or w.shape[1] != vv.shape[1]:
            raise RuntimeError(f"o_proj reconstruction shape mismatch: w={tuple(w.shape)} v={tuple(vv.shape)}")
        v_rep = vv.repeat_interleave(nrep, dim=0)
        ctx = torch.einsum("ht,htd->hd", w, v_rep).reshape(1, -1).to(dtype=oproj_dtype)
        my_out = fa.o_proj(ctx)[0].detach().float().cpu().numpy()
        real_out = cap.attn_out["final"]
        cos = float(np.dot(my_out, real_out) /
                    (np.linalg.norm(my_out) * np.linalg.norm(real_out) + 1e-9))
        return cos, commit in ("YES", "NO"), repr(tok.decode([gid]))
    finally:
        cap.remove()


# ── additive v-norm capture helpers (never raise into the registered path) ──────
def _vnorm_new_state():
    """Mutable state for the additive channel: GPU norm store, per-block fire
    counts, hook errors, and scoring diagnostics.

    Counter semantics: the per-(row, block) CAPTURE checks are counted once per
    block; only `ok` / `nonfinite` / `score_error` are per-(row, block, metric).
    `missing` (hook never fired), `multi_fire` (fired 2+ times) and `hook_failed`
    (fired but the body errored) are DISTINCT buckets — the previous version
    tested `vn is None` before the fire count, so a double-fire that also failed
    capture was misreported as `missing`.
    """
    return {
        "store": {},        # block index -> torch tensor [n_kv, T] fp32 (GPU, current row)
        "counts": {},       # block index -> hook fire count for the current row
        "errs": {},         # block index -> first hook error string (whole run)
        "global_errs": [],  # non-block-scoped failures (drain / removal / sidecar)
        "diag": {
            # ── CAPTURE outcome, exactly ONE per (row, block) ─────────────────
            # Mutually exclusive by construction: _vnorm_block_norms increments
            # exactly one of these and returns immediately, and the call site
            # never re-enters the capture phase for the same block. So
            # blocks_ok + missing + multi_fire + hook_failed + shape_mismatch
            # + capture_error == n_rows * n_layers whenever the channel is on.
            "blocks_ok": 0, "missing": 0, "multi_fire": 0, "hook_failed": 0,
            "shape_mismatch": 0, "capture_error": 0,
            # ── POST-CAPTURE failure, counted SEPARATELY ──────────────────────
            # A block can be captured successfully and still fail while scoring
            # or assigning. That is not a capture failure and must not be added
            # to n_blocks_failed, or captured + failed would exceed expected.
            "post_capture_error": 0,
            # ── per-(row, block, METRIC) ──────────────────────────────────────
            "ok": 0, "nonfinite": 0, "score_error": 0,
            # ── sidecar row-identity mirror ───────────────────────────────────
            "mirror_error": 0,
            "first_capture_error": None, "first_post_capture_error": None,
            "first_score_error": None, "first_mirror_error": None,
        },
    }


def _vnorm_exc_text(exc):
    """Format an exception with a CONSTANT fallback.

    `f"{exc}"` calls the exception's own `__str__`, which is arbitrary code and
    can itself raise (or raise again while being formatted). Every additive
    error string in this file is produced here, so a hostile `__str__` degrades
    to a constant instead of escaping into the registered path.
    """
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001
        try:
            return str(type(exc).__name__)
        except Exception:  # noqa: BLE001
            return "<unformattable exception>"


def _vnorm_note(state, counter=None, first_key=None, exc=None, text=None,
                block=None):
    """The ONE diagnostic sink for the additive path. Structurally non-raising.

    Every recording site routes through here: hook-body failures, drain
    failures, capture failures, post-capture failures, scoring failures, mirror
    failures, sidecar failures and hook-removal failures. Each of the three
    steps below (format / count / store) is independently guarded with a
    constant fallback, so no combination of a broken exception, a mangled state
    dict or a full container can produce an exception at a call site.
    """
    try:
        msg = text if text is not None else _vnorm_exc_text(exc)
        msg = str(msg)[:400]
    except Exception:  # noqa: BLE001
        msg = "<unformattable exception>"
    try:
        d = state["diag"]
        if counter is not None and counter in d:
            d[counter] += 1
        if first_key is not None and d.get(first_key) is None:
            d[first_key] = msg
    except Exception:  # noqa: BLE001
        pass
    try:
        if block is not None:
            state["errs"].setdefault(block, msg)
        elif first_key is None:
            state["global_errs"].append(msg)
    except Exception:  # noqa: BLE001
        pass


def _claim_cell_exclusive(json, claim_path, payload):
    """Atomically reserve one (out_dir, task, model) cell.

    `os.open(..., O_CREAT | O_EXCL)` is a single atomic syscall: exactly one of
    two concurrent invocations can create the file, so this closes the
    check-then-write race that `os.path.exists(npz)` alone cannot. The claim is
    STICKY — it is never removed on failure, matching the lane's fail-closed
    discipline, and it is rewritten to state="complete" once the registered
    artifact is on disk.

    Returns (ok, reason). Callers must treat ok=False as fatal BEFORE loading a
    model; this is deliberately not fail-soft, because its whole purpose is to
    stop a second writer.
    """
    fd = None
    try:
        os.makedirs(os.path.dirname(claim_path), exist_ok=True)
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = "<unreadable>"
        try:
            with open(claim_path) as f:
                existing = f.read()[:600]
        except Exception:  # noqa: BLE001
            pass
        return False, existing
    except Exception as exc:  # noqa: BLE001
        return False, f"claim create failed: {_vnorm_exc_text(exc)}"
    try:
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
    except Exception:  # noqa: BLE001
        pass  # the claim's EXISTENCE is the lock; its contents are advisory
    finally:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass
    return True, ""


def _claim_mark_complete(json, claim_path, payload):
    """Rewrite a held claim to state="complete". Best-effort: by the time this
    runs the registered artifact already exists, and the npz is the real
    completion evidence, so a failure here is recorded and ignored."""
    try:
        with open(claim_path + ".tmp", "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(claim_path + ".tmp", claim_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _vnorm_install_hooks(layers, n_kv, head_dim_expected, state):
    """Install observational forward hooks on EVERY block's `self_attn.v_proj`.

    Each hook reshapes the v_proj output `[1, T, n_kv*head_dim]` to
    `[T, n_kv, head_dim]`, casts to FP32 *before* squaring (sealed
    `_capture_value_norms` rationale: bf16/fp16 exponent range clips large value
    vectors), takes the L2 norm over `head_dim`, and stores ONLY the resulting
    `(n_kv, T)` tensor — the full V tensor is never retained. Norms stay on the
    GPU (~n_kv*T*4 bytes per block, ~3.6 MB total at 126 blocks / 8 KV / T=900)
    and are drained to numpy once per forward, so no hook forces a mid-forward
    GPU->CPU sync.

    The `[T, n_kv, head_dim]` (head-major) reshape is the SAME one the o_proj
    faithfulness gate already uses to rebuild the attention output from V — grid A
    via modal_app._Capture, grid B via _oproj_cos_gate_b's v_out.view(T, n_kv,
    d_head). That reconstruction is gated at cos >= 0.999 on rows 0-1 of every
    cell, so a wrong V layout would fail the gate before extraction ever starts.

    Returns `(handles, reason)`. `reason` is "" on success; a non-empty reason
    means NO hooks were installed and the additive channel must stay disabled.
    This function performs its structural pre-flight BEFORE installing anything
    and NEVER raises — a structural surprise disables the additive channel
    rather than aborting a registered extraction.
    """
    try:
        import torch  # bound for the closure below; inside the guard on purpose

        for li, blk in enumerate(layers):
            attn = getattr(blk, "self_attn", None)
            if attn is None:
                return [], f"block {li}: no .self_attn"
            vp = getattr(attn, "v_proj", None)
            if vp is None:
                return [], (f"block {li}: self_attn has no .v_proj "
                            f"(fused-QKV or non-standard attention layout)")
            # Mirror modal_app._Capture's refusal: raw v_proj must BE the value
            # tensor attention consumes (false for value-norm / K==V families).
            if hasattr(attn, "v_norm") or getattr(attn, "use_k_eq_v", False):
                return [], (f"block {li}: attention has v_norm/use_k_eq_v — raw "
                            f"v_proj capture would be unfaithful")
            of = getattr(vp, "out_features", None)
            if not isinstance(of, int) or of <= 0:
                return [], f"block {li}: v_proj.out_features unusable ({of!r})"
            if of % n_kv != 0:
                return [], (f"block {li}: v_proj out_features {of} not divisible "
                            f"by n_kv={n_kv}")
            if head_dim_expected is not None and (of // n_kv) != head_dim_expected:
                return [], (f"block {li}: derived head_dim {of // n_kv} != config "
                            f"head_dim {head_dim_expected}")
        store, counts = state["store"], state["counts"]
    except Exception as exc:  # noqa: BLE001
        return [], f"preflight {_vnorm_exc_text(exc)}"

    def _make(li):
        def _hook(_mod, _inp, out):
            # EVERY statement is inside the try. A hook that raises does so inside
            # the registered forward, where grid B's _extract_body would convert it
            # into a PERMANENT `aborted` terminal status. The fire count is the
            # first statement inside the try, so double-fire detection survives:
            # a block whose v_proj fires twice in one forward is ambiguous data and
            # must be dropped, not averaged.
            try:
                counts[li] = counts.get(li, 0) + 1
                v = out[0] if isinstance(out, tuple) else out
                if v.ndim != 3 or int(v.shape[0]) != 1:
                    raise RuntimeError(
                        f"v_proj output {tuple(v.shape)} != [1, T, n_kv*head_dim]")
                t_len, width = int(v.shape[1]), int(v.shape[2])
                if width % n_kv != 0:
                    raise RuntimeError(
                        f"v_proj width {width} not divisible by n_kv={n_kv}")
                hd = width // n_kv
                x = v.detach().float().reshape(t_len, n_kv, hd)
                nrm = torch.linalg.vector_norm(x, dim=-1)      # [T, n_kv] fp32
                store[li] = nrm.transpose(0, 1).contiguous()   # [n_kv, T]
                del x, nrm
            except Exception as exc:  # noqa: BLE001 — must never perturb the forward
                # Through the sink, never a bare setdefault: formatting `exc`
                # here would be the last unguarded statement inside the
                # registered forward.
                _vnorm_note(state, block=li, exc=exc)
        return _hook

    handles = []
    try:
        for li, blk in enumerate(layers):
            handles.append(blk.self_attn.v_proj.register_forward_hook(_make(li)))
    except Exception as exc:  # noqa: BLE001
        _vnorm_remove_hooks(handles, state)
        return [], f"register {_vnorm_exc_text(exc)}"
    return handles, ""


def _vnorm_remove_hooks(handles, state):
    """Best-effort hook removal. A RemovableHandle that refuses to detach must not
    take a finished extraction down with it."""
    try:
        for h in handles:
            try:
                h.remove()
            except Exception as exc:  # noqa: BLE001
                _vnorm_note(state, exc=exc)
    except Exception:  # noqa: BLE001
        pass


def _vnorm_row_reset(state):
    """Drop the previous row's captures so a hook that fails to fire is detected
    as missing rather than silently reusing a stale row (cross-row bleed)."""
    try:
        state["store"].clear()
        state["counts"].clear()
    except Exception:  # noqa: BLE001
        pass


def _vnorm_drain(state):
    """Move this forward's per-block norms to CPU numpy and release the GPU
    buffers. Returns {block index: np.ndarray (n_kv, T) float32}.

    Transfers are BATCHED PER DEVICE: every block on one device has the same
    (n_kv, T) shape within a row, so they stack into a single D2H copy. The naive
    per-block version issued n_layers copies per row (25,200 for a 126-block model
    at 200 rows) instead of one per device per row (200, or 1,600 on 8xA100).
    Falls back to per-block copies for a device whose stack fails. Never raises —
    including under MemoryError: `out = {}` is an ALLOCATION and therefore lives
    inside the try, with the import-time `_VNORM_EMPTY_MAP` as the return-path
    fallback so no failure handler ever has to allocate.
    """
    out = None
    store = None
    try:
        import torch

        out = {}
        store = state["store"]
        by_dev = {}
        for li, t in store.items():
            by_dev.setdefault(str(getattr(t, "device", "cpu")), []).append(li)
        for dev, lis in by_dev.items():
            try:
                stacked = torch.stack([store[li] for li in lis], dim=0).cpu().numpy()
                for j, li in enumerate(lis):
                    out[li] = stacked[j]
            except Exception as exc:  # noqa: BLE001 — ragged shapes / OOM on this device
                _vnorm_note(state, exc=exc,
                            text=f"drain[{dev}] {_vnorm_exc_text(exc)} "
                                 f"(fell back per-block)")
                for li in lis:
                    try:
                        out[li] = store[li].cpu().numpy()
                    except Exception as exc2:  # noqa: BLE001
                        _vnorm_note(state, block=li, exc=exc2)
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, exc=exc)
    finally:
        try:
            if store is not None:
                store.clear()
        except Exception:  # noqa: BLE001
            pass
    return out if out is not None else _VNORM_EMPTY_MAP


def _vnorm_block_norms(v_by_block, li, n_kv, T, state):
    """Validate ONE block's captured norms, once per (row, block). Returns the
    (n_kv, T) array or None; never raises.

    The fire count is tested BEFORE `vn is None` so the three failure modes stay
    distinct: never-fired (`missing`), fired-twice (`multi_fire`), fired-but-body-
    errored (`hook_failed`).
    """
    try:
        diag = state["diag"]
        c = int(state["counts"].get(li, 0))
        if c == 0:
            diag["missing"] += 1
            return None
        if c != 1:
            diag["multi_fire"] += 1
            return None
        vn = v_by_block.get(li)
        if vn is None:
            diag["hook_failed"] += 1
            return None
        if vn.ndim != 2 or int(vn.shape[0]) != n_kv or int(vn.shape[1]) != T:
            diag["shape_mismatch"] += 1
            return None
        diag["blocks_ok"] += 1
        return vn
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="capture_error",
                    first_key="first_capture_error", exc=exc)
        return None


def _vnorm_score_cell(SEAL, np, cell, caps, nkv_map, vn, state):
    """Score one additive v-norm cell. Returns a float or None; never raises.

    `caps` is the SAME weights dict the registered cells were scored from, so the
    additive cell reads exactly the same attention row. `v_norm_bos` and
    `v_norm_max` ignore `caps` entirely (they reduce v_norms alone) but the sealed
    kernel still fetches `captures[layer][step]` before dispatching on metric, so
    the weights dict must be present for all three.
    """
    try:
        diag = state["diag"]
        sc = SEAL._compute_attention_score(cell, caps, nkv_map,
                                           v_norm_captures={"final": [vn]})
        if sc is None or not np.isfinite(sc):
            diag["nonfinite"] += 1
            return None
        diag["ok"] += 1
        return float(sc)
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="score_error",
                    first_key="first_score_error", exc=exc)
        return None


def _vnorm_meta(state, mode, reason, enabled, n_rows, n_layers, n_metrics):
    """Build the additive diagnostics block. Returns only JSON primitives and
    never raises — on internal failure it returns a minimal explanatory dict.

    `mode` is the stable literal that also goes into registered meta; `reason`
    is the free-form disable/error text and lives ONLY here.
    """
    try:
        d = state["diag"]
        first_hook_err = None
        if state["errs"]:
            k = sorted(state["errs"])[0]
            first_hook_err = f"block {k}: {state['errs'][k]}"
        attempted = int(d["ok"] + d["nonfinite"] + d["score_error"])
        # CAPTURE buckets only — post_capture_error is NOT summed in here, or a
        # block could be counted as both captured and failed.
        blocks_failed = int(d["missing"] + d["multi_fire"] + d["hook_failed"]
                            + d["shape_mismatch"] + d["capture_error"])
        blocks_attempted = int(d["blocks_ok"]) + blocks_failed
        return {
            "enabled": bool(enabled),
            "hook_target": "self_attn.v_proj",
            "capture_shape": "(n_kv_heads, T) fp32 L2 norms over head_dim",
            "mode": str(mode),
            "reason": str(reason),
            # Full-coverage denominator, independent of whether the channel ran.
            # When the channel is disabled every counter below is 0 and this stays
            # at full size, so 0/N reads unambiguously as "captured nothing".
            "n_metric_cells_expected": int(n_rows) * int(n_layers) * int(n_metrics),
            "n_metric_cells_attempted": attempted,
            "n_scored": int(d["ok"]),
            "n_nonfinite": int(d["nonfinite"]),
            "n_score_errors": int(d["score_error"]),
            "n_blocks_expected": int(n_rows) * int(n_layers),
            # INVARIANT a consumer can check: when enabled,
            #   n_blocks_attempted == n_blocks_expected
            #   n_blocks_captured + n_blocks_failed == n_blocks_attempted
            "n_blocks_attempted": blocks_attempted,
            "n_blocks_captured": int(d["blocks_ok"]),
            "n_blocks_failed": blocks_failed,
            "n_missing": int(d["missing"]),
            "n_multi_fire": int(d["multi_fire"]),
            "n_hook_failed": int(d["hook_failed"]),
            "n_shape_mismatch": int(d["shape_mismatch"]),
            "n_capture_errors": int(d["capture_error"]),
            # Disjoint from every capture bucket above; these blocks WERE
            # captured and then failed downstream.
            "n_post_capture_errors": int(d["post_capture_error"]),
            "n_mirror_errors": int(d["mirror_error"]),
            "n_blocks_with_hook_errors": len(state["errs"]),
            "first_hook_error": first_hook_err,
            "first_capture_error": d["first_capture_error"],
            "first_post_capture_error": d["first_post_capture_error"],
            "first_score_error": d["first_score_error"],
            "first_mirror_error": d["first_mirror_error"],
            "global_errors": [str(e) for e in state["global_errs"][:5]],
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": bool(enabled), "mode": str(mode),
                "diagnostics_error": _vnorm_exc_text(exc)}


def _vnorm_build_mirror(np, state, labels, gen_token_ids, commit_p, yes_no):
    """Copy the registered row-identity columns for the sidecar (MK decision,
    2026-09-01). Never raises: a column that cannot be built is omitted and
    recorded, and the sidecar is written without it.

    These four columns are already in memory at sidecar-write time, so this adds
    no compute and no new failure mode. Dtypes deliberately match the registered
    write EXACTLY (`labels`/`gen_token_ids`/`yes_no` int64, `commit_p` float64)
    so a consumer can assert bitwise equality against the registered npz.

    Purpose: `sample_idx` is `arange(200)` in every banked file and therefore
    cannot detect a row permutation — a guard that fires everywhere and proves
    nothing. Within a cell these four columns ARE a content fingerprint.
    """
    out = None
    try:
        out = {}   # allocation: inside the guard, like every other one
        for name, src, dtype in (("labels", labels, "int64"),
                                 ("gen_token_ids", gen_token_ids, "int64"),
                                 ("commit_p", commit_p, "float64"),
                                 ("yes_no", yes_no, "int64")):
            try:
                out[name] = np.asarray(src).astype(dtype, copy=True)
            except Exception as exc:  # noqa: BLE001
                _vnorm_note(state, counter="mirror_error",
                            first_key="first_mirror_error", exc=exc,
                            text=f"mirror[{name}] {_vnorm_exc_text(exc)}")
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="mirror_error",
                    first_key="first_mirror_error", exc=exc)
    return out if out is not None else _VNORM_EMPTY_MAP


def _vnorm_write_sidecar(np, json, out_path, v_norm_scores, mirror, n_rows,
                         vmeta, state):
    """Serialize the additive channel to its OWN npz, in its OWN call, AFTER the
    registered artifact is already on disk.

    This is the isolation the registered write requires: the additive arrays no
    longer ride inside the same `np.savez` as `scores` / `labels` / `sample_idx` /
    `gen_token_ids` / `commit_p` / `yes_no` / `metrics` / `meta`. A malformed
    additive array can no longer destroy a complete registered artifact at the
    final write — on the 405B cell that would have been ~40 min of 8xA100 lost
    after all compute was already paid for.

    `mirror` carries the row-identity columns copied from the registered arrays
    (see _vnorm_build_mirror); it may be empty or partial, and the sidecar is
    written either way.

    Atomic (tmp + os.replace), same discipline as the registered write. Never
    raises. Returns (path_or_None, reason) where reason is ALWAYS a plain str
    produced by _vnorm_exc_text — no `f"...{exc}"` outside a guard.
    """
    tmp = None
    try:
        tmp = str(out_path) + ".tmp.npz"
        if v_norm_scores is None:
            return None, "additive array was never allocated"
        arrays = {
            "v_norm_scores": np.asarray(v_norm_scores, dtype=np.float64),
            "v_norm_metrics": json.dumps(list(ADDITIVE_DEPTH_CELL_DETAILS)),
            "sample_idx": np.arange(int(n_rows), dtype=np.int64),
            "meta": json.dumps(vmeta),
        }
        # Row-identity mirror. Merged LAST but cannot shadow the keys above.
        try:
            for k, v in (mirror or _VNORM_EMPTY_MAP).items():
                if k not in arrays:
                    arrays[k] = v
        except Exception as exc:  # noqa: BLE001
            _vnorm_note(state, counter="mirror_error",
                        first_key="first_mirror_error", exc=exc)
        np.savez(tmp, **arrays)
        os.replace(tmp, out_path)
        return out_path, ""
    except Exception as exc:  # noqa: BLE001
        reason = "<unformattable exception>"
        try:
            reason = _vnorm_exc_text(exc)
        except Exception:  # noqa: BLE001
            pass
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:  # noqa: BLE001
            pass
        _vnorm_note(state, text=f"sidecar {reason}")
        return None, reason


@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60 * 6)
def depth_extract(model_id: str, task: str, n: int = 200, precision: str = "nf4",
                  code_commit: str = "", out_dir: str = OUT_DIR_DEFAULT):
    import hashlib
    import json
    import sys
    import time

    import numpy as np
    import torch

    if PKG_REMOTE not in sys.path:
        sys.path.insert(0, PKG_REMOTE)
    import cc_modal_app as BASE  # June extractor, imported as a module (not run)

    SEAL, PIPE, CR, CC, io_plugins, target_layer_map = BASE._import_seal()

    # The four depth cells, taken from the SEALED panel by detail name (exact tuples).
    by_detail = {c[2]: tuple(c) for c in SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS}
    missing = [d for d in DEPTH_CELL_DETAILS if d not in by_detail]
    if missing:
        raise RuntimeError(f"sealed panel is missing expected cells: {missing}")
    cells = [by_detail[d] for d in DEPTH_CELL_DETAILS]

    # ADDITIVE channel: resolve the v-norm cell(s) from the SAME sealed panel.
    # A miss here disables the additive channel; it never blocks the registered run.
    # `v_norm_mode` is ALWAYS one of the two module literals (registered meta
    # reads it verbatim). `v_norm_reason` carries the free-form explanation and
    # is written to the SIDECAR only.
    add_cells, v_norm_mode, v_norm_reason = _VNORM_EMPTY_SEQ, _VNORM_MODE_ON, ""
    try:
        sealed_v = tuple(f"final_{m}" for m in SEAL.ATTENTION_METRICS_V_NORMS)
    except Exception as _vexc:  # noqa: BLE001
        sealed_v = None
    add_missing = [d for d in ADDITIVE_DEPTH_CELL_DETAILS if d not in by_detail]
    if add_missing:
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = f"sealed panel missing additive cells {add_missing}"
    elif sealed_v is None:
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = "could not read sealed ATTENTION_METRICS_V_NORMS"
    elif sealed_v != tuple(ADDITIVE_DEPTH_CELL_DETAILS):
        # The trailing axis of v_norm_scores is positional; if the sealed metric
        # order ever drifts from this file's literal, disable rather than emit a
        # correctly-shaped array whose columns mean something else.
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = (f"additive order {list(ADDITIVE_DEPTH_CELL_DETAILS)} "
                         f"!= sealed order {list(sealed_v)}")
    else:
        add_cells = [by_detail[d] for d in ADDITIVE_DEPTH_CELL_DETAILS]

    data = f"{MNT}/data/{task}_n{n}.jsonl"
    if not os.path.exists(data):
        raise FileNotFoundError(f"missing data file on volume: {data}")
    data_sha = hashlib.sha256(open(data, "rb").read()).hexdigest()
    frozen = FROZEN_DATA_SHA256.get(task)
    if frozen is None:
        raise RuntimeError(f"task {task!r} has no frozen data hash in this lane — refusing to run")
    if data_sha != frozen:
        raise RuntimeError(f"DATA HASH MISMATCH for {task}: volume file {data_sha} != frozen {frozen}")
    prompts, labels, dh = CR._load_calibration_jsonl(data)
    if len(prompts) != n:
        raise RuntimeError(f"expected n={n} rows, data file has {len(prompts)}")

    # ── output namespace + EXCLUSIVE CLAIM (both before the model is loaded) ──────
    # Grid A has no terminal-status mechanism, so nothing else stops a rerun from
    # clobbering a banked artifact. This is a new protection, not a relaxed one.
    slug = model_id.split("/")[-1]
    outdir = f"{MNT}/{out_dir}/{task}"
    npz_path = f"{outdir}/{slug}.depth.npz"
    claim_path = f"{outdir}/{slug}.claim.json"

    # Barrier 1 — cheap existence check. NOT sufficient on its own: two concurrent
    # invocations can both pass it (check-then-write TOCTOU).
    if os.path.exists(npz_path):
        raise RuntimeError(
            f"REFUSING TO OVERWRITE {npz_path}. Grid A's banked "
            f"{OUT_DIR_REGISTERED!r} artifacts are the inputs to every depth result "
            f"since 2026-08-16 (RESULTS.json, RESCORE_GRID_A.json, COLOCATION.json, "
            f"REDUNDANCY.json, DEPTH_COVERAGE.json). Pick a fresh --out-dir "
            f"(default {OUT_DIR_DEFAULT!r}).")

    # Barrier 2 — ATOMIC claim. os.open(O_CREAT|O_EXCL) is one syscall, so exactly
    # one of any number of concurrent invocations creates the file and the rest
    # fail. This is what actually closes the race; barrier 1 only gives a nicer
    # message in the common case.
    claim_payload = {
        "state": "in_progress", "model": model_id, "task": task, "out_dir": out_dir,
        "n": n, "precision": precision, "code_commit": code_commit or "<not passed>",
        "claimed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claimed_at_unix": time.time(), "pid": os.getpid(),
        "note": "STICKY LOCK. Never auto-removed. A claim left in state "
                "'in_progress' means the run was killed (spend limit, "
                "preemption, OOM); inspect it, confirm the process is dead, then "
                "delete this file to retry that cell. state='complete' means the "
                "npz beside it is final.",
    }
    ok, why = _claim_cell_exclusive(json, claim_path, claim_payload)
    if not ok:
        raise RuntimeError(
            f"CELL ALREADY CLAIMED: {claim_path} exists, so another invocation "
            f"holds this (out_dir, task, model). Existing claim:\n{why}\n"
            f"If that run is dead, delete the claim file to retry.")
    print(f"[depth] out_dir={out_dir!r} -> {npz_path} (claim held)", flush=True)

    tok, model, precision = BASE._load(model_id, False, precision)
    n_layers = len(model.model.layers)
    n_heads = int(model.config.num_attention_heads)
    n_kv = int(getattr(model.config, "num_key_value_heads", n_heads))
    tags = target_layer_map(n_layers)
    final_idx = n_layers - 1
    print(f"[depth] {model_id} {task} precision={precision} layers={n_layers} "
          f"H={n_heads} kv={n_kv} cells={list(DEPTH_CELL_DETAILS)} n={len(prompts)}", flush=True)

    # ── faithfulness gate on rows 0-1 (cos>=0.999 + YES/NO), fail-closed ─────────
    gate = {"rows": []}
    for i in (0, 1):
        cos, is_yn, commit = _oproj_cos_gate(BASE, model, tok, model_id, prompts[i], tags, final_idx)
        gate["rows"].append({"row": i, "oproj_recon_cos": round(cos, 5),
                             "commit_is_yes_no": bool(is_yn), "commit_token": commit})
        print(f"[depth] gate row {i}: cos={cos:.5f} yes_no={is_yn} commit={commit}", flush=True)
    gate["GATE_cos_ok"] = all(r["oproj_recon_cos"] >= 0.999 for r in gate["rows"])
    gate["GATE_yes_no_ok"] = all(r["commit_is_yes_no"] for r in gate["rows"])
    if not (gate["GATE_cos_ok"] and gate["GATE_yes_no_ok"]):
        raise RuntimeError(f"depth gate FAILED: {json.dumps(gate)}")

    # ── main loop: one t=0 forward per prompt, every block's last-query row ──────
    n_rows = len(prompts)
    scores = np.full((n_rows, n_layers, len(cells)), np.nan, dtype=np.float64)
    gen_ids = np.zeros(n_rows, dtype=np.int64)
    commit_p = np.zeros(n_rows, dtype=np.float64)
    yes_no = np.zeros(n_rows, dtype=bool)

    # ── ADDITIVE channel: parallel array + per-block v_proj hooks ────────────────
    # Installed AFTER the faithfulness gate so the gate's own v_proj/o_proj hooks
    # run exactly as before. Observational only (hooks return None), so the
    # registered four metrics are numerically untouched.
    # PREBIND INERT SENTINELS FIRST. Every name below is bound to an object that
    # already exists (None, or an import-time constant), so these four statements
    # cannot allocate and cannot raise. The failure handlers that follow rebind to
    # these same objects — they never construct a fallback, because constructing
    # the fallback is precisely what fails under MemoryError.
    v_state = None
    v_handles = _VNORM_EMPTY_SEQ
    v_norm_scores = None
    v_by_block = _VNORM_EMPTY_MAP
    try:
        v_state = _vnorm_new_state()   # allocates several dicts — must be guarded
    except Exception:  # noqa: BLE001 — handler is REBIND-ONLY, no allocation
        add_cells = _VNORM_EMPTY_SEQ
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = "additive state allocation failed"
    try:
        v_norm_scores = np.full((n_rows, n_layers, len(ADDITIVE_DEPTH_CELL_DETAILS)),
                                np.nan, dtype=np.float64)
    except Exception as _vexc:  # noqa: BLE001
        add_cells, v_norm_mode = _VNORM_EMPTY_SEQ, _VNORM_MODE_OFF
        try:
            v_norm_reason = f"alloc {_vnorm_exc_text(_vexc)}"
        except Exception:  # noqa: BLE001
            v_norm_reason = "additive score array allocation failed"
    if add_cells:
        try:
            head_dim_expected = getattr(model.config, "head_dim", None)
            if not isinstance(head_dim_expected, int):
                head_dim_expected = None
            v_handles, reason = _vnorm_install_hooks(
                model.model.layers, n_kv, head_dim_expected, v_state)
            if reason:
                add_cells, v_norm_mode = _VNORM_EMPTY_SEQ, _VNORM_MODE_OFF
                v_norm_reason = str(reason)
        except Exception as _vexc:  # noqa: BLE001 — handler must not allocate
            _vnorm_remove_hooks(v_handles, v_state)
            v_handles = _VNORM_EMPTY_SEQ
            add_cells = _VNORM_EMPTY_SEQ
            v_norm_mode = _VNORM_MODE_OFF
            try:
                v_norm_reason = f"setup {_vnorm_exc_text(_vexc)}"
            except Exception:  # noqa: BLE001
                v_norm_reason = "additive hook setup failed"
    # Guarded like every other additive statement: a broken stdout must not be
    # able to abort a registered cell from inside the additive channel.
    try:
        if add_cells:
            print(f"[depth][vnorm] additive channel ON: "
                  f"{list(ADDITIVE_DEPTH_CELL_DETAILS)} via {len(v_handles)} "
                  f"v_proj hooks", flush=True)
        else:
            print(f"[depth][vnorm] additive channel OFF — {v_norm_reason}", flush=True)
    except Exception:  # noqa: BLE001
        pass

    for i, prompt in enumerate(prompts):
        try:
            _vnorm_row_reset(v_state)
        except Exception:  # noqa: BLE001 — caller-level guard, no allocation
            pass
        ids = BASE._chat_ids(tok, prompt, model_id)
        if len(ids) > DEPTH_MAX_TOKENS:
            raise RuntimeError(f"row {i}: prompt has {len(ids)} tokens > {DEPTH_MAX_TOKENS} "
                               f"(memory-safe bound for all-blocks attention capture)")
        with torch.no_grad():
            out = model(torch.tensor([ids], device=model.device),
                        output_attentions=True, use_cache=False)
        # Drain the additive GPU norm buffers immediately after the forward (one
        # batched transfer; no mid-forward sync). The disabled branch and the
        # failure branch both REBIND to the import-time empty map — neither
        # allocates a `{}`, which is itself a MemoryError site.
        try:
            v_by_block = _vnorm_drain(v_state) if add_cells else _VNORM_EMPTY_MAP
        except Exception:  # noqa: BLE001
            v_by_block = _VNORM_EMPTY_MAP
        if out.attentions is None or len(out.attentions) != n_layers:
            got = None if out.attentions is None else len(out.attentions)
            raise RuntimeError(f"row {i}: attentions missing/short (got {got}, want {n_layers})")
        logits = out.logits[0, -1].float().cpu().numpy().astype(np.float64)
        p = np.exp(logits - logits.max()); p /= p.sum()
        gid = int(np.argmax(p))
        gen_ids[i] = gid
        commit_p[i] = float(p[gid])
        yes_no[i] = tok.decode([gid], skip_special_tokens=True).strip().upper() in ("YES", "NO")

        T = len(ids)
        for li in range(n_layers):
            att = out.attentions[li]
            if att is None or len(att.shape) != 4:
                raise RuntimeError(f"row {i} block {li}: bad attention tensor")
            if int(att.shape[1]) != n_heads or int(att.shape[-1]) != T or int(att.shape[-2]) != T:
                raise RuntimeError(f"row {i} block {li}: shape {tuple(att.shape)} != [1,{n_heads},{T},{T}]")
            w_last = att[0, :, -1, :].float().cpu().numpy()
            caps = {"final": [w_last]}
            nkv_map = {"final": n_kv}
            for k, cell in enumerate(cells):
                sc = SEAL._compute_attention_score(cell, caps, nkv_map, v_norm_captures=None)
                if sc is None or not np.isfinite(sc):
                    raise RuntimeError(f"row {i} block {li} cell {cell}: non-finite score {sc!r}")
                scores[i, li, k] = float(sc)
            # Additive channel. Validate the block's capture ONCE, then score every
            # additive metric off it. The outer try is a structural guarantee that
            # no additive statement can raise into the registered loop, on top of
            # the helpers already being individually non-raising.
            if add_cells:
                # PHASE 1 — capture validation. Increments exactly one capture
                # bucket. Its own outer guard uses `capture_error`, the same
                # bucket _vnorm_block_norms uses internally, so a block can never
                # land in two capture buckets.
                vn = None
                try:
                    vn = _vnorm_block_norms(v_by_block, li, n_kv, T, v_state)
                except Exception as _vexc:  # noqa: BLE001 — unreachable; see report
                    _vnorm_note(v_state, counter="capture_error",
                                first_key="first_capture_error", exc=_vexc)
                # PHASE 2 — scoring + assignment. Failures here are POST-capture
                # and are counted in a disjoint bucket, so
                # n_blocks_captured + n_blocks_failed can never exceed
                # n_blocks_expected.
                if vn is not None:
                    try:
                        for k, vcell in enumerate(add_cells):
                            vsc = _vnorm_score_cell(SEAL, np, vcell, caps,
                                                    nkv_map, vn, v_state)
                            if vsc is not None:
                                v_norm_scores[i, li, k] = vsc
                    except Exception as _vexc:  # noqa: BLE001
                        _vnorm_note(v_state, counter="post_capture_error",
                                    first_key="first_post_capture_error",
                                    exc=_vexc)
        v_by_block = _VNORM_EMPTY_MAP   # release the row's captures; no allocation
        del att, out
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if i % 25 == 0:
            print(f"[depth]  {i}/{n_rows} yes_no_so_far={int(yes_no[:i + 1].sum())}", flush=True)

    # Prebound to the import-time constant, not a fresh dict: the pre-bind itself
    # was an unguarded allocation.
    v_norm_diag = _VNORM_DIAG_FALLBACK
    try:
        _vnorm_remove_hooks(v_handles, v_state)
        v_handles = _VNORM_EMPTY_SEQ
        _vnorm_row_reset(v_state)
        v_norm_diag = _vnorm_meta(v_state, v_norm_mode, v_norm_reason,
                                  bool(add_cells), n_rows, n_layers,
                                  len(ADDITIVE_DEPTH_CELL_DETAILS))
        json.dumps(v_norm_diag)  # serialization probe — must not fail at write time
        print(f"[depth][vnorm] {json.dumps(v_norm_diag)}", flush=True)
    except Exception as _vexc:  # noqa: BLE001
        # Try for an informative dict; fall back to the import-time constant if
        # even that allocation fails.
        try:
            v_norm_diag = {"enabled": bool(add_cells), "mode": v_norm_mode,
                           "diagnostics_error": _vnorm_exc_text(_vexc)}
        except Exception:  # noqa: BLE001
            v_norm_diag = _VNORM_DIAG_FALLBACK

    # The additive channel is deliberately EXCLUDED from this check: NaN there is a
    # recorded gap, never a reason to fail a registered extraction.
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite scores present after loop — should be unreachable")
    frac_yn = float(yes_no.mean())
    print(f"[depth] done: yes_no_rate={frac_yn:.2%}", flush=True)
    if frac_yn < 0.5:
        raise RuntimeError(f"YES/NO commit rate {frac_yn:.2%} < 50% — task not attempted; refusing to save")

    # `slug`, `outdir` and `npz_path` were fixed before the model load, together
    # with the no-overwrite guard; do not recompute them from a different variable.
    os.makedirs(outdir, exist_ok=True)
    # runtime provenance (Codex audit MINOR-5)
    import bitsandbytes
    import transformers
    def _sha_file(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta = {
        "schema": "furnace-depth-curve/1.0", "model": model_id, "task": task,
        "precision": precision, "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv,
        "n_rows": n_rows, "metrics": list(DEPTH_CELL_DETAILS),
        "backend": "modal-torch", "comparable": False,
        "data_path": data, "data_sha256": data_sha, "data_hash_loader": dh,
        "yes_no_commit_rate": round(frac_yn, 4), "gate": gate,
        "prereg": "exploratory/depth-curve/PRE_REGISTRATION.md",
        "note": "per-layer t=0 last-query attention; sealed kernel per block under tag 'final'",
        "provenance": {
            "extractor_code_commit": code_commit or "<not passed>",
            "hf_model_revision": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "numpy": np.__version__,
            "seal_pri_calibrator_sha256": _sha_file(f"{SEAL_REMOTE}/pri_calibrator.py"),
            "seal_diagnose_sha256": _sha_file(f"{SEAL_REMOTE}/diagnose_inter_head_disagreement.py"),
            "depth_max_tokens": DEPTH_MAX_TOKENS,
        },
        # ── ADDITIVE pointers ONLY ──────────────────────────────────────────
        # A module-level list literal, a CLOSED-SET literal, and an f-string over
        # `slug`. `v_norm_capture_mode` is only ever _VNORM_MODE_ON or
        # _VNORM_MODE_OFF — no runtime-derived text, no exception message, no
        # unbounded string. All diagnostics and every disable/error reason live
        # in the SIDECAR meta.
        "additive_metrics": list(ADDITIVE_DEPTH_CELL_DETAILS),
        "v_norm_capture_mode": v_norm_mode,
        "additive_sidecar": f"{slug}.vnorm.npz",
    }
    # ── REGISTERED WRITE: exactly the eight arrays the frozen extractor wrote ─────
    # Nothing additive is serialized here. A malformed additive array can no longer
    # take this artifact down with it.
    np.savez(npz_path,
             scores=scores, labels=np.asarray(labels, dtype=np.int64),
             sample_idx=np.arange(n_rows, dtype=np.int64),
             gen_token_ids=gen_ids, commit_p=commit_p, yes_no=yes_no.astype(np.int64),
             metrics=json.dumps(list(DEPTH_CELL_DETAILS)), meta=json.dumps(meta))
    with open(f"{outdir}/{slug}.gates.json", "w") as f:
        json.dump(meta, f, indent=2)
    # COMMIT THE REGISTERED ARTIFACT BEFORE ANY ADDITIVE WORK. Everything below
    # is additive; if the container dies in it, the registered npz + gates are
    # already durable on the volume.
    _claim_mark_complete(json, claim_path, dict(claim_payload, state="complete",
                                                npz=npz_path))
    vol.commit()

    # ── ADDITIVE STANZA — ONE NON-RAISING BOUNDARY ───────────────────────────────
    # Everything from here to the end of the try is additive: vmeta construction,
    # the mirror build, the helper call, the tuple unpack, every print, and the
    # additive commit. A BrokenPipeError from a log call inside this block must
    # not escape (in grid B it would become a PERMANENT `aborted` status; here it
    # would skip nothing, because the registered commit already happened).
    vnorm_path, vnorm_reason = None, "not attempted"
    try:
        vmeta = {
            "schema": "furnace-depth-vnorm/1.0", "model": model_id, "task": task,
            "depth_npz": f"{slug}.depth.npz", "precision": precision,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv,
            "n_rows": n_rows,
            "additive_metrics": list(ADDITIVE_DEPTH_CELL_DETAILS),
            "v_norm_capture_mode": v_norm_mode,
            "additive_capture_diagnostics": v_norm_diag,
            "data_sha256": data_sha, "backend": "modal-torch", "comparable": False,
            "registered": False, "out_dir": out_dir,
            "row_identity_mirror": ["labels", "gen_token_ids", "commit_p", "yes_no"],
            "row_identity_check": "REQUIRED",
            "note": "UNREGISTERED descriptive channel (candidate #16). "
                    "(1) Index the trailing axis of v_norm_scores BY NAME via "
                    "v_norm_metrics, never by position. "
                    "(2) BEFORE joining to depth_npz you MUST assert exact "
                    "equality of labels/gen_token_ids/commit_p/yes_no against "
                    "the registered file and REFUSE the join on any mismatch. "
                    "sample_idx is arange(n) and cannot detect a permutation — "
                    "writing these columns without enforcing them would repeat "
                    "exactly the mistake that made sample_idx vacuous.",
        }
        v_mirror = _vnorm_build_mirror(np, v_state, labels, gen_ids, commit_p,
                                       yes_no)
        vnorm_path, vnorm_reason = _vnorm_write_sidecar(
            np, json, f"{outdir}/{slug}.vnorm.npz", v_norm_scores, v_mirror,
            n_rows, vmeta, v_state)
        try:
            print(f"[depth][vnorm] sidecar={vnorm_path!r} reason={vnorm_reason!r} "
                  f"mirror={sorted(v_mirror)}", flush=True)
        except Exception:  # noqa: BLE001
            pass
        vol.commit()
    except Exception as _vexc:  # noqa: BLE001 — additive boundary; never escapes
        vnorm_path = None
        try:
            vnorm_reason = _vnorm_exc_text(_vexc)
        except Exception:  # noqa: BLE001
            vnorm_reason = "<unformattable exception>"
        try:
            print(f"[depth][vnorm] sidecar stanza FAILED: {vnorm_reason} "
                  f"(registered artifact already written and committed)", flush=True)
        except Exception:  # noqa: BLE001
            pass
    result = {"model": model_id, "task": task, "precision": precision, "n_layers": n_layers,
              "n_rows": n_rows, "yes_no_commit_rate": round(frac_yn, 4),
              "gate_cos": [r["oproj_recon_cos"] for r in gate["rows"]],
              "out": npz_path, "out_dir": out_dir,
              "vnorm_out": vnorm_path, "v_norm_capture_mode": v_norm_mode,
              "NOTE": "gates only — scoring is local (depth_score.py) after ALL cells finish"}
    print("DEPTH_EXTRACT_RESULT\n" + json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-7B-Instruct", task: str = "anli_r1",
         n: int = 200, precision: str = "nf4", code_commit: str = "",
         out_dir: str = OUT_DIR_DEFAULT):
    print(depth_extract.remote(model_id, task, n, precision, code_commit, out_dir))
