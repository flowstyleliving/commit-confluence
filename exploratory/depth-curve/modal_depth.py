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
"""
import os
from pathlib import Path

import modal

APP_NAME = "cc-depth-curve"
VOL_NAME = "model-cache"
MNT = "/models"
SEAL_REMOTE = "/seal"
PKG_REMOTE = "/pkg"
OUT_DIR = "depth_curve"
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


@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60 * 6)
def depth_extract(model_id: str, task: str, n: int = 200, precision: str = "nf4",
                  code_commit: str = ""):
    import hashlib
    import json
    import sys

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

    for i, prompt in enumerate(prompts):
        ids = BASE._chat_ids(tok, prompt, model_id)
        if len(ids) > DEPTH_MAX_TOKENS:
            raise RuntimeError(f"row {i}: prompt has {len(ids)} tokens > {DEPTH_MAX_TOKENS} "
                               f"(memory-safe bound for all-blocks attention capture)")
        with torch.no_grad():
            out = model(torch.tensor([ids], device=model.device),
                        output_attentions=True, use_cache=False)
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
        del att, out
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if i % 25 == 0:
            print(f"[depth]  {i}/{n_rows} yes_no_so_far={int(yes_no[:i + 1].sum())}", flush=True)

    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite scores present after loop — should be unreachable")
    frac_yn = float(yes_no.mean())
    print(f"[depth] done: yes_no_rate={frac_yn:.2%}", flush=True)
    if frac_yn < 0.5:
        raise RuntimeError(f"YES/NO commit rate {frac_yn:.2%} < 50% — task not attempted; refusing to save")

    slug = model_id.split("/")[-1]
    outdir = f"{MNT}/{OUT_DIR}/{task}"
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
    }
    np.savez(f"{outdir}/{slug}.depth.npz",
             scores=scores, labels=np.asarray(labels, dtype=np.int64),
             sample_idx=np.arange(n_rows, dtype=np.int64),
             gen_token_ids=gen_ids, commit_p=commit_p, yes_no=yes_no.astype(np.int64),
             metrics=json.dumps(list(DEPTH_CELL_DETAILS)), meta=json.dumps(meta))
    with open(f"{outdir}/{slug}.gates.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()
    result = {"model": model_id, "task": task, "precision": precision, "n_layers": n_layers,
              "n_rows": n_rows, "yes_no_commit_rate": round(frac_yn, 4),
              "gate_cos": [r["oproj_recon_cos"] for r in gate["rows"]],
              "out": f"{outdir}/{slug}.depth.npz",
              "NOTE": "gates only — scoring is local (depth_score.py) after ALL cells finish"}
    print("DEPTH_EXTRACT_RESULT\n" + json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-7B-Instruct", task: str = "anli_r1",
         n: int = 200, precision: str = "nf4", code_commit: str = ""):
    print(depth_extract.remote(model_id, task, n, precision, code_commit))
