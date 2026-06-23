"""commit_dump.py — modal entrypoint for per-sample commit tokens across precision rungs.

    modal run cloud/_commit_dump.py --model-id Qwen/Qwen2.5-7B-Instruct --task anli_r1 --precision nf4

Writes a JSONL to the volume at:  commit_dump/<slug>__<precision>_<task>.jsonl
"""
import json, os

MNT = "/vol"
GPU_CONFIG = os.environ.get("CC_GPU_CONFIG", "A100-80GB")

import modal

app = modal.App("commit-dump")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5.0", "transformers>=4.46.0", "bitsandbytes>=0.44.0",
        "numpy", "accelerate",
    )
)
hf_secret = modal.Secret.from_name("huggingface")
vol = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60 * 6)
def dump_commits(model_id: str, task: str, n: int = 200, precision: str = "nf4"):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    PRECISIONS = ("nf4", "int8", "bf16", "fp32")

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    quantized = precision in ("nf4", "int8")
    parts = str(GPU_CONFIG).split(":", 1)
    n_gpu = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
    if (not quantized) and any(s in model_id.lower() for s in ("70b", "72b")) and n_gpu < 2:
        raise ValueError(
            f"{model_id} at precision={precision} will OOM; use nf4|int8 or 2 GPUs.")

    tok = AutoTokenizer.from_pretrained(model_id)
    compute_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    kw = dict(attn_implementation="eager", device_map="auto", torch_dtype=compute_dtype)
    if quantized:
        from transformers import BitsAndBytesConfig
        skip = ["lm_head", "embed_tokens"]
        if precision == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", llm_int8_skip_modules=skip)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        kw.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    def _chat_ids(tok, prompt):
        if tok.chat_template is None:
            raise RuntimeError(f"{model_id} tokenizer missing chat_template")
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return [int(t) for t in np.array(tok(text)["input_ids"]).reshape(-1)]

    # load prompts
    prompts, labels = [], []
    with open(f"{MNT}/data/{task}_n{n}.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            labels.append(rec["label"])

    commits = []
    for i, p_ in enumerate(prompts):
        ids = _chat_ids(tok, p_)
        with torch.no_grad():
            inp = torch.as_tensor([ids], device=model.device)
            logits = model(inp).logits[0, -1, :].detach().float().cpu().numpy()
        p = np.exp(logits - logits.max()); p /= p.sum()
        gid = int(np.argmax(p))
        commit_text = tok.decode([gid], skip_special_tokens=True).strip()
        commits.append({
            "idx": i,
            "label": int(labels[i]),
            "commit_token": commit_text,
            "is_yes_no": commit_text.upper() in ("YES", "NO"),
            "commit_p": round(float(p[gid]), 6),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(prompts)}", flush=True)

    base = model_id.split("/")[-1]
    out_slug = base if precision == "nf4" else f"{base}__{precision}"
    outdir = f"{MNT}/commit_dump"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/{out_slug}_{task}.jsonl"
    with open(outpath, "w") as f:
        for rec in commits:
            f.write(json.dumps(rec) + "\n")
    vol.commit()

    yes_no_rate = sum(1 for c in commits if c["is_yes_no"]) / len(commits)
    print(f"DONE {model_id} {task} precision={precision} n={len(commits)} yes_no_rate={yes_no_rate:.4f}")
    return {"n": len(commits), "yes_no_rate": yes_no_rate, "path": outpath}


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-7B-Instruct", task: str = "anli_r1",
         n: int = 200, precision: str = "nf4"):
    print(dump_commits.remote(model_id, task, n, precision))
