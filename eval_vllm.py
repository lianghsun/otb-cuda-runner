#!/usr/bin/env python3
"""OpenTWBench eval on NVIDIA B200 (or any modern CUDA GPU) via vLLM — for the
models still unrun after the Mac/MLX pass: the big ones (up to 72B), the
reasoning distills, and the driver-blocked architectures (mamba hybrids, LFM2,
Qwen3.5) that the old vLLM on the 3090 could not serve. A modern vLLM on
Blackwell serves all of them.

Protocol is IDENTICAL to the rest of the suite so numbers drop onto the same
leaderboard: box answer extraction, a deterministic per-question option shuffle
seeded by sha256(question), temperature 0, the same system prompt, and the
box-first + bare-letter lenient fallback. Full precision only (bfloat16); no
quantization anywhere.

Privacy by design: benchmark items are pulled from a PRIVATE HF repo with your
HF_TOKEN and never written to disk in the clear; results contain only a hash of
each question plus booleans (correct? box-parsed? lenient-parsed?), so they can
live in a public repo without leaking the benchmark.

    HF_TOKEN=hf_xxx python eval_vllm.py                # all default models
    HF_TOKEN=hf_xxx python eval_vllm.py --models A B    # a subset
    TP=8 HF_TOKEN=hf_xxx python eval_vllm.py            # tensor-parallel over 8 GPUs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import random
import re
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
DATA_REPO = "OpenTWBench/otb-mac-data"          # private; shared with the Mac runner
BENCHES = {"formosa": "formosa.parquet", "exam": "exam-sample.parquet"}

SYSTEM = (
    "你是一位專業的測驗作答助理。請仔細閱讀題目，以繁體中文（臺灣用語）思考，"
    "並在最後輸出 \\boxed{}，大括號內只填入唯一正確選項的英文字母，也就是 "
    "A、B、C、D 其中一個。除了 \\boxed{} 之外不要使用其他格式標示答案。")

_BOX = re.compile(r"\\boxed\{\s*([A-Da-d甲乙丙丁])\s*\}")
_ORD = {"甲": "A", "乙": "B", "丙": "C", "丁": "D"}
_LEAD = re.compile(r"^\s*[（(]?\s*([A-Da-d甲乙丙丁])\s*[)）.。、:：\s]")
_MARKED = re.compile(r"(?:答案|正確選項|正解|答|選)\s*(?:是|為|:|：)?\s*[（(]?\s*([A-Da-d甲乙丙丁])")
_ANY = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")

# The 22 models the Mac pass left unrun. All served in bfloat16 (no quant).
DEFAULT_MODELS = [
    # multimodal (MLX couldn't load it; vLLM serves the text path)
    "ornith-ai/Ornith-1.5-9B",
    "ornith-ai/Ornith-1.5-35B-A3B",               # ~72GB bf16 MoE (A3B); Qwen3.5-MoE-VL, text path
    # Qwen/Qwen3.8-27B (paper anchor) assigned to the H200 batch, not here.
    # big, non-reasoning
    "tencent/Hunyuan-7B-Instruct",                # (g?) trust_remote_code
    "THUDM/glm-4-9b-chat",
    "01-ai/Yi-1.5-9B-Chat",
    "MediaTek-Research/Llama-Breeze2-3B-Instruct",  # vision stripped — run prepare_breeze2.py first
    "mistralai/Mistral-Nemo-Instruct-2407",       # (g)
    "microsoft/Phi-4",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-30B-A3B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "mistralai/Mistral-Small-24B-Instruct-2501",  # (g)
    "google/gemma-3-27b-it",                       # (g)
    "Qwen/Qwen2.5-32B-Instruct",
    "zai-org/GLM-4-32B-0414",
    "Qwen/Qwen3-32B",
    "01-ai/Yi-1.5-34B-Chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",        # (g)
    "Qwen/Qwen2.5-72B-Instruct",
    "tencent/Hunyuan-A13B-Instruct",               # (g?) trust_remote_code
    # reasoning (get max_tokens 16384 — see MAX_TOKENS)
    # Phi-4-reasoning-plus cancelled: box 0% + ~random accuracy (its long CoT is
    # truncated before the \boxed{} answer, hash-only output can't diagnose it,
    # and the user opted not to re-run it with a bigger budget). Off the board.
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # R1-Distill 32B/70B cancelled per request (distills no longer needed;
    # 14B already boarded, kept). QwQ / Phi-4-reasoning are not distills.
]

# Reasoning models need room to finish thinking before the \boxed{} answer.
MAX_TOKENS = {
    "Qwen/QwQ-32B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 16384,
}

# Models vLLM can't load as-is and must be pre-converted to a local checkpoint.
# prepare_breeze2.py strips Breeze2's InternVL vision tower to a plain Llama;
# the board name stays the HF id, loading uses the local path.
LOCAL_ALIAS = {
    "MediaTek-Research/Llama-Breeze2-3B-Instruct": str(HERE / "breeze2-3b-text"),
}


def qhash(stem: str) -> str:
    return hashlib.sha256(stem.encode()).hexdigest()[:16]


def box(text: str):
    m = _BOX.search(text or "")
    if not m:
        return None
    g = m.group(1).upper()
    return _ORD.get(g, g)


def lenient(text: str):
    if not text:
        return None
    t = text.split("</think>")[-1]
    for rx in (_BOX, _LEAD, _MARKED):
        m = rx.search(t)
        if m:
            g = m.group(1).upper()
            return _ORD.get(g, g)
    m = _ANY.search(t)
    return m.group(1).upper() if m else None


def accel_downloads():
    """Enable hf_transfer only if importable — an enabled-but-missing hf_transfer
    hard-fails every download. hf_xet is used automatically when installed."""
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return "hf_transfer"
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        return None


def purge_model(model):
    """Delete a model's weights from the HF cache after scoring, so peak disk
    stays at one model. Only the model snapshot is removed."""
    import shutil
    cache = (os.environ.get("HF_HUB_CACHE")
             or (os.environ.get("HF_HOME") and pathlib.Path(os.environ["HF_HOME"]) / "hub")
             or pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    d = pathlib.Path(cache) / ("models--" + model.replace("/", "--"))
    if d.exists():
        try:
            freed = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except Exception:
            freed = 0
        shutil.rmtree(d, ignore_errors=True)
        print(f"    reclaimed weights: {model}  (~{freed / 1e9:.1f} GB)", flush=True)


def load_rows(parquet):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(DATA_REPO, parquet, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    return pq.read_table(p).to_pylist()


def shuffled(row):
    opts = [(k, row[k]) for k in "ABCD"]
    seed = int.from_bytes(hashlib.sha256(row["question"].encode()).digest()[:8], "big")
    random.Random(seed).shuffle(opts)
    correct_text = row[row["answer"]]
    out = {"question": row["question"]}
    ans = None
    for (ok, text), nk in zip(opts, "ABCD"):
        out[nk] = text
        if text == correct_text:
            ans = nk
    out["answer"] = ans
    return out


def run_model(model, benches, max_tokens):
    from vllm import LLM, SamplingParams
    tp = int(os.environ.get("TP", "0")) or _gpu_count()
    # reasoning models need a long context window; keep others tight to save KV
    reasoning = model in MAX_TOKENS
    max_model_len = 20480 if reasoning else 8192
    path = LOCAL_ALIAS.get(model, model)
    # Models that need a local conversion first (Breeze2: strip the vision tower).
    # prepare_breeze2.py writes ./breeze2-3b-text; run it once if it's not there.
    if path != model and not pathlib.Path(path).exists():
        prep = HERE / "prepare_breeze2.py"
        if prep.exists():
            print(f"    {path} missing — running {prep.name} to build it", flush=True)
            import subprocess
            subprocess.run([sys.executable, str(prep)], check=True)
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{path} still missing after prepare step; cannot load {model}")
    print(f"\n=== loading {model}  (tp={tp}, max_model_len={max_model_len})"
          + (f"  [local: {path}]" if path != model else ""), flush=True)
    t0 = time.time()
    # dtype="auto" keeps each model in its released precision — bf16 for full-
    # precision models, and the native fp8 / mxfp4 for frontier models that ship
    # only quantized (Policy B: native release precision is allowed and labelled;
    # post-hoc quantization of a bf16 model is not).
    # GPU_MEM_UTIL lets a shared GPU (e.g. home-srv's 3090 already hosting an
    # embedding model) cap vLLM to the free slice instead of the default 90%.
    mem_util = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
    llm = LLM(model=path, dtype="auto", trust_remote_code=True,
              tensor_parallel_size=tp, gpu_memory_utilization=mem_util,
              max_model_len=max_model_len, enforce_eager=False)
    print(f"    loaded in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out_dir = OUT / model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in benches:
        rows = load_rows(BENCHES[bench])
        path = out_dir / f"{bench}.jsonl"
        done = set()
        if path.exists():
            for line in path.open():
                try:
                    done.add(json.loads(line)["qh"])
                except Exception:
                    pass
        todo = [r for r in rows if qhash(r["question"]) not in done]
        print(f"    {bench}: {len(rows)} items, {len(done)} done, {len(todo)} to go",
              flush=True)
        t0 = time.time()
        # batch in chunks so results stream to disk and a restart resumes
        CHUNK = 2000
        with path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), CHUNK):
                batch = todo[i:i + CHUNK]
                convs = []
                for row in batch:
                    q = shuffled(row)
                    body = (f"題目：{q['question']}\nA. {q['A']}\nB. {q['B']}\n"
                            f"C. {q['C']}\nD. {q['D']}")
                    convs.append([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": body}])
                outs = llm.chat(convs, sp, use_tqdm=False)
                for row, out in zip(batch, outs):
                    q = shuffled(row)
                    text = out.outputs[0].text if out.outputs else ""
                    b = box(text)
                    le = lenient(text)
                    fh.write(json.dumps({
                        "qh": qhash(row["question"]),
                        "ok": le == q["answer"],
                        "boxp": b is not None,
                        "lenp": le is not None,
                    }) + "\n")
                fh.flush()
                rate = min(i + CHUNK, len(todo)) / max(1e-9, time.time() - t0)
                print(f"      {min(i + CHUNK, len(todo))}/{len(todo)}  {rate:.1f} q/s",
                      flush=True)
        print(f"    {bench} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # free the GPU before the next model loads
    try:
        import contextlib
        import gc
        import torch
        from vllm.distributed.parallel_state import destroy_model_parallel
        with contextlib.suppress(Exception):
            destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_count():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def main():
    accel = accel_downloads()
    print(f"   download accelerator: {accel or 'hf_xet/LFS'}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--benches", nargs="*", default=list(BENCHES))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--keep", action="store_true",
                    help="keep weights after each model (default: reclaim)")
    args = ap.parse_args()

    counts: dict[str, int] = {}

    def complete(model):
        d = OUT / model.replace("/", "_")
        for b in args.benches:
            counts.setdefault(b, len(load_rows(BENCHES[b])))
            p = d / f"{b}.jsonl"
            n = sum(1 for _ in p.open()) if p.exists() else 0
            if n < counts[b]:
                return False
        return True

    ok, failed, skipped = [], [], []
    for model in args.models:
        marker = OUT / model.replace("/", "_") / ".done"
        if marker.exists() or complete(model):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done\n")
            print(f"\n=== skip {model} (already done)", flush=True)
            skipped.append(model)
            continue
        try:
            run_model(model, args.benches, MAX_TOKENS.get(model, args.max_tokens))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done\n")
            ok.append(model)
            if not args.keep:
                purge_model(model)
        except KeyboardInterrupt:
            print(f"\n    interrupted during {model} — weights kept for resume",
                  flush=True)
            raise
        except Exception as e:
            print(f"    !! {model} failed: {type(e).__name__}: {str(e)[:300]}",
                  flush=True)
            failed.append((model, f"{type(e).__name__}: {str(e)[:300]}"))
            if not args.keep:
                purge_model(model)

    print("\n===== SUMMARY =====")
    for m in ok:
        print(f"  OK      {m}")
    for m in skipped:
        print(f"  SKIP    {m}")
    for m, why in failed:
        print(f"  FAILED  {m}  ({why})")
    (OUT / "_summary.json").write_text(json.dumps(
        {"ok": ok, "skipped": skipped, "failed": failed},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
