#!/usr/bin/env python3
"""OpenTWBench 補跑入口：包住上游 eval_vllm.py，只調整「怎麼把模型跑起來」，不動評測協定。

    ./run.sh                          # models.txt 全部
    ./run.sh LiquidAI/LFM2-700M       # 指定模型

eval_vllm.py 原封不動取自上游（見 UPSTREAM_*，啟動時會驗雜湊）。題目洗牌、prompt、
答案萃取、計分與結果格式都跟 GB200、Mac 那兩輪完全相同，產出可直接 fold 進榜。

本檔只做這些事：
  1. 多卡時同時跑多顆模型：每顆只拿「放得下的最少張卡」，其餘卡給別顆
     （小模型切到多卡只會多花通訊時間；OTB_PARALLEL=0 可關掉，改成逐顆用滿所有卡）
  2. TP 一定挑能整除 attention / KV / Mamba head 數的值（vLLM 硬性要求）
  3. 下載權重前先確認這版 vLLM 認得該架構、權重放得進顯卡——不行就立刻略過，不白下載
  4. 依單卡顯存調整批次參數：大卡開大提速，小卡保守避免啟動時 OOM
  5. 模型原生 context 比上游預設的 8192 短時，改用原生長度重試
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import preflight  # noqa: E402

UPSTREAM_REPO = "lianghsun/otb-b200-runner"
UPSTREAM_COMMIT = "6e0ef175be7a75fde552d743d3f7e0c972b27407"
UPSTREAM_SHA256 = "38ed77480996c7beb0e2ca55265f5d52a318660dfda8130fb50c0a64e1cb2e87"
RESULTS = HERE / "results"
PROGRESS_EVERY = 300        # 並行模式下每幾秒回報一次進度


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def model_dir(repo: str) -> pathlib.Path:
    return RESULTS / repo.replace("/", "_")


# ───────────────────────────── worker：在目前可見的卡上逐顆跑 ─────────────────────────────

def worker(models: list[str], token: str, eval_sha: str) -> int:
    import torch
    if not torch.cuda.is_available():
        print("✗ torch 看不到 GPU。請先執行 ./setup.sh，它會檢查驅動與 CUDA 是否相容。")
        return 4
    n_gpus = torch.cuda.device_count()
    per_gpu_gib = min(torch.cuda.get_device_properties(i).total_memory
                      for i in range(n_gpus)) / 2**30
    big_gpu = per_gpu_gib >= 70
    batched = int(os.environ.get("OTB_MAX_BATCHED_TOKENS", 65536 if big_gpu else 16384))
    seqs = int(os.environ.get("OTB_MAX_NUM_SEQS", 1024 if big_gpu else 256))
    eager = os.environ.get("ENFORCE_EAGER") == "1"

    import transformers
    import vllm
    from vllm import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())

    import eval_vllm as ev
    if ev.DATA_REPO != common.DATA_REPO or ev.OUT != RESULTS:
        print("✗ eval_vllm.py 的資料集或輸出路徑與 runner 設定不一致")
        return 3

    weights = dict(common.read_models())
    user_tp = int(os.environ.get("TP", "0"))

    # 上游 run_model() 在呼叫當下才 `from vllm import LLM`，換掉模組屬性即可生效。
    original_llm = vllm.LLM

    def llm_with_runner_defaults(*args, **kwargs):
        kwargs.setdefault("max_num_batched_tokens", batched)
        kwargs.setdefault("max_num_seqs", seqs)
        if eager:
            kwargs["enforce_eager"] = True
        try:
            return original_llm(*args, **kwargs)
        except Exception as exc:
            if "max_model_len" in kwargs and "max_model_len" in str(exc):
                print(f"    max_model_len={kwargs['max_model_len']} 超過模型上限，"
                      "改用模型原生長度重試", flush=True)
                kwargs.pop("max_model_len")
                return original_llm(*args, **kwargs)
            raise

    vllm.LLM = llm_with_runner_defaults

    # 上游 main() 以模組全域名稱呼叫 run_model，替換模組屬性即可攔截；這裡丟出的
    # 例外會被上游記為 FAILED，不影響清單中的其他模型。
    original_run = ev.run_model

    def run_model(model, benches, max_tokens):
        status, cfg = common.model_config(model, token)
        if cfg is None:
            raise RuntimeError(f"讀不到 config.json（HTTP {status}）——"
                               "gated 模型需先在 Hugging Face 頁面同意授權")
        archs = cfg.get("architectures") or []
        if archs and not any(a in supported for a in archs):
            raise RuntimeError(f"vLLM {vllm.__version__} 不支援架構 {archs}（未下載權重即略過）")
        tp = common.fit_tp(cfg, user_tp or n_gpus)
        if user_tp and tp != user_tp:
            print(f"    TP={user_tp} 無法整除 head 數，改用 TP={tp}", flush=True)
        fit, ratio = common.fit_status(weights.get(model, 0.0), tp, per_gpu_gib)
        if fit == "too_big":
            raise RuntimeError(f"權重 {weights[model]:.1f} GiB 佔 TP={tp} 顯存的 {ratio:.0%}，"
                               "放不下，略過（未下載權重）")
        if fit == "tight" and "GPU_MEM_UTIL" not in os.environ:
            print(f"    顯存偏緊（權重佔 {ratio:.0%}）；若載入失敗請改設 GPU_MEM_UTIL=0.95",
                  flush=True)
        os.environ["TP"] = str(tp)
        print(f"    [runner] TP={tp}  max_num_batched_tokens={batched}  max_num_seqs={seqs}",
              flush=True)
        return original_run(model, benches, max_tokens)

    ev.run_model = run_model

    # 溯源紀錄：讓收結果的人能判斷這批分數是在什麼條件下產生的
    started = dt.datetime.now(dt.timezone.utc)
    provenance = {
        "runner": "otb-cuda-runner",
        "runner_commit": git_rev(),
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "eval_vllm_sha256": eval_sha,
        "dataset_repo": common.DATA_REPO,
        "dataset_sha": common.dataset_sha(token),
        "started_utc": started.isoformat(timespec="seconds"),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "nvidia_driver": preflight.driver_version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "per_gpu_gib": round(per_gpu_gib, 1),
        "serving": {"max_num_batched_tokens": batched, "max_num_seqs": seqs,
                    "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.90")),
                    "enforce_eager": eager},
        "models": models,
    }
    prov_dir = RESULTS / "_provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / f"{started:%Y%m%d-%H%M%S}-{os.getpid()}.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2))

    print(f"== {len(models)} 顆模型｜{n_gpus}× {torch.cuda.get_device_name(0)} "
          f"({per_gpu_gib:.0f} GiB)｜vLLM {vllm.__version__}｜資料集 "
          f"{(provenance['dataset_sha'] or '?')[:10]}", flush=True)

    sys.argv = ["eval_vllm.py", "--models", *models]
    ev.main()
    return 0


# ─────────────────────── 排程：多卡時讓多顆模型各佔幾張卡同時跑 ───────────────────────

def plan_jobs(models: list[str], gpus: list[dict], token: str):
    """回傳 (可排程的 [(repo, tp)], 直接略過的 [(repo, 原因)])。"""
    per_gpu = min(g["gib"] for g in gpus)
    weights = dict(common.read_models())
    jobs, skipped = [], []
    for repo in models:
        status, cfg = common.model_config(repo, token)
        if cfg is None:
            skipped.append((repo, f"讀不到 config.json（HTTP {status}），gated 模型需先同意授權"))
            continue
        tp, fit, ratio = common.min_tp(cfg, len(gpus), weights.get(repo, 0.0), per_gpu)
        if fit == "too_big":
            skipped.append((repo, f"權重佔 {tp} 張卡顯存的 {ratio:.0%}，放不下"))
            continue
        jobs.append((repo, tp))
    # 先排吃卡多、權重大的，避免它們一直等不到足夠的空卡
    jobs.sort(key=lambda j: (-j[1], -weights.get(j[0], 0.0)))
    return jobs, skipped


def progress(repo: str) -> str:
    parts = []
    for bench in ("formosa", "exam"):
        path = model_dir(repo) / f"{bench}.jsonl"
        if path.exists():
            with path.open("rb") as fh:
                parts.append(f"{bench} {sum(1 for _ in fh)}")
    return " ".join(parts) or "載入中"


def orchestrate(models: list[str], gpus: list[dict], token: str) -> int:
    jobs, skipped = plan_jobs(models, gpus, token)
    for repo, why in skipped:
        print(f"⏭  略過 {repo}：{why}")
    if not jobs:
        return 1

    log_dir = HERE / "logs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    free = [g["index"] for g in gpus]
    queue = list(jobs)
    running: dict[str, tuple] = {}
    ok, failed = [], list(skipped)
    print(f"== {len(jobs)} 顆模型、{len(gpus)} 張卡並行｜每顆的完整輸出在 {log_dir}/", flush=True)

    last_report = time.time()
    try:
        while queue or running:
            # 能開就開：依序找第一個「需要的卡數 ≤ 空卡數」的模型
            launched = True
            while launched:
                launched = False
                for i, (repo, tp) in enumerate(queue):
                    if tp > len(free):
                        continue
                    ids, free = free[:tp], free[tp:]
                    log_path = log_dir / f"{repo.replace('/', '_')}.log"
                    log = log_path.open("w")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, ids)),
                               TP=str(tp), OTB_WORKER="1")
                    proc = subprocess.Popen([sys.executable, str(HERE / "run.py"), repo],
                                            env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[repo] = (proc, ids, time.time(), log, log_path)
                    print(f"▶  {repo}  GPU {','.join(map(str, ids))}  TP={tp}", flush=True)
                    queue.pop(i)
                    launched = True
                    break

            time.sleep(5)
            for repo in list(running):
                proc, ids, t0, log, log_path = running[repo]
                if proc.poll() is None:
                    continue
                log.close()
                del running[repo]
                free = sorted(free + ids)
                minutes = (time.time() - t0) / 60
                if (model_dir(repo) / ".done").exists():
                    ok.append(repo)
                    print(f"✓  {repo}  {minutes:.0f} 分鐘", flush=True)
                else:
                    tail = log_path.read_text(errors="replace").strip().splitlines()[-3:]
                    failed.append((repo, " / ".join(tail)[-300:]))
                    print(f"✗  {repo}  {minutes:.0f} 分鐘後失敗（完整 log：{log_path}）", flush=True)
                    for line in tail:
                        print(f"     {line}", flush=True)

            if running and time.time() - last_report > PROGRESS_EVERY:
                last_report = time.time()
                print("   進度｜" + "｜".join(f"{r.split('/')[-1]} {progress(r)}"
                                           for r in running), flush=True)
    finally:
        for proc, *_ in running.values():
            proc.terminate()

    print("\n===== SUMMARY =====")
    for repo in ok:
        print(f"  OK      {repo}")
    for repo, why in failed:
        print(f"  FAILED  {repo}  ({why})")
    # 各 worker 都寫過自己的 _summary.json（互相覆蓋），最後以整體結果為準
    (RESULTS / "_summary.json").write_text(json.dumps(
        {"ok": ok, "skipped": [], "failed": failed}, ensure_ascii=False, indent=2))
    return 0


# ─────────────────────────────────── 入口 ───────────────────────────────────

def main(argv: list[str]) -> int:
    common.load_env_file()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("✗ 請設定 HF_TOKEN（寫進 .env 或 export；評測題目在私有資料集）")
        return 2

    eval_sha = sha256(HERE / "eval_vllm.py")
    if eval_sha != UPSTREAM_SHA256 and os.environ.get("OTB_ALLOW_MODIFIED") != "1":
        print("✗ eval_vllm.py 與上游版本不一致，產出的分數無法保證能跟榜單比較。\n"
              f"  預期 {UPSTREAM_SHA256[:16]}…，實際 {eval_sha[:16]}…\n"
              "  若確定是刻意修改，設 OTB_ALLOW_MODIFIED=1 再執行。")
        return 3

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    models = argv or [repo for repo, _ in common.read_models()]

    if os.environ.get("OTB_WORKER") == "1":
        return worker(models, token, eval_sha)

    todo = [m for m in models if not (model_dir(m) / ".done").exists()]
    for m in models:
        if m not in todo:
            print(f"⏭  {m} 已完成")
    if not todo:
        print("全部都跑完了。下一步：./send_results.sh")
        return 0

    gpus = preflight.query_gpus()
    parallel = (len(gpus) > 1 and len(todo) > 1
                and os.environ.get("OTB_PARALLEL") != "0" and not os.environ.get("TP"))
    rc = orchestrate(todo, gpus, token) if parallel else worker(todo, token, eval_sha)

    left = [m for m in models if not (model_dir(m) / ".done").exists()]
    if not left:
        print("\n== 全部完成。下一步：./send_results.sh")
    else:
        print(f"\n== 還有 {len(left)} 顆未完成。可直接重跑 ./run.sh 續跑（已算的題目不會重算），"
              "或先用 ./send_results.sh 送回已完成的部分。")
    return rc or (1 if left else 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
