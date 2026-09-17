#!/usr/bin/env python3
"""開跑前的體檢：顯卡、驅動、記憶體、磁碟、HF 存取權，以及每顆模型放不放得下。

不需要先安裝任何東西（只用 Python 標準庫）：

    HF_TOKEN=hf_xxx python3 preflight.py
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

DRIVER_FOR_028 = 580        # vLLM 0.28 綁 torch 2.13（CUDA 13），驅動需 ≥ 580
MIN_RAM_GIB = 64
MIN_DISK_GIB = 100


def query_gpus() -> list[dict]:
    """用 nvidia-smi 列出 GPU；會尊重 CUDA_VISIBLE_DEVICES（nvidia-smi 本身不看它）。"""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, mem_mib, driver = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name,
                     "gib": int(mem_mib) / 1024, "driver": driver})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        keep = {int(x) for x in visible.split(",") if x.strip().isdigit()}
        gpus = [g for g in gpus if g["index"] in keep]
    return gpus


def driver_version() -> str | None:
    gpus = query_gpus()
    return gpus[0]["driver"] if gpus else None


def parallel_enabled(n_gpus: int) -> bool:
    """與 run.py 相同的判斷：多卡且沒關掉並行、沒指定 TP 時，多顆模型會同時跑。"""
    return n_gpus > 1 and os.environ.get("OTB_PARALLEL") != "0" and not os.environ.get("TP")


def host_ram_gib() -> float | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def hf_cache_dir() -> pathlib.Path:
    if os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return pathlib.Path(os.environ["HF_HOME"]) / "hub"
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def plan(models, gpus, token):
    """對每顆模型決定 TP、判斷放不放得下、檢查 HF 存取權。回傳逐列結果。"""
    n = len(gpus)
    per_gpu = min((g["gib"] for g in gpus), default=0.0)
    rows = []
    for repo, weights in models:
        status, cfg = common.model_config(repo, token)
        if cfg is None:
            reason = {401: "token 無效或未提供", 403: "未取得授權（gated，需先同意）",
                      404: "找不到此模型"}.get(status, f"HTTP {status}")
            rows.append({"repo": repo, "weights": weights, "tp": None, "fit": "no_access",
                         "ratio": 0.0, "note": reason})
            continue
        note = ""
        if not n:
            tp, fit, ratio = None, "no_gpu", 0.0
        elif parallel_enabled(n):
            tp, fit, ratio = common.min_tp(cfg, n, weights, per_gpu)
        else:
            want = int(os.environ.get("TP") or n)
            tp = common.fit_tp(cfg, want)
            fit, ratio = common.fit_status(weights, tp, per_gpu)
            if tp < want:
                need = common.divisors_needed(cfg)
                note = f"head 數 {sorted(set(need.values()))} 不整除 {want}，TP 降為 {tp}"
        rows.append({"repo": repo, "weights": weights, "tp": tp, "fit": fit,
                     "ratio": ratio, "note": note})
    return rows


def main() -> int:
    common.load_env_file()
    token = os.environ.get("HF_TOKEN")
    problems = 0

    print("── 顯卡")
    gpus = query_gpus()
    if not gpus:
        print("   ✗ 找不到 NVIDIA GPU（nvidia-smi 不存在或無輸出）。這台機器無法執行評測。")
        return 1
    for g in gpus:
        print(f"   GPU{g['index']}  {g['name']}  {g['gib']:.1f} GiB")
    driver = gpus[0]["driver"]
    major = int(driver.split(".")[0])
    vllm_ver = "0.28.0" if major >= DRIVER_FOR_028 else "0.13.0"
    print(f"   驅動 {driver} → setup.sh 會安裝 vLLM {vllm_ver}"
          + ("" if major >= DRIVER_FOR_028 else "（驅動 < 580，改用 CUDA 12.8 版本）"))

    print("── Hugging Face 存取")
    if not token:
        print("   ✗ 未設定 HF_TOKEN。評測題目放在私有資料集，一定要有 token（寫進 .env）。")
        return 1
    ds = common.dataset_sha(token)
    if ds:
        print(f"   ✓ 可讀取評測資料集 {common.DATA_REPO}（版本 {ds[:10]}）")
    else:
        print(f"   ✗ 無法讀取 {common.DATA_REPO}：token 沒有此私有資料集的讀取權")
        problems += 1

    parallel = parallel_enabled(len(gpus))
    print("── 模型" + ("（多卡並行：每顆只佔放得下的最少張卡，同時跑好幾顆）" if parallel else ""))
    rows = plan(common.read_models(), gpus, token)
    icon = {"ok": "✓", "tight": "△", "too_big": "⏭", "no_access": "✗", "no_gpu": "✗"}
    label = {"ok": "", "tight": "顯存偏緊，建議 GPU_MEM_UTIL=0.95",
             "too_big": "顯存不足，會自動略過", "no_access": "", "no_gpu": ""}
    runnable = []
    for r in rows:
        tp = f"TP={r['tp']}" if r["tp"] else "    "
        pct = f"{r['ratio']:>4.0%}" if r["ratio"] else "    "
        msg = "；".join(x for x in (label[r["fit"]], r["note"]) if x)
        print(f"   {icon[r['fit']]} {r['weights']:5.1f} GiB  {tp:<5} {pct}  {r['repo']}"
              + (f"  — {msg}" if msg else ""))
        if r["fit"] in ("ok", "tight"):
            runnable.append(r)
        problems += r["fit"] == "no_access"

    print("── 主機")
    ram = host_ram_gib()
    if ram is not None:
        mark = "✓" if ram >= MIN_RAM_GIB else "⚠"
        print(f"   {mark} 記憶體 {ram:.0f} GiB" + ("" if ram >= MIN_RAM_GIB else
              f"（建議 ≥ {MIN_RAM_GIB} GiB；載入模型時可能被 OOM killer 砍掉）"))
    # 每顆跑完會刪權重；並行時最壞情況是所有可跑的模型同時躺在磁碟上
    sizes = [r["weights"] for r in runnable] or [0.0]
    need = max(MIN_DISK_GIB, (sum(sizes) if parallel else max(sizes)) + 20)
    cache = hf_cache_dir()
    probe = cache if cache.exists() else pathlib.Path.home()
    free = shutil.disk_usage(probe).free / 2**30
    mark = "✓" if free >= need else "✗"
    print(f"   {mark} 模型快取 {cache} 剩餘 {free:.0f} GiB"
          + ("" if free >= need else f"（需 ≥ {need:.0f} GiB；可設 HF_HOME 指到較大的磁碟）"))
    problems += free < need

    print()
    print(f"可以跑 {len(runnable)}/{len(rows)} 顆。")
    if problems:
        print("有需要先處理的問題（上方 ✗）。")
        return 1
    print("沒有問題，下一步：./setup.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
