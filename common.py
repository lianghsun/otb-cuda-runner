"""共用小工具。只用標準庫——preflight.py 要在還沒建 .venv 之前就能跑。"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DATA_REPO = "OpenTWBench/otb-mac-data"

# 在 vLLM 裡會被 TP 切分、因此必須能被 TP 整除的欄位。
# 刻意不含 mamba_n_groups：vLLM 的 Mamba2 層在 n_groups=1 時會自動複製 group，
# 不受 TP 限制（granite-4.0-h 系列正是如此）。也不含 head_dim／mamba_d_head：
# 那是向量維度，不是 head 個數。
DIVISIBLE_KEYS = ("num_attention_heads", "num_key_value_heads", "mamba_n_heads")

# 權重佔「這個 TP 群組總顯存」的比例門檻。其餘留給 KV cache、CUDA graph 與啟動開銷。
FIT_OK = 0.70       # 以下：放心跑
FIT_TIGHT = 0.85    # 以下：可跑但緊，建議 GPU_MEM_UTIL=0.95；超過就不跑


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """讀 .env（KEY=VALUE 一行一個），填進環境變數；已經 export 的值優先。

    讓 HF_TOKEN 不必出現在指令列或 shell history 裡。
    """
    path = pathlib.Path(path or HERE / ".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_models(path: str | os.PathLike | None = None) -> list[tuple[str, float]]:
    """讀 models.txt，回傳 [(repo, 權重GiB), ...]，保留檔案中的順序。"""
    path = pathlib.Path(path or HERE / "models.txt")
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out.append((parts[0], float(parts[1]) if len(parts) > 1 else 0.0))
    return out


def hf_get(url: str, token: str | None = None, timeout: int = 30) -> tuple[int, bytes]:
    """GET 一個 Hugging Face URL，回傳 (HTTP 狀態碼, 內容)。不丟例外，方便逐項回報。"""
    headers = {"User-Agent": "otb-cuda-runner"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def model_config(repo: str, token: str | None = None) -> tuple[int, dict | None]:
    status, body = hf_get(f"https://huggingface.co/{repo}/resolve/main/config.json", token)
    if status != 200:
        return status, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def dataset_sha(token: str | None = None) -> str | None:
    status, body = hf_get(f"https://huggingface.co/api/datasets/{DATA_REPO}", token)
    if status != 200:
        return None
    try:
        return json.loads(body).get("sha")
    except ValueError:
        return None


def divisors_needed(cfg: dict) -> dict[str, int]:
    """收集設定檔中所有需被 TP 整除的 head 數（含多模態模型的 text/vision 子設定）。"""
    found = {}
    for scope in ("", "text_config", "vision_config"):
        section = cfg if not scope else cfg.get(scope)
        if not isinstance(section, dict):
            continue
        for key in DIVISIBLE_KEYS:
            value = section.get(key)
            if isinstance(value, int) and value > 0:
                found[f"{scope}.{key}" if scope else key] = value
    return found


def valid_tps(cfg: dict | None, n_gpus: int) -> list[int]:
    """1..n_gpus 中能整除所有需切分欄位的 TP，由小到大。

    vLLM 對此是硬性 assert（例如 total_num_kv_heads % tp_size == 0），不符合時錯誤
    會被包成難以追查的「Engine core initialization failed」。
    """
    need = divisors_needed(cfg or {})
    return [t for t in range(1, max(1, n_gpus) + 1) if all(v % t == 0 for v in need.values())]


def fit_tp(cfg: dict | None, n_gpus: int) -> int:
    """不超過 n_gpus 的最大合法 TP（單一 process 吃下所有卡時用）。"""
    return valid_tps(cfg, n_gpus)[-1]


def min_tp(cfg: dict | None, n_gpus: int, weights_gib: float,
           per_gpu_gib: float) -> tuple[int, str, float]:
    """放得下的最小合法 TP（多顆模型並行時用，省下的卡給別的模型）。

    優先找「ok」的 TP，其次「tight」；都放不下時回傳最大 TP 與 "too_big"。
    """
    tps = valid_tps(cfg, n_gpus)
    for wanted in ("ok", "tight"):
        for t in tps:
            status, ratio = fit_status(weights_gib, t, per_gpu_gib)
            if status == wanted:
                return t, status, ratio
    status, ratio = fit_status(weights_gib, tps[-1], per_gpu_gib)
    return tps[-1], status, ratio


def fit_status(weights_gib: float, tp: int, per_gpu_gib: float) -> tuple[str, float]:
    """回傳 ("ok" | "tight" | "too_big", 權重佔顯存比例)。"""
    if not weights_gib or not per_gpu_gib:
        return "ok", 0.0
    ratio = weights_gib / (tp * per_gpu_gib)
    if ratio <= FIT_OK:
        return "ok", ratio
    if ratio <= FIT_TIGHT:
        return "tight", ratio
    return "too_big", ratio
