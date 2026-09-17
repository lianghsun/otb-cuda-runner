#!/usr/bin/env bash
# 建立 .venv 並安裝 vLLM。依 NVIDIA 驅動版本自動挑 vLLM：
#   驅動 ≥ 580 → vLLM 0.28.0（torch 2.13 / CUDA 13，與 GB200 那輪相同）
#   驅動 < 580 → vLLM 0.13.0（torch 2.9 / CUDA 12.8）
# 要指定版本：VLLM_VERSION=0.13.0 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

die() { echo "✗ $*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "這份腳本只支援 Linux + NVIDIA GPU"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi，請先安裝 NVIDIA 驅動"

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
major=${driver%%.*}
[ "$major" -ge 550 ] || die "驅動 $driver 太舊，請升級到 ≥ 550（建議 ≥ 580）"

if [ -z "${VLLM_VERSION:-}" ]; then
  if [ "$major" -ge 580 ]; then VLLM_VERSION=0.28.0; else VLLM_VERSION=0.13.0; fi
fi
echo "── 驅動 $driver → vLLM $VLLM_VERSION"

if ! command -v uv >/dev/null; then
  echo "── 安裝 uv（Python 套件管理工具，官方安裝程式 https://astral.sh/uv）"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "uv 安裝失敗，請手動安裝後重跑"
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi

echo "── 安裝套件（第一次約 5–10 分鐘）"
uv pip install --python .venv/bin/python \
  "vllm==$VLLM_VERSION" pyarrow "huggingface_hub[hf_xet]" hf_transfer

echo "── 檢查安裝結果"
.venv/bin/python - <<'PY'
import sys
import torch
import transformers
import vllm

print(f"   vLLM {vllm.__version__} | torch {torch.__version__} (CUDA {torch.version.cuda})"
      f" | transformers {transformers.__version__}")
if not torch.cuda.is_available():
    sys.exit("✗ torch 看不到 GPU——通常是驅動版本太舊，撐不起這版 torch 的 CUDA。"
             "可試 VLLM_VERSION=0.13.0 ./setup.sh")
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device=f"cuda:{i}")
    (x @ x).sum().item()
    print(f"   ✓ GPU{i} {torch.cuda.get_device_name(i)} 可正常運算")

from vllm import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
need = ["Lfm2ForCausalLM", "GraniteMoeHybridForCausalLM", "LlamaForCausalLM",
        "Gemma3ForConditionalGeneration"]
missing = [a for a in need if a not in archs]
if missing:
    print(f"   ⚠ 這版 vLLM 不認得 {missing}，對應模型會被略過")
else:
    print("   ✓ 清單中模型的架構都支援")
PY

echo
echo "安裝完成，下一步：./run.sh"
