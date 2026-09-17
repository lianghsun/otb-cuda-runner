#!/usr/bin/env bash
# 跑評測。不帶參數＝models.txt 全部；也可指定模型：./run.sh LiquidAI/LFM2-700M
# 中斷後直接重跑即可，已完成的題目不會重算。建議在 tmux / screen 裡執行。
set -euo pipefail
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "✗ 尚未安裝，請先執行 ./setup.sh" >&2; exit 1; }

mkdir -p logs
log="logs/run-$(date +%Y%m%d-%H%M%S).log"
echo "log: $log"

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
.venv/bin/python run.py "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
