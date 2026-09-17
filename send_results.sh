#!/usr/bin/env bash
# 把驗證通過的結果送回：推到本 repo 的 results/<主機>-<時間> 分支，並另存一份 tar.gz。
#   ./send_results.sh              # 推分支 + 打包
#   NO_PUSH=1 ./send_results.sh    # 只打包，自己把 tar.gz 傳給對方
#
# 送出的只有題目雜湊與對錯布林值，不含題目、選項或模型輸出。
# 推送不會動到你的工作目錄或目前分支（用暫時的 index 建 commit）。
set -euo pipefail
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "✗ 找不到 .venv，請先 ./setup.sh" >&2; exit 1; }

echo "── 驗證結果"
.venv/bin/python verify_results.py || true
complete=()
while IFS= read -r d; do
  if [ -n "$d" ]; then complete+=("$d"); fi
done < <(.venv/bin/python verify_results.py --complete 2>/dev/null || true)
[ "${#complete[@]}" -gt 0 ] || { echo "✗ 沒有可送回的完整結果" >&2; exit 1; }

paths=()
for d in "${complete[@]}"; do paths+=("results/$d"); done
for extra in results/_provenance results/_summary.json; do
  if [ -e "$extra" ]; then paths+=("$extra"); fi
done

host=$(hostname -s 2>/dev/null || hostname)
host=$(printf '%s' "$host" | tr -c 'A-Za-z0-9._-' '-')
ts=$(date -u +%Y%m%d-%H%M%S)
tarball="otb-results-$host-$ts.tar.gz"
tar czf "$tarball" "${paths[@]}"
echo "── 已打包 ${#complete[@]} 顆模型 → $tarball"

if [ "${NO_PUSH:-0}" = 1 ]; then
  echo "NO_PUSH=1：請把 $tarball 傳給評測負責人。"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_INDEX_FILE="$tmp/index"
git read-tree HEAD
git add -f -- "${paths[@]}"
tree=$(git write-tree)
commit=$(git -c user.name=otb-cuda-runner -c user.email=otb-cuda-runner@users.noreply.github.com \
  commit-tree "$tree" -p HEAD -m "results: ${#complete[@]} models from $host ($ts)")
branch="results/$host-$ts"

if git push origin "$commit:refs/heads/$branch"; then
  echo "✓ 已推到分支 ${branch}，謝謝！"
else
  echo "✗ 推送失敗（可能沒有這個 repo 的寫入權限）。請改把 $tarball 傳給評測負責人。" >&2
  exit 1
fi
