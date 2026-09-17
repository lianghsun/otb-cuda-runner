#!/usr/bin/env python3
"""檢查 results/ 每顆模型是否完整、題目是否對得上目前的評測集，並印出正確率當 sanity check。

    .venv/bin/python verify_results.py              # 報告
    .venv/bin/python verify_results.py --complete   # 只列出可送回的目錄（send_results.sh 用）

「可送回」＝ 有 .done，且每個 bench 的題目雜湊與資料集完全一致（不缺、不多）。
多出來的題目代表用的是舊版題目，這正是這批模型要重跑的原因，所以視為錯誤。
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import eval_vllm as ev  # noqa: E402


def expected_hashes() -> dict[str, set[str]]:
    out = {}
    for bench, parquet in ev.BENCHES.items():
        out[bench] = {ev.qhash(r["question"]) for r in ev.load_rows(parquet)}
    return out


def inspect(model_dir: pathlib.Path, expected: dict[str, set[str]]) -> dict:
    report = {"done": (model_dir / ".done").exists(), "benches": {}, "ok": True}
    for bench, want in expected.items():
        path = model_dir / f"{bench}.jsonl"
        seen: dict[str, dict] = {}
        bad = 0
        if path.exists():
            for line in path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    seen.setdefault(rec["qh"], rec)
                except (ValueError, KeyError, TypeError):
                    bad += 1
        got = set(seen)
        missing, extra = len(want - got), len(got - want)
        scored = [seen[h] for h in want & got]
        acc = sum(bool(r.get("ok")) for r in scored) / len(scored) if scored else 0.0
        parsed = sum(bool(r.get("lenp")) for r in scored) / len(scored) if scored else 0.0
        report["benches"][bench] = {"n": len(want), "have": len(want & got), "missing": missing,
                                    "extra": extra, "bad_lines": bad, "acc": acc,
                                    "parsed": parsed}
        if missing or extra:
            report["ok"] = False
    report["ok"] = report["ok"] and report["done"]
    return report


def main(argv: list[str]) -> int:
    common.load_env_file()
    quiet = "--complete" in argv
    results = ev.OUT
    dirs = sorted(d for d in results.iterdir() if d.is_dir() and not d.name.startswith("_")) \
        if results.exists() else []
    if not dirs:
        if not quiet:
            print("results/ 裡還沒有任何模型結果。")
        return 1

    # 下載題目時 huggingface_hub 可能印進度條；--complete 模式的 stdout 要保持乾淨
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        expected = expected_hashes()

    names = {repo.replace("/", "_"): repo for repo, _ in common.read_models()}
    complete, broken = [], 0
    for d in dirs:
        r = inspect(d, expected)
        if r["ok"]:
            complete.append(d.name)
        if quiet:
            continue
        repo = names.get(d.name, d.name)
        if r["ok"]:
            head = "✓ 完成"
        elif any(b["extra"] for b in r["benches"].values()):
            head = "✗ 題目與目前資料集不符（混到舊版題目？）"
            broken += 1
        elif r["done"]:
            head = "✗ 有 .done 但缺題"
            broken += 1
        else:
            head = "… 未完成（重跑 ./run.sh 會從中斷處續跑）"
        print(f"{head}  {repo}")
        for bench, b in r["benches"].items():
            line = (f"    {bench:<8} {b['have']:>6}/{b['n']:<6} 正確率 {b['acc']:6.2%}"
                    f"  可解析 {b['parsed']:6.1%}")
            if b["extra"]:
                line += f"  多出 {b['extra']} 題"
            if b["bad_lines"]:
                line += f"  壞行 {b['bad_lines']}（中斷時寫到一半，無害）"
            print(line)
            if b["have"] and b["parsed"] < 0.5:
                print("    ⚠ 超過一半的回答抓不到選項。小模型偶爾如此；若較大的模型也這樣，"
                      "可能是環境問題，送回時請跟評測負責人提一聲")

    if quiet:
        print("\n".join(complete))
        return 0 if complete else 1
    print(f"\n可送回 {len(complete)}/{len(dirs)} 顆。")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
