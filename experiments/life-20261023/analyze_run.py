#!/usr/bin/env python3
"""Analyze a life run: session events + memory + rollout -> one metrics report.

Usage: python analyze_run.py <run_dir>
  <run_dir> contains session.ndjson, rollout/*.jsonl, data/memory.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_events(run: Path):
    f = run / "session.ndjson"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_rollout(run: Path):
    files = sorted((run / "rollout").glob("*.jsonl")) if (run / "rollout").exists() else []
    lines = []
    for f in files:
        lines.extend(json.loads(l) for l in f.read_text().splitlines() if l.strip())
    return lines


def load_memory(run: Path):
    f = run / "data" / "memory.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())


def main(run_dir: str) -> None:
    run = Path(run_dir)
    events = load_events(run)
    rollout = load_rollout(run)
    memory = load_memory(run)

    print(f"===== {run.name} 生命运行分析 =====\n")

    # 1. 事件概览
    counter = Counter(e.get("event") for e in events)
    print("[事件]", dict(counter))

    # 2. 轮次与裁决
    verdicts = Counter()
    for e in events:
        if e.get("event") == "exploration_end":
            verdict = e.get("status", "?").split("/")[-1]
            verdicts[verdict] += 1
    print("[裁决]", dict(verdicts))

    # 3. 节奏（rollout 时间戳）
    if rollout:
        ts = [r["ts"] for r in rollout]
        dur_min = None
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(ts[0])
            t1 = datetime.fromisoformat(ts[-1])
            dur_min = (t1 - t0).total_seconds() / 60
        except Exception:
            pass
        rounds = counter.get("exploration_end", 0)
        print(
            f"[节奏] 调用 {len(rollout)} 次 | 跨度 {dur_min and round(dur_min,1)} 分钟"
            f" | 轮 {rounds} | 分/轮 {dur_min and rounds and round(dur_min/rounds,1)}"
        )
        timeouts = sum(1 for r in rollout if (r.get("error") or "") and "wall clock" in str(r.get("error")))
        long_calls = [round(r["duration_ms"] / 1000) for r in rollout if r.get("duration_ms", 0) > 120000]
        print(f"[护栏] 超长调用(>120s): {len(long_calls)} 个 {long_calls[:8]}")
        # thinking 覆盖
        with_reasoning = sum(1 for r in rollout if r.get("reasoning"))
        print(f"[思考] 带思考轨迹的调用: {with_reasoning}/{len(rollout)}")

    # 4. 问题树
    questions = [m for m in memory if m.get("kind") in ("question", "contradiction")]
    statuses = Counter(m.get("status", "open") for m in questions)
    depths = Counter(m.get("depth", 0) for m in questions)
    resolved = statuses.get("resolved", 0) + statuses.get("decomposed", 0)
    rate = f"{resolved}/{len(questions)}" if questions else "0/0"
    print(f"[问题树] 总 {len(questions)} | 状态 {dict(statuses)} | 解决率 {rate} | 深度分布 {dict(depths)}")

    # 5. 知识沉淀
    beliefs = [
        m
        for m in memory
        if m.get("kind") in ("conjecture", "theory")
        and not str(m.get("content", "")).startswith("Run '")
    ]
    cred = Counter()
    for m in beliefs:
        c = float(m.get("credibility", 0))
        bucket = "≥0.8" if c >= 0.8 else "0.5-0.8" if c >= 0.5 else "0.3-0.5" if c >= 0.3 else "<0.3"
        cred[bucket] += 1
    print(f"[知识] 信念 {len(beliefs)} 条 | 置信分布 {dict(cred)}")

    # 6. 实验链
    experiments = [m for m in memory if m.get("kind") == "experiment"]
    print(f"[实验] 记录 {len(experiments)} 条")

    # 7. dream 与 wonder
    print(f"[梦] {counter.get('dream_end',0)} 次 | [好奇] 新问题 {sum(len(e.get('questions',[])) for e in events if e.get('event')=='wondered')} 个")

    # 8. 好奇列表顶部
    cur = run / "data" / "curiosity.md"
    if cur.exists():
        lines = [l for l in cur.read_text().splitlines() if l and l[0].isdigit()][:5]
        print("\n[当前最好奇]")
        for l in lines:
            print(" ", l[:110])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
