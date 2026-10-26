#!/usr/bin/env python3
"""Dev-loop runner for tau3 banking_knowledge: full transcripts + reward breakdown.

Unlike tau3-knowledge-20261023/run.py (A/B experiment), this runner exists to
support the iterative-improvement loop:

  run a batch of tasks -> dump FULL conversation + reward breakdown per task
  -> failure analysis on real transcripts -> harness/prompt improvement
  -> re-run the same batch and compare.

Usage:
  python run_dev.py --label base --tasks task_001 ... task_020
  python run_dev.py --label fix1 --tasks-file ids.txt

Outputs (under this directory):
  transcripts/<label>/<task>.json   full messages + reward_info breakdown
  results-<label>.jsonl             one line per task
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RUN_DIR = Path(__file__).parent
REPO_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

DOMAIN = "banking_knowledge"


def _env_from_dotenv():
    for line in Path(os.environ.get("SWAYBOT_ENV", REPO_ROOT / ".env")).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _setup(domain: str, retrieval: str | None):
    _env_from_dotenv()
    model = os.environ["SWAYBOT_MODEL"]
    os.environ["OPENAI_API_KEY"] = os.environ["SWAYBOT_API_KEY"]
    os.environ["OPENAI_API_BASE"] = os.environ["SWAYBOT_API_BASE"]
    os.environ["TAU2_DATA_DIR"] = os.path.expanduser(
        "~/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
    )
    os.environ.setdefault("SWAYBOT_LITELLM_TIMEOUT", "180")

    from benchmarks.tau3bench import registry  # noqa: F401
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import get_tasks, run_single_task

    def make_config():
        kwargs = {}
        if retrieval:
            kwargs["retrieval_config"] = retrieval
        return TextRunConfig(
            domain=domain,
            agent="swaybot",
            user="user_simulator",
            llm_agent=model,
            llm_user=f"openai/{model}",
            max_steps=int(os.environ.get("SWAYBOT_MAX_STEPS", "50")),
            num_trials=1,
            seed=0,
            **kwargs,
        )

    tasks = {t.id: t for t in get_tasks(domain)}
    return make_config, run_single_task, tasks


def _dump_reward_info(info) -> dict:
    if info is None:
        return {}
    try:
        return json.loads(info.model_dump_json(exclude_none=True))
    except Exception:
        return {"reward": float(getattr(info, "reward", 0.0))}


def _dump_messages(result) -> list[dict]:
    out = []
    for m in getattr(result, "messages", None) or []:
        entry = {"role": getattr(m, "role", "?")}
        tc = getattr(m, "tool_calls", None) or []
        if tc:
            entry["tool_calls"] = [
                {
                    "name": getattr(c, "name", ""),
                    "arguments": getattr(c, "arguments", {}),
                    "requestor": getattr(c, "requestor", ""),
                }
                for c in tc
            ]
        content = getattr(m, "content", "")
        if isinstance(content, str) and content:
            entry["content"] = content
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label, e.g. base / fix1")
    ap.add_argument("--domain", default=DOMAIN)
    ap.add_argument("--retrieval", default="bm25",
                    help="retrieval_config; pass 'none' to omit (airline/retail)")
    ap.add_argument("--tasks", nargs="*", default=[], help="task ids")
    ap.add_argument("--tasks-file", type=Path, default=None)
    ap.add_argument("--guidance", type=Path, default=None,
                    help="optional extra-guidance file (SWAYBOT_EXTRA_GUIDANCE_FILE)")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the domain")
    args = ap.parse_args()

    retrieval = None if args.retrieval == "none" else args.retrieval
    make_config, run_single_task, tasks = _setup(args.domain, retrieval)

    ids = list(args.tasks)
    if args.tasks_file:
        ids += [ln.strip() for ln in args.tasks_file.read_text().splitlines() if ln.strip()]
    if args.limit:
        ids += sorted(tasks)[: args.limit]
    missing = [i for i in ids if i not in tasks]
    if missing:
        sys.exit(f"unknown task ids: {missing}")

    if args.guidance:
        os.environ["SWAYBOT_EXTRA_GUIDANCE_FILE"] = str(args.guidance)
    else:
        os.environ.pop("SWAYBOT_EXTRA_GUIDANCE_FILE", None)

    tdir = RUN_DIR / "transcripts" / args.label
    tdir.mkdir(parents=True, exist_ok=True)
    rfile = RUN_DIR / f"results-{args.label}.jsonl"

    rewards = []
    for tid in ids:
        t0 = time.perf_counter()
        try:
            result = run_single_task(make_config(), tasks[tid])
            info = getattr(result, "reward_info", None)
            reward = float(getattr(info, "reward", 0.0)) if info else 0.0
            dump = {
                "task": tid,
                "reward": reward,
                "seconds": round(time.perf_counter() - t0, 1),
                "reward_info": _dump_reward_info(info),
                "messages": _dump_messages(result),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[{args.label}] {tid}: ERROR {type(exc).__name__}: {exc}", flush=True)
            reward = 0.0
            dump = {
                "task": tid,
                "reward": 0.0,
                "seconds": round(time.perf_counter() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "messages": [],
            }
        (tdir / f"{tid}.json").write_text(
            json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        with rfile.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task": tid, "reward": reward,
                                 "seconds": dump["seconds"]}, ensure_ascii=False) + "\n")
        rewards.append(reward)
        print(f"[{args.label}] {tid}: reward={reward:.2f} ({dump['seconds']}s)", flush=True)

    score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"== {args.label}: avg={score:.3f} over {len(rewards)} tasks ==", flush=True)


if __name__ == "__main__":
    main()
