#!/usr/bin/env python3
"""Run tau3-bench domains in N parallel lanes respecting a concurrency cap.

The remote glm endpoint is slow and the user caps concurrency at 2.  The
default ``run_evaluation.py`` runs domains sequentially.  This driver launches
up to ``--lanes`` (default 2) domains concurrently, each as a subprocess that
itself runs tasks sequentially, so total in-flight requests == lanes.

Results are written to ``tau3_results/parallel_evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DOMAINS: list[tuple[str, int, str | None]] = [
    ("mock", 10, None),
    ("airline", 20, None),
    ("retail", 20, None),
    ("telecom", 20, None),
    ("banking_knowledge", 20, "golden_retrieval"),
]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", env.get("SWAYBOT_API_KEY", ""))
    env.setdefault("OPENAI_API_BASE", env.get("SWAYBOT_API_BASE", ""))
    gateway = env.get("SWAYBOT_API_BASE", "")
    if gateway:
        host = re.sub(r"^https?://", "", gateway).split("/")[0].split(":")[0]
        no_proxy = ",".join(h for h in (host, "127.0.0.1", "localhost") if h)
        env.setdefault("NO_PROXY", no_proxy)
        env.setdefault("no_proxy", no_proxy)
    env.setdefault(
        "TAU2_DATA_DIR",
        str(
            Path.home()
            / ".cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
        ),
    )
    return env


def _run_domain(
    domain: str,
    limit: int,
    retrieval: str | None,
    llm_agent: str,
    llm_user: str,
    repo_root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    cmd = [
        "python",
        "-m",
        "benchmarks.tau3bench",
        "--domain",
        domain,
        "--llm-agent",
        llm_agent,
        "--llm-user",
        llm_user,
        "--limit",
        str(limit),
        "--max-steps",
        "50",
    ]
    if retrieval:
        cmd.extend(["--retrieval-config", retrieval])

    start = time.perf_counter()
    log_path = log_dir / f"{domain}.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            env=_env(),
            cwd=str(repo_root),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - start

    stdout = log_path.read_text(encoding="utf-8", errors="replace")

    summary: dict[str, Any] = {
        "domain": domain,
        "limit": limit,
        "elapsed_sec": round(elapsed, 1),
        "log_file": str(log_path),
    }
    # Even on non-zero exit, harvest per-task rewards from the log so a crash
    # on the *last* task doesn't erase the rewards of the tasks before it.
    rewards: list[float] = []
    for m in re.finditer(r"Task\s+\S+:\s+reward=([0-9.]+)", stdout):
        rewards.append(float(m.group(1)))
    summary["task_rewards"] = rewards
    summary["num_tasks_completed"] = len(rewards)
    if rewards:
        summary["avg_reward"] = sum(rewards) / len(rewards)

    if proc.returncode != 0:
        summary["error"] = f"exit code {proc.returncode}"
        summary["stderr_tail"] = stdout[-3000:]
        return summary

    avg_match = re.search(r"Average reward:\s+([0-9.]+)", stdout)
    tasks_match = re.search(r"Running\s+(\d+)\s+task\(s\)", stdout)
    if avg_match:
        summary["avg_reward"] = float(avg_match.group(1))
    if tasks_match:
        summary["num_tasks"] = int(tasks_match.group(1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tau3-bench domains in parallel lanes.",
    )
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--llm-agent", default="glm-5.1-fp8")
    parser.add_argument("--llm-user", default="openai/glm-5.1-fp8")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Subset of domain names to run (default: all). "
        "Useful for re-running only domains that failed previously.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path("tau3_results")
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    selected = DOMAINS
    if args.domains:
        wanted = set(args.domains)
        selected = [d for d in DOMAINS if d[0] in wanted]
        missing = wanted - {d[0] for d in selected}
        if missing:
            parser.error(f"Unknown domain(s): {', '.join(sorted(missing))}")

    import concurrent.futures

    summaries: list[dict[str, Any]] = [None] * len(selected)  # type: ignore[list-item]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.lanes) as pool:
        futures = {
            pool.submit(
                _run_domain,
                domain,
                limit,
                retrieval,
                args.llm_agent,
                args.llm_user,
                repo_root,
                log_dir,
            ): idx
            for idx, (domain, limit, retrieval) in enumerate(selected)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                summaries[idx] = fut.result()
            except Exception as exc:  # pragma: no cover
                summaries[idx] = {"domain": selected[idx][0], "error": str(exc)}
            print(
                f"[done] {summaries[idx].get('domain')}: "
                f"reward={summaries[idx].get('avg_reward')} "
                f"({summaries[idx].get('elapsed_sec')}s)",
                flush=True,
            )

    tag = "_".join(d[0] for d in selected) if args.domains else "all"
    output_path = output_dir / f"parallel_evaluation_{tag}.json"
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSummary written to {output_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
