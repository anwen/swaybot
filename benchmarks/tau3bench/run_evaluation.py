"""Run a sampled tau3-bench evaluation across domains.

This driver shells out to ``python -m benchmarks.tau3bench`` once per domain
so that tau2's module-level data directory is set cleanly for each run.  It
collects the per-domain average reward and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


DOMAINS = [
    ("mock", 10, None),
    ("airline", 20, None),
    ("retail", 20, None),
    ("telecom", 20, None),
    # banking_knowledge tasks are evaluated with oracle retrieval so we can
    # focus on reasoning and credit-card recommendation accuracy.
    ("banking_knowledge", 20, "golden_retrieval"),
]


def _env() -> dict[str, str]:
    """Return the current environment plus tau2/OpenAI aliases for LiteLLM."""
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", env.get("SWAYBOT_API_KEY", ""))
    env.setdefault("OPENAI_API_BASE", env.get("SWAYBOT_API_BASE", ""))
    # litellm (used by tau2's user_simulator) builds its own OpenAI client and
    # honors httpx's trust_env=True, which picks up the macOS system proxy at
    # 127.0.0.1:7890 and corrupts TLS to the gateway.  Bypass the proxy for the
    # gateway host so LiteLLM connects directly.  See swaybot.models._build_http_client
    # for the matching fix on SwayBot's own OpenAI client.
    gateway = env.get("SWAYBOT_API_BASE", "")
    if gateway:
        host = re.sub(r"^https?://", "", gateway).split("/")[0].split(":")[0]
        no_proxy = ",".join(
            h for h in (host, "127.0.0.1", "localhost") if h
        )
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


def run_domain(
    domain: str,
    limit: int,
    retrieval_config: str | None,
    *,
    llm_agent: str,
    llm_user: str,
) -> dict:
    """Run one domain and return a summary dict."""
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
    if retrieval_config:
        cmd.extend(["--retrieval-config", retrieval_config])

    print(f"\n=== Running {domain} (limit={limit}) ===")
    result = subprocess.run(
        cmd,
        env=_env(),
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    stdout = result.stdout
    stderr = result.stderr
    if result.returncode != 0:
        print(f"Domain {domain} failed with exit code {result.returncode}")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)
        return {
            "domain": domain,
            "limit": limit,
            "error": f"exit code {result.returncode}",
            "stderr_tail": stderr[-2000:],
        }

    avg_match = re.search(r"Average reward:\s+([0-9.]+)", stdout)
    tasks_match = re.search(r"Running\s+(\d+)\s+task\(s\)", stdout)
    avg_reward = float(avg_match.group(1)) if avg_match else None
    num_tasks = int(tasks_match.group(1)) if tasks_match else None

    summary = {
        "domain": domain,
        "limit": limit,
        "num_tasks": num_tasks,
        "avg_reward": avg_reward,
    }
    print(f"{domain}: avg_reward={avg_reward} over {num_tasks} tasks")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sampled tau3-bench evaluation across domains.",
    )
    parser.add_argument(
        "--llm-agent",
        default="glm-5.1-fp8",
        help="LLM model name for the SwayBot agent (default: glm-5.1-fp8)",
    )
    parser.add_argument(
        "--llm-user",
        default="openai/glm-5.1-fp8",
        help="LLM model name for the tau2 user simulator (default: openai/glm-5.1-fp8)",
    )
    args = parser.parse_args()

    output_dir = Path("tau3_results")
    output_dir.mkdir(exist_ok=True)
    summaries = []
    for domain, limit, retrieval_config in DOMAINS:
        summaries.append(
            run_domain(
                domain,
                limit,
                retrieval_config,
                llm_agent=args.llm_agent,
                llm_user=args.llm_user,
            )
        )

    output_path = output_dir / "sample_evaluation.json"
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSummary written to {output_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
