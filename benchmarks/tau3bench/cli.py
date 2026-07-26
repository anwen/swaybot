"""CLI entry point for running tau3-bench with SwayBot.

Usage::

    # Run mock domain with an LLM-backed SwayBot agent
    python -m benchmarks.tau3bench --domain mock --llm-agent gpt-4o

    # Run a specific task
    python -m benchmarks.tau3bench --domain airline --llm-agent gpt-4o \
        --task-ids create_task_1

    # Use SwayBot's EchoBrain (no LLM calls)
    python -m benchmarks.tau3bench --domain mock

Requires tau2-bench (Python >= 3.12, install via ``uv pip install tau2-bench``).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tau3-bench evaluation with SwayBot agent",
    )
    parser.add_argument(
        "--domain",
        default="mock",
        help="Domain to evaluate (mock, airline, retail, telecom, banking_knowledge)",
    )
    parser.add_argument(
        "--llm-agent",
        default=None,
        help="LLM model name for the SwayBot agent (e.g. gpt-4o). "
             "Omit to use EchoBrain (no LLM calls).",
    )
    parser.add_argument(
        "--llm-user",
        default="gpt-4o",
        help="LLM model for the user simulator (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM (or set SWAYBOT_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the LLM (or set SWAYBOT_API_BASE)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Specific task IDs to run (default: all tasks in domain)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum agent steps per task (default: 100)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Late import — tau2 requires Python >= 3.12
    try:
        from tau2.run import get_tasks, run_single_task
        from tau2.data_model.simulation import TextRunConfig
    except ImportError as exc:
        print(
            f"Error: {exc}\n\n"
            "tau2-bench requires Python >= 3.12. Install it with:\n"
            "  pip install tau2-bench\n"
            "or via uv:\n"
            "  uv pip install tau2-bench",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import and register the SwayBot agent factory
    from benchmarks.tau3bench.registry import create_swaybot_agent

    # Load tasks
    task_ids = args.task_ids if args.task_ids else None
    tasks = get_tasks(args.domain, task_ids=task_ids)
    if not tasks:
        print(f"No tasks found for domain '{args.domain}'", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(tasks)} task(s) from domain '{args.domain}'")

    config = TextRunConfig(
        domain=args.domain,
        agent="swaybot",
        user="llm_user",
        llm_agent=args.llm_agent or "",
        llm_user=args.llm_user,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
    )

    # Run each task
    results = []
    for task in tasks:
        result = run_single_task(config, task)
        results.append(result)
        reward = getattr(result, "reward", None)
        print(f"  Task {task.id}: reward={reward}")

    # Summary
    rewards = [getattr(r, "reward", 0) for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nAverage reward: {avg_reward:.4f} over {len(results)} tasks")


if __name__ == "__main__":
    main()
