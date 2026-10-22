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


def _domain_accepts_retrieval_variant(domain: str) -> bool:
    """Whether the domain's environment constructor takes retrieval_variant.

    Probe the signature rather than hardcoding a domain list, so this
    survives future tau2 signature changes in either direction.
    """
    try:
        import importlib
        import inspect

        module = importlib.import_module(f"tau2.domains.{domain}.environment")
        params = inspect.signature(module.get_environment).parameters
        return "retrieval_variant" in params
    except Exception:
        return True  # cannot probe: keep legacy behavior, let tau2 complain


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
        "--user",
        default="user_simulator",
        help="User simulator constructor to use (default: user_simulator)",
    )
    parser.add_argument(
        "--llm-user",
        default="gpt-4o",
        help="LLM model name for the user simulator (default: gpt-4o)",
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
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks in the domain",
    )
    parser.add_argument(
        "--retrieval-config",
        default=None,
        help="Retrieval config for banking_knowledge (e.g. bm25, no_knowledge)",
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

    # Import to register the SwayBot agent factory with tau2's registry.
    from benchmarks.tau3bench import registry  # noqa: F401

    # Load tasks
    task_ids = args.task_ids if args.task_ids else None
    tasks = get_tasks(args.domain, task_ids=task_ids)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print(f"No tasks found for domain '{args.domain}'", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(tasks)} task(s) from domain '{args.domain}'")

    config_kwargs = dict(
        domain=args.domain,
        agent="swaybot",
        user=args.user,
        llm_agent=args.llm_agent or "",
        llm_user=args.llm_user,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
    )
    if args.retrieval_config:
        if _domain_accepts_retrieval_variant(args.domain):
            config_kwargs["retrieval_config"] = args.retrieval_config
        else:
            # tau2's 2026-10 signature change: only banking_knowledge's
            # environment constructor takes retrieval_variant; airline/
            # retail/telecom reject the kwarg outright.
            print(
                f"  note: domain '{args.domain}' takes no retrieval variant; "
                f"ignoring --retrieval-config '{args.retrieval_config}'"
            )

    config = TextRunConfig(**config_kwargs)

    # Run each task
    results = []
    for task in tasks:
        try:
            result = run_single_task(config, task)
        except Exception as exc:
            # A single task must not abort the whole domain run.  The remote
            # endpoint throws intermittent 500s / connection errors, and the
            # NL-assertion evaluator can raise JSONDecodeError on non-JSON judge
            # output.  Record reward 0 and continue so we still get a domain
            # average over the surviving tasks.
            print(f"  Task {task.id}: reward=0.0 (error: {type(exc).__name__}: {exc})")
            results.append(None)
            continue
        results.append(result)
        reward = _reward_from_result(result)
        print(f"  Task {task.id}: reward={reward}")

    # Summary
    rewards = [_reward_from_result(r) for r in results if r is not None]
    failed = sum(1 for r in results if r is None)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    if failed:
        print(
            f"\nAverage reward: {avg_reward:.4f} over {len(rewards)} tasks "
            f"({failed} task(s) failed with errors and scored 0)"
        )
    else:
        print(f"\nAverage reward: {avg_reward:.4f} over {len(results)} tasks")


def _reward_from_result(result) -> float:
    """Extract the scalar reward from a tau2 SimulationRun result."""
    if result is None:
        return 0.0
    reward_info = getattr(result, "reward_info", None)
    if reward_info is not None:
        return float(getattr(reward_info, "reward", 0.0))
    return float(getattr(result, "reward", 0.0))


if __name__ == "__main__":
    main()
