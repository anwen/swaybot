"""CLI entry point for running AgentDojo benchmark with SwayBot.

Usage::

    # Run banking suite with gpt-4o
    python -m benchmarks.agentdojo --suite banking --model gpt-4o

    # Run all suites
    python -m benchmarks.agentdojo --model gpt-4o

    # Run with a specific attack
    python -m benchmarks.agentdojo --suite banking --model gpt-4o \
        --attack important_instructions

    # Run with defense
    python -m benchmarks.agentdojo --suite banking --model gpt-4o \
        --defense tool_filter
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgentDojo benchmark with SwayBot agent",
    )
    parser.add_argument(
        "--suite", "-s",
        nargs="*",
        default=None,
        help="Suite(s) to evaluate: banking, workspace, slack, travel. "
             "Default: all suites.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-05-13",
        help="Model name (default: gpt-4o-2024-05-13)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL (or set OPENAI_API_BASE)",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="Attack strategy (e.g. important_instructions, tool_knowledge, direct)",
    )
    parser.add_argument(
        "--defense",
        default=None,
        help="Defense mechanism (e.g. tool_filter, transformers_pi_detector)",
    )
    parser.add_argument(
        "--benchmark-version",
        default="v1.2.2",
        help="Benchmark version (default: v1.2.2)",
    )
    parser.add_argument(
        "--logdir",
        default="./runs/agentdojo",
        help="Directory for results (default: ./runs/agentdojo)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max parallel workers (default: 1)",
    )
    args = parser.parse_args()

    try:
        from agentdojo.agent_pipeline import AgentPipeline
        from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
        from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
        from agentdojo.scripts.benchmark import benchmark_suite, benchmark_suite_without_injections
        from agentdojo.task_suite.load_suites import get_suites
    except ImportError as exc:
        print(
            f"Error: {exc}\n\n"
            "agentdojo is required. Install it with:\n"
            "  pip install agentdojo",
            file=sys.stderr,
        )
        sys.exit(1)

    from benchmarks.agentdojo.adapter import SwayBotPipelineElement
    from swaybot.llm_brain import LLMBrain

    # Build SwayBot brain
    brain = LLMBrain(
        api_key=args.api_key,
        base_url=args.api_base,
        model=args.model,
    )
    swaybot_element = SwayBotPipelineElement(brain=brain)

    # Determine suites
    suite_names = args.suite or ["banking", "workspace", "slack", "travel"]
    suites = []
    for name in suite_names:
        try:
            suites.append(get_suites(name))
        except Exception:
            print(f"Warning: suite '{name}' not found, skipping", file=sys.stderr)

    if not suites:
        print("No valid suites specified", file=sys.stderr)
        sys.exit(1)

    # Build pipeline
    system_msg = "You are a helpful assistant. Use the available tools to complete the user's task."
    pipeline = AgentPipeline([
        SystemMessage(system_msg),
        InitQuery(),
        swaybot_element,
        ToolsExecutionLoop([ToolsExecutor(), swaybot_element]),
    ])

    # Run evaluation
    print(f"Running AgentDojo with suites: {suite_names}")
    for suite in suites:
        suite_name = getattr(suite, "name", str(suite))
        print(f"\n--- Suite: {suite_name} ---")
        if args.attack:
            results = benchmark_suite(
                pipeline=pipeline,
                suite=suite,
                attack=args.attack,
                logdir=args.logdir,
            )
        else:
            results = benchmark_suite_without_injections(
                pipeline=pipeline,
                suite=suite,
                logdir=args.logdir,
            )

        # Print results
        if hasattr(results, "utility_results"):
            utility = sum(1 for v in results.utility_results.values() if v) / max(1, len(results.utility_results))
            print(f"  Utility rate: {utility:.2%}")
        if hasattr(results, "security_results"):
            security = sum(1 for v in results.security_results.values() if v) / max(1, len(results.security_results))
            print(f"  Security rate: {security:.2%}")


if __name__ == "__main__":
    main()
