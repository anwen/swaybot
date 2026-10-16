import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .agent_events import Event, RunBegin, StepEnd, to_ndjson
from .brain import EchoBrain
from .dotenv import load_dotenv
from .explorer import Explorer
from .memory import MemoryStore
from .reflection import Reflector
from .run_log import format_run, load_runs, run_log_path_for_memory, select_runs
from .security import PermissionMode, PermissionPolicy
from .task_state import Environment
from .tools import format_action


def _default_data_dir() -> Path:
    return Path.home() / ".swaybot"


def _inspect_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="swaybot inspect",
        description="Inspect recorded runs without calling the LLM",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for SwayBot data (default: ~/.swaybot)",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Path to persistent memory store (default: DATA_DIR/memory.json)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Show runs whose task contains this text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of matching runs to show (default: 1; 0 shows all)",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Show only the latest matching run (default)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    memory_path = Path(args.memory) if args.memory else data_dir / "memory.json"
    run_log_path = run_log_path_for_memory(memory_path)
    runs = load_runs(run_log_path)
    limit = 1 if args.last else args.limit
    selected = select_runs(runs, task=args.task, limit=limit)
    if not selected:
        print(f"No runs found in {run_log_path}")
        return 1

    for index, record in enumerate(selected):
        if index:
            print("\n---")
        print(format_run(record))
    return 0


def _make_output_sink(args: argparse.Namespace) -> Callable[[Event], None]:
    """Render agent events live as NDJSON (json mode) or text lines."""
    run_type = "explore" if args.explore else "task"

    def sink(event: Event) -> None:
        if args.output_format == "json":
            payload = to_ndjson(event)
            if payload is None:
                return
            if isinstance(event, RunBegin):
                ordered = {
                    "event": "run_start",
                    "type": run_type,
                    "task": event.task,
                    "brain": args.brain,
                    "permission_mode": args.permission_mode,
                    "max_steps": event.max_steps,
                }
                if event.hypothesis:
                    ordered["hypothesis"] = event.hypothesis
                payload = ordered
            elif isinstance(event, StepEnd) and args.verbose:
                payload["raw_action"] = event.action
            print(json.dumps(payload, ensure_ascii=False))
            return
        if isinstance(event, RunBegin):
            if run_type == "explore":
                print(f"Exploration: {event.task}")
                if event.hypothesis:
                    print(f"Hypothesis: {event.hypothesis}")
        elif isinstance(event, StepEnd):
            line = f"Step {event.step}: {format_action(event.action)} → {event.result}"
            if args.verbose:
                line = f"{line}  {event.action}"
            print(line)

    return sink


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "inspect":
        return _inspect_main(argv[1:])

    parser = argparse.ArgumentParser(
        description="SwayBot: a lightweight self-evolving agent"
    )
    parser.add_argument("task", nargs="?", default="explore", help="Task for the agent")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps")
    parser.add_argument(
        "--brain",
        choices=["echo", "llm"],
        default="echo",
        help="Brain to use (default: echo)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for SwayBot data (default: ~/.swaybot)",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Path to persistent memory store (default: DATA_DIR/memory.json)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory persistence",
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=None,
        help="Directory of Markdown skill files to load (default: ./skills)",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Reflect on the run and store insights",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Ask the brain to produce a plan before acting",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Generate and run an exploratory task instead of using TASK",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="LLM API key (or SWAYBOT_API_KEY)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="LLM API base URL (or SWAYBOT_API_BASE)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="LLM model name (or SWAYBOT_MODEL)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="LLM request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw action dicts alongside formatted output",
    )
    parser.add_argument(
        "--permission-mode",
        choices=[m.value for m in PermissionMode],
        default=None,
        help="Permission mode: read-only, workspace-write, or danger (default: workspace-write)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format: human-readable text or NDJSON events (default: text)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()

    memory_path: Path | None = None
    if not args.no_memory:
        if args.memory:
            memory_path = Path(args.memory)
        else:
            memory_path = data_dir / "memory.json"

    brain = EchoBrain()
    if args.brain == "llm":
        from .llm_brain import LLMBrain

        llm_kwargs = {
            "api_key": args.api_key,
            "base_url": args.api_base,
            "model": args.model,
        }
        if args.timeout is not None:
            llm_kwargs["timeout"] = args.timeout
        brain = LLMBrain(**llm_kwargs)

    permission_policy = None
    if args.permission_mode is not None:
        permission_policy = PermissionPolicy(mode=args.permission_mode)

    memory = MemoryStore(path=memory_path) if memory_path else None
    reflector = Reflector(memory) if memory and (args.reflect or args.explore) else None
    skills = None
    if args.skills is not None:
        from .skills import SkillLibrary

        skills = SkillLibrary(args.skills)
    agent = Agent(
        brain=brain,
        memory=memory,
        reflector=reflector,
        permission_policy=permission_policy,
        skills=skills,
        on_event=_make_output_sink(args),
    )

    if args.explore:
        explorer = Explorer(agent, max_steps=args.max_steps)
        exploration, env = explorer.run()
    else:
        env = agent.run(
            args.task, max_steps=args.max_steps, reflect=args.reflect, plan=args.plan
        )
    return 0
