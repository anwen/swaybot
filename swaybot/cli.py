import argparse
import os
from pathlib import Path

from .agent import Agent
from .brain import EchoBrain
from .memory import MemoryStore
from .reflection import Reflector
from .tools import format_action


def _load_dotenv(path: Path | str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ.

    Does not overwrite existing environment variables. Values may be quoted.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_data_dir() -> Path:
    return Path.home() / ".swaybot"


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="SwayBot: a lightweight self-evolving agent"
    )
    parser.add_argument("task", help="Task for the agent")
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
        "--api-key", type=str, default=None, help="LLM API key (or SWAYBOT_API_KEY)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="LLM API base URL (or SWAYBOT_API_BASE)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="LLM model name (or SWAYBOT_MODEL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw action dicts alongside formatted output",
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

        brain = LLMBrain(api_key=args.api_key, base_url=args.api_base, model=args.model)

    memory = MemoryStore(path=memory_path) if memory_path else None
    reflector = Reflector(memory) if memory and args.reflect else None
    agent = Agent(brain=brain, memory=memory, reflector=reflector)
    env = agent.run(
        args.task, max_steps=args.max_steps, reflect=args.reflect, plan=args.plan
    )
    for entry in env.history:
        action = entry["action"]
        result = entry["result"]
        line = f"Step {entry['step']}: {format_action(action)} → {result}"
        if args.verbose:
            line = f"{line}  {action}"
        print(line)
    return 0
