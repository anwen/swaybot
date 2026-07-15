import argparse

from .agent import Agent
from .memory import MemoryStore
from .reflection import Reflector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SwayBot: a lightweight self-evolving agent"
    )
    parser.add_argument("task", help="Task for the agent")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps")
    parser.add_argument(
        "--memory", type=str, default=None, help="Path to persistent memory store"
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Reflect on the run and store insights (requires --memory)",
    )
    args = parser.parse_args(argv)

    memory = MemoryStore(path=args.memory) if args.memory else None
    reflector = Reflector(memory) if memory and args.reflect else None
    agent = Agent(memory=memory, reflector=reflector)
    env = agent.run(args.task, max_steps=args.max_steps, reflect=args.reflect)
    for entry in env.history:
        print(f"[{entry['step']}] {entry['action']} -> {entry['result']}")
    return 0
