import argparse

from .agent import Agent
from .brain import EchoBrain
from .memory import MemoryStore
from .reflection import Reflector


def main(argv: list[str] | None = None) -> int:
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
        "--memory", type=str, default=None, help="Path to persistent memory store"
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Reflect on the run and store insights (requires --memory)",
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
    args = parser.parse_args(argv)

    brain = EchoBrain()
    if args.brain == "llm":
        from .llm_brain import LLMBrain

        brain = LLMBrain(api_key=args.api_key, base_url=args.api_base, model=args.model)

    memory = MemoryStore(path=args.memory) if args.memory else None
    reflector = Reflector(memory) if memory and args.reflect else None
    agent = Agent(brain=brain, memory=memory, reflector=reflector)
    env = agent.run(args.task, max_steps=args.max_steps, reflect=args.reflect)
    for entry in env.history:
        print(f"[{entry['step']}] {entry['action']} -> {entry['result']}")
    return 0
