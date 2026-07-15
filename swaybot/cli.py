import argparse

from .agent import Agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SwayBot: a lightweight self-evolving agent"
    )
    parser.add_argument("task", help="Task for the agent")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps")
    args = parser.parse_args(argv)

    agent = Agent()
    env = agent.run(args.task, max_steps=args.max_steps)
    for entry in env.history:
        print(f"[{entry['step']}] {entry['action']} -> {entry['result']}")
    return 0
