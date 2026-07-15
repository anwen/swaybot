# SwayBot

> A lightweight agent that learns to grow and evolve on its own.

SwayBot is designed to be small in footprint, clear in structure, and open-ended in capability. Instead of trying to do everything at once, it starts with a minimal core and learns to extend itself—adding skills, refining behavior, and adapting to new tasks through experience.

## Vision

We believe the next generation of agents should not be giant monoliths. They should be:

- **Lightweight** — easy to run, easy to understand, easy to modify.
- **Elegant** — every part has a reason to exist.
- **Self-improving** — able to observe, remember, and evolve its own behavior over time.

SwayBot is an experiment in making that kind of agent real.

## Core Ideas

- **Minimal core, growing surface** — start small, expand by learning rather than hard-coding.
- **Experience as structure** — the agent turns what it learns into reusable patterns, tools, and workflows.
- **Human-aligned evolution** — growth is guided by intent, feedback, and clear boundaries.

## Getting Started

SwayBot is built with Python 3.10+ and has no required runtime dependencies.

```bash
# Clone the repository
git clone https://github.com/askender/swaybot.git
cd swaybot

# Install in editable mode
pip install -e .

# Run the agent
python -m swaybot "count to 3" --max-steps 5
```

### Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

The minimal loop is `perceive → think → act → observe → loop`:

- `Environment` holds the task, step counter, and observation history.
- `Brain` decides the next action. `EchoBrain` is the default deterministic brain and requires no API key.
- `ToolRegistry` dispatches actions to tools (`echo`, `add`, `done`).
- `Agent` wires them together and runs until the task signals completion or the step budget is exhausted.

## Roadmap

- [x] Define the minimal agent loop
- [ ] Build memory and reflection primitives
- [ ] Add self-improvement mechanisms
- [ ] Document growth patterns and examples

## License

MIT © 2026 anwen
