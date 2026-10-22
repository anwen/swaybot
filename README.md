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

- `Environment` (in `swaybot/task_state.py`) holds the task, step counter, and observation history.
- `Brain` (in `swaybot/brain.py`) decides the next action. `EchoBrain` is the default deterministic brain and requires no API key.
- `ToolRegistry` dispatches actions to tools. Built-ins include `echo`, `add`, `done`, `final_answer`, and optional workspace/shell/web tools (`read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `grep`, `run_shell_command`, `web_search`, `web_fetch`).
- `Agent` wires them together and runs until the task signals completion or the step budget is exhausted.

The CLI supports `--brain {echo,llm}`, `--reflect`, `--plan`, `--explore`, `--permission-mode {read-only,workspace-write,danger}`, `--output-format {text,json}`, and memory paths via `--data-dir` / `--memory` / `--no-memory`.

## Skills

SwayBot can load an independent `skills/` directory as a skill library. Each skill is a Markdown file (nanobot-style `skills/<name>/SKILL.md` or flat `skills/*.md`) with YAML frontmatter (`name`, `description`, `tags`). On each turn, skills whose name, description, or tags match the task are injected into the system prompt.

```bash
python -m swaybot "interact with github" --brain llm --skills ./skills
```

## Memory

SwayBot can optionally keep a `MemoryStore` to record experiences, facts, theories, conjectures, and inspirations. Memories are scoped as either `short_term` (raw per-run experiences) or `long_term` (validated knowledge produced by reflection). Each memory carries source, evidence, credibility, surprise, and tags so the agent can later retrieve relevant long-term context or search for counterexamples. Short-term steps are auto-compacted (summarized into long-term memories) when they exceed a step or context-size budget — on by default, disable with `--no-compaction`.

```python
from swaybot import Agent, MemoryStore

store = MemoryStore(path="~/.swaybot/memory.json")
agent = Agent(memory=store)
agent.run("explore a topic", max_steps=5)
```

## Reflection

After a run, SwayBot can reflect on what happened: summarize the experience, flag surprising events, detect contradictions in memory, and verify claims against stored facts. Reflections are stored as `long_term` `theory` memories, creating a self-improving loop where raw experience gradually turns into structured knowledge.

```bash
python -m swaybot "explore colors" --max-steps 5 --reflect
```

Output now reads as readable tool calls:

```text
Step 1: echo(message="thinking...") → thinking...
Step 2: done() → finished
```

## LLM Brain

SwayBot can use an OpenAI-compatible chat model as its brain. Install the optional dependency and run with `--brain llm`:

```bash
pip install -e ".[llm]"
export SWAYBOT_API_KEY="your-key"
export SWAYBOT_API_BASE="https://api.example.com/v1"
export SWAYBOT_MODEL="your-model"
python -m swaybot "What is 2+2?" --brain llm --max-steps 3 --reflect
```

Alternatively, create a `.env` file (see `.env.example`) in the project root. The CLI loads it automatically and reads `SWAYBOT_API_KEY`, `SWAYBOT_API_BASE`, and `SWAYBOT_MODEL` from there. By default, memory is persisted to `~/.swaybot/memory.json`; use `--data-dir` or `--memory` to customize the location, or `--no-memory` to disable persistence.

The default brain remains `EchoBrain`, so the core package stays dependency-free.

## HTTP API & WebUI

Run the server with:

```bash
pip install -e ".[api]"
python -m swaybot.api
```

It exposes an OpenAI-compatible `/v1/chat/completions` endpoint (including SSE streaming), session messages/history, and a long-running goal coordinator under `/v1/goals`.

## Roadmap

- [x] Define the minimal agent loop
- [x] Build memory primitives
- [x] Build reflection primitives
- [x] Add LLM-powered reasoning
- [x] Add self-improvement mechanisms
- [ ] Document growth patterns and examples

## License

MIT © 2026 anwen
