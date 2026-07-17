import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import JSONLBackend, StorageBackend

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def run_log_path_for_memory(memory_path: Path | str | None) -> Path | None:
    """Return the run-log path that lives next to a memory file."""
    if not memory_path:
        return None
    return Path(memory_path).with_name("runs.jsonl")


def _backend_for_path(path: Path | str | None) -> tuple[StorageBackend, str] | None:
    if not path:
        return None
    run_path = Path(path)
    return JSONLBackend(run_path.parent), run_path.stem


def append_run(path: Path | str, record: dict) -> None:
    """Append one JSON-lines run record."""
    backend_info = _backend_for_path(path)
    if backend_info is None:
        return
    backend, key = backend_info
    backend.append(key, record)


def load_runs(path: Path | str | None) -> list[dict]:
    """Load JSON-lines run records, skipping malformed lines."""
    backend_info = _backend_for_path(path)
    if backend_info is None:
        return []
    backend, key = backend_info
    return backend.load_stream(key)


def build_run_record(
    task: str,
    max_steps: int,
    steps: list[dict],
    hypothesis: str | None = None,
    reflections: list[str] | None = None,
) -> dict:
    """Build a compact, inspectable record for one agent run."""
    final_answer = None
    for step in reversed(steps):
        action = step.get("action") or {}
        if action.get("name") == "final_answer":
            args = action.get("args") or {}
            final_answer = args.get("answer", step.get("result"))
            break

    totals: dict[str, float] = {key: 0 for key in TOKEN_KEYS}
    duration_total = 0.0
    for step in steps:
        usage = step.get("token_usage") or {}
        if isinstance(usage, dict):
            for key in TOKEN_KEYS:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += value
        duration = step.get("duration_ms")
        if isinstance(duration, (int, float)):
            duration_total += duration

    return {
        "task": task,
        "hypothesis": hypothesis or "",
        "max_steps": max_steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "final_answer": final_answer,
        "reflections": reflections or [],
        "totals": {**totals, "duration_ms": duration_total},
    }


def select_runs(
    runs: list[dict], task: str | None = None, limit: int = 1
) -> list[dict]:
    """Select runs by optional task substring, keeping chronological order."""
    selected = runs
    if task:
        selected = [run for run in selected if task in str(run.get("task", ""))]
    if limit <= 0:
        return selected
    return selected[-limit:]


def format_run(record: dict) -> str:
    """Render one run record for the CLI."""
    lines = [f"Task: {record.get('task', '')}"]
    hypothesis = record.get("hypothesis")
    if hypothesis:
        lines.append(f"Hypothesis: {hypothesis}")
    created_at = record.get("created_at")
    if created_at:
        lines.append(f"Created: {created_at}")

    steps = record.get("steps") or []
    lines.append(f"Steps: {len(steps)}")
    final_answer = record.get("final_answer")
    if final_answer is not None:
        lines.append(f"Final answer: {final_answer}")

    totals = record.get("totals") or {}
    if totals:
        lines.append(
            "Totals: "
            f"prompt={_format_number(totals.get('prompt_tokens'))} "
            f"completion={_format_number(totals.get('completion_tokens'))} "
            f"total={_format_number(totals.get('total_tokens'))} "
            f"duration={_format_number(totals.get('duration_ms'))} ms"
        )

    reflections = record.get("reflections") or []
    if reflections:
        lines.append("Reflections:")
        for reflection in reflections:
            lines.append(f"  - {reflection}")

    for step in steps:
        action = step.get("action") or {}
        result = step.get("result")
        lines.append(f"Step {step.get('step')}: {_format_action(action)} → {result}")
        raw_output = step.get("raw_output")
        if raw_output:
            lines.append(f"  raw output: {raw_output}")
        usage = step.get("token_usage") or {}
        if usage:
            lines.append(
                "  tokens: "
                f"prompt={_format_number(usage.get('prompt_tokens'))} "
                f"completion={_format_number(usage.get('completion_tokens'))} "
                f"total={_format_number(usage.get('total_tokens'))}"
            )
        duration = step.get("duration_ms")
        if duration is not None:
            lines.append(f"  duration: {_format_number(duration)} ms")
        error = step.get("error")
        if error:
            lines.append(f"  error: {error}")
    return "\n".join(lines)


def _format_action(action: dict) -> str:
    name = action.get("name", "unknown")
    args = action.get("args") or {}
    args_str = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({args_str})"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
