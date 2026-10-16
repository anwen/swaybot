from pathlib import Path
from typing import Any

from .agent_events import RunBegin, RunEnd, StepEnd
from .storage import JSONLBackend, StorageBackend
from .tools import format_action
from .utils import utc_now_iso

TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class RunLogRecorder:
    """Agent-event consumer that writes the run log.

    Buffers StepEnd records and appends the full run record when RunEnd
    arrives — the same file ``python -m swaybot inspect`` reads. Construct
    via ``for_memory``; a disabled recorder's ``__call__`` is a no-op.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._steps: list[dict] = []
        self._task = ""
        self._max_steps = 0
        self._hypothesis: str | None = None

    @classmethod
    def for_memory(cls, memory_path: Path | str | None) -> "RunLogRecorder":
        return cls(run_log_path_for_memory(memory_path))

    def __call__(self, event: object) -> None:
        if self._path is None:
            return
        if isinstance(event, RunBegin):
            self._task = event.task
            self._max_steps = event.max_steps
            self._hypothesis = event.hypothesis
        elif isinstance(event, StepEnd):
            self._steps.append(self._record_step(event))
        elif isinstance(event, RunEnd):
            append_run(
                self._path,
                build_run_record(
                    self._task,
                    self._max_steps,
                    self._steps,
                    hypothesis=self._hypothesis,
                    reflections=list(event.reflections),
                ),
            )

    @staticmethod
    def _record_step(event: StepEnd) -> dict:
        record: dict = {
            "step": event.step,
            "action": event.action,
            "result": str(event.result),
        }
        if event.error:
            record["error"] = event.error
        metadata = event.metadata or {}
        for key in (
            "model_input_messages",
            "raw_output",
            "token_usage",
            "duration_ms",
        ):
            if metadata.get(key) is not None:
                record[key] = metadata[key]
        return record


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
        "created_at": utc_now_iso(),
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
