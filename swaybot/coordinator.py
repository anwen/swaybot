"""Sustained-goal / long-task coordinator with state-machine resume."""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .events import Event, EventBus, InMemoryEventBus
from .scheduler import OneShotSchedule, Scheduler
from .storage import InMemoryBackend, StorageBackend


GOAL_PENDING = "pending"
GOAL_PLANNING = "planning"
GOAL_RUNNING = "running"
GOAL_WAITING = "waiting"
GOAL_DONE = "done"
GOAL_FAILED = "failed"
GOAL_CANCELLED = "cancelled"

SUBTASK_PENDING = "pending"
SUBTASK_RUNNING = "running"
SUBTASK_DONE = "done"
SUBTASK_FAILED = "failed"


@dataclass
class Subtask:
    """A single step toward a larger goal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    status: str = SUBTASK_PENDING
    result: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Goal:
    """A sustained goal broken into subtasks."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    context: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    state: str = GOAL_PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def status(self) -> str:
        """Backward-compatible alias for ``state``."""
        return self.state


class GoalStateMachine:
    """Persist and transition goal state using a StorageBackend.

    Each goal is stored under the key ``goal:{id}``. The state machine only
    handles data; it does not execute subtasks.
    """

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend

    def create(self, goal: Goal) -> None:
        self._touch(goal)
        self.save(goal)

    def save(self, goal: Goal) -> None:
        self._touch(goal)
        self.backend.save(self._key(goal.id), asdict(goal))

    def load(self, goal_id: str) -> Goal:
        data = self.backend.load(self._key(goal_id))
        if not isinstance(data, dict):
            raise KeyError(goal_id)
        return self._deserialize(data)

    def list_goals(self) -> list[Goal]:
        goals: list[Goal] = []
        for key in self.backend.list_keys("goal:"):
            data = self.backend.load(key)
            if isinstance(data, dict):
                goals.append(self._deserialize(data))
        return goals

    def transition(
        self,
        goal: Goal,
        state: str | None = None,
        subtask: Subtask | None = None,
        subtask_status: str | None = None,
        subtask_result: str | None = None,
    ) -> None:
        if state is not None:
            goal.state = state
        if subtask is not None and subtask_status is not None:
            subtask.status = subtask_status
            if subtask_result is not None:
                subtask.result = subtask_result
        self.save(goal)

    @staticmethod
    def _key(goal_id: str) -> str:
        return f"goal:{goal_id}"

    @staticmethod
    def _touch(goal: Goal) -> None:
        goal.updated_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> Goal:
        subtasks = [
            Subtask(**{k: v for k, v in s.items() if k in Subtask.__dataclass_fields__})
            for s in data.get("subtasks", [])
        ]
        goal_fields = {k: v for k, v in data.items() if k in Goal.__dataclass_fields__}
        goal_fields["subtasks"] = subtasks
        return Goal(**goal_fields)


class GoalCoordinator:
    """Break a long goal into subtasks and execute them sequentially.

    ``agent_factory`` should return an ``Agent`` instance for each subtask.

    The coordinator can persist state to a ``StorageBackend`` and execute
    subtasks through an async ``Scheduler``. This allows goals to resume after
    a restart and failed subtasks to be retried.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        state_path: Path | str | None = None,
        max_subtask_steps: int = 10,
        backend: StorageBackend | None = None,
        scheduler: Scheduler | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.max_subtask_steps = max_subtask_steps

        # Backward compatibility: if no backend is supplied but a file path is,
        # keep the original append-only JSONL behavior for load_history().
        self._legacy_state_path = Path(state_path) if state_path else None
        if backend is not None:
            self.backend = backend
        elif state_path is not None:
            self.backend = InMemoryBackend()
        else:
            self.backend = InMemoryBackend()

        self.state_machine = GoalStateMachine(self.backend)
        self.scheduler = scheduler or Scheduler()
        self.event_bus = event_bus or InMemoryEventBus()
        self._owns_scheduler = scheduler is None

    def run(self, description: str, context: str = "") -> Goal:
        """Plan, execute, and record a goal."""
        return asyncio.run(self._run(description, context))

    async def arun(self, description: str, context: str = "") -> Goal:
        """Async version of ``run`` for use inside an event loop."""
        return await self._run(description, context)

    async def _run(self, description: str, context: str = "") -> Goal:
        goal = Goal(description=description, context=context)
        self.state_machine.create(goal)

        self.state_machine.transition(goal, GOAL_PLANNING)
        goal.subtasks = self._plan(goal)
        if not goal.subtasks:
            goal.subtasks.append(Subtask(description=description))
        self.state_machine.save(goal)

        self.state_machine.transition(goal, GOAL_RUNNING)
        await self._start_scheduler()

        try:
            for subtask in goal.subtasks:
                if goal.state == GOAL_CANCELLED:
                    break
                await self._execute_subtask(goal, subtask)
        finally:
            if self._owns_scheduler:
                await self.scheduler.stop()

        failed = any(s.status == SUBTASK_FAILED for s in goal.subtasks)
        final_state = GOAL_FAILED if failed else GOAL_DONE
        self.state_machine.transition(goal, final_state)
        self._legacy_save(goal)
        return goal

    async def resume(self, goal_id: str) -> Goal:
        """Resume a previously saved goal, retrying pending/failed subtasks."""
        goal = self.state_machine.load(goal_id)
        await self._start_scheduler()
        try:
            for subtask in goal.subtasks:
                if subtask.status in (SUBTASK_PENDING, SUBTASK_FAILED):
                    await self._execute_subtask(goal, subtask)
        finally:
            if self._owns_scheduler:
                await self.scheduler.stop()
        failed = any(s.status == SUBTASK_FAILED for s in goal.subtasks)
        self.state_machine.transition(
            goal, GOAL_FAILED if failed else GOAL_DONE
        )
        return goal

    async def retry(self, goal_id: str) -> Goal:
        """Reset failed subtasks to pending and re-run them."""
        goal = self.state_machine.load(goal_id)
        for subtask in goal.subtasks:
            if subtask.status == SUBTASK_FAILED:
                subtask.status = SUBTASK_PENDING
                subtask.result = ""
        self.state_machine.save(goal)
        return await self.resume(goal_id)

    async def cancel(self, goal_id: str) -> Goal:
        """Mark a goal as cancelled."""
        goal = self.state_machine.load(goal_id)
        self.state_machine.transition(goal, GOAL_CANCELLED)
        return goal

    async def _execute_subtask(self, goal: Goal, subtask: Subtask) -> None:
        self.state_machine.transition(
            goal, GOAL_RUNNING, subtask, SUBTASK_RUNNING
        )

        completed = asyncio.Event()
        payload: dict[str, Any] = {}

        async def on_complete(event: Event) -> None:
            if event.payload.get("subtask_id") == subtask.id:
                payload.update(event.payload)
                completed.set()

        self.event_bus.subscribe("subtask.completed", on_complete)
        self.scheduler.add_job(
            self._subtask_job,
            OneShotSchedule(),
            args=(goal.id, subtask.id),
            job_id=subtask.id,
        )
        await completed.wait()

        status = payload.get("status", SUBTASK_FAILED)
        result = payload.get("result", "")
        self.state_machine.transition(
            goal, GOAL_WAITING, subtask, status, result
        )

    async def _subtask_job(self, goal_id: str, subtask_id: str) -> None:
        """Worker executed by the scheduler for a single subtask."""
        goal = self.state_machine.load(goal_id)
        subtask = next(
            (s for s in goal.subtasks if s.id == subtask_id),
            None,
        )
        if subtask is None:
            await self.event_bus.publish(
                Event(
                    type="subtask.completed",
                    payload={
                        "goal_id": goal_id,
                        "subtask_id": subtask_id,
                        "status": SUBTASK_FAILED,
                        "result": "subtask not found",
                    },
                )
            )
            return

        agent = self.agent_factory()
        loop = asyncio.get_running_loop()
        try:
            env = await loop.run_in_executor(
                None,
                agent.run,
                subtask.description,
                self.max_subtask_steps,
                False,
            )
            final = env.history[-1]["result"] if env.history else ""
            has_error = any("error" in h for h in env.history) or str(
                final
            ).startswith("Error:")
            status = SUBTASK_FAILED if has_error else SUBTASK_DONE
            result = str(final)
        except Exception as exc:  # noqa: BLE001
            status = SUBTASK_FAILED
            result = f"Error: {exc}"

        self.state_machine.transition(goal, None, subtask, status, result)
        await self.event_bus.publish(
            Event(
                type="subtask.completed",
                payload={
                    "goal_id": goal_id,
                    "subtask_id": subtask_id,
                    "status": status,
                    "result": result,
                },
            )
        )

    async def _start_scheduler(self) -> None:
        if not self.scheduler._running:
            await self.scheduler.start()

    def _plan(self, goal: Goal) -> list[Subtask]:
        """Ask the agent's brain for a list of subtasks."""
        agent = self.agent_factory()
        perception = {
            "task": goal.description,
            "context": goal.context,
            "step": 0,
            "max_steps": 3,
            "planning": True,
            "tool_descriptions": agent.tools.schemas(),
            "available_tools": agent.tools.names(),
        }
        action = agent.brain.think(perception, agent.tools.names())
        if action.get("name") == "plan":
            steps = action.get("args", {}).get("steps", [])
            if steps:
                return [Subtask(description=str(s)) for s in steps]
        return []

    def _legacy_save(self, goal: Goal) -> None:
        """Append-only JSONL fallback for the original state_path API."""
        if self._legacy_state_path is None:
            return
        self._legacy_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._legacy_state_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(goal), ensure_ascii=False) + "\n")

    def load_history(self) -> list[dict[str, Any]]:
        """Return all saved goals as plain dicts."""
        if self._legacy_state_path is not None:
            if not self._legacy_state_path.exists():
                return []
            entries = []
            with self._legacy_state_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
        return [asdict(g) for g in self.state_machine.list_goals()]
