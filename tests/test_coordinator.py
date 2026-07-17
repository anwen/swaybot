import json
from pathlib import Path

import pytest

from swaybot.agent import Agent
from swaybot.brain import EchoBrain
from swaybot.coordinator import GoalCoordinator, GoalStateMachine, Goal, Subtask
from swaybot.storage import InMemoryBackend


def test_coordinator_runs_subtasks(tmp_path: Path):
    state = tmp_path / "goals.jsonl"
    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(),
        state_path=state,
        max_subtask_steps=2,
    )
    goal = coordinator.run("say hello")
    assert goal.status == "done"
    assert len(goal.subtasks) >= 1
    assert any(s.status == "done" for s in goal.subtasks)


def test_coordinator_saves_state(tmp_path: Path):
    state = tmp_path / "goals.jsonl"
    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(),
        state_path=state,
        max_subtask_steps=2,
    )
    coordinator.run("test goal")
    history = coordinator.load_history()
    assert len(history) == 1
    assert history[0]["description"] == "test goal"


def test_coordinator_plans_from_brain():
    class PlanBrain:
        def think(self, perception, available_tools, metadata=None):
            return {
                "name": "plan",
                "args": {"steps": ["step one", "step two"]},
            }

    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(brain=PlanBrain()),
        state_path=Path("/dev/null"),
        max_subtask_steps=1,
    )
    goal = coordinator.run("do something")
    assert len(goal.subtasks) == 2
    assert goal.subtasks[0].description == "step one"


def test_coordinator_records_failed_subtask(tmp_path: Path):
    class BrokenBrain:
        def think(self, perception, available_tools, metadata=None):
            return {"name": "nonexistent_tool", "args": {}}

    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(brain=BrokenBrain()),
        state_path=tmp_path / "goals.jsonl",
        max_subtask_steps=2,
    )
    goal = coordinator.run("fail")
    assert goal.status == "failed"
    assert any(s.status == "failed" for s in goal.subtasks)


def test_state_machine_round_trip():
    backend = InMemoryBackend()
    sm = GoalStateMachine(backend)
    goal = Goal(description="round trip")
    goal.subtasks = [Subtask(description="step one")]
    sm.create(goal)

    loaded = sm.load(goal.id)
    assert loaded.description == "round trip"
    assert len(loaded.subtasks) == 1
    assert loaded.subtasks[0].description == "step one"


@pytest.mark.asyncio
async def test_resume_runs_pending_subtasks():
    backend = InMemoryBackend()
    sm = GoalStateMachine(backend)
    goal = Goal(description="resume me")
    goal.subtasks = [Subtask(id="st1", description="do work")]
    sm.create(goal)

    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(brain=EchoBrain()),
        backend=backend,
        max_subtask_steps=2,
    )
    resumed = await coordinator.resume(goal.id)
    assert resumed.state == "done"
    assert resumed.subtasks[0].status == "done"


@pytest.mark.asyncio
async def test_retry_resets_failed_subtasks():
    backend = InMemoryBackend()
    sm = GoalStateMachine(backend)
    goal = Goal(description="retry me")
    goal.subtasks = [Subtask(id="st1", description="do work", status="failed", result="oops")]
    sm.create(goal)

    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(brain=EchoBrain()),
        backend=backend,
        max_subtask_steps=2,
    )
    retried = await coordinator.retry(goal.id)
    assert retried.state == "done"
    assert retried.subtasks[0].status == "done"
    assert retried.subtasks[0].result != "oops"


@pytest.mark.asyncio
async def test_cancel_marks_goal_cancelled():
    backend = InMemoryBackend()
    sm = GoalStateMachine(backend)
    goal = Goal(description="cancel me")
    sm.create(goal)

    coordinator = GoalCoordinator(
        agent_factory=lambda: Agent(brain=EchoBrain()),
        backend=backend,
        max_subtask_steps=2,
    )
    cancelled = await coordinator.cancel(goal.id)
    assert cancelled.state == "cancelled"
