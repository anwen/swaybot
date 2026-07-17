import json
from pathlib import Path

import pytest

from swaybot.agent import Agent
from swaybot.coordinator import GoalCoordinator, Goal


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
