"""Sustained-goal machinery: long-task coordinator + job scheduler.

Experimental extension layer, isolated from the core package on purpose
(PRINCIPLES.md): its only live consumer is the HTTP API's /v1/goals
surface. The agent loop, CLI, explorer, and benchmarks run without it.
"""

from .coordinator import (
    GOAL_CANCELLED,
    GOAL_DONE,
    GOAL_FAILED,
    GOAL_PENDING,
    GOAL_PLANNING,
    GOAL_RUNNING,
    GOAL_WAITING,
    Goal,
    GoalCoordinator,
    GoalStateMachine,
    Subtask,
)
from .scheduler import (
    CronSchedule,
    IntervalSchedule,
    Job,
    OneShotSchedule,
    Scheduler,
)

__all__ = [
    "GOAL_CANCELLED",
    "GOAL_DONE",
    "GOAL_FAILED",
    "GOAL_PENDING",
    "GOAL_PLANNING",
    "GOAL_RUNNING",
    "GOAL_WAITING",
    "CronSchedule",
    "Goal",
    "GoalCoordinator",
    "GoalStateMachine",
    "IntervalSchedule",
    "Job",
    "OneShotSchedule",
    "Scheduler",
    "Subtask",
]
