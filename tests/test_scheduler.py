import asyncio
from pathlib import Path

import pytest

from swaybot.scheduler import (
    CronSchedule,
    IntervalSchedule,
    Scheduler,
)
from swaybot.storage import InMemoryBackend


@pytest.mark.asyncio
async def test_scheduler_runs_interval_job():
    counter = {"value": 0}

    async def increment():
        counter["value"] += 1

    scheduler = Scheduler()
    scheduler.add_job(increment, IntervalSchedule(seconds=0.05))
    await scheduler.start()
    await asyncio.sleep(0.18)
    await scheduler.stop()

    assert counter["value"] >= 2


@pytest.mark.asyncio
async def test_scheduler_respects_max_running():
    running = {"count": 0, "max": 0}

    async def slow():
        running["count"] += 1
        running["max"] = max(running["max"], running["count"])
        await asyncio.sleep(0.2)
        running["count"] -= 1

    scheduler = Scheduler()
    scheduler.add_job(
        slow,
        IntervalSchedule(seconds=0.05),
        max_running=1,
    )
    await scheduler.start()
    await asyncio.sleep(0.25)
    await scheduler.stop()

    assert running["max"] == 1


@pytest.mark.asyncio
async def test_scheduler_persists_state(tmp_path: Path):
    backend = InMemoryBackend()
    counter = {"value": 0}

    async def increment():
        counter["value"] += 1

    scheduler = Scheduler(backend=backend)
    job_id = scheduler.add_job(increment, IntervalSchedule(seconds=0.05))
    await scheduler.start()
    await asyncio.sleep(0.12)
    await scheduler.stop()

    state = backend.load("scheduler_state")
    assert state[job_id]["last_run"] is not None
    assert state[job_id]["next_run"] is not None


@pytest.mark.asyncio
async def test_scheduler_reloads_state(tmp_path: Path):
    backend = InMemoryBackend()
    backend.save(
        "scheduler_state",
        {
            "job-1": {
                "last_run": None,
                "next_run": "2000-01-01T00:00:00+00:00",
                "running_count": 0,
            }
        },
    )

    counter = {"value": 0}

    async def increment():
        counter["value"] += 1

    scheduler = Scheduler(backend=backend)
    scheduler.add_job(
        increment,
        IntervalSchedule(seconds=60),
        job_id="job-1",
    )
    await scheduler.start()
    await asyncio.sleep(0.1)
    await scheduler.stop()

    assert counter["value"] == 1


@pytest.mark.asyncio
async def test_scheduler_run_job_now():
    counter = {"value": 0}

    async def increment():
        counter["value"] += 1

    scheduler = Scheduler()
    job_id = scheduler.add_job(
        increment,
        IntervalSchedule(seconds=3600),
    )
    await scheduler.start()
    await scheduler.run_job_now(job_id)
    await asyncio.sleep(0.05)
    await scheduler.stop()

    assert counter["value"] == 1


import sys


def test_cron_schedule_requires_croniter(monkeypatch):
    has_croniter = False
    try:
        import croniter  # noqa: F401

        has_croniter = True
    except ImportError:
        pass

    scheduler = Scheduler()
    if not has_croniter:
        with pytest.raises(RuntimeError):
            scheduler.add_job(
                lambda: None,
                CronSchedule(expression="0 * * * *"),
            )
        return

    job_id = scheduler.add_job(
        lambda: None,
        CronSchedule(expression="0 * * * *"),
    )
    job = scheduler._jobs[job_id]

    # Block the optional croniter import to exercise the fallback path.
    monkeypatch.setitem(sys.modules, "croniter", object())

    with pytest.raises(RuntimeError):
        scheduler._compute_next_run(job)
