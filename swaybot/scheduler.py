"""Generic async job scheduler with interval/cron support and concurrency limits."""

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .storage import InMemoryBackend, StorageBackend


@dataclass
class IntervalSchedule:
    """Run a job every ``seconds``."""

    seconds: float


@dataclass
class CronSchedule:
    """Run a job on a cron expression (requires ``croniter``)."""

    expression: str


Schedule = IntervalSchedule | CronSchedule


@dataclass
class Job:
    """A scheduled job."""

    id: str
    name: str
    schedule: Schedule
    fn: Callable
    args: tuple = ()
    max_running: int = 1
    enabled: bool = True
    last_run: str | None = None
    next_run: str | None = None
    running_count: int = 0


class Scheduler:
    """Async scheduler with max-running constraints and context cancellation.

    Jobs are registered in code; only runtime state (last_run, next_run) is
    persisted if a backend is provided. This keeps callables out of storage.
    """

    def __init__(
        self,
        backend: StorageBackend | None = None,
        state_key: str = "scheduler_state",
    ) -> None:
        self.backend = backend or InMemoryBackend()
        self._state_key = state_key
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()

    def add_job(
        self,
        fn: Callable,
        schedule: Schedule,
        *,
        name: str | None = None,
        args: tuple = (),
        max_running: int = 1,
        enabled: bool = True,
        job_id: str | None = None,
    ) -> str:
        """Register a job and return its id."""
        job_id = job_id or uuid.uuid4().hex
        name = name or job_id
        job = Job(
            id=job_id,
            name=name,
            schedule=schedule,
            fn=fn,
            args=args,
            max_running=max_running,
            enabled=enabled,
        )
        self._compute_next_run(job)
        self._jobs[job_id] = job
        return job_id

    def remove_job(self, job_id: str) -> None:
        """Remove a job and cancel any running instances."""
        job = self._jobs.pop(job_id, None)
        if job is None:
            return
        for task in list(self._tasks):
            if task.get_name() == f"job-{job_id}":
                task.cancel()
        self._persist_state()

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return metadata for all registered jobs."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "enabled": job.enabled,
                "max_running": job.max_running,
                "running_count": job.running_count,
                "last_run": job.last_run,
                "next_run": job.next_run,
            }
            for job in self._jobs.values()
        ]

    async def start(self) -> None:
        """Start the scheduling loop."""
        if self._running:
            return
        self._load_state()
        self._running = True
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler and cancel running jobs."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        for task in list(self._tasks):
            task.cancel()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None

    async def run_job_now(self, job_id: str) -> None:
        """Trigger a job immediately, outside of its schedule."""
        job = self._jobs.get(job_id)
        if job is None or not job.enabled:
            return
        if job.running_count >= job.max_running:
            return
        self._schedule_run(job)

    async def _loop(self) -> None:
        while self._running:
            self._wake_event.clear()
            now = datetime.now(timezone.utc)
            for job in list(self._jobs.values()):
                if not job.enabled:
                    continue
                if job.running_count >= job.max_running:
                    continue
                next_run = self._parse_time(job.next_run)
                if next_run is not None and now >= next_run:
                    self._schedule_run(job)
                    self._compute_next_run(job)

            next_due = self._next_due_time()
            if next_due is None:
                wait = 1.0
            else:
                wait = (next_due - datetime.now(timezone.utc)).total_seconds()
                if wait <= 0:
                    continue
            if await self._wait_for_signal(wait):
                break

    async def _wait_for_signal(self, timeout: float) -> bool:
        """Wait for stop, wake, or ``timeout`` seconds. Return True if stop."""
        stop_task = asyncio.create_task(self._stop_event.wait())
        wake_task = asyncio.create_task(self._wake_event.wait())
        done, pending = await asyncio.wait(
            {stop_task, wake_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        return stop_task in done

    def _schedule_run(self, job: Job) -> None:
        job.running_count += 1
        task = asyncio.create_task(
            self._run_job(job), name=f"job-{job.id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_job(self, job: Job) -> None:
        try:
            if inspect.iscoroutinefunction(job.fn):
                await job.fn(*job.args)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, job.fn, *job.args)
        except Exception:  # pragma: no cover - job errors should not crash scheduler
            pass
        finally:
            job.running_count -= 1
            job.last_run = datetime.now(timezone.utc).isoformat()
            self._persist_state()
            self._wake_event.set()

    def _compute_next_run(self, job: Job) -> None:
        now = datetime.now(timezone.utc)
        if isinstance(job.schedule, IntervalSchedule):
            last = self._parse_time(job.last_run)
            base = max(last, now) if last is not None else now
            job.next_run = (
                base + timedelta(seconds=job.schedule.seconds)
            ).isoformat()
        elif isinstance(job.schedule, CronSchedule):
            try:
                from croniter import croniter
            except ImportError as exc:
                raise RuntimeError(
                    "croniter is required for cron schedules"
                ) from exc
            itr = croniter(job.schedule.expression, now)
            job.next_run = itr.get_next(datetime).isoformat()

    def _next_due_time(self) -> datetime | None:
        now = datetime.now(timezone.utc)
        candidates: list[datetime] = []
        for job in self._jobs.values():
            if not job.enabled or job.running_count >= job.max_running:
                continue
            next_run = self._parse_time(job.next_run)
            if next_run is not None and next_run > now:
                candidates.append(next_run)
        return min(candidates) if candidates else None

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _persist_state(self) -> None:
        state = {
            job.id: {
                "last_run": job.last_run,
                "next_run": job.next_run,
                "running_count": job.running_count,
            }
            for job in self._jobs.values()
        }
        self.backend.save(self._state_key, state)

    def _load_state(self) -> None:
        state = self.backend.load(self._state_key)
        if not isinstance(state, dict):
            return
        for job_id, job_state in state.items():
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if isinstance(job_state, dict):
                job.last_run = job_state.get("last_run")
                job.next_run = job_state.get("next_run")
                job.running_count = job_state.get("running_count", 0)
