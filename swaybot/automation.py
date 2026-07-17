"""Lightweight background automation backed by the async Scheduler."""

import asyncio
import threading
from typing import Callable

from .scheduler import IntervalSchedule, Scheduler


class Automation:
    """Run jobs periodically in a background thread.

    This is a compatibility wrapper around ``Scheduler`` that preserves the
    original synchronous ``add_interval`` / ``start`` / ``stop`` API.
    """

    def __init__(self) -> None:
        self._scheduler = Scheduler()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def add_interval(
        self,
        seconds: float,
        fn: Callable,
        args: tuple = (),
    ) -> str:
        """Schedule ``fn(*args)`` every ``seconds``."""
        return self._scheduler.add_job(
            fn,
            IntervalSchedule(seconds=seconds),
            args=args,
        )

    def start(self) -> None:
        """Start the scheduler in a daemon thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._scheduler.start())
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread."""
        if self._loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._scheduler.stop(), self._loop
            )
            try:
                future.result(timeout=2)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
