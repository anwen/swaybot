"""Lightweight background automation with interval-based scheduling."""

import threading
import time
from typing import Callable


class Automation:
    """Run jobs periodically in a background thread."""

    def __init__(self) -> None:
        self._jobs: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add_interval(
        self,
        seconds: float,
        fn: Callable,
        args: tuple = (),
    ) -> None:
        """Schedule ``fn(*args)`` every ``seconds``."""
        self._jobs.append(
            {
                "interval": seconds,
                "fn": fn,
                "args": args,
                "last": 0.0,
            }
        )

    def start(self) -> None:
        """Start the scheduler thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            next_due = float("inf")
            for job in self._jobs:
                due = job["last"] + job["interval"]
                if now >= due:
                    job["last"] = now
                    try:
                        job["fn"](*job["args"])
                    except Exception:  # pragma: no cover
                        pass
                    due = now + job["interval"]
                next_due = min(next_due, due)
            wait = max(0.01, next_due - time.time())
            self._stop.wait(wait)

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
