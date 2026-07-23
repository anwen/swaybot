"""In-process per-path locks for filesystem tools.

Prevents concurrent writes/edits to the same file within the same process.
Locks are keyed by canonical resolved path and are released when the context
manager exits.
"""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class PathLockManager:
    """Manage per-path locks keyed by canonical resolved path."""

    _master_lock = threading.Lock()
    _locks: dict[str, threading.Lock] = {}

    @classmethod
    @contextmanager
    def lock(cls, path: str | Path) -> Generator[None, None, None]:
        """Acquire a lock for ``path`` for the duration of the with-block."""
        key = str(Path(path).resolve())
        with cls._master_lock:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._locks[key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
