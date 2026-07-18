"""Tests for per-path file locks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swaybot.file_locks import PathLockManager


def test_path_lock_synchronizes_concurrent_writes(tmp_path: Path) -> None:
    target = tmp_path / "counter.txt"
    target.write_text("0", encoding="utf-8")

    def increment() -> None:
        for _ in range(50):
            with PathLockManager.lock(target):
                value = int(target.read_text(encoding="utf-8"))
                target.write_text(str(value + 1), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(increment) for _ in range(4)]
        for future in futures:
            future.result()

    assert int(target.read_text(encoding="utf-8")) == 200


def test_path_lock_allows_concurrent_access_to_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("0", encoding="utf-8")
    b.write_text("0", encoding="utf-8")

    def write_once(path: Path, value: str) -> None:
        with PathLockManager.lock(path):
            path.write_text(value, encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_once, a, "a"),
            executor.submit(write_once, b, "b"),
        ]
        for future in futures:
            future.result()

    assert a.read_text(encoding="utf-8") == "a"
    assert b.read_text(encoding="utf-8") == "b"


def test_path_lock_keyed_by_resolved_path(tmp_path: Path) -> None:
    """Equivalent paths (with . or ..) share the same lock."""
    target = tmp_path / "dir" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_text("0", encoding="utf-8")

    alias = tmp_path / "dir" / "." / "file.txt"

    def increment() -> None:
        for _ in range(25):
            with PathLockManager.lock(alias):
                value = int(target.read_text(encoding="utf-8"))
                target.write_text(str(value + 1), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(increment),
            executor.submit(increment),
        ]
        for future in futures:
            future.result()

    assert int(target.read_text(encoding="utf-8")) == 50
