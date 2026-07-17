from pathlib import Path

import pytest

from swaybot.storage import (
    InMemoryBackend,
    JSONBackend,
    JSONLBackend,
    StorageError,
)


def test_json_backend_roundtrip(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    backend.save("memory", [{"a": 1}, {"b": 2}])
    assert backend.load("memory") == [{"a": 1}, {"b": 2}]


def test_json_backend_load_missing_returns_none(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    assert backend.load("missing") is None


def test_json_backend_append_to_list(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    backend.append("log", {"x": 1})
    backend.append("log", {"x": 2})
    assert backend.load_stream("log") == [{"x": 1}, {"x": 2}]


def test_json_backend_append_to_non_list_raises(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    backend.save("single", {"x": 1})
    with pytest.raises(StorageError):
        backend.append("single", {"x": 2})


def test_json_backend_query(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    backend.save("records", [
        {"tag": "a", "value": 1},
        {"tag": "b", "value": 2},
        {"tag": "a", "value": 3},
    ])
    assert backend.query("records", {"tag": "a"}) == [
        {"tag": "a", "value": 1},
        {"tag": "a", "value": 3},
    ]


def test_jsonl_backend_append_and_stream(tmp_path: Path):
    backend = JSONLBackend(tmp_path)
    backend.append("runs", {"step": 1})
    backend.append("runs", {"step": 2})
    assert backend.load_stream("runs") == [{"step": 1}, {"step": 2}]


def test_jsonl_backend_stream_limit(tmp_path: Path):
    backend = JSONLBackend(tmp_path)
    for i in range(5):
        backend.append("runs", {"step": i})
    assert backend.load_stream("runs", limit=2) == [{"step": 3}, {"step": 4}]


def test_jsonl_backend_skips_malformed_lines(tmp_path: Path):
    backend = JSONLBackend(tmp_path)
    path = tmp_path / "runs.jsonl"
    path.write_text('{"step":1}\nnot json\n{"step":2}\n', encoding="utf-8")
    assert backend.load_stream("runs") == [{"step": 1}, {"step": 2}]


def test_backend_rejects_invalid_key(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    with pytest.raises(StorageError):
        backend.save("../evil", [])
    with pytest.raises(StorageError):
        backend.save("", [])


def test_backend_list_keys(tmp_path: Path):
    backend = JSONBackend(tmp_path)
    backend.save("alpha", [])
    backend.save("beta", [])
    assert backend.list_keys() == ["alpha", "beta"]
    assert backend.list_keys(prefix="a") == ["alpha"]


def test_in_memory_backend_roundtrip():
    backend = InMemoryBackend()
    backend.save("x", [{"a": 1}])
    assert backend.load("x") == [{"a": 1}]
    backend.append("x", {"b": 2})
    assert backend.load_stream("x") == [{"a": 1}, {"b": 2}]


def test_in_memory_backend_query():
    backend = InMemoryBackend()
    backend.save("records", [{"tag": "a"}, {"tag": "b"}])
    assert backend.query("records", {"tag": "b"}) == [{"tag": "b"}]
