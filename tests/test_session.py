import json
from pathlib import Path

import pytest

from swaybot.bus import InboundMessage, OutboundMessage
from swaybot.session import SessionError, SessionManager


def test_session_manager_create_and_load(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    manager.create("s1")
    assert manager.exists("s1")
    assert manager.load("s1") == []


def test_session_manager_appends_messages(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    manager.append("s1", {"role": "user", "content": "hi"})
    manager.append("s1", {"role": "assistant", "content": "hello"})

    history = manager.load("s1")
    assert len(history) == 2
    assert history[0]["content"] == "hi"
    assert history[1]["content"] == "hello"


def test_session_manager_persists_bus_messages(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    manager.append("s1", InboundMessage(content="in", session_id="s1"))
    manager.append("s1", OutboundMessage(content="out", session_id="s1"))

    history = manager.load("s1")
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "in"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "out"


def test_session_manager_load_limit(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    for i in range(5):
        manager.append("s1", {"role": "user", "content": str(i)})

    assert len(manager.load("s1", limit=2)) == 2
    assert manager.load("s1", limit=2)[0]["content"] == "3"


def test_session_manager_rejects_invalid_session_id(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    with pytest.raises(SessionError):
        manager.append("../evil", {"role": "user", "content": "x"})
    with pytest.raises(SessionError):
        manager.append("", {"role": "user", "content": "x"})


def test_session_manager_list_and_delete(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    manager.create("a")
    manager.create("b")
    assert manager.list_sessions() == ["a", "b"]
    manager.delete("a")
    assert manager.list_sessions() == ["b"]


def test_session_manager_clear_keeps_file(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    manager.append("s1", {"role": "user", "content": "x"})
    manager.clear("s1")
    assert manager.exists("s1")
    assert manager.load("s1") == []


def test_session_manager_load_missing_returns_empty(tmp_path: Path):
    manager = SessionManager(base_dir=tmp_path)
    assert manager.load("missing") == []
