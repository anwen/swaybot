import json
from io import BytesIO
from unittest.mock import patch

from swaybot.sdk import SwayBotClient


class FakeOpener:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        return BytesIO(json.dumps(self.response_data).encode("utf-8"))


def test_sdk_health():
    client = SwayBotClient("http://localhost:9999")
    opener = FakeOpener({"status": "ok"})
    with patch("swaybot.sdk.urllib.request.urlopen", opener):
        assert client.health()["status"] == "ok"


def test_sdk_chat_completions():
    client = SwayBotClient("http://localhost:9999")
    opener = FakeOpener({"choices": [{"message": {"content": "hi"}}]})
    with patch("swaybot.sdk.urllib.request.urlopen", opener):
        result = client.chat_completions([{"role": "user", "content": "hello"}])
        assert result["choices"][0]["message"]["content"] == "hi"
        request = opener.calls[0]
        assert request.full_url == "http://localhost:9999/v1/chat/completions"
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["messages"][0]["content"] == "hello"


def test_sdk_post_message():
    client = SwayBotClient("http://localhost:9999")
    opener = FakeOpener({"status": "accepted"})
    with patch("swaybot.sdk.urllib.request.urlopen", opener):
        result = client.post_message("my-session", "hello")
        assert result["status"] == "accepted"
        request = opener.calls[0]
        assert "my-session" in request.full_url


def test_sdk_get_history():
    client = SwayBotClient("http://localhost:9999")
    opener = FakeOpener({"messages": [{"role": "assistant", "content": "ok"}]})
    with patch("swaybot.sdk.urllib.request.urlopen", opener):
        result = client.get_history("my-session", limit=5)
        assert result["messages"][0]["content"] == "ok"
        request = opener.calls[0]
        assert "limit=5" in request.full_url
