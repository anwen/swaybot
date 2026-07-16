import pytest

from swaybot.tools import web as web_tools


SAMPLE_HTML = """
<!doctype html>
<html>
<head><title>Test</title></head>
<body>
<script>alert('ignore');</script>
<p>Hello world</p>
<footer>skip this</footer>
</body>
</html>
"""


def test_safe_url_accepts_public_https():
    assert web_tools._safe_url("https://example.com/path") == "https://example.com/path"


def test_safe_url_rejects_localhost():
    with pytest.raises(ValueError, match="local"):
        web_tools._safe_url("http://localhost:8080/")


def test_safe_url_rejects_private_ip():
    with pytest.raises(ValueError, match="private"):
        web_tools._safe_url("http://192.168.1.1/")


def test_safe_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme"):
        web_tools._safe_url("file:///etc/passwd")


def test_web_fetch_extracts_visible_text(monkeypatch):
    monkeypatch.setattr(web_tools, "_fetch", lambda url: SAMPLE_HTML)
    result = web_tools.web_fetch("https://example.com")
    assert "Hello world" in result
    assert "alert" not in result
    assert "skip this" not in result


def test_web_fetch_trims_to_max_length(monkeypatch):
    monkeypatch.setattr(web_tools, "_fetch", lambda url: "<p>" + "x" * 10000 + "</p>")
    result = web_tools.web_fetch("https://example.com", max_length=100)
    assert len(result) <= 104
    assert result.endswith("...")


def test_web_fetch_returns_error_on_fetch_failure(monkeypatch):
    import urllib.error

    def fail(url):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(web_tools, "_fetch", fail)
    result = web_tools.web_fetch("https://example.com")
    assert result.startswith("Error fetching URL")


def test_web_search_parses_duckduckgo_lite(monkeypatch):
    html = """
    <a href="https://example.com/one">First</a>
    <td class="result-snippet">Snippet one</td>
    <a href="https://example.com/two">Second</a>
    <td class="result-snippet">Snippet two</td>
    """
    monkeypatch.setattr(web_tools, "_fetch", lambda url: html)
    result = web_tools.web_search("query", max_results=2)
    assert "First" in result
    assert "Second" in result
    assert "Snippet one" in result
    assert "Snippet two" in result


def test_web_search_returns_no_results_when_empty(monkeypatch):
    monkeypatch.setattr(web_tools, "_fetch", lambda url: "<html></html>")
    result = web_tools.web_search("query")
    assert result == "No results found."
