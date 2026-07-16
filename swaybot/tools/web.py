"""Basic web fetch and search tools with SSRF protection."""

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            self.texts.append(data)

    def text(self) -> str:
        return " ".join(self.texts)


def _is_private_host(hostname: str) -> bool:
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for _family, _type, _proto, _canon, sockaddr in addrinfo:
        ip = sockaddr[0]
        try:
            if ipaddress.ip_address(ip).is_private:
                return True
        except ValueError:
            continue
    return False


def _safe_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("URL has no host")
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Refusing to fetch local URL")
    if _is_private_host(hostname):
        raise ValueError("Refusing to fetch private URL")
    return url


def _fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; SwayBot/0.1; +https://example.com)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        data = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        return data.decode(charset, errors="replace")


def web_fetch(url: str, max_length: int = 4000) -> str:
    """Fetch a public URL and return visible text."""
    safe = _safe_url(url)
    try:
        html = _fetch(safe)
    except urllib.error.URLError as exc:
        return f"Error fetching URL: {exc}"
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:  # pragma: no cover
        pass
    text = " ".join(extractor.text().split())
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo lite and return result snippets."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    try:
        html = _fetch(url)
    except urllib.error.URLError as exc:
        return f"Error searching: {exc}"

    results: list[str] = []
    # DuckDuckGo lite result rows contain a link and a snippet.
    for link_match, snippet_match in zip(
        re.finditer(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html),
        re.finditer(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.S),
    ):
        href, title = link_match.groups()
        snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1))
        snippet = " ".join(snippet.split())
        results.append(f"{title.strip()} - {href.strip()}\n{snippet}")
        if len(results) >= max_results:
            break

    return "\n\n".join(results) if results else "No results found."
