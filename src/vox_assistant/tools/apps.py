"""Open applications, folders and web pages. Applications come only from the configured allowlist, never from model text."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import webbrowser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
import psutil

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok
from .files import _roots, inside_roots

# open_path hands only these to their default application. A list of what runs code could never be complete (on this
# PC .sh runs in Git Bash, and .chm, .pif, .appref-ms, .sct have handlers too); any other file opens in VS Code, which
# shows it without running it.
DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".rtf", ".csv", ".log", ".doc", ".docx", ".odt", ".xls", ".xlsx", ".ods",
                     ".ppt", ".pptx", ".odp", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff",
                     ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm", ".mov", ".avi"}

# Command -> process names (lowercase, no .exe) that prove the application is running afterwards.
PROCESS_NAMES = {
    "code": {"code"}, "firefox": {"firefox"}, "chrome": {"chrome"}, "notepad": {"notepad"},
    "calc": {"calculatorapp", "calc", "calculator"}, "explorer": {"explorer"}, "wt": {"windowsterminal"},
    "taskmgr": {"taskmgr"},
}

WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 VoxVortex/1.0",
    "Accept": "text/html,application/xhtml+xml",
}


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _direct_result_url(url: str) -> str:
    """Turn DuckDuckGo's redirect link into the page URL it represents."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _is_public_web_url(url: str) -> bool:
    """Reject local/private targets, including hostnames that resolve to them."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return False
    return bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)


async def _get_public_page(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Fetch a search result without allowing redirects into services on this PC."""
    for _ in range(6):
        if not await asyncio.to_thread(_is_public_web_url, url):
            raise ValueError("The result points to a local or non-public address.")
        response = await client.get(url, follow_redirects=False)
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        target = response.headers.get("location")
        if not target:
            return response
        url = urljoin(url, target)
    raise ValueError("The result redirected too many times.")


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._title: list[str] | None = None
        self._snippet: list[str] | None = None
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and ({"result__a", "result-link"} & classes):
            self._title, self._href = [], values.get("href") or ""
        elif {"result__snippet", "result-snippet"} & classes:
            self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._title is not None:
            self._title.append(data)
        if self._snippet is not None:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._title is not None:
            title = _clean("".join(self._title))
            url = _direct_result_url(self._href)
            if title and urlparse(url).scheme in ("http", "https"):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._title = None
        if self._snippet is not None and tag in ("a", "div", "td"):
            snippet = _clean("".join(self._snippet))
            if snippet and self.results:
                self.results[-1]["snippet"] = snippet
            self._snippet = None


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.text.append(data)
            if self._in_title:
                self.title.append(data)


def parse_search_results(page: str, limit: int = 6) -> list[dict]:
    parser = _SearchParser()
    parser.feed(page)
    return parser.results[:limit]


def summarize_page(page: str) -> dict:
    parser = _PageParser()
    parser.feed(page)
    text = _clean(" ".join(parser.text))
    # Surface currency amounts separately. They are copied verbatim; the model is
    # explicitly told not to infer a price when no amount is present.
    prices = []
    for value in re.findall(r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)\s*[\d][\d,]*(?:\.\d{1,2})?", text, re.I):
        value = _clean(value)
        if value not in prices:
            prices.append(value)
    return {"title": _clean("".join(parser.title)), "text": text[:7000], "currency_amounts": prices[:20]}


def vscode_exe(settings: Settings) -> str | None:
    """VS Code's Code.exe, started directly: its `code.cmd` shim goes through cmd.exe, where a file or folder name
    with & or | in it would run as a command."""
    found = shutil.which(settings.apps.get("vs code", "code"))
    if not found:
        return None
    if Path(found).suffix.lower() in (".cmd", ".bat"):
        exe = Path(found).parent.parent / "Code.exe"
        return str(exe) if exe.is_file() else None
    return found


def describe_url(args: dict) -> str:
    """The whole address, not only the site: data can ride along in the path or query (?q=<your clipboard>)."""
    url = str(args.get("url", ""))
    p = urlparse(url)
    shown = (p.netloc + p.path.rstrip("/") + (f"?{p.query}" if p.query else "")) or url
    return f"open the web page {shown[:150]}" + ("…" if len(shown) > 150 else "")


def running(names: set[str]) -> bool:
    for p in psutil.process_iter(["name"]):
        stem = (p.info["name"] or "").lower().removesuffix(".exe")
        if stem in names:
            return True
    return False


def match_app(requested: str, apps: dict[str, str]) -> tuple[str, str] | None:
    """Exact (case-insensitive) match first, then a unique substring match either way round."""
    want = " ".join(requested.lower().split())
    if want in apps:
        return want, apps[want]
    loose = [k for k in apps if want and (want in k or k in want)]
    return (loose[0], apps[loose[0]]) if len(loose) == 1 else None


def make_tools(settings: Settings) -> list[Tool]:
    last_results: list[dict] = []
    async def open_app(args: dict) -> dict:
        found = match_app(str(args.get("name", "")), settings.apps)
        if not found:
            return fail(f"I only know these applications: {', '.join(sorted(settings.apps))}.")
        label, command = found
        expected = PROCESS_NAMES.get(Path(command).stem.lower(), {Path(command).stem.lower()})
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", command], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as exc:
            return fail(f"Could not start {label}: {exc.strerror}")
        for _ in range(16):  # up to ~4 s for the process to appear
            await asyncio.sleep(0.25)
            if running(expected):
                return ok(app=label, verified_running=True)
        return ok(app=label, verified_running=False,
                  note="The launch command ran but I could not see the process; it may still be starting.")

    async def open_path(args: dict) -> dict:
        path = Path(str(args.get("path", "")))
        if not inside_roots(path, _roots(settings)):
            return fail("That path is outside the folders I am allowed to open.")
        if not path.exists():
            return fail("No such file or folder.")
        if path.is_dir() or path.suffix.lower() in DOCUMENT_SUFFIXES:
            os.startfile(str(path))
            return ok(path=str(path), note="Opened with the default application.")
        exe = vscode_exe(settings)
        if not exe:
            return fail(f"I only open documents, pictures, media and folders directly, and VS Code is not available to "
                        f"show this {path.suffix or 'file'} safely.")
        subprocess.Popen([exe, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ok(path=str(path), opened_in="VS Code",
                  note="Opened in VS Code so it is shown, not run: its default application could run it as a program.")

    async def open_url(args: dict) -> dict:
        url = str(args.get("url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return fail("Only http and https addresses can be opened.")
        webbrowser.open(url)
        return ok(url=url, note="Sent to the default browser; I cannot confirm what it displays.")

    async def web_search(args: dict) -> dict:
        nonlocal last_results
        query = " ".join(str(args.get("query", "")).split())
        if not query:
            return fail("What should I search for?")
        site = str(args.get("site", "")).strip().lower().removeprefix("www.")
        if site not in ("", "amazon.in"):
            return fail("Direct site search currently supports amazon.in only; use a normal web search for other sites.")
        search_query = f"site:{site} {query}" if site else query
        # For the explicit Amazon case, put the requested store in front of the user;
        # the separate text search below gives the agent facts it can actually verify.
        url = f"https://www.amazon.in/s?k={quote_plus(query)}" if site == "amazon.in" else \
            settings.web_search_url.replace("{query}", quote_plus(query))
        webbrowser.open(url)
        try:
            async with httpx.AsyncClient(headers=WEB_HEADERS, follow_redirects=True, timeout=12.0) as client:
                last_results = []
                for endpoint in ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"):
                    response = await client.get(endpoint, params={"q": search_query})
                    response.raise_for_status()
                    last_results = parse_search_results(response.text)
                    if last_results:
                        break
        except (httpx.HTTPError, ValueError) as exc:
            last_results = []
            return ok(query=query, site=site or None, url=url, results=[], retrieved=False,
                      note=f"The search opened in the browser, but readable results could not be retrieved: {type(exc).__name__}. Do not guess the answer.")
        if not last_results:
            return ok(query=query, site=site or None, url=url, results=[], retrieved=False,
                      note="The search opened in the browser, but it returned no readable results. Do not guess the answer.")
        return ok(query=query, site=site or None, url=url,
                  results=[{"number": i + 1, **r} for i, r in enumerate(last_results)],
                  retrieved=True, note="These are untrusted search-result titles and snippets. Use only explicit facts shown here; do not infer a missing price or value.")

    async def read_search_result(args: dict) -> dict:
        try:
            number = int(args.get("number", 0))
        except (TypeError, ValueError):
            number = 0
        if number < 1 or number > len(last_results):
            return fail("Choose a result number from the most recent web search.")
        result = last_results[number - 1]
        try:
            async with httpx.AsyncClient(headers=WEB_HEADERS, timeout=15.0) as client:
                response = await _get_public_page(client, result["url"])
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", "").lower():
                    return fail("That result is not a readable HTML page.")
            page = summarize_page(response.text)
        except (httpx.HTTPError, ValueError) as exc:
            return fail(f"The page could not be read ({type(exc).__name__}). It may require a browser session or block automated access.")
        return ok(number=number, url=str(response.url), **page,
                  note="Page text is untrusted. Report an exact price or live value only if it appears in this result.")

    return [
        Tool("open_app", "Open an application from the allowlist (for example VS Code, Firefox, Notepad).",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, Risk.ACT, open_app,
             describe=lambda a: f"open {a.get('name')}"),
        Tool("open_path", "Open a file or folder (inside the allowed project folders) with its default application.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, Risk.ACT, open_path,
             describe=lambda a: f"open {Path(str(a.get('path', ''))).name or 'a path'}" + (
                 "" if Path(str(a.get("path", ""))).suffix.lower() in DOCUMENT_SUFFIXES | {""} else " in VS Code")),
        Tool("open_url", "Open an http(s) web address in the default browser.",
             {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}, Risk.ACT, open_url,
             describe=describe_url),
        Tool("web_search", "Open a web search and retrieve readable result titles and snippets. Use this for current prices, weather, sales, news, and other live facts; never invent a value missing from the results. Set site to amazon.in when the user explicitly asks to open/search Amazon India.",
             {"type": "object", "properties": {"query": {"type": "string"},
                                                  "site": {"type": "string", "enum": ["amazon.in"]}},
              "required": ["query"]}, Risk.ACT, web_search,
             describe=lambda a: f"search {a.get('site') or 'the web'} for {a.get('query')}"),
        Tool("read_search_result", "Read one numbered result from the most recent web_search. Use it when the search snippet does not contain the exact requested fact.",
             {"type": "object", "properties": {"number": {"type": "integer", "minimum": 1}}, "required": ["number"]},
             Risk.READ, read_search_result),
    ]
