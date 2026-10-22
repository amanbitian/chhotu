"""Open applications, folders and web pages. Applications come only from the configured allowlist, never from model text."""

from __future__ import annotations

import asyncio
import os
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import psutil

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok
from .files import _roots, inside_roots

# Opening these would run code, not show a document. open_path refuses them even inside the allowed folders.
EXECUTABLE_SUFFIXES = {".exe", ".bat", ".cmd", ".com", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".msi",
                       ".msc", ".scr", ".lnk", ".url", ".reg", ".hta", ".jar", ".py", ".pyw", ".dll", ".cpl"}

# Command -> process names (lowercase, no .exe) that prove the application is running afterwards.
PROCESS_NAMES = {
    "code": {"code"}, "firefox": {"firefox"}, "chrome": {"chrome"}, "notepad": {"notepad"},
    "calc": {"calculatorapp", "calc", "calculator"}, "explorer": {"explorer"}, "wt": {"windowsterminal"},
    "taskmgr": {"taskmgr"},
}


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
        if path.is_file() and path.suffix.lower() in EXECUTABLE_SUFFIXES:
            return fail(f"I will not open {path.suffix} files, because that would run a program. Open it in an editor instead.")
        os.startfile(str(path))
        return ok(path=str(path), note="Opened with the default application.")

    async def open_url(args: dict) -> dict:
        url = str(args.get("url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return fail("Only http and https addresses can be opened.")
        webbrowser.open(url)
        return ok(url=url, note="Sent to the default browser; I cannot confirm what it displays.")

    async def web_search(args: dict) -> dict:
        query = " ".join(str(args.get("query", "")).split())
        if not query:
            return fail("What should I search for?")
        url = settings.web_search_url.replace("{query}", quote_plus(query))
        webbrowser.open(url)
        return ok(query=query, url=url, note="Search opened in the default browser; I cannot read the results.")

    return [
        Tool("open_app", "Open an application from the allowlist (for example VS Code, Firefox, Notepad).",
             {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, Risk.ACT, open_app,
             describe=lambda a: f"open {a.get('name')}"),
        Tool("open_path", "Open a file or folder (inside the allowed project folders) with its default application.",
             {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, Risk.ACT, open_path,
             describe=lambda a: f"open {Path(str(a.get('path', ''))).name or 'a path'}"),
        Tool("open_url", "Open an http(s) web address in the default browser.",
             {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}, Risk.ACT, open_url,
             describe=lambda a: f"open the web page {urlparse(str(a.get('url', ''))).netloc or a.get('url')}"),
        Tool("web_search", "Search the web for something in the default browser.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, Risk.ACT, web_search,
             describe=lambda a: f"search the web for {a.get('query')}"),
    ]
