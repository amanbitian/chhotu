"""Desktop windows: list them, bring one forward / minimize / maximize, or close one. Win32 via ctypes.

Closing is `destructive`: an application that does not ask about unsaved work would lose it.
Every action is checked against the window's real state afterwards.
"""

from __future__ import annotations

import asyncio
import ctypes
import re
import sys
from ctypes import wintypes
from pathlib import Path

import psutil

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok

SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE = 6, 3, 9
WM_CLOSE = 0x0010
DWMWA_CLOAKED = 14
GWL_EXSTYLE, WS_EX_TOOLWINDOW = -20, 0x00000080
# Windows that are part of the shell, not something the user would call "a window".
SHELL_PROCESSES = {"textinputhost", "shellexperiencehost", "searchhost", "startmenuexperiencehost", "lockapp"}


def list_windows() -> list[dict]:
    if sys.platform != "win32":
        return []
    user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
    out: list[dict] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):  # 4 = GW_OWNER: skip dialogs owned by others
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        cloaked = wintypes.DWORD()
        dwm.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if cloaked.value:
            return True  # suspended store apps and windows on other virtual desktops
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            proc = psutil.Process(pid.value).name().lower().removesuffix(".exe")
        except psutil.Error:
            proc = ""
        if proc in SHELL_PROCESSES or buf.value in ("Program Manager",):
            return True
        out.append({"id": int(hwnd), "title": buf.value, "app": proc, "minimized": bool(user32.IsIconic(hwnd)),
                    "maximized": bool(user32.IsZoomed(hwnd))})
        return True

    user32.EnumWindows(visit, 0)
    fg = user32.GetForegroundWindow()
    for w in out:
        w["focused"] = w["id"] == fg
    return out


def match_window(query: str, windows: list[dict], apps: dict[str, str] | None = None) -> dict | list[dict] | None:
    """One window, a list of candidates when ambiguous, or None. Matches app name first, then title words.
    `apps` (the allowlist, e.g. "vs code" -> "code") turns spoken app names into process names."""
    q = " ".join(re.sub(r"^the\s+|\s+window$", "", query.lower().strip()).split())
    if not q:
        return None
    if apps and q in apps:
        q = Path(apps[q]).stem.lower()
    by_app = [w for w in windows if w["app"] == q or w["app"] == q.replace(" ", "")]
    if len(by_app) == 1:
        return by_app[0]
    words = q.split()
    by_title = [w for w in (by_app or windows) if all(word in (w["title"] + " " + w["app"]).lower() for word in words)]
    if len(by_title) == 1:
        return by_title[0]
    return by_title or by_app or None


def _focus(hwnd: int) -> bool:
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    # Windows only lets the foreground process hand focus over; a synthetic Alt tap satisfies that rule.
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)
    user32.SetForegroundWindow(hwnd)
    return user32.GetForegroundWindow() == hwnd


def make_tools(settings: Settings) -> list[Tool]:
    def find(args: dict) -> tuple[dict | None, dict | None]:
        target = match_window(str(args.get("window", "")), list_windows(), settings.apps)
        if target is None:
            return None, fail("No open window matches that. Ask me to list the windows.")
        if isinstance(target, list):
            return None, fail("Several windows match; which one?", candidates=[{"title": w["title"], "app": w["app"]} for w in target[:8]])
        return target, None

    async def windows(args: dict) -> dict:
        found = await asyncio.to_thread(list_windows)
        return ok(windows=[{k: w[k] for k in ("title", "app", "focused", "minimized")} for w in found[:30]],
                  untrusted_content=True)  # titles come from other programs and web pages

    async def window_action(args: dict) -> dict:
        action = str(args.get("action", "")).lower()
        if action not in ("focus", "minimize", "maximize", "restore"):
            return fail("action must be focus, minimize, maximize or restore")
        target, err = find(args)
        if err:
            return err
        hwnd, user32 = target["id"], ctypes.windll.user32
        if action == "focus":
            done = _focus(hwnd)
        else:
            user32.ShowWindow(hwnd, {"minimize": SW_MINIMIZE, "maximize": SW_MAXIMIZE, "restore": SW_RESTORE}[action])
            await asyncio.sleep(0.2)
            done = {"minimize": bool(user32.IsIconic(hwnd)), "maximize": bool(user32.IsZoomed(hwnd)),
                    "restore": not user32.IsIconic(hwnd) and not user32.IsZoomed(hwnd)}[action]
        if not done:
            return fail(f"Windows did not let me {action} it.", window=target["title"])
        return ok(window=target["title"], app=target["app"], action=action, verified=True)

    async def close_window(args: dict) -> dict:
        target, err = find(args)
        if err:
            return err
        hwnd, user32 = target["id"], ctypes.windll.user32
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        for _ in range(12):
            await asyncio.sleep(0.25)
            if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
                return ok(window=target["title"], closed=True)
        return ok(window=target["title"], closed=False,
                  note="It is still open; it may be asking whether to save. Nothing was forced closed.")

    window_arg = {"type": "object", "properties": {"window": {"type": "string", "description": "app name or words from the title"}},
                  "required": ["window"]}
    return [
        Tool("list_windows", "List the open windows (title, app, which one is focused).",
             {"type": "object", "properties": {}}, Risk.READ, windows),
        Tool("window_action", "Bring a window to the front (focus), or minimize, maximize or restore it.",
             {"type": "object", "properties": {**window_arg["properties"],
                                               "action": {"type": "string", "enum": ["focus", "minimize", "maximize", "restore"]}},
              "required": ["window", "action"]}, Risk.ACT, window_action,
             describe=lambda a: f"{a.get('action', 'focus')} the {a.get('window')} window"),
        Tool("close_window", "Close a window, as if its close button was clicked.", window_arg, Risk.DESTRUCTIVE,
             close_window, describe=lambda a: f"close the {a.get('window')} window, which can lose unsaved work"),
    ]
