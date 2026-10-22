"""A system-wide hotkey (Win32 RegisterHotKey) that wakes the assistant: push-to-talk without a wake phrase."""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Callable

MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
NAMED_KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "pause": 0x13, "scrolllock": 0x91,
              **{f"f{i}": 0x6F + i for i in range(1, 25)}}


def parse(spec: str) -> tuple[int, int]:
    """'ctrl+alt+space' -> (modifier flags, virtual-key code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    *mods, key = parts
    flags = 0
    for m in mods:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r} in {spec!r}")
        flags |= MODIFIERS[m]
    if key in NAMED_KEYS:
        vk = NAMED_KEYS[key]
    elif len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    else:
        raise ValueError(f"unknown key {key!r} in {spec!r}")
    if not flags and not (key.startswith("f") or key in ("pause", "scrolllock")):
        raise ValueError("a hotkey needs a modifier (ctrl, alt, shift, win) unless it is a function, Pause or Scroll Lock key")
    return flags, vk


class GlobalHotkey:
    def __init__(self, spec: str, on_press: Callable[[], None]) -> None:
        self.spec, self.on_press = spec, on_press
        self.mods, self.vk = parse(spec)
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self.error: str | None = None

    def start(self) -> bool:
        """Register and listen on a background thread. False (with `error`) if another app already owns the key."""
        if sys.platform != "win32":
            self.error = "hotkeys are only implemented on Windows"
            return False
        self._thread = threading.Thread(target=self._run, name="vox-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self.error is None

    def _run(self) -> None:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, self.mods | MOD_NOREPEAT, self.vk):
            self.error = f"{self.spec} is already used by another program"
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    try:
                        self.on_press()
                    except Exception:
                        pass
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2.0)
