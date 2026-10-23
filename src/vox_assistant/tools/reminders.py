"""Local reminders. The store is SQLite, shared by the control panel and hands-free mode (separate processes).

Exactly one of them delivers each reminder: hands-free mode speaks them while it runs; otherwise the panel announces
them to the page and as a Windows notification. Taking a reminder is one transaction, so two processes polling at the
same moment never both get it.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok

LATE_AFTER_S = 120.0  # a reminder delivered this long after its time says when it was due (nothing was running)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Title and text travel in environment variables, never in the script text.
_TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName('text')
$t.Item(0).AppendChild($x.CreateTextNode($env:VOXA_TITLE)) > $null
$t.Item(1).AppendChild($x.CreateTextNode($env:VOXA_TEXT)) > $null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($x))
"""


def show_toast(title: str, text: str) -> bool:
    """A Windows notification; False if it could not be shown (not Windows, or PowerShell failed)."""
    if os.name != "nt":
        return False
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST], capture_output=True,
                           env={**os.environ, "VOXA_TITLE": title, "VOXA_TEXT": text}, timeout=20, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def is_late(reminder: dict, now: float | None = None) -> bool:
    return bool(reminder.get("due")) and (now or time.time()) - reminder["due"] > LATE_AFTER_S


class ReminderStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit, with an explicit transaction where one is needed (_take); wait out the other process's write.
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0, isolation_level=None)
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, due REAL, text TEXT,"
                             " state TEXT DEFAULT 'pending')")
            try:  # added in 0.2: when a reminder was claimed for speaking
                self._db.execute("ALTER TABLE reminders ADD COLUMN claimed_at REAL")
            except sqlite3.OperationalError:
                pass
            self._db.commit()

    def add(self, text: str, in_minutes: float) -> dict:
        due = self._clock() + in_minutes * 60
        with self._lock:
            cur = self._db.execute("INSERT INTO reminders (due, text) VALUES (?, ?)", (due, text))
            self._db.commit()
        return {"id": cur.lastrowid, "text": text, "due": due, "due_local": datetime.fromtimestamp(due).strftime("%H:%M")}

    def pending(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, due, text FROM reminders WHERE state='pending' ORDER BY due").fetchall()
        return [{"id": i, "due": d, "text": t, "due_local": datetime.fromtimestamp(d).strftime("%H:%M")} for i, d, t in rows]

    _DUE = ("SELECT id, due, text FROM reminders WHERE (state='pending' AND due<=?) "
            "OR (state='claimed' AND claimed_at<=?)")
    CLAIM_TIMEOUT_S = 120.0

    def due(self) -> list[dict]:
        """Return and mark as fired every reminder whose time has come (the web page: delivered once it is returned)."""
        return self._take("fired")

    def claim_due(self) -> list[dict]:
        """Return due reminders and mark them claimed; call `mark_fired` once one has been spoken. A claim older than
        two minutes (the process died before speaking) makes the reminder due again, so it is not lost."""
        return self._take("claimed")

    def mark_fired(self, reminder_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE reminders SET state='fired' WHERE id=?", (reminder_id,))
            self._db.commit()

    def _take(self, state: str) -> list[dict]:
        now = self._clock()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")  # the write lock first: the other process waits, then sees them taken
            try:
                rows = self._db.execute(self._DUE, (now, now - self.CLAIM_TIMEOUT_S)).fetchall()
                self._db.executemany("UPDATE reminders SET state=?, claimed_at=? WHERE id=?",
                                     [(state, now, r[0]) for r in rows])
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return [{"id": i, "due": d, "text": t, "due_local": datetime.fromtimestamp(d).strftime("%H:%M")} for i, d, t in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def cancel(self, reminder_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE reminders SET state='cancelled' WHERE id=? AND state='pending'", (reminder_id,))
            self._db.commit()
        return cur.rowcount > 0


def make_tools(settings: Settings, store: ReminderStore) -> list[Tool]:
    async def set_reminder(args: dict) -> dict:
        text = str(args.get("text", "")).strip()
        try:
            minutes = float(args.get("in_minutes"))
        except (TypeError, ValueError):
            return fail("in_minutes must be a number")
        if not text or not 0 < minutes <= 60 * 24 * 14:
            return fail("Give a reminder text and a delay between 1 minute and 14 days.")
        r = store.add(text, minutes)
        return ok(reminder=r, verified_stored=any(p["id"] == r["id"] for p in store.pending()))

    async def list_reminders(args: dict) -> dict:
        return ok(reminders=store.pending())

    async def cancel_reminder(args: dict) -> dict:
        try:
            rid = int(args.get("id"))
        except (TypeError, ValueError):
            return fail("id must be a number")
        return ok(cancelled=True) if store.cancel(rid) else fail("No pending reminder with that id.")

    return [
        Tool("set_reminder", "Remind the user after a delay. Convert times like 'in half an hour' to minutes.",
             {"type": "object", "properties": {"text": {"type": "string"}, "in_minutes": {"type": "number"}},
              "required": ["text", "in_minutes"]}, Risk.ACT, set_reminder,
             describe=lambda a: f"set a reminder in {a.get('in_minutes')} minutes: {a.get('text')}"),
        Tool("list_reminders", "List pending reminders.", {"type": "object", "properties": {}}, Risk.READ, list_reminders),
        Tool("cancel_reminder", "Cancel a pending reminder by id.",
             {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}, Risk.ACT, cancel_reminder,
             describe=lambda a: f"cancel reminder {a.get('id')}"),
    ]
