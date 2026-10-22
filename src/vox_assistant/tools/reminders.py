"""Local reminders. The store is SQLite; whichever runtime is active (voice loop or web app) polls `due()` and speaks them."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok


class ReminderStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, due REAL, text TEXT,"
                             " state TEXT DEFAULT 'pending')")
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

    def due(self) -> list[dict]:
        """Return and mark as fired every reminder whose time has come."""
        now = self._clock()
        with self._lock:
            rows = self._db.execute("SELECT id, due, text FROM reminders WHERE state='pending' AND due<=?", (now,)).fetchall()
            self._db.executemany("UPDATE reminders SET state='fired' WHERE id=?", [(r[0],) for r in rows])
            self._db.commit()
        return [{"id": i, "due": d, "text": t} for i, d, t in rows]

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
