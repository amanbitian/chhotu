"""Append-only SQLite log of every tool request, the permission decision, and the verified result."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, tool TEXT,"
                " risk TEXT, args TEXT, decision TEXT, ok INTEGER, summary TEXT)")
            self._db.commit()

    def record(self, kind: str, tool: str = "", risk: str = "", args: dict | None = None, decision: str = "",
               ok: bool | None = None, summary: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, kind, tool, risk, args, decision, ok, summary) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), kind, tool, risk, json.dumps(args or {}, ensure_ascii=False, default=str), decision,
                 None if ok is None else int(ok), summary[:500]))
            self._db.commit()

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, kind, tool, risk, args, decision, ok, summary FROM events ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        keys = ("id", "ts", "kind", "tool", "risk", "args", "decision", "ok", "summary")
        return [dict(zip(keys, r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
