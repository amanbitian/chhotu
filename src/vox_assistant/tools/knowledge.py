"""Personal knowledge: full-text search over your own notes and documents (text files under `knowledge_roots`).

The text of each file is cached in memory and re-read only when its modification time changes, so after the first
search a query costs a directory walk plus string matching (measured: the uncached scan of F:/Projects took 7-17 s).
Results are snippets marked as untrusted, because a document can contain instructions that are not the user's.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from pathlib import Path

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok
from .files import SECRET_NAMES, SKIP_DIRS

# Notes and documents. Data files (json, csv, logs) are not notes and made the scan slow.
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org", ".adoc"}
MAX_FILE_BYTES = 1_000_000
SNIPPET = 160

_cache: dict[str, tuple[float, str]] = {}  # path -> (mtime, lowercased text)
_text: dict[str, str] = {}  # path -> original text, for snippets
_lock = threading.Lock()


def _read(p: Path, mtime: float) -> str | None:
    key = str(p)
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    _cache[key], _text[key] = (mtime, text.lower()), text
    return _cache[key][1]


def search_text(roots: list[Path], query: str, *, limit: int = 6, budget_s: float = 6.0) -> list[dict]:
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    if not words:
        return []
    scored: list[tuple[tuple[int, int], dict]] = []
    deadline = time.monotonic() + budget_s
    with _lock:
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for name in filenames:
                    if time.monotonic() > deadline:
                        break
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in TEXT_SUFFIXES or SECRET_NAMES.search(name):
                        continue
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size > MAX_FILE_BYTES:
                        continue
                    low = _read(p, st.st_mtime)
                    if low is None:
                        continue
                    present = [w for w in words if w in low]
                    if len(present) < max(1, (len(words) + 1) // 2):  # at least half of the words
                        continue
                    count = sum(low.count(w) for w in present)
                    first = min(low.find(w) for w in present)
                    start = max(0, first - SNIPPET // 2)
                    snippet = " ".join(_text[str(p)][start:start + SNIPPET].split())
                    scored.append(((len(present), count), {"path": str(p), "snippet": snippet}))
                if time.monotonic() > deadline:
                    break
    scored.sort(key=lambda s: s[0], reverse=True)
    return [hit for _, hit in scored[:limit]]


def make_tools(settings: Settings) -> list[Tool]:
    async def search_notes(args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return fail("What should I look for?")
        roots = [Path(r) for r in settings.knowledge_roots]
        hits = await asyncio.to_thread(search_text, roots, query)
        return ok(query=query, results=hits, untrusted_content=True,
                  note=None if hits else "Nothing in your notes matched.")

    return [
        Tool("search_notes", "Search the text of the user's own notes and documents (markdown, text) for a topic, "
             "and return matching snippets with file paths. Use for questions about their projects or notes.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             Risk.READ, search_notes),
    ]


def warm_index(settings: Settings) -> None:
    """Fill the cache in the background at start-up, so the first spoken search is fast too."""
    search_text([Path(r) for r in settings.knowledge_roots], "warm index", limit=1, budget_s=120.0)
