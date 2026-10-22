"""Live check of the desktop tools on this PC, calling each tool's handler directly (no LLM, no permission prompt).

    .venv\\Scripts\\python.exe scripts\\desktop_check.py [--with-agent]

Visible side effects, all cleaned up: opens Notepad and closes it again (unsaved, empty), sets the clipboard and then
restores what was there, opens ONE browser tab with a web search, opens this project in VS Code, and takes one
screenshot that is described by the local vision model (nothing leaves the PC). --with-agent also starts the coding
agent in a throwaway folder with a do-nothing task and then stops it.
Not run here: load_model / unload_model (they would replace a running Dual_gpu lane's configuration).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import psutil

from vox_assistant.config import PROJECT_ROOT, Settings
from vox_assistant.tools import build_registry

RESULTS: list[tuple[str, str, str]] = []


def record(ok: bool | None, name: str, detail: str = "") -> None:
    status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}{' - ' + detail if detail else ''}", flush=True)


async def main() -> int:
    settings = Settings.load()
    tmp = Path(tempfile.mkdtemp(prefix="voxa_desk_"))
    settings.data_dir = tmp
    reg, store = build_registry(settings)
    call = lambda _tool, **a: reg.get(_tool).handler(a)  # noqa: E731

    print("Windows")
    r = await call("open_app", name="notepad")
    record(r["ok"] and r["verified_running"], "open_app notepad", str(r.get("verified_running")))
    await asyncio.sleep(1.0)
    r = await call("list_windows")
    note = [w for w in r["windows"] if w["app"] == "notepad"]
    record(bool(note), "list_windows sees Notepad", f"{len(r['windows'])} windows")
    for action in ("minimize", "maximize", "restore", "focus"):
        r = await call("window_action", window="notepad", action=action)
        record(r["ok"], f"window_action {action}", r.get("error", "verified"))
        await asyncio.sleep(0.3)
    r = await call("close_window", window="notepad")
    record(r["ok"] and r.get("closed"), "close_window notepad", r.get("note") or "closed")

    print("Clipboard")
    # Save the raw clipboard, not through read_clipboard: that tool caps what it returns at 6000 characters.
    from vox_assistant.tools.clipboard import _powershell

    get = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard -Raw"
    code, saved, _ = await _powershell(get)
    saved = saved.rstrip("\r\n") if code == 0 else ""
    r = await call("set_clipboard", text="VoxVortex clipboard check")
    back = await call("read_clipboard")
    record(r["ok"] and back.get("content") == "VoxVortex clipboard check", "set + read clipboard")
    if saved:
        await _powershell("Set-Clipboard -Value $env:VOXA_CLIP", {"VOXA_CLIP": saved})
        _, now, _ = await _powershell(get)
        record(now.rstrip("\r\n") == saved, "clipboard restored in full", f"{len(saved)} chars")

    print("Projects, web, knowledge, screen")
    r = await call("list_projects")
    record(r["ok"] and PROJECT_ROOT.name in r["projects"], "list_projects", f"{len(r['projects'])} projects")
    r = await call("open_project", project="vox vortex pc assistant")
    record(r["ok"], "open_project in VS Code", f"verified_open={r.get('verified_open')} {r.get('note') or ''}")
    r = await call("web_search", query="VoxCPM2 voice cloning")
    record(r["ok"], "web_search opens the browser", r.get("url", r.get("error", "")))
    t0 = time.perf_counter()
    r = await call("search_notes", query="VoxCPM2 cached synthesis RTF")
    record(r["ok"] and bool(r["results"]), "search_notes", f"{len(r['results'])} hits in {time.perf_counter() - t0:.1f} s; top {r['results'][0]['path'] if r['results'] else '-'}")
    t0 = time.perf_counter()
    r = await call("look_at_screen", question="What applications are visible? One sentence.")
    record(r["ok"], "look_at_screen (local vision model)",
           f"{time.perf_counter() - t0:.1f} s: {(r.get('description') or r.get('error', ''))[:140]!r}")

    print("Workflows and models (read-only)")
    r = await call("run_workflow", name="status_check")
    record(r["ok"] and r["all_ok"], "run_workflow status_check", ", ".join(s["step"] for s in r.get("steps", [])))
    r = await call("available_models", query="qwen3.5")
    record(r["ok"] and r["models"], "available_models",
           r.get("error") or f"{r.get('total_models')} in catalog; gpus {[g['lane'] for g in r.get('gpus', [])]}")

    if "--with-agent" in sys.argv:
        print("Coding agent")
        proj_root = Path(tempfile.mkdtemp(prefix="voxa_agent_"))
        (proj_root / "scratch_project").mkdir()
        settings.search_roots = [str(proj_root)]
        reg2, store2 = build_registry(settings)
        before_pids = {p.pid for p in psutil.process_iter()}
        r = await reg2.get("delegate_coding_task").handler({"project": "scratch project", "task": "Reply with the word ready and do nothing else."})
        record(r["ok"] and r.get("verified_running"), "delegate_coding_task starts the agent", r.get("note") or r.get("error", ""))
        await asyncio.sleep(2)
        for p in psutil.process_iter(["name", "cwd"]):  # stop what we started: the agent and its cmd host
            if p.pid in before_pids:
                continue
            try:
                if (p.info["name"] or "").lower().startswith(("opencode", "cmd")) and str(proj_root) in (p.cwd() or ""):
                    p.kill()
            except psutil.Error:
                pass
        store2.close()
        shutil.rmtree(proj_root, ignore_errors=True)

    store.close()
    shutil.rmtree(tmp, ignore_errors=True)
    counts = {k: sum(1 for s, *_ in RESULTS if s == k) for k in ("PASS", "FAIL", "SKIP")}
    print("\nSummary: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
