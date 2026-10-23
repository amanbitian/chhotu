"""Tracked coding-agent jobs for the desktop control panel.

The older project tool opened an untracked terminal.  Jobs started here run in the
project folder with captured output, so the web UI can show whether they are still
working, what they most recently reported, and their final exit status.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Settings
from .tools.projects import agent_command, cmd_task

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAVED_LINES = 200  # output lines kept per task in tasks.json


class TaskBusy(RuntimeError):
    """A coding task is already running in that project: two agents editing the same files would collide."""


@dataclass
class AgentTask:
    id: str
    project: Path
    prompt: str
    allow_changes: bool
    created_at: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = "running"
    exit_code: int | None = None
    process: subprocess.Popen | None = None
    output: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=1000))


class TaskManager:
    """Start and observe a small history of coding-agent jobs, kept in `tasks.json` across restarts."""

    def __init__(self, settings: Settings, history: int = 30) -> None:
        self.settings = settings
        self.history = history
        self._tasks: dict[str, AgentTask] = {}
        self._lock = threading.RLock()
        self._file = Path(settings.data_dir) / "tasks.json"
        self._load()

    # ---- persistence ---------------------------------------------------------------------------------------

    def _load(self) -> None:
        try:
            saved = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for t in saved[-self.history:] if isinstance(saved, list) else []:
            try:
                job = AgentTask(t["id"], Path(t["path"]), t["prompt"], bool(t["allow_changes"]), t["created_at"],
                                t["started_at"], t.get("ended_at"), t["status"], t.get("exit_code"))
            except (KeyError, TypeError):
                continue
            if job.status == "running":  # the panel stopped without seeing it end
                job.status, job.ended_at = "interrupted", job.ended_at or job.started_at
            job.output.extend(t.get("output") or [])
            self._tasks[job.id] = job

    def _save(self) -> None:
        with self._lock:
            data = [{**self.state(i), "output": list(self._tasks[i].output)[-SAVED_LINES:]} for i in self._tasks]
        try:
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._file)
        except OSError:
            pass  # history is a convenience; a full disk must not stop a task

    # ---- control -------------------------------------------------------------------------------------------

    def _command(self, prompt: str, allow_changes: bool) -> list[str]:
        template = list(self.settings.coding_agent)
        if not template:
            raise RuntimeError("The coding_agent setting is empty.")
        exe = shutil.which(template[0])
        if not exe:
            raise RuntimeError(f"The coding agent `{template[0]}` is not installed or not on PATH.")

        # OpenCode's interactive `--prompt` UI cannot be captured. Its `run`
        # command is the equivalent non-interactive mode intended for automation.
        if Path(exe).stem.lower() == "opencode":
            command = [exe, "run"]
            if allow_changes:
                command.append("--auto")
            command.append(cmd_task(prompt))
            return command
        if allow_changes:  # silently running without it would let the user believe changes are auto-approved
            raise ValueError(f"Letting the agent change files without asking is only wired up for OpenCode; set "
                             f"{Path(exe).stem}'s own permission flags in the coding_agent setting instead.")
        command = agent_command([exe, *template[1:]], prompt)
        return command

    @staticmethod
    def _popen_command(command: list[str]) -> list[str]:
        if os.name == "nt" and Path(command[0]).suffix.lower() in (".cmd", ".bat"):
            # CreateProcess cannot execute a batch shim directly. The prompt has
            # already passed through cmd_task, so cmd metacharacters are removed.
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", subprocess.list2cmdline(command)]
        return command

    def start(self, project: Path, prompt: str, *, allow_changes: bool = False) -> dict:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Describe the task first.")
        command = self._command(prompt, allow_changes)
        job = AgentTask(uuid.uuid4().hex[:12], project.resolve(), prompt, allow_changes)
        with self._lock:  # held across the start, so two requests cannot both pass the check
            busy = next((t for t in self._tasks.values() if t.project == job.project and t.status == "running"), None)
            if busy is not None:
                raise TaskBusy(f"A coding task is already running in {job.project.name}; stop it or wait for it to finish.")
            try:
                job.process = subprocess.Popen(
                    self._popen_command(command), cwd=str(job.project), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    bufsize=1, creationflags=NO_WINDOW,
                )
            except OSError as exc:
                raise RuntimeError(f"Could not start the coding agent: {exc}") from exc
            job.output.append(f"Started coding agent in {job.project.name}.")
            self._tasks[job.id] = job
            while len(self._tasks) > self.history:
                oldest = next(iter(self._tasks))
                if self._tasks[oldest].status == "running":
                    break
                self._tasks.pop(oldest)
        threading.Thread(target=self._pump, args=(job,), name=f"task-{job.id}", daemon=True).start()
        limit_min = self.settings.coding_task_timeout_min
        if limit_min > 0:
            timer = threading.Timer(limit_min * 60, self._stop, args=(job.id, "timed out"))
            timer.daemon = True
            timer.start()
        self._save()
        return self.state(job.id)

    def _pump(self, job: AgentTask) -> None:
        assert job.process is not None and job.process.stdout is not None
        for raw in job.process.stdout:
            line = _ANSI.sub("", raw).rstrip()
            if line:
                with self._lock:
                    job.output.append(line)
        code = job.process.wait()
        with self._lock:
            job.exit_code = code
            job.ended_at = time.time()
            if job.status == "running":
                job.status = "completed" if code == 0 else "failed"
            job.output.append({
                "completed": "Task completed successfully.", "cancelled": "Task was stopped.",
                "timed out": f"Task was stopped after {self.settings.coding_task_timeout_min} minutes "
                             "(setting coding_task_timeout_min).",
            }.get(job.status, f"Task exited with code {code}."))
        self._save()

    def state(self, task_id: str, *, include_output: bool = True) -> dict:
        with self._lock:
            job = self._tasks.get(task_id)
            if job is None:
                raise KeyError(task_id)
            end = job.ended_at or time.time()
            result = {
                "id": job.id, "project": job.project.name, "path": str(job.project), "prompt": job.prompt,
                "allow_changes": job.allow_changes, "status": job.status,
                "created_at": job.created_at, "started_at": job.started_at, "ended_at": job.ended_at,
                "elapsed_s": round(max(0.0, end - job.started_at), 1), "exit_code": job.exit_code,
                "pid": job.process.pid if job.process is not None and job.process.poll() is None else None,
                "last_line": job.output[-1] if job.output else "",
                "output_lines": len(job.output),
            }
            if include_output:
                result["output"] = list(job.output)
            return result

    def states(self) -> list[dict]:
        with self._lock:
            ids = list(self._tasks)
        return [self.state(task_id, include_output=False) for task_id in reversed(ids)]

    def stop(self, task_id: str) -> dict:
        return self._stop(task_id, "cancelled")

    def _stop(self, task_id: str, status: str) -> dict:
        with self._lock:
            job = self._tasks.get(task_id)
            if job is None:
                raise KeyError(task_id)
            proc = job.process
            if job.status != "running" or proc is None or proc.poll() is not None:
                return self.state(task_id)
            job.status = status
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=NO_WINDOW)
            if proc.poll() is None:
                proc.kill()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return self.state(task_id)

    def shutdown(self) -> None:
        for task in self.states():
            if task["status"] == "running":
                self.stop(task["id"])
        with self._lock:
            for job in self._tasks.values():
                if job.status == "cancelled" and job.ended_at is None:
                    job.ended_at = time.time()
        self._save()


class PanelTasks:
    """The control panel's TaskManager as seen from the hands-free child process (VOXA_PANEL_URL), so a coding task
    asked for by voice is tracked in the Tasks panel like one started from the page."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def start(self, project: Path, prompt: str, *, allow_changes: bool = False) -> dict:
        try:
            r = httpx.post(f"{self.url}/api/tasks", timeout=15.0,
                           json={"project": project.name, "task": prompt, "allow_changes": allow_changes})
        except httpx.HTTPError as exc:
            raise RuntimeError(f"The control panel did not answer ({type(exc).__name__}).") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("detail") or r.text
            except ValueError:
                detail = r.text
            raise RuntimeError(str(detail)[:300])
        return r.json()
