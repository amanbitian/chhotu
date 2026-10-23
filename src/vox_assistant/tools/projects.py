"""Projects: open one in VS Code ("switch to the Cutline project") and hand a coding task to a coding agent.

A project is a folder directly inside one of `search_roots`. The coding agent is a configured command
(`coding_agent`, OpenCode by default). With a task manager (the control panel, and hands-free mode started by it) the
job is tracked in the Tasks panel with its output and a stop button; without one it starts in a new, visible terminal,
so the user can watch and stop it. Letting a tracked agent change files without asking is said in the approval
question and is never pre-approved.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from ..config import Settings
from ..permissions import Risk, Tool, fail, ok
from .apps import vscode_exe
from .files import SKIP_DIRS

if TYPE_CHECKING:
    from ..tasks import TaskManager

# cmd.exe (and ';' for Windows Terminal) would treat these as syntax inside the task text; they become spaces.
_CMD_UNSAFE = re.compile(r'["&|<>^%!;\r\n]')


def _norm(name: str) -> str:
    return re.sub(r"[\W_]+", "", name.lower())


def list_projects(settings: Settings) -> list[Path]:
    out = []
    for root in settings.search_roots:
        try:
            out += [p for p in Path(root).iterdir() if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")]
        except OSError:
            continue
    return out


def match_project(name: str, projects: list[Path]) -> Path | list[Path] | None:
    """Ignores case, spaces, '_' and '-' ("vox vortex pc assistant" finds VoxVortex_pc_assistant)."""
    want = _norm(re.sub(r"^(the|my)\s+|\s+(project|repo|folder)$", "", name.strip(), flags=re.I))
    if not want:
        return None
    exact = [p for p in projects if _norm(p.name) == want]
    if len(exact) == 1:
        return exact[0]
    loose = [p for p in projects if want in _norm(p.name)]
    if len(loose) == 1:
        return loose[0]
    return loose or None


def allows_changes(args: dict) -> bool:
    """The model's `allow_changes`, read strictly: only a real true (or "true") counts; bool("false") would be True."""
    value = args.get("allow_changes", False)
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def describe_task(args: dict, tracked: bool = True) -> str:
    text = f"start the coding agent in {args.get('project')} with the task: {str(args.get('task', ''))[:120]}"
    # Said in the approval question: approving this lets the agent change files without asking you first. (Untracked,
    # the agent runs in a visible terminal and asks there itself.)
    return text + (", and let it change files without asking you" if tracked and allows_changes(args) else "")


def cmd_task(task: str) -> str:
    return " ".join(_CMD_UNSAFE.sub(" ", task).split())


def agent_command(template: list[str], task: str) -> list[str]:
    return [part.replace("{task}", cmd_task(task)) for part in template]


def launch_line(template: list[str], folder: Path, exe: str) -> str:
    """A `start` command line for a new console in `folder` running the agent.

    - The task is not in this text: it is read from %VOXA_TASK% with delayed expansion (!VOXA_TASK!), which cmd applies
      *after* parsing, so nothing in it can act as cmd syntax.
    - The agent is started by its full path (`exe`, resolved by Python): cmd's own PATH search stops at a PATH entry
      with a stray quote, which this PC has, and then cannot find npm tools.
    - `/k ""exe" args"`: cmd strips one outer pair of quotes from a /k command that has more than two.
    - Windows Terminal's `wt` is not used: it re-splits its arguments and drops the quotes."""
    def arg(part: str) -> str:
        return f'"{part}"'

    parts = [exe, *(p.replace("{task}", "!VOXA_TASK!") for p in template[1:])]
    return f'cmd /c start "" /D "{folder}" cmd /v:on /k "' + " ".join(arg(p) for p in parts) + '"'


def make_tools(settings: Settings, task_manager: TaskManager | None = None) -> list[Tool]:
    def resolve(args: dict) -> tuple[Path | None, dict | None]:
        found = match_project(str(args.get("project", "")), list_projects(settings))
        if found is None:
            return None, fail("I could not find a project folder with that name.",
                              projects=sorted(p.name for p in list_projects(settings))[:40])
        if isinstance(found, list):
            return None, fail("Several projects match; which one?", candidates=[p.name for p in found[:10]])
        return found, None

    async def projects(args: dict) -> dict:
        return ok(projects=sorted(p.name for p in list_projects(settings)))

    async def open_project(args: dict) -> dict:
        folder, err = resolve(args)
        if err:
            return err
        exe = vscode_exe(settings)
        if not exe:
            return fail("VS Code's `code` command is not on PATH.")
        before = {p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "code.exe"}
        subprocess.Popen([exe, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):  # VS Code reuses its main process, so look for its window title instead of a new process
            await asyncio.sleep(0.25)
            from .windows import list_windows

            if any(w["app"] == "code" and folder.name.lower() in w["title"].lower() for w in list_windows()):
                return ok(project=folder.name, path=str(folder), verified_open=True)
        running = bool(before) or any((p.info["name"] or "").lower() == "code.exe" for p in psutil.process_iter(["name"]))
        return ok(project=folder.name, path=str(folder), verified_open=False,
                  note="VS Code was asked to open it" + (" but I could not see its window yet." if running else
                                                         " but it does not seem to be running."))

    async def delegate_coding_task(args: dict) -> dict:
        folder, err = resolve(args)
        if err:
            return err
        task = str(args.get("task", "")).strip()
        if not task:
            return fail("What should the coding agent do?")
        if task_manager is not None:
            try:
                job = task_manager.start(folder, task, allow_changes=allows_changes(args))
            except (RuntimeError, ValueError) as exc:
                return fail(str(exc))
            return ok(project=folder.name, agent=settings.coding_agent[0], task_id=job["id"],
                      status=job["status"],
                      note="The task is visible in the Tasks panel, including its live output and stop control.")
        cmd = agent_command(settings.coding_agent, task)
        exe = shutil.which(cmd[0])
        if not exe:
            return fail(f"The coding agent `{cmd[0]}` is not installed or not on PATH (setting coding_agent).")
        # A new visible console in the project folder, so the user sees and approves what the agent does.
        try:
            subprocess.Popen(launch_line(settings.coding_agent, folder, exe), env={**os.environ, "VOXA_TASK": cmd_task(task)},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as exc:
            return fail(f"Could not start a terminal: {exc.strerror}")
        stem = Path(exe).stem.lower()
        for _ in range(24):
            await asyncio.sleep(0.25)
            if any(stem in (p.info["name"] or "").lower() for p in psutil.process_iter(["name"])):
                return ok(project=folder.name, agent=cmd[0], verified_running=True,
                          note="It runs in a new terminal window, where the user can watch and stop it.")
        return ok(project=folder.name, agent=cmd[0], verified_running=False,
                  note="The terminal was opened but I could not see the agent process yet.")

    project_arg = {"project": {"type": "string", "description": "project folder name, loosely matched"}}
    tracked = task_manager is not None
    return [
        Tool("list_projects", "List the project folders I can open.", {"type": "object", "properties": {}}, Risk.READ, projects),
        Tool("open_project", "Open (switch to) a project folder in VS Code.",
             {"type": "object", "properties": project_arg, "required": ["project"]}, Risk.ACT, open_project,
             describe=lambda a: f"open the {a.get('project')} project in VS Code"),
        Tool("delegate_coding_task", "Hand a coding task to the local coding agent and track it in the Tasks panel."
             if tracked else "Hand a coding task to the local coding agent, in a new terminal window in the project folder.",
             {"type": "object", "properties": {**project_arg, "task": {"type": "string"},
              "allow_changes": {"type": "boolean", "description": "Auto-approve the coding agent only when the user explicitly asks it to make changes."}},
              "required": ["project", "task"]},
             Risk.ACT, delegate_coding_task, describe=lambda a: describe_task(a, tracked),
             escalate=allows_changes if tracked else None),
    ]
