"""Start, stop and watch the processes the assistant runs on: the voice LLM, the cloned-voice service, hands-free mode.

The launcher (main.py) and the web app share one ServiceManager, so the page can show what is up and control it.
A service that is already answering its health URL when we are asked to start it (started by hand, or by
scripts/start_all.ps1) is reported as "external": it is shown, but never stopped from here.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import PROJECT_ROOT, Settings

LOG_LINES = 400
EVENT_LINES = 800
EVENT_PREFIX = "@@VOXA_EVENT@@"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Service:
    name: str
    title: str
    command: list[str]
    health_url: str | None  # None: no endpoint (hands-free mode); "running" then means the process is alive
    startup_s: float  # how long a start may take before it counts as failed
    needs: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None
    started_at: float = 0.0
    log: collections.deque = field(default_factory=lambda: collections.deque(maxlen=LOG_LINES))
    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=EVENT_LINES))
    event_seq: int = 0
    run_id: int = 0
    exit_code: int | None = None
    stopped_by_user: bool = False
    flags: dict = field(default_factory=dict)  # live state reported by the child, e.g. guest_mode switched by voice

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


class ServiceManager:
    def __init__(self, settings: Settings, python: str | None = None) -> None:
        self.settings = settings
        py = python or sys.executable
        self.services: dict[str, Service] = {
            "llm": Service("llm", "Voice LLM (Qwen on the R9700)", [py, "-m", "vox_assistant.cli", "llm-server"],
                           f"{settings.llm_base_url}/models", 120.0),
            "voice": Service("voice", "Cloned-voice service (VoxCPM2 + speech recognition)",
                             ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              str(PROJECT_ROOT / "scripts" / "start_tts.ps1")], f"{settings.tts_url}/health", 180.0),
            "listen": Service("listen", "Hands-free listening (microphone and speakers)",
                              [py, "-m", "vox_assistant.cli", "listen"], None, 60.0, needs=["llm"]),
        }
        self._lock = threading.Lock()
        # The control panel's address, set by whoever serves it: hands-free mode sends coding tasks there to be tracked.
        self.panel_url: str | None = None

    # ---- state -------------------------------------------------------------------------------------------------

    def _healthy(self, svc: Service) -> bool:
        if svc.health_url is None:
            return svc.alive()
        try:
            return httpx.get(svc.health_url, timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def state(self, name: str) -> dict:
        svc = self.services[name]
        healthy = self._healthy(svc)
        if svc.alive():
            if healthy:
                state = "running"
            elif svc.health_url is not None and time.monotonic() - svc.started_at > svc.startup_s:
                state = "failed"
            else:
                state = "starting"
        elif healthy:
            state = "external"
        else:
            state = "failed" if svc.exit_code not in (None, 0) and not svc.stopped_by_user else "stopped"
        return {"name": name, "title": svc.title, "state": state, "pid": svc.process.pid if svc.alive() else None,
                "exit_code": svc.exit_code, "controllable": state != "external"}

    def states(self) -> list[dict]:
        return [self.state(n) for n in self.services]

    def logs(self, name: str, lines: int = 120) -> list[str]:
        return list(self.services[name].log)[-max(1, min(lines, LOG_LINES)):]

    def events(self, name: str, after: int = 0, run_id: int | None = None) -> dict:
        """Structured child-process events for the web conversation.

        A run id prevents a browser from missing events when the listener is restarted and
        sequence numbers begin at one again.
        """
        svc = self.services[name]
        if run_id != svc.run_id:
            after = 0
        return {"run_id": svc.run_id, "events": [e for e in svc.events if e["seq"] > after]}

    # ---- control -----------------------------------------------------------------------------------------------

    def _env(self) -> dict:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        env.setdefault("VOXA_DATA_DIR", str(self.settings.data_dir))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]))
        if self.panel_url:
            env["VOXA_PANEL_URL"] = self.panel_url
        return env

    def _pump(self, svc: Service, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(line[len(EVENT_PREFIX):])
                    svc.event_seq += 1
                    svc.events.append({"seq": svc.event_seq, **event})
                    if event.get("kind") == "guest_mode":
                        svc.flags["guest_mode"] = bool(event.get("on"))
                except (json.JSONDecodeError, TypeError):
                    svc.log.append("[could not decode a hands-free event]")
                continue
            svc.log.append(line)
        svc.exit_code = proc.wait()
        svc.log.append(f"[exited with code {svc.exit_code}]")

    def start(self, name: str, extra_args: list[str] | None = None) -> dict:
        svc = self.services[name]
        with self._lock:
            current = self.state(name)["state"]
            if current in ("running", "starting", "external"):
                return self.state(name)
            for dep in svc.needs:  # hands-free mode is useless without the LLM; say so instead of failing inside the child
                if self.state(dep)["state"] not in ("running", "external"):
                    raise RuntimeError(f"Start the {self.services[dep].title} first.")
            svc.log.clear()
            svc.events.clear()
            svc.flags.clear()
            svc.event_seq = 0
            svc.run_id += 1
            svc.exit_code, svc.stopped_by_user = None, False
            env = self._env()
            if name == "listen":
                env["VOXA_EVENT_STREAM"] = "1"
            svc.process = subprocess.Popen(
                svc.command + list(extra_args or []), cwd=str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            svc.started_at = time.monotonic()
            threading.Thread(target=self._pump, args=(svc, svc.process), name=f"svc-{name}", daemon=True).start()
        return self.state(name)

    def stop(self, name: str) -> dict:
        svc = self.services[name]
        with self._lock:
            if svc.alive():
                svc.stopped_by_user = True
                _kill_tree(svc.process)
                try:
                    svc.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # taskkill can fail (for example while the child is still
                    # attaching its console). Ensure the direct child is gone
                    # before reporting the service as stopped.
                    svc.process.kill()
                    svc.process.wait(timeout=5)
        return self.state(name)

    def shutdown(self) -> None:
        """Stop what this manager started (hands-free first, then the services it relies on). External ones stay up."""
        for name in ("listen", "voice", "llm"):
            self.stop(name)


def _kill_tree(proc: subprocess.Popen) -> None:
    """llama-server and the voice service are grandchildren of the launched shell; end the whole tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        if proc.poll() is None:
            proc.kill()
    else:
        proc.terminate()
