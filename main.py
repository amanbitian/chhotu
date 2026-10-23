r"""VoxVortex Desktop launcher: one command opens the control panel, and the panel runs everything else.

    python main.py                               # automatically uses this project's .venv
    .venv\Scripts\python.exe main.py              # start the voice LLM and voice service, open the panel
    .venv\Scripts\python.exe main.py --listen     # ... and start hands-free listening too
    .venv\Scripts\python.exe main.py --no-services  # only the panel; start things from it yourself

Closing this window (Ctrl+C) stops what this launcher started. A service that was already running is left alone.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# People commonly use the Python shown by their editor's Run button. Keep the
# launcher one-click even when that is the global Python: the app's native/audio
# dependencies live in the project virtual environment.
def run_in_venv() -> int:
    """Ctrl+C reaches both processes (same console): the panel stops its services, so wait for it rather than
    leaving them half stopped. A second Ctrl+C ends it outright."""
    child = subprocess.Popen([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        return child.wait()
    except KeyboardInterrupt:
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.kill()
            return 130


if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    raise SystemExit(run_in_venv())

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from vox_assistant.config import Settings  # noqa: E402
from vox_assistant.services import ServiceManager  # noqa: E402
from vox_assistant.tasks import TaskManager  # noqa: E402
from vox_assistant.tools.knowledge import warm_index  # noqa: E402
from vox_assistant.webapp import API_VERSION, create_app  # noqa: E402


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def compatible_panel(url: str) -> bool:
    """True only when the listener is this version of the VoxVortex panel."""
    try:
        response = httpx.get(f"{url}/api/version", timeout=2.0)
        return response.status_code == 200 and response.json().get("api_version") == API_VERSION
    except (httpx.HTTPError, ValueError):
        return False


def next_available_port(preferred: int, attempts: int = 20) -> int | None:
    return next((port for port in range(preferred + 1, preferred + attempts + 1) if not port_in_use(port)), None)


def when_up(url: str, action, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            action()
            return
        except httpx.HTTPError:
            time.sleep(0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxVortex Desktop control panel")
    parser.add_argument("--no-browser", action="store_true", help="do not open the panel in the browser")
    parser.add_argument("--no-services", action="store_true", help="do not start the voice LLM and voice service")
    parser.add_argument("--listen", action="store_true", help="also start hands-free listening")
    parser.add_argument("--port", type=int, help="panel port (default: setting web_port)")
    args = parser.parse_args()

    settings = Settings.load()
    port = args.port or settings.web_port
    url = f"http://127.0.0.1:{port}"
    if port_in_use(port):
        if compatible_panel(url):
            print(f"VoxVortex Desktop is already running at {url}. Opening it.")
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        replacement = next_available_port(port)
        if replacement is None:
            print(f"Port {port} has an older or unrelated server, and no free nearby port was found.")
            return 1
        print(f"Port {port} has an older or unrelated server; starting the current panel on port {replacement} instead.")
        port = replacement
        url = f"http://127.0.0.1:{port}"

    services = ServiceManager(settings)
    services.panel_url = url  # hands-free mode sends coding tasks here, so they show in the Tasks panel
    tasks = TaskManager(settings)

    def start_services() -> None:
        for name in ("llm", "voice"):
            print(f"Starting {services.services[name].title}...")
            services.start(name)
        if args.listen:  # needs the LLM answering first
            when_up(f"{settings.llm_base_url}/models", lambda: print("Starting hands-free listening...") or services.start("listen"), 180.0)

    if not args.no_services:
        threading.Thread(target=start_services, name="autostart", daemon=True).start()
    threading.Thread(target=warm_index, args=(settings,), name="notes-index", daemon=True).start()
    if not args.no_browser:
        threading.Thread(target=when_up, args=(f"{url}/api/status", lambda: webbrowser.open(url)), daemon=True).start()

    print(f"VoxVortex Desktop: {url}   (Ctrl+C stops it and the services it started)")
    try:
        uvicorn.run(create_app(settings, services=services, tasks=tasks), host="127.0.0.1", port=port, log_level="warning")
    finally:
        print("Stopping services...")
        tasks.shutdown()
        services.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
