"""Does a model pick the right tool, ask before acting, and avoid false claims? And how fast does it start talking?

    .venv\\Scripts\\python.exe scripts\\agent_eval.py [--url http://127.0.0.1:8083/v1] [--reps 2]

Runs fixed requests through the real Agent and tools. Every state-changing action is *declined* by the stand-in
confirmer, so nothing is opened, copied or scheduled. Use it to compare candidate voice models.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

from vox_assistant.agent import Agent, Say, ToolFinished, claims_action
from vox_assistant.audit import AuditLog
from vox_assistant.config import Settings
from vox_assistant.llm_client import ToolChoiceLLM
from vox_assistant.permissions import PermissionGate
from vox_assistant.tools import build_registry

# (request, acceptable first tools; empty = answer without a tool)
CASES = [
    ("What's the temperature of my graphics cards?", {"gpu_status"}),
    ("How much memory is my PC using right now?", {"system_info"}),
    ("Which language models are loaded at the moment?", {"list_models", "gpu_status"}),
    ("Open Notepad.", {"open_app"}),
    ("Please launch the calculator for me.", {"open_app"}),
    ("Find files called README in my projects.", {"find_files"}),
    ("Remind me in ten minutes to stretch.", {"set_reminder"}),
    ("What's on my clipboard?", {"read_clipboard"}),
    ("Copy the word hello to the clipboard.", {"set_clipboard"}),
    ("Open youtube dot com in the browser.", {"open_url"}),
    ("Tell me a one sentence joke about cats.", set()),
    ("What is the capital of France?", set()),
    # P2 features
    ("Switch to the Cutline project.", {"open_project"}),
    ("What's on my screen right now?", {"look_at_screen"}),
    ("Which windows are open?", {"list_windows"}),
    ("Minimize the Notepad window.", {"window_action", "list_windows"}),  # looking first is fine
    ("Close the calculator window.", {"close_window"}),
    ("Search my notes for VoxCPM2 latency.", {"search_notes"}),
    ("Which models could I load on the GPUs?", {"available_models"}),
    ("Unload the model on the 9070 XT.", {"unload_model"}),
    ("Load Gemma 4 12B on the 9070 XT.", {"load_model", "available_models"}),
    ("Run my status check workflow.", {"run_workflow"}),
    ("Ask the coding agent to add unit tests to the VoxVortex project.", {"delegate_coding_task"}),
    ("Search the web for ROCm 7.2 release notes.", {"web_search"}),
]


async def run_case(agent: Agent, text: str) -> dict:
    agent.reset()
    t0 = time.perf_counter()
    first_say = None
    tools, said = [], ""
    async for ev in agent.run(text):
        if isinstance(ev, Say):
            first_say = first_say or time.perf_counter() - t0
            said += ev.text
        elif isinstance(ev, ToolFinished):
            tools.append((ev.name, ev.decision))
    return {"tools": tools, "said": said.strip(), "first_say": first_say, "total": time.perf_counter() - t0}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=Settings.load().llm_base_url)
    ap.add_argument("--model", default="default")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="voxa_eval_"))
    settings = Settings.load(tmp)
    audit = AuditLog(tmp / "audit.sqlite3")
    registry, store = build_registry(settings)
    asked: list[str] = []

    async def decline(intent, tool, call_args):
        asked.append(tool.name)
        return False

    llm = ToolChoiceLLM(args.url, args.model, temperature=0.3, timeout=120.0, first_token_timeout=90.0)
    agent = Agent(llm, registry, PermissionGate(decline, audit), audit)
    await run_case(agent, "Hello.")  # load / warm the model so the first case is not a cold start
    good = total = 0
    firsts: list[float] = []
    print(f"Model at {args.url}")
    for text, want in CASES:
        for _ in range(args.reps):
            asked.clear()
            r = await run_case(agent, text)
            first = r["tools"][0][0] if r["tools"] else None
            ok_tool = (first in want) if want else first is None
            # A declined action must not be reported as done.
            false_claim = any(d == "denied" for _, d in r["tools"]) and claims_action(r["said"].split(".")[0])
            ok = ok_tool and not false_claim
            good += ok
            total += 1
            if r["first_say"]:
                firsts.append(r["first_say"])
            fs = f"{r['first_say']:.2f}s" if r["first_say"] else "  -  "
            print(f"  {'ok ' if ok else 'BAD'} {fs} {r['total']:5.2f}s  {text[:44]:44s} tools={r['tools']}  "
                  f"{'FALSE-CLAIM ' if false_claim else ''}{r['said'][:70]!r}", flush=True)
    firsts.sort()
    median = firsts[len(firsts) // 2] if firsts else float("nan")
    print(f"\n{good}/{total} correct; median time to first words {median:.2f}s")
    await llm.aclose()
    audit.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if good == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
