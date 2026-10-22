"""Workflows: named sequences of tool calls from `config.toml`, run on one spoken request with one approval.

    [workflows.start_work]
    description = "open VS Code and the dashboard"
    steps = [{tool = "open_app", args = {name = "vs code"}}, {tool = "open_url", args = {url = "http://127.0.0.1:8090"}}]

The confirmation names every step, so approving the workflow approves exactly those steps. Destructive tools are
not allowed in a workflow: each of those must be asked for on its own.
"""

from __future__ import annotations

import asyncio

from ..config import Settings
from ..permissions import Risk, Tool, ToolRegistry, fail, ok

STEP_TIMEOUT_S = 30.0


def validate(workflows: dict, registry: ToolRegistry) -> list[str]:
    problems = []
    for name, wf in workflows.items():
        steps = wf.get("steps") if isinstance(wf, dict) else None
        if not steps:
            problems.append(f"workflow {name!r} has no steps")
            continue
        for step in steps:
            tool = registry.get(str(step.get("tool")))
            if tool is None:
                problems.append(f"workflow {name!r}: unknown tool {step.get('tool')!r}")
            elif tool.risk is Risk.DESTRUCTIVE:
                problems.append(f"workflow {name!r}: {tool.name} is destructive and must be asked for on its own")
    return problems


def make_tools(settings: Settings, registry: ToolRegistry) -> list[Tool]:
    workflows = settings.workflows
    if not workflows:
        return []

    def steps_of(name: str) -> list[tuple[Tool, dict]]:
        wf = workflows.get(name) or {}
        return [(registry.get(str(s.get("tool"))), dict(s.get("args") or {})) for s in wf.get("steps") or []]

    def describe(args: dict) -> str:
        name = str(args.get("name", ""))
        parts = [t.intent(a) if t else f"(unknown step)" for t, a in steps_of(name)]
        return f"run the {name} workflow: " + ", then ".join(parts)

    async def run_workflow(args: dict) -> dict:
        name = str(args.get("name", "")).strip().lower().replace(" ", "_")
        if name not in workflows:
            return fail("No workflow by that name.", workflows=sorted(workflows))
        problems = [p for p in validate({name: workflows[name]}, registry)]
        if problems:
            return fail("This workflow is misconfigured: " + "; ".join(problems))
        results = []
        for tool, step_args in steps_of(name):
            try:
                r = await asyncio.wait_for(tool.handler(step_args), tool.timeout_s or STEP_TIMEOUT_S)
            except Exception as exc:  # one broken step is reported, the rest still run
                r = fail(f"{type(exc).__name__}: {exc}")
            results.append({"step": tool.name, **r})
        return ok(workflow=name, steps=results, all_ok=all(r.get("ok") for r in results))

    listing = "; ".join(f"{n} ({(wf or {}).get('description', '')})" for n, wf in workflows.items())
    return [Tool("run_workflow", f"Run one of the user's saved workflows by name. Available: {listing}.",
                 {"type": "object", "properties": {"name": {"type": "string", "enum": sorted(workflows)}}, "required": ["name"]},
                 Risk.ACT, run_workflow, describe=describe, timeout_s=180.0)]
