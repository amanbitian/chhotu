"""Tools, their risk level, and the gate that decides what needs the user's explicit approval.

READ      observe only; runs without asking.
ACT       changes something reversible (open an app, set a reminder); asks unless the user pre-approved the tool.
DESTRUCTIVE  hard to undo (delete, overwrite, evict a running model); always asks, and cannot be pre-approved.

The gate never looks at who is speaking. A voice that sounds like the owner is not proof of identity.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .audit import AuditLog


class Risk(str, enum.Enum):
    READ = "read"
    ACT = "act"
    DESTRUCTIVE = "destructive"


Handler = Callable[[dict], Awaitable[dict]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    risk: Risk
    handler: Handler
    describe: Callable[[dict], str] | None = None  # plain-language "what I am about to do", spoken before asking
    timeout_s: float | None = None  # None: the agent's default (loading a model on a GPU needs longer)
    # True for these arguments: this call is as risky as a destructive one, so it is always asked, even if pre-approved.
    escalate: Callable[[dict], bool] | None = None

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def intent(self, args: dict) -> str:
        return self.describe(args) if self.describe else f"{self.name} {json.dumps(args, ensure_ascii=False)}"


Confirmer = Callable[[str, Tool, dict], Awaitable[bool]]


class PermissionGate:
    def __init__(self, confirm: Confirmer | None, audit: AuditLog, auto_approve: set[str] | None = None) -> None:
        self.confirm = confirm
        self.audit = audit
        self.auto_approve = set(auto_approve or ())

    async def authorize(self, tool: Tool, args: dict) -> tuple[bool, str]:
        """Returns (allowed, decision label). The agent records the decision and the result in the audit log."""
        if tool.risk is Risk.READ:
            decision = "auto"
        elif tool.risk is Risk.ACT and tool.name in self.auto_approve and not (tool.escalate and tool.escalate(args)):
            decision = "pre-approved"
        elif self.confirm is None:
            decision = "denied-no-confirmer"
        else:
            decision = "approved" if await self.confirm(tool.intent(args), tool, args) else "denied"
        return decision in ("auto", "pre-approved", "approved"), decision


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


def ok(**data: Any) -> dict:
    return {"ok": True, **data}


def fail(error: str, **data: Any) -> dict:
    return {"ok": False, "error": error, **data}
