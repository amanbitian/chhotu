"""Tool registry assembly."""

from __future__ import annotations

from ..config import Settings
from ..permissions import ToolRegistry
from . import apps, clipboard, files, knowledge, models, projects, reminders, screen, system, windows, workflows
from .reminders import ReminderStore


def build_registry(settings: Settings, store: ReminderStore | None = None) -> tuple[ToolRegistry, ReminderStore]:
    store = store or ReminderStore(settings.data_dir / "reminders.sqlite3")
    registry = ToolRegistry()
    for module_tools in (system.make_tools(settings), models.make_tools(settings), files.make_tools(settings),
                         knowledge.make_tools(settings), apps.make_tools(settings), projects.make_tools(settings),
                         windows.make_tools(settings), screen.make_tools(settings), clipboard.make_tools(settings),
                         reminders.make_tools(settings, store)):
        for tool in module_tools:
            registry.add(tool)
    for tool in workflows.make_tools(settings, registry):  # last: its steps refer to the tools above
        registry.add(tool)
    return registry, store
