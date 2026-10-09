"""
ToolRegistry - registers tools and dispatches calls by name.

This is what an agent orchestrator talks to instead of importing each
tool class directly - it depends on the BaseTool abstraction (dependency
inversion), so registering a fifth tool later never requires touching
the orchestrator's code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base_tool import BaseTool
from .exceptions import ToolInputError
from .tool_result import ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """A simple, explicit name -> BaseTool registry.

    Single responsibility: registration and dispatch. It does not decide
    *which* tool to call for a given query - that is the orchestrator's
    job (query planning), kept deliberately out of this class.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance under its `name`.

        Raises:
            ToolInputError: If `tool` is not a BaseTool, or if a tool
                with the same name is already registered (registration
                order bugs should fail loudly, not silently overwrite).
        """
        if not isinstance(tool, BaseTool):
            raise ToolInputError(f"Only BaseTool instances can be registered, got {type(tool)!r}")
        if tool.name in self._tools:
            raise ToolInputError(f"A tool named '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool:
        """Look up a tool by name.

        Raises:
            ToolInputError: If no tool is registered under `name`.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools)) or "<none>"
            raise ToolInputError(
                f"No tool registered under name '{name}'. Available tools: {available}"
            ) from exc

    def has(self, name: str) -> bool:
        """Return True if a tool with `name` is registered."""
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the schema of every registered tool, e.g. for an LLM's
        tool-use / function-calling prompt."""
        return [tool.to_schema() for tool in self._tools.values()]

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up `name` and invoke it with `kwargs`.

        This is the single entry point an orchestrator needs: it never
        has to import a concrete tool class.

        Raises:
            ToolInputError: If `name` is not registered.
            ToolExecutionError: Propagated from the tool itself if the
                underlying operation fails unexpectedly.
        """
        tool = self.get(name)
        return tool.run(**kwargs)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)
