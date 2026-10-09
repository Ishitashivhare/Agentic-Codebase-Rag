"""
BaseTool: the interface every tool implements.

This is the load-bearing abstraction for the whole tool layer:

- Open/closed principle: new tools (e.g. a future `run_tests` or
  `git_blame` tool) are added by writing a new BaseTool subclass, never
  by editing existing tools or the orchestrator.
- Liskov substitution: an orchestrator (or ToolRegistry) can treat any
  BaseTool interchangeably - it only ever calls `.run(**kwargs)` and
  reads a `ToolResult` back.
- Interface segregation: the interface is deliberately minimal (name,
  description, run) rather than forcing every tool to implement methods
  it doesn't need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .tool_result import ToolResult


class BaseTool(ABC):
    """Abstract base class for all agent-callable tools.

    Subclasses must set `name` and `description` as class attributes (or
    instance attributes set in `__init__`) and implement `run`.

    `name` is the stable identifier used for registration and lookup
    (e.g. "search_code"). `description` is a short, natural-language
    explanation intended for an LLM's tool-selection prompt - keep it
    accurate and specific, since the orchestrator's tool choice quality
    depends directly on it.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult.

        Implementations should:
          - Validate their own inputs and return `ToolResult.fail(...)`
            for expected/invalid-input failures rather than raising.
          - Raise `ToolExecutionError` only for unexpected runtime
            failures (I/O errors, downstream service failures) that the
            caller should be aware are exceptional, not just "no results".
          - Never raise a bare `Exception` - wrap it in a tool-layer
            exception so callers can catch predictably.
        """
        raise NotImplementedError

    def to_schema(self) -> Dict[str, Any]:
        """Return a minimal schema describing this tool, suitable for
        exposing to an LLM's tool-use / function-calling prompt or for
        a `list_tools()` introspection call. Subclasses may override to
        add parameter schemas without changing the base contract.
        """
        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return f"{self.__class__.__name__}(name={self.name!r})"
