"""
Uniform result envelope returned by every tool.

Every tool returns a ToolResult instead of raw data or raising for
expected failure cases (file not found, empty query, etc.). This gives
the orchestrator a single, predictable shape to branch on ("did this
step succeed?") instead of needing per-tool try/except logic - the
orchestrator only needs to catch ToolExecutionError for genuinely
unexpected failures (see exceptions.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolResult:
    """Immutable result of a single tool invocation.

    Attributes:
        success: Whether the tool considers the call to have succeeded.
        data: The payload on success (shape varies per tool - documented
            in each tool's docstring). None on failure.
        error: A human-readable failure reason. None on success.
        metadata: Small, tool-specific extras (e.g. result counts,
            the resolved path, the k used for a search) that are useful
            for logging or for an agent's sufficiency check, without
            polluting `data`.
    """

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any = None, **metadata: Any) -> "ToolResult":
        """Build a successful result."""
        return cls(success=True, data=data, error=None, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ToolResult":
        """Build a failed result. `error` should be safe to show to an
        LLM or end user - do not put stack traces or secrets in it."""
        return cls(success=False, data=None, error=error, metadata=metadata)

    def __bool__(self) -> bool:
        """Allows `if result:` as a shorthand for `if result.success:`."""
        return self.success
