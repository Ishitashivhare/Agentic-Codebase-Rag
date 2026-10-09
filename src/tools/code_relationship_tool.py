"""AST-based relationship and bounded call-graph traversal tool."""

from __future__ import annotations

from typing import Any

from src.code_index import CodeRelationshipIndex

from .base_tool import BaseTool
from .tool_result import ToolResult


class CodeRelationshipTool(BaseTool):
    name = "code_relationship"
    description = "Identify definitions, callers, callees, imports, class methods, and bounded call paths."

    def __init__(self, index: CodeRelationshipIndex):
        self._index = index

    def run(
        self,
        symbol: str,
        mode: str = "relationships",
        direction: str = "both",
        max_depth: int = 2,
        **_: Any,
    ) -> ToolResult:
        if not isinstance(symbol, str) or not symbol.strip():
            return ToolResult.fail("symbol must be a non-empty string")
        if mode not in {"relationships", "traverse"}:
            return ToolResult.fail("mode must be 'relationships' or 'traverse'")
        if direction not in {"both", "callers", "callees"}:
            return ToolResult.fail("direction must be 'both', 'callers', or 'callees'")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or not 1 <= max_depth <= 10:
            return ToolResult.fail("max_depth must be an integer from 1 to 10")

        if mode == "relationships":
            data = self._index.relationships_for(symbol)
            count = sum(len(value) for value in data.values() if isinstance(value, list))
        else:
            data = self._index.traverse(symbol, direction=direction, max_depth=max_depth)
            count = len(data)
        return ToolResult.ok(data, symbol=symbol, mode=mode, direction=direction, max_depth=max_depth, result_count=count)
