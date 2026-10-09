"""Exact AST symbol lookup."""

from __future__ import annotations

from typing import Any, Optional

from src.code_index import CodeRelationshipIndex

from .base_tool import BaseTool
from .tool_result import ToolResult


class SearchSymbolTool(BaseTool):
    name = "search_symbol"
    description = "Find exact function, method, class, or variable definitions using the AST index."

    def __init__(self, index: CodeRelationshipIndex):
        self._index = index

    def run(self, symbol: str, exact: bool = True, **_: Any) -> ToolResult:
        if not isinstance(symbol, str) or not symbol.strip():
            return ToolResult.fail("symbol must be a non-empty string")
        if not isinstance(exact, bool):
            return ToolResult.fail("exact must be a boolean")
        records = self._index.search_symbols(symbol, exact=exact)
        return ToolResult.ok(
            [record.to_dict() for record in records],
            symbol=symbol,
            exact=exact,
            result_count=len(records),
        )
