"""
Tool layer for the codebase RAG assistant.

Public API:

    from src.tools import (
        ToolRegistry,
        ToolResult,
        BaseTool,
        SearchCodeTool,
        ReadFileTool,
        FindFileTool,
        GetRepositoryStructureTool,
        SearchSymbolTool,
        CodeRelationshipTool,
        build_default_tool_registry,
    )
"""

from .base_tool import BaseTool
from .exceptions import PathSecurityError, ToolError, ToolExecutionError, ToolInputError
from .factory import build_default_tool_registry
from .find_file_tool import FindFileTool
from .read_file_tool import ReadFileTool
from .repository_structure_tool import GetRepositoryStructureTool
from .search_symbol_tool import SearchSymbolTool
from .code_relationship_tool import CodeRelationshipTool
from .search_code_tool import SearchCodeTool
from .tool_registry import ToolRegistry
from .tool_result import ToolResult
from .vector_store_protocol import SimilaritySearchable

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "ToolError",
    "ToolInputError",
    "ToolExecutionError",
    "PathSecurityError",
    "SimilaritySearchable",
    "SearchCodeTool",
    "ReadFileTool",
    "FindFileTool",
    "GetRepositoryStructureTool",
    "SearchSymbolTool",
    "CodeRelationshipTool",
    "build_default_tool_registry",
]
