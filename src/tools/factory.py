"""
Factory for assembling the default tool layer.

This is the only place that knows about all four concrete tools at once.
Everything else (ToolRegistry, an orchestrator) depends only on
BaseTool/ToolRegistry, not on these concrete classes - so this function
is where dependency injection actually happens.

Usage from the existing app.py (illustrative - not a required change):

    from src.embeddings import load_embeddings
    from src.vectordb import create_vector_db
    from src.tools.factory import build_default_tool_registry

    embeddings = load_embeddings()
    db = create_vector_db(chunks, embeddings)          # existing code, untouched
    registry = build_default_tool_registry(repo_root=repo_path, vector_store=db)

    result = registry.run("search_code", query="how does authentication work")
    if result.success:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.code_index import CodeRelationshipIndex

from .code_relationship_tool import CodeRelationshipTool
from .find_file_tool import FindFileTool
from .read_file_tool import ReadFileTool
from .repository_structure_tool import GetRepositoryStructureTool
from .search_symbol_tool import SearchSymbolTool
from .search_code_tool import SearchCodeTool
from .tool_registry import ToolRegistry
from .vector_store_protocol import SimilaritySearchable


def build_default_tool_registry(
    repo_root: Union[str, Path],
    vector_store: SimilaritySearchable,
    relationship_index: Optional[CodeRelationshipIndex] = None,
) -> ToolRegistry:
    """Build a ToolRegistry populated with the four standard tools.

    Args:
        repo_root: Root directory of the indexed repository. Backs
            `read_file`, `find_file`, and `get_repository_structure`.
        vector_store: The existing FAISS store (or anything else
            satisfying `SimilaritySearchable`), e.g. the object returned
            by `src/vectordb.py::create_vector_db` or `load_vector_db`.
            Passed straight through - this function does not build or
            mutate the store.

    Returns:
        A ToolRegistry with "search_code", "read_file", "find_file", and
        "get_repository_structure" registered.
    """
    registry = ToolRegistry()
    registry.register(SearchCodeTool(vector_store=vector_store))
    registry.register(ReadFileTool(repo_root=repo_root))
    registry.register(FindFileTool(repo_root=repo_root))
    registry.register(GetRepositoryStructureTool(repo_root=repo_root))
    index = relationship_index or CodeRelationshipIndex(repo_root)
    registry.register(SearchSymbolTool(index=index))
    registry.register(CodeRelationshipTool(index=index))
    return registry
