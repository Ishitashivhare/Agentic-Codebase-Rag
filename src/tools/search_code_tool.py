"""
search_code(query) - semantic search over the indexed codebase.

Reuses the vector store produced by the existing `src/vectordb.py` and
`src/embeddings.py` pipeline unchanged. This tool is a thin, validated
wrapper around `similarity_search` - it does not re-embed, re-chunk, or
rebuild the index.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool
from .exceptions import ToolExecutionError, ToolInputError
from .tool_result import ToolResult
from .vector_store_protocol import SimilaritySearchable

logger = logging.getLogger(__name__)


class SearchCodeTool(BaseTool):
    """Semantic search tool backed by an existing FAISS vector store.

    Single responsibility: translate a natural-language query into
    ranked code chunks. It does not know how the store was built and
    does not manage its lifecycle - that stays in `vectordb.py`.
    """

    name = "search_code"
    description = (
        "Semantic search over the indexed codebase. Given a natural-"
        "language query, returns the most similar code chunks. Best for "
        "'how does X work' or conceptual questions; use find_file for "
        "locating a file by name and read_file for exact contents."
    )

    def __init__(
        self,
        vector_store: SimilaritySearchable,
        default_k: int = 5,
        max_k: int = 20,
    ) -> None:
        """
        Args:
            vector_store: An already-built store exposing
                `similarity_search(query, k)`, e.g. the object returned
                by the existing `create_vector_db` / `load_vector_db`.
            default_k: Number of chunks to return when the caller does
                not specify one.
            max_k: Hard ceiling on `k`, regardless of what the caller
                requests, to bound latency and prompt size.

        Raises:
            ToolInputError: If `vector_store` is missing the required
                interface, or if the k bounds are invalid.
        """
        if vector_store is None or not hasattr(vector_store, "similarity_search"):
            raise ToolInputError(
                "vector_store must implement similarity_search(query, k)"
            )
        if default_k <= 0 or max_k <= 0 or default_k > max_k:
            raise ToolInputError(
                f"Invalid k bounds: default_k={default_k}, max_k={max_k}"
            )

        self._vector_store = vector_store
        self._default_k = default_k
        self._max_k = max_k

    def run(self, query: str, k: Optional[int] = None) -> ToolResult:
        """
        Args:
            query: Natural-language search query. Must be a non-empty
                string.
            k: Number of results to return. Defaults to `default_k`,
                clamped to `max_k`.

        Returns:
            ToolResult whose `data` is a list of dicts:
                {"rank": int, "content": str, "source": str, "metadata": dict}
            `source` falls back to "unknown" when the underlying chunks
            carry no metadata (true of the current `vectordb.py`, which
            calls `FAISS.from_texts` without metadata) - callers should
            not assume file provenance is always available yet.
        """
        if not isinstance(query, str) or not query.strip():
            return ToolResult.fail("query must be a non-empty string")

        search_k = self._default_k if k is None else k
        if not isinstance(search_k, int) or search_k <= 0:
            return ToolResult.fail("k must be a positive integer")
        search_k = min(search_k, self._max_k)

        try:
            documents = self._vector_store.similarity_search(query, k=search_k)
        except Exception as exc:
            logger.exception("search_code: similarity_search failed for query=%r", query)
            raise ToolExecutionError(f"search_code failed: {exc}") from exc

        results: List[Dict[str, Any]] = []
        for rank, doc in enumerate(documents, start=1):
            content = getattr(doc, "page_content", None)
            if content is None:
                content = str(doc)
            metadata = getattr(doc, "metadata", None) or {}
            results.append(
                {
                    "rank": rank,
                    "content": content,
                    "source": metadata.get("source", "unknown"),
                    "metadata": metadata,
                }
            )

        return ToolResult.ok(
            results,
            query=query,
            k=search_k,
            result_count=len(results),
        )
