"""
Structural interface for the vector store that `search_code` depends on.

This is the dependency-inversion piece: SearchCodeTool depends on this
narrow Protocol, not on `langchain.vectorstores.FAISS` directly. The
object built by the existing `src/vectordb.py::create_vector_db` already
satisfies this protocol (LangChain's FAISS wrapper implements
`similarity_search`), so nothing about the existing vector database code
needs to change - it's reused as-is and simply passed in.

Depending on the Protocol instead of the concrete FAISS class also means
SearchCodeTool can be tested with a lightweight fake, and would keep
working unchanged if the underlying store were ever swapped (e.g. for
Chroma or pgvector) as long as the replacement exposes the same method.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class SimilaritySearchable(Protocol):
    """Anything with a LangChain-style `similarity_search` method.

    Returned items are expected to duck-type as LangChain `Document`
    objects (i.e. expose `.page_content` and `.metadata`), which is what
    `FAISS.from_texts(...)` in the existing `vectordb.py` produces.
    """

    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        ...
