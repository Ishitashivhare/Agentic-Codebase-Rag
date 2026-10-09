"""
rag_pipeline.py - agentic RAG pipeline entry point.

    Question -> Planner Agent -> Tool Selection -> Retrieval
             -> Context Aggregation -> LLM -> Answer

"Tool Selection -> Retrieval -> Context Aggregation" is delegated
entirely to `AgentOrchestrator` (src/agent/orchestrator.py), which runs
the planner once, executes its plan, evaluates whether the result
actually covers the question, and performs additional targeted
retrieval if not - bounded by `max_iterations`. This module's job is
narrower: build the tool registry, run the orchestrator, then handle
the LLM prompt/response and append the citation block.

`ask_codebase(query, db, model, url, k=...)` keeps the exact call
signature app.py uses - no change to app.py is required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from langchain_community.llms import Ollama

from src.agent import AgentOrchestrator, PlannerAgent
from src.citations import format_citation_block
from src.tools import SearchCodeTool, ToolRegistry, build_default_tool_registry
from src.tools.citation_validator import validate_citations
from src.tools.vector_store_protocol import SimilaritySearchable

logger = logging.getLogger(__name__)

# Mirrors the path app.py already computes for the "Local Repository" tab
# (os.path.join(os.path.dirname(__file__), "data", "repo") from the
# project root). Used only when the caller doesn't supply `repo_root`.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent / "data" / "repo"

_PROMPT_TEMPLATE = """You are a helpful software engineer.

Explain the code clearly. Each context block below is labeled with the
file path it came from, and the class name and/or function name when
known, e.g.:
    [path: src/auth/auth_service.py | class: AuthService | function: login | lines 16-21]

When you reference code, name the specific file, class, and function it
came from directly in your explanation - for example: "Authentication is
implemented in src/auth/auth_service.py, in the AuthService.login method."

Do not add your own "Source:" section at the end of your answer - one is
appended automatically after you respond, so adding your own would
duplicate it.

For questions about supported upload or ingestion types, distinguish
between generic file-storage helpers and actual API endpoints. Verify
the answer against upload/connect routes, extension allowlists, and the
database adapter classes. Do not infer support for a file type merely
because a helper can store its bytes; explicitly state whether raw SQL
script files are supported separately from SQLite database files or SQL
database connections.

For symbol-location questions, name the file where a class or function is
defined, not a file that merely imports it. Distinguish internal dataclass
models from Pydantic API response models when both use similar names.

Context:
{context}

Question:
{query}
"""


def ask_codebase(
    query: str,
    db: SimilaritySearchable,
    model: str = "llama3",
    url: str = "http://localhost:11434",
    k: int = 5,
    repo_root: Optional[str] = None,
    planner: Optional[PlannerAgent] = None,
    llm_client: Optional[Any] = None,
    orchestrator: Optional[AgentOrchestrator] = None,
    max_iterations: int = 4,
) -> str:
    """Answer a question about the indexed codebase using the agentic pipeline.

    Args:
        query: The user's natural-language question.
        db: An already-built vector store exposing `similarity_search`
            (e.g. the object `vectordb.create_vector_db`/`load_vector_db`
            returns). Never rebuilt or re-embedded here.
        model: Ollama model name. Unused if `llm_client` is supplied.
        url: Ollama server URL. Unused if `llm_client` is supplied.
        k: Number of chunks each semantic search step should retrieve.
            Passed through to the orchestrator, so the Streamlit sidebar
            slider keeps controlling retrieval breadth exactly as before.
        repo_root: Root directory of the indexed repository, backing the
            filesystem tools. Defaults to the same "data/repo" path
            app.py already uses. If it doesn't exist (e.g. a GitHub-
            indexed repo whose files aren't there), those tools are
            simply not registered - the pipeline still works with
            search_code alone.
        planner: Inject a PlannerAgent (e.g. with a custom classifier).
            Ignored if `orchestrator` is supplied directly.
        llm_client: Inject an object exposing `.invoke(prompt) -> str`.
            Defaults to constructing `Ollama(model=model, base_url=url)`.
        orchestrator: Inject a fully-constructed AgentOrchestrator
            (mainly for tests) - when supplied, `repo_root`/`planner`/
            `max_iterations` are not used to build a new one.
        max_iterations: Hard cap on the orchestrator's total tool calls
            for this query. Ignored if `orchestrator` is supplied.

    Returns:
        The generated answer, with a "Source:" section appended when any
        retrieved context carried file provenance.

    Raises:
        ValueError: If `query` is empty/whitespace-only, or `k` is not
            a positive integer.
        RuntimeError: If the LLM call itself fails.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    if orchestrator is not None:
        active_orchestrator = orchestrator
    else:
        root = Path(repo_root) if repo_root else _DEFAULT_REPO_ROOT
        registry = _build_registry(root, db)
        active_orchestrator = AgentOrchestrator(
            tool_registry=registry,
            planner=planner or PlannerAgent(),
            max_iterations=max_iterations,
            default_k=k,
        )

    result = active_orchestrator.run(query, k=k)

    llm = llm_client or Ollama(model=model, base_url=url)
    prompt = _PROMPT_TEMPLATE.format(context=result.context_text, query=query)

    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        logger.exception("ask_codebase: LLM generation failed for query=%r", query)
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    citations = validate_citations(result.citations, repo_root or _DEFAULT_REPO_ROOT, result.context_text)
    source_block = format_citation_block(citations)
    if source_block:
        answer = f"{answer}\n\n{source_block}"

    return answer


def _build_registry(repo_root: Path, db: SimilaritySearchable) -> ToolRegistry:
    """Build the tool registry, degrading gracefully if `repo_root`
    isn't a real directory instead of raising and breaking the whole
    pipeline over three tools that need a filesystem."""
    if repo_root.exists() and repo_root.is_dir():
        return build_default_tool_registry(repo_root=repo_root, vector_store=db)

    logger.warning(
        "repo_root '%s' is not a valid directory; registering search_code "
        "only. File-system tools (read_file, find_file, "
        "get_repository_structure) require an indexed repository on disk.",
        repo_root,
    )
    registry = ToolRegistry()
    registry.register(SearchCodeTool(vector_store=db))
    return registry
