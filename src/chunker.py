"""
chunker.py - AST-aware code chunking with citation metadata.

Replaces the original blind character-based splitting
(`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` over
raw file text) with chunking along actual function/method/class
boundaries wherever a file parses as valid Python. Every chunk now
carries exactly the metadata source citations need: file path, function
name, class name, and the line range in the original file.

Files that don't parse (syntax errors, unusual syntax) fall back to the
original character-based splitter, still tagged with path and a computed
line range - just without a function/class name, since none could be
determined. This fallback preserves the original chunker's behavior as
a safety net rather than dropping unparseable files from the index.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.loader import CodeDocument

logger = logging.getLogger(__name__)

_FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Soft cap used only for logging a heads-up; oversized functions are
# still kept as a single chunk rather than mid-function split, since a
# split function loses exactly the boundary information this rewrite
# exists to preserve.
_LARGE_CHUNK_WARNING_CHARS = 1500


@dataclass(frozen=True)
class CodeChunk:
    """One retrievable unit of code plus its citation metadata."""

    content: str
    path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    start_line: int = 1
    end_line: int = 1

    def to_metadata(self) -> Dict[str, Any]:
        """Metadata dict for `FAISS.from_texts(..., metadatas=...)`.
        Keys match what search_code_tool.py returns unchanged (it
        already passes through whatever metadata FAISS carries) and
        what citations.py reads back out via `Citation.from_metadata`."""
        return {
            "source": self.path,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


def chunk_code(code_docs: List[CodeDocument]) -> List[CodeChunk]:
    """Chunk every document, returning a flat list of CodeChunk.

    NOTE: return type changed from `List[str]` to `List[CodeChunk]`.
    `vectordb.create_vector_db` was updated to accept this type; app.py
    needs no change since it only ever does `len(chunks)` and passes
    `chunks` straight through to `create_vector_db`.
    """
    all_chunks: List[CodeChunk] = []
    for doc in code_docs:
        try:
            all_chunks.extend(_chunk_document(doc))
        except Exception:
            logger.exception("Failed to chunk '%s'; skipping this file.", doc.get("path", "?"))
    return all_chunks


def _chunk_document(doc: CodeDocument) -> List[CodeChunk]:
    path = doc["path"]
    content = doc["content"]

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug(
            "'%s' is not valid Python (syntax error or unsupported syntax); "
            "falling back to text splitting.",
            path,
        )
        return _chunk_by_text(path, content)

    lines = content.splitlines()
    chunks = _walk_module(tree, path, lines)

    if not chunks:
        # Valid Python with nothing at function/class granularity (e.g. a
        # flat script of top-level statements, or an empty __init__.py) -
        # fall back to text splitting so the file is still searchable.
        return _chunk_by_text(path, content)

    return chunks


def _walk_module(tree: ast.Module, path: str, lines: List[str]) -> List[CodeChunk]:
    """Chunk at module scope: every top-level function becomes a chunk;
    every top-level class's methods each become their own chunk (falling
    back to one chunk for the whole class if it has no methods).

    Nested functions (a function defined inside another function) are
    intentionally NOT split into their own chunks - they stay part of
    their enclosing function's chunk, which is the more useful retrieval
    and citation unit; citing an inner helper alone, without the outer
    function that calls it, is rarely what's wanted.
    """
    chunks: List[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_node_to_chunk(node, path, lines, function_name=node.name, class_name=None))

        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                for method in methods:
                    chunks.append(
                        _node_to_chunk(method, path, lines, function_name=method.name, class_name=node.name)
                    )
            else:
                # A class with no methods (e.g. a dataclass or constants
                # holder) - chunk its whole body as one unit.
                chunks.append(_node_to_chunk(node, path, lines, function_name=None, class_name=node.name))

        # Other top-level statements (imports, module-level constants,
        # `if __name__ == "__main__":`) are intentionally not chunked
        # individually - low retrieval value on their own, and any file
        # made up ENTIRELY of such statements falls back to
        # `_chunk_by_text` via the empty-chunks check in `_chunk_document`.

    return chunks


def _node_to_chunk(
    node: ast.AST,
    path: str,
    lines: List[str],
    function_name: Optional[str],
    class_name: Optional[str],
) -> CodeChunk:
    start_line: int = node.lineno  # type: ignore[attr-defined]
    end_line: int = getattr(node, "end_lineno", start_line) or start_line
    segment = "\n".join(lines[start_line - 1 : end_line])

    if class_name:
        # Prepend the enclosing class as a synthetic comment line so
        # it's part of the embedded/searchable text - a method chunk's
        # own body (e.g. "def get_user(self, uid): return uid") often
        # never repeats its class name anywhere in its source lines, so
        # without this a query like "what does UserService do" has
        # nothing to match against in the chunk CONTENT, only in
        # metadata. This line is synthetic and not counted in
        # start_line/end_line - citations still point at the method's
        # real location in the file.
        segment = f"# class {class_name}\n{segment}"

    if len(segment) > _LARGE_CHUNK_WARNING_CHARS:
        label = f"{class_name}.{function_name}" if class_name and function_name else (function_name or class_name or "?")
        logger.debug(
            "'%s' in '%s' is %d chars (over the %d soft cap); kept as one "
            "chunk to preserve its function/class boundary rather than "
            "mid-split it.",
            label,
            path,
            len(segment),
            _LARGE_CHUNK_WARNING_CHARS,
        )

    return CodeChunk(
        content=segment,
        path=path,
        function_name=function_name,
        class_name=class_name,
        start_line=start_line,
        end_line=end_line,
    )


def _chunk_by_text(path: str, content: str) -> List[CodeChunk]:
    """Fallback for files that don't parse as Python, or that parse but
    have nothing at function/class granularity: the original character-
    based splitting, now tagged with path and a computed line range
    (never a function/class name, since none could be determined)."""
    pieces = _FALLBACK_SPLITTER.split_text(content)
    chunks: List[CodeChunk] = []
    search_from = 0

    for piece in pieces:
        offset = content.find(piece, search_from)
        if offset == -1:
            # Overlap between pieces can make a forward-only search miss;
            # fall back to a full search rather than mis-locating this
            # piece's line range.
            offset = content.find(piece)
        if offset == -1:
            # Content was rewritten by the splitter (shouldn't happen for
            # a pure character splitter, but stay defensive) - use the
            # previous line as a best-effort estimate rather than crash.
            offset = search_from

        start_line = content.count("\n", 0, offset) + 1
        end_line = start_line + piece.count("\n")
        chunks.append(
            CodeChunk(
                content=piece,
                path=path,
                function_name=None,
                class_name=None,
                start_line=start_line,
                end_line=end_line,
            )
        )
        search_from = offset + 1

    return chunks
