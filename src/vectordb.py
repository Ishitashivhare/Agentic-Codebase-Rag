"""
vectordb.py - build/load the FAISS vector store, now metadata-aware.

`create_vector_db` now accepts `List[CodeChunk]` (from chunker.py)
instead of `List[str]`, and passes each chunk's metadata through to
`FAISS.from_texts` so `search_code` can return file path / function
name / class name / line range for every result. This is the piece that
was previously missing entirely - the original called
`FAISS.from_texts(chunks, embeddings)` with no `metadatas`, so no
citation was ever possible.

`save_local`/`load_local`-backed caching and `load_vector_db` are
implemented here too. app.py already called `load_vector_db(embeddings)`
and `create_vector_db(chunks, embeddings, use_cache=False)` - neither
existed in the original file, so those call sites would have failed at
runtime. Implementing them isn't scope creep for this citation feature;
`create_vector_db`'s signature had to change anyway (chunks are now
`CodeChunk` objects, not strings), so this closes a real, adjacent gap
at essentially no extra cost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_community.vectorstores import FAISS

from src.chunker import CodeChunk

logger = logging.getLogger(__name__)

# Matches the path the README documents ("vector_index.faiss/" at the
# project root) and what app.py's cache-management tab already assumes exists.
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "vector_index.faiss"


def create_vector_db(
    chunks: List[CodeChunk],
    embeddings: Any,
    use_cache: bool = True,
    index_dir: Optional[str] = None,
) -> FAISS:
    """Build a FAISS index from CodeChunks, carrying each chunk's
    citation metadata (source, function_name, class_name, start_line,
    end_line) into the index.

    Args:
        chunks: Output of `chunker.chunk_code()`.
        embeddings: An embeddings model, e.g. from `embeddings.load_embeddings()`.
        use_cache: If True (default), persist the built index via
            `FAISS.save_local` to `index_dir` so a later `load_vector_db()`
            call can reuse it without re-embedding. app.py's GitHub tab
            already calls this with `use_cache=False` for one-off,
            non-persisted indexing.
        index_dir: Directory to persist the index to/load it from.
            Defaults to "<project_root>/vector_index.faiss".

    Returns:
        The built FAISS vector store.

    Raises:
        ValueError: If `chunks` is empty - `FAISS.from_texts` fails on
            an empty list with a confusing internal error, so this
            fails fast with a clear message instead.
    """
    if not chunks:
        raise ValueError("create_vector_db received no chunks to index")

    texts = [chunk.content for chunk in chunks]
    metadatas = [chunk.to_metadata() for chunk in chunks]

    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    if use_cache:
        _save_index(db, index_dir)

    return db


def load_vector_db(embeddings: Any, index_dir: Optional[str] = None) -> Optional[FAISS]:
    """Load a previously cached FAISS index.

    Args:
        embeddings: The same embeddings model used to build the index -
            required by FAISS to deserialize correctly.
        index_dir: Directory to load from. Defaults to
            "<project_root>/vector_index.faiss".

    Returns:
        The loaded FAISS store, or None if no cached index exists yet
        (a cold start, not an error condition) or if loading fails for
        any reason (logged, not raised, so a corrupted cache degrades to
        "please re-index" rather than crashing the app).
    """
    target_dir = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
    if not target_dir.exists():
        return None

    try:
        return FAISS.load_local(str(target_dir), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        logger.exception("Failed to load cached vector index from '%s'.", target_dir)
        return None


def _save_index(db: FAISS, index_dir: Optional[str]) -> None:
    target_dir = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(target_dir))
    except OSError:
        logger.exception(
            "Failed to persist vector index to '%s'; continuing with the "
            "in-memory index for this session only.",
            target_dir,
        )
