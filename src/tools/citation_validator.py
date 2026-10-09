"""Validation helpers for source citations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.citations import Citation


def validate_citations(citations: Iterable[Citation], repo_root: str | Path, context_text: str) -> List[Citation]:
    """Keep only citations whose file, lines, and contextual evidence exist."""
    root = Path(repo_root).resolve()
    valid: List[Citation] = []
    for citation in citations:
        candidate = (root / citation.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        try:
            line_count = len(candidate.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            continue
        if citation.start_line is not None or citation.end_line is not None:
            if citation.start_line is None or citation.end_line is None:
                continue
            if citation.start_line < 1 or citation.end_line < citation.start_line or citation.end_line > line_count:
                continue
        if citation.path not in context_text and citation.file_basename not in context_text:
            continue
        valid.append(citation)
    return valid
