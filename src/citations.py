"""
citations.py - the citation data model and formatting.

Kept as its own module (rather than folded into rag_pipeline.py) so the
citation format - and its correctness - doesn't get tangled up with
orchestration logic, and so anything else that surfaces retrieved code
(a future API response, a different UI) can reuse the same Citation
type and formatter instead of reinventing "how do we print a source".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Citation:
    """One place in the codebase an answer drew on.

    Attributes:
        path: File path as indexed (may be relative, e.g. "src/auth/auth_service.py").
        function_name: Enclosing function/method name, if known.
        class_name: Enclosing class name, if known.
        start_line: 1-indexed start line, if known.
        end_line: 1-indexed end line, if known.
    """

    path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    @property
    def file_basename(self) -> str:
        return Path(self.path).name

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> Optional["Citation"]:
        """Build a Citation from a search_code chunk's metadata dict.

        Returns None if there's no usable path - e.g. chunks indexed
        before citation metadata existed, or a chunk whose source
        genuinely couldn't be determined. Citing "unknown" would be
        worse than citing nothing.
        """
        path = metadata.get("source")
        if not path or path == "unknown":
            return None
        return cls(
            path=path,
            function_name=metadata.get("function_name") or None,
            class_name=metadata.get("class_name") or None,
            start_line=metadata.get("start_line") or None,
            end_line=metadata.get("end_line") or None,
        )

    def to_line(self) -> str:
        """Render as "<file>: lines <start>-<end>", with the enclosing
        class/function appended when known, e.g.:

            auth_service.py: lines 34-67
            jwt_service.py: lines 12-40 (class JWTService, function decode)
            utils.py: lines 5-9 (function helper)
        """
        if self.start_line and self.end_line:
            location = f"{self.file_basename}: lines {self.start_line}-{self.end_line}"
        else:
            location = self.file_basename

        if self.class_name and self.function_name:
            location += f" (class {self.class_name}, function {self.function_name})"
        elif self.function_name:
            location += f" (function {self.function_name})"
        elif self.class_name:
            location += f" (class {self.class_name})"

        return location

    def dedupe_key(self) -> Tuple[str, Optional[str], Optional[str], Optional[int], Optional[int]]:
        return (self.path, self.function_name, self.class_name, self.start_line, self.end_line)


def dedupe_citations(citations: List[Citation]) -> List[Citation]:
    """Remove exact duplicate citations while preserving first-seen order."""
    seen: set = set()
    result: List[Citation] = []
    for citation in citations:
        key = citation.dedupe_key()
        if key not in seen:
            seen.add(key)
            result.append(citation)
    return result


def format_citation_block(citations: List[Citation]) -> str:
    """Render the deterministic "Source:" section appended to every
    answer, e.g.:

        Source:
        auth_service.py: lines 34-67
        jwt_service.py: lines 12-40 (class JWTService, function decode)

    Returns an empty string if there are no citations - callers should
    skip appending anything in that case rather than print an empty header.
    """
    if not citations:
        return ""
    lines = "\n".join(citation.to_line() for citation in citations)
    return f"Source:\n{lines}"


def format_context_label(metadata: Dict[str, Any]) -> str:
    """Render a compact, LLM-facing label for one retrieved chunk, e.g.:

        [path: src/auth/auth_service.py | class: AuthService | function: verify_token | lines 34-67]

    This is what the model sees inline above each context block, so it
    has the file/function/class explicitly available to reference in
    its prose - the appended "Source:" block (format_citation_block) is
    the deterministic, code-generated guarantee; this label is what
    makes the model's own explanation likely to name the same things.
    """
    parts = [f"path: {metadata.get('source', 'unknown')}"]
    if metadata.get("class_name"):
        parts.append(f"class: {metadata['class_name']}")
    if metadata.get("function_name"):
        parts.append(f"function: {metadata['function_name']}")
    if metadata.get("start_line") and metadata.get("end_line"):
        parts.append(f"lines {metadata['start_line']}-{metadata['end_line']}")
    return "[" + " | ".join(parts) + "]"
