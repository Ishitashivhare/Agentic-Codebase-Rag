"""
read_file(path) - read a single file's contents (optionally a line range)
from the indexed repository.

Reuses the same "open as UTF-8 text" convention as the existing
`src/loader.py`, but scoped to one file at a time, with path-traversal
protection and a size cap - things the bulk loader doesn't need but a
tool exposed to an agent (or indirectly to user input) does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from .base_tool import BaseTool
from .exceptions import PathSecurityError, ToolExecutionError, ToolInputError
from .path_utils import resolve_safe_path
from .tool_result import ToolResult

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """Reads one file's contents, optionally restricted to a line range.

    Single responsibility: fetch exact file contents. It deliberately
    does not search or list files - that's `find_file` and
    `get_repository_structure` respectively.
    """

    name = "read_file"
    description = (
        "Read the full contents of a single file in the repository, "
        "optionally limited to a start/end line range. Use this after "
        "search_code or find_file has identified a specific path."
    )

    def __init__(self, repo_root: Union[str, Path], max_bytes: int = 2_000_000) -> None:
        """
        Args:
            repo_root: Root directory this tool is confined to. All
                paths passed to `run` are resolved relative to this and
                validated to stay inside it.
            max_bytes: Refuse to read files larger than this, to avoid
                blowing up an LLM context window or the process memory.

        Raises:
            ToolInputError: If `repo_root` does not exist or is not a
                directory.
        """
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ToolInputError(f"repo_root '{repo_root}' is not a valid directory")
        if max_bytes <= 0:
            raise ToolInputError("max_bytes must be positive")

        self._repo_root = root.resolve()
        self._max_bytes = max_bytes

    def run(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> ToolResult:
        """
        Args:
            path: File path relative to `repo_root`.
            start_line: 1-indexed, inclusive start line. Optional.
            end_line: 1-indexed, inclusive end line. Optional.

        Returns:
            ToolResult whose `data` is the file contents (str) for the
            requested range. `metadata` includes the resolved relative
            `path`, `total_lines`, and the `returned_lines` range as a
            (start, end) tuple.
        """
        if not isinstance(path, str) or not path.strip():
            return ToolResult.fail("path must be a non-empty string")

        try:
            resolved = resolve_safe_path(self._repo_root, path)
        except PathSecurityError as exc:
            return ToolResult.fail(str(exc))

        if not resolved.exists():
            return ToolResult.fail(f"File not found: {path}")
        if not resolved.is_file():
            return ToolResult.fail(f"Path is not a file: {path}")

        try:
            size = resolved.stat().st_size
            if size > self._max_bytes:
                return ToolResult.fail(
                    f"File exceeds max readable size "
                    f"({size} bytes > {self._max_bytes} bytes): {path}"
                )
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError as exc:
            logger.exception("read_file: failed to read %s", resolved)
            raise ToolExecutionError(f"read_file failed for '{path}': {exc}") from exc

        content, returned_range = self._select_range(lines, start_line, end_line)
        if content is None:
            return ToolResult.fail("start_line must be less than end_line")

        return ToolResult.ok(
            content,
            path=str(resolved.relative_to(self._repo_root)),
            total_lines=len(lines),
            returned_lines=returned_range,
        )

    @staticmethod
    def _select_range(
        lines: list,
        start_line: Optional[int],
        end_line: Optional[int],
    ) -> Tuple[Optional[str], Tuple[int, int]]:
        """Slice `lines` to the requested 1-indexed, inclusive range.
        Returns (None, ...) if the requested range is empty/invalid."""
        if start_line is None and end_line is None:
            return "".join(lines), (1, len(lines))

        start_idx = max((start_line or 1) - 1, 0)
        end_idx = min(end_line, len(lines)) if end_line is not None else len(lines)

        if start_idx >= end_idx:
            return None, (0, 0)

        selected = lines[start_idx:end_idx]
        return "".join(selected), (start_idx + 1, end_idx)
