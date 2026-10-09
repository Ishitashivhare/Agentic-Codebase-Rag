"""
find_file(filename) - locate files in the repository by name.

Walks the repository the same way `src/loader.py` does (`os.walk`), but
matches on filename instead of collecting every `.py` file's contents -
this is a lookup tool, not an indexing step, so it stays lightweight and
does not touch the vector store.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .base_tool import BaseTool
from .exceptions import ToolExecutionError, ToolInputError
from .path_utils import DEFAULT_IGNORED_DIRS
from .tool_result import ToolResult

logger = logging.getLogger(__name__)


class FindFileTool(BaseTool):
    """Finds files whose name matches or contains a given filename.

    Single responsibility: name-based file lookup. It does not read
    file contents (`read_file`'s job) or search code semantically
    (`search_code`'s job).
    """

    name = "find_file"
    description = (
        "Find files in the repository by filename (exact or substring "
        "match, case-insensitive). Returns matching paths relative to "
        "the repository root. Use this before read_file when you know "
        "a file's name but not its exact location."
    )

    def __init__(
        self,
        repo_root: Union[str, Path],
        ignored_dirs: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Args:
            repo_root: Root directory to search within.
            ignored_dirs: Directory names to skip while walking (e.g.
                `.git`, `node_modules`). Defaults to a sensible set.

        Raises:
            ToolInputError: If `repo_root` does not exist or is not a
                directory.
        """
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ToolInputError(f"repo_root '{repo_root}' is not a valid directory")

        self._repo_root = root.resolve()
        self._ignored_dirs = set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)

    def run(self, filename: str, exact_match: bool = False) -> ToolResult:
        """
        Args:
            filename: Filename or substring to search for, e.g.
                "auth.py" or "auth".
            exact_match: If True, only files whose name equals
                `filename` (case-insensitive) match. If False (default),
                substring matches are included too.

        Returns:
            ToolResult whose `data` is a list of matching paths, each
            relative to `repo_root`, sorted for stable output.
        """
        if not isinstance(filename, str) or not filename.strip():
            return ToolResult.fail("filename must be a non-empty string")

        target = filename.strip().lower()
        matches: List[str] = []

        try:
            for current_root, dirs, files in os.walk(self._repo_root):
                dirs[:] = [d for d in dirs if d not in self._ignored_dirs]
                for file_name in files:
                    candidate = file_name.lower()
                    is_match = candidate == target if exact_match else target in candidate
                    if is_match:
                        full_path = Path(current_root) / file_name
                        matches.append(str(full_path.relative_to(self._repo_root)))
        except OSError as exc:
            logger.exception("find_file: failed to walk %s", self._repo_root)
            raise ToolExecutionError(f"find_file failed: {exc}") from exc

        matches.sort()
        return ToolResult.ok(matches, filename=filename, match_count=len(matches))
