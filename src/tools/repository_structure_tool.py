"""
get_repository_structure() - hierarchical directory/file view of the
repository, for "repository understanding" style questions.

Also built on `os.walk`/`Path.iterdir`, matching the traversal
conventions used elsewhere in the codebase (`loader.py`, `find_file`),
so ignored-directory handling stays consistent across tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from .base_tool import BaseTool
from .exceptions import ToolExecutionError, ToolInputError
from .path_utils import DEFAULT_IGNORED_DIRS
from .tool_result import ToolResult

logger = logging.getLogger(__name__)


class GetRepositoryStructureTool(BaseTool):
    """Builds a nested directory/file tree for the repository.

    Single responsibility: describe the repo's *shape*, not its
    contents. Pairs with `find_file` (locate) and `read_file` (fetch)
    for repository-understanding questions like "what's the top-level
    layout of this project?".
    """

    name = "get_repository_structure"
    description = (
        "Return a hierarchical tree of the repository's directories and "
        "files, up to a configurable depth. Use this to understand "
        "overall project layout before deciding which files to read or "
        "search."
    )

    def __init__(
        self,
        repo_root: Union[str, Path],
        ignored_dirs: Optional[Iterable[str]] = None,
        max_depth: int = 6,
    ) -> None:
        """
        Args:
            repo_root: Root directory to describe.
            ignored_dirs: Directory names to exclude (e.g. `.git`).
                Defaults to a sensible set shared with `find_file`.
            max_depth: Default recursion depth cap, overridable per call.

        Raises:
            ToolInputError: If `repo_root` does not exist, is not a
                directory, or `max_depth` is negative.
        """
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ToolInputError(f"repo_root '{repo_root}' is not a valid directory")
        if max_depth < 0:
            raise ToolInputError("max_depth must be non-negative")

        self._repo_root = root.resolve()
        self._ignored_dirs = set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)
        self._default_max_depth = max_depth

    def run(self, max_depth: Optional[int] = None) -> ToolResult:
        """
        Args:
            max_depth: Overrides the instance's default recursion depth
                for this call only.

        Returns:
            ToolResult whose `data` is a nested dict:
                {"name": str, "type": "directory"|"file", "children": [...]}
            (files have no "children" key).
        """
        depth_limit = self._default_max_depth if max_depth is None else max_depth
        if not isinstance(depth_limit, int) or depth_limit < 0:
            return ToolResult.fail("max_depth must be a non-negative integer")

        try:
            tree = self._build_tree(self._repo_root, depth_limit)
        except OSError as exc:
            logger.exception("get_repository_structure: failed to walk %s", self._repo_root)
            raise ToolExecutionError(f"get_repository_structure failed: {exc}") from exc

        return ToolResult.ok(tree, root=str(self._repo_root), max_depth=depth_limit)

    def _build_tree(self, path: Path, depth_remaining: int) -> Dict[str, Any]:
        """Recursively build a nested dict describing `path`."""
        node: Dict[str, Any] = {"name": path.name or str(path), "type": "directory", "children": []}

        if depth_remaining < 0:
            return node

        try:
            entries = sorted(path.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
        except PermissionError:
            logger.warning("get_repository_structure: permission denied for %s", path)
            return node

        for entry in entries:
            if entry.is_dir():
                if entry.name in self._ignored_dirs:
                    continue
                node["children"].append(self._build_tree(entry, depth_remaining - 1))
            else:
                node["children"].append({"name": entry.name, "type": "file"})

        return node
