"""
Shared path-safety helper for filesystem-facing tools.

`read_file`, `find_file`, and `get_repository_structure` all need to
stay confined to a repository root - this is factored out once so the
traversal-prevention logic lives (and is tested) in exactly one place
instead of being copy-pasted into every tool.
"""

from __future__ import annotations

from pathlib import Path

from .exceptions import PathSecurityError


def resolve_safe_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `repo_root` and guarantee the
    result stays inside `repo_root`.

    Args:
        repo_root: The directory tools are scoped to.
        relative_path: A path supplied by a caller (agent/user), which
            may contain `..` segments, be absolute, or otherwise attempt
            to escape `repo_root`.

    Returns:
        The resolved, absolute Path, guaranteed to be inside `repo_root`.

    Raises:
        PathSecurityError: If the resolved path falls outside
            `repo_root`, e.g. via `../../etc/passwd` or an absolute path
            pointing elsewhere on disk.
    """
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path '{relative_path}' resolves outside the repository "
            f"root ('{root}') and was rejected."
        ) from exc

    return candidate


DEFAULT_IGNORED_DIRS = frozenset(
    {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}
)
