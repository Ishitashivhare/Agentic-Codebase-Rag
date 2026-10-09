"""
github_loader.py - clone and cache GitHub repositories for indexing.

app.py imports these four functions
(`clone_github_repo, get_repo_info, list_cached_repos, remove_github_cache`)
for its "GitHub Repository" tab. This is a minimal, `git clone`-based
implementation matching the interface app.py expects.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Matches the layout app.py's other caches use ("data/repo", "vector_index.faiss/").
_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "github_repos"

_GITHUB_URL_PATTERN = re.compile(
    r"^https?://github\.com/(?P<owner>[\w.\-]+)/(?P<name>[\w.\-]+?)(?:\.git)?/?$"
)


def get_repo_info(github_url: str) -> Dict[str, str]:
    """Parse {"owner", "name"} out of a GitHub URL.

    Raises:
        ValueError: If `github_url` doesn't look like a GitHub repo URL.
    """
    if not isinstance(github_url, str) or not github_url.strip():
        raise ValueError("github_url must be a non-empty string")

    match = _GITHUB_URL_PATTERN.match(github_url.strip())
    if not match:
        raise ValueError(f"'{github_url}' does not look like a GitHub repository URL")
    return {"owner": match.group("owner"), "name": match.group("name")}


def clone_github_repo(github_url: str, cache_root: Optional[str] = None) -> str:
    """Clone `github_url` into a per-repo cache directory, reusing an
    existing clone if one is already cached, and return the local path.

    Args:
        github_url: A GitHub repository URL, e.g. "https://github.com/owner/repo".
        cache_root: Directory clones are cached under. Defaults to
            "<project_root>/data/github_repos".

    Returns:
        The local filesystem path to the cloned repository, as a string.

    Raises:
        ValueError: If `github_url` isn't a recognizable GitHub URL.
        RuntimeError: If git isn't installed, or the clone fails (bad
            URL, network error, private repo without credentials, timeout).
    """
    info = get_repo_info(github_url)
    root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{info['owner']}__{info['name']}"

    if target.exists():
        logger.info("Reusing cached clone of '%s' at '%s'.", github_url, target)
        return str(target)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(target)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError("git clone timed out after 300 seconds") from exc

    return str(target)


def list_cached_repos(cache_root: Optional[str] = None) -> List[str]:
    """List the names of currently cached GitHub clones."""
    root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def remove_github_cache(repo_name: Optional[str] = None, cache_root: Optional[str] = None) -> None:
    """Remove one cached clone (`repo_name`), or every cached clone if
    `repo_name` is None - matching app.py's "Delete" (per-repo) and
    "Clear All Cache" (no argument) buttons."""
    root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
    if not root.exists():
        return

    if repo_name is None:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return

    target = root / repo_name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
