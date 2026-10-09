"""
loader.py - load a repository's Python files into path-tagged documents.

Changed from the original `load_repo`, which returned `List[str]` (raw
file contents, path discarded): it now returns `List[CodeDocument]`
(path + content), because the chunker needs the source path to attach
citation metadata to every chunk. This is the only place upstream of
the vector store that ever sees a file's actual path, so it's the only
place a citation's file path can originate from.

app.py is unaffected by this change - it only ever does
`len(files)` and passes `files` straight into `chunk_code(files)`,
never inspecting the list's element type.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, TypedDict

from src.tools.path_utils import DEFAULT_IGNORED_DIRS

logger = logging.getLogger(__name__)


class CodeDocument(TypedDict):
    """One loaded source file."""

    path: str  # Relative to the repo root, e.g. "src/auth/auth_service.py"
    content: str  # Raw file text


def load_repo(repo_path: str) -> List[CodeDocument]:
    """Walk `repo_path` and load every `.py` file's path and contents.

    Args:
        repo_path: Root directory to walk.

    Returns:
        A list of CodeDocument dicts. Files that can't be decoded as
        UTF-8, or can't be read for any other reason, are skipped (and
        logged) rather than aborting the whole load - one bad file
        shouldn't prevent indexing the rest of the repository.
    """
    repo_root = Path(repo_path)
    code_files: List[CodeDocument] = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORED_DIRS]

        for file_name in files:
            if not file_name.endswith(".py"):
                continue

            full_path = Path(root) / file_name
            try:
                content = full_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Skipping unreadable file '%s': %s", full_path, exc)
                continue

            relative_path = full_path.relative_to(repo_root)
            code_files.append({"path": str(relative_path), "content": content})

    return code_files
