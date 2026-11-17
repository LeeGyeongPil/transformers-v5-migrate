"""Find the files worth looking at and say what kind each one is."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from pathlib import PurePath

DEFAULT_EXCLUDES = (
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "*.egg-info",
)

_TEXT_PATTERNS = ("Dockerfile*", "*.Dockerfile", ".env", ".env.*", "*.sh", "*.bash", "Makefile")
_YAML_PATTERNS = ("*.yml", "*.yaml")
_NOT_CONFIG_JSON = ("package.json", "package-lock.json", "composer.json", "tsconfig*.json", "*.schema.json")


class PathError(Exception):
    """A path given on the command line does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"path does not exist: {path}")
        self.path = path


def classify(name: str) -> str | None:
    """Return the file kind, or None when the file is not a target.

    Kinds: ``py``, ``ipynb``, ``requirements``, ``pyproject``, ``environment`` (conda),
    ``yaml``, ``config`` (JSON training config) and ``text`` (Dockerfile, .env, shell, Makefile).
    """
    if name.endswith(".py"):
        return "py"
    if name.endswith(".ipynb"):
        return "ipynb"
    if name == "pyproject.toml":
        return "pyproject"
    if fnmatch.fnmatch(name, "requirements*.txt") or fnmatch.fnmatch(name, "constraints*.txt"):
        return "requirements"
    if fnmatch.fnmatch(name, "environment*.yml") or fnmatch.fnmatch(name, "environment*.yaml"):
        return "environment"
    if any(fnmatch.fnmatch(name, pattern) for pattern in _YAML_PATTERNS):
        return "yaml"
    if name.endswith(".json") and not any(fnmatch.fnmatch(name, pattern) for pattern in _NOT_CONFIG_JSON):
        return "config"
    if any(fnmatch.fnmatch(name, pattern) for pattern in _TEXT_PATTERNS):
        return "text"
    return None


def _excluded(rel: str, name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in patterns)


def collect(paths: Iterable[str], exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Expand files and directories into sorted ``(path, kind)`` pairs.

    Directories are walked without following symlinks, skipping :data:`DEFAULT_EXCLUDES`
    and any ``exclude`` glob (matched against the file name and the path relative to the
    directory given). A file named explicitly is always included when it is a known kind.
    """
    patterns = tuple(DEFAULT_EXCLUDES) + tuple(exclude)
    found: dict[str, str] = {}
    for given in paths:
        if os.path.isfile(given):
            kind = classify(os.path.basename(given))
            if kind:
                found[os.path.normpath(given)] = kind
            continue
        if not os.path.isdir(given):
            raise PathError(given)
        for root, dirs, files in os.walk(given):
            rel_root = os.path.relpath(root, given)
            dirs[:] = sorted(
                d for d in dirs if not _excluded(PurePath(rel_root, d).as_posix().removeprefix("./"), d, patterns)
            )
            for name in sorted(files):
                rel = PurePath(rel_root, name).as_posix().removeprefix("./")
                kind = classify(name)
                if kind and not _excluded(rel, name, patterns):
                    found[os.path.normpath(os.path.join(root, name))] = kind
    return sorted(found.items())
