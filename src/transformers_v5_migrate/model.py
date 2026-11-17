"""Plain data classes shared by the codemod, the scanners and the report renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ParseError(Exception):
    """Raised by :func:`migrate_source` when the code cannot be parsed."""

    def __init__(self, path: str, line: int, message: str) -> None:
        super().__init__(f"{path}:{line}: {message}")
        self.path = path
        self.line = line
        self.message = message


@dataclass(frozen=True)
class Fix:
    """One automatic edit (applied only with ``--write``)."""

    rule: str
    path: str
    line: int
    col: int
    before: str
    after: str
    cell: int | None = None


@dataclass(frozen=True)
class Finding:
    """One thing a human has to look at (never edited automatically)."""

    rule: str
    severity: str  # "error" | "warning" | "note"
    kind: str  # "manual" | "behavior" | "dependency"
    path: str
    line: int
    col: int
    message_key: str
    params: dict[str, Any] = field(default_factory=dict)
    cell: int | None = None
    fixable: bool = False


@dataclass
class Result:
    """Outcome of migrating one source string or notebook."""

    code: str
    fixes: list[Fix] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class FileDiff:
    """Before/after text of a file the tool would change (or did change)."""

    path: str
    before: str
    after: str


@dataclass
class Report:
    """Outcome of scanning a set of paths."""

    files_scanned: int = 0
    files_changed: list[str] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    diffs: list[FileDiff] = field(default_factory=list)
    written: bool = False
