"""Word-boundary scan of non-Python text: Dockerfile, .env, shell, YAML, Makefile, notebook magic lines.

Only two things are looked for, both report-only (these files are never rewritten):

* ``TV5M08`` environment variables that transformers v5 no longer reads
* ``TV5M10`` the removed ``transformers-cli`` command
"""

from __future__ import annotations

import re

from . import rules as R
from .model import Finding

_ENV_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(R.REMOVED_ENV_VARS) + r")(?![A-Za-z0-9_])")
_CLI_RE = re.compile(r"(?<![\w./-])transformers-cli(?![\w-])")


def scan_text(
    text: str,
    path: str,
    select: tuple[str, ...] | None = None,
    ignore: tuple[str, ...] | None = None,
    *,
    cell: int | None = None,
    skip_comments: bool = True,
) -> list[Finding]:
    """Scan ``text`` line by line; line numbers start at 1."""
    findings: list[Finding] = []
    want_env = R.is_enabled("TV5M08", select, ignore)
    want_cli = R.is_enabled("TV5M10", select, ignore)
    for number, line in enumerate(text.split("\n"), start=1):
        if skip_comments and line.lstrip().startswith("#"):
            continue
        if want_env:
            for match in _ENV_RE.finditer(line):
                findings.append(_finding("TV5M08", path, number, match.start() + 1, cell, name=match.group(1)))
        if want_cli:
            for match in _CLI_RE.finditer(line):
                findings.append(_finding("TV5M10", path, number, match.start() + 1, cell))
    return findings


def _finding(rule: str, path: str, line: int, col: int, cell: int | None, **params: str) -> Finding:
    spec = R.RULES[rule]
    return Finding(rule, spec.severity, "manual", path, line, col, f"msg.{rule}", dict(params), cell, False)
