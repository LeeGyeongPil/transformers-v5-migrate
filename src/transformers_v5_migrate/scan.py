"""Scan files and directories: the engine behind the command line."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable

from . import rules as R
from .codemod import migrate_source
from .deps import scan_pyproject, scan_requirements
from .model import FileDiff, Finding, ParseError, Report
from .notebook import migrate_notebook
from .walker import collect

_BOM = "﻿"


def _parse_finding(path: str, line: int, message: str, uses_transformers: bool = True) -> Finding:
    # a file that never mentions transformers only gets a note: the tool has nothing to say about it
    severity = R.RULES["TV5P01"].severity if uses_transformers else "note"
    return Finding("TV5P01", severity, "manual", path, line, 1, "msg.TV5P01", {"error": message})


def write_atomic(path: str, data: bytes) -> None:
    """Replace ``path`` with ``data`` without ever leaving a half-written file."""
    tmp = os.path.join(os.path.dirname(path) or ".", f".{os.path.basename(path)}.tv5m.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _scan_read_only(kind: str, text: str, path: str, sel: tuple[str, ...] | None, ign: tuple[str, ...] | None) -> list[Finding]:
    """Kinds that are only ever reported on, never rewritten."""
    if kind == "requirements":
        return scan_requirements(text, path, sel, ign)
    return scan_pyproject(text, path, sel, ign)


def scan_paths(
    paths: Iterable[str],
    *,
    write: bool = False,
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
) -> Report:
    """Analyse every target file under ``paths``.

    With ``write=True`` the fixed ``.py`` and ``.ipynb`` files are replaced in place
    (atomically); every other file kind is read only. Raises
    :class:`transformers_v5_migrate.walker.PathError` for a path that does not exist.
    """
    sel = tuple(select) if select else None
    ign = tuple(ignore) if ignore else None
    report = Report(written=write)
    for path, kind in collect(paths, exclude):
        report.files_scanned += 1
        with open(path, "rb") as handle:
            raw = handle.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            if R.is_enabled("TV5P01", sel, ign):
                report.findings.append(_parse_finding(path, 1, f"not valid UTF-8 ({exc.reason})", b"transformers" in raw))
            continue
        bom = text.startswith(_BOM)
        if bom:
            text = text[1:]

        new_text = text
        try:
            if kind == "py":
                result = migrate_source(text, path=path, select=sel, ignore=ign)
            elif kind == "ipynb":
                result = migrate_notebook(text, path=path, select=sel, ignore=ign)
            else:
                report.findings += _scan_read_only(kind, text, path, sel, ign)
                continue
        except ParseError as exc:
            if R.is_enabled("TV5P01", sel, ign):
                report.findings.append(_parse_finding(path, exc.line, exc.message, "transformers" in text))
            continue

        report.fixes += result.fixes
        report.findings += result.findings
        new_text = result.code
        if new_text != text:
            report.files_changed.append(path)
            report.diffs.append(FileDiff(path, text, new_text))
            if write:
                write_atomic(path, ((_BOM if bom else "") + new_text).encode("utf-8"))
    return report
