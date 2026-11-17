"""Jupyter notebooks: fix code cells, keep everything else byte for byte.

IPython magics (``!pip install ...``, ``%time ...``, ``%%bash``) are not Python and
make LibCST fail. Each magic line is swapped for a same-indentation placeholder
statement before parsing and swapped back afterwards, so the tree is parseable and
the magic text survives untouched. All code cells are analysed as one module (an
``import`` in cell 1 is visible in cell 7); cell boundaries are marker comments.

Cells are numbered from 1 over *all* cells (markdown included), like the order on screen.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import re
from collections.abc import Iterable
from typing import Any

import libcst as cst

from . import rules as R
from .codemod import analyze_module
from .model import Finding, ParseError, Result
from .textscan import scan_text

_LINES = re.compile(r"[^\n]*\n|[^\n]+")
_MAGIC_ASSIGN = re.compile(r"^[\w.\[\]]+\s*=\s*[!%]")
_HELP_LINE = re.compile(r"^(\?{1,2}[\w.]+|[\w.]+\?{1,2})$")
_MAGIC_MARK = re.compile(r"^[ \t]*(?:pass[ \t]+)?# __tv5m_magic_(\d+)__", re.MULTILINE)
_CELL_MARK = re.compile(r"^# __tv5m_cell_(\d+)__\n", re.MULTILINE)


def _is_magic(stripped: str) -> bool:
    return stripped.startswith(("!", "%")) or bool(_MAGIC_ASSIGN.match(stripped) or _HELP_LINE.match(stripped))


def _mask(text: str) -> tuple[str, dict[int, str], list[tuple[int, str]]]:
    """Return (masked text, marker id -> original line, [(line number, original line)])."""
    out: list[str] = []
    originals: dict[int, str] = {}
    magic_lines: list[tuple[int, str]] = []
    lines = _LINES.findall(text)
    i = 0
    while i < len(lines):
        raw = lines[i]
        body = raw.rstrip("\r\n")
        stripped = body.lstrip()
        if not _is_magic(stripped):
            out.append(raw)
            i += 1
            continue
        indent = body[: len(body) - len(stripped)]
        first = True
        while True:
            marker = len(originals)
            originals[marker] = body
            magic_lines.append((i + 1, body))
            placeholder = f"{indent}pass  # __tv5m_magic_{marker}__" if first else f"{indent}# __tv5m_magic_{marker}__"
            out.append(placeholder + raw[len(body) :])
            first = False
            if body.rstrip().endswith("\\") and i + 1 < len(lines):  # line continuation of a shell command
                i += 1
                raw = lines[i]
                body = raw.rstrip("\r\n")
                continue
            break
        i += 1
    return "".join(out), originals, magic_lines


def _unmask(text: str, originals: dict[int, str]) -> str:
    return _MAGIC_MARK.sub(lambda m: originals[int(m.group(1))], text)


def _cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = _LINES.findall(text) if isinstance(cell.get("source"), list) else text


def _dump_like(original: str, nb: dict[str, Any]) -> str:
    """Serialize ``nb`` with the indentation, escaping and final newline of ``original``."""
    head = re.match(r"\A\{[ \t]*\n([ \t]+)\"", original)
    if head:
        kwargs: dict[str, Any] = {"indent": head.group(1)}
    elif "\n" not in original.strip():
        kwargs = {"indent": None, "separators": (",", ":")}
    else:
        kwargs = {"indent": 1}
    text = json.dumps(nb, ensure_ascii="\\u" in original, **kwargs)
    return text + "\n" if original.endswith("\n") else text


def migrate_notebook(
    nb_json: str,
    *,
    path: str = "<string>",
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
) -> Result:
    """Migrate a notebook given as JSON text; ``Result.code`` is the new JSON text.

    Cells that cannot be parsed are skipped and reported as ``TV5P01``.
    Raises :class:`ParseError` when the file is not a notebook.
    """
    sel = tuple(select) if select else None
    ign = tuple(ignore) if ignore else None
    try:
        nb = json.loads(nb_json)
        cells = nb["cells"]
        if not isinstance(cells, list):
            raise TypeError("'cells' is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        raise ParseError(path, 1, f"not a valid nbformat 4 notebook ({exc})") from exc

    findings: list[Finding] = []
    pieces: list[str] = []  # marker + masked cell, in order
    starts: list[int] = []  # first line (in the combined module) of each piece's code
    cell_ids: list[int] = []
    info: dict[int, tuple[str, dict[int, str], bool]] = {}  # idx -> (original text, originals, added_newline)
    line_no = 1
    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        text = _cell_text(cell)
        if not text.strip():
            continue
        if text.lstrip().startswith("%%"):  # cell magic (%%bash, %%writefile, ...): not Python
            findings += scan_text(text, path, sel, ign, cell=idx)
            continue
        masked, originals, magic_lines = _mask(text)
        for number, body in magic_lines:
            findings += [
                dataclasses.replace(f, line=number)
                for f in scan_text(body, path, sel, ign, cell=idx, skip_comments=False)
            ]
        try:
            cst.parse_module(masked)
        except cst.ParserSyntaxError as exc:
            if R.is_enabled("TV5P01", sel, ign):
                findings.append(
                    Finding("TV5P01", R.RULES["TV5P01"].severity, "manual", path, exc.raw_line, exc.raw_column + 1,
                            "msg.TV5P01", {"error": exc.message}, idx, False)
                )
            continue
        added_nl = not masked.endswith("\n")
        piece = f"# __tv5m_cell_{idx}__\n{masked}{chr(10) if added_nl else ''}"
        pieces.append(piece)
        starts.append(line_no + 1)
        cell_ids.append(idx)
        info[idx] = (text, originals, added_nl)
        line_no += piece.count("\n")

    combined = "".join(pieces)
    result = Result(code=nb_json, findings=findings)
    if not combined:
        result.findings.sort(key=lambda f: (f.cell or 0, f.line, f.col, f.rule))
        return result
    module = cst.parse_module(combined)
    inner = analyze_module(module, combined, path, sel, ign)

    def locate(line: int) -> tuple[int, int]:
        pos = bisect.bisect_right(starts, line) - 1
        return cell_ids[pos], line - starts[pos] + 1

    for fix in inner.fixes:
        cell_id, local = locate(fix.line)
        result.fixes.append(dataclasses.replace(fix, line=local, cell=cell_id))
    for finding in inner.findings:
        cell_id, local = locate(finding.line)
        result.findings.append(dataclasses.replace(finding, line=local, cell=cell_id))
    result.findings.sort(key=lambda f: (f.cell or 0, f.line, f.col, f.rule))

    if inner.fixes:
        parts = _CELL_MARK.split(inner.code)  # ['', '3', code3, '5', code5, ...]
        changed = False
        for cell_id, new_masked in zip(parts[1::2], parts[2::2]):
            idx = int(cell_id)
            original, originals, added_nl = info[idx]
            if added_nl and new_masked.endswith("\n"):
                new_masked = new_masked[:-1]
            new_text = _unmask(new_masked, originals)
            if new_text != original:
                _set_source(cells[idx - 1], new_text)
                changed = True
        if changed:
            result.code = _dump_like(nb_json, nb)
    return result
