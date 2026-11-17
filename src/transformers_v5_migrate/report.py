"""Render a :class:`Report` as text or JSON (and the rule table)."""

from __future__ import annotations

import dataclasses
import difflib
import json
import unicodedata
from typing import Any

from . import __version__
from . import rules as R
from .i18n import t
from .model import Finding, Fix, Report

_ANSI = {"bold": "1", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36", "dim": "2"}
_SEVERITY_COLOR = {"error": "red", "warning": "yellow", "note": "blue"}


def summary(report: Report) -> dict[str, int]:
    kinds = [f.kind for f in report.findings]
    return {
        "fixable": len(report.fixes),
        "manual": kinds.count("manual"),
        "behavior": kinds.count("behavior"),
        "dependency": kinds.count("dependency"),
        "files_scanned": report.files_scanned,
        "files_changed": len(report.files_changed),
    }


def render_message(finding: Finding, lang: str) -> str:
    params = dict(finding.params)
    if "hint" in params:
        params["hint"] = t(f"hint.{params['hint']}", lang, **params)
    return t(finding.message_key, lang, **params)


def _location(item: Finding | Fix, lang: str) -> str:
    if item.cell is not None:
        return f"{item.path}:{t('label.cell', lang, cell=item.cell)}:{item.line}:{item.col}"
    return f"{item.path}:{item.line}:{item.col}"


def _unified_diff(path: str, before: str, after: str) -> list[str]:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=path, tofile=f"{path} (v5)"
    )
    return [line if line.endswith("\n") else line + "\n" for line in lines]


def render_text(report: Report, lang: str, *, diff: bool = True, color: bool = False) -> str:
    def paint(text: str, *styles: str) -> str:
        if not color:
            return text
        return "".join(f"\x1b[{_ANSI[s]}m" for s in styles) + text + "\x1b[0m"

    out: list[str] = []
    if report.fixes:
        out.append(paint("== " + t("sec.fixes_written" if report.written else "sec.fixes_dry", lang) + " ==", "bold"))
        for fix in report.fixes:
            rule = paint(fix.rule, "bold", _SEVERITY_COLOR[R.RULES[fix.rule].severity])
            out.append(f"{_location(fix, lang)}: {rule} {fix.before} -> {fix.after}")
        out.append("")
    if diff and report.diffs:
        out.append(paint("== " + t("sec.diff_written" if report.written else "sec.diff_dry", lang) + " ==", "bold"))
        for file_diff in report.diffs:
            for line in _unified_diff(file_diff.path, file_diff.before, file_diff.after):
                text = line.rstrip("\n")
                if line.startswith(("+++", "---")):
                    text = paint(text, "bold")
                elif line.startswith("@@"):
                    text = paint(text, "cyan")
                elif line.startswith("+"):
                    text = paint(text, "green")
                elif line.startswith("-"):
                    text = paint(text, "red")
                out.append(text)
        out.append("")
    for kind, section in (("manual", "sec.manual"), ("behavior", "sec.behavior"), ("dependency", "sec.dependency")):
        items = [f for f in report.findings if f.kind == kind]
        if not items:
            continue
        out.append(paint("== " + t(section, lang) + " ==", "bold"))
        for finding in items:
            rule = paint(f"{finding.rule} [{t('severity.' + finding.severity, lang)}]", "bold", _SEVERITY_COLOR[finding.severity])
            out.append(f"{_location(finding, lang)}: {rule} {render_message(finding, lang)}")
            out.append(paint(f"    {t('label.fix', lang)}: {t('fix.' + finding.rule, lang)}", "dim"))
        out.append("")
    counts = summary(report)
    key = ("scan.written" if report.written else "scan.dry") + ("_one" if counts["files_scanned"] == 1 else "")
    out.append(t(key, lang, scanned=counts["files_scanned"], changed=counts["files_changed"]))
    out.append(t("summary.written" if report.written else "summary.dry", lang, **counts))
    return "\n".join(out) + "\n"


def _fix_dict(fix: Fix, lang: str) -> dict[str, Any]:
    data = dataclasses.asdict(fix)
    data["message"] = t(f"msg.{fix.rule}", lang)
    return data


def _finding_dict(finding: Finding, lang: str) -> dict[str, Any]:
    data = dataclasses.asdict(finding)
    del data["message_key"]
    data["message"] = render_message(finding, lang)
    data["recommendation"] = t(f"fix.{finding.rule}", lang)
    return data


def render_json(report: Report, lang: str) -> str:
    payload = {
        "tool": "transformers-v5-migrate",
        "version": __version__,
        "target": R.TARGET_VERSION,
        "lang": lang,
        "written": report.written,
        "summary": summary(report),
        "fixes": [_fix_dict(f, lang) for f in report.fixes],
        "findings": [_finding_dict(f, lang) for f in report.findings],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_rules(lang: str) -> str:
    headers = [t(f"list.col.{c}", lang) for c in ("id", "kind", "severity", "verified", "source", "title")]
    rows = [
        [r.id, t(f"kind.{r.kind}", lang), t(f"severity.{r.severity}", lang), r.verified_in, t(f"source.{r.source}", lang), t(f"title.{r.id}", lang)]
        for r in R.RULES.values()
    ]
    widths = [max(_width(row[i]) for row in [headers, *rows]) for i in range(5)]

    def line(row: list[str]) -> str:
        cells = [row[i] + " " * (widths[i] - _width(row[i])) for i in range(5)]
        return "  ".join(cells + [row[5]])

    return "\n".join([t("list.title", lang, version=R.TARGET_VERSION), "", line(headers), *(line(r) for r in rows)]) + "\n"


def _width(text: str) -> int:
    """Display width: Hangul and other wide characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
