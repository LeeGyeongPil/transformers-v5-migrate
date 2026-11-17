"""Dependency files: does the pinned transformers (or huggingface-hub) allow v5?

* ``TV5D01`` the transformers specifier excludes 5.23.0 (``<5``, ``==4.57.1``, ``~=4.40``)
* ``TV5D02`` the huggingface-hub specifier cannot be satisfied together with what
  transformers 5.23.0 requires (``huggingface-hub>=1.31.0,<3.0``)

Dependency files are only ever reported on, never rewritten.
"""

from __future__ import annotations

import re
import tomllib
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from . import rules as R
from .model import Finding

_TARGET = Version(R.TARGET_VERSION)
_HUB = SpecifierSet(R.HUB_REQUIREMENT.split("huggingface-hub", 1)[1])


def _bump(version: Version) -> Version:
    parts = list(version.release) + [0, 0]
    parts[len(version.release) - 1] += 1
    return Version(".".join(map(str, parts[: len(version.release)])))


def _overlaps(spec: SpecifierSet, other: SpecifierSet) -> bool:
    """Can some version satisfy both? Checks the spec's own boundary versions plus a coarse grid."""
    candidates = {Version(v) for v in ("1.31.0", "1.31.1", "1.32.0", "1.99.0", "2.0.0", "2.9.0", "2.99.0")}
    for s in spec:
        base = Version(s.version.replace(".*", ""))
        candidates.update({base, _bump(base)})
    both = spec & other
    return any(both.contains(c, prereleases=True) for c in candidates)


def _check(name: str, spec_text: str, path: str, line: int, col: int) -> list[Finding]:
    try:
        spec = SpecifierSet(spec_text)
    except InvalidSpecifier:
        return []
    if not str(spec):
        return []
    canonical = canonicalize_name(name)
    if canonical == "transformers" and not spec.contains(_TARGET, prereleases=True):
        return [_finding("TV5D01", path, line, col, spec=f"{name}{spec}")]
    if canonical == "huggingface-hub" and not _overlaps(spec, _HUB):
        return [_finding("TV5D02", path, line, col, spec=f"{name}{spec}", required=R.HUB_REQUIREMENT)]
    return []


def _finding(rule: str, path: str, line: int, col: int, **params: str) -> Finding:
    spec = R.RULES[rule]
    return Finding(rule, spec.severity, "dependency", path, line, col, f"msg.{rule}", dict(params), None, False)


def scan_requirements(text: str, path: str, select: tuple[str, ...] | None, ignore: tuple[str, ...] | None) -> list[Finding]:
    """``requirements*.txt`` / ``constraints*.txt``."""
    findings: list[Finding] = []
    for number, raw in enumerate(text.split("\n"), start=1):
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        col = raw.find(req.name) + 1 or 1
        for f in _check(req.name, str(req.specifier), path, number, col):
            if R.is_enabled(f.rule, select, ignore):
                findings.append(f)
    return findings


def _poetry_to_pep440(spec: str) -> str | None:
    """Translate the common Poetry constraint forms; return None when unsure."""
    spec = spec.strip()
    if spec in ("", "*"):
        return None
    parts = [p.strip() for p in spec.split(",")]
    out: list[str] = []
    for p in parts:
        if p.startswith("^") or p.startswith("~") and not p.startswith("~="):
            nums = p[1:].split(".")
            if not all(n.isdigit() for n in nums):
                return None
            version = ".".join(nums)
            if p.startswith("^"):
                idx = next((i for i, n in enumerate(nums) if n != "0"), len(nums) - 1)
            else:
                idx = 0 if len(nums) == 1 else 1
            upper = nums[: idx + 1]
            upper[idx] = str(int(upper[idx]) + 1)
            out += [f">={version}", "<" + ".".join(upper)]
        elif re.match(r"^\d", p):
            out.append(f"=={p}")
        else:
            out.append(p)
    return ",".join(out)


def _locate(lines: list[str], name_pattern: str) -> list[tuple[int, int]]:
    """(line, column) of every place that looks like a dependency on ``name_pattern``."""
    found = []
    pattern = re.compile(rf"""(^|["'\s\[,]){name_pattern}(?=[\s\[<>=!~^"'=;,]|$)""", re.IGNORECASE)
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        match = pattern.search(line)
        if match:
            found.append((number, match.start() + len(match.group(1)) + 1))
    return found


def scan_pyproject(text: str, path: str, select: tuple[str, ...] | None, ignore: tuple[str, ...] | None) -> list[Finding]:
    """``pyproject.toml``: project, optional-dependencies, dependency-groups and Poetry tables."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    entries: list[tuple[str, str]] = []  # (name, specifier text)

    def add_pep508(items: Any) -> None:
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                try:
                    req = Requirement(item)
                except InvalidRequirement:
                    continue
                entries.append((req.name, str(req.specifier)))

    project = data.get("project", {})
    add_pep508(project.get("dependencies"))
    for group in project.get("optional-dependencies", {}).values():
        add_pep508(group)
    for group in data.get("dependency-groups", {}).values():
        add_pep508(group)

    poetry = data.get("tool", {}).get("poetry", {})
    tables = [poetry.get("dependencies", {}), poetry.get("dev-dependencies", {})]
    tables += [g.get("dependencies", {}) for g in poetry.get("group", {}).values() if isinstance(g, dict)]
    for table in tables:
        for name, value in table.items():
            raw = value.get("version") if isinstance(value, dict) else value
            if isinstance(raw, str) and (spec := _poetry_to_pep440(raw)) is not None:
                entries.append((name, spec))

    lines = text.split("\n")
    where = {
        "transformers": _locate(lines, r"transformers"),
        "huggingface-hub": _locate(lines, r"huggingface[-_.]hub"),
    }
    seen = {"transformers": 0, "huggingface-hub": 0}
    findings: list[Finding] = []
    for name, spec_text in entries:
        canonical = canonicalize_name(name)
        if canonical not in where:
            continue
        spots = where[canonical]
        index = seen[canonical]
        seen[canonical] += 1
        line, col = spots[index] if index < len(spots) else (1, 1)
        for f in _check(name, spec_text, path, line, col):
            if R.is_enabled(f.rule, select, ignore):
                findings.append(f)
    return findings
