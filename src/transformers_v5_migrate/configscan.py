"""JSON and YAML training configs (``HfArgumentParser`` input): removed or renamed argument keys.

Report-only (``TV5M14``). Keys are found line by line, so no YAML parser is needed;
JSON files must at least parse. ``torch_dtype`` is deliberately not looked for here:
a model's own ``config.json`` has that key and v5 still reads those files.
"""

from __future__ import annotations

import json
import re

from . import rules as R
from .model import Finding

_KEY = re.compile(r"""^\s*(?:-\s+)?["']?([A-Za-z_]\w*)["']?\s*:""")
MAX_BYTES = 2_000_000  # data dumps are not training configs


def _hint(key: str) -> dict[str, str]:
    if key in R.ARGS_RENAMES:
        return {"hint": "renamed", "new": R.ARGS_RENAMES[key]}
    if key == "use_auth_token":
        return {"hint": "renamed", "new": "token"}
    if key == "warmup_ratio":
        return {"hint": "warmup_ratio_config"}
    if key == "include_inputs_for_metrics":
        return {"hint": "include_inputs_for_metrics"}
    return {"hint": R.ARGS_REMOVED[key]}


def scan_config(
    text: str,
    path: str,
    kind: str,
    select: tuple[str, ...] | None = None,
    ignore: tuple[str, ...] | None = None,
) -> list[Finding]:
    """``kind`` is ``"config"`` (JSON) or ``"yaml"``."""
    if not R.is_enabled("TV5M14", select, ignore):
        return []
    if kind == "config":
        try:
            if not isinstance(json.loads(text), (dict, list)):
                return []
        except ValueError:
            return []
    found: list[tuple[int, str, int]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if line.lstrip().startswith("#"):
            continue
        match = _KEY.match(line)
        if match:
            found.append((number, match.group(1), match.start(1) + 1))
    looks_like_training = any(key == "output_dir" for _, key, _ in found)
    spec = R.RULES["TV5M14"]
    findings = []
    for number, key, col in found:
        if key in R.CONFIG_KEYS or (key == "use_auth_token" and looks_like_training):
            params = {"name": key, **_hint(key)}
            findings.append(Finding("TV5M14", spec.severity, "manual", path, number, col, "msg.TV5M14", params))
    return findings
