"""English and Korean messages, and how the language is chosen.

Order: ``--lang`` > ``TRANSFORMERS_V5_MIGRATE_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
A value starting with ``ko`` selects Korean, anything else English. All user-visible
text lives in ``locales/en.json`` and ``locales/ko.json`` (identical key sets).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Any

SUPPORTED = ("en", "ko")
ENV_VAR = "TRANSFORMERS_V5_MIGRATE_LANG"


def resolve_lang(cli_lang: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Pick ``"en"`` or ``"ko"``; the first non-empty source decides."""
    env = os.environ if env is None else env
    for value in (cli_lang, env.get(ENV_VAR), env.get("LC_ALL"), env.get("LC_MESSAGES"), env.get("LANG")):
        if value and value.strip():
            return "ko" if value.strip().lower().startswith("ko") else "en"
    return "en"


@lru_cache(maxsize=None)
def catalog(lang: str) -> dict[str, str]:
    """The message table for ``lang`` (falls back to English for unknown languages)."""
    name = lang if lang in SUPPORTED else "en"
    text = resources.files("transformers_v5_migrate").joinpath("locales", f"{name}.json").read_text(encoding="utf-8")
    return json.loads(text)


def t(key: str, lang: str, **params: Any) -> str:
    """Look up ``key`` and fill ``{placeholders}``; unknown keys come back as the key itself."""
    template = catalog(lang).get(key) or catalog("en").get(key) or key
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return template
