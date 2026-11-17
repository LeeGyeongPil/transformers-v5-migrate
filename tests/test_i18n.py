"""AC-7: English and Korean, same keys, clear precedence."""

import json
import re
import subprocess
import sys
from importlib import resources

import pytest

from transformers_v5_migrate import RULES
from transformers_v5_migrate.cli import main
from transformers_v5_migrate.i18n import catalog, resolve_lang, t


def _load(lang):
    return json.loads(resources.files("transformers_v5_migrate").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8"))


def test_catalog_keys_equal():
    en, ko = _load("en"), _load("ko")
    assert set(en) == set(ko)
    assert all(isinstance(v, str) and v for v in en.values())
    assert all(isinstance(v, str) and v for v in ko.values())


def test_placeholders_match_between_languages():
    en, ko = _load("en"), _load("ko")
    for key in en:
        assert set(re.findall(r"{(\w+)}", en[key])) == set(re.findall(r"{(\w+)}", ko[key])), key


def test_every_rule_has_title_message_and_recommendation_in_both_languages():
    for lang in ("en", "ko"):
        cat = catalog(lang)
        for rule_id in RULES:
            assert cat[f"title.{rule_id}"]
            assert cat[f"fix.{rule_id}"]
            assert f"msg.{rule_id}" in cat, (lang, rule_id)


def test_korean_texts_really_differ_from_english():
    en, ko = _load("en"), _load("ko")
    assert [k for k in en if en[k] == ko[k]] == ["list.col.id"]  # "ID" is the same in both


def test_lang_resolution_order():
    env = {"TRANSFORMERS_V5_MIGRATE_LANG": "en", "LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "ko", "LANG": "ko"}
    assert resolve_lang("ko", env) == "ko"  # --lang beats everything
    assert resolve_lang(None, env) == "en"  # tool variable beats locale
    del env["TRANSFORMERS_V5_MIGRATE_LANG"]
    assert resolve_lang(None, env) == "ko"  # LC_ALL
    assert resolve_lang(None, {"LC_ALL": "C", "LANG": "ko_KR.UTF-8"}) == "en"  # first non-empty wins, C is not Korean
    assert resolve_lang(None, {"LC_ALL": "", "LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"
    assert resolve_lang(None, {}) == "en"
    assert resolve_lang("EN", {"LANG": "ko"}) == "en"


def test_t_fills_placeholders_and_falls_back():
    assert "`x`" in t("msg.TV5M03", "en", name="x")
    assert t("no.such.key", "ko") == "no.such.key"
    assert t("msg.TV5M03", "fr", name="x") == t("msg.TV5M03", "en", name="x")


def test_cli_output_language(capsys, example_project):
    main([str(example_project), "--lang", "ko", "--fail-on", "never", "--no-diff"])
    ko = capsys.readouterr().out
    main([str(example_project), "--lang", "en", "--fail-on", "never", "--no-diff"])
    en = capsys.readouterr().out
    assert "손으로 고쳐야 할 곳" in ko and "자동 수정 가능" in ko.splitlines()[-1]
    assert "Manual changes needed" in en and "auto-fixable" in en.splitlines()[-1]
    assert not re.search(r"[가-힣]", en)
    # rule ids are not translated
    assert "TV5M01" in ko and "TV5M01" in en


def test_environment_variable_and_flag_switch_language(capsys, example_project, monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_V5_MIGRATE_LANG", "ko")
    main([str(example_project), "--fail-on", "never", "--no-diff"])
    assert "자동 수정 가능" in capsys.readouterr().out
    main([str(example_project), "--fail-on", "never", "--no-diff", "--lang", "en"])
    assert "auto-fixable" in capsys.readouterr().out


def test_help_is_translated(capsys):
    with pytest.raises(SystemExit):
        main(["--lang", "ko", "--help"])
    ko = capsys.readouterr().out
    assert "사용법:" in ko and "자동 수정을 .py·.ipynb 파일에 적용합니다" in ko and "usage:" not in ko
    with pytest.raises(SystemExit):
        main(["--lang", "en", "--help"])
    en = capsys.readouterr().out
    assert "usage:" in en and "--write" in en and "사용법" not in en


def test_json_messages_follow_language(capsys, example_project):
    main([str(example_project), "--lang", "ko", "--format", "json", "--fail-on", "never"])
    data = json.loads(capsys.readouterr().out)
    assert data["lang"] == "ko" and re.search(r"[가-힣]", data["findings"][0]["message"])
    assert data["findings"][0]["rule"].startswith("TV5")  # ids stay the same


def test_list_rules_in_korean(capsys):
    main(["--list-rules", "--lang", "ko"])
    out = capsys.readouterr().out
    assert "규칙" in out and "TV5R03" in out


def test_usage_error_is_translated(capsys):
    with pytest.raises(SystemExit):
        main(["--lang", "ko"])
    assert "오류" in capsys.readouterr().err


def test_help_through_a_real_process_uses_locale_env():
    proc = subprocess.run(
        [sys.executable, "-m", "transformers_v5_migrate", "--help"],
        capture_output=True, text=True, timeout=30, env={"PATH": "", "LANG": "ko_KR.UTF-8", "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0 and "사용법:" in proc.stdout
