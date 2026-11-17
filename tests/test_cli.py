"""AC-6 and AC-10: output formats and exit codes."""

import json
import re
import shutil
import subprocess
import sys

import pytest

from transformers_v5_migrate import RULES
from transformers_v5_migrate.cli import main

SUMMARY = re.compile(r"\d+ auto-fixable · \d+ manual · \d+ behavior · \d+ dependency")


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_json_shape(capsys, example_project):
    code, out, _ = run(capsys, str(example_project), "--format", "json")
    data = json.loads(out)
    assert code == 1
    assert data["tool"] == "transformers-v5-migrate" and data["target"] == "5.23.0" and data["lang"] == "en"
    assert set(data["summary"]) == {"fixable", "manual", "behavior", "dependency", "files_scanned", "files_changed"}
    assert data["summary"]["fixable"] == len(data["fixes"]) > 0
    assert data["summary"]["manual"] + data["summary"]["behavior"] + data["summary"]["dependency"] == len(data["findings"])
    first = data["findings"][0]
    assert {"rule", "severity", "kind", "path", "line", "col", "cell", "message", "recommendation", "fixable"} <= set(first)
    assert {"rule", "path", "line", "col", "before", "after", "message"} <= set(data["fixes"][0])


def test_sarif_shape(capsys, example_project, tmp_path):
    out_file = tmp_path / "out.sarif"
    code, out, err = run(capsys, str(example_project), "--format", "sarif", "--output", str(out_file))
    assert code == 1 and out == "" and str(out_file) in err
    sarif = json.loads(out_file.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    driver = sarif["runs"][0]["tool"]["driver"]
    declared = {r["id"] for r in driver["rules"]}
    assert declared == set(RULES)
    results = sarif["runs"][0]["results"]
    assert results and {r["ruleId"] for r in results} <= declared
    assert {r["level"] for r in results} <= {"error", "warning", "note"}
    plain = [r for r in results if r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].endswith(".py")]
    location = plain[0]["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] and location["region"]["startLine"] >= 1
    notebook = [r for r in results if r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].endswith(".ipynb")]
    assert notebook and all("cell" in r["properties"] for r in notebook)
    # cell lines are not notebook JSON lines: no region, the cell position is in the message
    assert all("region" not in r["locations"][0]["physicalLocation"] for r in notebook)
    assert all(r["message"]["text"].startswith(f"cell {r['properties']['cell']}:") for r in notebook)


def test_exit_codes(capsys, tmp_path, example_project):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.py").write_text("import transformers\nx = 1\n")
    assert run(capsys, str(clean))[0] == 0
    assert run(capsys, str(example_project))[0] == 1
    assert run(capsys, str(tmp_path / "missing"))[0] == 2
    assert run(capsys, str(example_project), "--fail-on", "never")[0] == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(capsys, str(empty))
    assert code == 2 and "no files" in err


def test_write_leaves_manual_errors_so_exit_is_still_one(capsys, tmp_path, example_project):
    work = tmp_path / "p"
    shutil.copytree(example_project, work)
    code, out, _ = run(capsys, str(work), "--write")
    assert code == 1  # manual errors remain
    assert re.search(r"\d+ auto-fixed · ", out)
    code, out, _ = run(capsys, str(work))
    assert code == 1
    assert "0 auto-fixable" in out


def test_fixed_only_input_exits_zero_after_write(capsys, tmp_path):
    (tmp_path / "t.py").write_text("from transformers import Trainer\nTrainer(model=m, tokenizer=t)\n")
    assert run(capsys, str(tmp_path))[0] == 1  # dry run: the fix is still an open error
    assert run(capsys, str(tmp_path), "--write")[0] == 0
    assert run(capsys, str(tmp_path))[0] == 0


def test_fail_on_levels(capsys, tmp_path):
    (tmp_path / "t.py").write_text("import transformers\nm = transformers.AutoModelForCausalLM.from_pretrained('x', dtype='auto')\ntok.batch_decode(i)\n")
    assert run(capsys, str(tmp_path), "--fail-on", "error")[0] == 0  # only a note
    assert run(capsys, str(tmp_path), "--fail-on", "note")[0] == 1


def test_select_ignore_and_validation(capsys, example_project):
    code, out, _ = run(capsys, str(example_project), "--select", "TV5D", "--format", "json")
    assert {f["rule"] for f in json.loads(out)["findings"]} == {"TV5D01", "TV5D02"}
    assert json.loads(out)["fixes"] == []
    code, out, _ = run(capsys, str(example_project), "--ignore", "TV5R,TV5M,TV5B,TV5P", "--format", "json")
    assert {f["rule"] for f in json.loads(out)["findings"]} == {"TV5D01", "TV5D02"}
    with pytest.raises(SystemExit) as info:
        main([str(example_project), "--select", "NOPE"])
    assert info.value.code == 2


def test_exclude_glob(capsys, example_project):
    _, out, _ = run(capsys, str(example_project), "--exclude", "*.ipynb", "--exclude", "Dockerfile", "--format", "json")
    paths = {f["path"] for f in json.loads(out)["findings"]}
    assert not any(p.endswith((".ipynb", "Dockerfile")) for p in paths)


def test_text_report_has_diff_sections_and_summary_last(capsys, example_project):
    code, out, _ = run(capsys, str(example_project), "--fail-on", "never")
    assert code == 0
    lines = out.rstrip("\n").splitlines()
    assert SUMMARY.fullmatch(lines[-1])
    assert "== Preview of the changes ==" in out and "-    no_cuda=True," in out and "+    use_cpu=True," in out
    assert "TV5M01" in out and "fix: " in out and "\x1b[" not in out
    _, no_diff, _ = run(capsys, str(example_project), "--no-diff", "--fail-on", "never")
    assert "Preview" not in no_diff


def test_color_always_and_no_color(capsys, example_project, monkeypatch):
    assert "\x1b[" in run(capsys, str(example_project), "--color", "always", "--fail-on", "never")[1]
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in run(capsys, str(example_project), "--fail-on", "never")[1]


def test_list_rules(capsys):
    code, out, _ = run(capsys, "--list-rules")
    assert code == 0
    for rule_id in RULES:
        assert rule_id in out
    assert "5.23.0" in out


def test_no_paths_is_usage_error(capsys):
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "transformers-v5-migrate 0.1.0" in capsys.readouterr().out


def test_first_run_command_prints_summary_and_exits_zero(example_project):
    """AC-10: the documented start command, run as a real process, finishes fast."""
    proc = subprocess.run(
        [sys.executable, "-m", "transformers_v5_migrate", str(example_project), "--fail-on", "never"],
        capture_output=True, text=True, timeout=30, env={"PATH": "", "LANG": "C"},
    )
    assert proc.returncode == 0, proc.stderr
    assert SUMMARY.fullmatch(proc.stdout.rstrip("\n").splitlines()[-1])


def test_only_unsupported_explicit_files_exit_zero(capsys, tmp_path):
    """Q-1: pre-commit passes every staged file that matches its pattern, even kinds we skip."""
    (tmp_path / "notes.txt").write_text("hello\n")
    (tmp_path / "package.json").write_text("{}\n")
    (tmp_path / "ruff.toml").write_text("x = 1\n")
    names = [str(tmp_path / n) for n in ("notes.txt", "package.json", "ruff.toml")]
    code, out, err = run(capsys, "--no-diff", "--lang", "en", *names)
    assert code == 0 and out == "" and "nothing to check" in err


def test_missing_path_and_empty_directory_still_exit_two(capsys, tmp_path):
    assert run(capsys, "--lang", "en", str(tmp_path / "nope.py"))[0] == 2
    assert run(capsys, "--lang", "en", str(tmp_path))[0] == 2


def test_unparsable_file_without_transformers_is_only_a_note(capsys, tmp_path):
    (tmp_path / "legacy.py").write_text('print "hello"\n')
    (tmp_path / "train.py").write_text("import transformers\nx = (\n")
    code, out, _ = run(capsys, str(tmp_path), "--no-diff", "--lang", "en", "--format", "json")
    levels = {f["path"].rsplit("/", 1)[-1]: f["severity"] for f in json.loads(out)["findings"]}
    assert levels == {"legacy.py": "note", "train.py": "error"} and code == 1


def test_single_file_summary_is_singular(capsys, tmp_path):
    (tmp_path / "ok.py").write_text("import transformers\n")
    _, out, _ = run(capsys, str(tmp_path / "ok.py"), "--lang", "en")
    assert "1 file checked" in out and "1 files" not in out


def test_removed_model_families_without_replacement_are_reported(capsys, tmp_path):
    (tmp_path / "a.py").write_text("from transformers import TvltFeatureExtractor, MCTCTFeatureExtractor\n")
    _, out, _ = run(capsys, str(tmp_path), "--lang", "en", "--format", "json")
    assert [f["rule"] for f in json.loads(out)["findings"]] == ["TV5M03", "TV5M03"]
