"""AC-2: report-only findings match the expectations written next to the code, nothing more."""

import re
from collections import Counter

from transformers_v5_migrate import migrate_source

EXPECT = re.compile(r"#\s*expect:\s*(.+)$")


def _expected(text: str) -> Counter:
    wanted: Counter = Counter()
    for number, line in enumerate(text.splitlines(), start=1):
        match = EXPECT.search(line)
        if match:
            for rule in match.group(1).split(","):
                wanted[(rule.strip(), number)] += 1
    return wanted


def test_expected_rule_lines(fixtures):
    text = (fixtures / "findings_v4.py").read_text(encoding="utf-8")
    result = migrate_source(text, path="findings_v4.py")
    actual = Counter((f.rule, f.line) for f in result.findings)
    assert actual == _expected(text)
    assert result.fixes == []  # report-only fixture: nothing is rewritten
    assert result.code == text


def test_expected_markers_cover_every_manual_and_behavior_rule(fixtures):
    text = (fixtures / "findings_v4.py").read_text(encoding="utf-8")
    rules = {rule for rule, _ in _expected(text)}
    wanted = {f"TV5M{n:02d}" for n in range(1, 9)} | {"TV5B01", "TV5B02"}
    assert rules >= wanted


def test_no_false_positives(fixtures):
    text = (fixtures / "no_false_positives.py").read_text(encoding="utf-8")
    result = migrate_source(text)
    assert result.findings == [] and result.fixes == []
    assert result.code == text


def test_names_without_transformers_import_are_not_reported(fixtures):
    text = (fixtures / "no_import.py").read_text(encoding="utf-8")
    result = migrate_source(text)
    assert result.findings == [] and result.fixes == [] and result.code == text


def test_guide_gap_fields_are_reported_as_removed():
    code = (
        "from transformers import TrainingArguments\n"
        "TrainingArguments(adafactor=True, fp16_opt_level='O1', group_by_length=True, save_safetensors=True, torchdynamo='eager')\n"
    )
    names = {f.params["name"] for f in migrate_source(code).findings}
    assert names == {"adafactor", "fp16_opt_level", "group_by_length", "save_safetensors", "torchdynamo"}


def test_findings_carry_severity_kind_and_location():
    code = "import os\nimport transformers\nos.environ['TRANSFORMERS_CACHE'] = 'x'\n"
    (finding,) = migrate_source(code, path="p.py").findings
    assert (finding.rule, finding.severity, finding.kind) == ("TV5M08", "warning", "manual")
    assert (finding.path, finding.line, finding.col) == ("p.py", 3, 12)
