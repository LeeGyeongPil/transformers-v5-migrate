"""AC-2/AC-3: dependency pins are checked but never rewritten. Manifests are built in tmp_path on purpose."""

import pytest

from transformers_v5_migrate import scan_paths
from transformers_v5_migrate.deps import scan_pyproject, scan_requirements


def _rules(findings):
    return [(f.rule, f.line) for f in findings]


def test_dependency_pins(tmp_path):
    (tmp_path / "requirements.txt").write_text("# pins\ntransformers==4.57.1\nhuggingface_hub<1.0\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["transformers>=4.40,<5"]\n')
    report = scan_paths([str(tmp_path)])
    found = {(f.path.rsplit("/", 1)[-1], f.rule, f.line) for f in report.findings}
    assert found == {
        ("requirements.txt", "TV5D01", 2),
        ("requirements.txt", "TV5D02", 3),
        ("pyproject.toml", "TV5D01", 3),
    }
    assert all(f.kind == "dependency" for f in report.findings)


def test_constraints_file_is_checked_too(tmp_path):
    (tmp_path / "constraints-dev.txt").write_text("transformers<5\n")
    assert [f.rule for f in scan_paths([str(tmp_path)]).findings] == ["TV5D01"]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("transformers==4.57.1", ["TV5D01"]),
        ("transformers<5", ["TV5D01"]),
        ("transformers~=4.40", ["TV5D01"]),
        ("transformers[torch]>=4.40,<5.0", ["TV5D01"]),
        ("Transformers == 4.*", ["TV5D01"]),
        ("transformers>=4.40", []),
        ("transformers>=5", []),
        ("transformers==5.23.0", []),
        ("transformers", []),
        ("huggingface_hub<1.0", ["TV5D02"]),
        ("huggingface-hub==0.25.0", ["TV5D02"]),
        ("huggingface-hub~=0.34", ["TV5D02"]),
        ("huggingface-hub>=0.20", []),
        ("huggingface-hub<2", []),
        ("huggingface-hub>=1.31.0,<3.0", []),
        ("# transformers<5", []),
        ("-r other.txt", []),
        ("torch<2", []),
        ("git+https://example.org/x.git", []),
    ],
)
def test_requirement_lines(line, expected):
    assert [f.rule for f in scan_requirements(line + "\n", "r.txt", None, None)] == expected


def test_pyproject_variants():
    text = """\
[project]
name = "x"
dependencies = ["transformers>=4.40,<5", "huggingface-hub<1"]

[project.optional-dependencies]
extra = ["transformers==4.57.1"]

[dependency-groups]
dev = ["transformers<5", {include-group = "extra"}]

[tool.poetry.dependencies]
python = "^3.11"
transformers = "^4.40"

[tool.poetry.group.test.dependencies]
transformers = { version = "~4.50", optional = true }
"""
    found = scan_pyproject(text, "pyproject.toml", None, None)
    assert [f.rule for f in found] == ["TV5D01", "TV5D02", "TV5D01", "TV5D01", "TV5D01", "TV5D01"]
    assert [f.line for f in found] == [3, 3, 6, 9, 13, 16]


def test_pyproject_that_allows_v5_is_clean():
    text = '[project]\nname = "x"\ndependencies = ["transformers>=4.40,<6"]\n[tool.poetry.dependencies]\ntransformers = ">=4.40"\n'
    assert scan_pyproject(text, "p", None, None) == []


def test_broken_toml_is_ignored():
    assert scan_pyproject("[project\n", "p", None, None) == []


def test_dependency_files_never_written(tmp_path):
    files = {
        "requirements.txt": "transformers==4.57.1\n",
        "constraints.txt": "huggingface_hub<1.0\n",
        "pyproject.toml": '[project]\nname="x"\ndependencies=["transformers<5"]\n',
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    report = scan_paths([str(tmp_path)], write=True)
    assert report.files_scanned == len(files)
    assert report.files_changed == []
    for name, text in files.items():
        assert (tmp_path / name).read_text() == text
    assert len(report.findings) == len(files)


def test_other_file_kinds_are_not_scanned(tmp_path):
    for name in ("Dockerfile", ".env", "run.sh", "ci.yml", "environment.yml", "Makefile", "train_args.json"):
        (tmp_path / name).write_text("TRANSFORMERS_CACHE=/x\ntransformers==4.57.1\n")
    (tmp_path / "ok.py").write_text("x = 1\n")
    report = scan_paths([str(tmp_path)])
    assert report.files_scanned == 1 and report.findings == []
