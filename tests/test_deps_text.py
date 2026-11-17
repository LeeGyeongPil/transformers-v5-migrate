"""AC-5: dependency pins and text files are checked but never rewritten."""

import pytest

from transformers_v5_migrate import scan_paths
from transformers_v5_migrate.deps import scan_pyproject, scan_requirements
from transformers_v5_migrate.textscan import scan_text


def _rules(findings):
    return [(f.rule, f.line) for f in findings]


def test_dependency_pins(example_project):
    report = scan_paths([str(example_project / "constraints.txt"), str(example_project / "pyproject.toml")])
    by_file = {(f.path.rsplit("/", 1)[-1], f.rule, f.line) for f in report.findings}
    assert by_file == {
        ("constraints.txt", "TV5D01", 2),
        ("constraints.txt", "TV5D02", 3),
    }
    assert all(f.kind == "dependency" for f in report.findings)


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


def test_env_and_cli_in_text_files(example_project):
    report = scan_paths(
        [str(example_project / n) for n in ("Dockerfile", ".env.example", "run.sh")], select=["TV5M"]
    )
    assert {(f.path.rsplit("/", 1)[-1], f.rule, f.line) for f in report.findings} == {
        ("Dockerfile", "TV5M08", 3),
        ("Dockerfile", "TV5M10", 11),
        (".env.example", "TV5M08", 3),
        ("run.sh", "TV5M08", 4),
    }


@pytest.mark.parametrize(
    "text, rules",
    [
        ("ENV TRANSFORMERS_CACHE=/x", ["TV5M08"]),
        ("export PYTORCH_TRANSFORMERS_CACHE=/x", ["TV5M08"]),
        ("ENV MY_TRANSFORMERS_CACHE_DIR=/x", []),  # different variable
        ("# TRANSFORMERS_CACHE=/x", []),  # comment
        ("RUN transformers-cli login", ["TV5M10"]),
        ("RUN pip install transformers-cli-tools", []),
        ("RUN ./transformers-cli", []),
        ("transformers env", []),
        ("HF_HOME=/x", []),
    ],
)
def test_text_scan_word_boundaries(text, rules):
    assert [f.rule for f in scan_text(text, "f")] == rules


def test_text_files_never_written(tmp_path):
    files = {
        "Dockerfile": "ENV TRANSFORMERS_CACHE=/x\nRUN transformers-cli env\n",
        ".env": "HF_HUB_ENABLE_HF_TRANSFER=1\n",
        "requirements.txt": "transformers==4.57.1\n",
        "pyproject.toml": '[project]\nname="x"\ndependencies=["transformers<5"]\n',
        "run.sh": "export TRANSFORMERS_CACHE=/x\n",
        "ci.yml": "env:\n  TRANSFORMERS_CACHE: /x\n",
        "Makefile": "setup:\n\ttransformers-cli env\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    report = scan_paths([str(tmp_path)], write=True)
    assert report.files_scanned == len(files)
    assert report.files_changed == []
    for name, text in files.items():
        assert (tmp_path / name).read_text() == text
    assert len(report.findings) >= len(files)


def test_sample_project_references_existing_files(example_project):
    """Regression: files named in the sample Dockerfile must exist (the requirements.txt rename broke it)."""
    import re

    dockerfile = (example_project / "Dockerfile").read_text()
    names = re.findall(r"^(?:COPY\s+(\S+)\s+\.|RUN\s+pip install -r\s+(\S+))", dockerfile, re.M)
    referenced = {n for pair in names for n in pair if n and n != "."}
    assert "constraints.txt" in referenced
    for name in referenced:
        assert (example_project / name).is_file(), f"Dockerfile references missing {name}"
