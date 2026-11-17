"""AC-9: the repository stays small, consistent and free of pinned dependency manifests."""

import re
import subprocess
import sys
import time
from pathlib import Path

import transformers_v5_migrate as pkg
from transformers_v5_migrate import RULES

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "dist", "build"}

V2_RULES = (
    {f"TV5R{n:02d}" for n in range(1, 10)}
    | {f"TV5M{n:02d}" for n in range(1, 9)}
    | {"TV5B01", "TV5B02", "TV5D01", "TV5D02", "TV5P01"}
)
MANIFEST_NAMES = ("requirements*.txt", "constraints*.txt", "pyproject.toml", "setup.py", "Pipfile*", "poetry.lock", "environment.yml")


def _files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS) and not path.name.endswith(".egg-info"):
            yield path


def test_rule_set_is_v2_scope():
    assert set(RULES) == V2_RULES and len(RULES) == 22


def test_version_consistent():
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    entries = re.findall(r"^## \[([^\]]+)\]", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert project == pkg.__version__ == entries[0]
    assert len(entries) == 1, "the changelog has exactly one release entry (the first public release)"


def test_no_dependency_manifests_outside_root():
    allowed = {ROOT / "pyproject.toml", ROOT / "uv.lock"}
    offenders = [
        str(p.relative_to(ROOT))
        for p in _files()
        if p not in allowed and any(p.match(pattern) for pattern in MANIFEST_NAMES)
    ]
    assert offenders == []


def test_images_small_and_relative():
    images = sorted((ROOT / "docs" / "images").glob("*.png"))
    assert 1 <= len(images) <= 2
    assert all(p.stat().st_size <= 200 * 1024 for p in images), [(p.name, p.stat().st_size) for p in images]
    for readme in ("README.md", "README.ko.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        assert "raw.githubusercontent.com" not in text
        links = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text)
        assert links, f"{readme} shows no image"
        for link in links:
            assert not link.startswith(("http://", "https://")), link
            assert (ROOT / link).is_file(), f"{readme} links to a missing image: {link}"


def test_removed_surfaces_absent():
    for rel in (
        ".pre-commit-hooks.yaml",
        ".github/workflows/surface-watch.yml",
        "src/transformers_v5_migrate/textscan.py",
        "src/transformers_v5_migrate/configscan.py",
    ):
        assert not (ROOT / rel).exists(), rel


def test_repo_size_budget():
    total = sum(p.stat().st_size for p in _files())
    assert total <= 700 * 1024, f"{total} bytes"


def test_first_run_command_is_fast_and_prints_a_summary():
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "transformers_v5_migrate", str(ROOT / "examples" / "v4_project"), "--fail-on", "never", "--lang", "en"],
        capture_output=True, text=True, timeout=30, env={"PATH": "", "LANG": "C"},
    )
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - start < 5
    assert re.fullmatch(r"\d+ auto-fixable · \d+ manual · \d+ behavior · \d+ dependency", proc.stdout.rstrip("\n").splitlines()[-1])
