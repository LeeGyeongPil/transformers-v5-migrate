from pathlib import Path

import pytest

TESTS = Path(__file__).parent
ROOT = TESTS.parent


@pytest.fixture
def fixtures() -> Path:
    return TESTS / "fixtures"


@pytest.fixture
def snapshots() -> Path:
    return TESTS / "snapshots"


@pytest.fixture
def example_project() -> Path:
    return ROOT / "examples" / "v4_project"


@pytest.fixture(autouse=True)
def _clean_language_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's locale."""
    for name in ("TRANSFORMERS_V5_MIGRATE_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
