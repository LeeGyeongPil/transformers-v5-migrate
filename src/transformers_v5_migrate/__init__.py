"""transformers-v5-migrate: codemod and risk report for Hugging Face Transformers v4 -> v5.

Public API::

    from transformers_v5_migrate import migrate_source
    result = migrate_source(open("train.py").read(), path="train.py")
    print(result.code)        # code with the mechanical fixes applied
    print(result.findings)    # things a human must look at
"""

__version__ = "0.1.0"

from .codemod import migrate_source  # noqa: E402
from .model import FileDiff, Finding, Fix, ParseError, Report, Result  # noqa: E402
from .notebook import migrate_notebook  # noqa: E402
from .rules import RULES, TARGET_VERSION, Rule  # noqa: E402
from .scan import scan_paths  # noqa: E402

__all__ = [
    "RULES",
    "TARGET_VERSION",
    "FileDiff",
    "Finding",
    "Fix",
    "ParseError",
    "Report",
    "Result",
    "Rule",
    "__version__",
    "migrate_notebook",
    "migrate_source",
    "scan_paths",
]
