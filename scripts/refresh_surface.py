#!/usr/bin/env python3
"""Regenerate ``data/v5_surface.json`` from two real transformers installs.

The rule table in ``rules.py`` is only trustworthy while it matches what the
installed libraries really do. This maintainer script compares the public API
surface of an old (v4) and a new (v5) transformers install and writes the
evidence snapshot that ``tests/test_rules_surface.py`` checks the rules against.

Typical use (two throw-away virtual environments, no torch needed)::

    uv venv -q --python 3.12 /tmp/tv4 && uv pip install -q --python /tmp/tv4/bin/python "transformers==4.57.1"
    uv venv -q --python 3.12 /tmp/tv5 && uv pip install -q --python /tmp/tv5/bin/python "transformers==5.23.0"
    python scripts/refresh_surface.py --v4-python /tmp/tv4/bin/python --v5-python /tmp/tv5/bin/python

Offline variant: pass two surface dumps produced by ``--dump``::

    /tmp/tv5/bin/python scripts/refresh_surface.py --dump v5_surface.json
    python scripts/refresh_surface.py --from-dumps v4_surface.json v5_surface.json
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "transformers_v5_migrate" / "data" / "v5_surface.json"

# Names the rules talk about: keep the snapshot small and focused.
RELEVANT_REMOVED = re.compile(r"^(TF|Flax)|FeatureExtractor$|ImageProcessorFast$|^AutoModel(ForVision2Seq|WithLMHead)$")
REPLACEMENT_HINTS = ("AutoModelForImageTextToText",)


def dump_surface(path: str) -> None:
    """Run inside an environment that has transformers installed: write its API surface."""
    import transformers
    from importlib.metadata import requires

    out: dict = {"version": transformers.__version__}
    from transformers import PreTrainedTokenizerBase, TrainingArguments
    from transformers.pipelines import SUPPORTED_TASKS

    out["TrainingArguments"] = sorted(f.name for f in dataclasses.fields(TrainingArguments))
    out["tokenizer_methods"] = sorted(n for n in dir(PreTrainedTokenizerBase) if not n.startswith("_"))
    out["pipeline_tasks"] = sorted(SUPPORTED_TASKS)
    out["top_level"] = sorted(n for n in dir(transformers) if not n.startswith("_"))
    out["hub_requirement"] = next((r for r in (requires("transformers") or []) if r.startswith("huggingface-hub")), "")
    sig = inspect.signature(PreTrainedTokenizerBase.apply_chat_template)
    out["apply_chat_template_return_dict_default"] = sig.parameters["return_dict"].default
    Path(path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def build_snapshot(v4: dict, v5: dict) -> dict:
    """Pure function: two surface dumps in, rule evidence out."""
    t4, t5 = set(v4["top_level"]), set(v5["top_level"])
    removed = sorted(n for n in t4 - t5 if RELEVANT_REMOVED.search(n))
    present = sorted(n for n in REPLACEMENT_HINTS if n in t5)
    # Replacement image processors for every removed image FeatureExtractor.
    present += sorted(n.replace("FeatureExtractor", "ImageProcessor") for n in removed
                      if n.endswith("FeatureExtractor") and n.replace("FeatureExtractor", "ImageProcessor") in t5)
    present += sorted(n.removesuffix("Fast") for n in removed if n.endswith("ImageProcessorFast") and n.removesuffix("Fast") in t5)
    return {
        "checked": date.today().isoformat(),
        "v4_version": v4["version"],
        "v5_version": v5["version"],
        "hub_requirement": v5.get("hub_requirement", ""),
        "apply_chat_template_return_dict_default": v5.get("apply_chat_template_return_dict_default"),
        "training_arguments": {"v4": sorted(v4["TrainingArguments"]), "v5": sorted(v5["TrainingArguments"])},
        "tokenizer_methods": {"removed": sorted(set(v4["tokenizer_methods"]) - set(v5["tokenizer_methods"]))},
        "pipeline_tasks": {"removed": sorted(set(v4["pipeline_tasks"]) - set(v5["pipeline_tasks"]))},
        "top_level": {"removed": removed, "present_in_v5": present},
    }


def compare_snapshots(old: dict, new: dict) -> list[str]:
    """Human-readable differences that matter for the rules (empty list: the rules still hold)."""
    problems: list[str] = []

    def sets(label: str, before: list[str], after: list[str]) -> None:
        added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
        if added:
            problems.append(f"{label}: new in the latest install: {', '.join(added)}")
        if removed:
            problems.append(f"{label}: gone from the latest install: {', '.join(removed)}")

    sets("TrainingArguments fields (v5)", old["training_arguments"]["v5"], new["training_arguments"]["v5"])
    sets("TrainingArguments fields (v4)", old["training_arguments"]["v4"], new["training_arguments"]["v4"])
    sets("removed tokenizer names", old["tokenizer_methods"]["removed"], new["tokenizer_methods"]["removed"])
    sets("removed pipeline tasks", old["pipeline_tasks"]["removed"], new["pipeline_tasks"]["removed"])
    sets("removed top-level names", old["top_level"]["removed"], new["top_level"]["removed"])
    sets("replacement names present in v5", old["top_level"]["present_in_v5"], new["top_level"]["present_in_v5"])
    for key in ("hub_requirement", "apply_chat_template_return_dict_default"):
        if old.get(key) != new.get(key):
            problems.append(f"{key}: {old.get(key)!r} -> {new.get(key)!r}")
    return problems


def _run_dump(python: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:  # removed again once the dump has been read
        target = Path(tmp) / "surface.json"
        subprocess.run([python, str(Path(__file__).resolve()), "--dump", str(target)], check=True)
        return json.loads(target.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", metavar="FILE", help="write the API surface of the running interpreter's transformers and exit")
    ap.add_argument("--v4-python", help="python executable of an environment with the old transformers")
    ap.add_argument("--v5-python", help="python executable of an environment with the new transformers")
    ap.add_argument("--from-dumps", nargs=2, metavar=("V4_JSON", "V5_JSON"), help="use two existing --dump files")
    ap.add_argument("--out", default=str(OUT), help="snapshot path (default: the packaged data file)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write: compare the fresh comparison with the committed snapshot and exit 1 on rule-relevant differences",
    )
    args = ap.parse_args(argv)

    if args.dump:
        dump_surface(args.dump)
        return 0
    if args.from_dumps:
        v4 = json.loads(Path(args.from_dumps[0]).read_text(encoding="utf-8"))
        v5 = json.loads(Path(args.from_dumps[1]).read_text(encoding="utf-8"))
    elif args.v4_python and args.v5_python:
        v4, v5 = _run_dump(args.v4_python), _run_dump(args.v5_python)
    else:
        ap.error("give --v4-python and --v5-python, or --from-dumps V4_JSON V5_JSON, or --dump FILE")
    snap = build_snapshot(v4, v5)
    if args.check:
        committed = json.loads(Path(args.out).read_text(encoding="utf-8"))
        problems = compare_snapshots(committed, snap)
        print(f"committed snapshot: transformers {committed['v5_version']}; latest install: {snap['v5_version']}")
        for problem in problems:
            print(f"DIFF {problem}")
        if problems:
            print("Update rules.py (and TARGET_VERSION), re-run this script without --check, then run the tests.")
            return 1
        print("No rule-relevant difference: the rules still match." + (
            f" Consider bumping TARGET_VERSION to {snap['v5_version']}." if snap["v5_version"] != committed["v5_version"] else ""))
        return 0
    Path(args.out).write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: transformers {snap['v4_version']} -> {snap['v5_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
