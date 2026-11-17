"""AC-8: the rule table agrees with the recorded 4.57.1 -> 5.23.0 install comparison."""

import json
from importlib import resources

import pytest

from transformers_v5_migrate import RULES, rules as R
from transformers_v5_migrate.deps import _HUB


@pytest.fixture(scope="module")
def surface():
    text = resources.files("transformers_v5_migrate").joinpath("data", "v5_surface.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_snapshot_is_for_the_versions_the_rules_claim(surface):
    assert surface["v5_version"] == R.TARGET_VERSION == "5.23.0"
    assert surface["v4_version"] == "4.57.1"


def test_rules_consistent_with_surface(surface):
    ta4, ta5 = set(surface["training_arguments"]["v4"]), set(surface["training_arguments"]["v5"])

    # TV5R05 / R06 / R07: old names existed in v4, are gone in v5; new names exist in v5
    renames = dict(R.ARGS_RENAMES, warmup_ratio="warmup_steps", include_inputs_for_metrics="include_for_metrics")
    for old, new in renames.items():
        assert old in ta4 and old not in ta5, old
        assert new in ta5, new

    # TV5M06: every listed field is really removed, and nothing removed is forgotten
    assert set(R.ARGS_REMOVED) <= ta4 - ta5
    handled = set(R.ARGS_REMOVED) | set(renames)
    assert ta4 - ta5 == handled, sorted((ta4 - ta5) ^ handled)
    assert set(R.ARGS_REMOVED_NOT_IN_GUIDE) <= set(R.ARGS_REMOVED)

    # TV5M05: removed pipeline tasks
    assert set(R.REMOVED_PIPELINE_TASKS) == set(surface["pipeline_tasks"]["removed"])

    # TV5M07: removed tokenizer names (encode_plus & batch_encode_plus are TV5R08)
    removed_tok = set(surface["tokenizer_methods"]["removed"])
    assert set(R.REMOVED_TOKENIZER_NAMES) <= removed_tok
    assert {"encode_plus", "batch_encode_plus"} <= removed_tok
    assert removed_tok - set(R.REMOVED_TOKENIZER_NAMES) - {"encode_plus", "batch_encode_plus"} == set()

    # TV5R09 / TV5M11: removed top-level classes and their replacements
    top = surface["top_level"]
    for old, new in R.CLASS_RENAMES.items():
        assert new in top["present_in_v5"], new
    removed_extractors = {n for n in top["removed"] if n.endswith("FeatureExtractor")}
    assert set(R.REMOVED_FEATURE_EXTRACTORS) <= removed_extractors
    # removed without an *ImageProcessor twin: deliberately not in the rule (no replacement to suggest)
    assert removed_extractors - set(R.REMOVED_FEATURE_EXTRACTORS) == set(R.REMOVED_NO_REPLACEMENT)
    for name in R.REMOVED_FEATURE_EXTRACTORS:
        assert name.replace("FeatureExtractor", "ImageProcessor") in top["present_in_v5"], name
    # TV5R11: every removed *ImageProcessorFast name has a plain twin that exists in v5
    fast = [n for n in top["removed"] if n.endswith("ImageProcessorFast")]
    assert len(fast) > 50
    for name in fast:
        assert R.class_rename(name) == ("TV5R11", name.removesuffix("Fast")), name
        assert name.removesuffix("Fast") in top["present_in_v5"], name
    assert "AutoModelWithLMHead" in top["removed"]
    assert any(n.startswith("TF") for n in top["removed"]) and any(n.startswith("Flax") for n in top["removed"])
    assert not any(n.startswith(("TF", "Flax")) for n in top["present_in_v5"])


def test_hub_requirement_and_chat_template_default_match_the_snapshot(surface):
    assert surface["hub_requirement"].replace("huggingface-hub", "").split(",").__len__() == 2
    assert set(str(_HUB).split(",")) == set(surface["hub_requirement"].replace("huggingface-hub", "").split(","))
    assert surface["apply_chat_template_return_dict_default"] is True


def test_every_rule_is_documented_and_verified():
    assert len(RULES) == 34
    for rule_id, rule in RULES.items():
        assert rule.id == rule_id
        assert rule.verified_in == "5.23.0"
        assert rule.doc_url.startswith("https://")
        assert rule.kind in R.KINDS and rule.severity in R.SEVERITIES and rule.source in ("guide", "probe", "both")


def test_rule_ids_follow_their_kind_prefix():
    prefix = {"fix": "TV5R", "manual": "TV5M", "behavior": "TV5B", "dependency": "TV5D", "parse": "TV5P"}
    for rule in RULES.values():
        assert rule.id.startswith(prefix[rule.kind])


def test_selector_validation():
    assert R.validate_selectors(["TV5R", "TV5M01", "TV5"]) == []
    assert R.validate_selectors(["TV5X", "nope"]) == ["TV5X", "nope"]
    assert R.is_enabled("TV5R03", ("TV5R",), None) and not R.is_enabled("TV5M01", ("TV5R",), None)
    assert not R.is_enabled("TV5R03", None, ("TV5R03",))


def test_refresh_script_builds_the_same_snapshot_from_dumps(tmp_path, surface):
    """The maintainer script's pure function reproduces the evidence shape the rules rely on."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("refresh_surface", Path(__file__).parent.parent / "scripts" / "refresh_surface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v4 = {"version": "4.0", "TrainingArguments": ["a", "b"], "tokenizer_methods": ["x", "y"], "pipeline_tasks": ["p", "q"],
          "top_level": ["TFThing", "ViTFeatureExtractor", "Keep"]}
    v5 = {"version": "5.0", "TrainingArguments": ["a"], "tokenizer_methods": ["x"], "pipeline_tasks": ["p"],
          "top_level": ["ViTImageProcessor", "Keep", "AutoModelForImageTextToText"], "hub_requirement": "huggingface-hub>=1"}
    snap = module.build_snapshot(v4, v5)
    assert snap["training_arguments"] == {"v4": ["a", "b"], "v5": ["a"]}
    assert snap["tokenizer_methods"]["removed"] == ["y"] and snap["pipeline_tasks"]["removed"] == ["q"]
    assert snap["top_level"]["removed"] == ["TFThing", "ViTFeatureExtractor"]
    assert "ViTImageProcessor" in snap["top_level"]["present_in_v5"]
    assert set(surface) == set(snap)
