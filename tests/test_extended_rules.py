"""Second-generation rules (M2): tokenizer/image-processor renames, Hub errors, config files, conda, --explain."""

import importlib.util
import json
from pathlib import Path

import pytest

from transformers_v5_migrate import RULES, migrate_source, scan_paths
from transformers_v5_migrate.cli import main
from transformers_v5_migrate.configscan import scan_config
from transformers_v5_migrate.deps import scan_environment

ROOT = Path(__file__).parent.parent


def test_additional_special_tokens_renamed():
    code = "from transformers import AutoTokenizer\nt = AutoTokenizer.from_pretrained('x', additional_special_tokens=['<a>'])\n"
    result = migrate_source(code)
    assert "extra_special_tokens=['<a>']" in result.code
    assert [f.rule for f in result.fixes] == ["TV5R10"]


def test_fast_image_processor_names_renamed_everywhere():
    code = (
        "import transformers as tf\n"
        "from transformers import CLIPImageProcessorFast, ViTImageProcessorFast as V\n"
        "a = CLIPImageProcessorFast.from_pretrained('x')\n"
        "b = tf.DetrImageProcessorFast()\n"
        "c = V()\n"
        "name = 'CLIPImageProcessorFast'\n"
    )
    result = migrate_source(code)
    assert "from transformers import CLIPImageProcessor, ViTImageProcessor as V\n" in result.code
    assert "a = CLIPImageProcessor.from_pretrained('x')" in result.code
    assert "b = tf.DetrImageProcessor()" in result.code
    assert "c = V()" in result.code and "name = 'CLIPImageProcessorFast'" in result.code
    assert {f.rule for f in result.fixes} == {"TV5R11"}


def test_use_fast_for_image_processors_becomes_backend():
    code = (
        "from transformers import AutoImageProcessor, CLIPImageProcessor\n"
        "a = AutoImageProcessor.from_pretrained('x', use_fast=True)\n"
        "b = CLIPImageProcessor.from_pretrained('x', use_fast=False)\n"
        "c = AutoImageProcessor.from_pretrained('x', use_fast=flag)\n"
        "d = AutoImageProcessor.from_pretrained('x', use_fast=True, backend='pil')\n"
    )
    result = migrate_source(code)
    assert "a = AutoImageProcessor.from_pretrained('x', backend=\"torchvision\")" in result.code
    assert "b = CLIPImageProcessor.from_pretrained('x', backend=\"pil\")" in result.code
    assert "use_fast=flag" in result.code and "use_fast=True, backend='pil'" in result.code
    assert [(f.rule, f.line, f.message_key) for f in result.findings] == [
        ("TV5R12", 4, "msg.TV5R12.manual"),
        ("TV5R12", 5, "msg.TV5R12.manual"),
    ]


def test_use_fast_for_tokenizers_is_a_note_and_untouched():
    code = "from transformers import AutoTokenizer\nt = AutoTokenizer.from_pretrained('x', use_fast=False)\n"
    result = migrate_source(code)
    assert result.code == code
    assert [(f.rule, f.severity, f.kind) for f in result.findings] == [("TV5B05", "note", "behavior")]


def test_requests_errors_around_hub_calls():
    code = (
        "import requests\n"
        "from requests import HTTPError\n"
        "from transformers import AutoModel\n"
        "try:\n"
        "    AutoModel.from_pretrained('x')\n"
        "except requests.HTTPError:\n"
        "    pass\n"
        "try:\n"
        "    AutoModel.from_pretrained('x')\n"
        "except (ValueError, HTTPError):\n"
        "    pass\n"
        "try:\n"
        "    print('no hub call here')\n"
        "except requests.HTTPError:\n"
        "    pass\n"
        "try:\n"
        "    AutoModel.from_pretrained('x')\n"
        "except OSError:\n"
        "    pass\n"
    )
    result = migrate_source(code)
    assert [(f.rule, f.line) for f in result.findings if f.rule == "TV5M13"] == [("TV5M13", 6), ("TV5M13", 10)]


def test_training_config_keys_in_json_and_yaml(tmp_path):
    (tmp_path / "train.json").write_text('{\n  "output_dir": "out",\n  "no_cuda": true,\n  "warmup_ratio": 0.1,\n  "logging_dir": "x",\n  "use_auth_token": true\n}\n')
    (tmp_path / "train.yaml").write_text("output_dir: out\nper_gpu_train_batch_size: 8\n# group_by_length: true\ngroup_by_length: true\n")
    (tmp_path / "config.json").write_text('{"model_type": "bert", "torch_dtype": "float32", "use_auth_token": null}\n')
    (tmp_path / "package.json").write_text('{"name": "x", "no_cuda": 1}\n')
    report = scan_paths([str(tmp_path)])
    found = sorted((Path(f.path).name, f.line, f.params["name"]) for f in report.findings if f.rule == "TV5M14")
    assert found == [
        ("train.json", 3, "no_cuda"),
        ("train.json", 4, "warmup_ratio"),
        ("train.json", 5, "logging_dir"),
        ("train.json", 6, "use_auth_token"),
        ("train.yaml", 2, "per_gpu_train_batch_size"),
        ("train.yaml", 4, "group_by_length"),
    ]


def test_config_scan_ignores_invalid_json_and_huge_files(tmp_path):
    assert scan_config("{not json", "x.json", "config") == []
    (tmp_path / "big.json").write_text('{"no_cuda": true, "pad": "' + "x" * 2_100_000 + '"}')
    assert scan_paths([str(tmp_path / "big.json")]).files_scanned == 0


def test_config_message_is_translated_with_the_new_name(capsys, tmp_path):
    (tmp_path / "t.yaml").write_text("output_dir: o\nno_cuda: true\n")
    main([str(tmp_path), "--lang", "ko", "--fail-on", "never", "--format", "json"])
    message = json.loads(capsys.readouterr().out)["findings"][0]["message"]
    assert "use_cpu" in message and "HfArgumentParser" in message


@pytest.mark.parametrize(
    "line, rules",
    [
        ("  - transformers=4.57.1", ["TV5D01"]),
        ("  - conda-forge::transformers=4.57", ["TV5D01"]),
        ("  - transformers=4.57.1=pyhd8ed1ab_0", ["TV5D01"]),
        ("    - transformers==4.57.1", ["TV5D01"]),
        ("    - huggingface-hub<1", ["TV5D02"]),
        ("    - transformers>=5", []),
        ("  - transformers", []),
        ("  - numpy=1.26", []),
    ],
)
def test_environment_yml(line, rules):
    assert [f.rule for f in scan_environment(line + "\n", "environment.yml", None, None)] == rules


def test_environment_yml_found_by_the_walker(tmp_path):
    (tmp_path / "environment.yml").write_text("name: x\ndependencies:\n  - python=3.11\n  - pip:\n    - transformers==4.57.1\n")
    report = scan_paths([str(tmp_path)])
    assert [(f.rule, f.line) for f in report.findings] == [("TV5D01", 5)]


def test_explain(capsys):
    assert main(["--explain", "tv5r03"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("TV5R03  ") and "processing_class" in out and "5.23.0" in out and "MIGRATION_GUIDE_V5" in out
    assert "before" in out and "after" in out
    main(["--explain", "TV5M06", "--lang", "ko"])
    ko = capsys.readouterr().out
    assert "<name>" in ko and "권장 수정" in ko
    with pytest.raises(SystemExit) as info:
        main(["--explain", "TV5X99"])
    assert info.value.code == 2


def test_explain_works_for_every_rule(capsys):
    for rule_id in RULES:
        assert main(["--explain", rule_id]) == 0
        assert rule_id in capsys.readouterr().out


def test_every_fix_rule_example_is_really_fixed_by_the_tool():
    """The before/after shown by --explain must be what the codemod does."""
    from transformers_v5_migrate import rules as R

    header = (
        "from transformers import AutoModel, AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, "
        "CLIPImageProcessorFast, Trainer, TrainingArguments\n"
        "name = tok = path = tokens = model = args = d = None\n"
        "trainer = Trainer(model=model)\n"
    )
    for rule_id, (before, after) in R.EXAMPLES.items():
        result = migrate_source(header + before + "\n")
        assert result.code.splitlines()[3:] == [after], rule_id  # the 3 header lines get renamed imports
        assert rule_id in {f.rule for f in result.fixes}, rule_id


def test_pre_commit_hooks_manifest():
    yaml = pytest.importorskip("yaml")
    hooks = yaml.safe_load((ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8"))
    assert [h["id"] for h in hooks] == ["transformers-v5-migrate", "transformers-v5-migrate-fix"]
    for hook in hooks:
        assert hook["language"] == "python" and hook["entry"].startswith("transformers-v5-migrate")
    assert "--write" in hooks[1]["entry"] and "--write" not in hooks[0]["entry"]


def test_surface_check_reports_new_removals():
    spec = importlib.util.spec_from_file_location("refresh_surface", ROOT / "scripts" / "refresh_surface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = json.loads((ROOT / "src" / "transformers_v5_migrate" / "data" / "v5_surface.json").read_text(encoding="utf-8"))
    assert module.compare_snapshots(committed, committed) == []
    changed = json.loads(json.dumps(committed))
    changed["training_arguments"]["v5"].remove("use_cpu")
    changed["pipeline_tasks"]["removed"].append("text-classification")
    changed["hub_requirement"] = "huggingface-hub>=2"
    problems = module.compare_snapshots(committed, changed)
    assert len(problems) == 3
    assert any("use_cpu" in p for p in problems) and any("text-classification" in p for p in problems)
