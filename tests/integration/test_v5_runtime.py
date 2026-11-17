"""AC-9: against a real transformers 5.23.0 the original code fails and the migrated code works.

Run with::

    uv run --with "transformers==5.23.0" pytest -q -m integration tests/integration

transformers is imported lazily so that the default test run (which deselects these
tests) never needs it. No torch, no model download, no network at test time.
"""

import pytest

from transformers_v5_migrate import migrate_source

pytestmark = pytest.mark.integration

ARGS_SNIPPET = """\
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir=OUT,
    no_cuda=True,
    per_gpu_train_batch_size=8,
    warmup_ratio=0.1,
)
"""

TOKENIZER_SNIPPET = """\
import transformers
enc = tok.encode_plus("hello world")
"""


@pytest.fixture(scope="module")
def transformers_v5():
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "5.23.0", f"integration tests are pinned to 5.23.0, found {transformers.__version__}"
    return transformers


@pytest.fixture
def tok(transformers_v5, tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]) + "\n")
    return transformers_v5.BertTokenizer(str(vocab))


def test_original_training_arguments_fail_on_v5(transformers_v5, tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        exec(ARGS_SNIPPET, {"OUT": str(tmp_path)})


def test_migrated_training_arguments_work_on_v5(transformers_v5, tmp_path):
    migrated = migrate_source(ARGS_SNIPPET)
    assert {f.rule for f in migrated.fixes} == {"TV5R05", "TV5R06"}
    namespace = {"OUT": str(tmp_path)}
    exec(migrated.code, namespace)
    args = namespace["args"]
    assert args.use_cpu is True
    assert args.per_device_train_batch_size == 8
    assert args.warmup_steps == 0.1


def test_original_encode_plus_fails_on_v5(tok):
    with pytest.raises(AttributeError, match="encode_plus"):
        exec(TOKENIZER_SNIPPET, {"tok": tok})


def test_migrated_tokenizer_call_works_on_v5(tok, transformers_v5):
    migrated = migrate_source(TOKENIZER_SNIPPET)
    assert [f.rule for f in migrated.fixes] == ["TV5R08"]
    namespace = {"tok": tok}
    exec(migrated.code, namespace)
    assert isinstance(namespace["enc"], transformers_v5.BatchEncoding)
    assert namespace["enc"]["input_ids"][0] == tok.cls_token_id


def test_dropped_include_inputs_false_is_accepted_by_v5(transformers_v5, tmp_path):
    code = "from transformers import TrainingArguments\nargs = TrainingArguments(output_dir=OUT, include_inputs_for_metrics=True)\n"
    namespace = {"OUT": str(tmp_path)}
    exec(migrate_source(code).code, namespace)
    assert namespace["args"].include_for_metrics == ["inputs"]


def test_removed_names_reported_by_the_tool_are_really_gone(transformers_v5):
    for name in ("AutoModelWithLMHead", "AutoModelForVision2Seq", "TFAutoModel", "CLIPFeatureExtractor"):
        assert not hasattr(transformers_v5, name), name
    for name in ("AutoModelForImageTextToText", "CLIPImageProcessor"):
        assert hasattr(transformers_v5, name), name
