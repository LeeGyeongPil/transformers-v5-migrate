"""AC-1: alias-aware automatic fixes."""

import pytest

from transformers_v5_migrate import ParseError, migrate_source


def test_sample_matches_snapshot(fixtures, snapshots):
    source = (fixtures / "sample_v4.py").read_text(encoding="utf-8")
    result = migrate_source(source, path="sample_v4.py")
    expected = (snapshots / "sample_v5.py").read_text(encoding="utf-8")
    assert result.code == expected
    assert result.findings == []
    assert {f.rule for f in result.fixes} == {
        "TV5R01", "TV5R02", "TV5R03", "TV5R04", "TV5R05", "TV5R06", "TV5R07", "TV5R08", "TV5R09",
    }


def test_non_trainer_tokenizer_kwarg_untouched(fixtures):
    source = (fixtures / "sample_v4.py").read_text(encoding="utf-8")
    new = migrate_source(source).code
    # look-alikes and text stay exactly as written, the comment survives
    assert "other = MyThing(tokenizer=tok)  # not a Trainer: must stay unchanged" in new
    assert "sft = trl.SFTTrainer(model=model, tokenizer=tok)" in new
    assert 'text = "tokenizer=tok, torch_dtype=x"' in new
    assert "torch_dtype=torch.float16" not in new
    assert "dtype=torch.float16,  # keep this comment" in new


@pytest.mark.parametrize(
    "import_line, call",
    [
        ("from transformers import Trainer as HFTrainer", "HFTrainer"),
        ("import transformers as tf", "tf.Trainer"),
        ("import transformers", "transformers.Trainer"),
        ("from transformers.trainer import Trainer", "Trainer"),
        ("from transformers import Seq2SeqTrainer", "Seq2SeqTrainer"),
    ],
)
def test_trainer_aliases_are_followed(import_line, call):
    new = migrate_source(f"{import_line}\nt = {call}(model=m, tokenizer=tok)\n").code
    assert "processing_class=tok" in new


def test_subclass_of_subclass_is_a_trainer():
    code = (
        "from transformers import Trainer\n"
        "class A(Trainer): pass\n"
        "class B(A): pass\n"
        "b = B(tokenizer=tok)\n"
    )
    assert "B(processing_class=tok)" in migrate_source(code).code


def test_trl_subclass_is_not_a_trainer():
    code = "import trl\nfrom transformers import Trainer\nclass S(trl.SFTTrainer): pass\ns = S(tokenizer=tok)\n"
    assert migrate_source(code).code == code


def test_training_arguments_subclass_is_followed():
    code = "from transformers import TrainingArguments\nclass A(TrainingArguments): pass\nA(no_cuda=True)\n"
    assert "A(use_cpu=True)" in migrate_source(code).code


def test_train_model_path_only_for_known_trainer_variables():
    code = (
        "from transformers import Trainer\n"
        "trainer = Trainer(model=m)\n"
        "trainer.train(model_path='a')\n"
        "other.train(model_path='b')\n"
    )
    result = migrate_source(code)
    assert "trainer.train(resume_from_checkpoint='a')" in result.code
    assert "other.train(model_path='b')" in result.code
    assert [(f.rule, f.line) for f in result.findings] == [("TV5R04", 4)]


def test_warmup_ratio_only_literals_in_zero_to_one():
    code = (
        "from transformers import TrainingArguments\n"
        "a = TrainingArguments(warmup_ratio=0.1)\n"
        "b = TrainingArguments(warmup_ratio=0)\n"
        "c = TrainingArguments(warmup_ratio=1.5)\n"
        "d = TrainingArguments(warmup_ratio=r)\n"
    )
    result = migrate_source(code)
    assert "a = TrainingArguments(warmup_steps=0.1)" in result.code
    assert "b = TrainingArguments(warmup_steps=0)" in result.code
    assert "c = TrainingArguments(warmup_ratio=1.5)" in result.code
    assert "d = TrainingArguments(warmup_ratio=r)" in result.code
    assert [(f.rule, f.line) for f in result.findings] == [("TV5M06", 4), ("TV5M06", 5)]


def test_include_inputs_false_drops_argument_without_dangling_comma():
    code = "from transformers import TrainingArguments\na = TrainingArguments(output_dir='o', include_inputs_for_metrics=False)\n"
    assert "a = TrainingArguments(output_dir='o')" in migrate_source(code).code


def test_encode_plus_becomes_call_and_keeps_arguments():
    code = "from transformers import AutoTokenizer\ntok = AutoTokenizer.from_pretrained('x')\nenc = tok.encode_plus('hi', return_tensors='pt')\n"
    assert "enc = tok('hi', return_tensors='pt')" in migrate_source(code).code


def test_vision2seq_renamed_everywhere_but_not_through_other_aliases():
    code = (
        "import transformers as tf\n"
        "from transformers import AutoModelForVision2Seq\n"
        "from transformers import AutoModelForVision2Seq as V\n"
        "a = AutoModelForVision2Seq.from_pretrained('x', dtype='auto')\n"
        "b = tf.AutoModelForVision2Seq.from_pretrained('x', dtype='auto')\n"
        "c = V.from_pretrained('x', dtype='auto')\n"
    )
    new = migrate_source(code).code
    assert "from transformers import AutoModelForImageTextToText\n" in new
    assert "from transformers import AutoModelForImageTextToText as V\n" in new
    assert "a = AutoModelForImageTextToText.from_pretrained" in new
    assert "b = tf.AutoModelForImageTextToText.from_pretrained" in new
    assert "c = V.from_pretrained" in new


def test_select_and_ignore_filter_rules():
    code = "from transformers import AutoModelForCausalLM\nm = AutoModelForCausalLM.from_pretrained('x', torch_dtype=t, use_auth_token=True)\n"
    only_r02 = migrate_source(code, select=["TV5R02"])
    assert "torch_dtype=t" in only_r02.code and "token=True" in only_r02.code
    not_r01 = migrate_source(code, ignore=["TV5R01"])
    assert "torch_dtype=t" in not_r01.code and "token=True" in not_r01.code
    prefix = migrate_source(code, select=["TV5R"])
    assert "dtype=t" in prefix.code and "torch_dtype" not in prefix.code


def test_duplicate_target_is_reported_not_rewritten():
    code = "from transformers import AutoModelForCausalLM\nm = AutoModelForCausalLM.from_pretrained('x', torch_dtype=a, dtype=b)\n"
    result = migrate_source(code)
    assert result.code == code
    assert [f.rule for f in result.findings] == ["TV5R01"]


def test_code_without_transformers_import_is_returned_untouched():
    code = "tok.encode_plus('x')\nTrainer(tokenizer=t)\n"
    result = migrate_source(code)
    assert result.code == code and not result.fixes and not result.findings


def test_syntax_error_raises_parse_error_with_line():
    with pytest.raises(ParseError) as info:
        migrate_source("import transformers\nx = (\n", path="bad.py")
    assert info.value.path == "bad.py"
    assert info.value.line >= 1


def test_metadata_is_read_through_the_transformer_not_wrapper_resolve():
    """Regression for the LazyValue trap: many calls in one module must not raise."""
    body = "".join(f"t{i} = Trainer(model=m, tokenizer=tok)\n" for i in range(50))
    result = migrate_source("from transformers import Trainer\n" + body)
    assert len(result.fixes) == 50
