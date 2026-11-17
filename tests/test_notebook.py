"""AC-4: notebooks keep magics, outputs and metadata; broken cells are reported, not fatal."""

import json

import pytest

from transformers_v5_migrate import ParseError, migrate_notebook


def _nb(*cells, source_as_list=True):
    out = []
    for kind, text in cells:
        lines = text.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {"tags": ["keep"]}, "source": lines if source_as_list else text}
        if kind == "code":
            cell["execution_count"] = 3
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["out\n"]}]
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def _dumps(nb, indent=1):
    return json.dumps(nb, indent=indent, ensure_ascii=False) + "\n"


MAGIC_CELL = "!pip install -q transformers==4.57.1 \\\n    datasets\n%env TRANSFORMERS_CACHE=/content/c\n"
CODE_CELL = (
    "from transformers import Trainer\n"
    "import torch\n"
    "if True:\n"
    "    %time x = 1\n"
    "    trainer = Trainer(model=m, tokenizer=tok)\n"
    "    !nvidia-smi\n"
    "files = !ls\n"
)


def test_magic_lines_restored():
    raw = _dumps(_nb(("markdown", "# title\n"), ("code", MAGIC_CELL), ("code", CODE_CELL)))
    result = migrate_notebook(raw, path="nb.ipynb")
    new = json.loads(result.code)
    assert new["cells"][2]["source"] == [
        "from transformers import Trainer\n",
        "import torch\n",
        "if True:\n",
        "    %time x = 1\n",
        "    trainer = Trainer(model=m, processing_class=tok)\n",
        "    !nvidia-smi\n",
        "files = !ls\n",
    ]
    assert new["cells"][1]["source"] == MAGIC_CELL.splitlines(keepends=True)  # untouched cell
    assert [(f.rule, f.cell, f.line) for f in result.fixes] == [("TV5R03", 3, 5)]


def test_outputs_metadata_unchanged():
    nb = _nb(("markdown", "# t\n"), ("code", CODE_CELL), ("code", "print('x')"))
    raw = _dumps(nb)
    new = json.loads(migrate_notebook(raw).code)
    for old_cell, new_cell in zip(nb["cells"], new["cells"]):
        assert {k: v for k, v in old_cell.items() if k != "source"} == {k: v for k, v in new_cell.items() if k != "source"}
    assert new["metadata"] == nb["metadata"]
    assert [c["cell_type"] for c in new["cells"]] == ["markdown", "code", "code"]
    assert new["cells"][2]["source"] == ["print('x')"]  # no trailing newline invented


def test_unparsable_cell_reported():
    broken = "from transformers import Trainer\nx = (\n"
    raw = _dumps(_nb(("code", broken), ("code", CODE_CELL)))
    result = migrate_notebook(raw, path="nb.ipynb")
    parse = [f for f in result.findings if f.rule == "TV5P01"]
    assert [(f.cell, f.severity) for f in parse] == [(1, "error")]
    new = json.loads(result.code)
    assert new["cells"][0]["source"] == broken.splitlines(keepends=True)
    assert "processing_class=tok" in "".join(new["cells"][1]["source"])


def test_transformers_cli_and_env_magic_reported_with_cell_and_line():
    raw = _dumps(_nb(("code", "%env TRANSFORMERS_CACHE=/c\n!transformers-cli env\n")))
    result = migrate_notebook(raw)
    assert [(f.rule, f.cell, f.line) for f in result.findings] == [("TV5M08", 1, 1), ("TV5M10", 1, 2)]
    assert result.code == raw  # report only, no rewrite


def test_cell_magic_is_scanned_as_text_not_parsed():
    raw = _dumps(_nb(("code", "%%bash\ntransformers-cli login\n")))
    result = migrate_notebook(raw)
    assert [(f.rule, f.cell, f.line) for f in result.findings] == [("TV5M10", 1, 2)]


def test_nothing_to_fix_is_byte_identical():
    raw = _dumps(_nb(("code", "import transformers\nx = 1\n")), indent=2)
    assert migrate_notebook(raw).code == raw


def test_import_in_an_earlier_cell_is_visible_later():
    raw = _dumps(_nb(("code", "import transformers as tf\n"), ("code", "t = tf.Trainer(model=m, tokenizer=tok)\n")))
    new = json.loads(migrate_notebook(raw).code)
    assert new["cells"][1]["source"] == ["t = tf.Trainer(model=m, processing_class=tok)\n"]


@pytest.mark.parametrize("indent", [1, 2, 4])
def test_json_layout_is_kept(indent):
    raw = _dumps(_nb(("code", CODE_CELL)), indent=indent)
    out = migrate_notebook(raw).code
    assert out.endswith("\n")
    assert out.splitlines()[1].startswith(" " * indent + '"')
    assert not out.splitlines()[1].startswith(" " * (indent + 1))


def test_string_source_form_is_kept():
    raw = _dumps(_nb(("code", "from transformers import Trainer\nTrainer(tokenizer=t)\n"), source_as_list=False))
    new = json.loads(migrate_notebook(raw).code)
    assert new["cells"][0]["source"] == "from transformers import Trainer\nTrainer(processing_class=t)\n"


def test_not_a_notebook_raises():
    with pytest.raises(ParseError):
        migrate_notebook("{not json", path="x.ipynb")
    with pytest.raises(ParseError):
        migrate_notebook('{"nbformat": 4}', path="x.ipynb")
