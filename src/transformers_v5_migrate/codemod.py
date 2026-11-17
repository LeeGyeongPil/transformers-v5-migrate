"""LibCST based rewriting and checking of Python source (the heart of the tool).

Two passes over the same tree:

1. ``_Prepass`` learns facts about the file: does it import transformers at all,
   which classes in the file subclass ``Trainer`` / ``TrainingArguments``, and
   which variables hold a ``Trainer``.
2. ``_V5`` rewrites what can be rewritten mechanically and reports the rest.

Names are resolved with LibCST's ``QualifiedNameProvider`` so that
``from transformers import Trainer as HFTrainer`` and ``import transformers as tf``
(``tf.Seq2SeqTrainer``) are followed, while ``MyThing(tokenizer=...)`` or
``trl.SFTTrainer(tokenizer=...)`` are left alone.

Inside a transformer the metadata is always read with ``self.get_metadata``;
calling ``wrapper.resolve`` while visiting raises ``LazyValue`` errors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, QualifiedNameProvider, QualifiedNameSource

from . import rules as R
from .model import Finding, Fix, ParseError, Result

_RENDER = cst.Module([])
_TF_OR_FLAX = re.compile(r"^(TF|Flax)[A-Z]")
_MODEL_CLASS = re.compile(r"AutoModel\w*|\w+For\w+|\w+Model")
_ENV_CALLS = {"os.getenv", "os.putenv", "os.environ.get", "os.environ.setdefault"}


def _code(node: cst.CSTNode) -> str:
    return _RENDER.code_for_node(node)


def _dotted(node: cst.CSTNode | None) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr.value}"
    return None


def _str_value(node: cst.CSTNode) -> str | None:
    """The value of a plain (non f-string) string literal, else None."""
    if isinstance(node, cst.SimpleString):
        try:
            value = node.evaluated_value
        except Exception:  # noqa: BLE001 - exotic prefixes are simply not matched
            return None
        return value if isinstance(value, str) else None
    return None


def _is_name(node: cst.CSTNode, value: str) -> bool:
    return isinstance(node, cst.Name) and node.value == value


def _tf_last(qnames: Iterable[Any]) -> set[str]:
    """Last path segment of every qualified name that lives in ``transformers``."""
    return {q.name.rsplit(".", 1)[-1] for q in qnames if q.name.startswith("transformers.")}


def _local(qnames: Iterable[Any]) -> set[str]:
    return {q.name for q in qnames if q.source == QualifiedNameSource.LOCAL}


@dataclass
class _Facts:
    has_import: bool = False
    trainer_like: set[str] = field(default_factory=set)
    args_like: set[str] = field(default_factory=set)
    trainer_vars: set[str] = field(default_factory=set)

    def classify(self, qnames: Iterable[Any]) -> str | None:
        qnames = list(qnames)
        last, local = _tf_last(qnames), _local(qnames)
        if last & R.TRAINER_CLASSES or local & self.trainer_like:
            return "trainer"
        if last & R.ARGS_CLASSES or local & self.args_like:
            return "args"
        return None


class _Prepass(cst.CSTTransformer):
    """Collect facts; never changes the tree."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.facts = _Facts()
        self._classes: list[tuple[str, list[list[Any]]]] = []
        self._assigns: list[tuple[str, list[Any]]] = []

    def _qn(self, node: cst.CSTNode) -> list[Any]:
        return list(self.get_metadata(QualifiedNameProvider, node, set()))

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if (_dotted(alias.name) or "").split(".")[0] == "transformers":
                self.facts.has_import = True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if not node.relative and (_dotted(node.module) or "").split(".")[0] == "transformers":
            self.facts.has_import = True

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._classes.append((node.name.value, [self._qn(b.value) for b in node.bases]))

    def visit_Assign(self, node: cst.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.value, cst.Call):
            self._assigns.append((_code(node.targets[0].target), self._qn(node.value.func)))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        facts = self.facts
        changed = True
        while changed:  # subclasses of subclasses
            changed = False
            for name, bases in self._classes:
                for base in bases:
                    last, local = _tf_last(base), _local(base)
                    if (last & R.TRAINER_CLASSES or local & facts.trainer_like) and name not in facts.trainer_like:
                        facts.trainer_like.add(name)
                        changed = True
                    if (last & R.ARGS_CLASSES or local & facts.args_like) and name not in facts.args_like:
                        facts.args_like.add(name)
                        changed = True
        for target, func_qn in self._assigns:
            if facts.classify(func_qn) == "trainer":
                facts.trainer_vars.add(target)
        return updated_node


class _V5(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (QualifiedNameProvider, PositionProvider)

    def __init__(self, path: str, facts: _Facts, select: tuple[str, ...] | None, ignore: tuple[str, ...] | None) -> None:
        super().__init__()
        self.path = path
        self.facts = facts
        self.select, self.ignore = select, ignore
        self.fixes: list[Fix] = []
        self.findings: list[Finding] = []
        self._skip_names: set[int] = set()

    # ---- helpers ----------------------------------------------------------------
    def _qn(self, node: cst.CSTNode) -> list[Any]:
        return list(self.get_metadata(QualifiedNameProvider, node, set()))

    def _pos(self, node: cst.CSTNode) -> tuple[int, int]:
        start = self.get_metadata(PositionProvider, node).start
        return start.line, start.column + 1

    def _on(self, rule: str) -> bool:
        return R.is_enabled(rule, self.select, self.ignore)

    def _fix(self, rule: str, node: cst.CSTNode, before: str, after: str) -> None:
        line, col = self._pos(node)
        self.fixes.append(Fix(rule, self.path, line, col, before, after))

    def _find(self, rule: str, node: cst.CSTNode, key: str | None = None, fixable: bool = False, **params: Any) -> None:
        if not self._on(rule):
            return
        spec = R.RULES[rule]
        kind = "manual" if spec.kind in ("fix", "parse") else spec.kind
        line, col = self._pos(node)
        self.findings.append(
            Finding(rule, spec.severity, kind, self.path, line, col, key or f"msg.{rule}", params, None, fixable)
        )

    def _call_anchor(self, call: cst.Call) -> cst.CSTNode:
        return call.func.attr if isinstance(call.func, cst.Attribute) else call.func

    # ---- calls ------------------------------------------------------------------
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        func = original_node.func
        qn = self._qn(func)
        kind = self.facts.classify(qn)
        attr = func.attr.value if isinstance(func, cst.Attribute) else None
        is_pipeline = "pipeline" in _tf_last(qn)
        existing = {a.keyword.value for a in original_node.args if a.keyword}
        star_kwargs = any(a.star == "**" for a in original_node.args)
        new_args = list(updated_node.args)
        dropped: set[int] = set()

        def rename(i: int, rule: str, new: str) -> None:
            arg, old = original_node.args[i], original_node.args[i].keyword.value  # type: ignore[union-attr]
            if not self._on(rule):
                return
            if new in existing:
                self._find(rule, arg, "msg.dup", old=old, new=new)
                return
            new_args[i] = new_args[i].with_changes(keyword=cst.Name(new))
            self._fix(rule, arg, old, new)

        for i, arg in enumerate(original_node.args):
            if arg.keyword is None:
                continue
            kw = arg.keyword.value
            if attr == "from_pretrained" and kw == "torch_dtype":
                rename(i, "TV5R01", "dtype")
            elif kw == "use_auth_token" and (attr in R.TOKEN_RENAME_METHODS or is_pipeline):
                rename(i, "TV5R02", "token")
            elif kind == "trainer" and kw == "tokenizer":
                rename(i, "TV5R03", "processing_class")
            elif attr == "train" and kw == "model_path":
                if _code(func.value) in self.facts.trainer_vars:  # type: ignore[union-attr]
                    rename(i, "TV5R04", "resume_from_checkpoint")
                else:
                    self._find("TV5R04", arg)
            elif kind == "args" and kw in R.ARGS_RENAMES:
                rename(i, "TV5R05", R.ARGS_RENAMES[kw])
            elif kind == "args" and kw == "warmup_ratio":
                self._warmup_ratio(i, arg, new_args, existing)
            elif kind == "args" and kw == "include_inputs_for_metrics":
                self._include_inputs(i, arg, new_args, existing, dropped)
            elif kind == "args" and kw in R.ARGS_REMOVED:
                self._find("TV5M06", arg, name=kw, hint=R.ARGS_REMOVED[kw])
            elif attr == "from_pretrained" and kw in ("load_in_8bit", "load_in_4bit"):
                self._find("TV5M01", arg, name=kw)
            elif attr == "from_pretrained" and kw in ("from_tf", "from_flax"):
                self._find("TV5M02", arg, name=kw)
            elif is_pipeline and kw == "task":
                task = _str_value(arg.value)
                if task in R.REMOVED_PIPELINE_TASKS:
                    self._find("TV5M05", arg, task=task)

        if is_pipeline and original_node.args and original_node.args[0].keyword is None:
            task = _str_value(original_node.args[0].value)
            if task in R.REMOVED_PIPELINE_TASKS:
                self._find("TV5M05", original_node.args[0], task=task)

        if _dotted(func) is not None and any(q.name in _ENV_CALLS for q in qn) and original_node.args:
            self._env_string(original_node.args[0].value)

        self._behavior(original_node, attr, qn, existing, star_kwargs)

        if dropped:
            kept = [a for i, a in enumerate(new_args) if i not in dropped]
            last = len(original_node.args) - 1
            if kept and last in dropped:  # do not leave a dangling comma behind
                kept[-1] = kept[-1].with_changes(comma=original_node.args[last].comma)
            new_args = kept

        result = updated_node.with_changes(args=new_args)
        if attr in ("encode_plus", "batch_encode_plus") and self._on("TV5R08"):
            assert isinstance(updated_node.func, cst.Attribute)
            self._fix("TV5R08", func.attr, _code(func), _code(func.value))  # type: ignore[union-attr]
            result = result.with_changes(func=updated_node.func.value)
        return result

    def _warmup_ratio(self, i: int, arg: cst.Arg, new_args: list[cst.Arg], existing: set[str]) -> None:
        if not self._on("TV5R06"):
            return
        value = arg.value
        number = value.evaluated_value if isinstance(value, (cst.Integer, cst.Float)) else None
        if number is None or not 0 <= number < 1:
            self._find("TV5M06", arg, name="warmup_ratio", hint="warmup_ratio")
        elif "warmup_steps" in existing:
            self._find("TV5M06", arg, name="warmup_ratio", hint="warmup_steps_conflict")
        else:
            new_args[i] = new_args[i].with_changes(keyword=cst.Name("warmup_steps"))
            self._fix("TV5R06", arg, f"warmup_ratio={_code(value)}", f"warmup_steps={_code(value)}")

    def _include_inputs(
        self, i: int, arg: cst.Arg, new_args: list[cst.Arg], existing: set[str], dropped: set[int]
    ) -> None:
        if not self._on("TV5R07"):
            return
        value = arg.value
        if "include_for_metrics" in existing or not (_is_name(value, "True") or _is_name(value, "False")):
            self._find("TV5M06", arg, name="include_inputs_for_metrics", hint="include_inputs_for_metrics")
        elif _is_name(value, "True"):
            inputs = cst.List([cst.Element(cst.SimpleString('"inputs"'))])
            new_args[i] = new_args[i].with_changes(keyword=cst.Name("include_for_metrics"), value=inputs)
            self._fix("TV5R07", arg, "include_inputs_for_metrics=True", 'include_for_metrics=["inputs"]')
        else:
            dropped.add(i)
            self._fix("TV5R07", arg, "include_inputs_for_metrics=False", "(removed, False is the default)")

    def _behavior(self, call: cst.Call, attr: str | None, qn: list[Any], existing: set[str], star_kwargs: bool) -> None:
        anchor = self._call_anchor(call)
        if attr == "from_pretrained" and not star_kwargs and not ({"dtype", "torch_dtype"} & existing):
            for q in qn:
                parts = q.name.split(".")
                if (
                    q.name.startswith("transformers.")
                    and len(parts) >= 3
                    and _MODEL_CLASS.fullmatch(parts[-2])
                    and not _TF_OR_FLAX.match(parts[-2])  # already reported as removed (TV5M03)
                ):
                    self._find("TV5B01", anchor)
                    break
        elif attr == "apply_chat_template" and not star_kwargs and "return_dict" not in existing:
            tokenize = next((a.value for a in call.args if a.keyword and a.keyword.value == "tokenize"), None)
            if tokenize is None or not _is_name(tokenize, "False"):
                self._find("TV5B02", anchor)

    # ---- attributes, names, imports, strings -------------------------------------
    def visit_Attribute(self, node: cst.Attribute) -> None:
        self._skip_names.add(id(node.attr))

    def visit_ImportAlias(self, node: cst.ImportAlias) -> None:
        self._skip_names.add(id(node.name))

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        attr = original_node.attr.value
        if attr in R.REMOVED_TOKENIZER_NAMES:
            self._find("TV5M07", original_node.attr, name=attr)
        names = {q.name for q in self._qn(original_node) if q.name.startswith("transformers.")}
        for name in names:
            self._removed_name(name.rsplit(".", 1)[-1], original_node.attr)
        renamed = R.class_rename(attr)
        if renamed and f"transformers.{attr}" in names and self._on(renamed[0]):
            self._fix(renamed[0], original_node.attr, attr, renamed[1])
            return updated_node.with_changes(attr=cst.Name(renamed[1]))
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
        old = original_node.value
        renamed = R.class_rename(old)
        if renamed and id(original_node) not in self._skip_names and self._on(renamed[0]):
            if any(q.name == f"transformers.{old}" for q in self._qn(original_node)):
                self._fix(renamed[0], original_node, old, renamed[1])
                return updated_node.with_changes(value=renamed[1])
        return updated_node

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.BaseSmallStatement:
        module = _dotted(original_node.module) or ""
        if original_node.relative or (module != "transformers" and not module.startswith("transformers.")):
            return updated_node
        if isinstance(original_node.names, cst.ImportStar):
            return updated_node
        aliases = list(updated_node.names)
        for i, alias in enumerate(original_node.names):
            name = _dotted(alias.name) or ""
            self._removed_name(name, alias)
            renamed = R.class_rename(name)
            if module == "transformers" and renamed and self._on(renamed[0]):
                self._fix(renamed[0], alias, name, renamed[1])
                aliases[i] = aliases[i].with_changes(name=cst.Name(renamed[1]))
        return updated_node.with_changes(names=aliases)

    def _removed_name(self, name: str, anchor: cst.CSTNode) -> None:
        if name == "AutoModelWithLMHead":
            self._find("TV5M04", anchor)
        elif _TF_OR_FLAX.match(name) or name in R.REMOVED_NO_REPLACEMENT:
            self._find("TV5M03", anchor, name=name)

    def leave_Subscript(self, original_node: cst.Subscript, updated_node: cst.Subscript) -> cst.BaseExpression:
        if any(q.name == "os.environ" for q in self._qn(original_node.value)) and len(original_node.slice) == 1:
            index = original_node.slice[0].slice
            if isinstance(index, cst.Index):
                self._env_string(index.value)
        return updated_node

    def _env_string(self, node: cst.CSTNode) -> None:
        value = _str_value(node)
        if value in R.REMOVED_ENV_VARS:
            self._find("TV5M08", node, name=value)


def _normalize(values: Iterable[str] | None) -> tuple[str, ...] | None:
    return tuple(values) if values else None


def _parse(code: str, path: str) -> cst.Module:
    try:
        return cst.parse_module(code)
    except cst.ParserSyntaxError as exc:
        raise ParseError(path, exc.raw_line, exc.message) from exc
    except Exception as exc:  # noqa: BLE001 - LibCST can raise other errors on unusual input
        raise ParseError(path, 1, str(exc)) from exc


def analyze_module(
    module: cst.Module,
    code: str,
    path: str,
    select: tuple[str, ...] | None,
    ignore: tuple[str, ...] | None,
) -> Result:
    """Run both passes over an already parsed module."""
    wrapper = MetadataWrapper(module)
    prepass = _Prepass()
    wrapper.visit(prepass)
    if not prepass.facts.has_import:
        return Result(code=code)
    transformer = _V5(path, prepass.facts, select, ignore)
    new_module = wrapper.visit(transformer)
    transformer.fixes.sort(key=lambda f: (f.line, f.col))
    transformer.findings.sort(key=lambda f: (f.line, f.col, f.rule))
    new_code = new_module.code if transformer.fixes else code
    return Result(new_code, transformer.fixes, transformer.findings)


def migrate_source(
    code: str,
    *,
    path: str = "<string>",
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
) -> Result:
    """Migrate one Python source string.

    Returns the new code plus the automatic fixes and the findings that need a
    human. Code that does not import ``transformers`` is returned unchanged.
    Raises :class:`ParseError` when the source cannot be parsed.
    """
    return analyze_module(_parse(code, path), code, path, _normalize(select), _normalize(ignore))
