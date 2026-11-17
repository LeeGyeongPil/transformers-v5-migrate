"""The v4 -> v5 rule table.

Every rule carries the transformers version it was checked against
(``verified_in``) and where the evidence comes from:

* ``guide``  - the official ``MIGRATION_GUIDE_V5.md``
* ``probe``  - a real install comparison (``data/v5_surface.json``)
* ``both``   - the guide and the install comparison agree

Where the guide and the installed library disagree, the library wins; the
disagreements are listed in the README ("guide vs. reality").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_VERSION = "5.23.0"
GUIDE_URL = "https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md"
HUB_REQUIREMENT = "huggingface-hub>=1.31.0,<3.0"

KINDS = ("fix", "manual", "behavior", "dependency", "parse")
SEVERITIES = ("note", "warning", "error")  # ascending


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str  # one of KINDS
    severity: str  # one of SEVERITIES
    verified_in: str
    source: str  # "guide" | "probe" | "both"
    doc_url: str = GUIDE_URL


def _r(id: str, kind: str, severity: str, source: str) -> tuple[str, Rule]:
    return id, Rule(id, kind, severity, TARGET_VERSION, source)


RULES: dict[str, Rule] = dict(
    [
        # automatic fixes
        _r("TV5R01", "fix", "warning", "both"),
        _r("TV5R02", "fix", "error", "both"),
        _r("TV5R03", "fix", "error", "both"),
        _r("TV5R04", "fix", "error", "guide"),
        _r("TV5R05", "fix", "error", "both"),
        _r("TV5R06", "fix", "error", "probe"),
        _r("TV5R07", "fix", "error", "probe"),
        _r("TV5R08", "fix", "error", "both"),
        _r("TV5R09", "fix", "error", "probe"),
        _r("TV5R10", "fix", "warning", "both"),
        _r("TV5R11", "fix", "warning", "both"),
        _r("TV5R12", "fix", "warning", "both"),
        # manual
        _r("TV5M01", "manual", "error", "both"),
        _r("TV5M02", "manual", "error", "guide"),
        _r("TV5M03", "manual", "error", "both"),
        _r("TV5M04", "manual", "error", "both"),
        _r("TV5M05", "manual", "error", "both"),
        _r("TV5M06", "manual", "error", "probe"),
        _r("TV5M07", "manual", "error", "probe"),
        _r("TV5M08", "manual", "warning", "guide"),
        _r("TV5M09", "manual", "warning", "guide"),
        _r("TV5M10", "manual", "error", "guide"),
        _r("TV5M11", "manual", "error", "probe"),
        _r("TV5M12", "manual", "warning", "probe"),
        _r("TV5M13", "manual", "warning", "both"),
        _r("TV5M14", "manual", "warning", "probe"),
        # behavior changes
        _r("TV5B01", "behavior", "note", "guide"),
        _r("TV5B02", "behavior", "warning", "both"),
        _r("TV5B03", "behavior", "note", "probe"),
        _r("TV5B04", "behavior", "note", "both"),
        _r("TV5B05", "behavior", "note", "probe"),
        # dependency files
        _r("TV5D01", "dependency", "note", "guide"),
        _r("TV5D02", "dependency", "warning", "probe"),
        # parsing
        _r("TV5P01", "parse", "error", "guide"),
    ]
)

# --- evidence tables used by the checks -------------------------------------------------

TOKEN_RENAME_METHODS = {"from_pretrained", "push_to_hub", "pipeline", "save_pretrained"}

TRAINER_CLASSES = {"Trainer", "Seq2SeqTrainer"}
ARGS_CLASSES = {"TrainingArguments", "Seq2SeqTrainingArguments"}

# TV5R05: plain renames of TrainingArguments fields.
ARGS_RENAMES = {
    "per_gpu_train_batch_size": "per_device_train_batch_size",
    "per_gpu_eval_batch_size": "per_device_eval_batch_size",
    "no_cuda": "use_cpu",
    "push_to_hub_token": "hub_token",
}

# TV5M06: removed TrainingArguments fields without a mechanical replacement -> hint key.
# The last five were removed in 5.23.0 but are missing from the official guide.
ARGS_REMOVED = {
    "overwrite_output_dir": "default",
    "logging_dir": "logging_dir",
    "tpu_num_cores": "tpu_num_cores",
    "ray_scope": "ray_scope",
    "fsdp_min_num_params": "fsdp",
    "fsdp_transformer_layer_cls_to_wrap": "fsdp",
    "include_tokens_per_second": "default",
    "jit_mode_eval": "jit_mode_eval",
    "tpu_metrics_debug": "tpu_metrics_debug",
    "mp_parameters": "default",
    "_n_gpu": "default",
    "past_index": "default",
    "use_mps_device": "default",
    "fp16_backend": "default",
    "half_precision_backend": "default",
    "use_legacy_prediction_loop": "default",
    "push_to_hub_model_id": "default",
    "push_to_hub_organization": "default",
    "adafactor": "default",
    "fp16_opt_level": "default",
    "group_by_length": "default",
    "save_safetensors": "default",
    "torchdynamo": "default",
}
ARGS_REMOVED_NOT_IN_GUIDE = ("adafactor", "fp16_opt_level", "group_by_length", "save_safetensors", "torchdynamo")

REMOVED_PIPELINE_TASKS = (
    "summarization",
    "translation",
    "text2text-generation",
    "question-answering",
    "image-to-text",
    "visual-question-answering",
    "image-to-image",
)

REMOVED_TOKENIZER_NAMES = (
    "prepare_seq2seq_batch",
    "as_target_tokenizer",
    "sanitize_special_tokens",
    "build_inputs_with_special_tokens",
    "create_token_type_ids_from_sequences",
    "prepare_for_model",
    "truncate_sequences",
    "all_special_tokens_extended",
    "special_tokens_map_extended",
)

REMOVED_ENV_VARS = (
    "TRANSFORMERS_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT",
    "HUGGINGFACE_CO_PREFIX",
    "HUGGINGFACE_CO_EXAMPLES_TELEMETRY",
    "HF_HUB_ENABLE_HF_TRANSFER",
)

# TV5M03: model families removed in v5 that have no replacement class (reported like the TF/Flax classes).
REMOVED_NO_REPLACEMENT = ("MCTCTFeatureExtractor", "TvltFeatureExtractor")

# TV5M11: image feature extractors that have an *ImageProcessor twin in v5. Audio ones such as
# WhisperFeatureExtractor stay, and so do MCTCTFeatureExtractor / TvltFeatureExtractor (removed, but no twin).
REMOVED_FEATURE_EXTRACTORS = (
    "BeitFeatureExtractor",
    "CLIPFeatureExtractor",
    "ChineseCLIPFeatureExtractor",
    "ConditionalDetrFeatureExtractor",
    "ConvNextFeatureExtractor",
    "DPTFeatureExtractor",
    "DeformableDetrFeatureExtractor",
    "DeiTFeatureExtractor",
    "DetrFeatureExtractor",
    "DonutFeatureExtractor",
    "FlavaFeatureExtractor",
    "GLPNFeatureExtractor",
    "ImageGPTFeatureExtractor",
    "LayoutLMv2FeatureExtractor",
    "LayoutLMv3FeatureExtractor",
    "LevitFeatureExtractor",
    "MaskFormerFeatureExtractor",
    "MobileNetV1FeatureExtractor",
    "MobileNetV2FeatureExtractor",
    "MobileViTFeatureExtractor",
    "OwlViTFeatureExtractor",
    "PerceiverFeatureExtractor",
    "PoolFormerFeatureExtractor",
    "SegformerFeatureExtractor",
    "ViTFeatureExtractor",
    "VideoMAEFeatureExtractor",
    "ViltFeatureExtractor",
    "YolosFeatureExtractor",
)

# TV5R09: renamed top-level classes.
CLASS_RENAMES = {"AutoModelForVision2Seq": "AutoModelForImageTextToText"}

_FAST_IMAGE_PROCESSOR = re.compile(r"(\w+ImageProcessor)Fast")


def class_rename(name: str) -> tuple[str, str] | None:
    """``(rule id, new name)`` when a transformers class was renamed in v5, else None.

    TV5R09 is the explicit table; TV5R11 is the v5 rule "the ``Fast`` suffix of image
    processors is gone" (``CLIPImageProcessorFast`` -> ``CLIPImageProcessor``).
    """
    if name in CLASS_RENAMES:
        return "TV5R09", CLASS_RENAMES[name]
    match = _FAST_IMAGE_PROCESSOR.fullmatch(name)
    return ("TV5R11", match.group(1)) if match else None


# TV5M12: dict keys that look like removed/renamed v4 arguments.
OLD_ARGUMENT_NAMES = frozenset(
    {"torch_dtype", "use_auth_token", "warmup_ratio", "include_inputs_for_metrics"}
    | set(ARGS_RENAMES)
    | set(ARGS_REMOVED)
)

# TV5M14: keys looked for in JSON/YAML training configs. ``torch_dtype`` is left out on purpose:
# every model's own config.json has it and v5 still reads those files.
CONFIG_KEYS = frozenset({"warmup_ratio", "include_inputs_for_metrics"} | set(ARGS_RENAMES) | set(ARGS_REMOVED)) - {
    "_n_gpu",
    "past_index",
}

# TV5M13: exceptions of the old HTTP library, and the calls that talk to the Hub.
REQUESTS_ERRORS = frozenset(
    {"requests.HTTPError", "requests.exceptions.HTTPError", "requests.RequestException", "requests.exceptions.RequestException"}
)
HUB_CALLS = frozenset({"from_pretrained", "hf_hub_download", "snapshot_download", "pipeline", "push_to_hub"})

# Before/after snippets shown by ``--explain`` (code, so the same in every language).
EXAMPLES: dict[str, tuple[str, str]] = {
    "TV5R01": ("AutoModel.from_pretrained(name, torch_dtype=torch.float16)", "AutoModel.from_pretrained(name, dtype=torch.float16)"),
    "TV5R02": ("AutoTokenizer.from_pretrained(name, use_auth_token=True)", "AutoTokenizer.from_pretrained(name, token=True)"),
    "TV5R03": ("Trainer(model=model, args=args, tokenizer=tok)", "Trainer(model=model, args=args, processing_class=tok)"),
    "TV5R04": ("trainer.train(model_path=path)", "trainer.train(resume_from_checkpoint=path)"),
    "TV5R05": ("TrainingArguments(output_dir=d, no_cuda=True)", "TrainingArguments(output_dir=d, use_cpu=True)"),
    "TV5R06": ("TrainingArguments(output_dir=d, warmup_ratio=0.1)", "TrainingArguments(output_dir=d, warmup_steps=0.1)"),
    "TV5R07": (
        "TrainingArguments(output_dir=d, include_inputs_for_metrics=True)",
        'TrainingArguments(output_dir=d, include_for_metrics=["inputs"])',
    ),
    "TV5R08": ('enc = tok.encode_plus("hello")', 'enc = tok("hello")'),
    "TV5R09": ("AutoModelForVision2Seq.from_pretrained(name)", "AutoModelForImageTextToText.from_pretrained(name)"),
    "TV5R10": (
        "AutoTokenizer.from_pretrained(name, additional_special_tokens=tokens)",
        "AutoTokenizer.from_pretrained(name, extra_special_tokens=tokens)",
    ),
    "TV5R11": ("CLIPImageProcessorFast.from_pretrained(name)", "CLIPImageProcessor.from_pretrained(name)"),
    "TV5R12": (
        "AutoImageProcessor.from_pretrained(name, use_fast=True)",
        'AutoImageProcessor.from_pretrained(name, backend="torchvision")',
    ),
}


def is_enabled(rule_id: str, select: tuple[str, ...] | None, ignore: tuple[str, ...] | None) -> bool:
    """Apply ``--select``/``--ignore`` (exact ids or prefixes such as ``TV5R``)."""
    if select and not any(rule_id.startswith(s) for s in select):
        return False
    if ignore and any(rule_id.startswith(s) for s in ignore):
        return False
    return True


def validate_selectors(values: list[str]) -> list[str]:
    """Return the entries that match no rule (empty list means all valid)."""
    return [v for v in values if not any(r.startswith(v) for r in RULES)]


def severity_rank(severity: str) -> int:
    return SEVERITIES.index(severity)
