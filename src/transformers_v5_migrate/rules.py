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
        # manual
        _r("TV5M01", "manual", "error", "both"),
        _r("TV5M02", "manual", "error", "guide"),
        _r("TV5M03", "manual", "error", "both"),
        _r("TV5M04", "manual", "error", "both"),
        _r("TV5M05", "manual", "error", "both"),
        _r("TV5M06", "manual", "error", "probe"),
        _r("TV5M07", "manual", "error", "probe"),
        _r("TV5M08", "manual", "warning", "guide"),
        # behavior changes
        _r("TV5B01", "behavior", "note", "guide"),
        _r("TV5B02", "behavior", "warning", "both"),
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

# TV5R09: renamed top-level classes.
CLASS_RENAMES = {"AutoModelForVision2Seq": "AutoModelForImageTextToText"}

def class_rename(name: str) -> tuple[str, str] | None:
    """``(rule id, new name)`` when a transformers class was renamed in v5, else None."""
    if name in CLASS_RENAMES:
        return "TV5R09", CLASS_RENAMES[name]
    return None


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
