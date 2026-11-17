# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-17

### Added

- `transformers-v5-migrate` command: checks Python files, Jupyter notebooks, `requirements*.txt`,
  `constraints*.txt`, `pyproject.toml`, conda `environment.yml`, Dockerfiles, `.env` files, shell scripts,
  Makefiles and JSON/YAML training configs for Hugging Face Transformers v5.
- Automatic, formatting-preserving fixes (LibCST, alias aware): `torch_dtype`, `use_auth_token`,
  `Trainer(tokenizer=)`, `train(model_path=)`, renamed `TrainingArguments` fields, `warmup_ratio`,
  `include_inputs_for_metrics`, `encode_plus`/`batch_encode_plus`, `AutoModelForVision2Seq`,
  `additional_special_tokens`, `*ImageProcessorFast`, image processor `use_fast`.
- Report-only rules: removed arguments, classes, pipeline tasks, tokenizer methods, environment variables,
  `transformers-cli`, `requests` errors around Hub calls, behavior changes (default dtype, chat template return
  type, shard size, `use_fast` on tokenizers) and dependency pins that exclude v5.
- Dry run by default (diff preview); `--write` applies fixes atomically to `.py` and `.ipynb` files only.
- Notebook support that keeps magics (`!pip`, `%time`, `%%bash`), outputs and metadata untouched.
- Output formats `text`, `json` and `sarif` (2.1.0), `--select`, `--ignore`, `--exclude`, `--fail-on`,
  `--list-rules`, `--explain RULE`.
- English and Korean output (`--lang`, `TRANSFORMERS_V5_MIGRATE_LANG`, `LC_ALL`/`LC_MESSAGES`/`LANG`).
- Python API: `migrate_source`, `migrate_notebook`, `scan_paths`, `RULES`.
- Rule evidence snapshot `data/v5_surface.json` (transformers 4.57.1 vs 5.23.0, checked 2026-11-17) and
  `scripts/refresh_surface.py` (`--check` for a scheduled comparison against the newest 5.x).
- pre-commit hooks (`.pre-commit-hooks.yaml`) and a weekly `surface-watch` workflow.

[Unreleased]: https://github.com/LeeGyeongPil/transformers-v5-migrate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LeeGyeongPil/transformers-v5-migrate/releases/tag/v0.1.0
