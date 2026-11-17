# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-17

First public release.

### Added

- `transformers-v5-migrate` command: checks Python files, Jupyter notebooks, `requirements*.txt`,
  `constraints*.txt` and `pyproject.toml` for Hugging Face Transformers v5.
- Automatic, formatting-preserving fixes (LibCST, alias aware): `torch_dtype`, `use_auth_token`,
  `Trainer(tokenizer=)`, `train(model_path=)`, renamed `TrainingArguments` fields, `warmup_ratio`,
  `include_inputs_for_metrics`, `encode_plus`/`batch_encode_plus`, `AutoModelForVision2Seq`.
- Report-only rules: removed arguments, classes, pipeline tasks, tokenizer methods and environment
  variables, two behavior changes (default dtype, chat template return type) and dependency pins that
  exclude v5.
- Dry run by default (diff preview); `--write` applies fixes atomically to `.py` and `.ipynb` files only.
- Notebook support that keeps magics (`!pip`, `%time`, `%%bash`), outputs and metadata untouched.
- Output formats `text` and `json`; options `--select`, `--ignore`, `--exclude`, `--fail-on`, `--list-rules`.
- English and Korean output (`--lang`, `TRANSFORMERS_V5_MIGRATE_LANG`, `LC_ALL`/`LC_MESSAGES`/`LANG`).
- Python API: `migrate_source`, `migrate_notebook`, `scan_paths`, `RULES`.
- Rule evidence snapshot `data/v5_surface.json` (transformers 4.57.1 vs 5.23.0, checked 2026-11-17) and
  `scripts/refresh_surface.py` (`--check` compares it with the newest 5.x).

[0.2.0]: https://github.com/LeeGyeongPil/transformers-v5-migrate/releases/tag/v0.2.0
