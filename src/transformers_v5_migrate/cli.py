"""Command line interface: ``transformers-v5-migrate [options] PATH...``."""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import sys
from collections.abc import Sequence

from . import __version__
from . import rules as R
from .i18n import SUPPORTED, resolve_lang, t
from .report import render_explain, render_json, render_rules, render_sarif, render_text
from .scan import scan_paths
from .walker import PathError


def _pre_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang X`` before the real parse, so that --help and errors are translated too."""
    for i, arg in enumerate(argv):
        if arg == "--":
            break
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


class _Formatter(argparse.HelpFormatter):
    """Help formatter with a translated ``usage:`` prefix."""

    def __init__(self, prog: str, usage_word: str = "usage", **kwargs: object) -> None:
        super().__init__(prog, **kwargs)  # type: ignore[arg-type]
        self._usage_word = usage_word

    def add_usage(self, usage, actions, groups, prefix=None):  # type: ignore[no-untyped-def]
        return super().add_usage(usage, actions, groups, prefix if prefix is not None else f"{self._usage_word}: ")


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: object, lang: str, **kwargs: object) -> None:
        self.lang = lang
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {t('cli.error', self.lang)}: {message}\n")


def build_parser(lang: str) -> argparse.ArgumentParser:
    kwargs: dict[str, object] = {}
    if "color" in inspect.signature(argparse.ArgumentParser.__init__).parameters:  # Python 3.14+
        kwargs["color"] = False
    parser = _Parser(
        prog="transformers-v5-migrate",
        description=t("cli.description", lang),
        epilog=t("cli.epilog", lang),
        add_help=False,
        formatter_class=functools.partial(_Formatter, usage_word=t("cli.usage", lang)),
        lang=lang,
        **kwargs,
    )
    parser._positionals.title = t("cli.positionals", lang)
    parser._optionals.title = t("cli.options", lang)
    parser.add_argument("-h", "--help", action="help", help=t("help.help", lang))
    parser.add_argument("paths", nargs="*", metavar="PATH", help=t("help.paths", lang))
    parser.add_argument("--write", action="store_true", help=t("help.write", lang))
    parser.add_argument("--format", choices=("text", "json", "sarif"), default="text", help=t("help.format", lang))
    parser.add_argument("--output", metavar="FILE", help=t("help.output", lang))
    parser.add_argument("--no-diff", action="store_true", help=t("help.no_diff", lang))
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help=t("help.color", lang))
    parser.add_argument("--select", action="append", default=[], metavar="RULE[,RULE]", help=t("help.select", lang))
    parser.add_argument("--ignore", action="append", default=[], metavar="RULE[,RULE]", help=t("help.ignore", lang))
    parser.add_argument(
        "--fail-on", choices=("error", "warning", "note", "never"), default="error", help=t("help.fail_on", lang)
    )
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB", help=t("help.exclude", lang))
    parser.add_argument("--list-rules", action="store_true", help=t("help.list_rules", lang))
    parser.add_argument("--explain", metavar="RULE", help=t("help.explain", lang))
    parser.add_argument("--lang", choices=SUPPORTED, help=t("help.lang", lang))
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}", help=t("help.version", lang))
    return parser


def _split(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _exit_code(report, fail_on: str) -> int:  # type: ignore[no-untyped-def]
    if fail_on == "never":
        return 0
    threshold = R.severity_rank(fail_on)
    severities = [f.severity for f in report.findings]
    if not report.written:  # fixes that were not applied are still open problems
        severities += [R.RULES[f.rule].severity for f in report.fixes]
    return 1 if any(R.severity_rank(s) >= threshold for s in severities) else 0


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    lang = resolve_lang(_pre_lang(args_list))
    parser = build_parser(lang)
    args = parser.parse_args(args_list)
    lang = resolve_lang(args.lang)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.list_rules:
        sys.stdout.write(render_rules(lang))
        return 0
    if args.explain:
        rule_id = args.explain.strip().upper()
        if rule_id not in R.RULES:
            parser.error(t("err.bad_rule", lang, rule=args.explain))
        sys.stdout.write(render_explain(rule_id, lang))
        return 0
    if not args.paths:
        parser.error(t("err.no_paths", lang))
    select, ignore = _split(args.select), _split(args.ignore)
    for bad in R.validate_selectors(select + ignore):
        parser.error(t("err.bad_rule", lang, rule=bad))

    try:
        report = scan_paths(args.paths, write=args.write, select=select, ignore=ignore, exclude=args.exclude)
    except PathError as exc:
        sys.stderr.write(f"{parser.prog}: {t('cli.error', lang)}: {t('err.missing_path', lang, path=exc.path)}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"{parser.prog}: {t('cli.error', lang)}: {t('err.io', lang, path=exc.filename or '', error=exc.strerror)}\n")
        return 2
    if report.files_scanned == 0 and all(os.path.isfile(p) for p in args.paths):
        # hooks pass every staged file that matches their pattern, including kinds we do not read
        sys.stderr.write(f"{parser.prog}: {t('note.no_targets', lang)}\n")
        return 0
    if report.files_scanned == 0:
        sys.stderr.write(f"{parser.prog}: {t('cli.error', lang)}: {t('err.no_targets', lang, paths=' '.join(args.paths))}\n")
        return 2

    if args.format == "json":
        text = render_json(report, lang)
    elif args.format == "sarif":
        text = render_sarif(report, lang)
    else:
        color = args.color == "always" or (
            args.color == "auto" and not args.output and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        )
        text = render_text(report, lang, diff=not args.no_diff, color=color)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except OSError as exc:
            sys.stderr.write(f"{parser.prog}: {t('cli.error', lang)}: {t('err.io', lang, path=args.output, error=exc.strerror)}\n")
            return 2
        sys.stderr.write(t("out.wrote", lang, path=args.output) + "\n")
    else:
        sys.stdout.write(text)
    return _exit_code(report, args.fail_on)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
