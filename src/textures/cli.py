"""Command-line entry point for textures."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from textures import __version__
from textures.definitions import Definitions, init_store
from textures.errors import Problem, SettingsError, TexturesError
from textures.settings import Settings, resolve_settings

PATH_FLAGS = {
    "definitions": "definitions store",
    "inputs": "inputs directory",
    "derivatives": "derivatives directory",
    "resources": "resource store",
    "items": "item-intermediate store",
    "hf_cache": "Hugging Face cache",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every textures command."""
    parser = argparse.ArgumentParser(
        prog="textures",
        description="Compute word- and region-level features of text.",
    )
    parser.add_argument("--version", action="version", version=f"textures {__version__}")
    for key, description in PATH_FLAGS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}",
            dest=key,
            metavar="PATH",
            help=f"override the {description} location",
        )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    site = commands.add_parser("site", help="inspect site settings")
    site_commands = site.add_subparsers(dest="site_command", metavar="COMMAND", required=True)
    show = site_commands.add_parser("show", help="print every resolved setting and its source")
    show.add_argument("--json", action="store_true", help="print JSON")
    show.set_defaults(handler=site_show)

    definitions = commands.add_parser("def", help="create and validate definitions stores")
    def_commands = definitions.add_subparsers(dest="def_command", metavar="COMMAND", required=True)
    init = def_commands.add_parser("init", help="create a definitions store")
    init.add_argument(
        "path", nargs="?", type=Path, help="store to create (default: selected store)"
    )
    init.set_defaults(handler=def_init)
    validate = def_commands.add_parser("validate", help="check every definition in a store")
    validate.add_argument(
        "path", nargs="?", type=Path, help="store to check (default: selected store)"
    )
    validate.add_argument("--json", action="store_true", help="print JSON")
    validate.set_defaults(handler=def_validate)
    return parser


def _settings(args: argparse.Namespace, **overrides: object) -> Settings:
    flags = {key: getattr(args, key) for key in PATH_FLAGS}
    flags.update({key: value for key, value in overrides.items() if value is not None})
    return resolve_settings(flags)


def site_show(args: argparse.Namespace) -> int:
    """Print every resolved setting with its source."""
    settings = _settings(args)
    rows = [
        (setting.key, "" if setting.value is None else str(setting.value), setting.source)
        for setting in settings.values.values()
    ]
    if args.json:
        payload = [
            {"key": key, "value": value or None, "source": source} for key, value, source in rows
        ]
        print(json.dumps(payload, indent=2))
        return 0
    widths = [max(len(row[index]) for row in rows) for index in range(2)]
    for key, value, source in rows:
        print(f"{key:<{widths[0]}}  {value or '(Hugging Face default)':<{widths[1]}}  {source}")
    return 0


def def_init(args: argparse.Namespace) -> int:
    """Create a definitions store at the given or selected location."""
    path = args.path.resolve() if args.path else _settings(args).definitions
    init_store(path)
    print(f"created definitions store {path}")
    return 0


def def_validate(args: argparse.Namespace) -> int:
    """Check a definitions store and report every problem found."""
    path = str(args.path.resolve()) if args.path else None
    problems: list[Problem] = []
    max_variants = 64
    try:
        settings = _settings(args, definitions=path)
        store = settings.definitions
        max_variants = settings.max_variants
    except SettingsError as error:
        if path is None:
            raise
        problems.append(Problem("settings", str(error)))
        store = Path(path)
    problems.extend(Definitions.load(store).validate(max_variants=max_variants))
    if args.json:
        payload = {
            "store": str(store),
            "valid": not problems,
            "problems": [{"location": p.location, "message": p.message} for p in problems],
        }
        print(json.dumps(payload, indent=2))
    elif problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s) in definitions store {store}", file=sys.stderr)
    else:
        print(f"definitions store {store} is valid")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the textures command line and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except TexturesError as error:
        print(f"textures: error: {error}", file=sys.stderr)
        return 1
