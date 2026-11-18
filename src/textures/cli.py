"""Command-line entry point for textures."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from textures import __version__
from textures.definitions import Definitions, init_store
from textures.engine import (
    Executor,
    TableStatus,
    item_status,
    plan_request,
    replanner,
    run_request,
    table_is_current,
    table_statuses,
)
from textures.errors import Problem, SettingsError, TexturesError
from textures.inputs import InputReport, load_input, read_input, scan_inputs
from textures.modules import RunContext
from textures.settings import Settings, resolve_settings
from textures.store import EntryStore

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

    inputs = commands.add_parser("inputs", help="list and check input files")
    input_commands = inputs.add_subparsers(dest="inputs_command", metavar="COMMAND", required=True)
    listing = input_commands.add_parser("list", help="list inputs with their sizes and status")
    listing.add_argument("--json", action="store_true", help="print JSON")
    listing.set_defaults(handler=inputs_list)
    check = input_commands.add_parser("check", help="validate one input and summarize it")
    check.add_argument("name", help="input basename")
    check.add_argument("--json", action="store_true", help="print JSON")
    check.set_defaults(handler=inputs_check)

    for command, help_text, handler in (
        ("plan", "show what a request needs and what is missing", plan_command),
        ("run", "compute a request and write its output table", run_command),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("name", help="input basename")
        sub.add_argument("-w", "--workflow", default="main", help="workflow ID (default: main)")
        sub.add_argument(
            "-m",
            "--module",
            dest="modules",
            action="append",
            metavar="MODULE",
            help="feature module to include; repeat for several (default: all in the workflow)",
        )
        if command == "plan":
            sub.add_argument("--json", action="store_true", help="print JSON")
        sub.set_defaults(handler=handler)

    status = commands.add_parser("status", help="report which output tables are current")
    status.add_argument("names", nargs="*", metavar="NAME", help="input basenames (default: all)")
    status.add_argument("--json", action="store_true", help="print JSON")
    status.set_defaults(handler=status_command)

    gc = commands.add_parser("gc", help="remove stale and broken output tables")
    gc.add_argument("--dry-run", action="store_true", help="list what would be removed")
    gc.set_defaults(handler=gc_command)
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
        sys.stdout.flush()
        print(f"{len(problems)} problem(s) in definitions store {store}", file=sys.stderr)
    else:
        print(f"definitions store {store} is valid")
    return 1 if problems else 0


def _summary(report: InputReport) -> dict[str, object]:
    corpus = report.corpus
    return {
        "name": report.name,
        "valid": corpus is not None,
        "layout": corpus.layout if corpus else None,
        "documents": len(corpus.documents) if corpus else None,
        "sentences": corpus.sentence_count if corpus else None,
        "regions": corpus.region_count if corpus else None,
        "metadata_columns": list(corpus.metadata_columns) if corpus else None,
        "digest": corpus.digest if corpus else None,
        "problems": [str(problem) for problem in report.problems],
        "warnings": [str(warning) for warning in report.warnings],
    }


def inputs_list(args: argparse.Namespace) -> int:
    """List every input with its layout, size, and validity."""
    root = _settings(args).inputs
    inventory = scan_inputs(root)
    reports = []
    for name, entry in inventory.inputs.items():
        if name in inventory.problems:
            reports.append(InputReport(name, None, tuple(inventory.problems[name]), ()))
        else:
            reports.append(read_input(entry))
    summaries = [_summary(report) for report in reports]
    if args.json:
        payload = {
            "inputs_directory": str(root),
            "inputs": summaries,
            "ignored": [str(path) for path in inventory.ignored],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if not summaries:
        print(f"no inputs in {root}")
    for summary in summaries:
        if summary["valid"]:
            status = f"{len(summary['warnings'])} warning(s)" if summary["warnings"] else "ok"
            print(
                f"{summary['name']}  {summary['layout']}  {summary['documents']} documents  "
                f"{summary['regions']} regions  {str(summary['digest'])[:12]}  {status}"
            )
        else:
            print(f"{summary['name']}  invalid: {len(summary['problems'])} problem(s)")
    for path in inventory.ignored:
        print(f"ignored: {path.name}")
    return 0


def inputs_check(args: argparse.Namespace) -> int:
    """Validate one input and print its problems, warnings, and a summary."""
    report = load_input(_settings(args).inputs, args.name)
    summary = _summary(report)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if summary["valid"] else 1
    for problem in report.problems:
        print(f"error: {problem}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    if report.corpus is None:
        sys.stdout.flush()
        print(f"{report.name} is invalid: {len(report.problems)} problem(s)", file=sys.stderr)
        return 1
    corpus = report.corpus
    print(f"{corpus.name}: {corpus.layout}")
    print(f"  documents: {len(corpus.documents)}")
    if corpus.sentence_count:
        print(f"  sentences: {corpus.sentence_count}")
    print(f"  regions: {corpus.region_count}")
    if corpus.metadata_columns:
        print(f"  metadata columns: {', '.join(corpus.metadata_columns)}")
    print(f"  digest: {corpus.digest}")
    return 0


def _definitions(settings: Settings) -> Definitions:
    """Load the selected definitions store, or the packaged definitions if it does not exist."""
    if settings.definitions.is_dir():
        return Definitions.load(settings.definitions)
    return Definitions.packaged()


def _stores(settings: Settings) -> tuple[EntryStore, EntryStore]:
    return EntryStore(settings.items), EntryStore(settings.resources)


def _plan(args: argparse.Namespace):
    settings = _settings(args)
    definitions = _definitions(settings)
    plan = plan_request(settings, definitions, args.name, args.workflow, args.modules)
    return settings, plan


def _variant_label(planned) -> str:
    labels = ", ".join(label.value for label in planned.variant.labels)
    return f"{planned.name} [{labels or planned.variant.config_id}]"


def plan_command(args: argparse.Namespace) -> int:
    """Print what a request needs: its table, resources, items, and columns."""
    settings, plan = _plan(args)
    items, resources = _stores(settings)
    statuses = {status.variant.identity: status for status in item_status(plan, items)}
    corpus = plan.corpus
    resource_rows = [
        {
            "kind": r.kind,
            "identity": r.identity,
            "present": resources.get(r.kind, r.identity) is not None,
        }
        for r in plan.resources()
    ]
    variant_rows = []
    for planned in plan.variants:
        status = statuses.get(planned.identity)
        variant_rows.append(
            {
                "variant": _variant_label(planned),
                "identity": planned.identity,
                "persisted": planned.module.persist,
                "present": status.present if status else None,
                "missing": status.missing if status else None,
            }
        )
    if args.json:
        payload = {
            "input": {"name": corpus.name, "digest": corpus.digest, "layout": corpus.layout},
            "documents": len(corpus.documents),
            "regions": corpus.region_count,
            "workflow": plan.workflow,
            "table": str(plan.table_directory),
            "table_current": table_is_current(plan),
            "resources": resource_rows,
            "variants": variant_rows,
            "columns": [column.name for column in plan.columns],
            "notes": list(plan.notes),
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(
        f"input {corpus.name}: {corpus.layout}, {len(corpus.documents)} documents, "
        f"{corpus.region_count} regions"
    )
    print(f"workflow {plan.workflow}")
    state = "current" if table_is_current(plan) else "to be written"
    print(f"table {plan.table_directory} ({state})")
    if resource_rows:
        print("resources:")
        for row in resource_rows:
            print(
                f"  {row['kind']} {row['identity'][:12]}: {'present' if row['present'] else 'missing'}"
            )
    print("variants:")
    for row in variant_rows:
        if row["persisted"]:
            detail = f"{row['present']} stored, {row['missing']} missing"
        else:
            detail = "computed when the table is written"
        print(f"  {row['variant']}: {detail}")
    print(f"columns: {', '.join(column.name for column in plan.columns)}")
    for note in plan.notes:
        print(f"note: {note}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Compute a request and write its output table."""
    settings, plan = _plan(args)
    items, resources = _stores(settings)
    for note in plan.notes:
        print(f"note: {note}", file=sys.stderr)

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    executor = Executor(
        plan, items, resources, RunContext(settings.device, settings.hf_cache), progress
    )
    directory, written = run_request(plan, executor, args.modules)
    if not written:
        print(f"{directory} is already current")
        return 0
    for planned in plan.variants:
        if planned.module.persist:
            counts = executor.counts[planned.identity]
            print(f"{_variant_label(planned)}: {counts.computed} computed, {counts.reused} reused")
    print(f"wrote {directory}")
    return 0


def _table_statuses(args: argparse.Namespace, names) -> list[TableStatus]:
    settings = _settings(args)
    definitions = _definitions(settings)
    return table_statuses(settings.derivatives, replanner(settings, definitions), names)


def status_command(args: argparse.Namespace) -> int:
    """Report whether each output table matches its request under current definitions."""
    statuses = _table_statuses(args, args.names or None)
    if args.json:
        payload = [
            {"path": str(status.path), "state": status.state, "reason": status.reason}
            for status in statuses
        ]
        print(json.dumps(payload, indent=2))
        return 0
    if not statuses:
        print("no output tables")
    for status in statuses:
        reason = f"  ({status.reason})" if status.reason else ""
        print(f"{status.state:<8} {status.path}{reason}")
    return 0


def gc_command(args: argparse.Namespace) -> int:
    """Remove stale and broken output tables."""
    removable = [s for s in _table_statuses(args, None) if s.state != "current"]
    for status in removable:
        if args.dry_run:
            print(f"would remove {status.path} ({status.state})")
        else:
            shutil.rmtree(status.path)
            print(f"removed {status.path} ({status.state})")
    if not removable:
        print("nothing to remove")
    return 0


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
