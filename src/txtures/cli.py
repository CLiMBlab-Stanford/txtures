"""Command-line entry point for txtures."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from txtures import __version__, api, environment, installation, shellenv
from txtures.corpus import SEGMENT_LEVELS
from txtures.definitions import Definitions, init_store
from txtures.definitions.store import init_home
from txtures.engine import (
    Executor,
    TableStatus,
    collection,
    item_status,
    plan_request,
    replanner,
    run_request,
    table_is_current,
    table_statuses,
)
from txtures.engine import slurm as slurm_jobs
from txtures.engine.execution import build_resource
from txtures.engine.planning import from_record, plan_resource
from txtures.engine.tables import EXPORT_FORMATS, export_format, export_table
from txtures.errors import Problem, SettingsError, TxturesError
from txtures.inputs import InputReport, link_input, load_input, read_input, scan_inputs
from txtures.settings import (
    LOCATION_KEYS,
    SITE_DOCUMENT,
    SPECS,
    Settings,
    home_path_file,
    legacy_settings_path,
    resolve_settings,
    set_site,
    unset_site,
    use_site,
    uses_default_locations,
)
from txtures.store import EntryStore

PATH_FLAGS = {
    "home": "txtures home",
    "definitions": "definitions store",
    "inputs": "inputs directory",
    "outputs": "outputs directory",
    "cache": "cache directory",
    "resources": "resources directory",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every txtures command."""
    parser = argparse.ArgumentParser(
        prog="txtures",
        description="Compute word- and region-level features of text.",
    )
    parser.add_argument("--version", action="version", version=f"txtures {__version__}")
    for key, description in PATH_FLAGS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}",
            dest=key,
            metavar="PATH",
            help=f"override the {description} location",
        )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    site = commands.add_parser("site", help="show and change settings")
    site_commands = site.add_subparsers(dest="site_command", metavar="COMMAND", required=True)
    listing = site_commands.add_parser(
        "ls", aliases=["show"], help="list every setting with its value, source, and meaning"
    )
    listing.add_argument("--json", action="store_true", help="print JSON")
    listing.set_defaults(handler=site_ls)
    use = site_commands.add_parser(
        "use", help="use a txtures home as your site, for every txtures you run"
    )
    use.add_argument("path", type=Path, help="the txtures home")
    use.set_defaults(handler=site_use)
    init = site_commands.add_parser(
        "init",
        help="create a txtures home, as a Git repository that tracks site.yml and "
        "definitions/ unless --no-git; PATH may be an existing directory without the "
        "home's entries",
    )
    init.add_argument("path", nargs="?", type=Path, help="home to create (default: your home)")
    init.add_argument(
        "--no-git",
        action="store_false",
        dest="git",
        help="do not make the home a Git repository",
    )
    init.set_defaults(handler=site_init)
    setter = site_commands.add_parser(
        "set",
        help="set settings in your site",
        usage="%(prog)s [-h] KEY=VALUE [KEY=VALUE ...]\n       %(prog)s [-h] KEY VALUE [VALUE ...]",
        description="Set settings in site.yml of your txtures home, as "
        "KEY=VALUE pairs, such as slurm.partition=gpu "
        "slurm.account=mylab, or set one setting as KEY VALUE. A list setting such as "
        "slurm.options takes several values: repeat its KEY=VALUE pair, or list them after KEY.",
    )
    setter.add_argument("words", nargs="+", metavar="KEY=VALUE", help="a setting and its value")
    setter.set_defaults(handler=site_set)
    unsetter = site_commands.add_parser("unset", help="remove settings from your site")
    unsetter.add_argument("keys", nargs="+", metavar="KEY", help="setting key")
    unsetter.set_defaults(handler=site_unset)

    shell_file = argparse.ArgumentParser(add_help=False)
    shell_file.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="the shell startup file; default ~/.zshrc, ~/.bashrc, or ~/.profile by $SHELL",
    )
    env = commands.add_parser(
        "env",
        help="override site settings with environment variables in your shell startup file",
        description="Keep environment variables that override your site, such as "
        "TXTURES_SLURM_PARTITION for slurm.partition, in a marked block of your shell "
        "startup file. Changes apply to shells started afterwards.",
    )
    env_commands = env.add_subparsers(dest="env_command", metavar="COMMAND", required=True)
    listing = env_commands.add_parser(
        "ls",
        aliases=["show"],
        parents=[shell_file],
        help="list the variables in your shell startup file",
    )
    listing.add_argument("--json", action="store_true", help="print JSON")
    listing.set_defaults(handler=env_ls)
    setter = env_commands.add_parser(
        "set",
        parents=[shell_file],
        help="set variables for settings in your shell startup file",
        usage="%(prog)s [-h] [--file PATH] KEY=VALUE [KEY=VALUE ...]\n"
        "       %(prog)s [-h] [--file PATH] KEY VALUE [VALUE ...]",
        description="Set the environment variables of settings, given as for "
        "`txtures site set`, in your shell startup file.",
    )
    setter.add_argument("words", nargs="+", metavar="KEY=VALUE", help="a setting and its value")
    setter.set_defaults(handler=env_set)
    unsetter = env_commands.add_parser(
        "unset", parents=[shell_file], help="remove variables from your shell startup file"
    )
    unsetter.add_argument("keys", nargs="+", metavar="KEY", help="setting key")
    unsetter.set_defaults(handler=env_unset)

    definitions = commands.add_parser("def", help="validate and create definitions stores")
    def_commands = definitions.add_subparsers(dest="def_command", metavar="COMMAND", required=True)
    init = def_commands.add_parser(
        "init",
        help="create a definitions store of its own, as a Git repository unless --no-git; "
        "`txtures site init` makes a home with one",
    )
    init.add_argument("path", type=Path, help="store to create")
    init.add_argument(
        "--no-git", action="store_false", dest="git", help="do not make the store a Git repository"
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
        ("get", "write a request's output table to a file or stdout", get_command),
    ):
        sub = commands.add_parser(command, help=help_text)
        if command in ("plan", "get"):
            sub.add_argument(
                "name", metavar="INPUT", help="input basename, input file path, or NAME=PATH"
            )
        else:
            sub.add_argument(
                "names",
                nargs="+",
                metavar="INPUT",
                help="input basename, or a .txt, .tsv, or .csv file as PATH or NAME=PATH, "
                "which is linked into the inputs directory; several inputs share one "
                "Slurm job array",
            )
        sub.add_argument("-w", "--workflow", default="main", help="workflow ID (default: main)")
        sub.add_argument(
            "-m",
            "--module",
            dest="modules",
            action="extend",
            nargs="+",
            metavar="MODULE",
            help="feature modules to include, each optionally as MODULE:WORKFLOW to use "
            "another workflow for it (default: all in the workflow)",
        )
        if command == "plan":
            sub.add_argument("--json", action="store_true", help="print JSON")
        else:
            if command == "get":
                sub.add_argument(
                    "-o",
                    "--output",
                    type=Path,
                    metavar="PATH",
                    help="file to write (default: stdout)",
                )
                sub.add_argument(
                    "--format",
                    choices=EXPORT_FORMATS,
                    help="output format (default: from the extension of PATH, else tsv)",
                )
                sub.add_argument(
                    "--run",
                    action="store_true",
                    help="compute the table first if it is not current, as `txtures run` "
                    "does, waiting for any Slurm job",
                )
            sub.add_argument(
                "--local",
                action="store_true",
                help="compute here even when the site submits to Slurm",
            )
            sub.add_argument(
                "--nworkers",
                dest="workers",
                type=int,
                metavar="N",
                help="Slurm array tasks that share the work (default: slurm.workers)",
            )
            sub.add_argument("--workers", dest="workers", type=int, help=argparse.SUPPRESS)
            _size_arguments(sub)
        sub.set_defaults(handler=handler)

    resources = commands.add_parser("resources", help="build the resources a request needs")
    resource_commands = resources.add_subparsers(dest="resources_command", required=True)
    build = resource_commands.add_parser(
        "build",
        help="build missing resources without computing items; Slurm workloads go to Slurm",
    )
    build.add_argument("name", nargs="?", help="input basename (not with --kind)")
    build.add_argument("-w", "--workflow", default="main", help="workflow ID (default: main)")
    build.add_argument(
        "--kind", help="build one resource of this kind, such as gcg_treebank, not a request's"
    )
    build.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a spec value of the --kind resource, read as YAML; repeat for several",
    )
    build.add_argument(
        "-m",
        "--module",
        dest="modules",
        action="append",
        metavar="MODULE",
        help="feature module whose resources to build; repeat for several (default: all)",
    )
    build.add_argument("--local", action="store_true", help="build everything here")
    _size_arguments(build)
    build.set_defaults(handler=resources_build_command)

    module = commands.add_parser(
        "module",
        help="list modules, or one module's config keys",
        description="`txtures module ls` lists the modules; "
        "`txtures module NAME ls` lists the config keys of module NAME.",
    )
    module.add_argument("words", nargs="+", metavar="ls | NAME ls")
    module.add_argument("--json", action="store_true", help="print JSON")
    module.set_defaults(handler=module_command)

    tools = commands.add_parser("tools", help="the Apptainer image of non-Python tools")
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)
    tools_status = tool_commands.add_parser("status", help="show the runtime and the image")
    tools_status.set_defaults(handler=tools_status_command)
    tools_build = tool_commands.add_parser(
        "build", help="build the image, as a Slurm job where Slurm is set up"
    )
    tools_build.add_argument("--local", action="store_true", help="build here")
    _size_arguments(tools_build)
    tools_build.set_defaults(handler=tools_build_command)
    tools_exec = tool_commands.add_parser("exec", help="run a command in the image")
    tools_exec.add_argument("argv", nargs=argparse.REMAINDER, metavar="COMMAND")
    tools_exec.add_argument(
        "--bind",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="mount PATH in the image; repeat for several (the working directory is always)",
    )
    tools_exec.set_defaults(handler=tools_exec_command)

    status = commands.add_parser("status", help="report which output tables are current")
    status.add_argument("names", nargs="*", metavar="NAME", help="input basenames (default: all)")
    status.add_argument("--json", action="store_true", help="print JSON")
    status.set_defaults(handler=status_command)

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("directory", type=Path)
    worker.add_argument("task", type=int)
    worker.set_defaults(handler=worker_command)

    gc = commands.add_parser(
        "gc", help="remove stale and broken tables and store entries, and finished jobs"
    )
    gc.add_argument("--dry-run", action="store_true", help="list what would be removed")
    gc.set_defaults(handler=gc_command)

    install = commands.add_parser(
        "install",
        help="update this installation from its checkout, keeping its choices",
        description="Run the checkout's ./install again for this installation. The prefix, "
        "PyTorch build, development tools, and txtures command location recorded by the "
        "last install are kept unless an option changes them. First-time installations "
        "use ./install in a checkout.",
    )
    install.add_argument(
        "--upgrade",
        action="store_true",
        help="first update the checkout to the newest release (see ./install --help)",
    )
    install.add_argument("--compute", choices=installation.COMPUTE, help="switch the PyTorch build")
    install.add_argument(
        "--dev",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add or remove the development tools",
    )
    install.add_argument(
        "--default",
        action="store_true",
        dest="make_default",
        help="make the txtures command run this installation",
    )
    homes = install.add_mutually_exclusive_group()
    homes.add_argument(
        "--home",
        type=Path,
        help="give this installation its own txtures home, used before your site",
    )
    homes.add_argument(
        "--no-home", action="store_true", help="use your site again instead of an own home"
    )
    install.set_defaults(handler=install_command)

    hf = commands.add_parser("hf", help="manage your Hugging Face token for gated models")
    hf_commands = hf.add_subparsers(dest="hf_command", metavar="COMMAND", required=True)
    login = hf_commands.add_parser("login", help="save your token, read from the terminal or stdin")
    login.set_defaults(handler=hf_login)
    logout = hf_commands.add_parser("logout", help="delete your saved token")
    logout.set_defaults(handler=hf_logout)
    whoami = hf_commands.add_parser("whoami", help="show which account your token belongs to")
    whoami.set_defaults(handler=hf_whoami)
    return parser


_noted = False


def _settings(
    args: argparse.Namespace,
    *,
    note: bool = True,
    tolerate_stale: bool = False,
    **overrides: object,
) -> Settings:
    """Resolve settings and set the process environment that txtures controls.

    With `note`, prints the default-locations notice when it applies. With
    `tolerate_stale`, as the `site` commands use, settings left by earlier
    versions are ignored instead of refused.
    """
    flags = {key: getattr(args, key) for key in PATH_FLAGS}
    flags.update({key: value for key, value in overrides.items() if value is not None})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", api.DefaultLocationsWarning)
        settings = api.settings(flags, tolerate_stale=tolerate_stale)
    environment.apply(settings)
    global _noted
    if note and not _noted and uses_default_locations(settings):
        _noted = True
        print(f"note: {api.DEFAULT_LOCATIONS_MESSAGE}", file=sys.stderr)
    return settings


def _show(value: object) -> str:
    if value is None:
        return "(unset)"
    if isinstance(value, list):
        return " ".join(value) if value else "(none)"
    return str(value)


def _json_value(value: object) -> object:
    return str(value) if isinstance(value, Path) else value


def _site_settings(args: argparse.Namespace) -> Settings:
    """Resolve settings for a `site` command, warning about settings left by earlier versions."""
    settings = _settings(args, note=False, tolerate_stale=True)
    if settings.legacy is not None:
        print(f"warning: {settings.legacy.message()}", file=sys.stderr)
    installed = installation.current()
    if installed is not None and installed.settings:
        print(f"warning: ignoring them: {installation.stale_message(installed)}", file=sys.stderr)
    return settings


def site_ls(args: argparse.Namespace) -> int:
    """Print the installation and every setting with its value, source, and meaning."""
    settings = _site_settings(args)
    installed = installation.current()
    if args.json:
        payload = {
            "installation": None
            if installed is None
            else {"prefix": str(installed.prefix), "compute": installed.compute},
            "home_path": str(home_path_file(os.environ, Path.home())),
            "site_settings": str(settings.site_document),
            "settings": [
                {
                    "key": spec.key,
                    "value": _json_value(settings.values[spec.key].value),
                    "source": settings.values[spec.key].source,
                    "description": spec.description,
                }
                for spec in SPECS
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if installed is None:
        print("installation: none (running outside an ./install environment)")
    else:
        print(f"installation: {installed.prefix} ({installed.compute})")
    print(f"home-path file: {home_path_file(os.environ, Path.home())}")
    print(f"site settings: {settings.site_document}")
    rows = [(spec, _show(settings.values[spec.key].value)) for spec in SPECS]
    key_width = max(len(spec.key) for spec, _ in rows)
    value_width = max(len(value) for _, value in rows)
    for spec, value in rows:
        source = settings.values[spec.key].source
        print(f"{spec.key:<{key_width}}  {value:<{value_width}}  {source}")
        print(f"{'':<{key_width}}  {spec.description}")
    if uses_default_locations(settings):
        print(f"\nnote: {api.DEFAULT_LOCATIONS_MESSAGE}")
    return 0


def _report_setting(settings: Settings, key: str, written: Path) -> None:
    """Print where `key` now comes from, noting a source that overrides `written`."""
    setting = settings.values[key]
    print(f"{key} is now {_show(setting.value)} ({setting.source})")
    if str(written) not in setting.source and setting.source != "default":
        print(f"note: {setting.source} overrides {written} for {key}", file=sys.stderr)


def site_use(args: argparse.Namespace) -> int:
    """Choose a txtures home as the user's site, whose settings then apply."""
    home = args.path.expanduser().absolute()
    if not home.is_dir():
        raise TxturesError(
            f"{home} is not a directory; give the path of a txtures home, "
            "or create one with `txtures site init PATH`"
        )
    path, legacy = use_site(home)
    print(f"set home to {home} in {path}")
    if legacy is not None:
        print(
            f"note: removed {legacy_settings_path(os.environ, Path.home())}, from an earlier "
            "txtures",
            file=sys.stderr,
        )
    missing = [name for name in (SITE_DOCUMENT, "definitions") if not (home / name).exists()]
    if missing:
        print(f"note: {home} has no {' or '.join(missing)}; `txtures site init` makes a full home")
    settings = _settings(args, note=False, tolerate_stale=True)
    _report_setting(settings, "home", path)
    for key in LOCATION_KEYS:
        setting = settings.values[key]
        print(f"  {key}: {setting.value} ({setting.source})")
    return 0


def site_init(args: argparse.Namespace) -> int:
    """Create a txtures home at the given location or the current one."""
    settings = _settings(args, note=False, tolerate_stale=True)
    path = args.path.expanduser().absolute() if args.path else settings.home
    note = init_home(path, git=args.git)
    print(
        f"created txtures home {path}" + (" as a Git repository" if args.git and not note else "")
    )
    if note:
        print(f"note: {note}", file=sys.stderr)
    if path != settings.home:
        print(f"note: to use it, run `txtures site use {path}`", file=sys.stderr)
    return 0


def _assignments(words: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Read `site set` arguments: KEY=VALUE pairs, or one KEY followed by its values.

    Keys never contain `=`, so the first word decides the form.

    Raises:
        SettingsError: The arguments fit neither form.
    """
    if "=" not in words[0]:
        if len(words) < 2:
            raise SettingsError(f"give a value for {words[0]}, as {words[0]}=VALUE")
        return [(words[0], list(words[1:]))]
    assignments = []
    for word in words:
        key, equals, value = word.partition("=")
        if not equals or not key:
            raise SettingsError(f"expected KEY=VALUE, got {word!r}")
        assignments.append((key, [value]))
    return assignments


def site_set(args: argparse.Namespace) -> int:
    """Set settings in the site's settings, all at once."""
    assignments = _assignments(args.words)
    path = set_site(_site_settings(args).home, assignments)
    print(f"wrote {path}")
    settings = _settings(args, note=False, tolerate_stale=True)
    for key in dict.fromkeys(key for key, _ in assignments):
        _report_setting(settings, key, path)
    return 0


def site_unset(args: argparse.Namespace) -> int:
    """Remove settings from the site's settings."""
    path, removed = unset_site(_site_settings(args).home, args.keys)
    if removed:
        print(f"wrote {path}")
    settings = _settings(args, note=False, tolerate_stale=True)
    for key in dict.fromkeys(args.keys):
        if key not in removed:
            print(f"{key} is not set in {path}")
        _report_setting(settings, key, path)
    return 0


def _shell_file(args: argparse.Namespace) -> Path:
    return args.file.expanduser().absolute() if args.file else shellenv.shell_config_path()


def env_ls(args: argparse.Namespace) -> int:
    """List the variables in the shell startup file and whether this shell has them."""
    path = _shell_file(args)
    variables = shellenv.read_env(path)
    rows = []
    for spec in SPECS:
        written = variables.get(spec.env)
        current = os.environ.get(spec.env)
        if written is None and current is None:
            continue
        if written is None:
            status = "set in this shell, not by txtures env"
        elif current == written:
            status = "in effect"
        elif current is None:
            status = "not in this shell yet"
        else:
            status = f"this shell has {current!r} instead"
        rows.append((spec, written, current, status))
    if args.json:
        payload = {
            "file": str(path),
            "variables": [
                {
                    "key": spec.key,
                    "variable": spec.env,
                    "written": written,
                    "current": current,
                    "status": status,
                }
                for spec, written, current, status in rows
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(f"shell startup file: {path}")
    if not rows:
        print("no txtures variables are set")
        return 0
    name_width = max(len(spec.env) for spec, *_ in rows)
    for spec, written, current, status in rows:
        value = written if written is not None else current
        print(f"{spec.env:<{name_width}}  {value}  ({spec.key}; {status})")
    if any(status == "not in this shell yet" or "instead" in status for *_, status in rows):
        print(f"\nnote: run `source {path}` or start a new shell to apply them")
    return 0


def _report_env(args: argparse.Namespace, path: Path, keys: Sequence[str], env) -> None:
    """Print where each of `keys` will come from in a shell that sources `path`."""
    settings = resolve_settings(
        {key: getattr(args, key) for key in PATH_FLAGS},
        env=env,
        tolerate_stale=True,
        installation_home=installation.home_of_current(),
    )
    for key in dict.fromkeys(keys):
        setting = settings.values[key]
        print(f"{key} will be {_show(setting.value)} ({setting.source})")
    print(f"note: run `source {path}` or start a new shell to apply this", file=sys.stderr)


def env_set(args: argparse.Namespace) -> int:
    """Set the variables of settings in the shell startup file, all at once."""
    assignments = _assignments(args.words)
    path = _shell_file(args)
    written = shellenv.set_env(path, assignments)
    print(f"wrote {', '.join(written)} to {path}")
    _report_env(args, path, [key for key, _ in assignments], {**os.environ, **written})
    return 0


def env_unset(args: argparse.Namespace) -> int:
    """Remove the variables of settings from the shell startup file."""
    path = _shell_file(args)
    removed = shellenv.unset_env(path, args.keys)
    if removed:
        print(f"removed {', '.join(removed)} from {path}")
    names = {spec.key: spec.env for spec in SPECS}
    for key in dict.fromkeys(args.keys):
        if names[key] not in removed:
            print(f"{names[key]} is not set in {path}")
    env = {name: value for name, value in os.environ.items() if name not in removed}
    _report_env(args, path, args.keys, env)
    return 0


def hf_login(args: argparse.Namespace) -> int:
    """Check a Hugging Face token and save it where only its owner can read it."""
    import getpass

    _settings(args, note=False)
    if sys.stdin.isatty():
        token = getpass.getpass("Hugging Face token: ").strip()
    else:
        token = sys.stdin.readline().strip()
    if not token:
        raise TxturesError("no token given")
    from huggingface_hub import HfApi

    try:
        name = HfApi().whoami(token=token)["name"]
    except Exception as error:
        raise TxturesError(f"the hub rejected the token: {error}") from error
    path = environment.token_path(os.environ, Path.home())
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(staging, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    os.replace(staging, path)
    print(f"saved the token for {name} to {path}")
    return 0


def hf_logout(args: argparse.Namespace) -> int:
    _settings(args, note=False)
    path = environment.token_path(os.environ, Path.home())
    if path.exists():
        path.unlink()
        print(f"deleted {path}")
    else:
        print("no token is saved")
    return 0


def hf_whoami(args: argparse.Namespace) -> int:
    _settings(args, note=False)
    path = environment.token_path(os.environ, Path.home())
    if not path.is_file():
        print("no token is saved; run `txtures hf login`")
        return 1
    from huggingface_hub import HfApi

    try:
        name = HfApi().whoami(token=path.read_text(encoding="utf-8").strip())["name"]
    except Exception as error:
        raise TxturesError(f"the hub rejected the saved token: {error}") from error
    print(name)
    return 0


def def_init(args: argparse.Namespace) -> int:
    """Create a definitions store of its own."""
    path = args.path.expanduser().absolute()
    note = init_store(path, git=args.git)
    print(
        f"created definitions store {path}"
        + (" as a Git repository" if args.git and not note else "")
    )
    if note:
        print(f"note: {note}", file=sys.stderr)
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
        "sections": corpus.segment_count("section") if corpus else None,
        "paragraphs": corpus.segment_count("paragraph") if corpus else None,
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
    for level in SEGMENT_LEVELS:
        if count := corpus.segment_count(level):
            print(f"  {level}s: {count}")
    print(f"  regions: {corpus.region_count}")
    if corpus.metadata_columns:
        print(f"  metadata columns: {', '.join(corpus.metadata_columns)}")
    print(f"  digest: {corpus.digest}")
    return 0


def _size_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Slurm job size",
        "override what modules, resources, and slurm settings ask for, in every job "
        "this command submits; no effect on work computed here",
    )
    group.add_argument("--time-hours", type=int, metavar="N", help="time limit, in hours")
    group.add_argument("--memory-gb", type=int, metavar="N", help="memory, in GB")
    group.add_argument("--ncores", type=int, metavar="N", help="CPU cores")
    group.add_argument("--ngpus", type=int, metavar="N", help="GPUs; 0 for none")


def _size(args: argparse.Namespace):
    """The job size the command line asks for, or `None`."""
    values = {
        key: getattr(args, key)
        for key in slurm_jobs.SIZE_FIELDS
        if getattr(args, key, None) is not None
    }
    return slurm_jobs.size(values)


def _stores(settings: Settings) -> tuple[EntryStore, EntryStore]:
    return EntryStore(settings.cache), EntryStore(settings.resource_store)


def _plan(args: argparse.Namespace):
    settings = _settings(args)
    definitions = api.load_definitions(settings)
    name = link_input(settings.inputs, args.name)
    plan = plan_request(settings, definitions, name, args.workflow, args.modules)
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


def _slurm() -> slurm_jobs.Slurm:
    return slurm_jobs.Slurm()


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_command(args: argparse.Namespace) -> int:
    """Write a request's output table in the requested format to a file or stdout.

    Without `--run`, the table must be current. With it, a table that is not
    current is computed first, here or by a Slurm job that is waited for.
    """
    format = export_format(args.format, args.output)
    settings, plan = _plan(args)
    current = table_is_current(plan)
    if current or not args.run:  # otherwise `api.run` reports them
        for note in plan.notes:
            print(f"note: {note}", file=sys.stderr)
    if not current:
        if not args.run:
            raise TxturesError(
                f"the table of {plan.corpus.name} for this request is not current; "
                "compute it with `txtures run` and the same arguments, or add --run"
            )
        if args.workers is not None and (args.local or settings.executor == "local"):
            raise TxturesError("--nworkers applies only to Slurm execution")
        api.run(
            plan.corpus.name,
            args.workflow,
            args.modules,
            **{key: getattr(args, key) for key in PATH_FLAGS},
            local=args.local,
            nworkers=args.workers,
            slurm={
                key: getattr(args, key)
                for key in slurm_jobs.SIZE_FIELDS
                if getattr(args, key, None) is not None
            }
            or None,
            progress=_progress,
        )
    export_table(plan.table_directory, format, args.output)
    if args.output is not None:
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Compute requests and write their output tables, here or as one Slurm job.

    With several inputs, one job array computes every input that is not
    current when any of them has Slurm work missing; otherwise each is
    computed here in turn.
    """
    settings = _settings(args)
    if args.workers is not None and (args.local or settings.executor == "local"):
        raise TxturesError("--nworkers applies only to Slurm execution")
    override = _size(args)
    definitions = api.load_definitions(settings)
    names = list(dict.fromkeys(link_input(settings.inputs, name) for name in args.names))
    plans = [
        plan_request(settings, definitions, name, args.workflow, args.modules) for name in names
    ]
    for note in dict.fromkeys(note for plan in plans for note in plan.notes):
        print(f"note: {note}", file=sys.stderr)
    decisions = [
        slurm_jobs.decide(settings, args.local, plan, *_stores(settings)) for plan in plans
    ]
    for note in dict.fromkeys(d.note for d in decisions if d.note):
        print(f"note: {note}", file=sys.stderr)
    if any(decision.submit for decision in decisions):
        return _submit(args, settings, plans, override)
    status = 0
    for plan in plans:
        executor = api.local_executor(plan, settings, _progress)
        status |= _report_run(plan, executor, *run_request(plan, executor, args.modules))
    return status


def _submit(args: argparse.Namespace, settings: Settings, plans, override) -> int:
    pending = []
    for plan in plans:
        if table_is_current(plan):
            print(f"{plan.table_directory} is already current")
        else:
            pending.append(plan)
    if not pending:
        return 0
    workers = args.workers if args.workers is not None else settings["slurm.workers"]
    if workers < 1:
        raise TxturesError("--nworkers must be at least 1")
    slurm = _slurm()
    if len(pending) == 1:
        directory = slurm_jobs.job_directory(settings.outputs, pending[0])
    else:
        directory = slurm_jobs.batch_directory(settings.outputs, pending)
    existing = slurm_jobs.read_job(directory)
    if existing is not None:
        states = slurm.states(existing.job_id)
        if slurm_jobs.is_active(states):
            what = "this table" if len(pending) == 1 else "these tables"
            print(
                f"job {existing.job_id} is already computing {what} "
                f"({slurm_jobs.summarize(states)})"
            )
            return 0
    job = slurm_jobs.submit(pending, settings, args.modules, workers, slurm, override=override)
    if len(pending) == 1:
        target = str(pending[0].table_directory)
    else:
        target = f"{len(pending)} tables ({', '.join(p.input_name for p in pending)})"
    for build in job.request.get("builds", []):
        print(f"submitted job {build['job_id']} to build resource {build['kind']}")
    for gpu in job.request.get("after_jobs", []):
        noun = "task" if workers == 1 else "tasks"
        print(f"submitted job {gpu['job_id']} ({workers} GPU {noun}) for the GPU work")
    noun = "task" if job.workers == 1 else "tasks"
    after = " after it" if job.request.get("after_jobs") else ""
    after = after or (" after the builds" if job.request.get("builds") else "")
    print(f"submitted job {job.job_id} ({job.workers} {noun}) to write {target}{after}")
    print(f"logs: {job.directory / 'logs'}; check progress with `txtures status`")
    return 0


def worker_command(args: argparse.Namespace) -> int:
    """Run one task of a submitted job.

    A build job builds its resource; a GPU job computes only GPU work; a
    request's job computes the rest and writes its tables. A task for text
    given to `tx.compute` writes no table; the waiting call assembles it
    from the cache.
    """
    # Before anything else, so neither this task nor any process it starts submits jobs.
    os.environ[slurm_jobs.TASK_VARIABLE] = str(args.directory)
    job = slurm_jobs.read_job(args.directory)
    if job is None:
        raise TxturesError(f"{args.directory} does not hold a submitted job")
    request = job.request
    flags = {key: value for key, value in request["settings"].items()}
    flags["execution.executor"] = "local"
    settings = resolve_settings(flags, installation_home=installation.home_of_current())
    environment.apply(settings)
    if job.source == "resource":
        return _build_task(job, settings)
    if not 0 <= args.task < job.workers:
        raise TxturesError(f"task {args.task} is outside the job's {job.workers} tasks")
    if job.source == "gpu":
        return _gpu_task(job, settings, args.task)
    if job.source == "texts":
        plan = api.job_plan(job, settings)
        if plan.table_identity != job.table_identity:
            raise TxturesError(
                "the input, definitions, or resolved resources changed after this job was "
                "submitted; run `txtures run` again"
            )
        executor = _task_executor(plan, settings, args.task, job.workers)
        _progress(f"task {args.task} of {job.workers} for {job.directory}")
        executor.prepare_resources()
        executor.compute_items(skip=set(job.request.get("skip", [])))
        _report_counts(plan, executor)
        print(f"computed the items of {job.directory}")
        return 0
    requests = job.request["requests"] if job.source == "batch" else [job.request]
    return _table_task(requests, settings, args.task, job.workers)


def _task_executor(
    plan, settings: Settings, task: int, workers: int, loaded=None, keep=()
) -> Executor:
    items, resources = _stores(settings)
    return Executor(
        plan,
        items,
        resources,
        environment.run_context(settings),
        _progress,
        shard=(task, workers),
        loaded=loaded,
        keep=keep,
    )


def _replan(request, settings: Settings, definitions):
    """Plan a recorded request again and check that it is the one submitted."""
    if request.get("texts"):
        plan = api.texts_plan(Path(request["texts"]), request, settings)
    else:
        plan = plan_request(
            settings, definitions, request["input"], request["workflow"], request["modules"]
        )
    if plan.table_identity != request["table_identity"]:
        raise TxturesError(
            "the input, definitions, or resolved resources changed after this job was "
            "submitted; run `txtures run` again"
        )
    return plan


def _gpu_task(job: slurm_jobs.Job, settings: Settings, task: int) -> int:
    """Run one task of a GPU job: only GPU work, never waiting for items others hold.

    The task takes its share of each request's GPU items, then any item no
    one holds, and ends when nothing is left for it, releasing the GPU.
    """
    definitions = api.load_definitions(settings)
    status = 0
    plans = {}
    for request in job.request["requests"]:
        try:
            plans[request["table_identity"]] = _replan(request, settings, definitions)
        except TxturesError as error:
            print(f"txtures: {request['input']}: {error}", file=sys.stderr)
            status = 1

    def resources_of(request) -> set[str]:
        plan = plans.get(request["table_identity"])
        if plan is None:
            return set()
        variants = set(request["variants"])
        return {
            r.identity
            for p in plan.variants
            if p.identity in variants
            for r in p.resources.values()
        }

    # Models stay loaded while a later request needs them, so the GPU never waits for a reload.
    loaded: dict = {}
    requests = job.request["requests"]
    for position, request in enumerate(requests):
        plan = plans.get(request["table_identity"])
        if plan is None:
            continue
        keep = set().union(*(resources_of(later) for later in requests[position + 1 :]))
        try:
            executor = _task_executor(plan, settings, task, job.workers, loaded, keep)
            _progress(f"GPU task {task} of {job.workers} for {request['input']}")
            executor.prepare_resources()
            executor.compute_items(only=set(request["variants"]), wait=False)
            _report_counts(plan, executor)
        except TxturesError as error:
            print(f"txtures: {request['input']}: {error}", file=sys.stderr)
            status = 1
    return status


def _table_task(requests, settings: Settings, task: int, workers: int) -> int:
    """Run one task of a job that writes tables: each request in turn.

    GPU work recorded in a request's `skip` was the GPU job's; a request
    whose GPU results are still missing is reported and its table not
    written, rather than computing GPU work here. A request whose input or
    definitions changed since submission is reported and skipped.
    """
    definitions = api.load_definitions(settings)
    items, _ = _stores(settings)
    status = 0
    for request in requests:
        try:
            plan = _replan(request, settings, definitions)
            skip = set(request.get("skip", []))
            executor = _task_executor(plan, settings, task, workers)
            _progress(f"task {task} of {workers} for {plan.table_directory}")
            if skip:
                executor.prepare_resources()
                executor.compute_items(skip=skip)
                missing = [
                    s.variant.name
                    for s in item_status(plan, items)
                    if s.variant.identity in skip and s.missing
                ]
                if missing:
                    raise TxturesError(
                        f"GPU results of {', '.join(dict.fromkeys(missing))} are missing "
                        "(see the GPU job's logs); run `txtures run` again to compute them"
                    )
            status |= _report_run(plan, executor, *run_request(plan, executor, request["modules"]))
        except TxturesError as error:
            print(f"txtures: {request['input']}: {error}", file=sys.stderr)
            status = 1
    return status


def _build_task(job: slurm_jobs.Job, settings: Settings) -> int:
    """Build the one resource a build job was submitted for, with any missing dependencies."""
    planned = from_record(job.request)
    context = environment.run_context(settings)
    _progress(f"building resource {planned.kind} {planned.identity[:12]} for {job.directory}")
    if not build_resource(_stores(settings)[1], planned, context, _progress):
        print(f"resource {planned.kind} {planned.identity[:12]} was already built")
    else:
        print(f"built resource {planned.kind} {planned.identity[:12]}")
    return 0


def resources_build_command(args: argparse.Namespace) -> int:
    """Build the resources a request needs, without computing its items.

    Resources whose workload is `slurm` are submitted as build jobs where
    Slurm is set up, unless `--local`; the others are built here.
    """
    if args.kind:
        settings = _settings(args)
        planned = [_planned_kind(args, settings)]
    else:
        if not args.name:
            raise TxturesError("give an input name, or --kind with a resource kind")
        settings, plan = _plan(args)
        for note in plan.notes:
            print(f"note: {note}", file=sys.stderr)
        planned = plan.resources()
    _, store = _stores(settings)
    missing = [r for r in planned if store.get(r.kind, r.identity) is None]
    if not missing:
        print("every resource is already built")
        return 0
    submit = not args.local and settings.executor != "local" and not slurm_jobs.in_task()
    submit = submit and slurm_jobs.available(settings)
    context = environment.run_context(settings)
    slurm = _slurm() if submit else None
    override = _size(args)
    for resource in missing:
        if submit and resource.resource.workload(resource.resolved).slurm:
            build = slurm_jobs.submit_build(resource, settings, slurm, override=override)
            print(
                f"job {build.job_id} builds resource {resource.kind} {resource.identity[:12]}; "
                f"logs: {build.directory / 'logs'}"
            )
            continue
        build_resource(store, resource, context, _progress)
        print(f"built resource {resource.kind} {resource.identity[:12]}")
    return 0


def _planned_kind(args: argparse.Namespace, settings: Settings):
    """The resource that `resources build --kind KIND --spec KEY=VALUE...` names."""
    import yaml

    from txtures.modules import resource_class

    spec = {}
    for item in args.spec:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise TxturesError(f"--spec {item!r} is not KEY=VALUE")
        spec[key] = yaml.safe_load(value)
    try:
        cls = resource_class(args.kind)
    except KeyError:
        raise TxturesError(f"there is no resource kind {args.kind!r}") from None
    return plan_resource(args.kind, cls(spec), environment.run_context(settings))


def _first_line(cls) -> str:
    """The first line of a class's docstring."""
    return (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""


def _type_name(annotation) -> str:
    """A short, readable form of a config field's type."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return " | ".join(str(value) for value in typing.get_args(annotation))
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return " | ".join(_type_name(arg) for arg in typing.get_args(annotation))
    if origin in (tuple, list):
        (item, *_) = typing.get_args(annotation) or (str,)
        return f"list of {_type_name(item)}"
    if annotation is type(None):
        return "null"
    return getattr(annotation, "__name__", str(annotation))


def _module_rows() -> list[dict]:
    from txtures.modules import REGISTRY

    rows = []
    for name in REGISTRY:
        cls = REGISTRY[name]
        module = cls()
        rows.append(
            {
                "name": name,
                "features": list(cls.features),
                "requires": list(cls.requires),
                "stored": cls.persist,
                "slurm": module.workload(cls.Config()).slurm,
                "contract_version": cls.contract_version,
                "description": _first_line(cls),
            }
        )
    return rows


def _config_rows(name: str) -> list[dict]:
    from pydantic_core import PydanticUndefined

    from txtures.modules import REGISTRY

    if name not in REGISTRY:
        raise TxturesError(
            f"no module named {name!r}; `txtures module ls` lists them: {', '.join(REGISTRY)}"
        )
    cls = REGISTRY[name]
    rows = []
    for key, info in cls.Config.model_fields.items():
        default = info.get_default(call_default_factory=True)
        if default is PydanticUndefined:
            default = None
        rows.append(
            {
                "key": key,
                "type": _type_name(info.annotation),
                "default": list(default) if isinstance(default, tuple) else default,
                "sweepable": key in cls.sweeps,
                "description": " ".join((info.description or "").split()),
            }
        )
    return rows


def module_command(args: argparse.Namespace) -> int:
    """List the modules (`module ls`) or one module's config keys (`module NAME ls`)."""
    words = args.words
    if words == ["ls"]:
        rows = _module_rows()
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        width = max(len(row["name"]) for row in rows)
        for row in rows:
            features = ", ".join(row["features"]) or "(no columns)"
            needs = f"; needs {', '.join(row['requires'])}" if row["requires"] else ""
            where = "; Slurm" if row["slurm"] else ""
            print(f"{row['name']:<{width}}  {features}{needs}{where}")
            if row["description"]:
                print(f"{'':<{width}}  {row['description']}")
        return 0
    if len(words) == 2 and "ls" in words:
        name = words[0] if words[1] == "ls" else words[1]
        rows = _config_rows(name)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not rows:
            print(f"{name} has no config keys")
        for row in rows:
            default = json.dumps(row["default"])
            sweep = ", sweepable" if row["sweepable"] else ""
            print(f"{row['key']}: {row['type']} (default {default}{sweep})")
            if row["description"]:
                print(f"  {row['description']}")
        return 0
    raise TxturesError("use `txtures module ls` or `txtures module NAME ls`")


def _tools_image(settings: Settings):
    from txtures.tools import ToolsImage

    return plan_resource("tools", ToolsImage(), environment.run_context(settings))


def tools_status_command(args: argparse.Namespace) -> int:
    """Show the Apptainer runtime and whether the image of this definition is built."""
    from txtures import tools

    settings = _settings(args)
    context = environment.run_context(settings)
    if context.runtime is not None:
        print(f"runtime: {context.runtime} (execution.runtime)")
    else:
        managed = tools.managed_runtime(settings.resources)
        where = tools.runtime_directory(settings.resources)
        state = "installed" if managed else "not installed; installed on first use"
        print(f"runtime: Apptainer {tools.MANAGED_RUNTIME['version']} in {where} ({state})")
    image = _tools_image(settings)
    print(f"definition: {tools.DEFINITION} (sha256 {image.resolved['definition_sha256'][:12]})")
    directory = _stores(settings)[1].get(image.kind, image.identity)
    if directory is None:
        print(f"image {image.identity[:12]}: not built; run `txtures tools build`")
        return 0
    manifest = json.loads((directory / tools.IMAGE_MANIFEST).read_text(encoding="utf-8"))
    print(f"image {image.identity[:12]}: {directory / tools.SANDBOX}")
    print(f"  built {manifest['built']} on {manifest['host']} with {manifest['apptainer']}")
    print(f"  kenlm {manifest['kenlm']['commit'][:12]}, {len(manifest['packages'])} packages")
    return 0


def tools_build_command(args: argparse.Namespace) -> int:
    """Build the tools image: as a Slurm job where Slurm is set up, unless `--local`."""
    settings = _settings(args)
    image = _tools_image(settings)
    store = _stores(settings)[1]
    if store.get(image.kind, image.identity) is not None:
        print(f"the tools image {image.identity[:12]} is already built")
        return 0
    if (
        not args.local
        and settings.executor != "local"
        and not slurm_jobs.in_task()
        and slurm_jobs.available(settings)
    ):
        job = slurm_jobs.submit_build(image, settings, _slurm(), override=_size(args))
        print(f"job {job.job_id} builds the tools image; logs: {job.directory / 'logs'}")
        return 0
    build_resource(store, image, environment.run_context(settings), _progress)
    print(f"built the tools image {image.identity[:12]}")
    return 0


def tools_exec_command(args: argparse.Namespace) -> int:
    """Run a command in the built tools image, with the working directory mounted."""
    settings = _settings(args)
    image = _tools_image(settings)
    directory = _stores(settings)[1].get(image.kind, image.identity)
    if directory is None:
        raise TxturesError("the tools image is not built; run `txtures tools build`")
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise TxturesError("give the command to run, as in `txtures tools exec -- lmplz --help`")
    loaded = image.resource.load(image.resolved, directory, environment.run_context(settings))
    return loaded.run(argv, binds=[Path.cwd(), *args.bind]).returncode


def _report_counts(plan, executor: Executor) -> None:
    for planned in plan.variants:
        if planned.module.persist:
            counts = executor.counts[planned.identity]
            print(f"{_variant_label(planned)}: {counts.computed} computed, {counts.reused} reused")


def _report_run(plan, executor: Executor, directory: Path, written: bool) -> int:
    if not written:
        print(f"{directory} is already current")
        return 0
    _report_counts(plan, executor)
    print(f"wrote {directory}")
    return 0


def _table_statuses(args: argparse.Namespace, names) -> list[TableStatus]:
    settings = _settings(args)
    definitions = api.load_definitions(settings)
    return table_statuses(settings.outputs, replanner(settings, definitions), names)


def _job_rows(settings: Settings, names) -> list[tuple[slurm_jobs.Job, dict[str, str], str]]:
    """Each recorded job with its task states and a one-line description."""
    found = slurm_jobs.jobs(settings.outputs, names)
    if not found:
        return []
    slurm = _slurm()
    rows = []
    for job in found:
        try:
            states = slurm.states(job.job_id)
            description = slurm_jobs.summarize(states)
        except TxturesError as error:
            states, description = {}, f"state unavailable: {error}"
        rows.append((job, states, description))
    return rows


def status_command(args: argparse.Namespace) -> int:
    """Report which output tables are current and the state of submitted jobs."""
    statuses = _table_statuses(args, args.names or None)
    jobs = _job_rows(_settings(args), args.names or None)
    if args.json:
        payload = {
            "tables": [
                {"path": str(status.path), "state": status.state, "reason": status.reason}
                for status in statuses
            ],
            "jobs": [
                {
                    "job_id": job.job_id,
                    "input": job.request["input"],
                    "workflow": job.request["workflow"],
                    "table": job.directory.name,
                    "tasks": states,
                    "logs": str(job.directory / "logs"),
                }
                for job, states, _ in jobs
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if not statuses:
        print("no output tables")
    for status in statuses:
        reason = f"  ({status.reason})" if status.reason else ""
        print(f"{status.state:<8} {status.path}{reason}")
    for job, _, description in jobs:
        where = f"{job.request['input']}/{job.directory.name}"
        print(f"job {job.job_id}  {where}: {description}  (logs: {job.directory / 'logs'})")
    return 0


def install_command(args: argparse.Namespace) -> int:
    """Replace this process with the checkout's `./install`, run with the recorded choices.

    Settings are not resolved, so this works even while they are invalid or
    left where earlier versions kept them.
    """
    installed = installation.current()
    if installed is None:
        raise TxturesError(
            "this txtures was not installed by ./install, so `txtures install` cannot update "
            "it; update it with the tool that installed it, such as pip or uv"
        )
    script = installed.checkout / "install"
    if not os.access(script, os.X_OK):
        raise TxturesError(
            f"{script} is missing; the checkout that {installed.prefix} was installed from "
            "has moved or was removed"
        )
    command = [
        str(script),
        *installation.reinstall_arguments(
            installed,
            upgrade=args.upgrade,
            compute=args.compute,
            dev=args.dev,
            make_default=args.make_default,
            home=args.home.expanduser().absolute() if args.home else None,
            no_home=args.no_home,
        ),
    ]
    print(f"running {shlex.join(command)}", flush=True)
    sys.stderr.flush()
    os.execv(script, command)
    return 0  # not reached


def gc_command(args: argparse.Namespace) -> int:
    """Remove stale and broken output tables and store entries, and finished jobs' records."""
    removable = [s for s in _table_statuses(args, None) if s.state != "current"]
    for status in removable:
        if args.dry_run:
            print(f"would remove {status.path} ({status.state})")
        else:
            shutil.rmtree(status.path)
            print(f"removed {status.path} ({status.state})")
    finished = [
        job
        for job, states, _ in _job_rows(_settings(args), None)
        if states and not slurm_jobs.is_active(states)
    ]
    for job in finished:
        if args.dry_run:
            print(f"would remove the record and logs of finished job {job.job_id}")
        else:
            shutil.rmtree(job.directory)
            print(f"removed the record and logs of finished job {job.job_id}")
    settings = _settings(args)
    items, resources = _stores(settings)
    # A store reached through a symbolic link is shared with another home, whose
    # (possibly older) txtures may still use entries this one finds stale.
    for name, directory, store in (
        ("cache", settings.cache, items),
        ("resources", settings.resources, resources),
    ):
        if directory.is_symlink():
            print(
                f"skipped the {name} store: {directory} links to {directory.resolve()}, shared with another home"
            )
            store.root = directory / ".gc-skipped-shared-store"
    garbage = collection.find(items, resources, api.load_definitions(settings).registry)
    freed = 0
    for found in garbage:
        size = found.size
        freed += size
        label = (
            f"{found.store} {found.kind} {found.path.name[:12]} ({found.reason}, {_bytes(size)})"
        )
        if args.dry_run:
            print(f"would remove {label}")
        else:
            collection.remove(found)
            print(f"removed {label}")
    if garbage:
        verb = "would free" if args.dry_run else "freed"
        print(f"{verb} {_bytes(freed)} in the stores")
    if not removable and not finished and not garbage:
        print("nothing to remove")
    return 0


def _bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    raise AssertionError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the txtures command line and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except TxturesError as error:
        print(f"txtures: error: {error}", file=sys.stderr)
        return 1
