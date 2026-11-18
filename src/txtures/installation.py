"""Installation records and launchers written by `./install`.

An installation lives under one prefix, `CHECKOUT/txtures-install/` by
default:

- `uv/`: the pinned uv binary;
- `python/`: the uv-managed CPython interpreter;
- `uv-cache/`: uv's download and build cache;
- `env/`: the environment, with txtures installed from the checkout;
- `bin/txtures`: the launcher, which runs the environment's interpreter in
  isolated mode;
- `install.json`: the choices made at installation, including any settings
  the installation fixes (`resources` and `execution.executor`, when given).

`./install` bootstraps uv and the environment and then runs this module's
`configure` command with the environment's interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtures import __version__
from txtures.errors import SettingsError
from txtures.settings import set_personal

RECORD = "install.json"
RECORD_VERSION = 1
LAUNCHER_MARKER = "# txtures launcher written by ./install"
COMPUTE = ("cuda", "cpu")
EXECUTORS = ("local", "slurm")


@dataclass(frozen=True)
class Installation:
    """A parsed installation record."""

    prefix: Path
    checkout: Path
    compute: str
    settings: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def source(self) -> str:
        return f"installation ({self.prefix / RECORD})"


def read(prefix: Path) -> Installation | None:
    """Read the record under `prefix`, or return `None` if there is none.

    Raises:
        SettingsError: The record exists but is invalid.
    """
    path = prefix / RECORD
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("record_version") != RECORD_VERSION:
            raise ValueError(f"unsupported record_version {record.get('record_version')!r}")
        return Installation(
            prefix=prefix,
            checkout=Path(record["checkout"]),
            compute=str(record["compute"]),
            settings=dict(record["settings"]),
            record=record,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise SettingsError(f"{path}: invalid installation record: {error}") from error


def current() -> Installation | None:
    """The installation whose environment runs this process, if any."""
    environment = Path(sys.prefix)
    if environment.name != "env":
        return None
    installation = read(environment.parent)
    if installation is None or Path(installation.record.get("environment", "")) != environment:
        return None
    return installation


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def launcher_text(target: Path, python: bool) -> str:
    """Text of a launcher that runs `target`: an interpreter, or another launcher."""
    command = f"{_quote(target)} -I -m txtures" if python else _quote(target)
    return (
        f'#!/bin/sh\n{LAUNCHER_MARKER}; rerun ./install instead of editing.\nexec {command} "$@"\n'
    )


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(text, encoding="utf-8")
    staging.chmod(0o755)
    os.replace(staging, path)


def install_user_launcher(bin_dir: Path, target: Path, make_default: bool) -> str:
    """Point `bin_dir/txtures` at an installation's launcher, if that is safe.

    A missing command is created. A txtures launcher for another
    installation is replaced only when `make_default` is set. Any other file
    is left alone. Returns a message describing what happened.
    """
    path = bin_dir / "txtures"
    text = launcher_text(target, python=False)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return f"left {path} unchanged: it is not a txtures launcher"
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if LAUNCHER_MARKER not in existing:
            return f"left {path} unchanged: it is not a txtures launcher"
        if existing == text:
            return f"{path} already runs this installation"
        if not make_default:
            return (
                f"left {path} pointing at another installation; "
                "rerun ./install with --default to make this one the default"
            )
    _write_executable(path, text)
    message = f"{path} now runs this installation"
    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        message += f"; add {bin_dir} to PATH to use it"
    return message


def configure(
    prefix: Path,
    checkout: Path,
    *,
    compute: str,
    resources: Path | None = None,
    executor: str | None = None,
    bin_dir: Path | None = None,
    make_default: bool = False,
    uv_version: str = "",
    site: Path | None = None,
) -> tuple[Installation, list[str]]:
    """Write the installation record and launchers.

    The record fixes only the settings given here or in an earlier run;
    other settings resolve as they would without an installation. `site`, a
    definitions store, is written to the user's personal settings, as
    `txtures site use` does. Returns the installation and messages to print.

    Raises:
        SettingsError: A choice is invalid, or `site` is not a directory.
    """
    if compute not in COMPUTE:
        raise SettingsError(f"compute must be one of {', '.join(COMPUTE)}")
    if executor is not None and executor not in EXECUTORS:
        raise SettingsError(f"executor must be one of {', '.join(EXECUTORS)}")
    previous = read(prefix)
    settings = dict(previous.settings) if previous else {}
    if site is not None and not site.expanduser().is_dir():
        raise SettingsError(f"site {site} is not a definitions store directory")
    if resources is not None:
        settings["resources"] = str(resources.expanduser().absolute())
    if executor is not None:
        settings["execution.executor"] = executor

    environment = prefix / "env"
    record = {
        "record_version": RECORD_VERSION,
        "checkout": str(checkout),
        "environment": str(environment),
        "compute": compute,
        "settings": settings,
        "txtures_version": __version__,
        "python_version": platform.python_version(),
        "uv_version": uv_version,
    }
    staging = prefix / f".{RECORD}.tmp"
    staging.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, prefix / RECORD)

    launcher = prefix / "bin" / "txtures"
    _write_executable(launcher, launcher_text(environment / "bin" / "python", python=True))
    messages = [f"installed txtures {__version__} ({compute}) in {prefix}"]
    messages += [f"  {key}: {value}" for key, value in sorted(settings.items())]
    if bin_dir is not None:
        messages.append(install_user_launcher(bin_dir, launcher, make_default))
    if site is not None:
        path = set_personal("definitions", [str(site.expanduser().absolute())])
        messages.append(f"set definitions to {site} in {path}")
    installation = read(prefix)
    assert installation is not None
    return installation, messages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m txtures.installation")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure", help="write the installation record and launchers")
    setup.add_argument("--prefix", type=Path, required=True)
    setup.add_argument("--checkout", type=Path, required=True)
    setup.add_argument("--compute", choices=COMPUTE, required=True)
    setup.add_argument("--resources", type=Path)
    setup.add_argument("--executor", choices=EXECUTORS)
    setup.add_argument("--bin-dir", type=Path)
    setup.add_argument("--default", action="store_true", dest="make_default")
    setup.add_argument("--uv-version", default="")
    setup.add_argument("--site", type=Path)
    args = parser.parse_args(argv)
    try:
        _, messages = configure(
            args.prefix.absolute(),
            args.checkout.absolute(),
            compute=args.compute,
            resources=args.resources,
            executor=args.executor,
            bin_dir=args.bin_dir,
            make_default=args.make_default,
            uv_version=args.uv_version,
            site=args.site,
        )
    except SettingsError as error:
        print(f"txtures: error: {error}", file=sys.stderr)
        return 1
    print("\n".join(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
