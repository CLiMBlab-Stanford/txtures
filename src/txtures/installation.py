"""Installation records and launchers written by `./install`.

An installation lives under one prefix, `CHECKOUT/txtures-install/` by
default:

- `uv/`: the pinned uv binary;
- `python/`: the uv-managed CPython interpreter;
- `uv-cache/`: uv's download and build cache;
- `env/`: the environment, with txtures installed from the checkout;
- `bin/txtures`: the launcher, which runs the environment's interpreter in
  isolated mode;
- `install.json`: facts about the installation: its checkout, environment,
  PyTorch build, whether it has the development tools, where `./install`
  put the `txtures` command, and versions. It holds no settings; those live
  in the site (see `txtures.settings`).

`./install` bootstraps uv and the environment and then runs this module's
`configure` command with the environment's interpreter. `txtures install`
runs the checkout's `./install` again with the choices in the record (see
`reinstall_arguments`).
"""

from __future__ import annotations

import argparse
import importlib.util
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
from txtures.settings import RENAMED, SPEC_BY_KEY, use_site

RECORD = "install.json"
RECORD_VERSION = 1
LAUNCHER_MARKER = "# txtures launcher written by ./install"
COMPUTE = ("cuda", "cpu")


@dataclass(frozen=True)
class Installation:
    """A parsed installation record.

    `settings` is empty except in records written by earlier versions, which
    could fix `resources` and `execution.executor`; see `stale_message`.
    """

    prefix: Path
    checkout: Path
    compute: str
    settings: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def source(self) -> str:
        return f"installation ({self.prefix / RECORD})"

    @property
    def dev(self) -> bool | None:
        """Whether the development tools were installed, or `None` if the record predates `dev`."""
        value = self.record.get("dev")
        return None if value is None else bool(value)


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
            settings=dict(record.get("settings", {})),
            record=record,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise SettingsError(f"{path}: invalid installation record: {error}") from error


def keep_advice(values: Mapping[str, Any]) -> str:
    """Say how to keep settings that an earlier version stored elsewhere, in the txtures home."""
    import shlex

    advice = []
    for key, value in values.items():
        key = RENAMED.get(key, key)
        if key == "home" or key not in SPEC_BY_KEY:
            continue
        if SPEC_BY_KEY[key].location:
            advice.append(f"move or symlink {value} to HOME/{key}")
        else:
            items = value if isinstance(value, list) else [value]
            pairs = " ".join(shlex.quote(f"{key}={item}") for item in items)
            advice.append(f"run `txtures site set {pairs}`")
    return "; ".join(advice)


def stale_message(installation: Installation) -> str:
    """Explain how to move settings out of an earlier installation record and into the home."""
    return (
        f"{installation.prefix / RECORD} sets {', '.join(installation.settings)}, but "
        "installations no longer hold settings; every setting lives in the txtures home. "
        f"Run `txtures install` to remove them; to keep them, "
        f"{keep_advice(installation.settings)}."
    )


def current() -> Installation | None:
    """The installation whose environment runs this process, if any."""
    environment = Path(sys.prefix)
    if environment.name != "env":
        return None
    installation = read(environment.parent)
    if installation is None or Path(installation.record.get("environment", "")) != environment:
        return None
    return installation


def reinstall_arguments(
    installation: Installation,
    *,
    upgrade: bool = False,
    compute: str | None = None,
    dev: bool | None = None,
    make_default: bool = False,
) -> list[str]:
    """The `./install` arguments that rebuild `installation` with the same choices.

    The prefix, the PyTorch build, the development tools, and the directory
    of the `txtures` command are kept unless `compute` or `dev` changes
    them. Records from before 0.5.0 do not say whether the development tools
    were installed; this process's environment is checked for pytest
    instead. They also do not say where the `txtures` command went, so
    `./install` uses its default there.
    """
    if dev is None:
        dev = installation.dev
    if dev is None:
        dev = importlib.util.find_spec("pytest") is not None
    arguments = ["--upgrade"] if upgrade else []
    arguments += ["--prefix", str(installation.prefix)]
    arguments += ["--compute", compute or installation.compute]
    if dev:
        arguments.append("--dev")
    if "bin_dir" in installation.record:
        bin_dir = installation.record["bin_dir"]
        arguments += ["--no-launcher"] if bin_dir is None else ["--bin-dir", str(bin_dir)]
    if make_default:
        arguments.append("--default")
    return arguments


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
                f"left {path} pointing at another installation; to make this one the "
                f"default, run `{target} install --default`"
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
    bin_dir: Path | None = None,
    make_default: bool = False,
    uv_version: str = "",
    site: Path | None = None,
    dev: bool = False,
) -> tuple[Installation, list[str]]:
    """Write the installation record and launchers.

    `dev` records whether `./install` installed the development tools, and
    `bin_dir` where it put the `txtures` command, or `None` for no command,
    so that `txtures install` can keep both. The record holds no settings. One written by an earlier version that did
    is rewritten without them, and a message names them. `site`, a
    txtures home, is chosen as the user's site, as `txtures site use` does. Returns the installation and messages to print.

    Raises:
        SettingsError: A choice is invalid, or `site` is not a directory.
    """
    if compute not in COMPUTE:
        raise SettingsError(f"compute must be one of {', '.join(COMPUTE)}")
    try:
        previous = read(prefix)
    except SettingsError:
        previous = None
    if site is not None and not site.expanduser().is_dir():
        raise SettingsError(f"site {site} is not a txtures home directory")

    environment = prefix / "env"
    record = {
        "record_version": RECORD_VERSION,
        "checkout": str(checkout),
        "environment": str(environment),
        "compute": compute,
        "dev": dev,
        "bin_dir": None if bin_dir is None else str(bin_dir),
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
    if previous is not None and previous.settings:
        messages.append(
            f"removed {', '.join(previous.settings)} from the installation record, which no "
            f"longer holds settings; to keep them, {keep_advice(previous.settings)}"
        )
    if bin_dir is not None:
        messages.append(install_user_launcher(bin_dir, launcher, make_default))
    if site is not None:
        path, legacy = use_site(site)
        messages.append(f"set home to {site} in {path}")
        if legacy:
            messages.append(
                f"removed the settings file of an earlier txtures, which set "
                f"{', '.join(legacy)}; to keep them, {keep_advice(legacy)}"
            )
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
    setup.add_argument("--bin-dir", type=Path)
    setup.add_argument("--default", action="store_true", dest="make_default")
    setup.add_argument("--uv-version", default="")
    setup.add_argument("--site", type=Path)
    setup.add_argument("--dev", action="store_true")
    args = parser.parse_args(argv)
    try:
        _, messages = configure(
            args.prefix.absolute(),
            args.checkout.absolute(),
            compute=args.compute,
            bin_dir=args.bin_dir,
            make_default=args.make_default,
            uv_version=args.uv_version,
            site=args.site,
            dev=args.dev,
        )
    except SettingsError as error:
        print(f"txtures: error: {error}", file=sys.stderr)
        return 1
    print("\n".join(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
