"""Setting environment variables in the user's shell startup file.

Every setting has an environment variable, such as `TXTURES_SLURM_PARTITION`
for `slurm.partition`, which overrides the site. `txtures env set` and
`txtures env unset` keep such variables as `export` lines in a marked block
of one startup file, so a user can depart from a shared site without
changing it. Lines outside the block are never changed.

The startup file follows the login shell named by `SHELL`: `~/.zshrc` for
zsh, `~/.bashrc` for bash, and `~/.profile` otherwise. A change applies to
shells started afterwards, or after the file is sourced again.
"""

from __future__ import annotations

import os
import shlex
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from txtures.errors import SettingsError
from txtures.settings import SPECS, SettingSpec, check_assignments

BEGIN = "# >>> txtures env >>>"
END = "# <<< txtures env <<<"
NOTE = "# Written by `txtures env set` and `txtures env unset`; change it with those commands."

SPEC_BY_ENV = {spec.env: spec for spec in SPECS}


def shell_config_path(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """The startup file of the user's login shell."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    shell = Path(env.get("SHELL", "")).name
    if shell == "zsh":
        return home / ".zshrc"
    if shell == "bash":
        return home / ".bashrc"
    return home / ".profile"


def _parts(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Split the file into the lines before the block, inside it, and after it."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    begins = [n for n, line in enumerate(lines) if line.strip() == BEGIN]
    ends = [n for n, line in enumerate(lines) if line.strip() == END]
    if not begins and not ends:
        return lines, [], []
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
        raise SettingsError(
            f"{path}: expected one block from {BEGIN!r} to {END!r}; repair it by hand"
        )
    begin, end = begins[0], ends[0]
    return lines[:begin], lines[begin + 1 : end], lines[end + 1 :]


def _parse(path: Path, block: Sequence[str]) -> dict[str, str]:
    variables = {}
    for line in block:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            words = []
        name, equals, value = words[1].partition("=") if len(words) == 2 else ("", "", "")
        if words[:1] != ["export"] or not equals or name not in SPEC_BY_ENV:
            raise SettingsError(
                f"{path}: unexpected line in the txtures block: {line!r}; repair it by hand"
            )
        variables[name] = value
    return variables


def read_env(path: Path) -> dict[str, str]:
    """The variables in the txtures block of `path`, by name.

    Raises:
        SettingsError: The block is malformed.
    """
    _, block, _ = _parts(path)
    return _parse(path, block)


def _env_text(spec: SettingSpec, value: Any) -> str:
    """A checked value as its environment variable spells it."""
    if spec.kind == "text_list":
        return shlex.join(value)
    return str(value)


def _write(path: Path, variables: Mapping[str, str]) -> None:
    """Rewrite the txtures block of `path` to hold `variables`, atomically."""
    target = path.resolve()  # keep a symbolic link to a dotfiles repository
    before, _, after = _parts(target)
    block = [
        f"export {spec.env}={shlex.quote(variables[spec.env])}"
        for spec in SPECS
        if spec.env in variables
    ]
    if block:
        if before and before[-1].strip():
            before = [*before, ""]
        lines = [*before, BEGIN, NOTE, *block, END, *after]
    else:
        while before and not before[-1].strip() and (not after or not after[0].strip()):
            before = before[:-1]
        lines = [*before, *after]
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{line}\n" for line in lines))
        os.chmod(staging, mode)
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def set_env(
    path: Path,
    assignments: Sequence[tuple[str, Sequence[str]]],
    *,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Set the variables of settings in the txtures block of `path`.

    Assignments are checked as `settings.check_assignments` checks them, so
    an invalid one changes nothing, and relative paths are made absolute.
    Any setting may be given, `definitions` included. Returns the variables
    written, by name.

    Raises:
        SettingsError: A key is unknown or repeated, a value is invalid, or
            the block is malformed.
    """
    checked = check_assignments(assignments, cwd=cwd)
    written = {}
    for key, value in checked.items():
        spec = next(spec for spec in SPECS if spec.key == key)
        written[spec.env] = _env_text(spec, value)
    variables = read_env(path)
    variables.update(written)
    _write(path, variables)
    return written


def unset_env(path: Path, keys: Sequence[str]) -> list[str]:
    """Remove the variables of settings `keys` from the txtures block of `path`.

    Returns the names of the variables that were there to remove; the block
    is removed once it is empty.

    Raises:
        SettingsError: A key is unknown or the block is malformed.
    """
    names = []
    for key in dict.fromkeys(keys):
        spec = next((spec for spec in SPECS if spec.key == key), None)
        if spec is None:
            raise SettingsError(f"unknown setting {key!r}; see `txtures site ls`")
        names.append(spec.env)
    variables = read_env(path)
    removed = [name for name in names if name in variables]
    if removed:
        for name in removed:
            del variables[name]
        _write(path, variables)
    return removed
