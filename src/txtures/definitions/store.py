"""Reading and creating definitions stores and txtures homes.

A definitions store is a directory with this layout::

    configs/MODULE/ID.yml     scientific configs, one directory per module
    workflows/ID.yml          workflows

It is usually the `definitions` directory of a txtures home (see
`txtures.settings`). This module reads the files and checks the layout, and
creates stores and homes. `txtures.definitions.resolution` interprets their
contents.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from txtures.errors import Problem, TxturesError
from txtures.modules import Registry
from txtures.settings import LOCATION_KEYS, SITE_DOCUMENT
from txtures.yamlio import YamlError, load_mapping

ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""Pattern for config and workflow IDs, which are also file stems."""

CONFIGS = "configs"
WORKFLOWS = "workflows"
SUFFIX = ".yml"
ALLOWED_TOP_LEVEL = {CONFIGS, WORKFLOWS, "README.md", ".git", ".gitignore"}


@dataclass(frozen=True)
class Definition:
    """The raw body of one config or workflow and where it came from.

    `path` is `None` for definitions that are not files: those packaged with
    txtures, and those given to the Python API, which `source` names.
    """

    id: str
    body: dict[str, Any]
    path: Path | None
    source: str | None = None

    @property
    def location(self) -> str:
        if self.path is not None:
            return str(self.path)
        return self.source or f"packaged {self.id}"


@dataclass
class StoreContents:
    """Every definition read from a store, plus layout problems found while reading."""

    root: Path | None
    configs: dict[str, dict[str, Definition]] = field(default_factory=dict)
    workflows: dict[str, Definition] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)


def _read_definition(path: Path, contents: StoreContents) -> Definition | None:
    """Read one definition file, recording a problem when its name or YAML is invalid."""
    if path.is_symlink():
        contents.problems.append(Problem(str(path), "symbolic links are not allowed"))
        return None
    if not path.is_file() or path.suffix != SUFFIX:
        contents.problems.append(Problem(str(path), f"expected a {SUFFIX} file"))
        return None
    if not ID_PATTERN.fullmatch(path.stem):
        contents.problems.append(
            Problem(str(path), f"ID {path.stem!r} must match {ID_PATTERN.pattern}")
        )
        return None
    try:
        body = load_mapping(path)
    except YamlError as error:
        contents.problems.append(Problem(str(path), str(error).removeprefix(f"{path}: ")))
        return None
    return Definition(path.stem, body, path)


def read_store(root: Path, registry: Registry) -> StoreContents:
    """Read every definition in the store at `root`.

    Layout problems, such as unknown files, invalid names, or YAML errors, are
    collected in the result rather than raised, so validation can report them
    all at once.

    Raises:
        TxturesError: `root` is not a directory.
    """
    if not root.is_dir():
        raise TxturesError(
            f"definitions store {root} does not exist; create a home with `txtures site init`"
        )
    contents = StoreContents(root)
    for entry in sorted(root.iterdir()):
        if entry.name == "site":
            contents.problems.append(
                Problem(
                    str(entry),
                    f"site settings now live in the txtures home's {SITE_DOCUMENT}, "
                    "not in the definitions store; move them and remove this directory",
                )
            )
        elif entry.name not in ALLOWED_TOP_LEVEL:
            contents.problems.append(Problem(str(entry), "unexpected entry in definitions store"))

    configs = root / CONFIGS
    if configs.is_dir():
        for module_dir in sorted(configs.iterdir()):
            if not module_dir.is_dir() or module_dir.name not in registry:
                contents.problems.append(Problem(str(module_dir), "not a known module directory"))
                continue
            found = contents.configs.setdefault(module_dir.name, {})
            for path in sorted(module_dir.iterdir()):
                definition = _read_definition(path, contents)
                if definition is not None:
                    found[definition.id] = definition

    workflows = root / WORKFLOWS
    if workflows.is_dir():
        for path in sorted(workflows.iterdir()):
            definition = _read_definition(path, contents)
            if definition is not None:
                contents.workflows[definition.id] = definition
    return contents


SITE_TEMPLATE = """\
# Site settings for txtures: execution and Slurm limits, which never change
# computed values. Uncomment a line to override the built-in default, or use
# `txtures site set` and `txtures site unset`.
#
# execution:
#   device: auto
#   executor: local
#   max_variants: 64
# slurm:
#   partition: gpu
#   account: mylab
#   time_hours: 24
#   memory_gb: 32
#   cpus: 4
#   gpus: 1
#   workers: 1
#   options: ["--exclude=node1"]
"""

EXAMPLE_CONFIG = """\
# An lm_scores config that scores text with Pythia 410M.
# Pin `revision` to a commit hash so results stay reproducible.
extends: main
model: EleutherAI/pythia-410m
"""

EXAMPLE_WORKFLOW = """\
# The main workflow, with lm_scores scored by the pythia410m config.
extends: main
lm_scores: pythia410m
"""

STORE_README = """\
# txtures definitions

This store holds txtures scientific definitions.

- `configs/MODULE/ID.yml`: configs for one module.
- `workflows/ID.yml`: workflows, which select a config for each module.

Run `txtures def validate` after editing.
"""

HOME_README = """\
# txtures home

Everything txtures keeps lives here:

- `site.yml`: execution and Slurm limits, edited by `txtures site set`.
- `definitions/`: configs and workflows; run `txtures def validate` after editing.
- `inputs/`: input files.
- `outputs/`: output tables.
- `cache/`: stored per-document results, reused by every run.
- `resources/`: models and the Hugging Face cache.

Each user chooses this home once with `txtures site use PATH`. A directory
can live elsewhere as a symbolic link. Git tracks `site.yml` and
`definitions/`; `.gitignore` leaves out the data directories.
"""

DEFINITIONS_FILES = {
    Path(CONFIGS) / "lm_scores" / f"pythia410m{SUFFIX}": EXAMPLE_CONFIG,
    Path(WORKFLOWS) / f"pythia{SUFFIX}": EXAMPLE_WORKFLOW,
    Path("README.md"): STORE_README,
}


def init_store(root: Path, *, git: bool = True) -> str | None:
    """Create a definitions store at `root` with examples.

    A txtures home made by `init_home` already holds one; this makes a store
    of its own, for use with `TXTURES_DEFINITIONS`. See `init_home` for `git`,
    the return value, and errors.
    """
    return _create(root, "definitions store", DEFINITIONS_FILES, (), git)


def init_home(root: Path, *, git: bool = True) -> str | None:
    """Create a txtures home at `root`: a site template, a definitions store, and data directories.

    With `git`, the home is also made a Git repository whose `.gitignore`
    leaves out the data directories, so changes to site settings and
    definitions can be tracked. The repository is shared with the group when
    the parent directory is group-writable. No commit is made. Git is
    skipped, with a note returned, when it is not installed or `root` would
    lie inside another repository.

    The home is assembled in a temporary sibling directory and renamed into
    place, so an interrupted call leaves nothing partial at `root`.

    Returns a note about Git to show the user, or `None`.

    Raises:
        TxturesError: `root` already exists, its parent does not, or
            `git init` fails.
    """
    files = {
        Path(SITE_DOCUMENT): SITE_TEMPLATE,
        Path("README.md"): HOME_README,
        Path(".gitignore"): "".join(f"/{key}/\n" for key in LOCATION_KEYS if key != "definitions"),
        **{Path("definitions") / path: text for path, text in DEFINITIONS_FILES.items()},
    }
    directories = [Path(key) for key in LOCATION_KEYS]
    return _create(root, "txtures home", files, directories, git)


def _create(
    root: Path,
    what: str,
    files: dict[Path, str],
    directories: Sequence[Path],
    git: bool,
) -> str | None:
    if root.exists() or root.is_symlink():
        raise TxturesError(f"refusing to create {what}: {root} already exists")
    parent = root.parent
    if not parent.is_dir():
        raise TxturesError(f"cannot create {what}: {parent} does not exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=parent))
    try:
        for directory in directories:
            (staging / directory).mkdir(parents=True, exist_ok=True)
        for relative, text in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        os.chmod(staging, 0o777 & ~_umask())
        note = _git_init(staging, parent, what) if git else None
        if root.exists():
            raise TxturesError(f"refusing to create {what}: {root} already exists")
        os.rename(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return note


def _git_init(directory: Path, parent: Path, what: str) -> str | None:
    """Make `directory` a Git repository, or explain why it was left without one."""
    git = shutil.which("git")
    if git is None:
        return f"git is not installed, so the {what} is not a Git repository"
    inside = subprocess.run(
        [git, "-C", str(parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if inside.returncode == 0:
        return (
            f"the {what} lies inside the Git repository {inside.stdout.strip()}, "
            "so it was not made a repository of its own"
        )
    command = [git, "init", "--quiet"]
    if parent.stat().st_mode & stat.S_IWGRP:
        command.append("--shared=group")
    result = subprocess.run([*command, str(directory)], capture_output=True, text=True)
    if result.returncode != 0:
        raise TxturesError(f"git init failed: {result.stderr.strip()}")
    return None


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
