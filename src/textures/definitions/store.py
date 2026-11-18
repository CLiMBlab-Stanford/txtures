"""Reading and creating definitions stores.

A definitions store is a directory with this layout::

    site/site.yml             site settings (see `textures.settings`)
    configs/MODULE/ID.yml     scientific configs, one directory per module
    workflows/ID.yml          workflows

This module reads the files and checks the layout. `textures.definitions.resolution`
interprets their contents.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textures.errors import Problem, TexturesError
from textures.modules import Registry
from textures.settings import SITE_DOCUMENT
from textures.yamlio import YamlError, load_mapping

ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""Pattern for config and workflow IDs, which are also file stems."""

CONFIGS = "configs"
WORKFLOWS = "workflows"
SUFFIX = ".yml"
ALLOWED_TOP_LEVEL = {"site", CONFIGS, WORKFLOWS, "README.md", ".git", ".gitignore"}


@dataclass(frozen=True)
class Definition:
    """The raw body of one config or workflow and where it came from.

    `path` is `None` for definitions packaged with textures.
    """

    id: str
    body: dict[str, Any]
    path: Path | None

    @property
    def location(self) -> str:
        return str(self.path) if self.path is not None else f"packaged {self.id}"


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
        TexturesError: `root` is not a directory.
    """
    if not root.is_dir():
        raise TexturesError(
            f"definitions store {root} does not exist; create it with `textures def init`"
        )
    contents = StoreContents(root)
    for entry in sorted(root.iterdir()):
        if entry.name not in ALLOWED_TOP_LEVEL:
            contents.problems.append(Problem(str(entry), "unexpected entry in definitions store"))
    site = root / "site"
    if site.is_dir():
        for entry in sorted(site.iterdir()):
            if entry.name != SITE_DOCUMENT.name:
                contents.problems.append(Problem(str(entry), "unexpected entry in site/"))

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
# Site settings for textures. They choose storage locations and execution
# limits and never change computed values. Relative paths resolve against
# this file's directory. Uncomment a line to override the built-in default.
#
# inputs: /path/to/inputs
# derivatives: /path/to/derivatives
# resources: /path/to/resources
# items: /path/to/items
# hf_cache: /path/to/huggingface/cache
# execution:
#   device: auto
#   max_variants: 64
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

README = """\
# textures definitions

This store holds textures site settings and scientific definitions.

- `site/site.yml`: storage locations and execution limits.
- `configs/MODULE/ID.yml`: configs for one module.
- `workflows/ID.yml`: workflows, which select a config for each module.

Run `textures def validate` after editing.
"""


def init_store(root: Path) -> None:
    """Create a definitions store at `root` with a site template and examples.

    The store is assembled in a temporary sibling directory and renamed into
    place, so an interrupted call leaves no partial store at `root`.

    Raises:
        TexturesError: `root` already exists, or its parent does not.
    """
    if root.exists() or root.is_symlink():
        raise TexturesError(f"refusing to create definitions store: {root} already exists")
    parent = root.parent
    if not parent.is_dir():
        raise TexturesError(f"cannot create definitions store: {parent} does not exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=parent))
    try:
        files = {
            SITE_DOCUMENT: SITE_TEMPLATE,
            Path(CONFIGS) / "lm_scores" / f"pythia410m{SUFFIX}": EXAMPLE_CONFIG,
            Path(WORKFLOWS) / f"pythia{SUFFIX}": EXAMPLE_WORKFLOW,
            Path("README.md"): README,
        }
        for relative, text in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        os.chmod(staging, 0o777 & ~_umask())
        if root.exists():
            raise TexturesError(f"refusing to create definitions store: {root} already exists")
        os.rename(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
