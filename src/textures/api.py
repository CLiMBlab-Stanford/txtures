"""The Python API: the same requests as the command line, from Python.

`run` is `textures run`: it computes a named input's request and writes the
same output table, then returns the table. `compute` computes features of
text given directly and returns the table without writing one. Both use the
configured item and resource stores, so repeated calls reuse earlier work.

The API always computes in the calling process, whatever
`execution.executor` says; submit Slurm jobs with `textures run`.

Like the command line, the API replaces the Hugging Face variables in the
process environment (see `textures.environment`) whenever it resolves
settings. Hugging Face libraries read some of them on import, so use textures
before importing those libraries yourself.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textures import environment, installation
from textures.definitions import Definitions
from textures.definitions.resolution import EXTENDS, MAIN
from textures.definitions.store import Definition, StoreContents
from textures.engine import Executor, Plan, plan_request, run_request
from textures.engine.tables import read_table, table_frame

__all__ = ["compute", "read_table", "run"]
from textures.errors import DefinitionError, InputError, Problem
from textures.inputs import InputSettings, read_texts
from textures.modules import REGISTRY, RunContext
from textures.settings import Settings, resolve_settings
from textures.store import EntryStore

if TYPE_CHECKING:
    import pandas as pd

Texts = str | Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]]


def settings(flags: Mapping[str, Any] | None = None) -> Settings:
    """Resolve settings, with `flags` taking precedence, and control the environment.

    `flags` maps setting keys, such as `items` or `execution.device`, to
    values. The installation running this interpreter, if any, supplies the
    settings it records.

    Raises:
        SettingsError: A settings document or value is invalid.
    """
    installed = installation.current()
    resolved = resolve_settings(
        flags,
        installation=installed.settings if installed else None,
        installation_source=installed.source if installed else "installation",
    )
    environment.apply(resolved)
    return resolved


def load_definitions(resolved: Settings) -> Definitions:
    """Load the selected definitions store, or the packaged definitions if it does not exist."""
    if resolved.definitions.is_dir():
        return Definitions.load(resolved.definitions)
    return Definitions.packaged()


def local_executor(plan: Plan, resolved: Settings, progress: Callable[[str], None] | None = None):
    """An executor for `plan` on this machine, using the configured stores."""
    return Executor(
        plan,
        EntryStore(resolved.items),
        EntryStore(resolved.resource_store),
        RunContext(resolved.device, resolved.hf_cache),
        progress,
    )


def _flags(locations: Mapping[str, Any], device: str | None) -> dict[str, Any]:
    flags = {key: str(value) for key, value in locations.items() if value is not None}
    if device is not None:
        flags["execution.device"] = device
    return flags


def run(
    name: str,
    workflow: str = MAIN,
    modules: Sequence[str] | None = None,
    *,
    definitions: Path | str | None = None,
    inputs: Path | str | None = None,
    derivatives: Path | str | None = None,
    items: Path | str | None = None,
    resources: Path | str | None = None,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Compute a request for the input `name` and return its output table.

    This is `textures run NAME --workflow WORKFLOW --modules ...`: the table
    is written to the derivatives directory, or reused if it is current, and
    then read with `read_table`. The keyword arguments override settings for
    this call; `progress`, if given, receives progress messages.

    Raises:
        TexturesError: The input, definitions, or settings are invalid, or a
            computation failed.
    """
    resolved = settings(
        _flags(
            {
                "definitions": definitions,
                "inputs": inputs,
                "derivatives": derivatives,
                "items": items,
                "resources": resources,
            },
            device,
        )
    )
    plan = plan_request(resolved, load_definitions(resolved), name, workflow, modules)
    _warn(plan.notes)
    directory, _ = run_request(plan, local_executor(plan, resolved, progress), modules)
    return read_table(directory)


def compute(
    modules: str | Sequence[str],
    texts: Texts,
    *,
    config: Mapping[str, Mapping[str, Any]] | None = None,
    workflow: str | None = None,
    region_delimiter: str | None = None,
    sentence_delimiter: str | None = None,
    region_joiner: str = " ",
    sentence_joiner: str = " ",
    definitions: Path | str | None = None,
    items: Path | str | None = None,
    resources: Path | str | None = None,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Compute the features of `modules` for text given directly, and return the table.

    `texts` is one document, a sequence of documents (with IDs `1`, `2`,
    ...), or a mapping from document ID to document. Each document is a
    string, divided into regions and sentences by the delimiters as an input
    file's sidecar would divide it, or a sequence of region strings.

    Configs come from one of two places:

    - `config` maps module names to config bodies, such as
      `{"lm_scores": {"model": "gpt2"}}`. Each body overrides the packaged
      defaults, may sweep parameters as a config file can, and ignores the
      definitions store, so the call alone determines the result. Modules
      not named use their packaged defaults.
    - Otherwise the definitions store's `workflow` (default `main`) chooses
      them, as for `textures run`.

    The table has the columns of an output table. No table is written, but
    computed items are stored and reused.

    Raises:
        TexturesError: The text, configs, or settings are invalid, or a
            computation failed.
    """
    if config is not None and workflow is not None:
        raise DefinitionError([Problem("compute", "give either config or workflow, not both")])
    requested = [modules] if isinstance(modules, str) else list(modules)
    report = read_texts(
        texts,
        InputSettings(
            region_delimiter=region_delimiter,
            sentence_delimiter=sentence_delimiter,
            region_joiner=region_joiner,
            sentence_joiner=sentence_joiner,
        ),
    )
    if report.corpus is None:
        raise InputError(report.problems)
    _warn(str(warning) for warning in report.warnings)
    resolved = settings(
        _flags({"definitions": definitions, "items": items, "resources": resources}, device)
    )
    chosen = load_definitions(resolved) if config is None else _given(config)
    plan = plan_request(
        resolved,
        chosen,
        report.corpus.name,
        workflow or MAIN,
        requested,
        corpus=report.corpus,
    )
    _warn(plan.notes)
    run_executor = local_executor(plan, resolved, progress)
    run_executor.prepare_resources()
    run_executor.compute_items()
    return table_frame(plan, run_executor)


def _given(config: Mapping[str, Mapping[str, Any]]) -> Definitions:
    """Packaged definitions whose `main` configs are the bodies given to `compute`."""
    contents = StoreContents(None)
    problems = []
    for module_name, body in config.items():
        where = f"config[{module_name!r}]"
        if module_name not in REGISTRY:
            problems.append(Problem(where, f"unknown module {module_name!r}"))
        elif not isinstance(body, Mapping):
            problems.append(Problem(where, "must be a mapping of config fields"))
        elif EXTENDS in body:
            problems.append(Problem(where, "extends is not available here"))
        else:
            contents.configs[module_name] = {MAIN: Definition(MAIN, dict(body), None, source=where)}
    if problems:
        raise DefinitionError(problems)
    return Definitions(contents, REGISTRY)


def _warn(messages) -> None:
    for message in messages:
        warnings.warn(message, stacklevel=3)
