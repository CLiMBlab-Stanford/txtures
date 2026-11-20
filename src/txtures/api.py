"""The Python API: the same requests as the command line, from Python.

`run` is `txtures run`: it computes a named input's request and writes the
same output table, then returns the table. `compute` computes features of
text given directly and returns the table without writing one. Both use the
configured item and resource stores, so repeated calls reuse earlier work.

When `execution.executor` is `slurm`, or `auto` and Slurm work is
missing where Slurm is set up, both submit a Slurm job for the missing
items and block until it ends; `local=True` computes in the calling
process instead. Interrupting the call cancels a job it submitted. The text
given to `compute` reaches the job's tasks through a file in the job
directory, and the results come back through the cache, so the returned
table is the one a local call returns. Inside a txtures Slurm task the API
always computes locally, so tasks never submit further jobs.

The API does not change the process environment. It passes its Hugging
Face cache and token to the Hugging Face libraries directly, so a host
program's own use of those libraries is unaffected.

When the definitions store, cache, and resources all come from
built-in defaults, the first call warns once and suggests how to point
txtures at shared locations.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from txtures import environment, installation
from txtures.definitions import Definitions
from txtures.definitions.resolution import EXTENDS, MAIN
from txtures.definitions.store import Definition, StoreContents
from txtures.engine import Executor, Plan, item_status, plan_request, run_request
from txtures.engine import slurm as slurm_jobs
from txtures.engine.tables import read_table, table_frame, table_is_current

__all__ = ["compute", "read_table", "run"]
from txtures.errors import DefinitionError, InputError, Problem, SettingsError, TxturesError
from txtures.inputs import InputSettings, read_texts
from txtures.modules import REGISTRY, Workload
from txtures.settings import Settings, resolve_settings, uses_default_locations
from txtures.store import EntryStore, read_json

if TYPE_CHECKING:
    import pandas as pd

Texts = str | Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]]


DEFAULT_LOCATIONS_MESSAGE = """\
txtures is using its default home, ~/txtures-home, so it shares no results with
others. To use a lab's shared home, run once:

    txtures site use /path/to/txtures-home

`txtures site ls` lists every setting. TXTURES_HOME, and the `home` argument
of the Python API, also choose a home."""

_warned_defaults = False


def settings(flags: Mapping[str, Any] | None = None, *, tolerate_stale: bool = False) -> Settings:
    """Resolve settings, with `flags` taking precedence.

    `flags` maps setting keys, such as `items` or `execution.device`, to
    values. Warns once per process when every shared location comes from
    built-in defaults.

    Settings that earlier versions kept outside the site, in a personal
    settings document or the installation record, are refused with a
    message saying how to move them; with `tolerate_stale` they are ignored
    instead.

    Raises:
        SettingsError: A settings document or value is invalid, or settings
            remain outside the site.
    """
    global _warned_defaults
    installed = installation.current()
    if installed is not None and installed.settings and not tolerate_stale:
        raise SettingsError(installation.stale_message(installed))
    resolved = resolve_settings(flags, tolerate_stale=tolerate_stale)
    if not _warned_defaults and uses_default_locations(resolved):
        _warned_defaults = True
        warnings.warn(DEFAULT_LOCATIONS_MESSAGE, DefaultLocationsWarning, stacklevel=3)
    return resolved


class DefaultLocationsWarning(UserWarning):
    """Every shared location comes from built-in defaults."""


def load_definitions(resolved: Settings) -> Definitions:
    """Load the definitions store in use, or the packaged definitions if it does not exist."""
    if resolved.definitions.is_dir():
        return Definitions.load(resolved.definitions)
    return Definitions.packaged()


def local_executor(plan: Plan, resolved: Settings, progress: Callable[[str], None] | None = None):
    """An executor for `plan` on this machine, using the configured stores."""
    return Executor(
        plan,
        EntryStore(resolved.cache),
        EntryStore(resolved.resource_store),
        environment.run_context(resolved),
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
    home: Path | str | None = None,
    definitions: Path | str | None = None,
    inputs: Path | str | None = None,
    outputs: Path | str | None = None,
    cache: Path | str | None = None,
    resources: Path | str | None = None,
    device: str | None = None,
    local: bool = False,
    nworkers: int | None = None,
    workers: int | None = None,
    slurm: Mapping[str, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Compute a request for the input `name` and return its output table.

    This is `txtures run NAME --workflow WORKFLOW --modules ...`: the table
    is written to the outputs directory, or reused if it is current, and
    then read with `read_table`. The location and `device` arguments
    override settings for this call.

    When the site's executor is `slurm`, a table that is not current is
    written by a Slurm job of `nworkers` tasks (default `slurm.workers`;
    `workers` is an older name for it),
    and the call blocks until the job ends. With `auto`, this happens only
    when Slurm work is missing and Slurm is set up here (see
    `slurm.decide`). If a job for the same table is
    already active, the call waits for it instead of submitting another.
    Interrupting the call cancels a job it submitted, but not one it only
    waits for. `local=True` computes in the calling process.

    `slurm` sets the size of every job the call submits, overriding what
    the modules and resources declare and the `slurm` settings: any of
    `time_hours`, `memory_gb`, `ncores`, and `ngpus`, as in
    `slurm={"ngpus": 0}`. It has no effect on work computed here.

    `progress`, if given, receives progress messages. Without it, messages
    about Slurm jobs and the choice of executor go to standard error, and
    others are discarded.

    Raises:
        TxturesError: The input, definitions, or settings are invalid, a
            computation or Slurm job failed, or `nworkers` was given with
            local execution.
    """
    resolved = settings(
        _flags(
            {
                "home": home,
                "definitions": definitions,
                "inputs": inputs,
                "outputs": outputs,
                "cache": cache,
                "resources": resources,
            },
            device,
        )
    )
    workers = _check_workers(resolved, local, nworkers, workers)
    override = slurm_jobs.size(slurm)
    plan = plan_request(resolved, load_definitions(resolved), name, workflow, modules)
    _warn(plan.notes)
    if _submits(plan, resolved, local, progress):
        if not table_is_current(plan):
            _on_slurm(
                plan,
                resolved,
                modules,
                workers,
                progress,
                lambda: table_is_current(plan),
                override=override,
            )
        return read_table(plan.table_directory)
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
    home: Path | str | None = None,
    definitions: Path | str | None = None,
    cache: Path | str | None = None,
    resources: Path | str | None = None,
    device: str | None = None,
    local: bool = False,
    nworkers: int | None = None,
    workers: int | None = None,
    slurm: Mapping[str, int] | None = None,
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
      them, as for `txtures run`.

    The table has the columns of an output table. No table is written, but
    computed items are stored and reused.

    When the site's executor is `slurm` and items are missing, a Slurm job
    of `nworkers` tasks (default `slurm.workers`) computes them, and the call
    blocks until the job ends. With `auto`, this happens only when
    Slurm work is missing and Slurm is set up here. The text and configs are written to the job
    directory for the tasks, so they must be JSON values. Waiting,
    interrupting, `local`, `slurm`, and `progress` behave as for `run`.

    Raises:
        TxturesError: The text, configs, or settings are invalid, a
            computation or Slurm job failed, or `nworkers` was given with
            local execution.
    """
    if config is not None and workflow is not None:
        raise DefinitionError([Problem("compute", "give either config or workflow, not both")])
    requested = [modules] if isinstance(modules, str) else list(modules)
    input_settings = InputSettings(
        region_delimiter=region_delimiter,
        sentence_delimiter=sentence_delimiter,
        region_joiner=region_joiner,
        sentence_joiner=sentence_joiner,
    )
    report = read_texts(texts, input_settings)
    if report.corpus is None:
        raise InputError(report.problems)
    _warn(str(warning) for warning in report.warnings)
    resolved = settings(
        _flags(
            {"home": home, "definitions": definitions, "cache": cache, "resources": resources},
            device,
        )
    )
    workers = _check_workers(resolved, local, nworkers, workers)
    override = slurm_jobs.size(slurm)
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
    submitting = _submits(plan, resolved, local, progress)
    if submitting and _missing(plan, resolved):
        given = {
            "documents": _documents(texts),
            "input_settings": input_settings.model_dump(mode="json"),
            "config": None if config is None else {k: dict(v) for k, v in config.items()},
        }
        _on_slurm(
            plan,
            resolved,
            requested,
            workers,
            progress,
            lambda: not _missing(plan, resolved),
            given,
            override=override,
        )
    run_executor = local_executor(plan, resolved, progress)
    run_executor.prepare_resources()
    if not submitting:
        run_executor.compute_items()
    return table_frame(plan, run_executor)


def job_plan(job: slurm_jobs.Job, resolved: Settings) -> Plan:
    """Plan the request of a submitted Slurm job again, as one of its tasks sees it.

    Raises:
        TxturesError: The input, text, definitions, or configs are now invalid.
    """
    request = job.request
    if job.source == "input":
        definitions = load_definitions(resolved)
        return plan_request(
            resolved, definitions, request["input"], request["workflow"], request["modules"]
        )
    return texts_plan(job.directory / "texts.json", request, resolved)


def texts_plan(path: Path, request: Mapping[str, Any], resolved: Settings) -> Plan:
    """Plan text given to `tx.compute` again from the `texts.json` written at submission.

    Raises:
        TxturesError: The text, definitions, or configs are now invalid.
    """
    given = read_json(path)
    report = read_texts(given["documents"], InputSettings(**given["input_settings"]))
    if report.corpus is None:
        raise InputError(report.problems)
    chosen = load_definitions(resolved) if given["config"] is None else _given(given["config"])
    return plan_request(
        resolved,
        chosen,
        report.corpus.name,
        request["workflow"],
        request["modules"],
        corpus=report.corpus,
    )


def _slurm() -> slurm_jobs.Slurm:
    return slurm_jobs.Slurm()


def _check_workers(
    resolved: Settings, local: bool, nworkers: int | None, workers: int | None
) -> int | None:
    """Return the number of array tasks asked for, from `nworkers` or its older name `workers`."""
    if nworkers is not None and workers is not None:
        raise TxturesError("give nworkers or workers, not both")
    count = nworkers if nworkers is not None else workers
    if count is None:
        return None
    if local or resolved.executor == "local":
        raise TxturesError("nworkers applies only to Slurm execution")
    if count < 1:
        raise TxturesError("nworkers must be at least 1")
    return count


def _submits(
    plan: Plan, resolved: Settings, local: bool, progress: Callable[[str], None] | None
) -> bool:
    """Whether a call submits to Slurm. Notes about the choice go to `progress` or standard error."""
    decision = slurm_jobs.decide(resolved, local, plan, EntryStore(resolved.cache))
    if decision.note:
        (progress or _stderr)(decision.note)
    return decision.submit


def _missing(plan: Plan, resolved: Settings) -> bool:
    """Whether any persisted item, or resource built on Slurm, of the plan is missing."""
    if slurm_jobs.slurm_resources(plan, EntryStore(resolved.resource_store)):
        return True
    return any(status.missing for status in item_status(plan, EntryStore(resolved.cache)))


def _documents(texts: Texts) -> dict[str, Any]:
    """The documents of `texts`, by ID, as JSON values that `read_texts` reads the same way."""
    if isinstance(texts, str):
        entries = {"1": texts}
    elif isinstance(texts, Mapping):
        entries = {str(document_id): value for document_id, value in texts.items()}
    else:
        entries = {str(number): value for number, value in enumerate(texts, start=1)}
    return {
        document_id: value if isinstance(value, str) else list(value)
        for document_id, value in entries.items()
    }


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _on_slurm(
    plan: Plan,
    resolved: Settings,
    modules: Sequence[str] | None,
    workers: int | None,
    progress: Callable[[str], None] | None,
    done: Callable[[], bool],
    texts: Mapping[str, Any] | None = None,
    *,
    override: Workload | None = None,
) -> None:
    """Compute a plan with a Slurm job, or wait for an active one, until `done()`.

    Raises:
        TxturesError: Submitting failed, or the job ended before `done()`.
        KeyboardInterrupt: The wait was interrupted; a job this call
            submitted has been cancelled.
    """
    report = progress or _stderr
    slurm = _slurm()
    name = plan.input_name if texts is None else slurm_jobs.COMPUTE_NAME
    existing = slurm_jobs.read_job(slurm_jobs.job_directory(resolved.outputs, plan, name))
    job = None
    if existing is not None:
        states = slurm.states(existing.job_id)
        if slurm_jobs.is_active(states):
            job = existing
            report(
                f"waiting for job {job.job_id}, which is already computing this request "
                f"({slurm_jobs.summarize(states)})"
            )
    submitted = job is None
    if job is None:
        count = workers if workers is not None else resolved["slurm.workers"]
        job = slurm_jobs.submit(
            plan, resolved, modules, count, slurm, texts=texts, override=override
        )
        noun = "task" if count == 1 else "tasks"
        report(f"submitted job {job.job_id} ({count} {noun}); logs: {job.directory / 'logs'}")
    try:
        slurm_jobs.wait(job, slurm, done, report)
    except KeyboardInterrupt:
        if submitted:
            try:
                slurm.cancel(job.job_id)
                report(f"cancelled job {job.job_id}")
            except TxturesError as error:
                report(f"could not cancel job {job.job_id}: {error}")
        else:
            report(f"stopped waiting; job {job.job_id} continues")
        raise


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
