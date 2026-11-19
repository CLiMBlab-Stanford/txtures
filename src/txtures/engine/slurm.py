"""Running a planned request as a Slurm job.

`submit` writes a job directory and submits one job array. Each array task
replans the request, checks that it still has the identity it was submitted
with, and computes its shard of the missing items and then any items left
over. For a named input, each task then writes the table if no other task
has: the first task to finish writes it, and the others find it current.
For text given to `tx.compute`, the tasks only compute items; the calling
process waits for the job with `wait` and assembles the table itself.

A job directory, `OUTPUTS/.txtures-jobs/NAME/WORKFLOW-DIGEST12/`, holds:

- `request.json`: the settings, request, and table identity the tasks use;
- `texts.json`: the text given to `tx.compute`, for those requests only;
- `job.sh`: the batch script;
- `job.json`: the job ID and submission time, written after `sbatch` succeeds;
- `logs/`: one log file per array task.

NAME is the input's name, or `tx.compute` for text given to `tx.compute`.

Slurm tasks never submit jobs. The batch script and `txtures worker` set
`TASK_VARIABLE`, which every process the task starts inherits; `submit`
refuses to run where it is set, and `submits` then chooses local execution.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtures.engine.planning import Plan, item_kind
from txtures.errors import TxturesError
from txtures.settings import Settings
from txtures.store import EntryStore, make_shared_dir, now, read_json, write_json

JOBS_DIRECTORY = ".txtures-jobs"
COMPUTE_NAME = "tx.compute"
"""The job directory name for requests from `tx.compute`, which have no input name."""
TASK_VARIABLE = "TXTURES_TASK"
"""Set, to the job directory, in Slurm tasks and every process they start."""
POLL_SECONDS = 10
"""How often `wait` checks a job's task states."""
GRACE_SECONDS = 120
"""How long `wait` lets a job show no active task before deciding it has failed.

`sacct` and `squeue` can omit a job for a short time after submission, and a
shared file system can show published results to the waiting host late."""
ACTIVE_STATES = frozenset(
    {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
)
FORWARDED_SETTINGS = (
    "home",
    "definitions",
    "inputs",
    "outputs",
    "cache",
    "resources",
    "execution.device",
    "execution.max_variants",
)
"""Settings a task receives from the submitting command, so it resolves the same request."""

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class Job:
    """A submitted job and the directory that records it."""

    directory: Path
    job_id: str
    workers: int
    table_identity: str
    request: Mapping[str, Any]

    @property
    def source(self) -> str:
        """`input` for a named input, or `texts` for text given to `tx.compute`."""
        return self.request.get("source", "input")


def job_directory(outputs: Path, plan: Plan, name: str | None = None) -> Path:
    """The job directory of a plan, under `name` (by default the plan's input name)."""
    return outputs / JOBS_DIRECTORY / (name or plan.input_name) / plan.table_directory.name


def in_task(env: Mapping[str, str] | None = None) -> bool:
    """Whether this process runs inside a txtures Slurm task."""
    env = os.environ if env is None else env
    return bool(env.get(TASK_VARIABLE))


def available(settings: Settings) -> bool:
    """Whether Slurm is set up here: `slurm.partition` is set and `sbatch` is on PATH."""
    return bool(settings["slurm.partition"]) and shutil.which("sbatch") is not None


def intensive_work(plan: Plan, items: EntryStore) -> list[str]:
    """Names of the intensive modules with results missing from `items`, in plan order."""
    found = []
    for planned in plan.variants:
        if not (planned.module.persist and planned.module.intensive) or planned.name in found:
            continue
        kind = item_kind(planned.name)
        if any(
            items.get(kind, planned.item_identity(document)) is None
            for document in plan.corpus.documents
        ):
            found.append(planned.name)
    return found


@dataclass(frozen=True)
class Decision:
    """Whether to submit a request to Slurm, and a note explaining an unexpected choice."""

    submit: bool
    note: str | None = None


def decide(settings: Settings, local: bool, plan: Plan, items: EntryStore) -> Decision:
    """Decide whether a request is submitted to Slurm or computed here.

    Nothing is submitted when `local` is true, when the executor is `local`,
    or when this process runs inside a txtures Slurm task. The `slurm`
    executor always submits. The `auto` executor submits when the plan has
    intensive work missing from `items` and Slurm is `available`.

    The note says why a request is computed here when Slurm execution was
    expected: inside a task, or with intensive work missing where Slurm is
    half set up (`sbatch` without `slurm.partition`, or the reverse). Where
    neither is present, `auto`, the default, computes here without a note.
    """
    if local or settings.executor == "local":
        return Decision(False)
    if in_task():
        return Decision(
            False,
            "computing here: this runs inside a txtures Slurm task, "
            "which never submits further jobs",
        )
    if settings.executor == "slurm":
        return Decision(True)
    work = intensive_work(plan, items)
    if not work:
        return Decision(False)
    if not available(settings):
        partition = bool(settings["slurm.partition"])
        if not partition and shutil.which("sbatch") is None:
            return Decision(False)
        missing = "no sbatch" if partition else "slurm.partition is unset"
        return Decision(False, f"computing {', '.join(work)} here: Slurm is not set up ({missing})")
    return Decision(True, f"submitting to Slurm: {', '.join(work)} results are missing")


class Slurm:
    """The Slurm commands txtures uses. `run` is replaceable for testing."""

    def __init__(self, run: Runner = subprocess.run):
        self._run = run

    def _call(self, command: list[str]) -> str:
        if self._run is subprocess.run and shutil.which(command[0]) is None:
            raise TxturesError(f"{command[0]} is not available; Slurm execution needs it")
        result = self._run(command, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise TxturesError(f"{shlex.join(command)} failed: {message}")
        return result.stdout

    def submit(self, script: Path) -> str:
        """Submit a batch script and return its job ID."""
        output = self._call(["sbatch", "--parsable", str(script)]).strip()
        return output.split(";")[0]

    def cancel(self, job_id: str) -> None:
        """Cancel every task of a job."""
        self._call(["scancel", job_id])

    def states(self, job_id: str) -> dict[str, str]:
        """Return the state of each task of a job, by task ID.

        Uses `sacct`, which remembers finished jobs, and falls back to
        `squeue`, which lists only active ones.
        """
        try:
            output = self._call(["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "JobID,State"])
            separator = "|"
        except TxturesError:
            output = self._call(["squeue", "-h", "-r", "-j", job_id, "-o", "%i %T"])
            separator = " "
        states = {}
        for line in output.splitlines():
            if separator in line:
                task, state = line.split(separator, 1)
                states[task.strip()] = state.split()[0] if state.split() else "UNKNOWN"
        return states


def summarize(states: Mapping[str, str]) -> str:
    """Describe task states compactly, such as `2 RUNNING, 1 PENDING`."""
    counter = Counter(states.values())
    return ", ".join(f"{n} {state}" for state, n in counter.most_common()) or "no tasks found"


def is_active(states: Mapping[str, str]) -> bool:
    return any(state in ACTIVE_STATES for state in states.values())


def batch_script(
    directory: Path, name: str, workers: int, slurm: Mapping[str, Any], command: Sequence[str]
) -> str:
    """The batch script for a request's job array."""
    if not slurm.get("partition"):
        raise TxturesError(
            "Slurm execution needs slurm.partition; set it with `txtures site set slurm.partition=NAME`"
        )
    options = [
        f"--job-name=txtures-{name}",
        f"--output={directory / 'logs' / '%A_%a.out'}",
        f"--array=0-{workers - 1}",
        f"--partition={slurm['partition']}",
        f"--time={slurm['time_hours'] * 60}",
        f"--mem={slurm['memory_gb']}G",
        f"--cpus-per-task={slurm['cpus']}",
    ]
    if slurm.get("account"):
        options.append(f"--account={slurm['account']}")
    if slurm.get("gpus"):
        options.append(f"--gres=gpu:{slurm['gpus']}")
    options += list(slurm.get("options") or [])
    lines = ["#!/bin/sh", *(f"#SBATCH {shlex.quote(option)}" for option in options)]
    lines.append(f"export {TASK_VARIABLE}={shlex.quote(str(directory))}")
    lines.append(f'exec {shlex.join(command)} "${{SLURM_ARRAY_TASK_ID:-0}}"')
    return "\n".join(lines) + "\n"


def read_job(directory: Path) -> Job | None:
    """Read a job directory's record, or `None` if the job was never submitted."""
    try:
        record = read_json(directory / "job.json")
        request = read_json(directory / "request.json")
        return Job(
            directory=directory,
            job_id=str(record["job_id"]),
            workers=int(request["workers"]),
            table_identity=str(request["table_identity"]),
            request=request,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def jobs(outputs: Path, names: Sequence[str] | None = None) -> list[Job]:
    """Every recorded job, or those of the given inputs."""
    root = outputs / JOBS_DIRECTORY
    if not root.is_dir():
        return []
    inputs = [root / name for name in names] if names else sorted(root.iterdir())
    found = []
    for input_directory in inputs:
        if input_directory.is_dir():
            for directory in sorted(input_directory.iterdir()):
                job = read_job(directory)
                if job is not None:
                    found.append(job)
    return found


def submit(
    plan: Plan,
    settings: Settings,
    modules: Sequence[str] | None,
    workers: int,
    slurm: Slurm,
    *,
    texts: Mapping[str, Any] | None = None,
) -> Job:
    """Submit a request as a job array of `workers` tasks.

    Without `texts`, the request is the plan's named input. With `texts`,
    it is text given to `tx.compute`: `texts` holds the `documents` (a
    mapping from document ID to document), the `input_settings` that divide
    them, and the `config` bodies, or `None` to use the definitions store.

    Raises:
        TxturesError: This process runs inside a txtures Slurm task, the
            Slurm settings are incomplete, or `sbatch` failed.
    """
    if in_task():
        raise TxturesError("a txtures Slurm task cannot submit Slurm jobs")
    name = plan.input_name if texts is None else COMPUTE_NAME
    directory = job_directory(settings.outputs, plan, name)
    make_shared_dir(directory / "logs")
    request = {
        "source": "input" if texts is None else "texts",
        "input": name,
        "workflow": plan.workflow,
        "modules": list(modules) if modules is not None else None,
        "table_identity": plan.table_identity,
        "workers": workers,
        "settings": {
            key: str(settings[key]) if isinstance(settings[key], Path) else settings[key]
            for key in FORWARDED_SETTINGS
        },
        "created": now(),
    }
    if texts is not None:
        # Key order is kept: it orders the documents and a config's sweeps.
        try:
            encoded = json.dumps(dict(texts), indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise TxturesError(
                f"the texts and configs must be JSON values for Slurm: {error}"
            ) from None
        (directory / "texts.json").write_text(encoded + "\n", encoding="utf-8")
    write_json(directory / "request.json", request)
    command = [sys.executable, "-I", "-m", "txtures", "worker", str(directory)]
    script = directory / "job.sh"
    script.write_text(
        batch_script(directory, name, workers, settings.group("slurm"), command),
        encoding="utf-8",
    )
    (directory / "job.json").unlink(missing_ok=True)
    job_id = slurm.submit(script)
    write_json(directory / "job.json", {"job_id": job_id, "submitted": now()})
    job = read_job(directory)
    assert job is not None
    return job


def wait(
    job: Job,
    slurm: Slurm,
    done: Callable[[], bool],
    report: Callable[[str], None],
    *,
    poll: float | None = None,
    grace: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until a job has no active task and `done()` is true.

    `report` receives the job's task states whenever they change. A task
    may fail while another completes its work, so success is judged by
    `done`, not by task states.

    Raises:
        TxturesError: The job has had no active task for `grace` seconds
            and `done()` is still false, or its states cannot be read.
    """
    poll = POLL_SECONDS if poll is None else poll
    grace = GRACE_SECONDS if grace is None else grace
    last = None
    quiet_since = clock()
    while True:
        states = slurm.states(job.job_id)
        summary = summarize(states)
        if states and summary != last:
            report(f"job {job.job_id}: {summary}")
            last = summary
        if is_active(states):
            quiet_since = clock()
        elif done():
            return
        elif clock() - quiet_since >= grace:
            raise TxturesError(
                f"job {job.job_id} ended without completing the request ({summary}); "
                f"see the logs in {job.directory / 'logs'}"
            )
        sleep(poll)
