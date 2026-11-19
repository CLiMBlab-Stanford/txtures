"""Running a planned request as a Slurm job.

`submit` writes a job directory and submits one job array. Each array task
replans the request, checks that it still has the identity it was submitted
with, computes its shard of the missing items and then any items left over,
and writes the table if no other task has. The first task to finish writes
it; the others find it current.

A job directory, `DERIVATIVES/.txtures-jobs/NAME/WORKFLOW-DIGEST12/`, holds:

- `request.json`: the settings, request, and table identity the tasks use;
- `job.sh`: the batch script;
- `job.json`: the job ID and submission time, written after `sbatch` succeeds;
- `logs/`: one log file per array task.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtures.engine.planning import Plan
from txtures.errors import TxturesError
from txtures.settings import Settings
from txtures.store import make_shared_dir, now, read_json, write_json

JOBS_DIRECTORY = ".txtures-jobs"
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


def job_directory(outputs: Path, plan: Plan) -> Path:
    return outputs / JOBS_DIRECTORY / plan.input_name / plan.table_directory.name


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
) -> Job:
    """Submit a request as a job array of `workers` tasks.

    Raises:
        TxturesError: Slurm settings are incomplete or `sbatch` failed.
    """
    directory = job_directory(settings.outputs, plan)
    make_shared_dir(directory / "logs")
    request = {
        "input": plan.input_name,
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
    write_json(directory / "request.json", request)
    command = [sys.executable, "-I", "-m", "txtures", "worker", str(directory)]
    script = directory / "job.sh"
    script.write_text(
        batch_script(directory, plan.input_name, workers, settings.group("slurm"), command),
        encoding="utf-8",
    )
    (directory / "job.json").unlink(missing_ok=True)
    job_id = slurm.submit(script)
    write_json(directory / "job.json", {"job_id": job_id, "submitted": now()})
    job = read_job(directory)
    assert job is not None
    return job
