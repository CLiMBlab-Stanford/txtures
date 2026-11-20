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

A missing resource whose workload is `slurm` is built by a job of its own,
submitted first: `OUTPUTS/.txtures-jobs/.resources/KIND-IDENTITY12/`, shaped
by the resource's workload (`job_options`). The request's job array then starts only after
every build succeeds, and Slurm cancels it if a build fails. A build job
already active for the same resource is waited for, not repeated.

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

from txtures.engine.planning import (
    Plan,
    PlannedResource,
    PlannedVariant,
    item_kind,
    resource_record,
)
from txtures.errors import TxturesError
from txtures.identity import digest
from txtures.modules import Workload
from txtures.settings import Settings
from txtures.store import EntryStore, make_shared_dir, now, read_json, write_json

JOBS_DIRECTORY = ".txtures-jobs"
RESOURCE_JOBS = ".resources"
"""The directory, inside `JOBS_DIRECTORY`, of resource build jobs."""
BATCH_JOBS = ".batches"
"""The directory, inside `JOBS_DIRECTORY`, of jobs that compute several inputs."""
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
    "execution.runtime",
    "corpora.ptb3",
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


def forwarded(settings: Settings) -> dict[str, Any]:
    """The `FORWARDED_SETTINGS` that are set, as JSON values."""
    return {
        key: str(settings[key]) if isinstance(settings[key], Path) else settings[key]
        for key in FORWARDED_SETTINGS
        if settings[key] is not None
    }


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


def slurm_resources(plan: Plan, resources: EntryStore) -> list[PlannedResource]:
    """Resources the plan needs, that `resources` lacks, and whose workload is `slurm`."""
    return [
        resource
        for resource in plan.resources()
        if resource.resource.workload(resource.resolved).slurm
        and resources.get(resource.kind, resource.identity) is None
    ]


def slurm_work(plan: Plan, items: EntryStore) -> list[PlannedVariant]:
    """Persisted variants whose workload is `slurm` and that have items missing from `items`."""
    found = []
    for planned in plan.variants:
        if not planned.module.persist or not planned.module.workload(planned.variant.config).slurm:
            continue
        kind = item_kind(planned.name)
        if any(
            items.get(kind, planned.item_identity(document)) is None
            for document in plan.corpus.documents
        ):
            found.append(planned)
    return found


SIZE_FIELDS = ("time_hours", "memory_gb", "ncores", "ngpus")
"""The `Workload` fields that shape a job, and that a call may override."""


def size(values: Mapping[str, Any] | None) -> Workload | None:
    """A per-call job size from `values`, such as `{"ngpus": 0}`, or `None` for none.

    Raises:
        TxturesError: A key is unknown or a value is not a whole number
            (at least 1, or at least 0 for `ngpus`).
    """
    if not values:
        return None
    unknown = sorted(set(values) - set(SIZE_FIELDS))
    if unknown:
        raise TxturesError(f"unknown job size {', '.join(unknown)}; use {', '.join(SIZE_FIELDS)}")
    for key, value in values.items():
        smallest = 0 if key == "ngpus" else 1
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < smallest
        ):
            raise TxturesError(f"{key} must be a whole number of at least {smallest}")
    return Workload(slurm=True, **dict(values))


def job_options(
    settings: Settings, workloads: Sequence[Workload], override: Workload | None = None
) -> dict[str, Any]:
    """The Slurm settings of a job: the site's, raised to what its `slurm` workloads need.

    Each numeric field takes the value set in `override`, the per-call size,
    if any; otherwise the largest value any `slurm` workload gives;
    otherwise the site's setting. A job that needs no GPU goes to
    `slurm.cpu_partition` when that is set.
    """
    slurm = settings.group("slurm")
    declared = [w for w in workloads if w.slurm]

    def largest(field: str, setting: str) -> Any:
        values = [getattr(w, field) for w in declared if getattr(w, field) is not None]
        return max(values) if values else slurm[setting]

    options = {
        **slurm,
        "time_hours": largest("time_hours", "time_hours"),
        "memory_gb": largest("memory_gb", "memory_gb"),
        "cpus": largest("ncores", "cpus"),
        "gpus": largest("ngpus", "gpus"),
    }
    if override is not None:
        for field, setting in zip(
            SIZE_FIELDS, ("time_hours", "memory_gb", "cpus", "gpus"), strict=True
        ):
            value = getattr(override, field)
            if value is not None:
                options[setting] = value
    if not options["gpus"] and slurm.get("cpu_partition"):
        options["partition"] = slurm["cpu_partition"]
    return options


def gpu_count(workload: Workload, settings: Settings, override: Workload | None = None) -> int:
    """How many GPUs a workload's job needs: from `override`, the workload, or `slurm.gpus`.

    A workload that does not run on Slurm needs none.
    """
    if not workload.slurm:
        return 0
    if override is not None and override.ngpus is not None:
        return override.ngpus
    return workload.ngpus if workload.ngpus is not None else settings["slurm.gpus"]


def cpu_only(options: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Job options with no GPU, on `slurm.cpu_partition` when that is set."""
    options = {**options, "gpus": 0}
    if settings["slurm.cpu_partition"]:
        options["partition"] = settings["slurm.cpu_partition"]
    return options


@dataclass(frozen=True)
class Decision:
    """Whether to submit a request to Slurm, and a note explaining an unexpected choice."""

    submit: bool
    note: str | None = None


def decide(
    settings: Settings,
    local: bool,
    plan: Plan,
    items: EntryStore,
    resources: EntryStore | None = None,
) -> Decision:
    """Decide whether a request is submitted to Slurm or computed here.

    Nothing is submitted when `local` is true, when the executor is `local`,
    or when this process runs inside a txtures Slurm task. The `slurm`
    executor always submits. The `auto` executor submits when the plan has
    Slurm work missing: items of a module whose workload is `slurm` missing
    from `items`, or such a resource missing from `resources` (by default
    the store in `settings`). Slurm must be `available`.

    The note says why a request is computed here when Slurm execution was
    expected: inside a task, or with Slurm work missing where Slurm is
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
    store = resources if resources is not None else EntryStore(settings.resource_store)
    modules = list(dict.fromkeys(planned.name for planned in slurm_work(plan, items)))
    builds = [f"resource {r.kind}" for r in slurm_resources(plan, store)]
    work = modules + builds
    if not work:
        return Decision(False)
    if not available(settings):
        partition = bool(settings["slurm.partition"])
        if not partition and shutil.which("sbatch") is None:
            # Without Slurm, light work is silent, but a build can take hours.
            note = f"building {', '.join(builds)} here: Slurm is not set up" if builds else None
            return Decision(False, note)
        missing = "no sbatch" if partition else "slurm.partition is unset"
        return Decision(False, f"computing {', '.join(work)} here: Slurm is not set up ({missing})")
    parts = [f"{name} results" for name in modules] + builds
    verb = "are" if modules or len(builds) > 1 else "is"
    return Decision(True, f"submitting to Slurm: {', '.join(parts)} {verb} missing")


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
    directory: Path,
    name: str,
    workers: int,
    slurm: Mapping[str, Any],
    command: Sequence[str],
    after: Sequence[str] = (),
    after_any: Sequence[str] = (),
) -> str:
    """The batch script for a job array.

    The array starts only after the jobs in `after` succeed, and Slurm
    cancels it if one fails; it also waits for the jobs in `after_any` to
    end, however they end.
    """
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
    dependencies = []
    if after:
        dependencies.append(f"afterok:{':'.join(after)}")
    if after_any:
        dependencies.append(f"afterany:{':'.join(after_any)}")
    if dependencies:
        options.append(f"--dependency={','.join(dependencies)}")
    if after:
        options.append("--kill-on-invalid-dep=yes")
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


def _subrequest(plan: Plan, modules: Sequence[str] | None) -> dict[str, Any]:
    return {
        "input": plan.input_name,
        "workflow": plan.workflow,
        "modules": list(modules) if modules is not None else None,
        "table_identity": plan.table_identity,
    }


def batch_directory(outputs: Path, plans: Sequence[Plan]) -> Path:
    """The job directory of a batch: its workflow and a digest of its tables' identities."""
    identity = digest([plan.table_identity for plan in plans])
    return outputs / JOBS_DIRECTORY / BATCH_JOBS / f"{plans[0].workflow}-{identity[:12]}"


def gpu_directory(directory: Path) -> Path:
    """The directory of the GPU job that a request's job waits for."""
    return directory.with_name(directory.name + ".gpu")


def _write_job(
    directory: Path,
    request: dict[str, Any],
    name: str,
    workers: int,
    options: Mapping[str, Any],
    slurm: Slurm,
    after: Sequence[str] = (),
    after_any: Sequence[str] = (),
) -> Job:
    make_shared_dir(directory / "logs")
    write_json(directory / "request.json", request)
    command = [sys.executable, "-I", "-m", "txtures", "worker", str(directory)]
    script = directory / "job.sh"
    script.write_text(
        batch_script(directory, name, workers, options, command, after, after_any),
        encoding="utf-8",
    )
    (directory / "job.json").unlink(missing_ok=True)
    job_id = slurm.submit(script)
    write_json(directory / "job.json", {"job_id": job_id, "submitted": now()})
    job = read_job(directory)
    assert job is not None
    return job


def submit(
    plans: Plan | Sequence[Plan],
    settings: Settings,
    modules: Sequence[str] | None,
    workers: int,
    slurm: Slurm,
    *,
    texts: Mapping[str, Any] | None = None,
    resources: EntryStore | None = None,
    override: Workload | None = None,
) -> Job:
    """Submit the jobs that compute one or more requests, and return the last.

    GPU hours cost money, so GPU work runs in a job of its own that never
    waits idle:

    1. Each missing resource with a `slurm` workload, and each missing
       resource that GPU work needs, is built by a job of its own
       (`submit_build`), on CPUs unless its workload asks for GPUs.
    2. If any request has GPU work missing (items of a `slurm` workload
       that needs GPUs, `gpu_count`), a GPU job array of `workers` tasks
       starts after the builds. Each task computes only those items, for
       every request, taking its share first and then any item nobody
       holds, and exits as soon as nothing is left for it to take.
    3. A CPU job then computes everything else and writes the tables. It
       starts once the GPU job has ended, however it ended, and writes only
       tables whose GPU results are all present. Text given to `tx.compute`
       gets this job only when it has CPU work on Slurm, since the caller
       builds its own table.

    The GPU job is shaped by its workloads (`job_options`) and `override`.
    The CPU job has no GPU and uses `slurm.cpu_partition` when it is set; it
    has `workers` tasks when it has `slurm` work of its own, and one
    otherwise. `override`'s time, memory, and cores apply to every job.

    Without `texts`, the requests are the plans' named inputs. With `texts`
    (one plan), it is text given to `tx.compute`: `texts` holds the
    `documents`, the `input_settings` that divide them, and the `config`
    bodies, or `None` to use the definitions store.

    Raises:
        TxturesError: This process runs inside a txtures Slurm task, the
            Slurm settings are incomplete, or `sbatch` failed.
    """
    if in_task():
        raise TxturesError("a txtures Slurm task cannot submit Slurm jobs")
    plans = [plans] if isinstance(plans, Plan) else list(plans)
    store = resources if resources is not None else EntryStore(settings.resource_store)
    items = EntryStore(settings.cache)
    if texts is not None:
        name, directory = COMPUTE_NAME, job_directory(settings.outputs, plans[0], COMPUTE_NAME)
    elif len(plans) == 1:
        name, directory = plans[0].input_name, job_directory(settings.outputs, plans[0])
    else:
        name, directory = "batch", batch_directory(settings.outputs, plans)

    memo: dict[str, Job] = {}
    builds: dict[str, Job] = {}
    gpu_variants: list[list[str]] = []
    gpu_loads: list[Workload] = []
    cpu_loads: list[Workload] = []
    for plan in plans:
        work = slurm_work(plan, items)
        gpu = [p for p in work if gpu_count(p.module.workload(p.variant.config), settings) > 0]
        if override is not None and override.ngpus is not None:
            gpu = work if override.ngpus > 0 else []
        gpu_variants.append(sorted({p.identity for p in gpu}))
        gpu_loads += [p.module.workload(p.variant.config) for p in gpu]
        cpu_loads += [p.module.workload(p.variant.config) for p in work if p not in gpu]
        needed = {r.identity: r for r in slurm_resources(plan, store)}
        for planned in gpu:
            for resource in plan.resources():
                if resource.identity in {r.identity for r in planned.resources.values()} and (
                    store.get(resource.kind, resource.identity) is None
                ):
                    needed.setdefault(resource.identity, resource)
        for resource in needed.values():
            build = submit_build(resource, settings, slurm, memo, override)
            builds[build.job_id] = build
    build_records = [
        {"job_id": b.job_id, "kind": b.request["kind"], "logs": str(b.directory / "logs")}
        for b in builds.values()
    ]
    base = {
        "workflow": plans[0].workflow,
        "modules": list(modules) if modules is not None else None,
        "settings": forwarded(settings),
        "created": now(),
        "builds": build_records,
    }

    gpu_job = None
    if any(gpu_variants):
        gdir = gpu_directory(directory)
        make_shared_dir(gdir / "logs")
        subrequests = []
        for plan, variants in zip(plans, gpu_variants, strict=True):
            sub = {**_subrequest(plan, modules), "variants": variants}
            if texts is not None:
                sub["texts"] = str(gdir / "texts.json")
            subrequests.append(sub)
        if texts is not None:
            _write_texts(gdir / "texts.json", texts)
        request = {
            **base,
            "source": "gpu",
            "input": ",".join(plan.input_name for plan in plans) if texts is None else name,
            "requests": subrequests,
            "table_identity": digest([plan.table_identity for plan in plans]),
            "workers": workers,
        }
        gpu_job = _write_job(
            gdir,
            request,
            f"{name}-gpu",
            workers,
            job_options(settings, gpu_loads, override),
            slurm,
            after=list(builds),
        )
    if texts is not None and not cpu_loads and gpu_job is not None:
        return gpu_job

    skips = {
        plan.table_identity: variants for plan, variants in zip(plans, gpu_variants, strict=True)
    }
    cpu_workers = workers if cpu_loads or gpu_job is None else 1
    if texts is not None:
        _write_texts(directory / "texts.json", texts, make=True)
        request = {
            **base,
            "source": "texts",
            "input": name,
            "table_identity": plans[0].table_identity,
            "skip": gpu_variants[0],
        }
    elif len(plans) == 1:
        request = {**base, "source": "input", **_subrequest(plans[0], modules)}
        request["skip"] = gpu_variants[0]
    else:
        request = {
            **base,
            "source": "batch",
            "input": ",".join(plan.input_name for plan in plans),
            "requests": [
                {**_subrequest(p, modules), "skip": skips[p.table_identity]} for p in plans
            ],
            "table_identity": digest([plan.table_identity for plan in plans]),
        }
    request["workers"] = cpu_workers
    if gpu_job is not None:
        request["after_jobs"] = [
            {"job_id": gpu_job.job_id, "logs": str(gpu_job.directory / "logs")}
        ]
    options = cpu_only(job_options(settings, cpu_loads, override), settings)
    return _write_job(
        directory,
        request,
        name,
        cpu_workers,
        options,
        slurm,
        after=list(builds),
        after_any=[gpu_job.job_id] if gpu_job is not None else [],
    )


def _write_texts(path: Path, texts: Mapping[str, Any], make: bool = False) -> None:
    """Write the text and configs given to `tx.compute`, keeping key order, which
    orders the documents and a config's sweeps."""
    if make:
        make_shared_dir(path.parent)
    try:
        encoded = json.dumps(dict(texts), indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise TxturesError(
            f"the texts and configs must be JSON values for Slurm: {error}"
        ) from None
    path.write_text(encoded + "\n", encoding="utf-8")


def build_directory(outputs: Path, resource: PlannedResource) -> Path:
    """The job directory of a resource build."""
    return outputs / JOBS_DIRECTORY / RESOURCE_JOBS / f"{resource.kind}-{resource.identity[:12]}"


def _build_options(
    settings: Settings, workload: Workload, override: Workload | None
) -> dict[str, Any]:
    """A build job's options: on CPUs unless its workload declares GPUs."""
    options = job_options(settings, [workload], override)
    if workload.slurm and workload.ngpus:
        return {**options, "gpus": workload.ngpus}
    return cpu_only(options, settings)


def submit_build(
    resource: PlannedResource,
    settings: Settings,
    slurm: Slurm,
    memo: dict[str, Job] | None = None,
    override: Workload | None = None,
) -> Job:
    """Submit a job that builds one resource, or return the active job already building it.

    Each missing dependency whose workload is `slurm` gets a build job of
    its own first, and this job starts after those succeed; other missing
    dependencies are built inside this job. `memo` maps identities to jobs
    already submitted by the caller, so a shared dependency is submitted once.

    Raises:
        TxturesError: This process runs inside a txtures Slurm task, the
            Slurm settings are incomplete, or `sbatch` failed.
    """
    if in_task():
        raise TxturesError("a txtures Slurm task cannot submit Slurm jobs")
    memo = {} if memo is None else memo
    if resource.identity in memo:
        return memo[resource.identity]
    directory = build_directory(settings.outputs, resource)
    existing = read_job(directory)
    if existing is not None and is_active(slurm.states(existing.job_id)):
        memo[resource.identity] = existing
        return existing
    store = EntryStore(settings.resource_store)
    after = [
        submit_build(dependency, settings, slurm, memo, override).job_id
        for dependency in resource.dependencies.values()
        if dependency.resource.workload(dependency.resolved).slurm
        and store.get(dependency.kind, dependency.identity) is None
    ]
    make_shared_dir(directory / "logs")
    request = {
        "source": "resource",
        "input": "resources",
        "workflow": resource.kind,
        **resource_record(resource),
        "after": after,
        "table_identity": resource.identity,
        "workers": 1,
        "settings": forwarded(settings),
        "created": now(),
    }
    write_json(directory / "request.json", request)
    command = [sys.executable, "-I", "-m", "txtures", "worker", str(directory)]
    script = directory / "job.sh"
    script.write_text(
        batch_script(
            directory,
            f"build-{resource.kind}",
            1,
            _build_options(settings, resource.resource.workload(resource.resolved), override),
            command,
            after=after,
        ),
        encoding="utf-8",
    )
    (directory / "job.json").unlink(missing_ok=True)
    job_id = slurm.submit(script)
    write_json(directory / "job.json", {"job_id": job_id, "submitted": now()})
    job = read_job(directory)
    assert job is not None
    memo[resource.identity] = job
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
            builds = "".join(
                f" and of job {b['job_id']} in {b['logs']}"
                for b in [*job.request.get("builds", []), *job.request.get("after_jobs", [])]
            )
            raise TxturesError(
                f"job {job.job_id} ended without completing the request ({summary}); "
                f"see the logs in {job.directory / 'logs'}{builds}"
            )
        sleep(poll)
