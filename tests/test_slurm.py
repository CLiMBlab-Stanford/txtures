import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fake_modules
import pytest
from conftest import write
from fake_modules import ORDER
from test_cli import fake_engine, isolated_settings  # noqa: F401 - fixtures

from txtures.cli import main
from txtures.engine import slurm as slurm_jobs
from txtures.engine.slurm import Slurm, batch_script, summarize
from txtures.errors import TxturesError
from txtures.modules import Workload


def on_slurm(self, spec):
    return Workload(slurm=True)


def here(self, spec):
    return Workload()


SLURM = {
    "partition": "gpu",
    "account": None,
    "time_hours": 2,
    "memory_gb": 16,
    "cpus": 4,
    "gpus": 1,
    "workers": 1,
    "options": [],
}


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


# Batch scripts and Slurm commands


def test_batch_script_requests_resources_and_runs_a_task():
    script = batch_script(
        Path("/jobs/x"),
        "doc",
        3,
        {**SLURM, "account": "lab", "options": ["--exclude=node 1"]},
        ["/env/bin/python", "-I", "-m", "txtures", "worker", "/jobs/x"],
    )
    lines = script.splitlines()
    assert lines[0] == "#!/bin/sh"
    assert "#SBATCH --array=0-2" in lines
    assert "#SBATCH --partition=gpu" in lines
    assert "#SBATCH --time=120" in lines
    assert "#SBATCH --mem=16G" in lines
    assert "#SBATCH --gres=gpu:1" in lines
    assert "#SBATCH --account=lab" in lines
    assert "#SBATCH '--exclude=node 1'" in lines
    assert lines[-2] == "export TXTURES_TASK=/jobs/x"
    assert lines[-1] == (
        'exec /env/bin/python -I -m txtures worker /jobs/x "${SLURM_ARRAY_TASK_ID:-0}"'
    )


def test_batch_script_without_gpus_or_partition():
    script = batch_script(Path("/j"), "doc", 1, {**SLURM, "gpus": 0}, ["txtures"])
    assert "--gres" not in script
    with pytest.raises(TxturesError, match="slurm.partition"):
        batch_script(Path("/j"), "doc", 1, {**SLURM, "partition": None}, ["txtures"])


def test_states_from_sacct_and_squeue():
    calls = []

    def run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "sacct":
            return completed("7_0|COMPLETED\n7_1|RUNNING\n7_2|CANCELLED by 12\n")
        raise AssertionError(command)

    states = Slurm(run).states("7")
    assert states == {"7_0": "COMPLETED", "7_1": "RUNNING", "7_2": "CANCELLED"}
    assert slurm_jobs.is_active(states)
    assert summarize(states) == "1 COMPLETED, 1 RUNNING, 1 CANCELLED"

    def without_sacct(command, **kwargs):
        if command[0] == "sacct":
            return completed(returncode=1, stderr="accounting storage is disabled")
        return completed("8_0 PENDING\n")

    assert Slurm(without_sacct).states("8") == {"8_0": "PENDING"}


def test_submit_reports_sbatch_failure():
    def run(command, **kwargs):
        return completed(returncode=1, stderr="invalid partition")

    with pytest.raises(TxturesError, match="invalid partition"):
        Slurm(run).submit(Path("/j/job.sh"))


def test_cancel_calls_scancel():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return completed()

    Slurm(run).cancel("7")
    assert calls == [["scancel", "7"]]


# Waiting for a job


def _job(tmp_path):
    return slurm_jobs.Job(tmp_path, "7", 1, "x", {})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_returns_once_no_task_is_active_and_the_work_is_done(tmp_path):
    class States(Slurm):
        def __init__(self, sequence):
            super().__init__(run=None)
            self.sequence = list(sequence)

        def states(self, job_id):
            return self.sequence.pop(0) if len(self.sequence) > 1 else self.sequence[0]

    clock = Clock()
    reports = []
    slurm = States([{}, {"7_0": "PENDING"}, {"7_0": "RUNNING"}, {"7_0": "COMPLETED"}])
    slurm_jobs.wait(
        _job(tmp_path),
        slurm,
        lambda: len(slurm.sequence) == 1,
        reports.append,
        poll=10,
        grace=60,
        clock=clock,
        sleep=clock.sleep,
    )
    assert reports == ["job 7: 1 PENDING", "job 7: 1 RUNNING", "job 7: 1 COMPLETED"]


def test_wait_fails_when_the_job_ends_without_the_work(tmp_path):
    class Ended(Slurm):
        def __init__(self):
            super().__init__(run=None)

        def states(self, job_id):
            return {"7_0": "FAILED"}

    clock = Clock()
    with pytest.raises(TxturesError, match="ended without completing the request"):
        slurm_jobs.wait(
            _job(tmp_path),
            Ended(),
            lambda: False,
            lambda m: None,
            poll=10,
            grace=60,
            clock=clock,
            sleep=clock.sleep,
        )
    assert clock.now == 60


# Tasks never submit


def test_a_task_never_submits(monkeypatch, tmp_path):
    from txtures.settings import resolve_settings

    settings = resolve_settings(
        {"home": str(tmp_path), "execution.executor": "slurm"}, env={}, user_home=tmp_path
    )
    # These choices are made without looking at the plan.
    assert slurm_jobs.decide(settings, False, None, None).submit
    assert not slurm_jobs.decide(settings, True, None, None).submit
    monkeypatch.setenv(slurm_jobs.TASK_VARIABLE, "/jobs/x")
    decision = slurm_jobs.decide(settings, False, None, None)
    assert not decision.submit and "inside a txtures Slurm task" in decision.note
    with pytest.raises(TxturesError, match="cannot submit"):
        slurm_jobs.submit(None, settings, None, 1, FakeSlurm())


# Shards


def test_a_shard_computes_its_own_items_first(tmp_path):
    from test_engine import DOCS, add_input, definitions

    from txtures.engine import Executor, plan_request, run_request
    from txtures.modules import RunContext
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    locations = {key: str(tmp_path / key) for key in ("inputs", "outputs", "cache")}
    locations["resources"] = str(tmp_path / "resources")
    locations["definitions"] = str(tmp_path / "definitions")
    (tmp_path / "definitions").mkdir()
    settings = resolve_settings(locations, env={}, user_home=tmp_path)
    add_input(settings, DOCS)
    fake_modules.reset()
    plan = plan_request(settings, definitions(settings), "doc")
    executor = Executor(
        plan,
        EntryStore(settings.cache),
        EntryStore(settings.resource_store),
        RunContext(),
        shard=(1, 2),
    )
    run_request(plan, executor, None)
    # Documents 0, 1, 2; shard 1 of 2 owns document 1 and then takes the rest.
    assert ORDER == ["It fell.", "The horse raced.", "The end."]


# Through the command line, with a fake Slurm


class FakeSlurm(Slurm):
    def __init__(self):
        super().__init__(run=None)
        self.scripts: list[str] = []
        self.task_states: dict[str, str] = {}

    def submit(self, script: Path) -> str:
        self.scripts.append(script.read_text())
        return str(41 + len(self.scripts))

    def states(self, job_id: str) -> dict[str, str]:
        return dict(self.task_states)


@pytest.fixture
def fake_slurm(monkeypatch, fake_engine):  # noqa: F811
    from txtures import cli

    slurm = FakeSlurm()
    monkeypatch.setattr(cli, "_slurm", lambda: slurm)
    write(
        Path(os.environ["TXTURES_HOME"]) / "site.yml",
        "execution:\n  executor: slurm\nslurm:\n  partition: gpu\n  workers: 2\n",
    )
    return slurm


def test_run_submits_and_tasks_write_the_table(capsys, fake_slurm, fake_engine):  # noqa: F811
    assert main(["run", "doc"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 42 (2 tasks)" in out
    assert "#SBATCH --array=0-1" in fake_slurm.scripts[0]
    (job,) = slurm_jobs.jobs(fake_engine / "outputs")
    assert job.job_id == "42" and job.workers == 2
    assert not list((fake_engine / "outputs" / "doc").glob("main-*"))

    fake_slurm.task_states = {"42_0": "PENDING", "42_1": "PENDING"}
    assert main(["run", "doc"]) == 0
    assert "job 42 is already computing this table (2 PENDING)" in capsys.readouterr().out
    assert len(fake_slurm.scripts) == 1

    fake_slurm.task_states = {"42_0": "RUNNING", "42_1": "RUNNING"}
    assert main(["worker", str(job.directory), "1"]) == 0
    assert main(["worker", str(job.directory), "0"]) == 0
    out = capsys.readouterr().out
    assert out.count("wrote ") == 1 and "is already current" in out
    assert fake_modules.CALLS["fake_scores"] == 2

    fake_slurm.task_states = {"42_0": "COMPLETED", "42_1": "COMPLETED"}
    assert main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [table["state"] for table in payload["tables"]] == ["current"]
    assert payload["jobs"][0]["tasks"] == {"42_0": "COMPLETED", "42_1": "COMPLETED"}
    assert main(["run", "doc"]) == 0
    assert "is already current" in capsys.readouterr().out
    assert main(["gc"]) == 0
    assert "removed the record and logs of finished job 42" in capsys.readouterr().out
    assert slurm_jobs.jobs(fake_engine / "outputs") == []


def test_task_refuses_a_changed_request(capsys, fake_slurm, fake_engine):  # noqa: F811
    assert main(["run", "doc"]) == 0
    (job,) = slurm_jobs.jobs(fake_engine / "outputs")
    write(fake_engine / "inputs" / "doc.txt", "Changed.\n")
    capsys.readouterr()
    assert main(["worker", str(job.directory), "0"]) == 1
    assert "changed after this job was submitted" in capsys.readouterr().err


def test_run_inside_a_task_computes_here(capsys, monkeypatch, fake_slurm):
    monkeypatch.setenv(slurm_jobs.TASK_VARIABLE, "/jobs/x")
    assert main(["run", "doc"]) == 0
    out = capsys.readouterr()
    assert "inside a txtures Slurm task" in out.err and "wrote " in out.out
    assert fake_slurm.scripts == []


def test_worker_marks_its_process_as_a_task(capsys, fake_slurm, fake_engine):  # noqa: F811
    assert main(["run", "doc"]) == 0
    (job,) = slurm_jobs.jobs(fake_engine / "outputs")
    assert not slurm_jobs.in_task()
    assert main(["worker", str(job.directory), "0"]) == 0
    assert os.environ[slurm_jobs.TASK_VARIABLE] == str(job.directory)


def test_local_flag_and_workers_option(capsys, fake_slurm):
    assert main(["run", "doc", "--local"]) == 0
    assert "wrote " in capsys.readouterr().out
    assert fake_slurm.scripts == []
    assert main(["run", "doc", "--local", "--workers", "2"]) == 1
    assert "--nworkers applies only to Slurm" in capsys.readouterr().err


# Through the Python API, with a fake Slurm that runs the tasks


class TaskRunningSlurm(FakeSlurm):
    """Runs every task of a submitted job, in this process, when its states are first read."""

    def __init__(self, active_checks=0):
        super().__init__()
        self.directories: list[Path] = []
        self.pending: dict[str, Path] = {}
        self.cancelled: list[str] = []
        self.active_checks = active_checks

    def submit(self, script: Path) -> str:
        job_id = super().submit(script)
        self.directories.append(script.parent)
        self.pending[job_id] = script.parent
        return job_id

    def states(self, job_id: str) -> dict[str, str]:
        if self.active_checks:
            self.active_checks -= 1
            return {f"{job_id}_0": "RUNNING"}
        directory = self.pending.pop(job_id, None)
        if directory is not None:
            job = slurm_jobs.read_job(directory)
            for task in range(job.workers):
                saved = os.environ.get(slurm_jobs.TASK_VARIABLE)
                try:
                    assert main(["worker", str(directory), str(task)]) == 0
                finally:
                    os.environ.pop(slurm_jobs.TASK_VARIABLE, None)
                    if saved is not None:
                        os.environ[slurm_jobs.TASK_VARIABLE] = saved
            return {f"{job_id}_{task}": "COMPLETED" for task in range(job.workers)}
        return {f"{job_id}_0": "COMPLETED"}

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


@pytest.fixture
def api_slurm(monkeypatch, fake_slurm):
    from txtures import api

    slurm = TaskRunningSlurm()
    monkeypatch.setattr(api, "REGISTRY", fake_modules.registry())
    monkeypatch.setattr(api, "_slurm", lambda: slurm)
    monkeypatch.setattr(slurm_jobs, "POLL_SECONDS", 0)
    monkeypatch.setattr(slurm_jobs, "GRACE_SECONDS", 0)
    return slurm


TEXTS = {"b": "The horse raced.", "a": ["It", "fell."]}
CONFIG = {"fake_scores": {"scale": [3, 1]}}


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_compute_on_slurm_returns_the_local_table(api_slurm, fake_engine, tmp_path):  # noqa: F811
    import pandas as pd

    import txtures as tx

    messages = []
    table = tx.compute("fake_feature", TEXTS, config=CONFIG, progress=messages.append)
    (directory,) = api_slurm.directories
    assert directory.parent.name == "tx.compute"
    assert "#SBATCH --array=0-1" in api_slurm.scripts[0]
    assert messages[0].startswith("submitted job 42 (2 tasks)")
    assert messages[1] == "job 42: 2 COMPLETED"
    assert fake_modules.CALLS["fake_scores"] == 4
    assert not (fake_engine / "outputs" / "texts").exists()

    local = tx.compute("fake_feature", TEXTS, config=CONFIG, cache=tmp_path / "other", local=True)
    pd.testing.assert_frame_equal(table, local)
    assert list(table["document_id"]) == ["b"] * 3 + ["a"] * 2
    assert list(table.columns)[-4:] == [
        "length_scale3",
        "double_scale3",
        "length_scale1",
        "double_scale1",
    ]

    again = tx.compute("fake_feature", TEXTS, config=CONFIG)
    pd.testing.assert_frame_equal(again, table)
    assert len(api_slurm.scripts) == 1


def test_run_on_slurm_waits_for_the_table(api_slurm, fake_engine, capsys):  # noqa: F811
    import txtures as tx

    with pytest.warns(UserWarning, match="revision is not pinned"):
        table = tx.run("doc", modules=["fake_feature"], workers=1)
    assert "submitted job 42 (1 task)" in capsys.readouterr().err
    (written,) = (fake_engine / "outputs" / "doc").glob("main-*/doc.tsv")
    assert list(table["text"]) == ["The", "horse", "raced.", "It", "fell."]
    assert len(written.read_text().splitlines()) == 6
    with pytest.warns(UserWarning, match="revision is not pinned"):
        tx.run("doc", modules=["fake_feature"])
    assert len(api_slurm.scripts) == 1


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_interrupting_cancels_a_submitted_job(api_slurm, monkeypatch):
    import txtures as tx

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    messages = []
    monkeypatch.setattr(slurm_jobs, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tx.compute("fake_feature", "a b", progress=messages.append)
    assert api_slurm.cancelled == ["42"]
    assert messages[-1] == "cancelled job 42"


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_a_call_waits_for_an_active_job_and_leaves_it_running(api_slurm, monkeypatch):
    import txtures as tx

    monkeypatch.setattr(
        slurm_jobs, "wait", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        tx.compute("fake_feature", "a b")
    api_slurm.active_checks = 1
    messages = []
    with pytest.raises(KeyboardInterrupt):
        tx.compute("fake_feature", "a b", progress=messages.append)
    assert messages[0].startswith("waiting for job 42, which is already computing")
    assert messages[-1] == "stopped waiting; job 42 continues"
    assert api_slurm.cancelled == ["42"] and len(api_slurm.scripts) == 1


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_a_failed_job_raises(api_slurm, monkeypatch):
    import txtures as tx

    monkeypatch.setattr(api_slurm, "states", lambda job_id: {f"{job_id}_0": "FAILED"})
    with pytest.raises(TxturesError, match="ended without completing the request"):
        tx.compute("fake_feature", "a b")
    assert fake_modules.CALLS["fake_scores"] == 0


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_the_api_inside_a_task_computes_here(api_slurm, monkeypatch):
    import txtures as tx

    monkeypatch.setenv(slurm_jobs.TASK_VARIABLE, "/jobs/x")
    messages = []
    table = tx.compute("fake_feature", "a bb", workers=3, progress=messages.append)
    assert list(table["length"]) == [1.0, 2.0]
    assert "inside a txtures Slurm task" in messages[0]
    assert api_slurm.scripts == []


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_api_workers_need_slurm(api_slurm):
    import txtures as tx

    with pytest.raises(TxturesError, match="workers applies only to Slurm"):
        tx.compute("fake_feature", "a b", local=True, workers=2)
    with pytest.raises(TxturesError, match="at least 1"):
        tx.compute("fake_feature", "a b", workers=0)


# The auto executor


@pytest.fixture
def auto_site(api_slurm, monkeypatch):
    """A site whose executor is auto, on a host with sbatch, where fake_scores runs on Slurm."""
    write(
        Path(os.environ["TXTURES_HOME"]) / "site.yml",
        "execution:\n  executor: auto\nslurm:\n  partition: gpu\n",
    )
    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(fake_modules.FakeScores, "workload", on_slurm)
    return api_slurm


def test_lm_scores_and_parse_run_on_slurm_and_surprisal_does_not():
    from txtures.modules import REGISTRY

    for name, expected in (("lm_scores", True), ("parse", True), ("surprisal", False)):
        module = REGISTRY[name]()
        assert module.workload(module.Config()).slurm is expected


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_auto_submits_only_while_slurm_work_is_missing(auto_site):
    import txtures as tx

    messages = []
    tx.compute("fake_feature", "a bb", progress=messages.append)
    assert messages[0] == "submitting to Slurm: fake_scores results are missing"
    # A CPU job builds the resource the GPU work needs; text needs no CPU job.
    build, gpu = auto_site.scripts
    assert "--gres" not in build and "#SBATCH --gres=gpu:1" in gpu

    messages = []
    table = tx.compute("fake_feature", "a bb", progress=messages.append)
    assert list(table["length"]) == [1.0, 2.0]
    assert messages == [] and len(auto_site.scripts) == 2


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_auto_computes_light_work_here(auto_site, monkeypatch):
    import txtures as tx

    monkeypatch.setattr(fake_modules.FakeScores, "workload", here)
    messages = []
    tx.compute("fake_feature", "a bb", workers=2, progress=messages.append)
    assert not any("Slurm" in message for message in messages)
    assert auto_site.scripts == []
    assert fake_modules.CALLS["fake_scores"] == 1


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_auto_computes_here_without_slurm(auto_site, monkeypatch):
    import txtures as tx

    def notes(texts):
        messages = []
        tx.compute("fake_feature", texts, progress=messages.append)
        return [message for message in messages if "Slurm" in message]

    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: None)
    assert notes("a bb") == ["computing fake_scores here: Slurm is not set up (no sbatch)"]
    # The default site: auto, without a partition. Where there is no Slurm, no note.
    (Path(os.environ["TXTURES_HOME"]) / "site.yml").unlink()
    assert notes("a b c") == []
    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert notes("a b c d") == [
        "computing fake_scores here: Slurm is not set up (slurm.partition is unset)"
    ]
    assert auto_site.scripts == []


def test_auto_through_the_command_line(capsys, auto_site, fake_engine):  # noqa: F811
    assert main(["run", "doc", "--workers", "2"]) == 0
    captured = capsys.readouterr()
    assert "note: submitting to Slurm: fake_scores results are missing" in captured.err
    assert "submitted job 43 (2 GPU tasks) for the GPU work" in captured.out
    assert "submitted job 44 (1 task) to write" in captured.out
    assert main(["run", "doc", "--local", "--workers", "2"]) == 1
    assert "--nworkers applies only to Slurm" in capsys.readouterr().err


# On a real cluster


@pytest.mark.integration
def test_real_slurm_job(tmp_path):
    """Submit a two-task job that scores a toy model on CPU.

    Needs `TXTURES_TEST_SLURM_PARTITION` and `TXTURES_TEST_SLURM_DIR`, a
    directory that compute nodes can read and write, and
    `TXTURES_TEST_SLURM_ACCOUNT` where the cluster requires an account.
    """
    import toy_lm

    partition = os.environ.get("TXTURES_TEST_SLURM_PARTITION")
    shared = os.environ.get("TXTURES_TEST_SLURM_DIR")
    account = os.environ.get("TXTURES_TEST_SLURM_ACCOUNT")
    if not partition or not shared or shutil.which("sbatch") is None:
        pytest.skip("set TXTURES_TEST_SLURM_PARTITION and TXTURES_TEST_SLURM_DIR")
    root = Path(shared) / f"test-{os.getpid()}-{time.time_ns()}"
    try:
        model = toy_lm.build(root / "model")
        write(root / "definitions" / "configs" / "lm_scores" / "toy.yml", f"model: {model}\n")
        write(root / "definitions" / "workflows" / "toy.yml", "lm_scores: toy\nsurprisal: main\n")
        rows = "".join(f"{n}\tab ba b a\n" if n % 2 else f"{n}\tb ab\n" for n in range(6))
        write(root / "inputs" / "doc.tsv", "document_id\ttext\n" + rows)
        write(
            root / "site.yml",
            f"execution:\n  executor: slurm\n  device: cpu\nslurm:\n  partition: {partition}\n"
            "  gpus: 0\n  cpus: 1\n  memory_gb: 4\n  time_hours: 1\n"
            + (f"  account: {account}\n" if account else ""),
        )
        env = {
            **os.environ,
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "home" / ".config"),
            "TXTURES_HOME": str(root),
        }
        txtures = [sys.executable, "-I", "-m", "txtures"]
        submitted = subprocess.run(
            [*txtures, "run", "doc", "-w", "toy", "-m", "surprisal", "--workers", "2"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert submitted.returncode == 0, submitted.stderr
        assert "submitted job" in submitted.stdout
        deadline = time.monotonic() + 20 * 60
        while time.monotonic() < deadline:
            status = json.loads(
                subprocess.run(
                    [*txtures, "status", "--json"], env=env, capture_output=True, text=True
                ).stdout
            )
            (job,) = status["jobs"]
            if job["tasks"] and not slurm_jobs.is_active(job["tasks"]):
                break
            time.sleep(10)
        logs = "\n".join(p.read_text() for p in Path(job["logs"]).glob("*.out"))
        assert set(job["tasks"].values()) == {"COMPLETED"}, logs
        assert [table["state"] for table in status["tables"]] == ["current"], logs
        assert logs.count("wrote ") == 1
        (table,) = (root / "outputs" / "doc").glob("toy-*/doc.tsv")
        assert len(table.read_text().splitlines()) == 1 + 3 * 4 + 3 * 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


COMPUTE_SCRIPT = """
import json, sys
import txtures as tx

model, local_cache = sys.argv[1], sys.argv[2]
texts = {"b": "ab ba b a", "a": ["b", "ab"]}
config = {"lm_scores": {"model": model}}
remote = tx.compute("surprisal", texts, config=config, workers=2)
local = tx.compute("surprisal", texts, config=config, cache=local_cache, local=True)
print(json.dumps({"remote": remote.to_dict("list"), "local": local.to_dict("list")}))
"""


@pytest.mark.integration
def test_real_slurm_compute(tmp_path):
    """Compute in-memory text with a two-task job, blocking, and compare with local results.

    Needs the same environment variables as `test_real_slurm_job`.
    """
    import toy_lm

    partition = os.environ.get("TXTURES_TEST_SLURM_PARTITION")
    shared = os.environ.get("TXTURES_TEST_SLURM_DIR")
    account = os.environ.get("TXTURES_TEST_SLURM_ACCOUNT")
    if not partition or not shared or shutil.which("sbatch") is None:
        pytest.skip("set TXTURES_TEST_SLURM_PARTITION and TXTURES_TEST_SLURM_DIR")
    root = Path(shared) / f"test-{os.getpid()}-{time.time_ns()}"
    try:
        model = toy_lm.build(root / "model")
        write(
            root / "site.yml",
            f"execution:\n  executor: slurm\n  device: cpu\nslurm:\n  partition: {partition}\n"
            "  gpus: 0\n  cpus: 1\n  memory_gb: 4\n  time_hours: 1\n"
            + (f"  account: {account}\n" if account else ""),
        )
        env = {
            **os.environ,
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "home" / ".config"),
            "TXTURES_HOME": str(root),
        }
        result = subprocess.run(
            [sys.executable, "-I", "-c", COMPUTE_SCRIPT, str(model), str(root / "local-cache")],
            env=env,
            capture_output=True,
            text=True,
            timeout=20 * 60,
        )
        logs = "\n".join(p.read_text() for p in root.glob("outputs/.txtures-jobs/*/*/logs/*.out"))
        assert result.returncode == 0, result.stderr + logs
        assert "submitted job" in result.stderr
        tables = json.loads(result.stdout.splitlines()[-1])
        remote, local = tables["remote"], tables["local"]
        # The task may run on another CPU type, whose float32 kernels can
        # differ from the login node's in the last bits; 1e-5 relative allows
        # that and nothing larger.
        surprisal = [name for name in remote if name.startswith("surprisal")]
        for name in surprisal:
            assert remote.pop(name) == pytest.approx(local.pop(name), rel=1e-5)
        assert surprisal and remote == local
        assert tables["remote"]["document_id"] == ["b"] * 4 + ["a"] * 2
        assert logs.count("computed the items of") == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


# Resources built on Slurm


@pytest.fixture
def slurm_weights(monkeypatch, fake_slurm):
    """Make the fake engine's resource build on Slurm, with needs of its own."""
    monkeypatch.setattr(
        fake_modules.Weights,
        "workload",
        lambda self, resolved: Workload(
            slurm=True, ncores=16, memory_gb=200, time_hours=48, ngpus=0
        ),
    )
    write(
        Path(os.environ["TXTURES_HOME"]) / "site.yml",
        "execution:\n  executor: slurm\n"
        "slurm:\n  partition: gpu\n  cpu_partition: cpu\n  workers: 2\n",
    )
    return fake_slurm


def test_slurm_resource_is_built_by_its_own_job_first(
    capsys,
    slurm_weights,
    fake_engine,  # noqa: F811
):
    assert main(["run", "doc"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 42 to build resource fake_weights" in out
    assert "submitted job 43 (2 tasks)" in out and "after the builds" in out
    build, request = slurm_weights.scripts
    assert "#SBATCH --partition=cpu" in build
    assert "#SBATCH --mem=200G" in build and "#SBATCH --cpus-per-task=16" in build
    assert "#SBATCH --time=2880" in build and "--gres" not in build
    # No module here needs a GPU, so the request's job runs on CPUs.
    assert "#SBATCH --partition=cpu" in request and "--gres" not in request
    assert "#SBATCH --dependency=afterok:42" in request
    assert "#SBATCH --kill-on-invalid-dep=yes" in request

    jobs = {job.job_id: job for job in slurm_jobs.jobs(fake_engine / "outputs")}
    assert jobs["42"].source == "resource"
    assert jobs["43"].request["builds"][0]["job_id"] == "42"
    assert main(["worker", str(jobs["42"].directory), "0"]) == 0
    assert "built resource fake_weights" in capsys.readouterr().out
    assert fake_modules.CALLS["build"] == 1
    assert main(["worker", str(jobs["43"].directory), "0"]) == 0
    assert main(["worker", str(jobs["43"].directory), "1"]) == 0
    assert fake_modules.CALLS["build"] == 1
    assert main(["status"]) == 0
    assert "job 42  resources/fake_weights-" in capsys.readouterr().out


def test_build_task_refuses_a_resource_that_builds_differently(
    capsys,
    slurm_weights,
    fake_engine,  # noqa: F811
):
    assert main(["run", "doc"]) == 0
    build = next(j for j in slurm_jobs.jobs(fake_engine / "outputs") if j.source == "resource")
    request = json.loads((build.directory / "request.json").read_text())
    request["resolved"]["scale"] = 99
    (build.directory / "request.json").write_text(json.dumps(request))
    capsys.readouterr()
    assert main(["worker", str(build.directory), "0"]) == 1
    assert "builds differently" in capsys.readouterr().err


def test_an_active_build_job_is_reused(slurm_weights, fake_engine):  # noqa: F811
    from txtures.settings import resolve_settings

    assert main(["run", "doc"]) == 0
    slurm_weights.task_states = {"42": "RUNNING"}
    settings = resolve_settings({})
    resource = fake_resource(settings)
    assert slurm_jobs.submit_build(resource, settings, slurm_weights).job_id == "42"
    assert len(slurm_weights.scripts) == 2


def fake_resource(settings):
    from txtures import api
    from txtures.engine import plan_request

    plan = plan_request(settings, api.load_definitions(settings), "doc", "main", None)
    (resource,) = plan.resources()
    return resource


def test_job_options_follow_workloads(slurm_weights, fake_engine):  # noqa: F811
    from txtures.settings import resolve_settings

    settings = resolve_settings({})
    options = slurm_jobs.job_options(settings, [Workload(slurm=True, ncores=16, ngpus=0)])
    assert (options["partition"], options["gpus"], options["cpus"]) == ("cpu", 0, 16)
    options = slurm_jobs.job_options(settings, [Workload(slurm=True, ngpus=2)])
    assert (options["partition"], options["gpus"]) == ("gpu", 2)
    assert options["memory_gb"] == settings["slurm.memory_gb"]
    # The largest need wins; fields of workloads not on Slurm are ignored.
    options = slurm_jobs.job_options(
        settings,
        [
            Workload(slurm=True, memory_gb=64),
            Workload(slurm=True, memory_gb=128),
            Workload(memory_gb=999),
        ],
    )
    assert options["memory_gb"] == 128
    assert slurm_jobs.job_options(settings, [])["cpus"] == settings["slurm.cpus"]


def test_request_array_follows_module_workloads(
    capsys,
    monkeypatch,
    fake_slurm,
    fake_engine,  # noqa: F811
):
    monkeypatch.setattr(
        fake_modules.FakeScores,
        "workload",
        lambda self, config: Workload(slurm=True, memory_gb=96, time_hours=3),
    )
    assert main(["run", "doc"]) == 0
    _, gpu, tables = fake_slurm.scripts
    assert "#SBATCH --mem=96G" in gpu and "#SBATCH --time=180" in gpu
    assert "#SBATCH --gres=gpu:1" in gpu and "--gres" not in tables


def test_resources_build_submits_slurm_resources(
    capsys,
    monkeypatch,
    slurm_weights,
    fake_engine,  # noqa: F811
):
    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert main(["resources", "build", "doc"]) == 0
    assert "job 42 builds resource fake_weights" in capsys.readouterr().out
    assert len(slurm_weights.scripts) == 1 and fake_modules.CALLS["build"] == 0
    assert main(["resources", "build", "doc", "--local"]) == 0
    assert "built resource fake_weights" in capsys.readouterr().out
    assert fake_modules.CALLS["build"] == 1
    assert main(["resources", "build", "doc"]) == 0
    assert "every resource is already built" in capsys.readouterr().out


def test_resources_build_builds_light_resources_here(
    capsys,
    monkeypatch,
    fake_slurm,
    fake_engine,  # noqa: F811
):
    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert main(["resources", "build", "doc"]) == 0
    assert "built resource fake_weights" in capsys.readouterr().out
    assert fake_slurm.scripts == []


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_auto_submits_while_a_slurm_resource_is_missing(auto_site, monkeypatch):
    import txtures as tx

    monkeypatch.setattr(fake_modules.FakeScores, "workload", here)
    monkeypatch.setattr(fake_modules.Weights, "workload", on_slurm)
    messages = []
    tx.compute("fake_feature", "a bb", progress=messages.append)
    assert messages[0] == "submitting to Slurm: resource fake_weights is missing"
    assert len(auto_site.scripts) == 2


def test_a_build_without_slurm_is_announced(fake_engine, monkeypatch):  # noqa: F811
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    monkeypatch.setattr(slurm_jobs.shutil, "which", lambda name: None)
    monkeypatch.setattr(fake_modules.Weights, "workload", on_slurm)
    from txtures import api
    from txtures.engine import plan_request

    settings = resolve_settings({})
    planned = plan_request(settings, api.load_definitions(settings), "doc", "main", None)
    decision = slurm_jobs.decide(settings, False, planned, EntryStore(settings.cache))
    assert decision == slurm_jobs.Decision(
        False, "building resource fake_weights here: Slurm is not set up"
    )


def test_dependency_with_a_slurm_workload_gets_its_own_build_job(
    capsys,
    monkeypatch,
    fake_slurm,
    fake_engine,  # noqa: F811
):
    from test_tools import BUILT, Base, Derived

    from txtures.engine.planning import plan_resource
    from txtures.modules import RunContext
    from txtures.settings import resolve_settings

    monkeypatch.setattr(Base, "workload", on_slurm)
    settings = resolve_settings({})
    planned = plan_resource("derived", Derived({"n": 1}), RunContext())
    job = slurm_jobs.submit_build(planned, settings, fake_slurm)
    base_script, derived_script = fake_slurm.scripts
    assert "--dependency" not in base_script
    assert "#SBATCH --dependency=afterok:42" in derived_script
    assert job.job_id == "43" and job.request["after"] == ["42"]
    BUILT.clear()
    base_job = slurm_jobs.read_job(
        slurm_jobs.build_directory(settings.outputs, planned.dependencies["base"])
    )
    assert main(["worker", str(base_job.directory), "0"]) == 0
    assert main(["worker", str(job.directory), "0"]) == 0
    assert BUILT == ["base", "derived"]
    assert "built resource test_derived" in capsys.readouterr().out


# Per-call job sizes


def test_a_per_call_size_overrides_workloads_and_settings(slurm_weights, fake_engine):  # noqa: F811
    from txtures.settings import resolve_settings

    settings = resolve_settings({})
    declared = [Workload(slurm=True, ngpus=2, memory_gb=200)]
    options = slurm_jobs.job_options(settings, declared, slurm_jobs.size({"ngpus": 0}))
    assert (options["gpus"], options["partition"], options["memory_gb"]) == (0, "cpu", 200)
    assert slurm_jobs.size(None) is None and slurm_jobs.size({}) is None
    with pytest.raises(TxturesError, match="unknown job size gpus"):
        slurm_jobs.size({"gpus": 0})
    with pytest.raises(TxturesError, match="ncores must be a whole number of at least 1"):
        slurm_jobs.size({"ncores": 0})
    with pytest.raises(TxturesError, match="ngpus must be"):
        slurm_jobs.size({"ngpus": True})


def test_size_flags_shape_every_job_a_run_submits(capsys, slurm_weights, fake_engine):  # noqa: F811
    assert main(["run", "doc", "--ngpus", "0", "--memory-gb", "8", "--time-hours", "1"]) == 0
    build, request = slurm_weights.scripts
    for script in (build, request):
        assert "#SBATCH --mem=8G" in script and "#SBATCH --time=60" in script
        assert "--gres" not in script and "#SBATCH --partition=cpu" in script


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_api_size_applies_to_submitted_jobs(auto_site):
    import txtures as tx

    tx.compute("fake_feature", "a bb", slurm={"ncores": 12, "ngpus": 0})
    (script,) = auto_site.scripts
    assert "#SBATCH --cpus-per-task=12" in script and "--gres" not in script
    with pytest.raises(TxturesError, match="unknown job size"):
        tx.compute("fake_feature", "a", slurm={"cpus": 2})


# GPU work in a job of its own


@pytest.fixture
def gpu_site(monkeypatch, fake_slurm):
    """fake_scores needs a GPU; the site has a CPU partition."""
    monkeypatch.setattr(fake_modules.FakeScores, "workload", on_slurm)
    write(
        Path(os.environ["TXTURES_HOME"]) / "site.yml",
        "execution:\n  executor: slurm\n"
        "slurm:\n  partition: gpu\n  cpu_partition: cpu\n  workers: 2\n",
    )
    return fake_slurm


def test_gpu_work_runs_in_its_own_job_and_tables_on_cpus(
    capsys,
    gpu_site,
    fake_engine,  # noqa: F811
):
    assert main(["run", "doc"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 42 to build resource fake_weights" in out
    assert "submitted job 43 (2 GPU tasks) for the GPU work" in out
    assert "submitted job 44 (1 task) to write" in out and "after it" in out
    build, gpu, tables = gpu_site.scripts
    assert "#SBATCH --partition=cpu" in build and "--gres" not in build
    assert "#SBATCH --partition=gpu" in gpu and "#SBATCH --gres=gpu:1" in gpu
    assert "#SBATCH --dependency=afterok:42" in gpu
    assert "#SBATCH --partition=cpu" in tables and "--gres" not in tables
    assert "#SBATCH --dependency=afterok:42,afterany:43" in tables

    jobs = {job.job_id: job for job in slurm_jobs.jobs(fake_engine / "outputs")}
    assert main(["worker", str(jobs["42"].directory), "0"]) == 0
    # Task 0 takes its share, then the item nobody holds; task 1 finds nothing left.
    assert main(["worker", str(jobs["43"].directory), "0"]) == 0
    assert fake_modules.CALLS["fake_scores"] == 2
    assert main(["worker", str(jobs["43"].directory), "1"]) == 0
    assert fake_modules.CALLS["fake_scores"] == 2
    capsys.readouterr()
    assert main(["worker", str(jobs["44"].directory), "0"]) == 0
    assert "wrote " in capsys.readouterr().out
    assert fake_modules.CALLS["fake_scores"] == 2


def test_a_gpu_task_leaves_held_items_and_the_table_job_reports_them(
    capsys,
    gpu_site,
    fake_engine,  # noqa: F811
):
    from txtures.store import EntryStore

    assert main(["run", "doc"]) == 0
    jobs = {job.job_id: job for job in slurm_jobs.jobs(fake_engine / "outputs")}
    assert main(["worker", str(jobs["42"].directory), "0"]) == 0
    items = EntryStore(fake_engine / "cache")
    from txtures import api
    from txtures.engine import plan_request
    from txtures.settings import resolve_settings

    settings = resolve_settings({})
    plan = plan_request(settings, api.load_definitions(settings), "doc", "main", None)
    scores = next(p for p in plan.variants if p.name == "fake_scores")
    held = scores.item_identity(plan.corpus.documents[1])
    assert items.try_lock("fake_scores", held)
    try:
        started = time.monotonic()
        assert main(["worker", str(jobs["43"].directory), "0"]) == 0
        assert time.monotonic() - started < 5
        assert fake_modules.CALLS["fake_scores"] == 1
        capsys.readouterr()
        assert main(["worker", str(jobs["44"].directory), "0"]) == 1
        err = capsys.readouterr().err
        assert "GPU results of fake_scores are missing" in err
        assert fake_modules.CALLS["fake_scores"] == 1
        assert not list((fake_engine / "outputs" / "doc").glob("main-*/doc.tsv"))
    finally:
        items.unlock("fake_scores", held)


def test_several_inputs_share_one_gpu_job_and_one_table_job(
    capsys,
    gpu_site,
    fake_engine,  # noqa: F811
):
    write(fake_engine / "inputs" / "other.txt", "Another one.\n")
    write(fake_engine / "inputs" / "third.txt", "A third.\n")
    assert main(["run", "doc", "other", "third", "--nworkers", "3"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 43 (3 GPU tasks) for the GPU work" in out
    assert "to write 3 tables (doc, other, third)" in out
    jobs = {job.job_id: job for job in slurm_jobs.jobs(fake_engine / "outputs")}
    assert jobs["43"].source == "gpu" and len(jobs["43"].request["requests"]) == 3
    assert jobs["44"].source == "batch"
    for job_id, tasks in (("42", 1), ("43", 3), ("44", 1)):
        for task in range(tasks):
            assert main(["worker", str(jobs[job_id].directory), str(task)]) == 0
    assert fake_modules.CALLS["fake_scores"] == 4
    # One GPU task did all three inputs with one load of the model.
    assert fake_modules.CALLS["load"] == 1
    for name in ("doc", "other", "third"):
        assert list((fake_engine / "outputs" / name).glob("main-*/" + name + ".tsv"))
    assert main(["run", "doc", "other", "third"]) == 0
    assert capsys.readouterr().out.count("is already current") == 3
