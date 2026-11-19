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
    assert slurm_jobs.submits(settings)
    assert not slurm_jobs.submits(settings, local=True)
    monkeypatch.setenv(slurm_jobs.TASK_VARIABLE, "/jobs/x")
    assert not slurm_jobs.submits(settings)
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
    assert "--workers applies only to Slurm" in capsys.readouterr().err


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
    with pytest.warns(UserWarning, match="inside a txtures Slurm task"):
        table = tx.compute("fake_feature", "a bb", workers=3)
    assert list(table["length"]) == [1.0, 2.0]
    assert api_slurm.scripts == []


@pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")
def test_api_workers_need_slurm(api_slurm):
    import txtures as tx

    with pytest.raises(TxturesError, match="workers applies only to Slurm"):
        tx.compute("fake_feature", "a b", local=True, workers=2)
    with pytest.raises(TxturesError, match="at least 1"):
        tx.compute("fake_feature", "a b", workers=0)


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
            [*txtures, "run", "doc", "-w", "toy", "--workers", "2"],
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
