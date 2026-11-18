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


# Shards


def test_a_shard_computes_its_own_items_first(tmp_path):
    from test_engine import DOCS, add_input, definitions

    from txtures.engine import Executor, plan_request, run_request
    from txtures.modules import RunContext
    from txtures.settings import resolve_settings
    from txtures.store import EntryStore

    locations = {key: str(tmp_path / key) for key in ("inputs", "derivatives", "items")}
    locations["resources"] = str(tmp_path / "resources")
    locations["definitions"] = str(tmp_path / "definitions")
    (tmp_path / "definitions").mkdir()
    settings = resolve_settings(locations, env={}, home=tmp_path)
    add_input(settings, DOCS)
    fake_modules.reset()
    plan = plan_request(settings, definitions(settings), "doc")
    executor = Executor(
        plan,
        EntryStore(settings.items),
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
        fake_engine / "home" / ".config" / "txtures" / "settings.yml",
        "execution:\n  executor: slurm\nslurm:\n  partition: gpu\n  workers: 2\n",
    )
    return slurm


def test_run_submits_and_tasks_write_the_table(capsys, fake_slurm, fake_engine):  # noqa: F811
    assert main(["run", "doc"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 42 (2 tasks)" in out
    assert "#SBATCH --array=0-1" in fake_slurm.scripts[0]
    (job,) = slurm_jobs.jobs(fake_engine / "derivatives")
    assert job.job_id == "42" and job.workers == 2
    assert not list((fake_engine / "derivatives" / "doc").glob("main-*"))

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
    assert slurm_jobs.jobs(fake_engine / "derivatives") == []


def test_task_refuses_a_changed_request(capsys, fake_slurm, fake_engine):  # noqa: F811
    assert main(["run", "doc"]) == 0
    (job,) = slurm_jobs.jobs(fake_engine / "derivatives")
    write(fake_engine / "inputs" / "doc.txt", "Changed.\n")
    capsys.readouterr()
    assert main(["worker", str(job.directory), "0"]) == 1
    assert "changed after this job was submitted" in capsys.readouterr().err


def test_local_flag_and_workers_option(capsys, fake_slurm):
    assert main(["run", "doc", "--local"]) == 0
    assert "wrote " in capsys.readouterr().out
    assert fake_slurm.scripts == []
    assert main(["run", "doc", "--local", "--workers", "2"]) == 1
    assert "--workers applies only to Slurm" in capsys.readouterr().err


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
            root / "home" / ".config" / "txtures" / "settings.yml",
            f"execution:\n  executor: slurm\n  device: cpu\nslurm:\n  partition: {partition}\n"
            "  gpus: 0\n  cpus: 1\n  memory_gb: 4\n  time_hours: 1\n"
            + (f"  account: {account}\n" if account else ""),
        )
        env = {
            **os.environ,
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "home" / ".config"),
            **{
                f"TXTURES_{key.upper()}": str(root / key)
                for key in ("definitions", "inputs", "derivatives", "items", "resources")
            },
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
        (table,) = (root / "derivatives" / "doc").glob("toy-*/doc.tsv")
        assert len(table.read_text().splitlines()) == 1 + 3 * 4 + 3 * 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
