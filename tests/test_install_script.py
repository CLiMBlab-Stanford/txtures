"""`./install --upgrade`, run against throwaway repositories.

Each release in these repositories replaces `install` with a stub that
reports how it was run, so the tests check which release the checkout
moves to and what the updated script receives, without installing.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL = Path(__file__).resolve().parents[1] / "install"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def release(origin_work: Path, tag: str) -> None:
    stub = origin_work / "install"
    stub.write_text(f'#!/bin/sh\necho "{tag} install $*"\n')
    stub.chmod(0o755)
    git(origin_work, "add", "install")
    git(origin_work, "commit", "--quiet", "-m", tag)
    git(origin_work, "tag", "-a", tag, "-m", tag)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A clone of a repository whose first commit holds the real install script."""
    work = tmp_path / "origin"
    work.mkdir()
    git(work, "init", "--quiet", "--initial-branch=main")
    git(work, "config", "user.name", "Test")
    git(work, "config", "user.email", "test@example.invalid")
    shutil.copy(INSTALL, work / "install")
    git(work, "add", "install")
    git(work, "commit", "--quiet", "-m", "start")
    git(work, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", str(work), str(clone)], check=True)
    git(clone, "config", "user.name", "Test")
    git(clone, "config", "user.email", "test@example.invalid")
    return clone


def run(clone: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(clone / "install"), *args], capture_output=True, text=True, cwd=clone
    )


def test_upgrade_moves_main_to_the_newest_release_and_runs_its_script(checkout):
    origin = checkout.parent / "origin"
    release(origin, "v0.2.0")
    release(origin, "v0.10.0")
    release(origin, "v0.11.0-rc1")
    result = run(checkout, "--upgrade", "--prefix", "/somewhere", "--dev")
    assert result.returncode == 0, result.stderr
    assert "updated the checkout to v0.10.0" in result.stdout
    assert result.stdout.strip().endswith("v0.10.0 install --prefix /somewhere --dev")
    assert git(checkout, "describe", "--tags") == "v0.10.0"
    assert git(checkout, "symbolic-ref", "--short", "HEAD") == "main"


def test_upgrade_moves_a_detached_release_checkout(checkout):
    origin = checkout.parent / "origin"
    git(checkout, "checkout", "--quiet", "--detach", "v0.1.0")
    release(origin, "v0.2.0")
    result = run(checkout, "--upgrade")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("v0.2.0 install")
    assert git(checkout, "describe", "--tags") == "v0.2.0"


def test_upgrade_refuses_uncommitted_changes_and_other_branches(checkout):
    origin = checkout.parent / "origin"
    release(origin, "v0.2.0")
    (checkout / "install").write_text((checkout / "install").read_text() + "# edit\n")
    result = run(checkout, "--upgrade")
    assert result.returncode == 1
    assert "uncommitted changes" in result.stderr
    git(checkout, "checkout", "--quiet", "--", "install")

    git(checkout, "checkout", "--quiet", "-b", "dev")
    result = run(checkout, "--upgrade")
    assert result.returncode == 1
    assert "on branch dev" in result.stderr
    assert git(checkout, "describe", "--tags") == "v0.1.0"


def test_help_never_upgrades(checkout):
    release(checkout.parent / "origin", "v0.2.0")
    result = run(checkout, "--upgrade", "--help")
    assert result.returncode == 0
    assert "--upgrade" in result.stdout
    assert git(checkout, "describe", "--tags") == "v0.1.0"
