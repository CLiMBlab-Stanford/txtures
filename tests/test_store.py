import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import write

from txtures.definitions import Definitions, init_store
from txtures.definitions.store import init_home
from txtures.errors import TxturesError


def messages(store) -> list[str]:
    return [str(problem) for problem in Definitions.load(store).validate()]


def test_init_creates_a_valid_store(tmp_path):
    root = tmp_path / "defs"
    init_store(root)
    assert sorted(p.name for p in root.iterdir()) == [".git", "README.md", "configs", "workflows"]
    assert (root / "configs" / "lm_scores" / "pythia410m.yml").is_file()
    assert (root / "workflows" / "pythia.yml").is_file()
    assert messages(root) == []
    assert [p.name for p in tmp_path.iterdir()] == ["defs"]
    assert os.stat(root).st_mode & 0o777 != 0o700


def git_config(root, key: str) -> str:
    result = subprocess.run(["git", "-C", str(root), "config", key], capture_output=True, text=True)
    return result.stdout.strip()


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@needs_git
def test_init_makes_the_store_a_git_repository_shared_like_its_parent(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o755)
    os.chmod(private, 0o755)
    assert init_store(private / "defs") is None
    assert (private / "defs" / ".git").is_dir()
    assert git_config(private / "defs", "core.sharedRepository") == ""
    assert messages(private / "defs") == []

    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o775)
    assert init_store(shared / "defs") is None
    assert git_config(shared / "defs", "core.sharedRepository") in {"1", "group"}


@needs_git
def test_init_leaves_git_alone_inside_another_repository_or_when_asked(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path / "project")], check=True)
    note = init_store(tmp_path / "project" / "defs")
    assert note is not None and "lies inside the Git repository" in note
    assert not (tmp_path / "project" / "defs" / ".git").exists()
    assert init_store(tmp_path / "plain", git=False) is None
    assert not (tmp_path / "plain" / ".git").exists()


def test_init_without_git_installed_still_creates_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert (
        init_store(tmp_path / "defs")
        == "git is not installed, so the definitions store is not a Git repository"
    )
    assert (tmp_path / "defs" / "workflows" / "pythia.yml").is_file()


def tracked(root) -> list[str]:
    """The files Git would add in `root`."""
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line[3:] for line in status.splitlines())


HOME_FILES = [
    ".gitignore",
    "README.md",
    "definitions/README.md",
    "definitions/configs/lm_scores/pythia410m.yml",
    "definitions/workflows/pythia.yml",
    "site.yml",
]


@needs_git
def test_init_home_tracks_only_settings_and_definitions(tmp_path):
    txh = tmp_path / "txh"
    assert init_home(txh) is None
    assert messages(txh / "definitions") == []
    for name in ("inputs", "outputs", "cache", "resources"):
        assert (txh / name).is_dir()
    write(txh / "cache" / "x" / "entry", "data")
    write(txh / "outputs" / "doc" / "t.tsv", "data")
    write(txh / "notes.txt", "mine")
    assert tracked(txh) == HOME_FILES


@needs_git
def test_init_home_fills_a_directory_that_holds_a_checkout(tmp_path):
    txh = tmp_path / "txh"
    subprocess.run(["git", "init", "--quiet", str(txh / "code")], check=True)
    write(txh / "code" / "README.md", "the code\n")
    write(txh / "notes.txt", "mine\n")
    assert init_home(txh) is None
    assert (txh / "notes.txt").read_text() == "mine\n"
    assert (txh / "code" / "README.md").read_text() == "the code\n"
    assert not [p for p in txh.iterdir() if p.name.startswith(".txtures-init")]
    assert (txh / ".git").is_dir()
    assert tracked(txh) == HOME_FILES
    assert messages(txh / "definitions") == []


def test_init_home_leaves_an_existing_repository_alone(tmp_path):
    txh = tmp_path / "txh"
    (txh / ".git").mkdir(parents=True)
    note = init_home(txh)
    assert note == f"{txh} is already a Git repository, so it was left as it is"
    assert (txh / "site.yml").is_file() and (txh / ".gitignore").is_file()


@needs_git
def test_init_home_inside_a_repository_leaves_git_alone(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path / "project")], check=True)
    note = init_home(tmp_path / "project" / "txh")
    assert note is not None and "lies inside the Git repository" in note
    assert not (tmp_path / "project" / "txh" / ".git").exists()


def test_init_home_refuses_a_directory_with_home_entries(tmp_path):
    txh = tmp_path / "txh"
    write(txh / "cache" / "x", "")
    write(txh / "README.md", "")
    write(txh / ".gitignore", "")
    write(txh / "other", "")
    with pytest.raises(TxturesError, match="it already has README.md, .gitignore, cache"):
        init_home(txh, git=False)
    assert sorted(p.name for p in txh.iterdir()) == [".gitignore", "README.md", "cache", "other"]
    with pytest.raises(TxturesError, match="is not a directory"):
        init_home(txh / "other")


def test_init_home_puts_back_what_it_placed_when_it_fails(tmp_path, monkeypatch):
    txh = tmp_path / "txh"
    write(txh / "keep", "")
    real = os.rename

    def failing(source, target):
        if Path(target).name == "outputs":
            raise OSError("disk full")
        return real(source, target)

    monkeypatch.setattr(os, "rename", failing)
    with pytest.raises(OSError, match="disk full"):
        init_home(txh, git=False)
    assert sorted(p.name for p in txh.iterdir()) == ["keep"]


def test_init_example_workflow_resolves_to_pythia(tmp_path):
    root = tmp_path / "defs"
    init_store(root)
    (lm,) = Definitions.load(root).expand("pythia").variants["lm_scores"]
    assert lm.config.model == "EleutherAI/pythia-410m"


def test_init_refuses_an_existing_path(tmp_path):
    root = tmp_path / "defs"
    root.mkdir()
    with pytest.raises(TxturesError, match="already exists"):
        init_store(root)


def test_missing_store_is_an_error(tmp_path):
    with pytest.raises(TxturesError, match="txtures site init"):
        Definitions.load(tmp_path / "absent")


def test_empty_store_is_valid(store):
    assert messages(store) == []


def test_layout_problems_are_all_reported(store):
    write(store / "notes.txt", "")
    write(store / "site" / "site.yml", "")
    write(store / "configs" / "parser" / "a.yml", "")
    write(store / "configs" / "lm_scores" / "bad-name.yml", "")
    write(store / "configs" / "lm_scores" / "wrong.yaml", "")
    write(store / "workflows" / "broken.yml", "lm_scores: [\n")
    found = "\n".join(messages(store))
    assert "notes.txt: unexpected entry in definitions store" in found
    assert "site: site settings now live in the txtures home's site.yml" in found
    assert "parser: not a known module directory" in found
    assert "ID 'bad-name' must match" in found
    assert "wrong.yaml: expected a .yml file" in found
    assert "broken.yml:" in found


def test_symlinked_definitions_are_rejected(store, tmp_path):
    target = write(tmp_path / "elsewhere.yml", "model: gpt2\n")
    link = store / "configs" / "lm_scores" / "linked.yml"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert any("symbolic links are not allowed" in m for m in messages(store))


def test_validate_checks_unused_configs_and_every_workflow(store):
    write(store / "configs" / "lm_scores" / "unused.yml", "dtype: float8\n")
    write(store / "configs" / "surprisal" / "main.yml", "extends: other\n")
    write(store / "workflows" / "w.yml", "lm_scores: nothere\n")
    found = "\n".join(messages(store))
    assert "unused.yml: dtype:" in found
    assert "main cannot use extends" in found
    assert "w.yml: no lm_scores config named 'nothere'" in found


def test_heartbeat_keeps_held_locks_fresh(tmp_path):
    from txtures.store import Heartbeat

    path = tmp_path / "entry.lock"
    path.write_text("{}")
    os.utime(path, (0, 0))
    heartbeat = Heartbeat(interval=3600)
    heartbeat.add(path)
    heartbeat.beat()
    assert path.stat().st_mtime > 1e9
    heartbeat.discard(path)
    os.utime(path, (0, 0))
    heartbeat.beat()
    assert path.stat().st_mtime == 0
