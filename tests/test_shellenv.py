from pathlib import Path

import pytest
from conftest import write

from txtures.errors import SettingsError
from txtures.settings import resolve_settings
from txtures.shellenv import BEGIN, END, read_env, set_env, shell_config_path, unset_env


@pytest.mark.parametrize(
    ("shell", "name"),
    [
        ("/bin/zsh", ".zshrc"),
        ("/usr/bin/bash", ".bashrc"),
        ("/bin/sh", ".profile"),
        ("", ".profile"),
    ],
)
def test_startup_file_follows_the_login_shell(tmp_path, shell, name):
    assert shell_config_path({"SHELL": shell}, tmp_path) == tmp_path / name


def test_set_env_keeps_the_rest_of_the_file_and_its_mode(tmp_path):
    path = write(tmp_path / ".bashrc", f"before\n\n{BEGIN}\n{END}\nafter\n")
    path.chmod(0o600)
    assert set_env(path, [("slurm.cpus", ["4"]), ("execution.device", ["cuda:1"])]) == {
        "TXTURES_SLURM_CPUS": "4",
        "TXTURES_EXECUTION_DEVICE": "cuda:1",
    }
    lines = path.read_text().splitlines()
    assert lines[0] == "before" and lines[-1] == "after"
    assert lines.index("export TXTURES_EXECUTION_DEVICE=cuda:1") < lines.index(
        "export TXTURES_SLURM_CPUS=4"
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_env(path) == {"TXTURES_SLURM_CPUS": "4", "TXTURES_EXECUTION_DEVICE": "cuda:1"}
    assert unset_env(path, ["slurm.cpus", "slurm.cpus", "slurm.gpus"]) == ["TXTURES_SLURM_CPUS"]
    assert read_env(path) == {"TXTURES_EXECUTION_DEVICE": "cuda:1"}


def test_set_env_writes_through_a_symbolic_link(tmp_path):
    target = write(tmp_path / "dotfiles" / "bashrc", "x\n")
    link = tmp_path / ".bashrc"
    link.symlink_to(target)
    set_env(link, [("slurm.account", ["lab"])])
    assert link.is_symlink()
    assert "export TXTURES_SLURM_ACCOUNT=lab" in target.read_text()


def test_list_variables_round_trip_through_shell_words(tmp_path):
    path = tmp_path / ".profile"
    options = ["--exclude=n1,n2", "--comment=a b", "it's"]
    written = set_env(path, [("slurm.options", options)])
    assert read_env(path) == written
    settings = resolve_settings(env=written, user_home=tmp_path)
    assert settings["slurm.options"] == options


@pytest.mark.parametrize(
    "text",
    [
        f"{BEGIN}\n",
        f"{END}\n{BEGIN}\n",
        f"{BEGIN}\n{END}\n{BEGIN}\n{END}\n",
        f"{BEGIN}\nexport PATH=/x\n{END}\n",
        f"{BEGIN}\necho hi\n{END}\n",
    ],
)
def test_malformed_blocks_are_left_for_the_user_to_repair(tmp_path, text):
    path = write(tmp_path / ".bashrc", text)
    with pytest.raises(SettingsError, match="repair it by hand"):
        set_env(path, [("slurm.cpus", ["4"])])
    assert path.read_text() == text


def test_invalid_assignments_change_nothing(tmp_path):
    path: Path = tmp_path / ".bashrc"
    with pytest.raises(SettingsError, match="unknown setting"):
        set_env(path, [("colour", ["blue"])])
    with pytest.raises(SettingsError, match="unknown setting"):
        unset_env(path, ["colour"])
    assert not path.exists()
