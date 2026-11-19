import json
import os
import subprocess
from pathlib import Path

import pytest

from txtures import environment, installation
from txtures.errors import SettingsError
from txtures.installation import LAUNCHER_MARKER, configure, install_user_launcher, read
from txtures.settings import resolve_settings


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    path = tmp_path / "txtures-install"
    path.mkdir()
    return path


def configured(prefix, tmp_path, **options):
    options.setdefault("compute", "cpu")
    return configure(prefix, tmp_path / "checkout", **options)


# Environment


def test_apply_replaces_inherited_hugging_face_variables(tmp_path):
    settings = resolve_settings({"resources": str(tmp_path / "res")}, env={}, user_home=tmp_path)
    env = {
        "HF_HOME": "/elsewhere",
        "HF_HUB_OFFLINE": "1",
        "HF_TOKEN": "secret",
        "TRANSFORMERS_CACHE": "/old",
        "HUGGINGFACE_HUB_CACHE": "/old",
        "PATH": "/bin",
    }
    environment.apply(settings, env, home=tmp_path)
    assert env == {
        "PATH": "/bin",
        "HF_HOME": str(tmp_path / "res" / "huggingface"),
        "HF_HUB_CACHE": str(tmp_path / "res" / "huggingface" / "hub"),
        "HF_TOKEN_PATH": str(tmp_path / ".config" / "txtures" / "hf-token"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


# Installation records


def test_first_configuration_records_only_given_settings(prefix, tmp_path):
    installed, messages = configured(prefix, tmp_path)
    assert installed.compute == "cpu"
    assert installed.settings == {}
    record = json.loads((prefix / "install.json").read_text())
    assert record["environment"] == str(prefix / "env")
    assert messages[0].startswith("installed txtures")


def add_old_settings(prefix: Path, settings: dict) -> None:
    """Make the record look like one written when installations held settings."""
    path = prefix / "install.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "settings": settings}))


def test_reconfiguration_removes_settings_left_by_earlier_versions(prefix, tmp_path):
    configured(prefix, tmp_path)
    add_old_settings(prefix, {"resources": "/shared/res", "execution.executor": "slurm"})
    assert read(prefix).settings == {"resources": "/shared/res", "execution.executor": "slurm"}
    installed, messages = configured(prefix, tmp_path, compute="cuda")
    assert installed.compute == "cuda"
    assert installed.settings == {}
    assert "settings" not in json.loads((prefix / "install.json").read_text())
    assert (
        "removed resources, execution.executor from the installation record, which no longer "
        "holds settings; to keep them, move or symlink /shared/res to HOME/resources; "
        "run `txtures site set execution.executor=slurm`"
    ) in messages


def test_settings_in_an_earlier_record_are_refused(prefix, tmp_path, monkeypatch):
    from txtures import api, installation

    configured(prefix, tmp_path)
    add_old_settings(prefix, {"resources": "/shared/res"})
    monkeypatch.setattr(installation, "current", lambda: read(prefix))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    with pytest.raises(SettingsError, match="installations no longer hold settings"):
        api.settings()
    assert api.settings(tolerate_stale=True).resources == (
        tmp_path / "home" / "txtures-home" / "resources"
    )


def test_invalid_record_is_reported(prefix):
    (prefix / "install.json").write_text('{"record_version": 99}')
    with pytest.raises(SettingsError, match="invalid installation record"):
        read(prefix)


def test_current_finds_the_record_of_the_running_environment(prefix, tmp_path, monkeypatch):
    configured(prefix, tmp_path)
    monkeypatch.setattr(installation.sys, "prefix", str(tmp_path / "elsewhere" / "env"))
    assert installation.current() is None
    monkeypatch.setattr(installation.sys, "prefix", str(prefix / "env"))
    assert installation.current().prefix == prefix


# Launchers


def test_launcher_runs_the_environment_in_isolated_mode(prefix, tmp_path):
    # A stand-in interpreter that prints the arguments it receives.
    python = prefix / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    configured(prefix, tmp_path)
    launcher = prefix / "bin" / "txtures"
    result = subprocess.run(
        [launcher, "run", "two words"], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == ["-I", "-m", "txtures", "run", "two words"]


def test_user_launcher_is_created_then_kept_for_another_installation(tmp_path):
    bin_dir = tmp_path / "bin"
    first, second = tmp_path / "a" / "bin" / "txtures", tmp_path / "b" / "bin" / "txtures"
    assert "now runs" in install_user_launcher(bin_dir, first, make_default=False)
    assert "already runs" in install_user_launcher(bin_dir, first, make_default=False)
    assert "--default" in install_user_launcher(bin_dir, second, make_default=False)
    assert str(first) in (bin_dir / "txtures").read_text()
    assert "now runs" in install_user_launcher(bin_dir, second, make_default=True)
    assert str(second) in (bin_dir / "txtures").read_text()


def test_user_launcher_never_replaces_another_command(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "txtures").write_text("#!/bin/sh\necho mine\n")
    message = install_user_launcher(bin_dir, tmp_path / "bin" / "x", make_default=True)
    assert "not a txtures launcher" in message
    assert (bin_dir / "txtures").read_text() == "#!/bin/sh\necho mine\n"
    assert LAUNCHER_MARKER not in (bin_dir / "txtures").read_text()


def test_site_option_chooses_the_site(prefix, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    txh = tmp_path / "txh"
    txh.mkdir()
    _, messages = configured(prefix, tmp_path, site=txh)
    assert messages[-1].startswith(f"set home to {txh}")
    pointer = tmp_path / "home" / ".config" / "txtures" / "home-path"
    assert pointer.read_text() == f"{txh}\n"
    with pytest.raises(SettingsError, match="not a txtures home directory"):
        configured(prefix, tmp_path, site=tmp_path / "missing")


def test_run_context_carries_the_saved_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    settings = resolve_settings({}, env={}, user_home=tmp_path)
    assert environment.run_context(settings).hf_token is None
    path = environment.token_path(dict(os.environ), tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("hf_secret\n")
    context = environment.run_context(settings)
    assert context.hf_token == "hf_secret"
    assert "hf_secret" not in repr(context)


# Reinstalling with the recorded choices


def test_the_record_keeps_dev_and_the_command_directory(prefix, tmp_path):
    installed, _ = configured(prefix, tmp_path, dev=True, bin_dir=tmp_path / "bin")
    assert installed.dev is True
    assert installed.record["bin_dir"] == str(tmp_path / "bin")
    assert installation.reinstall_arguments(installed) == [
        "--prefix",
        str(prefix),
        "--compute",
        "cpu",
        "--dev",
        "--bin-dir",
        str(tmp_path / "bin"),
    ]


def test_reinstall_options_change_the_recorded_choices(prefix, tmp_path):
    installed, _ = configured(prefix, tmp_path, dev=True)
    arguments = installation.reinstall_arguments(
        installed, upgrade=True, compute="cuda", dev=False, make_default=True
    )
    assert arguments == [
        "--upgrade",
        "--prefix",
        str(prefix),
        "--compute",
        "cuda",
        "--no-launcher",
        "--default",
    ]


def test_records_before_dev_was_recorded(prefix, tmp_path, monkeypatch):
    configured(prefix, tmp_path)
    path = prefix / "install.json"
    record = json.loads(path.read_text())
    del record["dev"], record["bin_dir"]
    path.write_text(json.dumps(record))
    installed = read(prefix)
    assert installed.dev is None
    arguments = installation.reinstall_arguments(installed)
    # This test environment has pytest, which marks the development tools.
    assert arguments == ["--prefix", str(prefix), "--compute", "cpu", "--dev"]
    monkeypatch.setattr(installation.importlib.util, "find_spec", lambda name: None)
    assert "--dev" not in installation.reinstall_arguments(installed)
