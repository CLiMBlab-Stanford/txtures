import json
import os
import subprocess
from pathlib import Path

import pytest

from textures import environment, installation
from textures.errors import SettingsError
from textures.installation import LAUNCHER_MARKER, configure, install_user_launcher, read
from textures.settings import resolve_settings


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    path = tmp_path / "textures-install"
    path.mkdir()
    return path


def configured(prefix, tmp_path, **options):
    options.setdefault("compute", "cpu")
    return configure(prefix, tmp_path / "checkout", **options)


# Environment


def test_apply_replaces_inherited_hugging_face_variables(tmp_path):
    settings = resolve_settings({"resources": str(tmp_path / "res")}, env={}, home=tmp_path)
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
        "HF_TOKEN_PATH": str(tmp_path / ".config" / "textures" / "hf-token"),
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
    assert messages[0].startswith("installed textures")


def test_reconfiguration_keeps_settings_not_given(prefix, tmp_path):
    configured(prefix, tmp_path, resources=tmp_path / "shared", executor="slurm")
    installed, _ = configured(prefix, tmp_path, compute="cuda")
    assert installed.compute == "cuda"
    assert installed.settings == {
        "resources": str(tmp_path / "shared"),
        "execution.executor": "slurm",
    }


def test_invalid_record_is_reported(prefix):
    (prefix / "install.json").write_text('{"record_version": 99}')
    with pytest.raises(SettingsError, match="invalid installation record"):
        read(prefix)


def test_current_is_none_outside_an_installation():
    assert installation.current() is None


# Launchers


def test_launcher_runs_the_environment_in_isolated_mode(prefix, tmp_path):
    # A stand-in interpreter that prints the arguments it receives.
    python = prefix / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    configured(prefix, tmp_path)
    launcher = prefix / "bin" / "textures"
    result = subprocess.run(
        [launcher, "run", "two words"], capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == ["-I", "-m", "textures", "run", "two words"]


def test_user_launcher_is_created_then_kept_for_another_installation(tmp_path):
    bin_dir = tmp_path / "bin"
    first, second = tmp_path / "a" / "bin" / "textures", tmp_path / "b" / "bin" / "textures"
    assert "now runs" in install_user_launcher(bin_dir, first, make_default=False)
    assert "already runs" in install_user_launcher(bin_dir, first, make_default=False)
    assert "--default" in install_user_launcher(bin_dir, second, make_default=False)
    assert str(first) in (bin_dir / "textures").read_text()
    assert "now runs" in install_user_launcher(bin_dir, second, make_default=True)
    assert str(second) in (bin_dir / "textures").read_text()


def test_user_launcher_never_replaces_another_command(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "textures").write_text("#!/bin/sh\necho mine\n")
    message = install_user_launcher(bin_dir, tmp_path / "bin" / "x", make_default=True)
    assert "not a textures launcher" in message
    assert (bin_dir / "textures").read_text() == "#!/bin/sh\necho mine\n"
    assert LAUNCHER_MARKER not in (bin_dir / "textures").read_text()


def test_site_option_writes_personal_settings(prefix, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    store = tmp_path / "definitions"
    store.mkdir()
    _, messages = configured(prefix, tmp_path, site=store)
    assert messages[-1].startswith(f"set definitions to {store}")
    personal = tmp_path / "home" / ".config" / "textures" / "settings.yml"
    assert f"definitions: {store}" in personal.read_text()
    with pytest.raises(SettingsError, match="not a definitions store directory"):
        configured(prefix, tmp_path, site=tmp_path / "missing")


def test_run_context_carries_the_saved_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    settings = resolve_settings({}, env={}, home=tmp_path)
    assert environment.run_context(settings).hf_token is None
    path = environment.token_path(dict(os.environ), tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("hf_secret\n")
    context = environment.run_context(settings)
    assert context.hf_token == "hf_secret"
    assert "hf_secret" not in repr(context)
