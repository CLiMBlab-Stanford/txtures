from pathlib import Path

import pytest
from conftest import write

from txtures.errors import SettingsError
from txtures.settings import (
    resolve_settings,
    set_personal,
    unset_personal,
    uses_default_locations,
)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def resolve(home, flags=None, env=None, cwd=None):
    return resolve_settings(flags, env=env or {}, home=home, cwd=cwd or home)


def test_defaults_are_in_the_home_directory(home):
    settings = resolve(home)
    assert settings.definitions == home / "txtures-definitions"
    assert settings.inputs == home / "txtures-store" / "inputs"
    assert settings.items == home / "txtures-store" / "items"
    assert settings.resources == home / "txtures-resources"
    assert settings.resource_store == home / "txtures-resources" / "store"
    assert settings.hf_cache == home / "txtures-resources" / "huggingface" / "hub"
    assert settings.executor == "local"
    assert settings.max_variants == 64
    assert all(setting.source == "default" for setting in settings.values.values())


def test_shared_locations_are_flagged_only_when_all_are_defaults(home):
    assert uses_default_locations(resolve(home))
    assert not uses_default_locations(resolve(home, env={"TXTURES_ITEMS": "/shared/items"}))


def test_precedence_flag_env_personal_site_default(home, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "inputs: site-inputs\nitems: site-items\nresources: r\n")
    write(
        home / ".config" / "txtures" / "settings.yml",
        f"definitions: {store}\ninputs: personal-inputs\nitems: personal-items\n",
    )
    settings = resolve(
        home,
        flags={"inputs": "/flag/inputs"},
        env={"TXTURES_INPUTS": "/env/inputs", "TXTURES_ITEMS": "/env/items"},
    )
    assert settings.inputs == Path("/flag/inputs")
    assert settings.values["inputs"].source == "command line"
    assert settings.items == Path("/env/items")
    assert settings.values["items"].source == "environment (TXTURES_ITEMS)"
    assert settings.definitions == store
    assert settings.values["definitions"].source.startswith("personal settings")
    assert settings.resources == store / "site" / "r"
    assert settings.values["resources"].source.startswith("site settings")
    assert settings.derivatives == home / "txtures-store" / "derivatives"


def test_relative_paths_resolve_against_their_source(home, tmp_path):
    personal = write(home / ".config" / "txtures" / "settings.yml", "items: cache/items\n")
    cwd = tmp_path / "work"
    settings = resolve(home, flags={"inputs": "texts"}, cwd=cwd)
    assert settings.inputs == cwd / "texts"
    assert settings.items == personal.parent / "cache" / "items"


def test_xdg_config_home_locates_personal_settings(home, tmp_path):
    xdg = tmp_path / "xdg"
    write(xdg / "txtures" / "settings.yml", "inputs: /xdg/inputs\n")
    settings = resolve(home, env={"XDG_CONFIG_HOME": str(xdg)})
    assert settings.inputs == Path("/xdg/inputs")


def test_nested_execution_settings_merge_by_key(home, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "execution:\n  device: cuda\n  max_variants: 8\n")
    write(home / ".config" / "txtures" / "settings.yml", "execution:\n  device: cpu\n")
    settings = resolve(home, flags={"definitions": str(store)})
    assert settings.device == "cpu"
    assert settings.max_variants == 8


def test_site_settings_cannot_choose_the_definitions_store(home, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "definitions: /elsewhere\n")
    with pytest.raises(SettingsError, match="cannot be set in site settings"):
        resolve(home, flags={"definitions": str(store)})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("colour: blue\n", "unknown setting 'colour'"),
        ("execution:\n  threads: 4\n", "unknown setting 'execution.threads'"),
        ("execution:\n  max_variants: 0\n", "positive integer"),
        ("execution:\n  max_variants: true\n", "positive integer"),
        ("inputs: 3\n", "non-empty path"),
        ("execution: fast\n", "unknown setting 'execution'"),
    ],
)
def test_invalid_settings_documents_are_rejected(home, text, message):
    write(home / ".config" / "txtures" / "settings.yml", text)
    with pytest.raises(SettingsError, match=message):
        resolve(home)


def test_installation_overrides_personal_and_site_but_not_env(home, tmp_path):
    write(home / ".config" / "txtures" / "settings.yml", "resources: /personal/res\n")
    installed = {"resources": "/installed/res", "execution.executor": "slurm"}
    settings = resolve_settings(
        env={}, home=home, installation=installed, installation_source="inst"
    )
    assert settings.resources == Path("/installed/res")
    assert settings.values["resources"].source == "inst"
    assert settings.executor == "slurm"
    settings = resolve_settings(
        env={"TXTURES_RESOURCES": "/env/res"}, home=home, installation=installed
    )
    assert settings.resources == Path("/env/res")


def test_executor_must_be_a_known_choice(home):
    with pytest.raises(SettingsError, match="must be one of local, slurm"):
        resolve(home, flags={"execution.executor": "pbs"})


def test_set_personal_writes_nested_yaml_that_resolves_back(home, tmp_path):
    path = set_personal("execution.device", ["cuda:1"], env={}, home=home)
    set_personal("slurm.options", ["--exclude=n1"], env={}, home=home)
    set_personal("inputs", ["texts"], env={}, home=home, cwd=tmp_path)
    text = path.read_text()
    assert "execution:\n  device: cuda:1\n" in text
    settings = resolve(home)
    assert settings.device == "cuda:1"
    assert settings["slurm.options"] == ["--exclude=n1"]
    assert settings.inputs == tmp_path / "texts"
    assert unset_personal("execution.device", env={}, home=home) == (path, True)
    assert unset_personal("execution.device", env={}, home=home) == (path, False)
    assert resolve(home).device == "auto"


@pytest.mark.parametrize(
    ("key", "values", "message"),
    [
        ("colour", ["blue"], "unknown setting"),
        ("slurm.cpus", ["0"], "positive integer"),
        ("execution.executor", ["pbs"], "must be one of"),
        ("items", ["a", "b"], "takes one value"),
    ],
)
def test_set_personal_rejects_invalid_values(home, key, values, message):
    with pytest.raises(SettingsError, match=message):
        set_personal(key, values, env={}, home=home)
    assert not (home / ".config" / "txtures" / "settings.yml").exists()
