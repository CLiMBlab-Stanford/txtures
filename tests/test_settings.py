from pathlib import Path

import pytest
from conftest import write

from textures.errors import SettingsError
from textures.settings import resolve_settings


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def no_lab(tmp_path: Path) -> Path:
    return tmp_path / "missing-lab"


def resolve(home, lab_root, flags=None, env=None, cwd=None):
    return resolve_settings(flags, env=env or {}, home=home, cwd=cwd or home, lab_root=lab_root)


def test_defaults_without_lab_storage_are_personal(home, no_lab):
    settings = resolve(home, no_lab)
    assert settings.definitions == home / "textures-definitions"
    assert settings.inputs == home / "textures-store" / "inputs"
    assert settings.items == home / "textures-store" / "items"
    assert settings.resources == home / "textures-resources"
    assert settings.resource_store == home / "textures-resources" / "store"
    assert settings.hf_cache == home / "textures-resources" / "huggingface" / "hub"
    assert settings.executor == "local"
    assert settings.max_variants == 64
    assert all(setting.source == "default" for setting in settings.values.values())


def test_defaults_with_writable_lab_storage_share_resources_and_items(home, tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    settings = resolve(home, lab)
    assert settings.definitions == lab / "textures-definitions"
    assert settings.resources == lab / "textures-resources"
    assert settings.items == lab / "textures-store" / "items"
    assert settings.inputs == home / "textures-store" / "inputs"


def test_precedence_flag_env_personal_site_default(home, no_lab, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "inputs: site-inputs\nitems: site-items\nresources: r\n")
    write(
        home / ".config" / "textures" / "settings.yml",
        f"definitions: {store}\ninputs: personal-inputs\nitems: personal-items\n",
    )
    settings = resolve(
        home,
        no_lab,
        flags={"inputs": "/flag/inputs"},
        env={"TEXTURES_INPUTS": "/env/inputs", "TEXTURES_ITEMS": "/env/items"},
    )
    assert settings.inputs == Path("/flag/inputs")
    assert settings.values["inputs"].source == "command line"
    assert settings.items == Path("/env/items")
    assert settings.values["items"].source == "environment (TEXTURES_ITEMS)"
    assert settings.definitions == store
    assert settings.values["definitions"].source.startswith("personal settings")
    assert settings.resources == store / "site" / "r"
    assert settings.values["resources"].source.startswith("site settings")
    assert settings.derivatives == home / "textures-store" / "derivatives"


def test_relative_paths_resolve_against_their_source(home, no_lab, tmp_path):
    personal = write(home / ".config" / "textures" / "settings.yml", "items: cache/items\n")
    cwd = tmp_path / "work"
    settings = resolve(home, no_lab, flags={"inputs": "texts"}, cwd=cwd)
    assert settings.inputs == cwd / "texts"
    assert settings.items == personal.parent / "cache" / "items"


def test_xdg_config_home_locates_personal_settings(home, no_lab, tmp_path):
    xdg = tmp_path / "xdg"
    write(xdg / "textures" / "settings.yml", "inputs: /xdg/inputs\n")
    settings = resolve(home, no_lab, env={"XDG_CONFIG_HOME": str(xdg)})
    assert settings.inputs == Path("/xdg/inputs")


def test_nested_execution_settings_merge_by_key(home, no_lab, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "execution:\n  device: cuda\n  max_variants: 8\n")
    write(home / ".config" / "textures" / "settings.yml", "execution:\n  device: cpu\n")
    settings = resolve(home, no_lab, flags={"definitions": str(store)})
    assert settings.device == "cpu"
    assert settings.max_variants == 8


def test_site_settings_cannot_choose_the_definitions_store(home, no_lab, tmp_path):
    store = tmp_path / "store"
    write(store / "site" / "site.yml", "definitions: /elsewhere\n")
    with pytest.raises(SettingsError, match="cannot be set in site settings"):
        resolve(home, no_lab, flags={"definitions": str(store)})


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
def test_invalid_settings_documents_are_rejected(home, no_lab, text, message):
    write(home / ".config" / "textures" / "settings.yml", text)
    with pytest.raises(SettingsError, match=message):
        resolve(home, no_lab)


def test_installation_overrides_personal_and_site_but_not_env(home, no_lab, tmp_path):
    write(home / ".config" / "textures" / "settings.yml", "resources: /personal/res\n")
    installed = {"resources": "/installed/res", "execution.executor": "slurm"}
    settings = resolve_settings(
        env={}, home=home, lab_root=no_lab, installation=installed, installation_source="inst"
    )
    assert settings.resources == Path("/installed/res")
    assert settings.values["resources"].source == "inst"
    assert settings.executor == "slurm"
    settings = resolve_settings(
        env={"TEXTURES_RESOURCES": "/env/res"}, home=home, lab_root=no_lab, installation=installed
    )
    assert settings.resources == Path("/env/res")


def test_executor_must_be_a_known_choice(home, no_lab):
    with pytest.raises(SettingsError, match="must be one of local, slurm"):
        resolve(home, no_lab, flags={"execution.executor": "pbs"})
