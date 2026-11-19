from pathlib import Path

import pytest
from conftest import write

from txtures.errors import SettingsError
from txtures.settings import (
    LOCATION_KEYS,
    resolve_settings,
    set_site,
    unset_site,
    use_site,
    uses_default_locations,
)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """The user's home directory."""
    path = tmp_path / "home"
    path.mkdir()
    return path


def resolve(home, flags=None, env=None, cwd=None, **options):
    return resolve_settings(flags, env=env or {}, user_home=home, cwd=cwd or home, **options)


def pointer(home: Path, txh: Path | str) -> Path:
    return write(home / ".config" / "txtures" / "home-path", f"{txh}\n")


def test_defaults_are_in_txtures_home(home):
    settings = resolve(home)
    txh = home / "txtures-home"
    assert settings.home == txh
    assert settings.site_document == txh / "site.yml"
    assert settings.definitions == txh / "definitions"
    assert settings.inputs == txh / "inputs"
    assert settings.outputs == txh / "outputs"
    assert settings.cache == txh / "cache"
    assert settings.resource_store == txh / "resources" / "store"
    assert settings.hf_cache == txh / "resources" / "huggingface" / "hub"
    assert settings.executor == "auto"
    assert settings.max_variants == 64
    assert all(setting.source == "default" for setting in settings.values.values())


def test_shared_locations_are_flagged_only_when_all_are_defaults(home):
    assert uses_default_locations(resolve(home))
    assert not uses_default_locations(resolve(home, env={"TXTURES_CACHE": "/shared/cache"}))
    assert not uses_default_locations(resolve(home, env={"TXTURES_HOME": "/lab"}))
    assert uses_default_locations(resolve(home, env={"TXTURES_INPUTS": "/mine"}))


def test_locations_follow_the_home_unless_overridden(home, tmp_path):
    txh = tmp_path / "lab"
    write(txh / "site.yml", "slurm:\n  partition: lab\n  cpus: 2\n")
    pointer(home, txh)
    settings = resolve(
        home,
        flags={"inputs": "/flag/inputs", "slurm.cpus": "8"},
        env={"TXTURES_INPUTS": "/env/inputs", "TXTURES_CACHE": "rel", "TXTURES_SLURM_ACCOUNT": "x"},
    )
    assert settings.home == txh
    assert settings.values["home"].source.startswith("home-path")
    assert settings.inputs == Path("/flag/inputs")
    assert settings.values["inputs"].source == "command line"
    assert settings.cache == home / "rel"
    assert settings.values["cache"].source == "environment (TXTURES_CACHE)"
    for key in ("definitions", "outputs", "resources"):
        assert settings[key] == txh / key
        assert settings.values[key].source == "home"
    assert settings["slurm.partition"] == "lab"
    assert settings.values["slurm.partition"].source == f"site settings ({txh / 'site.yml'})"
    assert settings["slurm.cpus"] == 8
    assert settings["slurm.account"] == "x"

    # TXTURES_HOME outranks the home-path file and moves every location with it.
    other = resolve(home, env={"TXTURES_HOME": str(tmp_path / "other")})
    assert all(other[key] == tmp_path / "other" / key for key in LOCATION_KEYS)
    assert other["slurm.partition"] is None


def test_xdg_config_home_locates_the_home_path_file(home, tmp_path):
    xdg = tmp_path / "xdg"
    write(xdg / "txtures" / "home-path", "/xdg/home\n")
    settings = resolve(home, env={"XDG_CONFIG_HOME": str(xdg)})
    assert settings.home == Path("/xdg/home")


def test_the_home_path_file_holds_one_path(home, tmp_path):
    pointer(home, "~/lab")
    assert resolve(home).home == Path("~/lab").expanduser()
    pointer(home, "/a\n/b")
    with pytest.raises(SettingsError, match="must hold one line"):
        resolve(home)
    written, legacy = use_site(tmp_path / "lab", env={}, user_home=home)
    assert written.read_text() == f"{tmp_path / 'lab'}\n"
    assert legacy is None
    assert resolve(home).home == tmp_path / "lab"


def test_nested_execution_settings_merge_by_key(home, tmp_path):
    write(tmp_path / "lab" / "site.yml", "execution:\n  device: cuda\n  max_variants: 8\n")
    settings = resolve(home, flags={"home": str(tmp_path / "lab"), "execution.device": "cpu"})
    assert settings.device == "cpu"
    assert settings.max_variants == 8


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("colour: blue\n", "unknown setting 'colour'"),
        ("execution:\n  threads: 4\n", "unknown setting 'execution.threads'"),
        ("execution:\n  max_variants: 0\n", "positive integer"),
        ("execution:\n  max_variants: true\n", "positive integer"),
        ("execution: fast\n", "unknown setting 'execution'"),
        ("home: /x\n", "'home' cannot be a site setting"),
        ("cache: /x\n", "'cache' cannot be a site setting"),
        ("items: /x\n", "'items' is now called 'cache'"),
    ],
)
def test_invalid_site_settings_are_rejected(home, tmp_path, text, message):
    write(tmp_path / "lab" / "site.yml", text)
    with pytest.raises(SettingsError, match=message):
        resolve(home, flags={"home": str(tmp_path / "lab")})


def test_old_names_are_refused_with_their_new_names(home):
    with pytest.raises(SettingsError, match="'derivatives' is now called 'outputs'"):
        resolve(home, flags={"derivatives": "/x"})
    with pytest.raises(SettingsError, match="TXTURES_DERIVATIVES is now called TXTURES_OUTPUTS"):
        resolve(home, env={"TXTURES_DERIVATIVES": "/x"})


def test_executor_must_be_a_known_choice(home):
    with pytest.raises(SettingsError, match="must be one of local, slurm"):
        resolve(home, flags={"execution.executor": "pbs"})


def test_set_site_writes_nested_yaml_and_keeps_other_values(home, tmp_path):
    txh = tmp_path / "lab"
    write(txh / "site.yml", "# comment\nslurm:\n  account: lab\n")
    path = set_site(txh, [("execution.device", ["cuda:1"])])
    set_site(txh, [("slurm.options", ["--exclude=n1"]), ("slurm.cpus", ["2"])])
    text = path.read_text()
    assert path == txh / "site.yml"
    assert "execution:\n  device: cuda:1\n" in text
    settings = resolve(home, flags={"home": str(txh)})
    assert settings.device == "cuda:1"
    assert settings["slurm.options"] == ["--exclude=n1"]
    assert settings["slurm.account"] == "lab"
    assert settings["slurm.cpus"] == 2
    assert unset_site(txh, ["execution.device", "slurm.gpus"]) == (path, ["execution.device"])
    assert unset_site(txh, ["execution.device"]) == (path, [])
    assert resolve(home, flags={"home": str(txh)}).device == "auto"


def test_set_site_needs_an_existing_home(tmp_path):
    with pytest.raises(SettingsError, match="does not exist; create it with `txtures site init`"):
        set_site(tmp_path / "absent", [("slurm.cpus", ["2"])])


@pytest.mark.parametrize(
    ("key", "values", "message"),
    [
        ("colour", ["blue"], "unknown setting"),
        ("slurm.cpus", ["0"], "positive integer"),
        ("execution.executor", ["pbs"], "must be one of"),
        ("slurm.partition", ["a", "b"], "takes one value"),
        ("home", ["/x"], "txtures site use PATH"),
        ("outputs", ["/x"], "outputs is always HOME/outputs"),
        ("derivatives", ["/x"], "now called 'outputs'"),
    ],
)
def test_set_site_rejects_invalid_values_without_writing(tmp_path, key, values, message):
    txh = tmp_path / "lab"
    txh.mkdir()
    with pytest.raises(SettingsError, match=message):
        set_site(txh, [(key, values)])
    assert not (txh / "site.yml").exists()


# Settings kept by earlier versions


def legacy(home: Path, text: str) -> Path:
    return write(home / ".config" / "txtures" / "settings.yml", text)


def test_personal_settings_of_earlier_versions_are_refused_with_steps(home, tmp_path):
    lab = tmp_path / "lab"
    path = legacy(
        home,
        f"definitions: {lab / 'definitions'}\ninputs: {lab / 'inputs'}\n"
        f"derivatives: {lab / 'derivatives'}\nitems: /elsewhere/items\n"
        "slurm:\n  options: [-a, -b]\n",
    )
    with pytest.raises(SettingsError) as raised:
        resolve(home)
    message = str(raised.value)
    assert f"{path} is from an earlier txtures" in message
    assert f"To move to a home at {lab}:" in message
    assert f"1. move or symlink {lab / 'derivatives'} to {lab / 'outputs'}" in message
    assert f"2. move or symlink /elsewhere/items to {lab / 'cache'}" in message
    assert f"3. txtures site use {lab}  (this also removes {path})" in message
    assert "4. txtures site set slurm.options=-a slurm.options=-b" in message
    assert "inputs" not in message.split("To move")[1]  # already in place

    settings = resolve(home, tolerate_stale=True)
    assert settings.legacy is not None
    assert settings.home == home / "txtures-home"

    written, removed = use_site(lab, env={}, user_home=home)
    assert written == home / ".config" / "txtures" / "home-path"
    assert removed["cache"] == Path("/elsewhere/items")
    assert not path.exists()
    assert resolve(home).home == lab


def test_site_settings_inside_the_definitions_store_are_refused(home, tmp_path):
    lab = tmp_path / "lab"
    write(lab / "definitions" / "site" / "site.yml", "slurm:\n  partition: p\n")
    pointer(home, lab)
    with pytest.raises(SettingsError) as raised:
        resolve(home)
    message = str(raised.value)
    assert f"1. txtures site use {lab}\n" in message
    assert "2. txtures site set slurm.partition=p" in message
    assert f"3. remove {lab / 'definitions' / 'site'}" in message
    assert resolve(home, tolerate_stale=True)["slurm.partition"] is None
