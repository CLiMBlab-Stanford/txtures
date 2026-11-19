import json
from pathlib import Path

import pytest
from conftest import write

import txtures
from txtures import installation
from txtures.cli import main
from txtures.settings import SPECS


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep CLI tests away from the real home directory and lab storage."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    for spec in SPECS:
        monkeypatch.delenv(spec.env, raising=False)
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("TXTURES_HOME", str(tmp_path / "txh"))


@pytest.fixture
def txh(tmp_path) -> Path:
    """The txtures home the CLI tests use."""
    return tmp_path / "txh"


def test_version_flag_reports_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"txtures {txtures.__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: txtures" in capsys.readouterr().out


def test_site_ls_reports_values_sources_and_meanings(capsys, monkeypatch, tmp_path, txh):
    # The tests may run inside an ./install environment, which has a record.
    monkeypatch.setattr(installation, "current", lambda: None)
    assert main(["--cache", str(tmp_path / "c"), "site", "ls", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installation"] is None
    assert payload["site_settings"] == str(txh / "site.yml")
    rows = {row["key"]: row for row in payload["settings"]}
    assert rows["cache"] == {
        "key": "cache",
        "value": str(tmp_path / "c"),
        "source": "command line",
        "description": "stored per-document results",
    }
    assert rows["home"]["source"] == "environment (TXTURES_HOME)"
    assert rows["resources"] == {
        "key": "resources",
        "value": str(txh / "resources"),
        "source": "home",
        "description": "models and the Hugging Face cache",
    }
    assert main(["site", "show"]) == 0
    out = capsys.readouterr().out
    assert "slurm.partition" in out and "partition to submit to" in out


def test_site_init_creates_the_home_layout(capsys, tmp_path, txh):
    assert main(["site", "init", "--no-git"]) == 0
    assert f"created txtures home {txh}" in capsys.readouterr().out
    assert sorted(p.name for p in txh.iterdir()) == [
        ".gitignore",
        "README.md",
        "cache",
        "definitions",
        "inputs",
        "outputs",
        "resources",
        "site.yml",
    ]
    assert not (txh / ".git").exists()
    assert main(["def", "validate"]) == 0
    capsys.readouterr()
    assert main(["site", "init"]) == 1
    assert "it already has site.yml, README.md, .gitignore, definitions" in capsys.readouterr().err
    assert main(["site", "init", "--no-git", str(tmp_path / "other")]) == 0
    assert f"txtures site use {tmp_path / 'other'}" in capsys.readouterr().err


def test_site_set_and_unset_edit_the_site_in_use(capsys, tmp_path, txh):
    site = txh / "site.yml"
    txh.mkdir()
    assert main(["site", "set", "slurm.partition", "gpu"]) == 0
    assert main(["site", "set", "slurm.options", "--", "--exclude=n1", "--qos=low"]) == 0
    out = capsys.readouterr().out
    assert f"wrote {site}" in out
    assert f"slurm.partition is now gpu (site settings ({site}))" in out
    assert "slurm:\n  partition: gpu\n  options:\n  - --exclude=n1\n" in site.read_text()
    assert not (tmp_path / "home" / ".config" / "txtures").exists()

    assert main(["site", "set", "slurm.gpus", "-1"]) == 1
    assert "non-negative integer" in capsys.readouterr().err
    assert main(["site", "set", "colour", "blue"]) == 1
    assert "see `txtures site ls`" in capsys.readouterr().err
    assert main(["site", "set", "home", "/x"]) == 1
    assert "txtures site use PATH" in capsys.readouterr().err
    assert main(["site", "set", "cache=/x"]) == 1
    err = capsys.readouterr().err
    assert "cache is always HOME/cache" in err and "txtures env set cache=PATH" in err
    assert main(["site", "set", "items=/x"]) == 1
    assert "'items' is now called 'cache'" in capsys.readouterr().err

    assert main(["site", "unset", "slurm.partition", "slurm.cpus"]) == 0
    out = capsys.readouterr().out
    assert "slurm.partition is now (unset) (default)" in out
    assert "slurm.cpus is not set in" in out
    assert "partition" not in site.read_text()


def test_site_set_needs_an_existing_home(capsys):
    assert main(["site", "set", "slurm.cpus=2"]) == 1
    assert "does not exist; create it with `txtures site init`" in capsys.readouterr().err


def test_site_settings_cannot_hold_locations(capsys, txh):
    write(txh / "site.yml", "inputs: /x\n")
    assert main(["status"]) == 1
    assert "'inputs' cannot be a site setting" in capsys.readouterr().err


def test_site_set_takes_several_key_value_pairs(capsys, txh):
    site = txh / "site.yml"
    txh.mkdir()
    args = ["slurm.partition=gpu", "slurm.account=lab", "slurm.options=--exclude=n1"]
    assert main(["site", "set", *args, "slurm.options=--qos=low"]) == 0
    out = capsys.readouterr().out
    assert "slurm.partition is now gpu" in out and "slurm.account is now lab" in out
    text = site.read_text()
    assert "partition: gpu" in text and "account: lab" in text
    assert "options:\n  - --exclude=n1\n  - --qos=low\n" in text

    # One invalid pair writes nothing.
    assert main(["site", "set", "slurm.partition=cpu", "slurm.gpus=-1"]) == 1
    assert "non-negative integer" in capsys.readouterr().err
    assert site.read_text() == text
    assert main(["site", "set", "slurm.partition=a", "slurm.partition=b"]) == 1
    assert "slurm.partition is given more than once" in capsys.readouterr().err
    assert main(["site", "set", "slurm.partition=a", "gpu"]) == 1
    assert "expected KEY=VALUE, got 'gpu'" in capsys.readouterr().err
    assert main(["site", "set", "slurm.partition"]) == 1
    assert "give a value for slurm.partition" in capsys.readouterr().err


def test_env_set_ls_and_unset_edit_the_shell_startup_file(capsys, tmp_path, txh, monkeypatch):
    bashrc = write(tmp_path / "home" / ".bashrc", "alias ll='ls -l'\n")
    txh.mkdir()
    assert main(["site", "set", "slurm.partition=gpu"]) == 0
    capsys.readouterr()

    args = ["slurm.partition=debug", "slurm.options=--qos=low", "slurm.options=--exclude=n 1"]
    assert main(["env", "set", *args, "cache=rel"]) == 0
    captured = capsys.readouterr()
    assert f"to {bashrc}" in captured.out
    assert "slurm.partition will be debug (environment (TXTURES_SLURM_PARTITION))" in captured.out
    assert f"source {bashrc}" in captured.err
    text = bashrc.read_text()
    assert text.startswith("alias ll='ls -l'\n\n# >>> txtures env >>>\n")
    assert "export TXTURES_SLURM_PARTITION=debug\n" in text
    assert f"export TXTURES_CACHE={Path.cwd() / 'rel'}\n" in text
    assert "export TXTURES_SLURM_OPTIONS='--qos=low '\"'\"'--exclude=n 1'\"'\"''\n" in text
    assert (txh / "site.yml").read_text().count("gpu") == 1

    assert main(["env", "ls", "--json"]) == 0
    rows = {row["key"]: row for row in json.loads(capsys.readouterr().out)["variables"]}
    assert rows["slurm.partition"]["status"] == "not in this shell yet"

    # A shell that sources the file sees the variables, which override the site.
    monkeypatch.setenv("TXTURES_SLURM_PARTITION", "debug")
    monkeypatch.setenv("TXTURES_SLURM_OPTIONS", "--qos=low '--exclude=n 1'")
    assert main(["site", "ls", "--json"]) == 0
    rows = {row["key"]: row for row in json.loads(capsys.readouterr().out)["settings"]}
    assert rows["slurm.partition"]["value"] == "debug"
    assert rows["slurm.options"]["value"] == ["--qos=low", "--exclude=n 1"]
    assert main(["env", "ls"]) == 0
    out = capsys.readouterr().out
    assert "TXTURES_SLURM_PARTITION  debug  (slurm.partition; in effect)" in out
    assert "TXTURES_HOME" in out and "not by txtures env" in out

    assert main(["env", "unset", "slurm.partition", "slurm.cpus"]) == 0
    out = capsys.readouterr().out
    assert "removed TXTURES_SLURM_PARTITION" in out
    assert "TXTURES_SLURM_CPUS is not set" in out
    assert "slurm.partition will be gpu (site settings" in out
    assert main(["env", "unset", "slurm.options", "cache"]) == 0
    assert bashrc.read_text() == "alias ll='ls -l'\n"

    assert main(["env", "set", "slurm.gpus=-1"]) == 1
    assert "non-negative integer" in capsys.readouterr().err
    assert bashrc.read_text() == "alias ll='ls -l'\n"


def test_env_writes_the_file_given_and_takes_locations(capsys, tmp_path):
    profile = tmp_path / "profile"
    assert main(["env", "set", "--file", str(profile), "definitions=/lab/defs"]) == 0
    assert "export TXTURES_DEFINITIONS=/lab/defs\n" in profile.read_text()
    assert "definitions will be /lab/defs" in capsys.readouterr().out


def test_old_environment_variables_are_refused(capsys, monkeypatch):
    monkeypatch.setenv("TXTURES_ITEMS", "/x")
    assert main(["status"]) == 1
    assert "TXTURES_ITEMS is now called TXTURES_CACHE" in capsys.readouterr().err


def test_site_use_chooses_the_home_and_the_environment_still_overrides(
    capsys, tmp_path, monkeypatch
):
    lab = tmp_path / "lab"
    write(lab / "site.yml", "slurm:\n  partition: lab\n")
    (lab / "definitions").mkdir()
    assert main(["site", "use", str(lab)]) == 0
    captured = capsys.readouterr()
    assert "TXTURES_HOME" in captured.err  # overrides what was written
    pointer = tmp_path / "home" / ".config" / "txtures" / "home-path"
    assert pointer.read_text() == f"{lab}\n"

    monkeypatch.delenv("TXTURES_HOME")
    monkeypatch.setenv("TXTURES_CACHE", str(tmp_path / "mine"))
    assert main(["site", "use", str(lab)]) == 0
    out = capsys.readouterr().out
    assert f"home is now {lab} (home-path ({pointer}))" in out
    assert f"  definitions: {lab / 'definitions'} (home)" in out
    assert f"  cache: {tmp_path / 'mine'} (environment (TXTURES_CACHE))" in out
    assert main(["site", "use", str(tmp_path / "missing")]) == 1
    assert "txtures site init PATH" in capsys.readouterr().err


def test_settings_of_earlier_versions_block_runs_until_moved(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("TXTURES_HOME")
    lab = tmp_path / "lab"
    (lab / "definitions").mkdir(parents=True)
    write(lab / "definitions" / "site" / "site.yml", "slurm:\n  cpus: 2\n")
    personal = tmp_path / "home" / ".config" / "txtures" / "settings.yml"
    write(
        personal,
        f"definitions: {lab / 'definitions'}\nitems: {lab / 'items'}\nslurm:\n  account: a\n",
    )
    assert main(["status"]) == 1
    err = capsys.readouterr().err
    assert "are from an earlier txtures" in err
    assert f"1. move or symlink {lab / 'items'} to {lab / 'cache'}" in err
    assert f"2. txtures site use {lab}  (this also removes {personal})" in err
    assert "3. txtures site set slurm.cpus=2 slurm.account=a" in err
    assert f"4. remove {lab / 'definitions' / 'site'}" in err

    assert main(["site", "use", str(lab)]) == 0
    assert "from an earlier txtures" in capsys.readouterr().err
    assert not personal.exists()
    assert main(["site", "set", "slurm.account=a", "slurm.cpus=2"]) == 0
    assert "warning:" in capsys.readouterr().err
    assert main(["status"]) == 1  # the old site document remains
    (lab / "definitions" / "site" / "site.yml").unlink()
    (lab / "definitions" / "site").rmdir()
    assert main(["status"]) == 0


def test_default_locations_draw_one_note(capsys, monkeypatch, tmp_path):
    from txtures import cli

    monkeypatch.delenv("TXTURES_HOME")
    monkeypatch.setattr(cli, "_noted", False)
    (tmp_path / "home" / "txtures-home" / "inputs").mkdir(parents=True)
    assert main(["status"]) == 0
    err = capsys.readouterr().err
    assert err.count("txtures site use") == 1
    assert main(["site", "use", str(tmp_path / "home" / "txtures-home")]) == 0
    monkeypatch.setattr(cli, "_noted", False)
    assert main(["status"]) == 0
    assert "site use" not in capsys.readouterr().err


def test_def_init_then_validate(capsys, tmp_path):
    store = tmp_path / "standalone"
    assert main(["def", "init", "--no-git", str(store)]) == 0
    assert "created definitions store" in capsys.readouterr().out
    assert sorted(p.name for p in store.iterdir()) == ["README.md", "configs", "workflows"]
    assert main(["def", "validate", str(store)]) == 0
    assert "is valid" in capsys.readouterr().out


def test_def_validate_reports_problems_and_fails(capsys, txh):
    write(txh / "definitions" / "workflows" / "w.yml", "lm_scores: nothere\n")
    write(txh / "definitions" / "site" / "site.yml", "")
    assert main(["def", "validate", "--json", str(txh / "definitions")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    messages = [problem["message"] for problem in payload["problems"]]
    assert any("nothere" in message for message in messages)
    assert any("site settings now live in the txtures home" in message for message in messages)


def test_def_validate_uses_the_sites_variant_limit(capsys, txh):
    write(txh / "site.yml", "execution:\n  max_variants: 1\n")
    write(txh / "definitions" / "configs" / "lm_scores" / "main.yml", "model: [a, b]\n")
    assert main(["def", "validate"]) == 1
    assert "more than execution.max_variants (1)" in capsys.readouterr().out


def test_errors_are_reported_without_tracebacks(capsys):
    assert main(["def", "validate"]) == 1
    assert "txtures: error: definitions store" in capsys.readouterr().err


def test_inputs_list_reports_valid_invalid_and_ignored(capsys, tmp_path):
    root = tmp_path / "inputs"
    write(root / "good.txt", "The horse raced.\n")
    write(root / "bad.txt", "a\n\nb\n")
    write(root / "notes.md", "")
    assert main(["--inputs", str(root), "inputs", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {entry["name"]: entry for entry in payload["inputs"]}
    assert by_name["good"]["valid"] is True
    assert (by_name["good"]["documents"], by_name["good"]["regions"]) == (1, 3)
    assert by_name["bad"]["valid"] is False
    assert payload["ignored"] == [str(root / "notes.md")]


def test_inputs_check_summarizes_or_fails(capsys, tmp_path):
    root = tmp_path / "inputs"
    write(
        root / "r.tsv", "sentence_id\tregion_id\ttext\titem\n1\t1\tThe horse\t7\n2\t1\tfell.\t7\n"
    )
    write(root / "bad.tsv", "doc_id\tregion\nx\ty\n")
    assert main(["--inputs", str(root), "inputs", "check", "r"]) == 0
    out = capsys.readouterr().out
    assert "r: region-per-row" in out
    assert "sentences: 2" in out
    assert "metadata columns: item" in out
    assert main(["--inputs", str(root), "inputs", "check", "bad"]) == 1
    out = capsys.readouterr().out
    assert "column doc_id is now document_id" in out
    assert "column region is now text" in out


def test_inputs_check_unknown_name_is_an_error(capsys, tmp_path):
    (tmp_path / "inputs").mkdir()
    assert main(["--inputs", str(tmp_path / "inputs"), "inputs", "check", "nope"]) == 1
    assert "no input named 'nope'" in capsys.readouterr().err


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """Run CLI engine commands against the stand-in modules in temporary stores."""
    import fake_modules

    from txtures import api
    from txtures.definitions import Definitions

    fake_modules.reset()
    monkeypatch.setattr(
        api, "load_definitions", lambda settings: Definitions.packaged(fake_modules.registry())
    )
    for key in ("inputs", "outputs", "resources", "cache"):
        monkeypatch.setenv(f"TXTURES_{key.upper()}", str(tmp_path / key))
    write(tmp_path / "inputs" / "doc.txt", "The horse raced.\nIt fell.\n")
    yield tmp_path
    fake_modules.reset()


def test_plan_run_status_and_gc(capsys, fake_engine):
    assert main(["plan", "doc", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["table_current"] is False
    assert plan["columns"] == ["length", "double"]
    scores = next(v for v in plan["variants"] if v["persisted"])
    assert (scores["present"], scores["missing"]) == (0, 2)
    assert any("revision is not pinned" in note for note in plan["notes"])

    assert main(["run", "doc"]) == 0
    out = capsys.readouterr()
    assert "fake_scores [main]: 2 computed, 0 reused" in out.out
    assert "computing 2 of 2 items" in out.err
    table = Path(plan["table"])
    assert (table / "doc.tsv").is_file()

    assert main(["run", "doc"]) == 0
    assert "is already current" in capsys.readouterr().out

    write(fake_engine / "inputs" / "doc.txt", "Something else.\n")
    assert main(["run", "doc"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    status = capsys.readouterr().out
    assert status.count("current") == 1 and status.count("stale") == 1

    assert main(["gc", "--dry-run"]) == 0
    assert f"would remove {table} (stale)" in capsys.readouterr().out
    assert table.exists()
    assert main(["gc"]) == 0
    assert not table.exists()
    assert main(["gc"]) == 0
    assert "nothing to remove" in capsys.readouterr().out


def test_run_rejects_unknown_workflow(capsys, fake_engine):
    assert main(["run", "doc", "-w", "missing"]) == 1
    assert "no workflow named 'missing'" in capsys.readouterr().err


# txtures install


def test_install_runs_the_checkout_script_with_the_recorded_choices(capsys, monkeypatch, tmp_path):
    from txtures import cli

    checkout = tmp_path / "checkout"
    script = write(checkout / "install", "#!/bin/sh\n")
    script.chmod(0o755)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    installed, _ = installation.configure(prefix, checkout, compute="cuda", dev=False)
    monkeypatch.setattr(installation, "current", lambda: installed)
    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: calls.append((path, argv)))
    # Invalid settings do not stop it.
    write(tmp_path / "txh" / "site.yml", "execution: [\n")

    assert main(["install", "--upgrade", "--dev"]) == 0
    expected = [str(script), "--upgrade", "--prefix", str(prefix), "--compute", "cuda"]
    assert calls == [(script, [*expected, "--dev", "--no-launcher"])]
    assert capsys.readouterr().out.startswith(f"running {script} --upgrade")


def test_install_needs_an_installation_and_its_checkout(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(installation, "current", lambda: None)
    assert main(["install"]) == 1
    assert "not installed by ./install" in capsys.readouterr().err

    prefix = tmp_path / "prefix"
    prefix.mkdir()
    installed, _ = installation.configure(prefix, tmp_path / "gone", compute="cpu")
    monkeypatch.setattr(installation, "current", lambda: installed)
    assert main(["install"]) == 1
    assert "has moved or was removed" in capsys.readouterr().err
