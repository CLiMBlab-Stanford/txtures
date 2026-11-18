import json
from pathlib import Path

import pytest
from conftest import write

import textures
from textures.cli import main


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep CLI tests away from the real home directory and lab storage."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    for name in ("DEFINITIONS", "INPUTS", "DERIVATIVES", "ITEMS", "RESOURCES"):
        monkeypatch.delenv(f"TEXTURES_{name}", raising=False)
    monkeypatch.setenv("TEXTURES_DEFINITIONS", str(tmp_path / "defs"))


def test_version_flag_reports_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"textures {textures.__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: textures" in capsys.readouterr().out


def test_site_show_reports_sources(capsys, tmp_path):
    assert main(["--items", str(tmp_path / "items"), "site", "show", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installation"] is None
    rows = {row["key"]: row for row in payload["settings"]}
    assert rows["items"] == {
        "key": "items",
        "value": str(tmp_path / "items"),
        "source": "command line",
    }
    assert rows["definitions"]["source"] == "environment (TEXTURES_DEFINITIONS)"
    assert rows["resources"]["source"] == "default"
    assert rows["execution.executor"]["value"] == "local"


def test_def_init_then_validate(capsys, tmp_path):
    assert main(["def", "init"]) == 0
    assert (tmp_path / "defs" / "site" / "site.yml").is_file()
    assert main(["def", "validate"]) == 0
    assert "is valid" in capsys.readouterr().out


def test_def_validate_reports_problems_and_fails(capsys, tmp_path):
    write(tmp_path / "defs" / "workflows" / "w.yml", "lm_scores: nothere\n")
    assert main(["def", "validate", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert any("nothere" in problem["message"] for problem in payload["problems"])


def test_def_validate_uses_the_stores_variant_limit(capsys, tmp_path):
    defs = tmp_path / "defs"
    write(defs / "site" / "site.yml", "execution:\n  max_variants: 1\n")
    write(defs / "configs" / "lm_scores" / "main.yml", "model: [a, b]\n")
    assert main(["def", "validate"]) == 1
    assert "more than execution.max_variants (1)" in capsys.readouterr().out


def test_errors_are_reported_without_tracebacks(capsys):
    assert main(["def", "validate"]) == 1
    assert "textures: error: definitions store" in capsys.readouterr().err


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
    write(root / "r.tsv", "sentence_id\tregion\titem\n1\tThe horse\t7\n2\tfell.\t7\n")
    write(root / "bad.tsv", "text\tregion\nx\ty\n")
    assert main(["--inputs", str(root), "inputs", "check", "r"]) == 0
    out = capsys.readouterr().out
    assert "r: region-per-row" in out
    assert "sentences: 2" in out
    assert "metadata columns: item" in out
    assert main(["--inputs", str(root), "inputs", "check", "bad"]) == 1
    assert "has both text and region columns" in capsys.readouterr().out


def test_inputs_check_unknown_name_is_an_error(capsys, tmp_path):
    (tmp_path / "inputs").mkdir()
    assert main(["--inputs", str(tmp_path / "inputs"), "inputs", "check", "nope"]) == 1
    assert "no input named 'nope'" in capsys.readouterr().err


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """Run CLI engine commands against the stand-in modules in temporary stores."""
    import fake_modules

    from textures import cli
    from textures.definitions import Definitions

    fake_modules.reset()
    monkeypatch.setattr(
        cli, "_definitions", lambda settings: Definitions.packaged(fake_modules.registry())
    )
    for key in ("inputs", "derivatives", "resources", "items"):
        monkeypatch.setenv(f"TEXTURES_{key.upper()}", str(tmp_path / key))
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
