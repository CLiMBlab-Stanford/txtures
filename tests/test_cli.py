import json

import pytest
from conftest import write

import textures
from textures.cli import main


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep CLI tests away from the real home directory and lab storage."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    for name in ("DEFINITIONS", "INPUTS", "DERIVATIVES", "RESOURCES", "ITEMS", "HF_CACHE"):
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
    rows = {row["key"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["items"] == {
        "key": "items",
        "value": str(tmp_path / "items"),
        "source": "command line",
    }
    assert rows["definitions"]["source"] == "environment (TEXTURES_DEFINITIONS)"
    assert rows["hf_cache"]["value"] is None


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
