"""Tests for the `lvpcfg_grammar` resource; the trainer itself is tested in prsr."""

from __future__ import annotations


def test_grammar_resource_builds_saves_and_loads(tmp_path, monkeypatch, capsys):
    import numpy as np
    from prsr import Grammar
    from test_gcg import fake_ptb

    from txtures.cli import main

    for key in ("inputs", "outputs", "resources", "cache", "definitions"):
        monkeypatch.setenv(f"TXTURES_{key.upper()}", str(tmp_path / key))
    monkeypatch.setenv("TXTURES_CORPORA_PTB3", str(fake_ptb(tmp_path / "ptb")))
    monkeypatch.setenv("TXTURES_EXECUTION_EXECUTOR", "local")
    monkeypatch.setenv("TXTURES_EXECUTION_DEVICE", "cpu")
    args = ["resources", "build", "--kind", "lvpcfg_grammar"]
    args += [
        "--spec",
        "treebank={sections: [2]}",
        "--spec",
        "cycles=1",
        "--spec",
        "updates=[2, 1, 1]",
    ]
    assert main(args) == 0
    assert "built resource lvpcfg_grammar" in capsys.readouterr().out
    [entry] = list((tmp_path / "resources" / "store" / "lvpcfg_grammar").glob("*/*"))
    trained = Grammar.load(entry)
    assert trained.training["trees"] == 7  # the FAIL tree is dropped
    assert [p["phase"] for p in trained.training["phases"]] == ["split", "merge", "smooth"]
    rule = next(iter(trained.rules.binary.values()))
    assert np.all(np.isfinite(rule)) and trained.rules.smoother is not None
    assert trained.lexicon.total == 46  # the 49 words less the 3 of the FAIL tree
