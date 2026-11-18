import math

import fake_modules
import numpy as np
import pandas as pd
import pytest
from conftest import write
from fake_modules import CALLS
from test_cli import fake_engine, isolated_settings  # noqa: F401 - fixtures

import txtures as tx
from txtures import api
from txtures.errors import DefinitionError, InputError, TxturesError
from txtures.inputs import InputSettings, read_texts

pytestmark = pytest.mark.filterwarnings("ignore:.*revision is not pinned:UserWarning")


@pytest.fixture
def fake_api(monkeypatch, fake_engine):  # noqa: F811
    monkeypatch.setattr(api, "REGISTRY", fake_modules.registry())
    return fake_engine


# Text given directly


def test_texts_as_strings_mappings_and_region_lists():
    report = read_texts(["The horse raced.", "It fell."])
    assert [d.document_id for d in report.corpus.documents] == ["1", "2"]
    report = read_texts({"a": ["New York", "is", "big."]})
    (document,) = report.corpus.documents
    assert document.text == "New York is big."
    assert [document.region_text(i) for i in range(3)] == ["New York", "is", "big."]
    report = read_texts("One | two. Three | four.", InputSettings(region_delimiter="|"))
    assert len(report.corpus.documents[0].regions) == 3


def test_text_identity_covers_ids_and_division():
    one = read_texts(["a b"]).corpus.digest
    assert read_texts(["a b"]).corpus.digest == one
    assert read_texts({"x": "a b"}).corpus.digest != one
    assert read_texts([["a b"]]).corpus.digest != one


def test_invalid_texts_are_reported():
    report = read_texts({"a": "", "b": ["ok", " "], "c": 3})
    messages = [str(problem) for problem in report.problems]
    assert messages == [
        "document a: document is empty",
        "document b: region 2 is empty or contains a tab or line break",
        "document c: must be a string or a sequence of region strings",
    ]
    assert read_texts([]).corpus is None
    assert "repeats" in str(read_texts({1: "a", "1": "b"}).problems[0])


# compute


def test_compute_returns_the_table_and_reuses_items(fake_api):
    table = tx.compute(
        "fake_feature", ["The horse raced.", "It fell."], config={"fake_scores": {"scale": 2}}
    )
    assert list(table.columns) == [
        "document_id",
        "region_index",
        "text",
        "char_start",
        "char_end",
        "length",
        "double",
    ]
    assert list(table["text"]) == ["The", "horse", "raced.", "It", "fell."]
    assert list(table["length"]) == [6.0, 10.0, 12.0, 4.0, 10.0]
    assert table["region_index"].dtype == "int64"
    assert CALLS["fake_scores"] == 2
    tx.compute("fake_feature", ["It fell."], config={"fake_scores": {"scale": 2}})
    assert CALLS["fake_scores"] == 2
    assert not (fake_api / "derivatives").exists()


def test_compute_sweeps_and_notes(fake_api):
    with pytest.warns(UserWarning, match="revision is not pinned"):
        table = tx.compute("fake_feature", "a bb", config={"fake_scores": {"scale": [1, 3]}})
    assert list(table["length_scale1"]) == [1.0, 2.0]
    assert list(table["length_scale3"]) == [3.0, 6.0]


def test_compute_rejects_invalid_configs(fake_api):
    with pytest.raises(DefinitionError) as error:
        tx.compute("fake_feature", "a", config={"nope": {}, "fake_scores": {"extends": "x"}})
    assert str(error.value) == (
        "config['nope']: unknown module 'nope'\n"
        "config['fake_scores']: extends is not available here"
    )
    with pytest.raises(DefinitionError, match=r"config\['fake_scores'\]"):
        tx.compute("fake_feature", "a", config={"fake_scores": {"scale": "big"}})
    with pytest.raises(DefinitionError, match="not both"):
        tx.compute("fake_feature", "a", config={}, workflow="main")
    with pytest.raises(InputError, match="document is empty"):
        tx.compute("fake_feature", "  ")


def test_compute_uses_the_store_workflow_without_config(fake_api, monkeypatch):
    from txtures.definitions import Definitions

    store = fake_api / "store"
    write(store / "configs" / "fake_scores" / "big.yml", "scale: 5\n")
    write(store / "workflows" / "big.yml", "fake_scores: big\n")
    monkeypatch.setattr(
        api, "load_definitions", lambda s: Definitions.load(store, fake_modules.registry())
    )
    table = tx.compute("fake_feature", "ab", workflow="big")
    assert list(table["length"]) == [10.0]


# run and read_table


def test_run_writes_the_table_compute_would_return(fake_api):
    table = tx.run("doc", modules=["fake_feature"])
    assert len(list((fake_api / "derivatives" / "doc").glob("main-*/doc.tsv"))) == 1
    computed = tx.compute("fake_feature", ["The horse raced.", "It fell."])
    pd.testing.assert_frame_equal(table, computed)


def test_read_table_keeps_text_and_parses_features(tmp_path):
    directory = tmp_path / "table"
    write(
        directory / "x.tsv",
        "document_id\tregion_index\ttext\tchar_start\tchar_end\tnote\ts\tflag\n"
        "007\t1\tNA\t0\t2\tnull\t1.5\tTRUE\n"
        "007\t2\tnull\t3\t7\tNA\tNA\tNA\n"
        "007\t3\tNone\t8\t12\t\t-Inf\tFALSE\n",
    )
    write(
        directory / "x.columns.tsv",
        "column\tmodule\tfeature\tlabels\tvariant\ns\tm\ts\t\tv\nflag\tm\tflag\t\tv\n",
    )
    table = tx.read_table(directory)
    assert list(table["document_id"]) == ["007"] * 3
    assert list(table["text"]) == ["NA", "null", "None"]
    assert list(table["note"]) == ["null", "NA", ""]
    assert table["s"][0] == 1.5 and math.isnan(table["s"][1]) and table["s"][2] == -np.inf
    assert list(table["flag"].astype(object).fillna("missing")) == [True, "missing", False]
    assert tx.read_table(directory / "x.tsv").equals(table)
    with pytest.raises(TxturesError, match="not an output table"):
        tx.read_table(tmp_path)


def test_the_api_leaves_the_process_environment_alone(fake_api, monkeypatch):
    import os

    monkeypatch.setenv("HF_HOME", "/host/hf")
    monkeypatch.setenv("HF_TOKEN", "host-token")
    before = dict(os.environ)
    tx.compute("fake_feature", "a b")
    assert dict(os.environ) == before


def test_default_locations_warn_once(fake_api, monkeypatch):
    for key in ("ITEMS", "RESOURCES"):
        monkeypatch.delenv(f"TXTURES_{key}")
    monkeypatch.delenv("TXTURES_DEFINITIONS")
    monkeypatch.setattr(api, "_warned_defaults", False)
    with pytest.warns(api.DefaultLocationsWarning, match="txtures site use"):
        tx.compute("fake_feature", "a b")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", api.DefaultLocationsWarning)
        tx.compute("fake_feature", "a b")
