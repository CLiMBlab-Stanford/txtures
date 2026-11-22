"""Tests of `word_length`."""

from test_cli import isolated_settings  # noqa: F401 - fixture

import txtures as tx
from txtures.inputs import read_texts
from txtures.modules.word_length import region_lengths


def document(texts):
    (document,) = read_texts(texts).corpus.documents
    return document


def test_counts_every_character_including_punctuation():
    lengths = region_lengths(document("The horse, it fell."))
    assert lengths.tolist() == [3, 6, 2, 5]


def test_counts_code_points_after_nfc_normalization():
    decomposed = "café"
    assert len(decomposed) == 5
    assert region_lengths(document(f"{decomposed} au lait")).tolist() == [4, 2, 4]


def test_regions_given_explicitly_keep_internal_spaces():
    lengths = region_lengths(document({"d": ["New York", "is", "big."]}))
    assert lengths.tolist() == [8, 2, 4]


def test_table_column(isolated_settings):  # noqa: F811
    table = tx.compute("word_length", ["The horse raced.", "It fell."])
    assert table["word_length"].tolist() == [3, 5, 6, 2, 5]
    assert list(table.columns)[-1] == "word_length"
