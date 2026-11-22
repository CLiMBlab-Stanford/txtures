"""Tests of `parse` alignment and sentence handling, without Stanza models."""

import pytest

from txtures.errors import TxturesError
from txtures.inputs import InputSettings, read_texts
from txtures.modules.parse import ParsedWord, align, sentence_text, snap_sentences


def document(texts, settings=None):
    (document,) = read_texts(texts, settings or InputSettings()).corpus.documents
    return document


def word(start, end, token, form, head, deprel="dep", upos="X", ner="O"):
    return ParsedWord(start, end, token, form, form, upos, upos, "_", head, deprel, ner)


def test_sentence_splits_inside_a_region_move_to_its_end():
    doc = document("Hi.there you. Bye now")
    # A parser that ends sentences after "Hi." (inside the first region) and after "you.".
    assert snap_sentences(doc, [3, 13]) == [(0, 1), (1, 2), (2, 4)]
    assert snap_sentences(doc, []) == [(0, 4)]
    assert snap_sentences(doc, [21]) == [(0, 4)]


def test_sentence_text_replaces_line_breaks_and_keeps_offsets():
    doc = document("a b\nc d", InputSettings(sentence_delimiter="<s>"))
    offset, text = sentence_text(doc, 0, len(doc.regions))
    assert offset == 0
    assert text == "a b c d"
    assert len(text) == len(doc.text)


def test_align_maps_words_to_regions_heads_and_sentences():
    doc = document("I don't. It fell.")
    first = [
        word(0, 1, 0, "I", 2, "nsubj"),
        word(2, 7, 1, "do", 2, "aux"),
        word(2, 7, 1, "n't", 2, "advmod"),
        word(7, 8, 2, ".", 2, "punct"),
    ]
    second = [word(0, 2, 0, "It", 2, "nsubj"), word(3, 7, 1, "fell", 0, "root")]
    second.append(word(7, 8, 2, ".", 2, "punct"))
    result = align(doc, [(0, 2), (2, 4)], [first, second])
    assert result["region"].tolist() == [0, 1, 1, 1, 2, 3, 3]
    assert result["head"].tolist() == [1, 1, 1, 1, 5, -1, 5]
    assert result["token"].tolist() == [0, 1, 1, 2, 3, 4, 5]
    assert result["sentence"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert result["word_start"].tolist()[4:] == [9, 12, 16]
    assert result["form"].tolist() == ["I", "do", "n't", ".", "It", "fell", "."]
    assert result["sentence_start"].tolist() == [0, 2]
    assert result["sentence_stop"].tolist() == [2, 4]


def test_token_spanning_two_regions_is_rejected():
    doc = document({"d": ["New", "York"]})
    with pytest.raises(TxturesError, match="spans more than one region"):
        align(doc, [(0, 2)], [[word(0, 8, 0, "New York", 0, "root")]])


def test_head_outside_the_sentence_is_rejected():
    doc = document("a b")
    with pytest.raises(TxturesError, match="outside its sentence"):
        align(doc, [(0, 2)], [[word(0, 1, 0, "a", 3), word(2, 3, 1, "b", 0)]])
