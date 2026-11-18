import json
from pathlib import Path

import pytest
from conftest import write

from txtures.errors import InputError, TxturesError
from txtures.inputs import (
    DOCUMENT_PER_ROW,
    PLAIN_TEXT,
    REGION_PER_ROW,
    load_input,
    scan_inputs,
)


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    root.mkdir()
    return root


def tsv(*rows: str) -> str:
    return "".join(f"{row}\n" for row in rows)


def sidecar(root: Path, name: str, **settings) -> None:
    write(root / f"{name}.json", json.dumps(settings))


def regions(document) -> list[str]:
    return [document.region_text(i) for i in range(len(document.regions))]


def corpus(root: Path, name: str):
    report = load_input(root, name)
    assert report.problems == (), report.problems
    return report.corpus


def problems(root: Path, name: str) -> str:
    report = load_input(root, name)
    assert report.corpus is None
    return "\n".join(str(problem) for problem in report.problems)


def warnings(root: Path, name: str) -> str:
    return "\n".join(str(warning) for warning in load_input(root, name).warnings)


# Discovery


def test_scan_finds_inputs_sidecars_and_ignored_files(inputs):
    write(inputs / "a.tsv", tsv("text", "x"))
    write(inputs / "a.json", "{}")
    write(inputs / "b.txt", "x\n")
    write(inputs / "orphan.json", "{}")
    write(inputs / "notes.md", "")
    write(inputs / ".hidden.tsv", "")
    inventory = scan_inputs(inputs)
    assert sorted(inventory.inputs) == ["a", "b"]
    assert inventory.inputs["a"].sidecar == inputs / "a.json"
    assert inventory.inputs["b"].sidecar is None
    assert [path.name for path in inventory.ignored] == ["notes.md", "orphan.json"]
    assert inventory.problems == {}


def test_shared_basename_is_a_problem(inputs):
    write(inputs / "a.tsv", tsv("text", "x"))
    write(inputs / "a.txt", "x\n")
    with pytest.raises(InputError, match="several files: a.tsv, a.txt"):
        load_input(inputs, "a")


def test_invalid_basename_is_a_problem(inputs):
    write(inputs / "bad name.txt", "x\n")
    with pytest.raises(InputError, match="must match"):
        load_input(inputs, "bad name")


def test_missing_input_and_directory(inputs, tmp_path):
    with pytest.raises(TxturesError, match="no input named 'x'"):
        load_input(inputs, "x")
    with pytest.raises(TxturesError, match="does not exist"):
        load_input(tmp_path / "absent", "x")


# Plain text


def test_plain_text_documents_are_lines_and_regions_are_words(inputs):
    write(inputs / "p.txt", "  The horse  raced.\nIt fell.\r\n")
    result = corpus(inputs, "p")
    assert result.layout == PLAIN_TEXT
    first, second = result.documents
    assert (first.doc_id, first.text) == ("1", "The horse  raced.")
    assert regions(first) == ["The", "horse", "raced."]
    assert (second.doc_id, second.text, regions(second)) == ("2", "It fell.", ["It", "fell."])
    assert first.sentences == ()


def test_blank_line_inside_plain_text_is_an_error(inputs):
    write(inputs / "p.txt", "a\n\nb\n")
    assert "p.txt:2: document is empty" in problems(inputs, "p")


def test_delimiters_define_regions_and_sentences(inputs):
    write(inputs / "p.txt", "The cat | sat<s>on the|mat\n")
    sidecar(inputs, "p", region_delimiter="|", sentence_delimiter="<s>")
    (document,) = corpus(inputs, "p").documents
    assert document.text == "The cat sat on the mat"
    assert regions(document) == ["The cat", "sat", "on the", "mat"]
    assert [(s.sentence_id, s.start, s.stop) for s in document.sentences] == [
        ("1", 0, 2),
        ("2", 2, 4),
    ]


def test_joiners_control_the_model_text(inputs):
    write(inputs / "p.txt", "walk|ed<s>home\n")
    sidecar(
        inputs,
        "p",
        region_delimiter="|",
        sentence_delimiter="<s>",
        region_joiner="",
        sentence_joiner="  ",
    )
    (document,) = corpus(inputs, "p").documents
    assert document.text == "walked  home"
    assert regions(document) == ["walk", "ed", "home"]


def test_sentence_delimiter_alone_keeps_whitespace_words(inputs):
    write(inputs / "p.txt", "One  two. // Three.\n")
    sidecar(inputs, "p", sentence_delimiter="//")
    (document,) = corpus(inputs, "p").documents
    assert document.text == "One  two. Three."
    assert regions(document) == ["One", "two.", "Three."]
    assert [(s.start, s.stop) for s in document.sentences] == [(0, 2), (2, 3)]


def test_empty_regions_and_sentences_are_errors(inputs):
    write(inputs / "p.txt", "a||b<s><s>c\n")
    sidecar(inputs, "p", region_delimiter="|", sentence_delimiter="<s>")
    found = problems(inputs, "p")
    assert "sentence 1, region 2 is empty" in found
    assert "sentence 2 is empty" in found


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"region_joiner": "-"},
            "region_joiner: Value error, must be empty or contain only whitespace",
        ),
        ({"region_delimiter": ""}, "region_delimiter: String should have at least 1 character"),
        ({"region_delimiter": "|", "sentence_delimiter": "|"}, "must differ"),
        ({"colour": "blue"}, "colour: Extra inputs are not permitted"),
    ],
)
def test_invalid_sidecars_are_rejected(inputs, settings, message):
    write(inputs / "p.txt", "x\n")
    sidecar(inputs, "p", **settings)
    assert message in problems(inputs, "p")


def test_malformed_sidecar_json_is_rejected(inputs):
    write(inputs / "p.txt", "x\n")
    write(inputs / "p.json", "{not json")
    assert "cannot read sidecar" in problems(inputs, "p")


# Document per row


def test_document_rows_keep_ids_metadata_and_quotes(inputs):
    write(
        inputs / "d.tsv",
        tsv("doc_id\ttext\tcondition", 'a\tShe said "hi".\tx', "b\tThe end.\ty"),
    )
    result = corpus(inputs, "d")
    assert result.layout == DOCUMENT_PER_ROW
    assert result.metadata_columns == ("condition",)
    first, second = result.documents
    assert (first.doc_id, first.text, first.metadata) == ("a", 'She said "hi".', {"condition": "x"})
    assert regions(first) == ["She", "said", '"hi".']
    assert second.doc_id == "b"


def test_document_rows_are_numbered_without_doc_id(inputs):
    write(inputs / "d.tsv", tsv("text", "one", "two"))
    assert [d.doc_id for d in corpus(inputs, "d").documents] == ["1", "2"]


def test_repeated_doc_id_and_bad_rows_are_errors(inputs):
    write(inputs / "d.tsv", tsv("doc_id\ttext", "a\tx", "a\ty", "b\tz\textra", "\tw"))
    found = problems(inputs, "d")
    assert "d.tsv:3: doc_id 'a' repeats line 2" in found
    assert "d.tsv:4: has 3 fields; the header has 2" in found
    assert "d.tsv:5: doc_id is empty" in found


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("text\ttext", "repeats column text"),
        ("text\tregion", "has both text and region columns"),
        ("words", "needs a text column"),
        ("text\tregion_index", "column 'region_index' is reserved"),
        ("text\t", "has an empty column name"),
    ],
)
def test_invalid_headers_are_rejected(inputs, header, message):
    fields = header.count("\t") + 1
    write(inputs / "d.tsv", tsv(header, "\t".join(["x"] * fields)))
    assert message in problems(inputs, "d")


def test_empty_and_header_only_files_are_rejected(inputs):
    write(inputs / "e.tsv", "")
    write(inputs / "h.tsv", tsv("text"))
    assert "is empty; the first line must be a header" in problems(inputs, "e")
    assert "contains no documents" in problems(inputs, "h")


def test_invalid_utf8_is_rejected(inputs):
    (inputs / "d.tsv").write_bytes(b"text\n\xff\xfe\n")
    assert "is not valid UTF-8" in problems(inputs, "d")


def test_byte_order_mark_is_ignored(inputs):
    (inputs / "d.tsv").write_bytes("﻿text\nhello\n".encode())
    assert corpus(inputs, "d").documents[0].text == "hello"


# Region per row


def test_region_rows_group_into_documents_and_sentences(inputs):
    write(
        inputs / "r.tsv",
        tsv(
            "doc_id\tsentence_id\tregion\titem",
            "d1\ts1\tThe horse\t1",
            "d1\ts1\traced past\t1",
            "d1\ts2\tIt fell.\t1",
            "d2\ts1\tHello\t2",
        ),
    )
    result = corpus(inputs, "r")
    assert result.layout == REGION_PER_ROW
    first, second = result.documents
    assert first.text == "The horse raced past It fell."
    assert regions(first) == ["The horse", "raced past", "It fell."]
    assert [(s.sentence_id, s.start, s.stop) for s in first.sentences] == [
        ("s1", 0, 2),
        ("s2", 2, 3),
    ]
    assert first.region_metadata == ({"item": "1"},) * 3
    assert (second.doc_id, second.text) == ("d2", "Hello")


def test_region_rows_without_doc_id_form_one_document(inputs):
    write(inputs / "r.tsv", tsv("region", "a", "b"))
    (document,) = corpus(inputs, "r").documents
    assert (document.doc_id, document.text, document.sentences) == ("1", "a b", ())


def test_region_rows_must_be_contiguous(inputs):
    write(
        inputs / "r.tsv",
        tsv("doc_id\tsentence_id\tregion", "a\t1\tx", "b\t1\ty", "a\t2\tz", "b\t2\tw", "b\t1\tv"),
    )
    found = problems(inputs, "r")
    assert "r.tsv:4: doc_id 'a' resumes after other documents" in found
    assert "r.tsv:6: sentence_id '1' resumes within document 'b'" in found


def test_region_rows_reject_delimiters_and_empty_values(inputs):
    write(inputs / "r.tsv", tsv("sentence_id\tregion", "1\t", "\tx"))
    sidecar(inputs, "r", region_delimiter="|")
    found = problems(inputs, "r")
    assert "region_delimiter does not apply to region-per-row inputs" in found
    assert "r.tsv:2: region is empty" in found
    assert "r.tsv:3: sentence_id is empty" in found


# Warnings and identity


def test_suspicious_content_produces_warnings(inputs):
    write(inputs / "d.tsv", tsv("text", '"quoted by R"', "non breaking", "café"))
    found = warnings(inputs, "d")
    assert "text is wrapped in double quotes" in found
    assert "whitespace other than spaces and tabs" in found
    assert "not in Unicode NFC normal form" in found
    assert load_input(inputs, "d").corpus is not None


def test_input_digest_covers_file_and_sidecar(inputs):
    write(inputs / "p.txt", "a|b\n")
    plain = corpus(inputs, "p").digest
    sidecar(inputs, "p", region_delimiter="|")
    delimited = corpus(inputs, "p").digest
    sidecar(inputs, "p", region_delimiter="|", region_joiner=" ")
    assert delimited != plain
    assert corpus(inputs, "p").digest == delimited


def test_document_digest_ignores_ids_and_metadata(inputs):
    write(inputs / "a.tsv", tsv("doc_id\ttext\tc", "x\tsame text\t1"))
    write(inputs / "b.tsv", tsv("doc_id\ttext\tc", "y\tsame text\t2"))
    write(inputs / "c.tsv", tsv("doc_id\ttext\tc", "x\tother text\t1"))
    a, b, c = (corpus(inputs, name).documents[0] for name in "abc")
    assert a.digest == b.digest
    assert a.digest != c.digest


def test_delimited_region_with_a_tab_is_rejected(inputs):
    write(inputs / "p.txt", "a\tb|c\n")
    sidecar(inputs, "p", region_delimiter="|")
    assert "region 1 contains a tab" in problems(inputs, "p")
