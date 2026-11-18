import json
from pathlib import Path

import pytest
from conftest import write

from txtures.errors import InputError, TxturesError
from txtures.identity import digest
from txtures.inputs import (
    DOCUMENT_PER_ROW,
    PARAGRAPH_PER_ROW,
    PLAIN_TEXT,
    REGION_PER_ROW,
    SECTION_PER_ROW,
    SENTENCE_PER_ROW,
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
    assert (first.document_id, first.text) == ("1", "The horse  raced.")
    assert regions(first) == ["The", "horse", "raced."]
    assert (second.document_id, second.text, regions(second)) == ("2", "It fell.", ["It", "fell."])
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
    assert [(s.id, s.start, s.stop) for s in document.sentences] == [
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


# TSV headers and rows


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("text\ttext", "repeats column text"),
        ("words", "needs a text column"),
        ("text\tregion_index", "column 'region_index' is reserved"),
        ("text\t", "has an empty column name"),
        ("doc_id\ttext", "column doc_id is now document_id"),
        ("region", "column region is now text"),
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


def test_bad_rows_and_ids_are_errors(inputs):
    write(
        inputs / "d.tsv",
        tsv("document_id\tsentence_id\ttext", "a\t1\tx", "b\t1\tz\textra", "\t1\tw", "a\ts2\ty"),
    )
    found = problems(inputs, "d")
    assert "d.tsv:3: has 4 fields; the header has 3" in found
    assert "d.tsv:4: document_id is empty" in found
    assert "d.tsv:5: sentence_id 's2' is not an integer" in found


# Document per row


def test_document_rows_keep_ids_metadata_and_quotes(inputs):
    write(
        inputs / "d.tsv",
        tsv("document_id\ttext\tcondition", 'a\tShe said "hi".\tx', "b\tThe end.\ty"),
    )
    result = corpus(inputs, "d")
    assert result.layout == DOCUMENT_PER_ROW
    assert result.metadata_columns == ("condition",)
    first, second = result.documents
    assert (first.document_id, first.text) == ("a", 'She said "hi".')
    assert regions(first) == ["She", "said", '"hi".']
    assert first.region_metadata == ({"condition": "x"},) * 3
    assert second.document_id == "b"


def test_document_rows_are_numbered_without_any_id_column(inputs):
    write(inputs / "d.tsv", tsv("text", "one", "two"))
    result = corpus(inputs, "d")
    assert result.layout == DOCUMENT_PER_ROW
    assert [d.document_id for d in result.documents] == ["1", "2"]
    assert result.documents[0].region_metadata == ()


# Sentence, paragraph, and section per row


def test_sentence_rows_are_split_into_words_and_sorted_by_id(inputs):
    write(
        inputs / "s.tsv",
        tsv(
            "document_id\tsentence_id\ttext\tcond",
            "a\t2\tIt fell.\tx",
            "b\t1\tHello there.\ty",
            "a\t10\tThe end.\tz",
            "a\t1\tThe horse raced.\tw",
        ),
    )
    result = corpus(inputs, "s")
    assert result.layout == SENTENCE_PER_ROW
    first, second = result.documents
    assert first.document_id == "a"
    assert first.text == "The horse raced. It fell. The end."
    assert [(s.id, s.start, s.stop) for s in first.sentences] == [
        ("1", 0, 3),
        ("2", 3, 5),
        ("10", 5, 7),
    ]
    assert [m["cond"] for m in first.region_metadata] == ["w"] * 3 + ["x"] * 2 + ["z"] * 2
    assert (second.document_id, second.text) == ("b", "Hello there.")


def test_sentence_rows_take_the_region_delimiter_but_not_the_sentence_delimiter(inputs):
    write(inputs / "s.tsv", tsv("sentence_id\ttext", "1\tThe horse | raced", "2\tfell."))
    sidecar(inputs, "s", region_delimiter="|")
    (document,) = corpus(inputs, "s").documents
    assert regions(document) == ["The horse", "raced", "fell."]
    sidecar(inputs, "s", sentence_delimiter="|")
    assert "sentence_delimiter does not apply to sentence rows" in problems(inputs, "s")


def test_paragraph_rows_are_split_into_sentences_numbered_within_each(inputs):
    write(
        inputs / "p.tsv",
        tsv(
            "section_id\tparagraph_id\ttext",
            "1\t1\tOne. // Two.",
            "1\t2\tThree.",
            "2\t1\tFour. // Five.",
        ),
    )
    sidecar(inputs, "p", sentence_delimiter="//")
    result = corpus(inputs, "p")
    assert result.layout == PARAGRAPH_PER_ROW
    (document,) = result.documents
    assert document.document_id == "1"
    assert document.text == "One. Two.\nThree.\n\nFour. Five."
    assert [(s.id, s.start, s.stop) for s in document.sections] == [("1", 0, 3), ("2", 3, 5)]
    assert [(s.id, s.start, s.stop) for s in document.paragraphs] == [
        ("1", 0, 2),
        ("2", 2, 3),
        ("1", 3, 5),
    ]
    assert [s.id for s in document.sentences] == ["1", "2", "1", "1", "2"]


def test_the_outermost_boundary_between_regions_chooses_the_joiner(inputs):
    write(
        inputs / "j.tsv",
        tsv(
            "section_id\tparagraph_id\tsentence_id\ttext",
            "1\t1\t1\ta b",
            "1\t1\t2\tc",
            "1\t2\t1\td",
            "2\t1\t1\te",
        ),
    )
    joiners = {"sentence_joiner": " ", "paragraph_joiner": "\n"}
    sidecar(inputs, "j", **joiners, section_joiner="\n\n")
    (document,) = corpus(inputs, "j").documents
    assert document.text == "a b c\nd\n\ne"
    assert regions(document) == ["a", "b", "c", "d", "e"]
    assert warnings(inputs, "j") == ""
    sidecar(inputs, "j", section_joiner="-")
    assert "section_joiner: Value error, must be empty or contain only whitespace" in problems(
        inputs, "j"
    )


def test_section_rows(inputs):
    write(inputs / "s.tsv", tsv("document_id\tsection_id\ttext", "a\t1\tx y", "a\t2\tz"))
    result = corpus(inputs, "s")
    assert result.layout == SECTION_PER_ROW
    (document,) = result.documents
    assert [(s.start, s.stop) for s in document.sections] == [(0, 2), (2, 3)]
    assert document.sentences == document.paragraphs == ()


# Region per row


def test_repeated_ids_make_rows_regions_in_file_order(inputs):
    write(
        inputs / "r.tsv",
        tsv(
            "document_id\tsentence_id\ttext\titem",
            "d1\t2\tIt fell.\t1",
            "d1\t1\tThe horse\t1",
            "d1\t1\traced past\t1",
            "d2\t1\tHello\t2",
        ),
    )
    result = corpus(inputs, "r")
    assert result.layout == REGION_PER_ROW
    first, second = result.documents
    assert first.text == "The horse raced past It fell."
    assert regions(first) == ["The horse", "raced past", "It fell."]
    assert [(s.id, s.start, s.stop) for s in first.sentences] == [("1", 0, 2), ("2", 2, 3)]
    assert first.region_ids == ()
    assert first.region_metadata == ({"item": "1"},) * 3
    assert (second.document_id, second.text) == ("d2", "Hello")


def test_words_per_row_with_only_document_ids(inputs):
    write(inputs / "w.tsv", tsv("document_id\ttext", "a\tThe", "a\tcat", "b\tHi"))
    result = corpus(inputs, "w")
    assert result.layout == REGION_PER_ROW
    assert [(d.document_id, d.text, d.sentences) for d in result.documents] == [
        ("a", "The cat", ()),
        ("b", "Hi", ()),
    ]


def test_region_ids_sort_regions_and_need_no_repeats(inputs):
    write(
        inputs / "r.tsv",
        tsv(
            "document_id\tsentence_id\tregion_id\ttext",
            "a\t2\t1\tfell.",
            "a\t1\t10\tbarn",
            "a\t1\t2\traced",
            "a\t1\t1\tThe horse",
        ),
    )
    result = corpus(inputs, "r")
    assert result.layout == REGION_PER_ROW
    (document,) = result.documents
    assert regions(document) == ["The horse", "raced", "barn", "fell."]
    assert document.region_ids == ("1", "2", "10", "1")
    assert [(s.id, s.start, s.stop) for s in document.sentences] == [("1", 0, 3), ("2", 3, 4)]


def test_ids_are_nested_within_their_parents(inputs):
    write(
        inputs / "r.tsv",
        tsv(
            "document_id\tparagraph_id\tsentence_id\ttext",
            "a\t1\t1\tx",
            "a\t1\t1\ty",
            "a\t2\t1\tz",
            "b\t1\t1\tw",
        ),
    )
    first, second = corpus(inputs, "r").documents
    assert [(s.id, s.start, s.stop) for s in first.sentences] == [("1", 0, 2), ("1", 2, 3)]
    assert [(p.id, p.start, p.stop) for p in first.paragraphs] == [("1", 0, 2), ("2", 2, 3)]
    assert second.text == "w"


def test_region_rows_without_document_id_form_one_document(inputs):
    write(inputs / "r.tsv", tsv("region_id\ttext", "2\tb", "1\ta"))
    (document,) = corpus(inputs, "r").documents
    assert (document.document_id, document.text, document.sentences) == ("1", "a b", ())


def test_region_rows_without_region_id_must_be_contiguous(inputs):
    write(
        inputs / "r.tsv",
        tsv("document_id\tsentence_id\ttext", "a\t1\tx", "b\t1\ty", "a\t1\tz", "a\t2\tw"),
    )
    assert "r.tsv:4: document 'a', sentence 1 resumes after other rows (it began at line 2)" in (
        problems(inputs, "r")
    )


def test_non_contiguous_rows_are_fine_with_region_ids_but_repeats_are_not(inputs):
    write(
        inputs / "r.tsv",
        tsv("document_id\tregion_id\ttext", "a\t2\ty", "b\t1\tz", "a\t1\tx"),
    )
    assert [d.text for d in corpus(inputs, "r").documents] == ["x y", "z"]
    write(inputs / "r.tsv", tsv("document_id\tregion_id\ttext", "a\t1\tx", "a\t01\ty"))
    assert "r.tsv:3: document 'a', region 01 repeats line 2" in problems(inputs, "r")


def test_region_rows_reject_delimiters_and_empty_regions(inputs):
    write(inputs / "r.tsv", tsv("sentence_id\ttext", "1\t", "1\tx"))
    sidecar(inputs, "r", region_delimiter="|")
    found = problems(inputs, "r")
    assert "region_delimiter does not apply to region rows" in found
    assert "r.tsv:2: region is empty" in found


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
    write(inputs / "a.tsv", tsv("document_id\ttext\tc", "x\tsame text\t1"))
    write(inputs / "b.tsv", tsv("document_id\ttext\tc", "y\tsame text\t2"))
    write(inputs / "c.tsv", tsv("document_id\ttext\tc", "x\tother text\t1"))
    a, b, c = (corpus(inputs, name).documents[0] for name in "abc")
    assert a.digest == b.digest
    assert a.digest != c.digest


def test_delimited_region_with_a_tab_is_rejected(inputs):
    write(inputs / "p.txt", "a\tb|c\n")
    sidecar(inputs, "p", region_delimiter="|")
    assert "region 1 contains a tab" in problems(inputs, "p")


def test_document_digest_includes_only_the_levels_a_document_defines(inputs):
    write(inputs / "a.tsv", tsv("sentence_id\ttext", "1\tx y", "2\tz"))
    write(inputs / "b.tsv", tsv("paragraph_id\tsentence_id\ttext", "1\t1\tx y", "1\t2\tz"))
    a, b = (corpus(inputs, name).documents[0] for name in "ab")
    assert a.digest == digest(
        {"text": "x y z", "regions": [[0, 1], [2, 3], [4, 5]], "sentences": [[0, 2], [2, 3]]}
    )
    assert b.digest != a.digest
