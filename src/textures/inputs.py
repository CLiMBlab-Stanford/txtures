"""Reading and validating input files.

An input is a file in the inputs directory, addressed by its basename:

- `NAME.tsv` with a `text` column: one document per row.
- `NAME.tsv` with a `region` column: one region per row, grouped into
  documents by `doc_id` and into sentences by `sentence_id`.
- `NAME.txt`: one document per line.

An optional `NAME.json` sidecar sets delimiters and joiners. TSV files are
read strictly: tab-separated, UTF-8, first line a header, and no quote
processing, so quote characters in text are kept as written. Fields therefore
cannot contain tabs or line breaks.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from textures.corpus import RESERVED_COLUMNS, Corpus, Document, Sentence
from textures.errors import InputError, Problem, TexturesError
from textures.identity import digest

NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
"""Pattern for input basenames."""

READER_VERSION = 1
"""Version of the reading rules. Increment it when the same file would read differently."""

INPUT_SUFFIXES = (".tsv", ".txt")
SIDECAR_SUFFIX = ".json"

DOCUMENT_PER_ROW = "document-per-row"
REGION_PER_ROW = "region-per-row"
PLAIN_TEXT = "plain-text"


class InputSettings(BaseModel):
    """Corpus-level settings read from an input's JSON sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    region_delimiter: str | None = Field(default=None, min_length=1)
    """Separates regions within a document. When unset, regions are whitespace-delimited words."""

    sentence_delimiter: str | None = Field(default=None, min_length=1)
    """Separates sentences within a document. When unset, the input defines no sentences."""

    region_joiner: str = " "
    """Whitespace placed between regions of a sentence in the text the model sees."""

    sentence_joiner: str = " "
    """Whitespace placed between sentences in the text the model sees."""

    language: str | None = Field(default=None, min_length=1)
    """Language of the input, as an IETF language tag such as `en`."""

    @field_validator("region_joiner", "sentence_joiner")
    @classmethod
    def _joiner_is_whitespace(cls, value: str) -> str:
        if value and not value.isspace():
            raise ValueError("must be empty or contain only whitespace")
        return value


# Discovery


@dataclass(frozen=True)
class InputFile:
    """An input file found in the inputs directory, with its optional sidecar."""

    name: str
    path: Path
    sidecar: Path | None


@dataclass
class Inventory:
    """Every input found in an inputs directory.

    `problems` describes files that cannot be inputs at all, such as two
    inputs sharing a basename. `ignored` lists files that are not inputs or
    sidecars; hidden files are skipped without being listed.
    """

    root: Path
    inputs: dict[str, InputFile] = field(default_factory=dict)
    problems: dict[str, list[Problem]] = field(default_factory=dict)
    ignored: list[Path] = field(default_factory=list)


def scan_inputs(root: Path) -> Inventory:
    """Find the inputs in `root`.

    Raises:
        TexturesError: `root` is not a directory.
    """
    if not root.is_dir():
        raise TexturesError(f"inputs directory {root} does not exist")
    inventory = Inventory(root)
    candidates: dict[str, list[Path]] = {}
    sidecars: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue
        if path.suffix in INPUT_SUFFIXES and path.is_file():
            candidates.setdefault(path.stem, []).append(path)
        elif path.suffix == SIDECAR_SUFFIX and path.is_file():
            sidecars[path.stem] = path
        else:
            inventory.ignored.append(path)
    for name, paths in candidates.items():
        problems = inventory.problems.setdefault(name, [])
        if not NAME_PATTERN.fullmatch(name):
            problems.append(
                Problem(str(paths[0]), f"input name {name!r} must match {NAME_PATTERN.pattern}")
            )
        if len(paths) > 1:
            listed = ", ".join(path.name for path in paths)
            problems.append(Problem(str(root), f"input name {name!r} has several files: {listed}"))
        inventory.inputs[name] = InputFile(name, paths[0], sidecars.pop(name, None))
        if not problems:
            del inventory.problems[name]
    inventory.ignored.extend(sidecars.values())
    inventory.ignored.sort()
    return inventory


# Reading


@dataclass
class _Report:
    """Problems and warnings collected while reading one input."""

    problems: list[Problem] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)

    def error(self, location: str, message: str) -> None:
        self.problems.append(Problem(location, message))

    def warn(self, location: str, message: str) -> None:
        self.warnings.append(Problem(location, message))


@dataclass(frozen=True)
class InputReport:
    """The result of reading one input: the corpus, or the problems that prevented it.

    `corpus` is `None` whenever `problems` is non-empty. Warnings describe
    content that is valid but often unintended.
    """

    name: str
    corpus: Corpus | None
    problems: tuple[Problem, ...]
    warnings: tuple[Problem, ...]


def _read_settings(sidecar: Path | None, report: _Report) -> InputSettings | None:
    if sidecar is None:
        return InputSettings()
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        report.error(str(sidecar), f"cannot read sidecar: {error}")
        return None
    if not isinstance(raw, dict):
        report.error(str(sidecar), "sidecar must be a JSON object")
        return None
    try:
        settings = InputSettings.model_validate(raw)
    except ValidationError as error:
        for detail in error.errors():
            where = ".".join(str(part) for part in detail["loc"])
            report.error(str(sidecar), f"{where}: {detail['msg']}" if where else detail["msg"])
        return None
    delimiters = (settings.region_delimiter, settings.sentence_delimiter)
    if None not in delimiters and delimiters[0] == delimiters[1]:
        report.error(str(sidecar), "region_delimiter and sentence_delimiter must differ")
        return None
    return settings


def _check_text(text: str, location: str, report: _Report) -> None:
    """Warn about characters that commonly change tokenization without being visible."""
    if any(char.isspace() and char not in " \t" for char in text):
        report.warn(location, "contains whitespace other than spaces and tabs")
    if unicodedata.normalize("NFC", text) != text:
        report.warn(location, "is not in Unicode NFC normal form")


class _DocumentBuilder:
    """Assembles one document's text and spans from regions."""

    def __init__(self, settings: InputSettings):
        self.settings = settings
        self.parts: list[str] = []
        self.length = 0
        self.regions: list[tuple[int, int]] = []
        self.sentences: list[Sentence] = []
        self._sentence_start: int | None = None

    def _append(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def start_sentence(self, sentence_id: str | None) -> None:
        if sentence_id is not None:
            self.sentences.append(Sentence(sentence_id, len(self.regions), len(self.regions)))
        self._sentence_start = len(self.regions)

    def end_sentence(self) -> None:
        if self.sentences:
            last = self.sentences[-1]
            self.sentences[-1] = Sentence(last.sentence_id, last.start, len(self.regions))

    def add_region(self, text: str) -> None:
        if self.regions:
            first_in_sentence = self._sentence_start == len(self.regions)
            joiner = (
                self.settings.sentence_joiner
                if first_in_sentence and self.sentences
                else self.settings.region_joiner
            )
            self._append(joiner)
        start = self.length
        self._append(text)
        self.regions.append((start, self.length))

    def add_words(self, text: str) -> None:
        """Add a run of text whose regions are its whitespace-delimited words."""
        if self.regions:
            first_in_sentence = self._sentence_start == len(self.regions)
            self._append(
                self.settings.sentence_joiner
                if first_in_sentence and self.sentences
                else self.settings.region_joiner
            )
        offset = self.length
        self._append(text)
        for match in re.finditer(r"\S+", text):
            self.regions.append((offset + match.start(), offset + match.end()))

    def build(self, doc_id: str, **kwargs) -> Document:
        return Document(
            doc_id, "".join(self.parts), tuple(self.regions), tuple(self.sentences), **kwargs
        )


def _split_document(
    raw: str, settings: InputSettings, location: str, report: _Report
) -> _DocumentBuilder | None:
    """Divide one document's raw text into sentences and regions using the delimiters."""
    builder = _DocumentBuilder(settings)
    text = raw.strip()
    if not text:
        report.error(location, "document is empty")
        return None
    if settings.sentence_delimiter is None:
        chunks: Sequence[tuple[str | None, str]] = [(None, text)]
    else:
        pieces = text.split(settings.sentence_delimiter)
        chunks = [(str(index), piece.strip()) for index, piece in enumerate(pieces, start=1)]
    ok = True
    for sentence_id, chunk in chunks:
        if not chunk:
            report.error(location, f"sentence {sentence_id} is empty")
            ok = False
            continue
        builder.start_sentence(sentence_id)
        if settings.region_delimiter is None:
            builder.add_words(chunk)
        else:
            for number, region in enumerate(chunk.split(settings.region_delimiter), start=1):
                region = region.strip()
                if not region:
                    where = f"sentence {sentence_id}, " if sentence_id else ""
                    report.error(location, f"{where}region {number} is empty")
                    ok = False
                    continue
                if "\t" in region:
                    where = f"sentence {sentence_id}, " if sentence_id else ""
                    report.error(
                        location, f"{where}region {number} contains a tab, which TSV cannot hold"
                    )
                    ok = False
                    continue
                builder.add_region(region)
        builder.end_sentence()
    return builder if ok else None


def _read_lines(path: Path, report: _Report) -> list[str] | None:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        report.error(str(path), f"is not valid UTF-8: {error}")
        return None
    except OSError as error:
        report.error(str(path), f"cannot be read: {error}")
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def _read_plain_text(path: Path, settings: InputSettings, report: _Report):
    lines = _read_lines(path, report)
    if lines is None:
        return None
    documents = []
    for number, line in enumerate(lines, start=1):
        location = f"{path}:{number}"
        builder = _split_document(line, settings, location, report)
        if builder is not None:
            _check_text("".join(builder.parts), location, report)
            documents.append(builder.build(str(number)))
    return documents, ()


def _rows(path: Path, lines: list[str], report: _Report) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield `(line number, row)` for each data line, checking field counts."""
    header = lines[0].split("\t")
    for number, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) != len(header):
            report.error(
                f"{path}:{number}", f"has {len(values)} fields; the header has {len(header)}"
            )
            continue
        for column, value in zip(header, values, strict=True):
            if len(value) > 1 and value.startswith('"') and value.endswith('"'):
                report.warn(
                    f"{path}:{number}",
                    f"{column} is wrapped in double quotes, which are kept as text; "
                    "write TSV files without quoting (in R, quote = FALSE)",
                )
        yield number, dict(zip(header, values, strict=True))


def _check_header(path: Path, header: list[str], report: _Report) -> bool:
    ok = True
    if len(set(header)) != len(header):
        repeated = sorted({column for column in header if header.count(column) > 1})
        report.error(f"{path}:1", f"repeats column {', '.join(repeated)}")
        ok = False
    if any(not column for column in header):
        report.error(f"{path}:1", "has an empty column name")
        ok = False
    if "text" in header and "region" in header:
        report.error(f"{path}:1", "has both text and region columns; use one layout")
        ok = False
    elif "text" not in header and "region" not in header:
        report.error(f"{path}:1", "needs a text column (one document per row) or a region column")
        ok = False
    return ok


def _metadata_columns(path: Path, header: list[str], used: set[str], report: _Report):
    columns = tuple(column for column in header if column not in used)
    for column in columns:
        if column in RESERVED_COLUMNS:
            report.error(f"{path}:1", f"column {column!r} is reserved for output tables")
    return columns


def _read_document_rows(path, lines, settings, report):
    header = lines[0].split("\t")
    metadata_columns = _metadata_columns(path, header, {"text", "doc_id"}, report)
    documents = []
    seen: dict[str, int] = {}
    for number, row in _rows(path, lines, report):
        location = f"{path}:{number}"
        doc_id = row.get("doc_id", str(len(seen) + 1)).strip()
        if not doc_id:
            report.error(location, "doc_id is empty")
            continue
        if doc_id in seen:
            report.error(location, f"doc_id {doc_id!r} repeats line {seen[doc_id]}")
            continue
        seen[doc_id] = number
        builder = _split_document(row["text"], settings, location, report)
        if builder is None:
            continue
        _check_text("".join(builder.parts), location, report)
        metadata = {column: row[column] for column in metadata_columns}
        documents.append(builder.build(doc_id, metadata=metadata))
    return documents, metadata_columns


def _read_region_rows(path, lines, settings, report):
    header = lines[0].split("\t")
    for key in ("region_delimiter", "sentence_delimiter"):
        if getattr(settings, key) is not None:
            report.error(str(path), f"{key} does not apply to region-per-row inputs")
    metadata_columns = _metadata_columns(path, header, {"region", "doc_id", "sentence_id"}, report)
    has_sentences = "sentence_id" in header

    groups: list[tuple[str, list[tuple[int, dict[str, str]]]]] = []
    finished: dict[str, int] = {}
    for number, row in _rows(path, lines, report):
        doc_id = row.get("doc_id", "1").strip()
        if not doc_id:
            report.error(f"{path}:{number}", "doc_id is empty")
            continue
        if groups and groups[-1][0] == doc_id:
            groups[-1][1].append((number, row))
            continue
        if doc_id in finished:
            report.error(
                f"{path}:{number}",
                f"doc_id {doc_id!r} resumes after other documents; its rows must be contiguous "
                f"(it ended at line {finished[doc_id]})",
            )
            continue
        if groups:
            finished[groups[-1][0]] = groups[-1][1][-1][0]
        groups.append((doc_id, [(number, row)]))

    documents = []
    for doc_id, rows in groups:
        builder = _DocumentBuilder(settings)
        region_metadata = []
        current: str | None = None
        closed: set[str] = set()
        ok = True
        for number, row in rows:
            location = f"{path}:{number}"
            region = row["region"].strip()
            if not region:
                report.error(location, "region is empty")
                ok = False
                continue
            if has_sentences:
                sentence_id = row["sentence_id"].strip()
                if not sentence_id:
                    report.error(location, "sentence_id is empty")
                    ok = False
                    continue
                if sentence_id != current:
                    if sentence_id in closed:
                        report.error(
                            location,
                            f"sentence_id {sentence_id!r} resumes within document {doc_id!r}; "
                            "its rows must be contiguous",
                        )
                        ok = False
                        continue
                    if current is not None:
                        builder.end_sentence()
                        closed.add(current)
                    builder.start_sentence(sentence_id)
                    current = sentence_id
            elif current is None:
                builder.start_sentence(None)
                current = ""
            builder.add_region(region)
            region_metadata.append({column: row[column] for column in metadata_columns})
        if has_sentences and current is not None:
            builder.end_sentence()
        if ok and builder.regions:
            _check_text("".join(builder.parts), f"{path}: document {doc_id}", report)
            documents.append(builder.build(doc_id, region_metadata=tuple(region_metadata)))
    return documents, metadata_columns


def read_input(entry: InputFile) -> InputReport:
    """Read and validate one input, collecting every problem and warning.

    Never raises for invalid content; problems are returned in the report.
    """
    report = _Report()
    settings = _read_settings(entry.sidecar, report)
    if settings is None:
        return InputReport(entry.name, None, tuple(report.problems), tuple(report.warnings))

    if entry.path.suffix == ".txt":
        layout = PLAIN_TEXT
        result = _read_plain_text(entry.path, settings, report)
    else:
        lines = _read_lines(entry.path, report)
        result = None
        layout = DOCUMENT_PER_ROW
        if lines is not None:
            if not lines:
                report.error(str(entry.path), "is empty; the first line must be a header")
            else:
                header = lines[0].split("\t")
                if _check_header(entry.path, header, report):
                    if "region" in header:
                        layout = REGION_PER_ROW
                        result = _read_region_rows(entry.path, lines, settings, report)
                    else:
                        result = _read_document_rows(entry.path, lines, settings, report)

    if result is not None and not result[0] and not report.problems:
        report.error(str(entry.path), "contains no documents")
    if report.problems or result is None:
        return InputReport(entry.name, None, tuple(report.problems), tuple(report.warnings))

    documents, metadata_columns = result
    file_digest = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    corpus = Corpus(
        name=entry.name,
        layout=layout,
        documents=tuple(documents),
        metadata_columns=tuple(metadata_columns),
        digest=digest(
            {
                "reader": READER_VERSION,
                "file": file_digest,
                "settings": settings.model_dump(mode="json"),
            }
        ),
    )
    return InputReport(entry.name, corpus, (), tuple(report.warnings))


def load_input(root: Path, name: str) -> InputReport:
    """Find and read the input `name` in the inputs directory `root`.

    Raises:
        TexturesError: The inputs directory or the named input does not exist.
        InputError: The name is shared by several files or is not a valid name.
    """
    inventory = scan_inputs(root)
    if name not in inventory.inputs:
        raise TexturesError(f"no input named {name!r} in {root}")
    if name in inventory.problems:
        raise InputError(inventory.problems[name])
    return read_input(inventory.inputs[name])
