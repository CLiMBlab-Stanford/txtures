"""Reading and validating input files.

An input is a file in the inputs directory, addressed by its basename:

- `NAME.txt`: one document per line.
- `NAME.tsv`: a `text` column and optional ID columns `document_id`,
  `section_id`, `paragraph_id`, `sentence_id`, and `region_id`, which nest in
  that order. Each row is a region when a `region_id` column is present or
  when two rows share every ID; otherwise each row is a unit of the innermost
  level with an ID column, or a document when there is none.

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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from txtures.corpus import (
    ID_COLUMNS,
    LEVELS,
    RESERVED_COLUMNS,
    SEGMENT_LEVELS,
    Corpus,
    Document,
    Segment,
)
from txtures.errors import InputError, Problem, TxturesError
from txtures.identity import digest

NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
"""Pattern for input basenames."""

READER_VERSION = 2
"""Version of the reading rules. Increment it when the same file would read differently."""

INPUT_SUFFIXES = (".tsv", ".txt")
SIDECAR_SUFFIX = ".json"

DOCUMENT_PER_ROW = "document-per-row"
SECTION_PER_ROW = "section-per-row"
PARAGRAPH_PER_ROW = "paragraph-per-row"
SENTENCE_PER_ROW = "sentence-per-row"
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

    paragraph_joiner: str = "\n"
    """Whitespace placed between paragraphs in the text the model sees."""

    section_joiner: str = "\n\n"
    """Whitespace placed between sections in the text the model sees."""

    language: str | None = Field(default=None, min_length=1)
    """Language of the input, as an IETF language tag such as `en`."""

    @field_validator("region_joiner", "sentence_joiner", "paragraph_joiner", "section_joiner")
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
        TxturesError: `root` is not a directory.
    """
    if not root.is_dir():
        raise TxturesError(f"inputs directory {root} does not exist")
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
    """Assembles one document's text and spans from regions.

    Two regions are joined with the joiner of the outermost segment boundary
    between them: section, paragraph, or sentence. Regions with no boundary
    between them are joined with the region joiner.
    """

    def __init__(self, settings: InputSettings):
        self.settings = settings
        self.parts: list[str] = []
        self.length = 0
        # The parts that came from the input, excluding joiners.
        self.content: list[str] = []
        self.regions: list[tuple[int, int]] = []
        self.region_ids: list[str] = []
        self.region_metadata: list[Mapping[str, str]] = []
        self.starts: dict[str, list[tuple[str, int]]] = {level: [] for level in SEGMENT_LEVELS}
        # The outermost level of segment opened since the last region, if any.
        self._boundary: str | None = None

    def _append(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def _join(self) -> None:
        if self.regions:
            level = self._boundary or "region"
            self._append(getattr(self.settings, f"{level}_joiner"))
            self.content.append(" ")
        self._boundary = None

    def open(self, level: str, segment_id: str) -> None:
        """Start a segment of `level` at the next region."""
        self.starts[level].append((segment_id, len(self.regions)))
        if self._boundary is None or LEVELS.index(level) < LEVELS.index(self._boundary):
            self._boundary = level

    def add_region(
        self, text: str, region_id: str | None = None, metadata: Mapping[str, str] | None = None
    ) -> None:
        self._join()
        start = self.length
        self._append(text)
        self.content.append(text)
        self.regions.append((start, self.length))
        if region_id is not None:
            self.region_ids.append(region_id)
        if metadata is not None:
            self.region_metadata.append(metadata)

    def add_words(self, text: str, metadata: Mapping[str, str] | None = None) -> None:
        """Add a run of text whose regions are its whitespace-delimited words."""
        self._join()
        offset = self.length
        self._append(text)
        self.content.append(text)
        for match in re.finditer(r"\S+", text):
            self.regions.append((offset + match.start(), offset + match.end()))
            if metadata is not None:
                self.region_metadata.append(metadata)

    def text(self) -> str:
        return "".join(self.parts)

    def input_text(self) -> str:
        """The document's text from the input, with each joiner replaced by a space."""
        return "".join(self.content)

    def build(self, document_id: str) -> Document:
        segments = {}
        for level, starts in self.starts.items():
            stops = [start for _, start in starts[1:]] + [len(self.regions)] * bool(starts)
            segments[level] = tuple(
                Segment(segment_id, start, stop)
                for (segment_id, start), stop in zip(starts, stops, strict=True)
            )
        return Document(
            document_id,
            self.text(),
            tuple(self.regions),
            sentences=segments["sentence"],
            paragraphs=segments["paragraph"],
            sections=segments["section"],
            region_ids=tuple(self.region_ids),
            region_metadata=tuple(self.region_metadata),
        )


def _split(
    raw: str,
    level: str,
    builder: _DocumentBuilder,
    location: str,
    report: _Report,
    metadata: Mapping[str, str] | None = None,
) -> bool:
    """Divide one `level` unit of raw text into sentences and regions using the delimiters.

    The sentence delimiter applies only above sentence level. Returns whether
    the text was valid; problems are added to `report`.
    """
    settings = builder.settings
    text = raw.strip()
    if not text:
        report.error(location, f"{level} is empty")
        return False
    if settings.sentence_delimiter is None or level == "sentence":
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
        if sentence_id is not None:
            builder.open("sentence", sentence_id)
        if settings.region_delimiter is None:
            builder.add_words(chunk, metadata)
            continue
        where = f"sentence {sentence_id}, " if sentence_id else ""
        for number, region in enumerate(chunk.split(settings.region_delimiter), start=1):
            region = region.strip()
            if not region:
                report.error(location, f"{where}region {number} is empty")
                ok = False
            elif "\t" in region:
                report.error(
                    location, f"{where}region {number} contains a tab, which TSV cannot hold"
                )
                ok = False
            else:
                builder.add_region(region, metadata=metadata)
    return ok


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
        builder = _DocumentBuilder(settings)
        if _split(line, "document", builder, location, report):
            _check_text(builder.input_text(), location, report)
            documents.append(builder.build(str(number)))
    return PLAIN_TEXT, documents, ()


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


RENAMED_COLUMNS = {
    "doc_id": "column doc_id is now document_id",
    "region": "column region is now text; rows are read as regions when their IDs repeat "
    "or a region_id column is present",
}
"""Columns of earlier input formats, with the message that explains their replacement."""


def _check_header(path: Path, header: list[str], report: _Report) -> bool:
    ok = True
    if len(set(header)) != len(header):
        repeated = sorted({column for column in header if header.count(column) > 1})
        report.error(f"{path}:1", f"repeats column {', '.join(repeated)}")
        ok = False
    if any(not column for column in header):
        report.error(f"{path}:1", "has an empty column name")
        ok = False
    for column, message in RENAMED_COLUMNS.items():
        if column in header:
            report.error(f"{path}:1", message)
            ok = False
    if "text" not in header and "region" not in header:
        report.error(f"{path}:1", "needs a text column")
        ok = False
    return ok


def _metadata_columns(path: Path, header: list[str], report: _Report) -> tuple[str, ...]:
    used = {"text", *ID_COLUMNS}
    columns = tuple(column for column in header if column not in used)
    for column in columns:
        if column in RESERVED_COLUMNS:
            report.error(f"{path}:1", f"column {column!r} is reserved for output tables")
    return columns


INTEGER_ID = re.compile(r"-?[0-9]+")
"""Pattern for section, paragraph, sentence, and region IDs."""


@dataclass
class _Row:
    """One valid data row of a TSV input, with its IDs checked."""

    number: int
    document_id: str
    ids: dict[str, str]
    """The section, paragraph, sentence, and region IDs the file gives, as written."""
    order: tuple[int, ...]
    """The integer values of `ids`, outermost first, which order rows within a document."""
    text: str
    metadata: Mapping[str, str] | None


def _describe(row: _Row, levels: Sequence[str]) -> str:
    parts = [f"document {row.document_id!r}"]
    parts += [f"{level} {row.ids[level]}" for level in levels if level in row.ids]
    return ", ".join(parts)


def _parse_rows(path, lines, levels, metadata_columns, report) -> list[_Row]:
    """Read each data row's IDs, text, and metadata, dropping rows with invalid IDs.

    Without any ID column, each row is its own document, numbered from 1.
    With ID columns but no `document_id`, the file is one document, `1`.
    """
    rows = []
    for number, row in _rows(path, lines, report):
        location = f"{path}:{number}"
        ids: dict[str, str] = {}
        for level in levels:
            column = f"{level}_id"
            value = row[column].strip()
            if not value:
                report.error(location, f"{column} is empty")
            elif level != "document" and not INTEGER_ID.fullmatch(value):
                report.error(location, f"{column} {value!r} is not an integer")
            else:
                ids[level] = value
        if len(ids) < len(levels):
            continue
        document_id = ids.pop("document", "1") if levels else str(len(rows) + 1)
        metadata = {column: row[column] for column in metadata_columns}
        order = tuple(int(value) for value in ids.values())
        rows.append(_Row(number, document_id, ids, order, row["text"], metadata or None))
    return rows


def _row_level(rows: Sequence[_Row], levels: Sequence[str]) -> str:
    """Infer which level each row of the file holds.

    A `region_id` column, or two rows with the same IDs at every level the
    file gives, means each row is a region. Otherwise each row is a unit of
    the innermost level with an ID column, or a document when there is none.
    """
    if not levels:
        return "document"
    if "region" in levels:
        return "region"
    keys = [(row.document_id, row.order) for row in rows]
    return "region" if len(set(keys)) < len(keys) else levels[-1]


def _check_order(path, rows: Sequence[_Row], levels, has_region_ids, report) -> list[_Row]:
    """Drop region rows whose position cannot be determined, reporting why.

    With `region_id`, every region's full set of IDs must be unique. Without
    it, the file order of a group's rows is their order, so each innermost
    group's rows must be contiguous in the file.
    """
    kept = []
    seen: dict[tuple, int] = {}
    previous: tuple | None = None
    for row in rows:
        key = (row.document_id, row.order)
        if has_region_ids and key in seen:
            report.error(
                f"{path}:{row.number}", f"{_describe(row, levels)} repeats line {seen[key]}"
            )
            continue
        if not has_region_ids and key != previous and key in seen:
            report.error(
                f"{path}:{row.number}",
                f"{_describe(row, levels)} resumes after other rows (it began at line "
                f"{seen[key]}); without region_id, a group's rows must be contiguous",
            )
            continue
        seen.setdefault(key, row.number)
        previous = key
        kept.append(row)
    return kept


def _read_table(path, lines, settings, report):
    header = lines[0].split("\t")
    levels = [level for level in LEVELS if f"{level}_id" in header]
    metadata_columns = _metadata_columns(path, header, report)
    rows = _parse_rows(path, lines, levels, metadata_columns, report)
    row_level = _row_level(rows, levels)
    if row_level == "region":
        for key in ("region_delimiter", "sentence_delimiter"):
            if getattr(settings, key) is not None:
                report.error(str(path), f"{key} does not apply to region rows")
        rows = _check_order(path, rows, levels, "region" in levels, report)
    elif row_level == "sentence" and settings.sentence_delimiter is not None:
        report.error(str(path), "sentence_delimiter does not apply to sentence rows")

    by_document: dict[str, list[_Row]] = {}
    for row in rows:
        by_document.setdefault(row.document_id, []).append(row)
    segment_levels = [level for level in SEGMENT_LEVELS if level in levels]
    documents = []
    for document_id, group in by_document.items():
        builder = _DocumentBuilder(settings)
        ok = True
        previous: tuple[int, ...] | None = None
        for row in sorted(group, key=lambda row: row.order):
            location = f"{path}:{row.number}"
            for depth, level in enumerate(segment_levels, start=1):
                if previous is None or row.order[:depth] != previous[:depth]:
                    builder.open(level, row.ids[level])
            previous = row.order
            if row_level != "region":
                ok = _split(row.text, row_level, builder, location, report, row.metadata) and ok
            elif not row.text.strip():
                report.error(location, "region is empty")
                ok = False
            else:
                builder.add_region(row.text.strip(), row.ids.get("region"), row.metadata)
        if ok:
            where = location if len(group) == 1 else f"{path}: document {document_id}"
            _check_text(builder.input_text(), where, report)
            documents.append(builder.build(document_id))
    return f"{row_level}-per-row", documents, metadata_columns


def read_input(entry: InputFile) -> InputReport:
    """Read and validate one input, collecting every problem and warning.

    Never raises for invalid content; problems are returned in the report.
    """
    report = _Report()
    settings = _read_settings(entry.sidecar, report)
    if settings is None:
        return InputReport(entry.name, None, tuple(report.problems), tuple(report.warnings))

    result: tuple[str, list[Document], tuple[str, ...]] | None = None
    if entry.path.suffix == ".txt":
        result = _read_plain_text(entry.path, settings, report)
    else:
        lines = _read_lines(entry.path, report)
        if lines is not None:
            if not lines:
                report.error(str(entry.path), "is empty; the first line must be a header")
            elif _check_header(entry.path, lines[0].split("\t"), report):
                result = _read_table(entry.path, lines, settings, report)

    if result is not None and not result[1] and not report.problems:
        report.error(str(entry.path), "contains no documents")
    if report.problems or result is None:
        return InputReport(entry.name, None, tuple(report.problems), tuple(report.warnings))

    layout, documents, metadata_columns = result
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
        TxturesError: The inputs directory or the named input does not exist.
        InputError: The name is shared by several files or is not a valid name.
    """
    inventory = scan_inputs(root)
    if name not in inventory.inputs:
        raise TxturesError(f"no input named {name!r} in {root}")
    if name in inventory.problems:
        raise InputError(inventory.problems[name])
    return read_input(inventory.inputs[name])


# Text given directly

MEMORY_LAYOUT = "memory"


def read_texts(
    texts: str | Sequence[str | Sequence[str]] | Mapping[str, str | Sequence[str]],
    settings: InputSettings | None = None,
    name: str = "texts",
) -> InputReport:
    """Read documents given in memory, as `tx.compute` receives them.

    `texts` is one document, a sequence of documents (with IDs `1`, `2`, ...),
    or a mapping from document ID to document. A document is either a string,
    divided by the delimiters in `settings` like a line of a `.txt` input, or
    a sequence of region strings, which are joined with the region joiner.

    Never raises for invalid content; problems are returned in the report.
    """
    settings = settings or InputSettings()
    report = _Report()
    if isinstance(texts, str):
        entries: list[tuple[str, object]] = [("1", texts)]
    elif isinstance(texts, Mapping):
        entries = [(str(document_id), value) for document_id, value in texts.items()]
    else:
        entries = [(str(number), value) for number, value in enumerate(texts, start=1)]
    documents = []
    seen: set[str] = set()
    for document_id, value in entries:
        location = f"document {document_id}"
        if not document_id.strip() or any(char in document_id for char in "\t\n\r"):
            report.error(location, "document IDs must be non-empty and contain no tabs or breaks")
            continue
        if document_id in seen:
            report.error(location, "document ID repeats")
            continue
        seen.add(document_id)
        builder = _DocumentBuilder(settings)
        if isinstance(value, str):
            ok = _split(value, "document", builder, location, report)
        elif isinstance(value, Sequence) and all(isinstance(region, str) for region in value):
            ok = _region_list(value, builder, location, report)
        else:
            report.error(location, "must be a string or a sequence of region strings")
            continue
        if ok:
            _check_text(builder.input_text(), location, report)
            documents.append(builder.build(document_id))
    if not entries:
        report.error(name, "contains no documents")
    if report.problems:
        return InputReport(name, None, tuple(report.problems), tuple(report.warnings))
    corpus = Corpus(
        name=name,
        layout=MEMORY_LAYOUT,
        documents=tuple(documents),
        metadata_columns=(),
        digest=digest(
            {
                "reader": READER_VERSION,
                "documents": [[d.document_id, d.digest] for d in documents],
                "settings": settings.model_dump(mode="json"),
            }
        ),
    )
    return InputReport(name, corpus, (), tuple(report.warnings))


def _region_list(
    regions: Sequence[str], builder: _DocumentBuilder, location: str, report: _Report
) -> bool:
    ok = bool(regions)
    if not regions:
        report.error(location, "document is empty")
    for number, region in enumerate(regions, start=1):
        region = region.strip()
        if not region or any(char in region for char in "\t\n\r"):
            report.error(location, f"region {number} is empty or contains a tab or line break")
            ok = False
            continue
        builder.add_region(region)
    return ok
