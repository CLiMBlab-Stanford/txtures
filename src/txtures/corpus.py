"""The in-memory form of an input: documents, their segments, and regions.

Every layout of input file is read into this form. Modules see only this
form, never the file it came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from txtures.identity import digest

LEVELS = ("document", "section", "paragraph", "sentence", "region")
"""The levels of an input's structure, from outermost to innermost. Each nests in the one before."""

SEGMENT_LEVELS = ("section", "paragraph", "sentence")
"""The levels between document and region, which group runs of regions."""

ID_COLUMNS = tuple(f"{level}_id" for level in LEVELS)
"""The ID column of each level, in `LEVELS` order."""

RESERVED_COLUMNS = frozenset({*ID_COLUMNS, "region_index", "text", "char_start", "char_end"})
"""Column names that output tables use for their own purposes."""


@dataclass(frozen=True)
class Segment:
    """A run of regions within one document: a section, paragraph, or sentence.

    `start` and `stop` are region indices: the segment holds
    `document.regions[start:stop]`.
    """

    id: str
    start: int
    stop: int


@dataclass(frozen=True)
class Document:
    """One language-model context and the regions it is divided into.

    `regions` holds character spans `(start, end)` into `text`, in order and
    non-overlapping. Characters between regions are joiner whitespace.

    `sections`, `paragraphs`, and `sentences` are each empty when the input
    does not define that level. Otherwise the level's segments cover every
    region in order, and each segment lies within one segment of every
    level above it.

    `region_ids` holds the input's `region_id` of each region, or is empty
    when the input gives none. `region_metadata` holds the metadata of the
    row each region was read from, one mapping per region, and is empty for
    inputs without metadata columns.
    """

    document_id: str
    text: str
    regions: tuple[tuple[int, int], ...]
    sentences: tuple[Segment, ...] = ()
    paragraphs: tuple[Segment, ...] = ()
    sections: tuple[Segment, ...] = ()
    region_ids: tuple[str, ...] = ()
    region_metadata: tuple[Mapping[str, str], ...] = ()

    def region_text(self, index: int) -> str:
        start, end = self.regions[index]
        return self.text[start:end]

    def segments(self, level: str) -> tuple[Segment, ...]:
        """The segments of `level`, which is one of `SEGMENT_LEVELS`."""
        return {"section": self.sections, "paragraph": self.paragraphs, "sentence": self.sentences}[
            level
        ]

    def segment_ids(self, level: str) -> list[str]:
        """The ID of the `level` segment holding each region, or `[]` if the level is undefined."""
        ids = [""] * len(self.regions)
        for segment in self.segments(level):
            ids[segment.start : segment.stop] = [segment.id] * (segment.stop - segment.start)
        return ids if self.segments(level) else []

    @cached_property
    def digest(self) -> str:
        """Identity of the document's content: text, region spans, and segment spans.

        IDs and metadata are excluded, so renaming a document or editing its
        metadata does not invalidate computed results. Section and paragraph
        spans are included only when defined, so a document without them
        keeps the identity it had before those levels existed.
        """
        content = {
            "text": self.text,
            "regions": [list(span) for span in self.regions],
            "sentences": [[s.start, s.stop] for s in self.sentences],
        }
        for name, segments in (("paragraphs", self.paragraphs), ("sections", self.sections)):
            if segments:
                content[name] = [[s.start, s.stop] for s in segments]
        return digest(content)


@dataclass(frozen=True)
class Corpus:
    """Every document read from one input, with the input's identity.

    `layout` names how the input's rows map onto documents, such as
    `sentence-per-row`. `metadata_columns` lists the input's metadata column
    names in file order. `digest` covers the input file's bytes and its
    normalized sidecar.
    """

    name: str
    layout: str
    documents: tuple[Document, ...]
    metadata_columns: tuple[str, ...]
    digest: str

    @property
    def region_count(self) -> int:
        return sum(len(document.regions) for document in self.documents)

    def segment_count(self, level: str) -> int:
        """The number of segments of `level`, one of `SEGMENT_LEVELS`, across all documents."""
        return sum(len(document.segments(level)) for document in self.documents)

    @property
    def sentence_count(self) -> int:
        return self.segment_count("sentence")

    @property
    def has_region_ids(self) -> bool:
        return any(document.region_ids for document in self.documents)
