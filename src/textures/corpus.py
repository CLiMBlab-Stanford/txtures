"""The in-memory form of an input: documents, sentences, and regions.

Every layout of input file is read into this form. Modules see only this
form, never the file it came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

from textures.identity import digest

RESERVED_COLUMNS = frozenset(
    {"doc_id", "sentence_id", "region_index", "region", "char_start", "char_end", "text"}
)
"""Column names that output tables use for their own purposes."""


@dataclass(frozen=True)
class Sentence:
    """A run of regions within one document.

    `start` and `stop` are region indices: the sentence holds
    `document.regions[start:stop]`.
    """

    sentence_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class Document:
    """One language-model context and the regions it is divided into.

    `regions` holds character spans `(start, end)` into `text`, in order and
    non-overlapping. Characters between regions are joiner whitespace.
    `sentences` is empty when the input defines no sentences; otherwise the
    sentences cover every region in order.

    `region_metadata` holds one mapping per region for region-per-row inputs
    and is empty otherwise. `metadata` holds document-level values for
    document-per-row inputs.
    """

    doc_id: str
    text: str
    regions: tuple[tuple[int, int], ...]
    sentences: tuple[Sentence, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    region_metadata: tuple[Mapping[str, str], ...] = ()

    def region_text(self, index: int) -> str:
        start, end = self.regions[index]
        return self.text[start:end]

    @cached_property
    def digest(self) -> str:
        """Identity of the document's content: text, region spans, and sentence spans.

        `doc_id` and metadata are excluded, so renaming a document or editing
        its metadata does not invalidate computed results.
        """
        return digest(
            {
                "text": self.text,
                "regions": [list(span) for span in self.regions],
                "sentences": [[s.start, s.stop] for s in self.sentences],
            }
        )


@dataclass(frozen=True)
class Corpus:
    """Every document read from one input, with the input's identity.

    `metadata_columns` lists the input's metadata column names in file order.
    `digest` covers the input file's bytes and its normalized sidecar.
    """

    name: str
    layout: str
    documents: tuple[Document, ...]
    metadata_columns: tuple[str, ...]
    digest: str

    @property
    def region_count(self) -> int:
        return sum(len(document.regions) for document in self.documents)

    @property
    def sentence_count(self) -> int:
        return sum(len(document.sentences) for document in self.documents)
