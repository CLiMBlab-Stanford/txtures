"""The sentences of each document, shared by every module that works on sentences.

When the input defines sentences, they are used as given. Otherwise Stanza's
tokenizer splits the document, and a split that falls inside a region moves
to the end of that region, so every sentence is a run of whole regions.
`parse` and `ngram_surprisal` read the result, so a document is split once
and every module sees the same sentences.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
from pydantic import Field

from txtures.corpus import Document
from txtures.modules.base import Item, Module, ModuleConfig, RunContext, Workload
from txtures.modules.stanza_pipeline import StanzaModels, StanzaPipeline


class SentencesConfig(ModuleConfig):
    """Parameters for splitting documents into sentences with Stanza."""

    language: str = Field(default="en", pattern=r"^[a-z]{2,3}(_[a-z]+)?$")
    """Stanza language code of the input, such as `en`."""

    package: str = Field(default="default", min_length=1)
    """Stanza package, which selects the treebank the tokenizer was trained on.
    `default` is Stanza's choice for the language."""


def snap_sentences(document: Document, ends: Sequence[int]) -> list[tuple[int, int]]:
    """Turn the end offsets of a splitter's sentences into runs of whole regions.

    A sentence ending inside a region is extended to that region's end.
    Returns `(start, stop)` region ranges covering every region in order.
    """
    region_ends = [end for _, end in document.regions]
    stops = set()
    for end in ends:
        region = bisect.bisect_left(region_ends, end)
        stops.add(min(region + 1, len(region_ends)))
    stops.add(len(region_ends))
    spans, start = [], 0
    for stop in sorted(stops):
        if stop > start:
            spans.append((start, stop))
            start = stop
    return spans


def sentence_spans(result: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """The `(start, stop)` region range of each sentence in a `sentences` result."""
    return list(zip(result["start"].tolist(), result["stop"].tolist(), strict=True))


def sentences_result(spans: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
    """The `sentences` result for sentences with these `(start, stop)` region ranges."""
    return {
        "start": np.array([start for start, _ in spans], dtype=np.int64),
        "stop": np.array([stop for _, stop in spans], dtype=np.int64),
    }


class Sentences(Module):
    """Find the sentences of each document: the input's, or Stanza's split."""

    name = "sentences"
    contract_version = 1
    Config = SentencesConfig
    persist = True
    batch_size = 64

    def workload(self, config: SentencesConfig) -> Workload:
        return Workload(slurm=True, ngpus=0)

    def resources(self, config: SentencesConfig):
        return {"stanza": StanzaPipeline({"language": config.language, "package": config.package})}

    def run_batch(
        self, config: SentencesConfig, items: Sequence[Item], context: RunContext
    ) -> list[dict[str, np.ndarray]]:
        results = []
        for item in items:
            document = item.document
            if document.sentences:
                spans = [(s.start, s.stop) for s in document.sentences]
            elif not document.regions:
                spans = []
            else:
                models: StanzaModels = item.resources["stanza"]
                split = models.splitter(document.text)
                ends = [sentence.tokens[-1].end_char for sentence in split.sentences]
                spans = snap_sentences(document, ends)
            results.append(sentences_result(spans))
        return results
