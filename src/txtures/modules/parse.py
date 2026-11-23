"""Universal Dependencies analyses of each document from Stanza.

`parse` tokenizes, tags, lemmatizes, parses, and finds named entities in
each sentence, and stores one record per syntactic word. It has no feature
columns; `pos`, `named_entity`, and `dependency_locality` read its results.

Sentences come from the `sentences` module: the input's when it defines
them, otherwise Stanza's split moved to region ends. Each sentence is parsed
on its own.

Every Stanza token must lie inside one region. Words of a multiword token
(*don't* → *do* *n't*) share the token's character span.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from txtures.corpus import Document
from txtures.errors import TxturesError
from txtures.modules.base import Item, Module, ModuleConfig, RunContext, Workload
from txtures.modules.sentences import sentence_spans
from txtures.modules.stanza_pipeline import StanzaModels, StanzaPipeline

STRING_FIELDS = ("form", "lemma", "upos", "xpos", "feats", "deprel", "ner")
"""Per-word text fields of a stored parse. Missing values are stored as `_`."""


class ParseConfig(ModuleConfig):
    """Parameters for parsing with Stanza."""

    language: str = Field(default="en", pattern=r"^[a-z]{2,3}(_[a-z]+)?$")
    """Stanza language code of the input, such as `en`."""

    package: str = Field(default="default", min_length=1)
    """Stanza package, which selects the treebank the models were trained on.
    `default` is Stanza's choice for the language."""


@dataclass(frozen=True)
class ParsedWord:
    """One syntactic word of a parsed sentence, as the parser reports it.

    `start` and `end` are the character span of the word's token in the
    sentence text. `token` numbers tokens from 0 within the sentence. `head`
    numbers words from 1 within the sentence, with 0 for the root.
    """

    start: int
    end: int
    token: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    ner: str


def sentence_text(document: Document, start: int, stop: int) -> tuple[int, str]:
    """Return the character offset and text of regions `start:stop`.

    Line breaks are replaced by spaces, which keeps character offsets and
    stops the parser from splitting the sentence at them.
    """
    first = document.regions[start][0]
    text = document.text[first : document.regions[stop - 1][1]]
    return first, text.replace("\r", " ").replace("\n", " ")


def align(
    document: Document,
    sentences: Sequence[tuple[int, int]],
    parsed: Sequence[Sequence[ParsedWord]],
) -> dict[str, np.ndarray]:
    """Combine per-sentence parses into one stored result for `document`.

    Returns arrays with one entry per word: `word_start` and `word_end`
    (character span of its token in the document text), `token` (token
    index in the document), `region`, `sentence`, `head` (word index in the
    document, or -1 for a root), and the fields in `STRING_FIELDS`; plus
    `sentence_start` and `sentence_stop`, the region range of each sentence.

    Raises:
        TxturesError: A token spans more than one region, or a head lies
            outside its sentence.
    """
    region_starts = [start for start, _ in document.regions]
    integer_fields = ("word_start", "word_end", "token", "region", "sentence", "head")
    columns: dict[str, list] = {name: [] for name in (*integer_fields, *STRING_FIELDS)}
    tokens = 0
    for index, ((start, stop), words) in enumerate(zip(sentences, parsed, strict=True)):
        offset, _ = sentence_text(document, start, stop)
        first_word = len(columns["head"])
        sentence_tokens = 0
        for word in words:
            begin, end = word.start + offset, word.end + offset
            region = bisect.bisect_right(region_starts, begin) - 1
            if not (start <= region < stop) or end > document.regions[region][1]:
                raise TxturesError(
                    f"document {document.document_id!r}: parser token "
                    f"{document.text[begin:end]!r} spans more than one region"
                )
            if not 0 <= word.head <= len(words):
                raise TxturesError(
                    f"document {document.document_id!r}: parser head {word.head} lies "
                    "outside its sentence"
                )
            columns["word_start"].append(begin)
            columns["word_end"].append(end)
            columns["token"].append(tokens + word.token)
            columns["region"].append(region)
            columns["sentence"].append(index)
            columns["head"].append(first_word + word.head - 1 if word.head else -1)
            for name in STRING_FIELDS:
                columns[name].append(getattr(word, name) or "_")
            sentence_tokens = max(sentence_tokens, word.token + 1)
        tokens += sentence_tokens
    result = {name: np.array(columns[name], dtype=np.int64) for name in integer_fields}
    for name in STRING_FIELDS:
        result[name] = np.array(columns[name], dtype=str)
    result["sentence_start"] = np.array([start for start, _ in sentences], dtype=np.int64)
    result["sentence_stop"] = np.array([stop for _, stop in sentences], dtype=np.int64)
    return result


def stanza_words(sentence) -> list[ParsedWord]:
    """Convert one Stanza sentence into `ParsedWord`s."""
    words = []
    for token_index, token in enumerate(sentence.tokens):
        for word in token.words:
            words.append(
                ParsedWord(
                    start=token.start_char,
                    end=token.end_char,
                    token=token_index,
                    form=word.text,
                    lemma=word.lemma or "_",
                    upos=word.upos or "_",
                    xpos=word.xpos or "_",
                    feats=word.feats or "_",
                    head=int(word.head or 0),
                    deprel=word.deprel or "_",
                    ner=token.ner or "O",
                )
            )
    return words


class Parse(Module):
    """Parse each document with Stanza into Universal Dependencies."""

    name = "parse"
    contract_version = 1
    Config = ParseConfig
    requires = ("sentences",)
    persist = True
    batch_size = 8

    def workload(self, config: ParseConfig) -> Workload:
        return Workload(slurm=True)

    def resources(self, config: ParseConfig):
        return {"stanza": StanzaPipeline({"language": config.language, "package": config.package})}

    def run_batch(
        self, config: ParseConfig, items: Sequence[Item], context: RunContext
    ) -> list[dict[str, np.ndarray]]:
        if not items:
            return []
        models: StanzaModels = items[0].resources["stanza"]
        plans = []
        texts = []
        for item in items:
            document = item.document
            spans = sentence_spans(item.upstream["sentences"])
            plans.append(spans)
            texts.extend(sentence_text(document, start, stop)[1] for start, stop in spans)
        parsed = models.parser.bulk_process(texts) if texts else []
        results, position = [], 0
        for item, spans in zip(items, plans, strict=True):
            sentences = []
            for _ in spans:
                stanza_document = parsed[position]
                position += 1
                words = [w for s in stanza_document.sentences for w in stanza_words(s)]
                if len(stanza_document.sentences) > 1:
                    raise TxturesError(
                        f"document {item.document.document_id!r}: the parser split a "
                        "sentence it was given as one"
                    )
                sentences.append(words)
            results.append(align(item.document, spans, sentences))
        return results
