"""Word tokens and sentences for n-gram models.

The corpus that trains an n-gram model and the text it scores must be
tokenized the same way, so both go through this module. Text is normalized
to NFC, curly apostrophes become `'`, and with `lowercase` it is lowercased.
Tokens follow the Penn Treebank's conventions for English: punctuation is
split from words, numbers such as `3.5` and `1,000` stay whole, words keep
internal hyphens and apostrophes, and the clitics `n't`, `'s`, `'re`,
`'ve`, `'ll`, `'d`, and `'m` are split off (*don't* → *do* *n't*). Tokens
never contain whitespace.

Increment `NGRAM_TEXT_VERSION` whenever the same text would tokenize or
split differently; it is part of the identity of every n-gram corpus.
"""

from __future__ import annotations

import re
import unicodedata

NGRAM_TEXT_VERSION = 1

_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|\w+(?:['\-]\w+)*|[^\w\s]")
_CLITIC = re.compile(r"(?i)^(.+?)('s|'re|'ve|'ll|'d|'m)$")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})

ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd",
        "co", "corp", "gen", "gov", "sen", "rep", "rev", "lt", "col", "sgt", "capt", "no",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
    }
)  # fmt: skip
"""Words that a full stop does not end a sentence after, without the stop, lowercase."""

_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")


def normalize(text: str, lowercase: bool = True) -> str:
    """NFC-normalize `text`, straighten apostrophes, and optionally lowercase it."""
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    return text.lower() if lowercase else text


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split normalized `text` into word tokens."""
    tokens = []
    for token in _TOKEN.findall(normalize(text, lowercase)):
        tokens += _split_clitics(token)
    return tokens


def _split_clitics(token: str) -> list[str]:
    """Split trailing clitics off a word, repeatedly (*i'd've* → *i* *'d* *'ve*)."""
    lower = token.lower()
    if len(token) > 3 and lower.endswith("n't"):
        return [*_split_clitics(token[:-3]), token[-3:]]
    match = _CLITIC.match(token)
    if match and lower not in ("'s", "'re", "'ve", "'ll", "'d", "'m"):
        return [*_split_clitics(match.group(1)), match.group(2)]
    return [token]


def sentence_ends(text: str) -> list[int]:
    """Character offsets just after each sentence of `text`, the last one at its end.

    A sentence ends at `.`, `!`, or `?` (with any closing quotes or brackets)
    before whitespace and a capital letter or digit, unless the word before
    a full stop is an abbreviation or a single letter, and at every line
    break.
    """
    ends = []
    for line in re.finditer(r"[^\n]+", text):
        for match in _SENTENCE_END.finditer(line.group()):
            before = line.group()[: match.start()].rsplit(None, 1)
            word = before[-1].lower().strip("\"'([") if before else ""
            if match.group().startswith(".") and (word in ABBREVIATIONS or len(word) == 1):
                continue
            ends.append(line.start() + match.end())
        if line.group().strip():
            ends.append(line.end())
    return sorted(set(ends))


def sentences(text: str) -> list[str]:
    """Split `text` into sentences with `sentence_ends`, without surrounding whitespace."""
    found, start = [], 0
    for end in sentence_ends(text):
        piece = text[start:end].strip()
        if piece:
            found.append(piece)
        start = end
    return found
