"""The length of each region in characters.

Every character of the region's text counts, punctuation included. Lengths
are counted in Unicode code points after NFC normalization, so a letter with
a combining accent counts once however the input encodes it.
"""

from __future__ import annotations

import unicodedata

import numpy as np

from txtures.corpus import Document
from txtures.modules.base import Item, Module, ModuleConfig, RunContext


class WordLengthConfig(ModuleConfig):
    """`word_length` has no parameters."""


def region_lengths(document: Document) -> np.ndarray:
    """Return the number of characters in each region of `document`."""
    lengths = [
        len(unicodedata.normalize("NFC", document.region_text(index)))
        for index in range(len(document.regions))
    ]
    return np.array(lengths, dtype=np.int64)


class WordLength(Module):
    """Count the characters of each region."""

    name = "word_length"
    contract_version = 1
    Config = WordLengthConfig
    features = ("word_length",)
    persist = False

    def run(self, config: WordLengthConfig, item: Item, context: RunContext) -> np.ndarray:
        return region_lengths(item.document)

    def columns(self, config: WordLengthConfig, document: Document, result: np.ndarray):
        return {"word_length": result}
