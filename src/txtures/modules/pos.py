"""The part of speech of each region.

A region's tag is the tag of its head word in the SUD tree: the word whose
head lies outside the region, or, when several do, the first that is not
punctuation. In SUD a function word heads its phrase, so a region such as
*don't* gets the tag of *do*.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from txtures.corpus import Document
from txtures.modules.base import Item, Module, ModuleConfig, RunContext
from txtures.modules.syntax import region_heads, sud_tree


class PosConfig(ModuleConfig):
    """Parameters for region part-of-speech tags."""

    tagset: Literal["upos", "xpos"] = "upos"
    """`upos` for Universal POS tags, or `xpos` for the treebank's own tags
    (Penn Treebank tags for English)."""


class Pos(Module):
    """Tag each region with the part of speech of its head word."""

    name = "pos"
    contract_version = 1
    Config = PosConfig
    requires = ("parse",)
    features = ("pos",)
    sweeps = {"tagset": ("tagset",)}
    persist = False

    def run(self, config: PosConfig, item: Item, context: RunContext) -> np.ndarray:
        parse = item.upstream["parse"]
        heads, _ = sud_tree(parse)
        words = region_heads(parse, heads, len(item.document.regions))
        tags = parse[config.tagset]
        return np.array([tags[w] if w >= 0 else None for w in words], dtype=object)

    def columns(self, config: PosConfig, document: Document, result: np.ndarray):
        return {"pos": list(result)}
