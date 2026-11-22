"""Whether each region is part of a named entity.

A region is 1 if any of its words lies inside an entity that the parser's
named-entity recognizer found, and 0 otherwise. Stanza's English models tag
the 18 OntoNotes types. By default only names count; the numeric and
temporal types (`DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`,
`CARDINAL`) do not.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from txtures.corpus import Document
from txtures.modules.base import Item, Module, ModuleConfig, RunContext

NAME_TYPES = (
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
)


class NamedEntityConfig(ModuleConfig):
    """Parameters for the named-entity flag."""

    types: tuple[str, ...] = Field(default=NAME_TYPES, min_length=1)
    """Entity types that count, as the recognizer names them."""


def entity_type(tag: str) -> str | None:
    """The entity type of a BIOES tag such as `B-GPE`, or `None` for `O`."""
    if tag in ("O", "_", ""):
        return None
    return tag.split("-", 1)[1] if "-" in tag else tag


class NamedEntity(Module):
    """Flag the regions that belong to a named entity."""

    name = "named_entity"
    contract_version = 1
    Config = NamedEntityConfig
    requires = ("parse",)
    features = ("named_entity",)
    persist = False

    def run(self, config: NamedEntityConfig, item: Item, context: RunContext) -> np.ndarray:
        parse = item.upstream["parse"]
        flags = np.zeros(len(item.document.regions), dtype=np.int64)
        types = set(config.types)
        for region, tag in zip(parse["region"], parse["ner"], strict=True):
            if entity_type(str(tag)) in types:
                flags[region] = 1
        return flags

    def columns(self, config: NamedEntityConfig, document: Document, result: np.ndarray):
        return {"named_entity": result}
