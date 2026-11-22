"""Tests of `pos` and `named_entity` on hand-built parses."""

from test_parse import document, word

from txtures.modules.base import Item, RunContext
from txtures.modules.named_entity import NamedEntity, NamedEntityConfig, entity_type
from txtures.modules.parse import align
from txtures.modules.pos import Pos, PosConfig


def parsed_document():
    """ "I don't live in New York." as Stanza would parse it."""
    doc = document("I don't live in New York.")
    words = [
        word(0, 1, 0, "I", 4, "nsubj", "PRON"),
        word(2, 7, 1, "do", 4, "aux", "AUX"),
        word(2, 7, 1, "n't", 4, "advmod", "PART"),
        word(8, 12, 2, "live", 0, "root", "VERB"),
        word(13, 15, 3, "in", 7, "case", "ADP"),
        word(16, 19, 4, "New", 7, "compound", "PROPN", "B-GPE"),
        word(20, 24, 5, "York", 4, "obl", "PROPN", "E-GPE"),
        word(24, 25, 6, ".", 4, "punct", "PUNCT"),
    ]
    return doc, align(doc, [(0, len(doc.regions))], [words])


def run(module, config, doc, parse):
    return module.run(config, Item(doc, {"parse": parse}, {}), RunContext())


def test_region_takes_the_tag_of_its_sud_head():
    doc, parse = parsed_document()
    tags = run(Pos(), PosConfig(), doc, parse)
    # In SUD `do` heads `don't`, and `York` heads `York.` over the full stop.
    assert list(tags) == ["PRON", "AUX", "VERB", "ADP", "PROPN", "PROPN"]


def test_xpos_tagset():
    doc, parse = parsed_document()
    assert run(Pos(), PosConfig(tagset="xpos"), doc, parse)[1] == "AUX"


def test_named_entity_flags_regions_inside_names():
    doc, parse = parsed_document()
    flags = run(NamedEntity(), NamedEntityConfig(), doc, parse)
    assert flags.tolist() == [0, 0, 0, 0, 1, 1]


def test_named_entity_types_filter_tags():
    doc, parse = parsed_document()
    flags = run(NamedEntity(), NamedEntityConfig(types=("PERSON",)), doc, parse)
    assert flags.tolist() == [0] * 6


def test_entity_type_of_tags():
    assert entity_type("O") is None
    assert entity_type("S-PERSON") == "PERSON"
    assert entity_type("I-WORK_OF_ART") == "WORK_OF_ART"
