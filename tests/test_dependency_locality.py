"""Tests of `dependency_locality`."""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from test_parse import document

from txtures.modules.dependency_locality import (
    Coordination,
    DependencyLocalityConfig,
    arcs,
    coordination_weight,
    region_costs,
    span_weight,
    word_costs,
)
from txtures.modules.parse import ParsedWord, align

FIN = "VerbForm=Fin"


def parse(text: str, rows: str):
    """A one-sentence parse of `text` from rows of `form upos head deprel [feats]`."""
    doc = document(text)
    words, position = [], 0
    for token, line in enumerate(rows.strip().splitlines()):
        form, upos, head, deprel, *feats = line.split()
        start = doc.text.index(form, position)
        position = start + len(form)
        words.append(
            ParsedWord(
                start,
                position,
                token,
                form,
                form.lower(),
                upos,
                upos,
                feats[0] if feats else "_",
                int(head),
                deprel,
                "O",
            )
        )
    return doc, align(doc, [(0, len(doc.regions))], [words])


OBJECT_RELATIVE = parse(
    "The reporter who the senator attacked admitted the error .",
    f"""
    The DET 2 det
    reporter NOUN 7 nsubj
    who PRON 6 obj PronType=Rel
    the DET 5 det
    senator NOUN 6 nsubj
    attacked VERB 2 acl:relcl {FIN}
    admitted VERB 0 root {FIN}
    the DET 9 det
    error NOUN 7 obj
    . PUNCT 7 punct
    """,
)

SUBJECT_RELATIVE = parse(
    "The reporter who attacked the senator admitted the error .",
    f"""
    The DET 2 det
    reporter NOUN 7 nsubj
    who PRON 4 nsubj PronType=Rel
    attacked VERB 2 acl:relcl {FIN}
    the DET 6 det
    senator NOUN 4 obj
    admitted VERB 0 root {FIN}
    the DET 9 det
    error NOUN 7 obj
    . PUNCT 7 punct
    """,
)


def costs(parsed, **config):
    return word_costs(parsed[1], DependencyLocalityConfig(**config)).tolist()


@pytest.mark.parametrize("relative_clauses", ["gibson", "sud"])
def test_gibsons_worked_example(relative_clauses):
    # Gibson (2000): object relative 0 1 0 0 1 3 3 0 1, subject relative 0 1 0 1 0 1 3 0 1.
    assert costs(OBJECT_RELATIVE, relative_clauses=relative_clauses) == [
        0, 1, 0, 0, 1, 3, 3, 0, 1, 0,
    ]  # fmt: skip
    assert costs(SUBJECT_RELATIVE, relative_clauses=relative_clauses) == [
        0, 1, 0, 1, 0, 1, 3, 0, 1, 0,
    ]  # fmt: skip


def test_gibson_rewiring_links_the_relative_word_to_its_antecedent():
    _, parsed = OBJECT_RELATIVE
    gibson = {(a.head, a.dependent) for a in arcs(parsed, "gibson").arcs}
    plain = {(a.head, a.dependent) for a in arcs(parsed, "sud").arcs}
    assert plain - gibson == {(1, 5)}
    assert gibson - plain == {(1, 2)}
    assert arcs(parsed, "gibson").gaps == {(5, 2)}
    assert arcs(parsed, "gibson").gaps and not arcs(parsed, "sud").gaps


def test_pied_piped_relative_word_and_clause_without_one():
    _, parsed = parse(
        "the house in which I live",
        f"""
        the DET 2 det
        house NOUN 0 root
        in ADP 4 case
        which PRON 6 obl PronType=Rel
        I PRON 6 nsubj
        live VERB 2 acl:relcl {FIN}
        """,
    )
    found = arcs(parsed, "gibson")
    pairs = {(a.head, a.dependent) for a in found.arcs}
    assert (1, 3) in pairs and (1, 5) not in pairs
    assert not found.gaps
    _, reduced = parse(
        "the senator the reporter attacked",
        f"""
        the DET 2 det
        senator NOUN 0 root
        the DET 4 det
        reporter NOUN 5 nsubj
        attacked VERB 2 acl:relcl {FIN}
        """,
    )
    assert {(a.head, a.dependent) for a in arcs(reduced, "gibson").arcs} == {
        (a.head, a.dependent) for a in arcs(reduced, "sud").arcs
    }


def test_words_distance_counts_intervening_words_without_a_cost_of_its_own():
    assert costs(SUBJECT_RELATIVE, distance="words", relative_clauses="sud") == [
        0, 0, 0, 1, 0, 1, 4, 0, 1, 0,
    ]  # fmt: skip


def test_verbs_variant_weighs_finite_verbs_two_and_needs_referents():
    # admitted: 2 for itself, then attacked (2) and senator (1) lie between it and reporter.
    assert costs(SUBJECT_RELATIVE, verbs=True)[6] == 2 + 2 + 1
    with pytest.raises(ValidationError, match="needs distance: referents"):
        DependencyLocalityConfig(distance="words", verbs=True)


QUICKLY = parse(
    "Quickly , John ran",
    f"""
    Quickly ADV 4 advmod
    , PUNCT 4 punct
    John PROPN 4 nsubj
    ran VERB 0 root {FIN}
    """,
)


def test_modifiers_variant_ignores_arcs_to_preceding_modifiers():
    assert costs(QUICKLY)[3] == 1 + 1
    assert costs(QUICKLY, modifiers=True)[3] == 1


APPLES = parse(
    "I saw apples and pears yesterday",
    f"""
    I PRON 2 nsubj
    saw VERB 0 root {FIN}
    apples NOUN 2 obj
    and CCONJ 5 cc
    pears NOUN 3 conj
    yesterday NOUN 2 obl:tmod
    """,
)


def test_coordination_variant_weighs_a_coordination_as_its_heaviest_conjunct():
    assert costs(APPLES)[5] == 1 + 2
    assert costs(APPLES, coordination=True)[5] == 1 + 1


def test_coordination_weight_and_spans():
    weights = np.array([1.0, 0, 1, 1, 0, 1, 1, 1])
    inner = Coordination(5, (frozenset({5}), frozenset({6})), frozenset({5, 6}))
    outer = Coordination(2, (frozenset({2, 3}), frozenset({4, 5, 6})), frozenset(range(2, 7)))
    structures = [inner, outer]
    # The second conjunct holds the inner coordination, which weighs 1, so it weighs 1.
    assert coordination_weight(outer, weights, structures) == 2
    assert coordination_weight(inner, weights, structures) == 1
    assert span_weight(1, 8, weights, [outer], structures) == 0 + 2 + 1
    # A coordination that starts before the span still counts once, whole.
    assert span_weight(3, 8, weights, [outer], structures) == 2 + 1
    assert span_weight(7, 8, weights, [outer], structures) == 1


def test_regions_sum_their_words_and_log_transforms_the_sum():
    doc, parsed = parse(
        "I don't know",
        f"""
        I PRON 4 nsubj
        do AUX 4 aux {FIN}
        n't PART 4 advmod
        know VERB 0 root VerbForm=Inf
        """,
    )
    config = DependencyLocalityConfig(distance="words")
    words = word_costs(parsed, config)
    regions = region_costs(doc, parsed, config)
    # `don't` holds two words; in SUD `do` heads the clause.
    assert regions.tolist() == [words[0], words[1] + words[2], words[3]]
    logged = region_costs(doc, parsed, DependencyLocalityConfig(distance="words", transform="log"))
    assert logged.tolist() == pytest.approx([math.log1p(v) for v in regions])
