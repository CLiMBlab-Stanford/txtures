"""Tests of conjunction-headed coordination."""

from test_dependency_locality import FIN, parse

from txtures.coordination import conjunction_heads
from txtures.modules.dependency_locality import arcs
from txtures.sud import analyze_sentence


def heads_of(rows: str):
    """Convert rows of `form upos head deprel [feats]`; return `(form, head form, relation)`."""
    parsed = [line.split() for line in rows.strip().splitlines()]
    forms = [row[0] for row in parsed]
    heads, labels, shared = analyze_sentence(
        forms,
        [form.lower() for form in forms],
        [row[1] for row in parsed],
        [row[4] if len(row) > 4 else "_" for row in parsed],
        [int(row[2]) for row in parsed],
        [row[3] for row in parsed],
    )
    new_heads, new_labels, _ = conjunction_heads([h - 1 for h in heads], labels, shared)
    return [
        (form, forms[head] if head >= 0 else "ROOT", label)
        for form, head, label in zip(forms, new_heads, new_labels, strict=True)
    ]


def test_conjunction_heads_conjuncts_and_shared_dependents():
    result = heads_of(
        f"""
        Police NOUN 2 nsubj
        arrested VERB 0 root {FIN}
        and CCONJ 4 cc
        questioned VERB 2 conj {FIN}
        the DET 6 det
        student NOUN 2 obj
        """
    )
    assert result == [
        ("Police", "and", "subj"),
        ("arrested", "and", "conj:coord"),
        ("and", "ROOT", "root"),
        ("questioned", "and", "conj:coord"),
        ("the", "student", "det"),
        ("student", "and", "comp:obj"),
    ]


def test_conjunct_without_conjunction_attaches_to_the_structure_before_it():
    result = heads_of(
        """
        apples NOUN 0 root
        , PUNCT 3 punct
        pears NOUN 1 conj
        and CCONJ 5 cc
        plums NOUN 1 conj
        """
    )
    assert result == [
        ("apples", "and", "conj:coord"),
        (",", "pears", "punct"),
        ("pears", "apples", "conj:coord"),
        ("and", "ROOT", "root"),
        ("plums", "and", "conj:coord"),
    ]


def test_conjunctions_nest_from_the_left():
    result = heads_of(
        """
        tea NOUN 0 root
        and CCONJ 3 cc
        coffee NOUN 1 conj
        or CCONJ 5 cc
        juice NOUN 1 conj
        """
    )
    assert result == [
        ("tea", "and", "conj:coord"),
        ("and", "or", "conj:coord"),
        ("coffee", "and", "conj:coord"),
        ("or", "ROOT", "root"),
        ("juice", "or", "conj:coord"),
    ]


def test_asyndetic_coordination_keeps_the_first_conjunct_as_head():
    result = heads_of(
        """
        red ADJ 0 root
        , PUNCT 3 punct
        white ADJ 1 conj
        """
    )
    assert result[2] == ("white", "red", "conj:coord")


def test_correlative_attaches_to_the_conjunction():
    result = heads_of(
        f"""
        both CCONJ 2 cc:preconj
        John PROPN 5 nsubj
        and CCONJ 4 cc
        Mary PROPN 2 conj
        left VERB 0 root {FIN}
        """
    )
    assert result[0] == ("both", "and", "cc@preconj")
    assert result[2] == ("and", "left", "subj")
    assert result[1] == ("John", "and", "conj:coord")


def test_private_dependents_stay_with_their_conjunct():
    result = heads_of(
        """
        the DET 3 det
        old ADJ 3 amod
        men NOUN 0 root
        and CCONJ 7 cc
        the DET 7 det
        young ADJ 7 amod
        women NOUN 3 conj
        """
    )
    assert result[0] == ("the", "men", "det")
    assert result[1] == ("old", "men", "mod")


def test_shared_relative_word_is_rewired_from_the_conjunction():
    _, parsed = parse(
        "the reporter who attacked and insulted the senator",
        f"""
        the DET 2 det
        reporter NOUN 0 root
        who PRON 4 nsubj PronType=Rel
        attacked VERB 2 acl:relcl {FIN}
        and CCONJ 6 cc
        insulted VERB 4 conj {FIN}
        the DET 8 det
        senator NOUN 4 obj
        """,
    )
    found = arcs(parsed, "gibson", "conjunction")
    pairs = {(a.head, a.dependent) for a in found.arcs}
    assert (1, 2) in pairs and (1, 4) not in pairs
    assert {(4, 3), (4, 5), (4, 7)} <= pairs
    assert found.conjunctions == {4}
