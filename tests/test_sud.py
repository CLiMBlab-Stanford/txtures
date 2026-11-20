"""Tests of the UD to SUD conversion."""

import io
import os
import tarfile
import urllib.request
from pathlib import Path

import pytest

from txtures.sud import convert_sentence

UD_EWT = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.19/en_ewt-ud-{}.conllu"
SUD_EWT = "https://grew.fr/download/SUD_2.19/SUD_English-EWT.tgz"


def convert(rows: str):
    """Convert rows of `form upos head deprel [feats]`; return `(form, head form, relation)`."""
    parsed = [line.split() for line in rows.strip().splitlines()]
    forms = [row[0] for row in parsed]
    feats = [row[4] if len(row) > 4 else "_" for row in parsed]
    heads, labels = convert_sentence(
        forms,
        [form.lower() for form in forms],
        [row[1] for row in parsed],
        feats,
        [int(row[2]) for row in parsed],
        [row[3] for row in parsed],
    )
    return [
        (form, forms[head - 1] if head else "ROOT", label)
        for form, head, label in zip(forms, heads, labels, strict=True)
    ]


def test_auxiliary_heads_the_verb_and_takes_its_subject():
    assert convert(
        """
        John PROPN 3 nsubj
        has AUX 3 aux VerbForm=Fin
        eaten VERB 0 root VerbForm=Part
        """
    ) == [("John", "has", "subj"), ("has", "ROOT", "root"), ("eaten", "has", "comp:aux")]


def test_auxiliaries_chain_from_the_outermost():
    result = convert(
        """
        it PRON 4 nsubj:pass
        has AUX 4 aux VerbForm=Fin
        been AUX 4 aux:pass VerbForm=Part
        eaten VERB 0 root VerbForm=Part
        """
    )
    assert result == [
        ("it", "has", "subj@pass"),
        ("has", "ROOT", "root"),
        ("been", "has", "comp:aux"),
        ("eaten", "been", "comp:aux@pass"),
    ]


def test_adposition_heads_its_noun():
    assert convert(
        """
        sat VERB 0 root VerbForm=Fin
        in ADP 4 case
        the DET 4 det
        house NOUN 1 obl
        """
    ) == [
        ("sat", "ROOT", "root"),
        ("in", "sat", "udep"),
        ("the", "house", "det"),
        ("house", "in", "comp"),
    ]


def test_copula_heads_the_predicate_and_takes_the_subject():
    assert convert(
        """
        John PROPN 3 nsubj
        is AUX 3 cop VerbForm=Fin
        happy ADJ 0 root
        """
    ) == [("John", "is", "subj"), ("is", "ROOT", "root"), ("happy", "is", "comp:pred")]


def test_complementizer_heads_its_clause():
    result = convert(
        """
        I PRON 2 nsubj
        think VERB 0 root VerbForm=Fin
        that SCONJ 5 mark
        John PROPN 5 nsubj
        left VERB 2 ccomp VerbForm=Fin
        """
    )
    assert result[2:] == [
        ("that", "think", "comp:obj"),
        ("John", "left", "subj"),
        ("left", "that", "comp"),
    ]


def test_conjuncts_form_a_chain():
    result = convert(
        """
        apples NOUN 0 root
        , PUNCT 3 punct
        pears NOUN 1 conj
        and CCONJ 5 cc
        plums NOUN 1 conj
        """
    )
    assert result[2] == ("pears", "apples", "conj:coord")
    assert result[4] == ("plums", "pears", "conj:coord")


def test_shared_right_dependent_moves_to_the_nearest_conjunct():
    result = convert(
        """
        I PRON 2 nsubj
        bought VERB 0 root VerbForm=Fin
        and CCONJ 4 cc
        ate VERB 2 conj VerbForm=Fin
        apples NOUN 2 obj
        """
    )
    assert result[4] == ("apples", "ate", "comp:obj")


def test_relative_clause_is_a_modifier():
    result = convert(
        """
        the DET 2 det
        reporter NOUN 0 root
        who PRON 6 obj PronType=Rel
        the DET 5 det
        senator NOUN 6 nsubj
        attacked VERB 2 acl:relcl VerbForm=Fin
        """
    )
    assert result[5] == ("attacked", "reporter", "mod@relcl")
    assert result[2] == ("who", "attacked", "comp:obj")


def test_coordinated_infinitives_attach_to_the_auxiliary_as_grew_does():
    # Grew never treats two VerbForm=Inf values as equal, so `write` is raised to `can`.
    result = convert(
        """
        you PRON 3 nsubj
        can AUX 3 aux VerbForm=Fin
        tour VERB 0 root VerbForm=Inf
        and CCONJ 5 cc
        write VERB 3 conj VerbForm=Inf
        """
    )
    assert result[4] == ("write", "can", "conj:coord")
    finite = convert(
        """
        you PRON 3 nsubj
        have AUX 3 aux VerbForm=Fin
        toured VERB 0 root VerbForm=Part
        and CCONJ 5 cc
        written VERB 3 conj VerbForm=Part
        """
    )
    assert finite[4] == ("written", "toured", "conj:coord")


def test_fixed_expression_keeps_its_internal_relations():
    result = convert(
        """
        left VERB 0 root VerbForm=Fin
        because SCONJ 5 case
        of ADP 2 fixed
        the DET 5 det
        rain NOUN 1 obl
        """
    )
    assert result[1] == ("because", "left", "udep")
    assert result[2] == ("of", "because", "unk")
    assert result[4] == ("rain", "because", "comp")


# Comparison with the official conversion


def _read_conllu(text: str) -> dict[str, list[list[str]]]:
    sentences, sentence_id, rows = {}, None, []
    for line in text.splitlines():
        if line.startswith("# sent_id"):
            sentence_id = line.split("=", 1)[1].strip()
        elif not line:
            if rows:
                sentences[sentence_id] = rows
            sentence_id, rows = None, []
        elif not line.startswith("#"):
            fields = line.split("\t")
            if "-" not in fields[0] and "." not in fields[0]:
                rows.append(fields)
    if rows:
        sentences[sentence_id] = rows
    return sentences


def _fetch(url: str, cache: Path) -> bytes:
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


@pytest.mark.integration
def test_matches_the_official_conversion_of_english_ewt(tmp_path_factory):
    cache = Path(os.environ.get("TXTURES_TEST_DATA", tmp_path_factory.mktemp("ewt")))
    cache.mkdir(parents=True, exist_ok=True)
    gold = {}
    with tarfile.open(fileobj=io.BytesIO(_fetch(SUD_EWT, cache))) as archive:
        for member in archive.getmembers():
            if member.name.endswith(".conllu"):
                gold.update(_read_conllu(archive.extractfile(member).read().decode("utf-8")))
    words = same = 0
    for split in ("train", "dev", "test"):
        ud = _read_conllu(_fetch(UD_EWT.format(split), cache).decode("utf-8"))
        for sentence_id, rows in ud.items():
            expected = gold[sentence_id]
            assert [r[1] for r in rows] == [r[1] for r in expected]
            heads, labels = convert_sentence(
                [r[1] for r in rows],
                [r[2] for r in rows],
                [r[3] for r in rows],
                [r[5] for r in rows],
                [int(r[6]) for r in rows],
                [r[7] for r in rows],
            )
            for head, label, row in zip(heads, labels, expected, strict=True):
                words += 1
                same += head == int(row[6]) and label == row[7]
    # One word of 254,820 differs: Grew leaves one "as well as" unchained.
    assert words == 254_820
    assert words - same <= 1
