"""Coordination headed by the coordinating conjunction.

SUD heads a coordination by its first conjunct and chains the others
(*A* → *B* → *C*). `conjunction_heads` rewrites such chains so that each
coordinating conjunction heads the structure before it and the conjunct after
it. Several conjunctions nest from the left: *A and B or C* becomes
*or*(*and*(*A*, *B*), *C*). A conjunct with no conjunction of its own (*A, B
and C*, or an asyndetic *A, B*) attaches to the structure before it, as in
SUD.

The top conjunction takes the coordination's place in the tree: the head and
relation of the first conjunct. Dependents that SUD's rules mark
`Shared=Yes` (a subject or preposed phrase of the first conjunct that the
other conjuncts lack, a shared right dependent, a correlative such as
*both*) attach to the top conjunction. Other dependents stay where SUD puts
them.
"""

from __future__ import annotations

from collections.abc import Sequence

from txtures.sud import Label

COORDINATED = "conj:coord"


def _coordinated(label: str) -> Label | None:
    parsed = Label.parse(label)
    return parsed if parsed.rel == "conj" and parsed.sub == "coord" else None


def _children(heads: Sequence[int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for word, head in enumerate(heads):
        children.setdefault(int(head), []).append(word)
    return children


def _chains(heads: Sequence[int], labels: Sequence[str]) -> list[list[int]]:
    """SUD conjunct chains: plain chains first, then embedded ones by depth."""
    children = _children(heads)

    def following(word: int, embedded: bool) -> int | None:
        for child in children.get(word, []):
            label = _coordinated(labels[child])
            if label is not None and (label.deep == "emb") == embedded:
                return child
        return None

    def depth(word: int) -> int:
        count = 0
        while heads[word] >= 0:
            word, count = int(heads[word]), count + 1
        return count

    chains = []
    for embedded in (False, True):
        found = []
        for start in range(len(heads)):
            if following(start, embedded) is None:
                continue
            own = _coordinated(labels[start])
            if own is not None and (own.deep == "emb") == embedded:
                continue
            chain = [start]
            while (next_word := following(chain[-1], embedded)) is not None:
                chain.append(next_word)
            found.append(chain)
        chains.extend(sorted(found, key=lambda chain: depth(chain[0])))
    return chains


def conjunction_heads(
    heads: Sequence[int], labels: Sequence[str], shared: Sequence[str | None]
) -> tuple[list[int], list[str], frozenset[int]]:
    """Rewrite SUD coordination so that conjunctions head their conjuncts.

    `heads` holds each word's head index (-1 for a root), and `shared` the
    `Shared` values from `txtures.sud.analyze_sentence`. Returns the new
    heads and relations, and the conjunctions that now head a coordination.
    """
    heads, labels = [int(h) for h in heads], list(labels)
    conjunctions: set[int] = set()
    for chain in _chains(heads, labels):
        children = _children(heads)
        markers = []
        for conjunct in chain[1:]:
            ccs = sorted(
                c
                for c in children.get(conjunct, [])
                if Label.parse(labels[c]).rel == "cc"
                and Label.parse(labels[c]).deep != "preconj"
                and c < conjunct
            )
            markers.append(ccs)
        if not any(markers):
            continue
        governor, relation = heads[chain[0]], labels[chain[0]]
        top = chain[0]
        for conjunct, ccs in zip(chain[1:], markers, strict=True):
            if ccs:
                conjunction = ccs[0]
                heads[top], labels[top] = conjunction, COORDINATED
                heads[conjunct], labels[conjunct] = conjunction, COORDINATED
                for extra in ccs[1:]:
                    heads[extra] = conjunction
                conjunctions.add(conjunction)
                top = conjunction
            else:
                heads[conjunct], labels[conjunct] = top, COORDINATED
        heads[top], labels[top] = governor, relation
        for conjunct in chain:
            for child in children.get(conjunct, []):
                if shared[child] == "Yes" and child not in conjunctions and child not in chain:
                    heads[child] = top
    return heads, labels, frozenset(conjunctions)
