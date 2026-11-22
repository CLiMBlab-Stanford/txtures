"""Integration cost under Dependency Locality Theory (Gibson 2000).

The definition follows the ModelBlocks implementation (`calcDLT.py`) used by
Shain et al. (2022), applied to SUD trees instead of categorial-grammar
trees. With `distance: referents`, the cost at a word *w* is its own weight
as a discourse referent plus, for every arc between *w* and an earlier word
*h*, the weight of the words strictly between them. An arc that ends at an
object gap also counts *w*. With `distance: words`, every word weighs 1 and
a word has no cost of its own. Punctuation weighs nothing, and arcs to
punctuation are ignored.

Arcs come from the SUD conversion of the `parse` module's trees, with
coordinating conjunctions as the heads of coordinations unless
`coordination_head: conjunct` (`txtures.coordination`), and relative clauses
rewired to Gibson's analysis unless `relative_clauses: sud`.

Variants:

- `verbs` (V): with `distance: referents`, non-finite verbs weigh 1 and
  finite verbs 2.
- `coordination` (C): a complete coordination that reaches into an arc's
  span counts as its heaviest conjunct, and an arc from a later conjunct to
  a word before the coordination skips the conjuncts before its own.
- `modifiers` (M): arcs from *w* to an earlier word that modifies it are
  ignored.

`docs/dependency-locality.md` describes the rules for users and must be kept
in sync with this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from pydantic import model_validator

from txtures.coordination import conjunction_heads
from txtures.corpus import Document
from txtures.modules.base import Item, Module, ModuleConfig, RunContext
from txtures.modules.syntax import sentence_ranges, sud_analysis
from txtures.sud import Label, parse_feats


class DependencyLocalityConfig(ModuleConfig):
    """Parameters for DLT integration cost."""

    distance: Literal["words", "referents"] = "referents"
    """Whether words weigh 1 each or only discourse referents count."""

    verbs: bool = False
    """V: non-finite verbs weigh 1 and finite verbs 2. Needs `distance: referents`."""

    coordination: bool = False
    """C: weigh a complete coordination in an arc's span as its heaviest conjunct,
    and skip earlier conjuncts on arcs from a later conjunct to a word before it."""

    modifiers: bool = False
    """M: ignore arcs from a word to an earlier word that modifies it."""

    transform: Literal["none", "log"] = "none"
    """`log` reports log(1 + cost), natural log, of each region's cost."""

    relative_clauses: Literal["gibson", "sud"] = "gibson"
    """`gibson` links a relative pronoun to its antecedent; `sud` keeps the SUD tree."""

    coordination_head: Literal["conjunction", "conjunct"] = "conjunction"
    """`conjunction` makes coordinating conjunctions head their conjuncts;
    `conjunct` keeps SUD's analysis, headed by the first conjunct."""

    @model_validator(mode="after")
    def _verbs_need_referents(self) -> Self:
        if self.verbs and self.distance != "referents":
            raise ValueError("verbs: true needs distance: referents")
        return self


@dataclass(frozen=True)
class Arc:
    """A dependency between two words of a document, as head and dependent indices."""

    head: int
    dependent: int
    label: str

    @property
    def modifier(self) -> bool:
        return Label.parse(self.label).rel == "mod"


def relative_clauses(
    parse: Mapping[str, np.ndarray], first: int, stop: int
) -> list[tuple[int, int, int]]:
    """Find relative clauses with a relative word in words `first:stop` of the UD tree.

    Returns `(antecedent, clause, relative)` word indices for every
    `acl:relcl` clause whose relative word the rules find.
    """
    heads = parse["head"]
    deprels = [str(d) for d in parse["deprel"]]
    children: dict[int, list[int]] = {}
    for word in range(first, stop):
        children.setdefault(int(heads[word]), []).append(word)

    def is_clause(word: int) -> bool:
        if deprels[word].startswith("acl:relcl"):
            return True
        return deprels[word] == "conj" and heads[word] >= 0 and is_clause(int(heads[word]))

    found = []
    for clause in range(first, stop):
        if not deprels[clause].startswith("acl:relcl") or heads[clause] < 0:
            continue
        words, pending = [], [clause]
        while pending:
            word = pending.pop()
            words.append(word)
            pending.extend(c for c in children.get(word, []) if not is_clause(c))
        relatives = [
            w
            for w in sorted(words)
            if w <= clause and parse_feats(str(parse["feats"][w])).get("PronType") == "Rel"
        ]
        if relatives:
            found.append((int(heads[clause]), clause, relatives[0]))
    return found


@dataclass(frozen=True)
class Arcs:
    """The arcs costs are computed over, and the SUD tree they come from.

    `gaps` holds the `(head, dependent)` pairs of arcs that end at an object
    gap: the arc from a relative word (or the phrase it heads) to the verb
    whose object it is. Such an arc also counts the verb itself.
    """

    arcs: list[Arc]
    heads: np.ndarray
    labels: list[str]
    gaps: frozenset[tuple[int, int]]
    conjunctions: frozenset[int] = frozenset()


def arcs(parse: Mapping[str, np.ndarray], relative: str, coordination: str = "conjunct") -> Arcs:
    """Return the arcs costs are computed over.

    With `coordination == "conjunction"`, coordinating conjunctions first
    become the heads of their conjuncts (`txtures.coordination`).

    With `relative == "gibson"`, the arc from each relative clause's
    antecedent to its SUD head is replaced by an arc from the antecedent to
    the clause's relative word. Other arcs keep their head and dependent,
    which is equivalent, for every cost, to reversing the path between the
    two words.
    """
    heads, labels, shared = sud_analysis(parse)
    conjunctions: frozenset[int] = frozenset()
    if coordination == "conjunction":
        new_heads, labels, conjunctions = conjunction_heads(heads, labels, shared)
        heads = np.array(new_heads, dtype=np.int64)
    result = {
        int(d): Arc(int(heads[d]), int(d), labels[d]) for d in range(len(heads)) if heads[d] >= 0
    }
    extra, gaps = [], set()
    if relative == "gibson":
        for first, stop in sentence_ranges(parse):
            for antecedent, clause, word in relative_clauses(parse, first, stop):
                top = clause
                seen = set()
                while top >= 0 and heads[top] != antecedent and top not in seen:
                    seen.add(top)
                    top = int(heads[top])
                if top < 0 or heads[top] != antecedent or top == word:
                    continue
                if top in conjunctions and heads[word] != top:
                    # Conjoined clauses, each with its own relative word, stay as they are.
                    continue
                ancestor = word
                while ancestor >= 0 and ancestor != top:
                    ancestor = int(heads[ancestor])
                if ancestor != top:
                    continue
                label = result.pop(top).label
                extra.append(Arc(antecedent, word, label))
                step = word
                while step != top:
                    if labels[step].split("@")[0] == "comp:obj":
                        gaps.add((int(heads[step]), step))
                        break
                    step = int(heads[step])
    return Arcs([*result.values(), *extra], heads, labels, frozenset(gaps), conjunctions)


@dataclass(frozen=True)
class Coordination:
    """A coordinate structure: its head, each conjunct's words, and all its words.

    `extent` is the conjuncts' words, plus, when a conjunction heads the
    structure, the conjunction and everything between the first and last
    conjunct.
    """

    head: int
    conjuncts: tuple[frozenset[int], ...]
    extent: frozenset[int]

    @property
    def words(self) -> frozenset[int]:
        return self.extent

    @property
    def end(self) -> int:
        return max(self.extent)


def coordinations(
    heads: np.ndarray, labels: Sequence[str], conjunctions: frozenset[int] = frozenset()
) -> list[Coordination]:
    """Find every coordinate structure in a tree.

    A conjunction in `conjunctions` heads a structure whose conjuncts are the
    subtrees of its `conj:coord` dependents. Elsewhere, conjuncts form SUD
    chains of `conj:coord` arcs (`conj:coord@emb` for an embedded
    coordination), and each conjunct holds its head's subtree without the
    next conjunct's.
    """
    children: dict[int, list[int]] = {}
    for word, head in enumerate(heads):
        children.setdefault(int(head), []).append(word)

    def subtree(word: int) -> set[int]:
        found, pending = set(), [word]
        while pending:
            current = pending.pop()
            found.add(current)
            pending.extend(children.get(current, []))
        return found

    def coordinated(word: int) -> Label | None:
        label = Label.parse(labels[word])
        return label if label.rel == "conj" and label.sub == "coord" else None

    def following(word: int, embedded: bool) -> int | None:
        for child in children.get(word, []):
            label = coordinated(child)
            if label is not None and (label.deep == "emb") == embedded:
                return child
        return None

    result = []
    for conjunction in sorted(conjunctions):
        conjuncts = tuple(
            frozenset(subtree(c)) for c in children.get(conjunction, []) if coordinated(c)
        )
        if len(conjuncts) < 2:
            continue
        inside = frozenset().union(*conjuncts) | {conjunction}
        result.append(
            Coordination(conjunction, conjuncts, frozenset(range(min(inside), max(inside) + 1)))
        )
    for start in range(len(heads)):
        if start in conjunctions:
            continue
        for embedded in (False, True):
            if following(start, embedded) is None:
                continue
            own = coordinated(start)
            if (
                own is not None
                and (own.deep == "emb") == embedded
                and int(heads[start]) not in conjunctions
            ):
                continue
            chain = [start]
            while (next_word := following(chain[-1], embedded)) is not None:
                chain.append(next_word)
            excluded: set[int] = set()
            if embedded and (outer := following(start, False)) is not None:
                excluded = subtree(outer)
            conjuncts = []
            for index, conjunct in enumerate(chain):
                words = subtree(conjunct) - excluded
                if index + 1 < len(chain):
                    words -= subtree(chain[index + 1])
                conjuncts.append(frozenset(words))
            result.append(Coordination(start, tuple(conjuncts), frozenset().union(*conjuncts)))
    return result


def word_weights(parse: Mapping[str, np.ndarray], config: DependencyLocalityConfig) -> np.ndarray:
    """Return the weight of every word under `config`'s distance and V variant.

    With `distance: referents`, nouns and proper nouns weigh 1 and finite
    verbs and auxiliaries 1 (2 with V); non-finite verbs and auxiliaries
    weigh 0 (1 with V); other words, including pronouns, weigh 0. With
    `distance: words`, every word weighs 1. Punctuation always weighs 0.
    """
    weights = np.zeros(len(parse["form"]), dtype=np.float64)
    for index, (upos, feats) in enumerate(zip(parse["upos"], parse["feats"], strict=True)):
        upos = str(upos)
        if upos == "PUNCT":
            continue
        if config.distance == "words":
            weights[index] = 1
            continue
        finite = parse_feats(str(feats)).get("VerbForm") == "Fin"
        if upos in ("NOUN", "PROPN"):
            weights[index] = 1
        elif upos in ("VERB", "AUX"):
            if config.verbs:
                weights[index] = 2 if finite else 1
            else:
                weights[index] = 1 if finite else 0
    return weights


def coordination_weight(
    structure: Coordination, weights: np.ndarray, structures: Sequence[Coordination]
) -> float:
    """Weight of a coordination: the weight of its heaviest conjunct.

    Coordinations nested in a conjunct count as their own heaviest conjunct.
    """

    def weigh(words: frozenset[int]) -> float:
        nested = [s for s in structures if s.words < words]
        outermost = [s for s in nested if not any(s.words < o.words for o in nested)]
        covered = frozenset().union(*(s.words for s in outermost)) if outermost else frozenset()
        total = float(sum(weights[w] for w in words - covered))
        return total + sum(coordination_weight(s, weights, structures) for s in outermost)

    return max(weigh(conjunct) for conjunct in structure.conjuncts)


def span_weight(
    start: int,
    stop: int,
    weights: np.ndarray,
    complete: Sequence[Coordination],
    structures: Sequence[Coordination],
) -> float:
    """Weight of words `start..stop - 1` under C.

    Each coordination in `complete` that reaches into the span counts once,
    as its heaviest conjunct, even when it starts before `start`. `complete`
    holds the outermost coordinations that end before the current word,
    ordered by their first word.
    """
    total, position = 0.0, start
    for structure in complete:
        first, last = min(structure.words), max(structure.words)
        if stop < last:
            break
        if last < position:
            continue
        if first > position:
            total += float(weights[position:first].sum())
        position = last + 1
        total += coordination_weight(structure, weights, structures)
    if stop > position:
        total += float(weights[position:stop].sum())
    return total


def word_costs(parse: Mapping[str, np.ndarray], config: DependencyLocalityConfig) -> np.ndarray:
    """Return the integration cost at every word of a parsed document.

    The cost at a word is its own weight as a discourse referent (with
    `distance: referents`) plus, for every arc to an earlier word, the weight
    of the words strictly between the two. An arc from a relative word to
    the verb whose object it is also counts the verb, because the object gap
    follows it.
    """
    found = arcs(parse, config.relative_clauses, config.coordination_head)
    weights = word_weights(parse, config)
    upos = [str(u) for u in parse["upos"]]
    structures = (
        coordinations(found.heads, found.labels, found.conjunctions) if config.coordination else []
    )
    costs = weights.copy() if config.distance == "referents" else np.zeros(len(weights))
    for arc in found.arcs:
        early, late = sorted((arc.head, arc.dependent))
        if upos[early] == "PUNCT" or upos[late] == "PUNCT":
            continue
        if config.modifiers and arc.modifier and arc.dependent == early:
            continue
        stop = late + 1 if (arc.head, arc.dependent) in found.gaps else late
        if not config.coordination:
            costs[late] += weights[early + 1 : stop].sum()
            continue
        done = [s for s in structures if max(s.words) < late]
        complete = sorted(
            (s for s in done if not any(s.words < o.words for o in done)),
            key=lambda s: min(s.words),
        )
        open_ = [
            s
            for s in structures
            if late in s.words and early not in s.words and early < min(s.words)
        ]
        if open_:
            outer = max(open_, key=lambda s: len(s.words))
            # A word in no conjunct, such as the heading conjunction, skips nothing.
            conjunct = next((c for c in outer.conjuncts if late in c), outer.words)
            costs[late] += span_weight(early + 1, min(outer.words), weights, complete, structures)
            costs[late] += span_weight(min(conjunct), stop, weights, complete, structures)
        else:
            costs[late] += span_weight(early + 1, stop, weights, complete, structures)
    return costs


def region_costs(
    document: Document, parse: Mapping[str, np.ndarray], config: DependencyLocalityConfig
) -> np.ndarray:
    """Sum word costs into regions and apply `transform`. Regions without words are NaN."""
    costs = word_costs(parse, config)
    result = np.full(len(document.regions), np.nan)
    for region in range(len(document.regions)):
        words = np.flatnonzero(parse["region"] == region)
        if len(words):
            result[region] = costs[words].sum()
    if config.transform == "log":
        result = np.log1p(result)
    return result


class DependencyLocality(Module):
    """Compute DLT integration cost for each region."""

    name = "dependency_locality"
    contract_version = 1
    Config = DependencyLocalityConfig
    requires = ("parse",)
    features = ("dlt",)
    sweeps = {
        "distance": ("distance",),
        "verbs": ("verbs",),
        "coordination": ("coordination",),
        "modifiers": ("modifiers",),
        "transform": ("transform",),
        "relative_clauses": ("relative_clauses",),
        "coordination_head": ("coordination_head",),
    }
    persist = False

    def run(self, config: DependencyLocalityConfig, item: Item, context: RunContext):
        return region_costs(item.document, item.upstream["parse"], config)

    def columns(self, config: DependencyLocalityConfig, document: Document, result):
        return {"dlt": result}
