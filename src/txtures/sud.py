"""Conversion of Universal Dependencies (UD) trees to Surface-Syntactic UD (SUD).

This is a port of the official Grew conversion `UD_to_SUD.grs` from
https://github.com/surfacesyntacticud/conversions (commit `df5aca2`), limited
to what changes heads and relation labels. Rules that only set features
(`Shared`, `Subject`, `ExtPos`) are left out. The steps run in the order of
the official strategy:

1. relabel UD relations as SUD relations (`sub_relations`,
   `rel_extensions`, `relations`);
2. reverse `aux`, `case`, `cop`, and `mark`, so that function words head
   their phrases, and raise dependents of the old head that SUD attaches to
   the new one (`reverse_relations`);
3. attach right dependents shared by conjuncts to the nearest conjunct
   (`minimize_right_conj-dep`);
4. mark embedded coordinations and turn bouquets of `conj`, `fixed`, and
   `flat` dependents into chains (`add_conj_emb`, `chained_relations`);
5. rename the remaining UD relations to `unk` (`normalize`).

Grew treats a fixed expression as one unit with the part of speech in its
`ExtPos` feature. Here the unit is its first word: dependents of the other
words of the expression are moved to it first, its `ExtPos` (or its own
UPOS) is used when rules test its part of speech, and the expression's
internal `fixed` relations are never moved.

Each step applies its rules until none applies, as Grew's `Onf` does. Rule
names follow the official file so that the two can be compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

NOMINAL = frozenset({"NOUN", "PRON", "PROPN", "NUM", "X"})
FUNCTION_RELATIONS = frozenset({"aux", "case", "cop", "mark"})
KEPT_RELATIONS = frozenset(
    {
        "appos",
        "cc",
        "clf",
        "comp",
        "compound",
        "conj",
        "det",
        "discourse",
        "dislocated",
        "flat",
        "goeswith",
        "list",
        "mod",
        "orphan",
        "parataxis",
        "punct",
        "reparandum",
        "root",
        "subj",
        "udep",
        "unk",
        "vocative",
    }
)
"""First components of relations that `normalize` keeps; all others become `unk`."""


@dataclass
class Label:
    """A relation: its name, its subtype after `:`, and its deep extension after `@`."""

    rel: str
    sub: str | None = None
    deep: str | None = None

    @classmethod
    def parse(cls, text: str) -> Label:
        base, _, deep = text.partition("@")
        rel, _, sub = base.partition(":")
        return cls(rel, sub or None, deep or None)

    def __str__(self) -> str:
        text = self.rel + (f":{self.sub}" if self.sub else "")
        return text + (f"@{self.deep}" if self.deep else "")

    def is_(self, *texts: str) -> bool:
        """Whether the whole label equals one of `texts`, as Grew's `-[a|b]->` tests."""
        return str(self) in texts

    def copy(self) -> Label:
        return Label(self.rel, self.sub, self.deep)


@dataclass
class Word:
    """One syntactic word of a sentence being converted."""

    form: str
    lemma: str
    upos: str
    feats: dict[str, str]
    head: int
    label: Label
    reversed: bool = False
    """Grew's `REV=Yes`: the arc was created by reversing a function relation."""
    level: bool = False
    """Grew's `level=10`: the arc is a function relation still to be reversed."""
    shared: str | None = None
    """Grew's `Shared` feature: `Yes` for a dependent of the first or last
    conjunct that all conjuncts share, `No` for one that only its conjunct
    has, `None` when the rules do not decide."""


@dataclass
class Graph:
    """A dependency tree with words numbered from 1; head 0 is the root."""

    words: list[Word]
    idiom_members: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> Word:
        return self.words[index - 1]

    def ids(self) -> range:
        return range(1, len(self.words) + 1)

    def children(self, head: int) -> list[int]:
        return [i for i in self.ids() if self[i].head == head]

    def upos(self, index: int) -> str:
        word = self[index]
        return word.feats.get("ExtPos", word.upos)

    def move(self, dependent: int, head: int) -> None:
        """Attach `dependent` to `head`, keeping its relation."""
        self[dependent].head = head


def same_value(a: str | None, b: str | None) -> bool:
    """Whether two feature values are equal in a Grew pattern such as `D.VerbForm = N.VerbForm`.

    Grew does not treat two `Inf` values as equal; the official SUD
    conversion of UD_English-EWT raises every conjunct infinitive that the
    rules would otherwise keep. Matching that output needs the same test.
    """
    return a is not None and a == b and a != "Inf"


def parse_feats(text: str) -> dict[str, str]:
    if not text or text == "_":
        return {}
    pairs = (item.split("=", 1) for item in text.split("|") if "=" in item)
    return {key: value for key, value in pairs}


Rule = Callable[[Graph], bool]
"""A rule applies itself once if it can, and reports whether it did."""


def normal_form(graph: Graph, rules: Sequence[Rule]) -> None:
    """Apply `rules` until none applies, as Grew's `Onf` does.

    When several rules apply, the one declared last in the official file
    wins, which reproduces Grew's choices on UD_English-EWT.
    """
    for _ in range(100 * max(len(graph), 1) + 100):
        if not any(rule(graph) for rule in reversed(rules)):
            return
    raise RuntimeError("SUD conversion did not reach a normal form")


# Idioms (irregular_expr.grs, reduced to their effect on heads)


def gather_idioms(graph: Graph) -> None:
    """Move dependents of the inner words of fixed expressions to the expression's head."""
    for index in graph.ids():
        if graph[index].label.rel == "fixed":
            graph.idiom_members.add(index)
    for member in sorted(graph.idiom_members):
        head = graph[member].head
        for child in graph.children(member):
            if child not in graph.idiom_members:
                graph.move(child, head)


# sub_relations


def _relabel(test: Callable[[Graph, int], bool], make: Callable[[Label], Label]) -> Rule:
    def rule(graph: Graph) -> bool:
        for d in graph.ids():
            if test(graph, d):
                graph[d].label = make(graph[d].label)
                return True
        return False

    return rule


def _has_child(graph: Graph, head: int, test: Callable[[int], bool]) -> bool:
    return any(test(c) for c in graph.children(head))


def _sub_relations() -> list[Rule]:
    def lab(d, g):
        return g[d].label

    def comp(g, d):
        return lab(d, g).is_("advcl:cleft", "ccomp:obj", "ccomp:obl", "ccomp:pred")

    def comp_xcomp(g, d):
        h = g[d].head
        return (
            lab(d, g).is_("xcomp:obj", "xcomp:obl", "xcomp:pred")
            and g.upos(d) == "VERB"
            and not _has_child(g, h, lambda o: g[o].label.is_("obj") and g.upos(o) in NOMINAL)
        )

    def comp_xcomp_obj(g, d):
        h = g[d].head
        return (
            lab(d, g).is_("xcomp:obl", "xcomp:pred")
            and g.upos(d) == "VERB"
            and _has_child(g, h, lambda o: g[o].label.is_("obj") and g.upos(o) in NOMINAL)
        )

    def comp_xcomp_cop(g, d):
        return (
            lab(d, g).is_("xcomp:obj", "xcomp:obl", "xcomp:pred")
            and g.upos(d) != "VERB"
            and _has_child(g, d, lambda c: g[c].label.is_("cop"))
        )

    def comp_dep(g, d):
        h = g[d].head
        return h > 0 and g.upos(h) not in ("NOUN", "PRON") and lab(d, g).is_("dep:comp")

    def expl_pv(g, d):
        return lab(d, g).is_("expl:pv", "expl:comp")

    def expl_pass(g, d):
        return lab(d, g).is_("expl:pass")

    def impersonal(g, d, *, passive: bool):
        h = g[d].head
        label = lab(d, g)
        if passive:
            if not (label.rel in ("csubj", "nsubj") and label.sub == "pass"):
                return False
        elif not label.is_("nsubj", "csubj"):
            return False
        if h == 0 or not _has_child(g, h, lambda s: g[s].label.is_("expl:subj")):
            return False
        if passive:
            return True
        expls = [s for s in g.children(h) if g[s].label.is_("expl:subj")]
        marks = [c for c in g.children(h) if g[c].label.is_("mark")]
        for c in marks:
            for s in expls:
                if d < c < s or s < c < d:
                    return False
        return True

    def obl_arg(g, d):
        label = lab(d, g)
        return label.rel == "obl" and label.sub in ("arg", "comp")

    def obl_agent(g, d):
        label = lab(d, g)
        return label.rel == "obl" and label.sub == "agent"

    def dep_nom(g, d):
        h = g[d].head
        return h > 0 and g.upos(h) in ("NOUN", "PRON") and lab(d, g).is_("dep:comp")

    def xcomp_pred(g, d):
        return (
            lab(d, g).is_("xcomp:pred")
            and g.upos(d) not in ("AUX", "VERB")
            and not _has_child(g, d, lambda c: g[c].label.is_("cop"))
        )

    def ccomp_pred(g, d):
        return lab(d, g).is_("ccomp:pred")

    def obl_mod(g, d):
        return lab(d, g).is_("obl:mod")

    def relcl(g, d):
        return lab(d, g).is_("acl:relcl")

    def clf(g, d):
        return lab(d, g).is_("clf")

    def subj_expl(g, d):
        h = g[d].head
        if not lab(d, g).is_("expl:subj"):
            return False
        subjects = [
            s for s in g.children(h) if g[s].label.is_("csubj", "nsubj", "csubj:pass", "nsubj:pass")
        ]
        if not subjects:
            return True
        marks = [c for c in g.children(h) if g[c].label.is_("mark")]
        plain = [s for s in g.children(h) if g[s].label.is_("csubj", "nsubj")]
        for c in marks:
            for s in plain:
                if s < c < d and not any(c < s1 for s1 in subjects):
                    return True
                if d < c < s and not any(s1 < c for s1 in subjects):
                    return True
        return False

    def other_expl(g, d):
        label = lab(d, g)
        return label.rel == "expl" and label.sub not in (None, "subj", "comp", "pass", "pv")

    return [
        _relabel(comp, lambda x: Label("comp", x.sub, x.deep)),
        _relabel(comp_xcomp, lambda x: Label("comp", x.sub, x.deep)),
        _relabel(comp_xcomp_obj, lambda x: Label("comp", x.sub, x.deep)),
        _relabel(comp_xcomp_cop, lambda x: Label("comp", x.sub, x.deep)),
        _relabel(comp_dep, lambda x: Label("comp")),
        _relabel(expl_pv, lambda x: Label("comp", None, "expl")),
        _relabel(expl_pass, lambda x: Label("comp", None, "pass")),
        _relabel(
            lambda g, d: impersonal(g, d, passive=False), lambda x: Label("comp", "obj", "agent")
        ),
        _relabel(
            lambda g, d: impersonal(g, d, passive=True), lambda x: Label("comp", "obj", "pass")
        ),
        _relabel(obl_arg, lambda x: Label("comp", "obl", x.deep)),
        _relabel(obl_agent, lambda x: Label("comp", "obl", "agent")),
        _relabel(dep_nom, lambda x: Label("comp", "obl")),
        _relabel(xcomp_pred, lambda x: Label("comp", "pred")),
        _relabel(ccomp_pred, lambda x: Label("comp", "pred", x.deep)),
        _relabel(obl_mod, lambda x: Label("mod", None, x.deep)),
        _relabel(relcl, lambda x: Label("mod", None, "relcl")),
        _relabel(clf, lambda x: Label("mod", "clf")),
        _relabel(subj_expl, lambda x: Label("subj", None, "expl")),
        _relabel(other_expl, lambda x: Label("unk", None, "expl")),
    ]


# rel_extensions


def _rel_extensions() -> list[Rule]:
    def test(g, d):
        label = g[d].label
        return (
            label.sub is not None
            and label.rel not in ("aux", "comp", "subj")
            and label.sub not in ("poss", "outer", "clf")
        )

    return [_relabel(test, lambda x: Label(x.rel, None, x.sub))]


# relations


def _relations() -> list[Rule]:
    def lab(g, d):
        return g[d].label

    def head_upos(g, d):
        h = g[d].head
        return g.upos(h) if h > 0 else None

    def has_obj(g, h):
        return _has_child(g, h, lambda o: g[o].label.is_("obj", "comp:obj"))

    def has_aux_pass(g, h):
        return _has_child(g, h, lambda o: g[o].label.is_("aux@pass", "comp:aux@pass"))

    def marked_by_adp(g, d):
        return _has_child(g, d, lambda m: g[m].label.is_("mark") and g.upos(m) == "ADP")

    def has_cop(g, d):
        return _has_child(g, d, lambda c: g[c].label.is_("cop"))

    def has_mark(g, d):
        return _has_child(g, d, lambda c: g[c].label.is_("mark"))

    def appos(g, d):
        return lab(g, d).is_("appos")

    def comp_expl(g, d):
        return lab(g, d).is_("expl") and head_upos(g, d) == "VERB" and g.upos(d) == "PRON"

    def obj(g, d):
        label = lab(g, d)
        h = g[d].head
        if label.rel != "obj":
            return False
        return not (
            label.deep == "agent" and _has_child(g, h, lambda a: g[a].label.is_("aux:caus"))
        )

    def obj_agent(g, d):
        h = g[d].head
        return lab(g, d).is_("obj@agent") and _has_child(g, h, lambda a: g[a].label.is_("aux:caus"))

    def ccomp_obj(g, d):
        h = g[d].head
        label = lab(g, d)
        return (
            label.rel == "ccomp"
            and head_upos(g, d) == "VERB"
            and not has_obj(g, h)
            and label.deep != "agent"
            and not marked_by_adp(g, d)
            and not has_aux_pass(g, h)
        )

    def xcomp_obj(g, d):
        h = g[d].head
        return (
            lab(g, d).rel == "xcomp"
            and head_upos(g, d) == "VERB"
            and g.upos(d) == "VERB"
            and not has_cop(g, d)
            and not has_obj(g, h)
            and not marked_by_adp(g, d)
            and not has_aux_pass(g, h)
        )

    def xcomp_cop(g, d):
        h = g[d].head
        return (
            lab(g, d).rel == "xcomp"
            and head_upos(g, d) == "VERB"
            and has_cop(g, d)
            and not has_obj(g, h)
            and not marked_by_adp(g, d)
            and not has_aux_pass(g, h)
        )

    def ccomp_obl(g, d):
        return lab(g, d).rel == "ccomp" and head_upos(g, d) == "VERB" and marked_by_adp(g, d)

    def ccomp_non_verb(g, d):
        return lab(g, d).rel == "ccomp" and g[d].head > 0 and head_upos(g, d) != "VERB"

    def ccomp_agent(g, d):
        label = lab(g, d)
        return label.rel == "ccomp" and label.deep == "agent"

    def ccomp_with_obj(g, d):
        return lab(g, d).rel == "ccomp" and has_obj(g, g[d].head)

    def ccomp_pass(g, d):
        # `H1 -[aux@pass|comp:aux@pass]-> H`: the head is itself a passive auxiliary's dependent.
        h = g[d].head
        if lab(g, d).rel != "ccomp" or h == 0:
            return False
        return g[h].label.is_("aux@pass", "comp:aux@pass")

    def iobj(g, d):
        return lab(g, d).rel == "iobj"

    def xcomp_obl(g, d):
        return lab(g, d).rel == "xcomp" and g.upos(d) == "VERB" and has_mark(g, d)

    def xcomp_obl_cop(g, d):
        return (
            lab(g, d).rel == "xcomp"
            and has_mark(g, d)
            and _has_child(g, d, lambda c: g[c].label.is_("cop") and g.upos(c) == "AUX")
        )

    def xcomp_obl_obj(g, d):
        return (
            lab(g, d).rel == "xcomp"
            and g.upos(d) == "VERB"
            and has_obj(g, g[d].head)
            and not has_mark(g, d)
        )

    def xcomp_pred(g, d):
        return lab(g, d).rel == "xcomp" and g.upos(d) != "VERB" and not has_cop(g, d)

    def xcomp_part(g, d):
        return (
            lab(g, d).rel == "xcomp"
            and g.upos(d) == "VERB"
            and g[d].feats.get("VerbForm") == "Part"
        )

    def conj(g, d):
        return lab(g, d).is_("conj")

    def mod(g, d):
        return lab(g, d).rel in ("advcl", "advmod", "amod", "nummod")

    def marked(g, d):
        return _has_child(g, d, lambda c: g[c].label.is_("case", "mark"))

    def acl_nmod(g, d):
        return lab(g, d).rel in ("acl", "nmod") and not marked(g, d)

    def subj(g, d):
        return lab(g, d).rel in ("nsubj", "csubj")

    def udep(g, d):
        return lab(g, d).rel in ("acl", "nmod") and marked(g, d)

    def udep_advmod_det(g, d):
        h = g[d].head
        return (
            lab(g, d).is_("advmod")
            and h > 0
            and _has_child(g, h, lambda c: g[c].label.is_("det"))
            and marked(g, d)
        )

    def udep_advmod_noun(g, d):
        return (
            lab(g, d).is_("advmod")
            and head_upos(g, d) in ("NOUN", "NUM", "PRON", "PROPN", "X")
            and marked(g, d)
        )

    def udep_obl(g, d):
        label = lab(g, d)
        return label.rel == "obl" and label.deep != "agent"

    def unk(g, d):
        return lab(g, d).rel in ("dep", "fixed")

    def unk_expl(g, d):
        return lab(g, d).is_("expl") and not (head_upos(g, d) == "VERB" and g.upos(d) == "PRON")

    def keep(rel, sub=None):
        return lambda x: Label(rel, sub if sub is not None else x.sub, x.deep)

    return [
        _relabel(appos, lambda x: Label("conj", "appos")),
        _relabel(comp_expl, lambda x: Label("comp", None, "expl")),
        _relabel(obj, lambda x: Label("comp", "obj", x.deep)),
        _relabel(obj_agent, lambda x: Label("comp", "obj", x.deep)),
        _relabel(ccomp_obj, lambda x: Label("comp", "obj", x.deep)),
        _relabel(xcomp_obj, lambda x: Label("comp", "obj", x.deep)),
        _relabel(xcomp_cop, lambda x: Label("comp", "obj", x.deep)),
        _relabel(ccomp_obl, lambda x: Label("comp", "obl", x.deep)),
        _relabel(ccomp_non_verb, lambda x: Label("comp", "obl", x.deep)),
        _relabel(ccomp_agent, lambda x: Label("comp", "obl", x.deep)),
        _relabel(ccomp_with_obj, lambda x: Label("comp", "obl", x.deep)),
        _relabel(ccomp_pass, lambda x: Label("comp", "obl", x.deep)),
        _relabel(iobj, lambda x: Label("comp", "obl", x.deep)),
        _relabel(xcomp_obl, lambda x: Label("comp", "obl", x.deep)),
        _relabel(xcomp_obl_cop, lambda x: Label("comp", "obl", x.deep)),
        _relabel(xcomp_obl_obj, lambda x: Label("comp", "obl", x.deep)),
        _relabel(xcomp_pred, lambda x: Label("comp", "pred", x.deep)),
        _relabel(xcomp_part, lambda x: Label("comp", "pred")),
        _relabel(conj, lambda x: Label("conj", "coord", x.deep)),
        _relabel(mod, keep("mod")),
        _relabel(acl_nmod, keep("mod")),
        _relabel(subj, keep("subj")),
        _relabel(udep, keep("udep")),
        _relabel(udep_advmod_det, lambda x: Label("udep")),
        _relabel(udep_advmod_noun, lambda x: Label("udep")),
        _relabel(udep_obl, keep("udep")),
        _relabel(unk, keep("unk")),
        _relabel(unk_expl, lambda x: Label("unk", None, "expl")),
    ]


# reverse_relations


def set_levels(graph: Graph) -> None:
    for d in graph.ids():
        if graph[d].label.rel in FUNCTION_RELATIONS:
            graph[d].level = True


def _pending(graph: Graph, head: int) -> list[int]:
    return [c for c in graph.children(head) if graph[c].level and not graph[c].reversed]


def reverse_edges(graph: Graph) -> bool:
    """Reverse one function relation (`reverse_edges.r`).

    Of a head's function dependents, the nearest one after it is reversed
    first, then the nearest one before it.
    """
    for h in graph.ids():
        pending = _pending(graph, h)
        if not pending:
            continue
        after = [f for f in pending if f > h]
        funct = min(after) if after else max(pending)
        head_word, funct_word = graph[h], graph[funct]
        label = funct_word.label.copy()
        # shift_in: the function word takes the head's place.
        funct_word.head, funct_word.label = head_word.head, head_word.label
        funct_word.reversed, funct_word.level = head_word.reversed, head_word.level
        head_word.head, head_word.label = funct, label
        head_word.reversed, head_word.level = True, True
        # shift_out: the head's other function dependents without a deep extension.
        for c in graph.children(h):
            word = graph[c]
            if word.label.rel in FUNCTION_RELATIONS and not word.label.deep and not word.reversed:
                word.head = funct
        return True
    return False


def _reversed_arcs(graph: Graph):
    """Yield `(h, d)` for every reversed arc `h -> d`."""
    for d in graph.ids():
        if graph[d].reversed and graph[d].head > 0:
            yield graph[d].head, d


def _rise(name: str, test: Callable[[Graph, int, int, int], bool]) -> Rule:
    """A `shift_out_edges` rule: move `d -> n` to `h -> n` where `h -[REV]-> d`."""

    def rule(graph: Graph) -> bool:
        for h, d in list(_reversed_arcs(graph)):
            for n in graph.children(d):
                if graph[n].reversed:
                    continue
                if test(graph, h, d, n):
                    graph.move(n, h)
                    return True
        return False

    rule.__name__ = name
    return rule


def _shift_out_edges() -> list[Rule]:
    def rel(g, i):
        return g[i].label

    def has_later_child(g, d, n):
        return any(x > n for x in g.children(d))

    def has_earlier_child(g, d, n):
        return any(x < n for x in g.children(d))

    def discourse_orphan(g, h, d, n):
        return rel(g, n).is_("discourse", "orphan")

    def cc(g, h, d, n):
        return rel(g, n).is_("cc") and n < h

    def agent_aux_caus(g, h, d, n):
        arc = rel(g, d)
        return arc.rel == "aux" and arc.sub == "caus" and rel(g, n).deep == "agent"

    def agent_cop(g, h, d, n):
        return rel(g, d).rel == "cop" and rel(g, n).deep == "agent"

    def all_mark_embedded(g, h, d, n):
        if rel(g, d).rel not in ("cop", "aux", "case"):
            return False
        if rel(g, n).rel not in ("dislocated", "parataxis", "subj", "vocative"):
            return False
        # `H1 -[1 <> cop]-> H` also holds when H is the root.
        return not (
            rel(g, n).is_("parataxis") and g[h].label.rel != "cop" and rel(g, d).rel == "case"
        )

    def comp_cleft(g, h, d, n):
        return rel(g, d).rel in ("cop", "aux", "case") and rel(g, n).is_("comp:cleft")

    def prep_nom_comp(g, h, d, n):
        return (
            g.upos(h) == "ADP"
            and g.upos(d) in ("NOUN", "NUM", "PROPN", "X")
            and rel(g, d).rel == "case"
            and rel(g, n).rel == "comp"
        )

    def mark_outside_left(g, h, d, n):
        return (
            rel(g, d).rel == "mark"
            and rel(g, n).rel in ("dislocated", "subj", "vocative")
            and n < h < d
        )

    def mark_outside_right(g, h, d, n):
        return rel(g, d).rel == "mark" and rel(g, n).rel in ("dislocated", "subj") and d < h < n

    def h_d_n_mod(g, h, d, n):
        arc, label = rel(g, d), rel(g, n)
        if arc.rel == "mark" or label.rel != "mod" or not (h < d < n):
            return False
        if label.is_("mod@relcl") or has_later_child(g, d, n):
            return False
        if arc.rel == "case" and g.upos(n) not in ("ADP", "SCONJ"):
            return False
        if arc.rel == "cop" and g.upos(n) not in ("ADP", "ADV", "SCONJ"):
            return False
        return not (arc.rel in ("case", "mark") and n in g.idiom_members)

    def matching_open(g, d, n):
        closing = {")": "(", "»": "«", "]": "["}
        for x in g.children(d):
            if x < d and rel(g, x).is_("punct"):
                if g[n].lemma == g[x].lemma or closing.get(g[n].lemma) == g[x].lemma:
                    return True
        return False

    def h_d_n_punct(g, h, d, n):
        return (
            rel(g, n).rel == "punct"
            and h < d < n
            and not has_later_child(g, d, n)
            and not matching_open(g, d, n)
        )

    def h_n_d(g, h, d, n):
        arc = rel(g, d)
        if arc.rel not in ("aux", "cop") or not rel(g, n).is_("mod") or not (h < n < d):
            return False
        if arc.rel == "cop" and g.upos(n) == "ADJ" and g.upos(d) == "NOUN":
            return False
        return not any(h < x < n for x in g.children(d))

    def d_h_n(g, h, d, n):
        label = rel(g, n)
        return label.rel in ("mod", "punct") and d < h < n and label.deep != "relcl"

    def d_n_h(g, h, d, n):
        return rel(g, d).rel == "cop" and rel(g, n).rel == "mod" and d < n < h

    def n_h_d(g, h, d, n):
        return rel(g, n).rel in ("mod", "orphan", "punct", "reparandum", "udep") and n < h < d

    def n_d_h(g, h, d, n):
        label = rel(g, n)
        if label.rel not in ("mod", "punct") or not (n < d < h):
            return False
        if has_earlier_child(g, d, n):
            return False
        return not (label.rel == "mod" and g.upos(n) not in ("ADP", "ADV"))

    def conjunct(label: Label) -> bool:
        return label.rel == "conj" and label.sub == "coord"

    def conj(g, h, d, n):
        if not conjunct(rel(g, n)):
            return False
        du, nu = g.upos(d), g.upos(n)
        if du == nu:
            return False
        dv, nv = g[d].feats.get("VerbForm"), g[n].feats.get("VerbForm")
        if du in NOMINAL and nu in NOMINAL:
            return False
        if du in ("ADJ", "ADP") and nu == "VERB" and nv == "Part":
            return False
        if du == "VERB" and dv == "Part" and nu in ("ADJ", "ADP"):
            return False
        if du in ("ADP", "ADV") and nu in ("ADP", "ADV"):
            return False
        return not (du in ("AUX", "VERB") and nu in ("AUX", "VERB") and same_value(dv, nv))

    def conj_verb(g, h, d, n):
        if not conjunct(rel(g, n)) or not h < d:
            return False
        dv, nv = g[d].feats.get("VerbForm"), g[n].feats.get("VerbForm")
        if g.upos(d) not in ("AUX", "VERB") or not dv:
            return False
        return not (g.upos(n) in ("AUX", "VERB") and same_value(dv, nv))

    return [
        _rise("discourse_orphan", discourse_orphan),
        _rise("cc", cc),
        _rise("agent_aux_caus", agent_aux_caus),
        _rise("agent_cop", agent_cop),
        _rise("all_mark_embedded", all_mark_embedded),
        _rise("comp_cleft", comp_cleft),
        _rise("prep_nom_comp", prep_nom_comp),
        _rise("all_mark_outside_left", mark_outside_left),
        _rise("all_mark_outside_right", mark_outside_right),
        _rise("H_D_N_mod", h_d_n_mod),
        _rise("H_D_N_punct", h_d_n_punct),
        _rise("H_N_D", h_n_d),
        _rise("D_H_N", d_h_n),
        _rise("D_N_H", d_n_h),
        _rise("N_H_D", n_h_d),
        _rise("N_D_H", n_d_h),
        _rise("conj", conj),
        _rise("conj_verb", conj_verb),
    ]


def rename_reversed(graph: Graph) -> None:
    """Clear the reversal marks and rename reversed relations (`cleaning`, `rename_rel`)."""
    for d in graph.ids():
        word = graph[d]
        word.reversed = word.level = False
        label = word.label
        if label.rel in ("case", "mark"):
            word.label = Label("comp")
        elif label.rel == "cop":
            word.label = Label("comp", "pred", label.deep)
        elif label.is_("aux"):
            word.label = Label("comp", "aux")
        elif label.rel == "aux" and label.sub:
            word.label = Label("comp", "aux", label.sub)


# minimize_right_conj-dep


def _conjuncts(graph: Graph, head: int) -> list[int]:
    return [
        c
        for c in graph.children(head)
        if graph[c].label.rel == "conj" and graph[c].label.sub == "coord"
    ]


def _minimize(graph: Graph) -> bool:
    for c1 in graph.ids():
        conjuncts = _conjuncts(graph, c1)
        for c2 in conjuncts:
            # direct
            for d in graph.children(c1):
                label = graph[d].label
                if label.is_("conj:coord", "discourse", "parataxis", "punct") or d <= c2:
                    continue
                if any(c2 < c < d for c in conjuncts):
                    continue
                graph.move(d, c2)
                graph[d].shared = "Yes"
                return True
            # indirect through a mod or comp:obj dependent of each conjunct
            for first_test, second_test in (
                (lambda lab: lab.is_("mod"), lambda lab: lab.is_("mod")),
                (lambda lab: lab.is_("comp:obj"), lambda lab: lab.is_("comp:obj")),
            ):
                is_mod = first_test(Label("mod"))
                for d1 in graph.children(c1):
                    if not first_test(graph[d1].label) or not d1 < c2:
                        continue
                    for d2 in graph.children(c2):
                        if not second_test(graph[d2].label):
                            continue
                        if is_mod and not d2 < c2:
                            continue
                        for d in graph.children(d1):
                            label = graph[d].label
                            if label.is_("conj", "discourse", "parataxis", "punct") or d <= c2:
                                continue
                            if any(c2 < c < d for c in conjuncts):
                                continue
                            graph.move(d, d2)
                            graph[d].shared = "Yes"
                            return True
    return False


# add_conj_emb and chained_relations


def mark_embedded(graph: Graph) -> bool:
    for n2 in graph.ids():
        if not graph[n2].label.is_("conj:coord", "conj:coord@emb"):
            continue
        for n3 in graph.children(n2):
            if graph[n3].label.is_("conj:coord"):
                graph[n3].label = Label("conj", "coord", "emb")
                return True
    return False


def _chain(test: Callable[[Graph, int], bool]) -> Rule:
    def rule(graph: Graph) -> bool:
        for h in graph.ids():
            members = [c for c in graph.children(h) if test(graph, c)]
            if len(members) >= 2:
                graph.move(members[-1], members[-2])
                return True
        return False

    return rule


def _chained_relations() -> list[Rule]:
    def conj(g, c):
        return g[c].label.rel == "conj" and g[c].label.sub == "coord"

    def fixed(g, c):
        return g[c].label.is_("unk") and c in g.idiom_members

    def flat(g, c):
        return g[c].label.rel == "flat"

    return [_chain(conj), _chain(fixed), _chain(flat)]


# shared_left_conj-dep, unshared_left_conj-dep, unshared_right_conj-dep


def _coordinated(label: Label) -> bool:
    return label.rel == "conj" and label.sub == "coord"


def _links(graph: Graph, first_only: bool):
    """Yield `(c1, c2)` for every conjunct arc; with `first_only`, only from a first conjunct."""
    for c2 in graph.ids():
        c1 = graph[c2].head
        if c1 <= 0 or not _coordinated(graph[c2].label):
            continue
        if first_only and str(graph[c1].label) == str(graph[c2].label):
            continue
        yield c1, c2


def _pron_type(graph: Graph, index: int, *values: str) -> bool:
    return graph[index].feats.get("PronType") in values


def _set_shared(value: str, test: Callable[[Graph, int, int, int], bool], first_only=True) -> Rule:
    """A rule that sets `Shared=value` on a dependent `d` of `c1` where `test(g, c1, c2, d)`."""

    def rule(graph: Graph) -> bool:
        for c1, c2 in _links(graph, first_only):
            for d in graph.children(c1):
                if graph[d].shared == value or not test(graph, c1, c2, d):
                    continue
                graph[d].shared = value
                return True
        return False

    return rule


def _shared_rules() -> list[list[Rule]]:
    """The three feature packages, in strategy order."""

    def left_pp(g, conjunct):
        return any(
            g[x].label.rel in ("mod", "udep") and g.upos(x) in ("ADP", "SCONJ") and x < conjunct
            for x in g.children(conjunct)
        )

    def gen(g, c1, c2, d):
        rel = g[d].label.rel
        if rel not in ("cc", "mod", "parataxis", "udep", "subj"):
            return False
        if d in g.idiom_members or g[d].shared is not None or not d < c1:
            return False
        if rel == "subj" and any(g[x].label.rel == "subj" for x in g.children(c2)):
            return False
        if rel in ("mod", "udep"):
            return g.upos(d) in ("ADP", "SCONJ") and not left_pp(g, c2)
        return True

    def punct(g, c1, c2, d):
        if not g[d].label.is_("punct") or not d < c1 or g[d].shared is not None:
            return False
        return not any(g[p].label.is_("punct") and c1 < p < c2 for p in g.children(c1))

    def double_yes(g, c1, c2, d):
        if not d < c1 or g[d].shared is not None:
            return False
        return any(
            d < d1 < c1 and g.upos(d1) != "PUNCT" and g[d1].shared == "Yes" for d1 in g.children(c1)
        )

    def det1(g, c1, c2, d):
        return (
            g[d].label.rel == "det"
            and d not in g.idiom_members
            and d < c1
            and g.upos(c2) not in ("NOUN", "PROPN")
        )

    def det2(g, c1, c2, d):
        return (
            g[d].label.rel == "det"
            and d not in g.idiom_members
            and d < c1
            and any(g[x].label.rel == "det" for x in g.children(c2))
        )

    def pron_pers(g, c1, c2, d):
        return (
            g[d].label.rel in ("comp", "mod", "udep")
            and g.upos(d) == "PRON"
            and d not in g.idiom_members
            and _pron_type(g, d, "Prs")
            and d < c1
        )

    def int_rel(g, c1, c2, d):
        if not (g.upos(d) in ("ADV", "PRON") and _pron_type(g, d, "Int", "Rel") and d < c1):
            return False
        return any(
            g.upos(x) in ("ADV", "PRON") and _pron_type(g, x, "Int", "Rel") and x < c2
            for x in g.children(c2)
        )

    def spec(g, c1, c2, d):
        return g[d].label.rel in ("compound", "reparandum") and d < c1

    def subj(g, c1, c2, d):
        return (
            g[d].label.rel == "subj"
            and d not in g.idiom_members
            and d < c1
            and any(g[x].label.rel == "subj" for x in g.children(c2))
        )

    def punct_no(g, c1, c2, d):
        return (
            g[d].label.rel == "punct"
            and d < c1
            and any(g[p].label.rel == "punct" and c1 < p < c2 for p in g.children(c1))
        )

    def double_no(g, c1, c2, d):
        if g[d].label.is_("punct") or not d < c1:
            return False
        return any(d2 < d and g[d2].shared == "No" for d2 in g.children(c1))

    def right_no(graph: Graph) -> bool:
        for _, c2 in _links(graph, False):
            if any(_coordinated(graph[x].label) for x in graph.children(c2)):
                continue
            for d in graph.children(c2):
                if d > c2 and graph[d].shared is None:
                    graph[d].shared = "No"
                    return True
        return False

    return [
        [_set_shared("Yes", gen), _set_shared("Yes", punct), _set_shared("Yes", double_yes, False)],
        [
            _set_shared("No", det1),
            _set_shared("No", det2),
            _set_shared("No", pron_pers),
            _set_shared("No", int_rel),
            _set_shared("No", spec),
            _set_shared("No", subj),
            _set_shared("No", punct_no),
            _set_shared("No", double_no, False),
        ],
        [right_no],
    ]


# normalize


def normalize(graph: Graph) -> None:
    for d in graph.ids():
        label = graph[d].label
        if label.rel not in KEPT_RELATIONS:
            graph[d].label = Label("unk")
        elif label.rel == "subj" and label.sub == "outer":
            graph[d].label = Label("subj", None, label.deep)


# The whole conversion


def to_graph(
    forms: Sequence[str],
    lemmas: Sequence[str],
    upos: Sequence[str],
    feats: Sequence[str],
    heads: Sequence[int],
    deprels: Sequence[str],
) -> Graph:
    """Build a graph from one sentence's UD columns; `heads` number words from 1, 0 for root."""
    words = [
        Word(form, lemma, pos, parse_feats(feat), head, Label.parse(deprel))
        for form, lemma, pos, feat, head, deprel in zip(
            forms, lemmas, upos, feats, heads, deprels, strict=True
        )
    ]
    return Graph(words)


def convert(graph: Graph) -> Graph:
    """Convert a UD graph to SUD in place and return it."""
    gather_idioms(graph)
    normal_form(graph, _sub_relations())
    normal_form(graph, _rel_extensions())
    normal_form(graph, _relations())
    set_levels(graph)
    normal_form(graph, [reverse_edges])
    normal_form(graph, _shift_out_edges())
    rename_reversed(graph)
    normal_form(graph, [_minimize])
    normal_form(graph, [mark_embedded])
    normal_form(graph, _chained_relations())
    for package in _shared_rules():
        normal_form(graph, package)
    normalize(graph)
    return graph


def convert_sentence(
    forms: Sequence[str],
    lemmas: Sequence[str],
    upos: Sequence[str],
    feats: Sequence[str],
    heads: Sequence[int],
    deprels: Sequence[str],
) -> tuple[list[int], list[str]]:
    """Convert one UD sentence; return SUD heads (from 1, 0 for root) and relations."""
    heads, labels, _ = analyze_sentence(forms, lemmas, upos, feats, heads, deprels)
    return heads, labels


def analyze_sentence(
    forms: Sequence[str],
    lemmas: Sequence[str],
    upos: Sequence[str],
    feats: Sequence[str],
    heads: Sequence[int],
    deprels: Sequence[str],
) -> tuple[list[int], list[str], list[str | None]]:
    """Convert one UD sentence; return SUD heads, relations, and `Shared` values."""
    graph = convert(to_graph(forms, lemmas, upos, feats, heads, deprels))
    words = graph.words
    return [w.head for w in words], [str(w.label) for w in words], [w.shared for w in words]
