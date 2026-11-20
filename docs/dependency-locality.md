# Dependency locality

`dependency_locality` measures the integration cost of Dependency Locality
Theory (Gibson 2000) at each region, with the variants of Shain et al.
(2022). It reads the `parse` module's Universal Dependencies (UD) analysis,
converts each sentence to Surface-Syntactic UD (SUD), makes coordinating
conjunctions the heads of coordinations, and rewires relative clauses, as
described below. In SUD, adpositions, auxiliaries, copulas, and
subordinators head their phrases. The output column is `dlt`.

| Key | Default | Meaning |
| --- | --- | --- |
| `distance` | `referents` | `referents` counts discourse referents; `words` counts words |
| `verbs` | `false` | V: finite verbs weigh 2 and non-finite verbs 1; needs `distance: referents` |
| `coordination` | `false` | C: weigh coordinations by their heaviest conjunct |
| `modifiers` | `false` | M: ignore arcs to preceding modifiers |
| `transform` | `none` | `none`, or `log` for log(1 + cost) with the natural log |
| `relative_clauses` | `gibson` | `gibson` or `sud`; see [Relative clauses](#relative-clauses) |
| `coordination_head` | `conjunction` | `conjunction` or `conjunct`; see [Coordination](#coordination) |

Every key is sweepable. The variants of Shain et al. (2022) take two
configs, because V applies only to referents:

```yaml
# configs/dependency_locality/words.yml
distance: words
coordination: {plain: false, c: true}
modifiers: {plain: false, m: true}
```

```yaml
# configs/dependency_locality/referents.yml
verbs: {plain: false, v: true}
coordination: {plain: false, c: true}
modifiers: {plain: false, m: true}
```

```yaml
# workflows/dlt.yml
extends: main
dependency_locality: [words, referents]
```

This workflow gives 12 columns, such as `dlt_words_plain_m` and
`dlt_referents_v_c_m`.

## Integration cost

The definition follows the ModelBlocks implementation (`calcDLT.py`) that
Shain et al. (2022) used, applied to SUD trees.

A discourse referent is a noun or proper noun, or a finite verb or
auxiliary (`VerbForm=Fin`). Pronouns are not discourse referents. With
`distance: referents`, a referent weighs 1 and other words 0; with V, finite
verbs and auxiliaries weigh 2 and non-finite ones 1. With `distance: words`,
every word weighs 1. Punctuation always weighs 0.

The cost at a word *w* is the sum of:

- with `distance: referents`, the weight of *w* itself;
- for every arc between *w* and an earlier word *h*, whichever of the two is
  the head, the weight of the words strictly between *h* and *w*.

An arc is counted only at its later word, and arcs to punctuation are
ignored. An arc that ends at an object gap also counts *w*, because the gap
follows the verb (see [Relative clauses](#relative-clauses)). A region's cost
is the sum of its words' costs; `transform: log` is applied to that sum.

With M, an arc from *w* to an earlier word that modifies it (a SUD `mod`
relation) is ignored. Determiners are not modifiers.

With C, a coordination counts as its heaviest conjunct (see
[Coordination](#coordination) for what the conjuncts are). When an arc's
span reaches into a coordination that ends before *w*, the whole
coordination counts once, as the weight of its heaviest conjunct, with
nested coordinations counted the same way. When *w* lies in a conjunct of a
coordination that *h* precedes, the span skips the conjuncts before *w*'s
own.

## Coordination

With `coordination_head: conjunction`, the default, each coordinating
conjunction heads the structure before it and the conjunct after it, and
takes the coordination's place in the tree. In "Police arrested and
questioned the student", *and* heads *arrested* and *questioned*, and
*Police* and *student* depend on *and*. With `coordination_head: conjunct`,
txtures keeps SUD's analysis, in which the first conjunct heads a chain of
the others.

- Several conjunctions nest from the left: in "tea and coffee or juice",
  *or* heads *and* and *juice*, and *and* heads *tea* and *coffee*.
- A conjunct with no conjunction of its own attaches to the structure
  before it, as in SUD. In "apples, pears and plums", *and* heads *apples*
  and *plums*, and *pears* depends on *apples*. A coordination with no
  conjunction at all ("red, white, blue") keeps its first conjunct as head.
- A correlative such as *both* in "both John and Mary" depends on the
  conjunction.
- A dependent moves from a conjunct to the conjunction only when it is
  shared by the conjuncts. UD attaches shared dependents to the first
  conjunct without marking them, so txtures uses the rules of the official
  SUD conversion that set its `Shared` feature. A preceding subject is
  shared when the next conjunct has none, a preceding prepositional or
  subordinate phrase when the next conjunct has none of its own, and a
  dependent after the last conjunct that UD attached to the first is
  shared. Determiners and adjectives stay with their conjunct ("the old men
  and women"). These rules are heuristic, because UD does not record the
  difference.

With conjunction heads, a conjunct is the subtree of a conjunction's
conjunct dependent, and the coordination spans its conjuncts and
conjunction. In SUD's analysis, a conjunct is its head's subtree without
the next conjunct.

## Relative clauses

UD attaches a relative clause to the noun it modifies through the clause's
head, and the relative pronoun is a dependent of the verb only. Gibson's
analysis links the pronoun to the noun, and the verb to the pronoun through
a gap: after the verb for an object, before it for a subject. With
`relative_clauses: gibson`, the default, txtures rewrites each tree to that
analysis. With `relative_clauses: sud`, it uses the SUD tree unchanged.

The two can differ. In "The reporter who the senator from Ohio attacked
admitted the error", *attacked* costs 5 referents with `gibson`: 1 for
itself, 3 for the object gap back to *who* (*senator*, *Ohio*, and
*attacked*), and 1 for its subject *senator* (*Ohio*). With `sud` it costs
6, since its arcs reach both *reporter* and *who*. In Gibson's own example,
without *from Ohio*, both give his 3.

The rules find relative clauses on the UD tree, where they are marked, and
then change the SUD tree:

1. A relative clause is a word *C* attached by `acl:relcl`, together with its
   subtree. Its antecedent *N* is the head of *C*. A clause conjoined to it
   by `conj` ("who have been detained or who are at large") is attached to
   the first clause, not to *N*, and is left unchanged.
2. The clause's relative word *R* is the first word of the clause, in text
   order, that has `PronType=Rel` and is *C* or comes before it. Words in a
   nested relative clause or a conjoined clause are skipped, because they
   belong to that clause. Words after *C* are free relatives or question
   words ("who can manage to do what I want"), not the clause's relative
   word. A clause with no relative word ("the senator the reporter
   attacked") is left unchanged: its arc from *N* already links the verb to
   the antecedent, as Gibson's analysis does when there is no relative
   pronoun.
3. The clause's head *H* is the word of the clause attached to *N* after the
   SUD conversion and the coordination step. It differs from *C* when SUD
   promotes a function word, as in "who has attacked", where *H* is *has*,
   or when the clause is a coordination, as in "who attacked and insulted
   the senator", where *H* is *and*. If *H* is a conjunction, the clause is
   rewired only when *R* depends on *H*, that is, when the conjuncts share
   it.
4. The arc *N*–*H* is replaced by an arc *N*–*R* with the relative-clause
   relation. To keep a tree, the arcs on the path from *H* down to *R* are
   reversed, so *R* depends on *N* and heads the rest of the clause. If *R*
   is *H*, nothing changes.

Reversing an arc keeps its relation, and for the `modifiers` variant the arc
keeps the word that was originally the dependent as its modifier. Costs
depend only on which pairs of words are linked, so the reversal does not
change any cost.

If the relative word, or the phrase it belongs to, is the object of a verb
in the clause (SUD `comp:obj`), that arc ends at an object gap.

The relative word may be deep in the clause. In "the house in which I live",
*R* is *which* and SUD makes *in* its head, so *house* links to *which* and the
path *which*–*in*–*live* is reversed. In "the book that I wanted to read",
*that* is the object of *read*, so the path from *wanted* through *to* and
*read* to *that* is reversed. In "the man whose dog barked", *R* is *whose*
and the path *whose*–*dog*–*barked* is reversed.

These rules cover English. In the gold UD English Web Treebank (2,341
`acl:relcl` clauses), 910 clauses have no relative word, 1,424 have exactly
one, 2 have the relative word as *C*, and 5 have several, where the rule's
choice is the clause's relative word in each case. About 20 of the 182
conjoined clauses have their own relative word, which the rules leave
unattached to the antecedent. On parsed text, the result depends on the
parser's attachment of relative clauses and its `PronType` features.

## SUD conversion

The conversion is a port of the official Grew rules (`UD_to_SUD.grs` in
[surfacesyntacticud/conversions](https://github.com/surfacesyntacticud/conversions)),
limited to what changes heads and relations. Applied to the gold UD English
Web Treebank 2.19, it reproduces the official SUD English-EWT 2.19 for every
word but one of 254,820.

The official output has one property that follows from Grew rather than
from the written rules: two infinitives are never treated as having the
same `VerbForm`. In "You can tour Chernobyl and write your own story", SUD
therefore attaches *write* to *can*, not to *tour*. txtures reproduces this.

