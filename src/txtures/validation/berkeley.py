"""The latent-variable PCFG trainer against the Berkeley parser's `GrammarTrainer`."""

from __future__ import annotations

import shutil
import time

import numpy as np

from txtures.lvpcfg import splitmerge as sm
from txtures.lvpcfg.em import iterate
from txtures.validation import BERKELEY_COMMIT, BERKELEY_REPOSITORY, DATA, Paths, dumps, run

TRAINING_SECTIONS = [f"{n:02d}" for n in range(2, 22)]
INPUT = "wsj02to21.gcg15.nol.prtrm"


def build(paths: Paths) -> None:
    """Clone the Berkeley parser at the pinned commit and compile the harnesses against its jar."""
    source = paths.berkeley / "src"
    if not source.is_dir():
        run(
            f"git clone -q {BERKELEY_REPOSITORY} {source} && git -C {source} checkout -q {BERKELEY_COMMIT}"
        )
    classes = paths.berkeley / "classes"
    if classes.exists():
        shutil.rmtree(classes)
    classes.mkdir()
    sources = " ".join(str(p) for p in (DATA / "java").rglob("*.java"))
    run(f"javac -nowarn -cp {paths.jar} -d {classes} {sources}")


def prepare(paths: Paths) -> None:
    """The trainer's input as ModelBlocks makes it: gcg15 WSJ 02-21, then `nol`, `prtrm`,
    `extrparplusops` (`resource-incrsem/Makefile`, `resource-lvpcfg/Makefile`), and the
    trees Berkeley trains on (`DumpTrees`)."""
    out = paths.berkeley
    parts = " ".join(str(paths.gcg15 / f"wsj{s}.gcg15.senttrees") for s in TRAINING_SECTIONS)
    run(
        f"""
        cd {out}
        cat {parts} > wsj02to21.gcg15.senttrees
        perl -pe 's/[-+]l.(?=[^\\)]* )//g;s/-x[^-}} ][^ |]*[|][^- ]*//g' < wsj02to21.gcg15.senttrees > wsj02to21.gcg15.nol.senttrees
        perl -pe 's/\\(([^\\(\\) ]*) ([^\\(\\) ]*)\\)/\\(\\1-PRTRM \\2\\)/g;s/\\([^ ]*,[^ -]*/\\(Pc/g;s/\\(:/\\(Pk/g' < wsj02to21.gcg15.nol.senttrees > {INPUT}.senttrees
        egrep '^\\(' {INPUT}.senttrees | sed 's/^\\((.*\\)$/(\\1)/g' | perl -pe 's/\\-(?=[a-z][^\\)]* )/+/g;s/\\-PRTRM/+PRTRM/g' > {INPUT}.extrparplusops.senttrees
        java -cp {paths.classpath} DumpTrees {INPUT}.extrparplusops.senttrees 2>/dev/null | grep '^(' > wsj02to21.berkeley-training-trees.txt
        """
    )


def train(paths: Paths, seed: int, cycles: int = 3) -> None:
    """`GrammarTrainer -SMcycles N -treebank SINGLEFILE` as ModelBlocks runs it, then the text dump."""
    run(
        f"""
        cd {paths.berkeley}
        java -Xmx20g -cp {paths.jar} edu.berkeley.nlp.PCFGLA.GrammarTrainer -SMcycles {cycles} \\
          -path {INPUT}.extrparplusops.senttrees -treebank SINGLEFILE -randSeed {seed} -out seed{seed}.{cycles}sm.gr
        java -Xmx20g -cp {paths.jar} edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile seed{seed}.{cycles}sm.gr seed{seed}.{cycles}sm
        """
    )


def dump(paths: Paths, mode: str, grammar: str) -> None:
    """Run a harness on a saved grammar: `onestep` (one EM iteration), `merge`, `split`
    (randomness 0), or `nodes` (inside and outside scores per node)."""
    out = paths.berkeley / "dumps"
    out.mkdir(exist_ok=True)
    trees = paths.berkeley / f"{INPUT}.extrparplusops.senttrees"
    source = paths.berkeley / grammar
    if mode == "onestep":
        command = f"edu.berkeley.nlp.PCFGLA.OneStep {trees} {source} {out / grammar}.before {out / grammar}.after"
    else:
        command = (
            f"edu.berkeley.nlp.PCFGLA.SplitMergeStep {mode} {trees} {source} {out / grammar}.{mode}"
        )
    run(f"java -Xmx20g -Dtrees=100000 -cp {paths.classpath} {command}")


def compare_onestep(paths: Paths, grammar: str) -> float:
    """txtures' EM iteration from a dumped grammar, against the Java iteration's result."""
    out = paths.berkeley / "dumps"
    before = dumps.load(out / f"{grammar}.before")
    corpus = dumps.training_corpus(paths.berkeley / f"{INPUT}.senttrees", before.categories)
    start = time.time()
    ll, g, lex = iterate(corpus, before.grammar(), before.lexicon())
    print(f"iteration {time.time() - start:.1f}s")
    after = dumps.load(out / f"{grammar}.after")
    print(f"log-likelihood txtures {ll!r} java {after.log_likelihood!r}")
    worst = abs(ll - after.log_likelihood) / abs(after.log_likelihood)
    trained = {k for k in g.unary}
    worst = max(
        worst,
        dumps.compare(g.binary, after.binary, "binary rules"),
        # Berkeley's rule lists also hold unary closures for parsing, which no tree uses
        dumps.compare(
            g.unary, {k: v for k, v in after.unary.items() if k in trained}, "unary rules"
        ),
        dumps.compare(dumps.flat(lex.word_tag), dumps.flat(after.counts["L"]), "word-tag counts"),
        dumps.compare(
            dumps.flat(lex.unseen_word_tag), dumps.flat(after.counts["LU"]), "signature counts"
        ),
        dumps.compare(lex.tag_count, after.counts["T"], "tag counts"),
        dumps.compare(lex.unseen_tag_count, after.counts["TU"], "unseen tag counts"),
        dumps.compare(
            {k: np.array([v]) for k, v in lex.word_count.items()},
            {k: np.array([v]) for k, v in after.counts["W"].items()},
            "word counts",
        ),
    )
    return worst


def compare_split_merge(paths: Paths, mode: str, grammar: str) -> None:
    """txtures' split (randomness 0) or merge from a dumped grammar, against the harness's."""
    out = paths.berkeley / "dumps"
    before = dumps.load(out / f"{grammar}.before")
    after = dumps.load(out / f"{grammar}.{mode}", out / f"{grammar}.{mode}.extra")
    grammar_, lexicon = before.grammar(), before.lexicon()
    if mode == "split":
        g, lex, trees = sm.split(
            grammar_, lexicon, before.split_trees, np.random.default_rng(0), 0.0
        )
        print("substates equal:", g.substates == after.substates)
        dumps.compare(g.binary, after.binary, "binary")
        dumps.compare(g.unary, {k: v for k, v in after.unary.items() if k in g.unary}, "unary")
        dumps.compare(dumps.flat(lex.word_tag), dumps.flat(after.counts["L"]), "word-tag counts")
        print("split trees equal:", trees == after.split_trees)
        return
    corpus = dumps.training_corpus(paths.berkeley / f"{INPUT}.senttrees", before.categories)
    stats = sm.merge_statistics(corpus, grammar_, lexicon)
    dumps.compare(stats.weights, after.merge_weights, "merge weights")
    deltas = {k: v for k, v in after.deltas.items() if v.size}
    big = {k: np.where(np.abs(v) >= 1e-6, v, 0) for k, v in deltas.items()}
    mine = {k: np.where(np.abs(stats.deltas[k]) >= 1e-6, stats.deltas[k], 0) for k in big}
    dumps.compare(mine, big, "merge losses of at least 1e-6")
    pairs = {c: [k for k, x in enumerate(v) if x] for c, v in after.pairs.items() if v.size}
    print(
        "Berkeley's rule on Java's losses gives Java's pairs:",
        sm.merge_pairs(deltas, rule="berkeley") == {c: pairs.get(c, []) for c in deltas},
    )
    g, lex, trees = sm.merge(grammar_, lexicon, before.split_trees, pairs, stats.weights)
    print("substates equal:", g.substates == after.substates)
    dumps.compare(g.binary, after.binary, "merged binary")
    dumps.compare(g.unary, {k: v for k, v in after.unary.items() if k in g.unary}, "merged unary")
    print("split trees equal:", trees == after.split_trees)
    ll, _, lex = iterate(corpus, g, lex, lexicon_only=True)
    print(f"lexicon pass log-likelihood txtures {ll!r} java {after.log_likelihood!r}")
    dumps.compare(
        dumps.flat(lex.word_tag), dumps.flat(after.counts["L"]), "retrained word-tag counts"
    )
