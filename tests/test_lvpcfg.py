"""Tests for the latent-variable PCFG trainer.

`tests/data/lvpcfg/berkeley-trees.txt` was printed by the Berkeley parser
1.7 (`Corpus` with SINGLEFILE, then `Corpus.binarizeAndFilterTrees` with
vertical 1, horizontal 0, RIGHT binarization, as `GrammarTrainer` runs) from
`trees.txt` escaped as ModelBlocks does. `trees.txt` holds the hand-written
GCG test trees after ModelBlocks' `nol` and `prtrm` steps, and one tree that
exercises binarization, empty nodes, and X-over-X unaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from txtures.lvpcfg.trees import Tree, annotate, escape, normalize, parse, training_tree

DATA = Path(__file__).parent / "data" / "lvpcfg"


def test_training_trees_match_berkeley():
    lines = (DATA / "trees.txt").read_text().splitlines()
    expected = (DATA / "berkeley-trees.txt").read_text().splitlines()
    assert [str(training_tree(line)) for line in lines] == expected


def test_escape_keeps_operators_through_normalization():
    assert escape("(N-aD-PRTRM dog)") == "((N+aD+PRTRM dog))"
    assert str(normalize(parse("((NP-SBJ-1 (DT a)))"))) == "(ROOT (NP (DT a)))"


def test_binarization_is_a_left_branching_chain():
    tree = annotate(parse("((A (X a) (Y b) (Z c)))"))
    assert str(tree) == "(ROOT (A^g (@A^g (X a) (Y b)) (Z c)))"


def test_normalize_drops_empty_subtrees():
    assert normalize(parse("((S (-NONE- *T*)))")) is None
    assert normalize(Tree("S", [Tree("NP", [Tree("-NONE-", [Tree("*")])]), Tree("a")])) == Tree(
        "S", [Tree("a")]
    )


# Inside scores and expected counts, against brute-force enumeration


def _brute_force(trees, binary, unary, lexical):
    """Sum over every substate assignment of every node; return likelihoods and counts."""
    import itertools

    import numpy as np

    from txtures.lvpcfg.inside import LEXICAL, UNARY

    S = lexical.shape[1]
    likelihoods = []
    counts = [np.zeros_like(binary), np.zeros_like(unary), np.zeros_like(lexical)]
    for tree in trees:
        total, partial = 0.0, [np.zeros_like(binary), np.zeros_like(unary), np.zeros_like(lexical)]
        for assignment in itertools.product(range(S), repeat=len(tree)):
            if assignment[-1] != 0:  # the root has one substate
                continue
            weight, used = 1.0, []
            for position, (kind, index, left, right) in enumerate(tree):
                a = assignment[position]
                if kind == LEXICAL:
                    term, key = lexical[index, a], (2, (index, a))
                elif kind == UNARY:
                    term, key = unary[index, a, assignment[left]], (1, (index, a, assignment[left]))
                else:
                    b, c = assignment[left], assignment[right]
                    term, key = binary[index, a, b, c], (0, (index, a, b, c))
                weight *= term
                used.append(key)
            total += weight
            for which, key in used:
                partial[which][key] += weight
        likelihoods.append(total)
        for which in range(3):
            counts[which] += partial[which] / total
    return np.array(likelihoods), counts


def test_inside_and_counts_match_brute_force():
    import numpy as np
    import torch

    from txtures.lvpcfg.inside import (
        BINARY,
        LEXICAL,
        UNARY,
        encode,
        expected_counts,
        log_likelihoods,
    )

    rng = np.random.default_rng(0)
    S = 3
    binary, unary, lexical = rng.random((2, S, S, S)), rng.random((2, S, S)), rng.random((4, S))
    trees = [
        # ROOT -> X (unary 0); X -> a b (binary 0)
        [(LEXICAL, 0, -1, -1), (LEXICAL, 1, -1, -1), (BINARY, 0, 0, 1), (UNARY, 0, 2, -1)],
        # deeper: X -> (Y -> c d) e, then a unary chain to the root
        [
            (LEXICAL, 2, -1, -1),
            (LEXICAL, 3, -1, -1),
            (BINARY, 1, 0, 1),
            (LEXICAL, 0, -1, -1),
            (BINARY, 0, 2, 3),
            (UNARY, 1, 4, -1),
            (UNARY, 0, 5, -1),
        ],
    ]
    encoded = encode(trees)
    tensors = [torch.tensor(x, dtype=torch.float64) for x in (binary, unary, lexical)]
    expected_l, expected_c = _brute_force(trees, binary, unary, lexical)
    found = log_likelihoods(encoded, *tensors).numpy()
    np.testing.assert_allclose(found, np.log(expected_l), rtol=1e-12)
    total, *counts = expected_counts(encoded, *tensors)
    assert abs(float(total) - np.log(expected_l).sum()) < 1e-10
    for mine, reference in zip(counts, expected_c, strict=True):
        np.testing.assert_allclose(mine.numpy(), reference, rtol=1e-10, atol=1e-12)


def test_inside_does_not_underflow_on_long_trees():
    import torch

    from txtures.lvpcfg.inside import BINARY, LEXICAL, encode, log_likelihoods

    # a right-branching tree of 400 words, each with probability 1e-3
    nodes = [(LEXICAL, 0, -1, -1)]
    for _ in range(399):
        nodes.append((LEXICAL, 0, -1, -1))
        nodes.append((BINARY, 0, len(nodes) - 1, len(nodes) - 2))
    binary = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    lexical = torch.full((1, 1), 1e-3, dtype=torch.float64)
    found = log_likelihoods(
        encode([nodes]), binary, torch.ones(1, 1, 1, dtype=torch.float64), lexical
    )
    assert abs(float(found[0]) - 400 * torch.log(torch.tensor(1e-3, dtype=torch.float64))) < 1e-8


# Lexicon and M-step


def test_signatures_follow_berkeley_level_5():
    from txtures.lvpcfg.lexicon import signature

    def known(word):
        return word in {"running"}

    assert signature("Running", 0, known) == "UNK-INITC-KNOWNLC-ing"
    assert signature("Running", 3, known) == "UNK-CAPS-ing"
    assert signature("walked", 2, known) == "UNK-LC-ed"
    assert signature("1980s", 4, known) == "UNK-LC-NUM-s"  # the s is lowercase
    assert signature("class", 1, known) == "UNK-LC"  # -ss is not a plural
    assert signature("well-being", 1, known) == "UNK-LC-DASH"
    assert signature("IBM", 0, known) == "UNK-CAPS"


def test_rare_words_share_the_unseen_distribution():
    import numpy as np

    from txtures.lvpcfg.lexicon import Lexicon

    lexicon = Lexicon({0: 2})
    for word, weights in [("cat", [0.9, 0.1]), ("dog", [0.2, 0.8])]:
        lexicon.train_token(word, 0, 1, np.array(weights), previous_counts={})
    assert lexicon.total == 2 and lexicon.total_unseen == 2
    lexicon.tie_rare_words()
    np.testing.assert_allclose(lexicon.word_tag[0]["cat"], [0.55, 0.45])
    np.testing.assert_allclose(lexicon.word_tag[0]["dog"], [0.55, 0.45])
    # P(word | tag substate) by Bayes' rule, nearly the relative frequency without smoothing
    np.testing.assert_allclose(lexicon.score("cat", 0, 1, no_smoothing=True), [0.5, 0.5], rtol=1e-3)


def test_normalize_divides_by_parent_totals_and_prunes():
    import numpy as np

    from txtures.lvpcfg.em import normalize

    binary = {(0, 1, 2): np.array([[[3.0]], [[1e-40]]])}
    unary = {(0, 3): np.array([[1.0], [2.0]]), (0, 0): np.array([[5.0, 0.0], [0.0, 5.0]])}
    b, u = normalize(binary, unary)
    np.testing.assert_allclose(b[(0, 1, 2)].ravel(), [0.75, 0.0])  # 1e-40 / 2 is pruned
    np.testing.assert_allclose(u[(0, 3)].ravel(), [0.25, 1.0])
    np.testing.assert_allclose(
        u[(0, 0)], [[1.25, 0.0], [0.0, 2.5]]
    )  # self-loops are not in the totals


# Split, merge, and smoothing


def test_split_noise_keeps_parent_totals_and_is_small():
    import numpy as np

    from txtures.lvpcfg.em import Grammar
    from txtures.lvpcfg.lexicon import Lexicon
    from txtures.lvpcfg.splitmerge import split

    grammar = Grammar(
        {0: 1, 1: 1, 2: 1}, {(1, 2, 2): np.array([[[0.6]]])}, {(1, 2): np.array([[0.4]])}
    )
    lexicon = Lexicon({0: 1, 1: 1, 2: 1}, tag_count={2: np.array([10.0])})
    new, lex, trees = split(grammar, lexicon, {0: 0, 1: 0, 2: 0}, np.random.default_rng(1))
    assert new.substates == {0: 1, 1: 2, 2: 2}
    b, u = new.binary[(1, 2, 2)], new.unary[(1, 2)]
    # each parent substate keeps its old total, and each part is within 0.5% of an even share
    np.testing.assert_allclose(b.sum(axis=(1, 2)) + u.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(b, 0.15, rtol=0.01)
    np.testing.assert_allclose(lex.tag_count[2], [5.0, 5.0])
    assert trees[1] == [0, [0, 1]] and trees[0] == [0, [0]]


def test_smoother_mixes_within_the_first_split():
    import numpy as np

    from txtures.lvpcfg.splitmerge import smoother_matrix

    tree = [0, [[0, [0, 1]], [1, [2, 3]]]]
    d = smoother_matrix(tree, 0.1)
    np.testing.assert_allclose(d[:2, :2], [[0.9, 0.1], [0.1, 0.9]])
    assert d[0, 2] == 0 and d[3, 3] == 0.9


def test_merge_sums_children_and_averages_parents():
    import numpy as np

    from txtures.lvpcfg.em import Grammar
    from txtures.lvpcfg.lexicon import Lexicon
    from txtures.lvpcfg.splitmerge import merge

    binary = {(1, 1, 1): np.arange(8, dtype=float).reshape(2, 2, 2)}
    grammar = Grammar({0: 1, 1: 2}, binary, {})
    lexicon = Lexicon({0: 1, 1: 2}, tag_count={1: np.array([1.0, 3.0])})
    weights = {1: np.array([0.25, 0.75])}
    new, lex, trees = merge(grammar, lexicon, {1: [0, [0, 1]]}, {1: [0]}, weights)
    # children summed over both substates, parents averaged 1:3
    expected = 0.25 * binary[(1, 1, 1)][0].sum() + 0.75 * binary[(1, 1, 1)][1].sum()
    np.testing.assert_allclose(new.binary[(1, 1, 1)].ravel(), [expected])
    np.testing.assert_allclose(lex.tag_count[1], [4.0])
    assert trees[1] == [0, [0]]


def test_merge_rules():
    import numpy as np
    import pytest

    from txtures.lvpcfg.splitmerge import merge_pairs

    # category 1: two useless splits (one rounds to exactly 0, one to 1e-15);
    # category 2: three splits that matter
    deltas = {1: np.array([0.0, 1e-15]), 2: np.array([5.0, 0.2, 9.0])}
    # Berkeley: exact zeros are skipped; the threshold is sorted[⌊4/2⌋] of the 4 nonzero (5.0)
    assert merge_pairs(deltas, rule="berkeley") == {1: [1], 2: [0, 1]}
    # default: both useless splits count as zero; the threshold is sorted[⌊5/2⌋] of 5 (0.2)
    assert merge_pairs(deltas) == {1: [0, 1], 2: [1]}
    with pytest.raises(ValueError, match="merge rule"):
        merge_pairs(deltas, rule="other")


def test_training_runs_and_likelihood_improves():
    from txtures.lvpcfg.em import index_trees
    from txtures.lvpcfg.trainer import train
    from txtures.lvpcfg.trees import training_tree

    lines = (DATA / "trees.txt").read_text().splitlines()
    trees = [training_tree(line) for line in lines] * 3
    labels = {"ROOT": 0}
    for tree in trees:
        stack = [tree]
        while stack:
            t = stack.pop()
            if not t.is_leaf:
                labels.setdefault(t.label, len(labels))
                stack.extend(t.children)
    corpus = index_trees(trees, labels)
    grammar, lexicon, split_trees, log = train(corpus, labels, cycles=2, seed=0, updates=(4, 2, 2))
    assert [p["phase"] for p in log.phases] == ["split", "merge", "smooth"] * 2
    first, last = log.phases[0]["log_likelihoods"], log.phases[-1]["log_likelihoods"]
    assert last[-1] > first[0]
    assert grammar.substates[0] == 1 and max(grammar.substates.values()) <= 4
    assert set(split_trees) == set(labels.values())


def test_grammar_resource_builds_saves_and_loads(tmp_path, monkeypatch, capsys):
    import numpy as np
    from test_gcg import fake_ptb

    from txtures.cli import main
    from txtures.lvpcfg.resource import load

    for key in ("inputs", "outputs", "resources", "cache", "definitions"):
        monkeypatch.setenv(f"TXTURES_{key.upper()}", str(tmp_path / key))
    monkeypatch.setenv("TXTURES_CORPORA_PTB3", str(fake_ptb(tmp_path / "ptb")))
    monkeypatch.setenv("TXTURES_EXECUTION_EXECUTOR", "local")
    monkeypatch.setenv("TXTURES_EXECUTION_DEVICE", "cpu")
    args = ["resources", "build", "--kind", "lvpcfg_grammar"]
    args += [
        "--spec",
        "treebank={sections: [2]}",
        "--spec",
        "cycles=1",
        "--spec",
        "updates=[2, 1, 1]",
    ]
    assert main(args) == 0
    assert "built resource lvpcfg_grammar" in capsys.readouterr().out
    [entry] = list((tmp_path / "resources" / "store" / "lvpcfg_grammar").glob("*/*"))
    trained = load(entry)
    assert trained.training["trees"] == 7  # the FAIL tree is dropped
    assert [p["phase"] for p in trained.training["phases"]] == ["split", "merge", "smooth"]
    rule = next(iter(trained.grammar.binary.values()))
    assert np.all(np.isfinite(rule)) and trained.grammar.smoother is not None
    assert trained.lexicon.total == 46  # the 49 words less the 3 of the FAIL tree


def _toy_training(cycles=1, updates=(2, 1, 1)):
    from txtures.lvpcfg.em import index_trees
    from txtures.lvpcfg.resource import category_numbers
    from txtures.lvpcfg.trainer import train
    from txtures.lvpcfg.trees import training_tree

    trees = [training_tree(line) for line in (DATA / "trees.txt").read_text().splitlines()] * 3
    labels = category_numbers(trees)
    corpus = index_trees(trees, labels)
    grammar, lexicon, split_trees, _ = train(corpus, labels, cycles=cycles, seed=0, updates=updates)
    return corpus, grammar, lexicon, split_trees


def test_tensor_iteration_matches_the_dictionary_one():
    import numpy as np

    from txtures.lvpcfg import fast
    from txtures.lvpcfg import splitmerge as sm
    from txtures.lvpcfg.em import iterate

    corpus, grammar, lexicon, split_trees = _toy_training()
    grammar, lexicon, split_trees = sm.split(
        grammar, lexicon, split_trees, np.random.default_rng(3)
    )
    for smoothing in (False, True):
        if smoothing:
            grammar.smoother = sm.smoothers(split_trees, 0.01)
            lexicon.smoother = sm.smoothers(split_trees, 0.1)
        ll, g_dict, l_dict = iterate(corpus, grammar, lexicon)
        state = fast.to_state(corpus, grammar, lexicon)
        ll_fast, new = fast.iterate(corpus, state)
        g_fast, l_fast = fast.to_dicts(corpus, new, grammar, lexicon)
        assert abs(ll - ll_fast) < 1e-9 * abs(ll)
        for key, value in g_dict.binary.items():
            np.testing.assert_allclose(g_fast.binary[key], value, rtol=1e-10, atol=1e-14)
        for key, value in g_dict.unary.items():
            np.testing.assert_allclose(g_fast.unary[key], value, rtol=1e-10, atol=1e-14)
        for tag, words in l_dict.word_tag.items():
            for word, value in words.items():
                np.testing.assert_allclose(
                    l_fast.word_tag[tag][word], value, rtol=1e-10, atol=1e-14
                )
        for tag, value in l_dict.unseen_tag_count.items():
            np.testing.assert_allclose(l_fast.unseen_tag_count[tag], value, rtol=1e-10, atol=1e-14)
        for word, count in l_dict.word_count.items():
            assert abs(l_fast.word_count[word] - count) < 1e-10
        assert abs(l_fast.total - l_dict.total) < 1e-9


def test_tensor_split_matches_the_dictionary_one_without_noise():
    import numpy as np
    import torch

    from txtures.lvpcfg import fast
    from txtures.lvpcfg import splitmerge as sm

    corpus, grammar, lexicon, split_trees = _toy_training()
    g_dict, l_dict, _ = sm.split(grammar, lexicon, split_trees, np.random.default_rng(0), 0.0)
    state = fast.split(
        fast.to_state(corpus, grammar, lexicon), torch.Generator().manual_seed(0), 0.0
    )
    g_fast, l_fast = fast.to_dicts(corpus, state, g_dict, l_dict)
    assert state.substates == g_dict.substates
    for key, value in g_dict.binary.items():
        np.testing.assert_allclose(g_fast.binary[key], value, rtol=1e-12, atol=1e-15)
    for key, value in g_dict.unary.items():
        np.testing.assert_allclose(g_fast.unary[key], value, rtol=1e-12, atol=1e-15)
    for tag, value in l_dict.tag_count.items():
        np.testing.assert_allclose(l_fast.tag_count[tag], value)


def test_tensor_split_noise_keeps_totals():
    import torch

    from txtures.lvpcfg import fast

    corpus, grammar, lexicon, _ = _toy_training()
    state = fast.to_state(corpus, grammar, lexicon)
    new = fast.split(state, torch.Generator().manual_seed(1))
    old_totals = state.binary.sum(dim=(2, 3))
    new_totals = new.binary.sum(dim=(2, 3))
    # each new parent substate keeps the old one's binary total; no part moves by more than 0.5%
    for r in range(state.binary.shape[0]):
        n = new.substates[int(state.binary_parent[r])]
        o = state.substates[int(state.binary_parent[r])]
        expected = old_totals[r, :o].repeat_interleave(2)[:n] if n > o else old_totals[r, :o]
        torch.testing.assert_close(new_totals[r, :n], expected)


def test_tensor_merge_matches_the_dictionary_one():
    import numpy as np

    from txtures.lvpcfg import fast
    from txtures.lvpcfg import splitmerge as sm

    corpus, grammar, lexicon, split_trees = _toy_training()
    grammar, lexicon, split_trees = sm.split(
        grammar, lexicon, split_trees, np.random.default_rng(5)
    )
    stats = sm.merge_statistics(corpus, grammar, lexicon)
    state = fast.to_state(corpus, grammar, lexicon)
    weights, deltas = fast.merge_statistics(corpus, state)
    for c, value in stats.weights.items():
        np.testing.assert_allclose(weights[c], value, rtol=1e-10, atol=1e-14)
    for c, value in stats.deltas.items():
        np.testing.assert_allclose(deltas[c], value, rtol=1e-8, atol=1e-10)
    pairs = sm.merge_pairs(stats.deltas)
    g_dict, l_dict, _ = sm.merge(grammar, lexicon, split_trees, pairs, stats.weights)
    merged = fast.merge(corpus, state, pairs, stats.weights)
    assert merged.substates == g_dict.substates
    g_fast, l_fast = fast.to_dicts(corpus, merged, g_dict, l_dict)
    for key, value in g_dict.binary.items():
        np.testing.assert_allclose(g_fast.binary[key], value, rtol=1e-12, atol=1e-15)
    for key, value in g_dict.unary.items():
        np.testing.assert_allclose(g_fast.unary[key], value, rtol=1e-12, atol=1e-15)
    for tag, words in l_dict.word_tag.items():
        for word, value in words.items():
            np.testing.assert_allclose(l_fast.word_tag[tag][word], value, rtol=1e-12, atol=1e-15)


# Left-corner compilation


def _toy_pcfg():
    from txtures.lvpcfg.leftcorner import Pcfg

    # phrases S, NP, VP; preterminals d, n, v, x (as GCG's -PRTRM tags are distinct)
    grammar = [
        "ROOT_0 -> S_0 1.0",
        "S_0 -> NP_0 VP_0 0.7",
        "S_0 -> S_0 x_0 0.3",
        "NP_0 -> d_0 NP_0 0.4",
        "NP_0 -> n_0 0.6",
        "VP_0 -> VP_0 NP_0 0.5",
        "VP_0 -> v_0 0.5",
    ]
    lexicon = ["n dogs [1.0]", "d the [1.0]", "v bark [1.0]", "x . [1.0]"]
    return Pcfg.from_berkeley_text(grammar, lexicon)


def test_left_corner_tables_are_distributions():
    import torch

    from txtures.lvpcfg import leftcorner as lc

    pcfg = _toy_pcfg()
    assert pcfg.root == {"S_0": 1.0} and pcfg.terminal["n_0"] == 1.0
    model = lc.compile_model(pcfg.without_unaries())
    # delta is a probability of expanding within the bound, and grows with room to expand
    assert torch.all(model.delta <= 1 + 1e-12)
    s = model.index["S_0"]
    assert model.delta[4, lc.LEFT, s] < model.delta[1, lc.LEFT, s]
    for d in range(lc.MAX_DEPTH + 1):
        f0, f1 = lc.f_table(model, d)
        j0, j1 = lc.j_table(model, d)
        assert torch.all((f0 >= 0) & (f1 >= 0)) and torch.all((j0 >= 0) & (j1 >= 0))
    # T forks to its left descendants (f = 1) at depth 0: the sentence's first word
    f0, f1 = lc.f_table(model, 0)
    assert f1[model.index["T"]] > 0 and f0[model.index["T"]] == 0
    for table in (lc.b_conditional(model, 1, lc.LEFT), lc.b_conditional(model, 1, lc.RIGHT)):
        for row in table.values():
            assert abs(sum(row.values()) - 1) < 1e-12


def test_viterbi_unary_closure_takes_the_best_path_of_at_most_two_steps():
    import numpy as np

    from txtures.lvpcfg.leftcorner import viterbi_unary_closure

    unary = {
        (1, 2): np.array([[0.1]]),  # direct A -> B
        (1, 3): np.array([[0.5]]),  # A -> C
        (3, 2): np.array([[0.6]]),  # C -> B: A -> C -> B is 0.3, better than 0.1
        (2, 4): np.array([[0.9]]),  # B -> D
    }
    closed = viterbi_unary_closure(unary, {1: 1, 2: 1, 3: 1, 4: 1})
    assert closed[(1, 2)][0, 0] == 0.3
    assert closed[(1, 3)][0, 0] == 0.5 and closed[(3, 2)][0, 0] == 0.6
    # A -> B -> D (0.09) counts; A -> C -> B -> D (0.27) is three steps and does not
    np.testing.assert_allclose(closed[(1, 4)], [[0.1 * 0.9]])
    np.testing.assert_allclose(closed[(3, 4)], [[0.6 * 0.9]])


def test_trained_grammar_exports_to_a_compilable_pcfg():
    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.resource import category_numbers
    from txtures.lvpcfg.trees import training_tree

    corpus, grammar, lexicon, _ = _toy_training()
    trees = [training_tree(line) for line in (DATA / "trees.txt").read_text().splitlines()] * 3
    pcfg = lc.pcfg_from_trained(category_numbers(trees), grammar, lexicon)
    assert pcfg.root and pcfg.binary and pcfg.lexicon
    assert all(not name.endswith("^g") and "+" not in name for name in pcfg.binary)
    model = lc.compile_model(pcfg.without_unaries())
    f0, f1 = lc.f_table(model, 0)
    assert f1[model.index["T"]] > 0


def test_synproc_model_stores_the_pcfg_and_compiles_on_load(tmp_path):
    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.resource import load_pcfg, save_pcfg

    pcfg = _toy_pcfg()
    save_pcfg(tmp_path, pcfg)
    loaded = load_pcfg(tmp_path)
    assert loaded.binary == pcfg.binary and loaded.unary == pcfg.unary
    assert loaded.root == pcfg.root and loaded.lexicon == pcfg.lexicon
    assert loaded.terminal == pcfg.terminal
    a, b = lc.compile_model(pcfg.without_unaries()), lc.compile_model(loaded.without_unaries())
    assert a.symbols == b.symbols and (a.plus == b.plus).all()


# The synproc reference parser


def _binary_toy_pcfg():
    from txtures.lvpcfg.leftcorner import Pcfg

    grammar = [
        "ROOT_0 -> S_0 1.0",
        "S_0 -> NP_0 VP_0 0.7",
        "S_0 -> S_0 x_0 0.3",
        "NP_0 -> d_0 n_0 0.4",
        "NP_0 -> a_0 n_0 0.6",
        "VP_0 -> v_0 NP_0 0.5",
        "VP_0 -> v_0 x_0 0.5",
    ]
    lexicon = [
        "d the [1.0]",
        "n dogs [0.5]",
        "n cats [0.5]",
        "a big [1.0]",
        "v see [1.0]",
        "x . [1.0]",
    ]
    return Pcfg.from_berkeley_text(grammar, lexicon)


def _inside(pcfg, words):
    """CKY inside probability of a sentence under a binary PCFG with lexicon (brute force)."""
    from collections import defaultdict

    n = len(words)
    chart = defaultdict(float)
    for i, word in enumerate(words):
        for tag, entries in pcfg.lexicon.items():
            if word in entries:
                chart[(i, i + 1, tag)] += pcfg.terminal.get(tag, 0) * entries[word]
    for width in range(2, n + 1):
        for i in range(n - width + 1):
            k = i + width
            for parent, rules in pcfg.binary.items():
                for (left, right), p in rules.items():
                    for m in range(i + 1, k):
                        chart[(i, k, parent)] += p * chart[(i, m, left)] * chart[(m, k, right)]
    return sum(p * chart[(0, n, c)] for c, p in pcfg.root.items())


def test_reference_parser_sums_to_the_pcfg_inside_probability():
    import math

    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.synproc import Tables, reference_parse

    pcfg = _binary_toy_pcfg().without_unaries()
    tables = Tables(lc.compile_model(pcfg))
    for sentence in ("the dogs see big cats .", "big cats see the dogs", "the cats see ."):
        words = sentence.split()
        inside = _inside(pcfg, words)
        assert inside > 0
        parse = reference_parse(tables, words, beam=10**9, recombine="sum")
        # each word's surprisal is -log2 P(w_t | w_<t); they sum to -log2 P(sentence)
        assert abs(sum(parse.surprisal) + math.log2(inside)) < 1e-9, sentence


def test_reference_parser_finds_the_best_sequence():
    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.synproc import Tables, reference_parse

    tables = Tables(lc.compile_model(_binary_toy_pcfg().without_unaries()))
    words = "the dogs see big cats .".split()
    parse = reference_parse(tables, words, beam=50)
    assert all(state is not None for state in parse.states)
    tag, fork, join, store = parse.states[-1]
    assert len(store) + fork - join == 0  # the last word's join empties the store
    assert [tables.symbols[s[0]] for s in parse.states] == [
        "d_0",
        "n_0",
        "v_0",
        "a_0",
        "n_0",
        "x_0",
    ]


@pytest.mark.parametrize("beam", [3, 50])
def test_batched_parser_matches_the_reference(beam):
    import math

    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.parser import Parser
    from txtures.lvpcfg.synproc import Tables, reference_parse

    for pcfg in (_binary_toy_pcfg(), _toy_pcfg()):
        model = lc.compile_model(pcfg.without_unaries())
        tables = Tables(model)
        sentences = [
            s.split()
            for s in (
                "the dogs see big cats .",
                "big cats see the dogs",
                "the cats see .",
                "dogs bark .",
                "the dogs bark",
            )
        ]
        sentences = [s for s in sentences if all(tables.words(w) for w in s)]
        results = Parser(model, beam=beam, working=2).parse(sentences)
        for words, result in zip(sentences, results, strict=True):
            expected = reference_parse(tables, words, beam=beam)
            for a, b in zip(result["surprisal"], expected.surprisal, strict=True):
                assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9, (words, a, b)
            assert result["states"] == expected.states


def test_lexicon_merge_once_and_as_synproc_does():
    from txtures.lvpcfg.synproc import MergedLexicon, berkeley_signature

    assert berkeley_signature("market") == "UNK-LC"
    assert berkeley_signature("UNK-LC") == "UNK-CAPS-DASH"
    assert berkeley_signature("UNK-CAPS-DASH") == "UNK-CAPS-DASH"  # its own signature
    lexicon = {
        "a": {"market": 0.5, "UNK-LC": 0.1},
        "b": {"UNK-CAPS-DASH": 0.2, "dog": 0.3},
    }
    index = {"a": 0, "b": 1}
    once = MergedLexicon(lexicon, index, "once")
    assert once.entries("market") == {0: 0.5 + 0.1}
    assert once.entries("unknown") == {0: 0.1}  # an unknown word takes its signature's
    synproc = MergedLexicon(lexicon, index, "synproc")
    # map order (sorted, uppercase first): UNK-LC, market (from a), then UNK-CAPS-DASH, dog
    # UNK-LC comes before UNK-CAPS-DASH, so it gets that signature's entries unmerged
    assert synproc.entries("unknown") == {0: 0.1, 1: 0.2}
    # market comes after UNK-LC, so it gets UNK-LC's merged entries: the cascade
    assert synproc.entries("market") == {0: 0.5 + 0.1, 1: 0.2}
    # UNK-CAPS-DASH is its own signature and doubles
    assert synproc.merged("UNK-CAPS-DASH") == {1: 0.4}


@pytest.mark.parametrize(("max_sentences", "window"), [(1, 4096), (2, 4096), (2, 1)])
def test_continuous_batching_keeps_results_and_order(max_sentences, window):
    import math

    from txtures.lvpcfg import leftcorner as lc
    from txtures.lvpcfg.parser import Parser
    from txtures.lvpcfg.synproc import Tables, reference_parse

    model = lc.compile_model(_binary_toy_pcfg().without_unaries())
    tables = Tables(model)
    sentences = [
        s.split()
        for s in (
            "the dogs see big cats .",
            "big cats see",
            "",
            "the cats see .",
            "big dogs see the cats .",
        )
    ]
    parser = Parser(model, beam=20, working=4)
    streamed = list(parser.parse_stream(sentences, max_sentences=max_sentences, window=window))
    assert [index for index, _ in streamed] == list(range(len(sentences)))
    for words, (_, result) in zip(sentences, streamed, strict=True):
        if not words:
            assert result == {"surprisal": [], "states": []}
            continue
        expected = reference_parse(tables, words, beam=20)
        for a, b in zip(result["surprisal"], expected.surprisal, strict=True):
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9
        assert result["states"] == expected.states
