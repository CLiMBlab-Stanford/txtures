"""The `lvpcfg_grammar` resource: a latent-variable PCFG trained on a treebank.

It depends on a `gcg_treebank`. Its trees go through ModelBlocks' `nol` and
`prtrm` steps (operators and morphology tags removed, preterminals marked)
and Berkeley's preparation, then `txtures.lvpcfg.trainer.train`.

Spec keys: `treebank` (the `gcg_treebank` spec), `cycles` (3), `seed` (2),
`merge_rule` (`negligible` or `berkeley`), `fail_trees` (`drop`, the
default, or `keep`, as ModelBlocks trains), and `updates` (`[49, 19, 9]`, the
EM updates of the split, merge, and smooth phases).

Stored files (no pickles): `grammar.json` (categories, substates, split
trees, smoothing), `rules.npz` (rule keys and their substate weights),
`lexicon.npz` and `lexicon.json` (the lexicon's counts), and `training.json`
(the log-likelihood of every EM iteration).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from txtures.errors import TxturesError
from txtures.modules.base import Resource, RunContext, Workload
from txtures.store import write_json

LVPCFG_VERSION = 2
"""Increment whenever the same spec and treebank would train a different grammar."""

DEFAULTS = {
    "cycles": 3,
    "seed": 2,
    "merge_rule": "negligible",
    "fail_trees": "drop",
    "updates": [49, 19, 9],
}


@dataclass
class TrainedGrammar:
    """A loaded `lvpcfg_grammar`."""

    categories: dict[str, int]
    grammar: Any  # txtures.lvpcfg.em.Grammar
    lexicon: Any  # txtures.lvpcfg.lexicon.Lexicon
    split_trees: dict[int, Any]
    training: dict[str, Any]


class LvpcfgGrammar(Resource):
    """A split-merge latent-variable PCFG trained on a GCG treebank."""

    kind = "lvpcfg_grammar"
    version = 1

    def dependencies(self) -> Mapping[str, Resource]:
        from txtures.gcg.treebank import GcgTreebank

        return {"treebank": GcgTreebank(self.spec.get("treebank", {}))}

    def resolve(self, context: RunContext) -> dict[str, Any]:
        spec = {**DEFAULTS, **{k: v for k, v in self.spec.items() if k != "treebank"}}
        if spec["merge_rule"] not in ("negligible", "berkeley"):
            raise TxturesError(
                f"merge_rule must be negligible or berkeley, not {spec['merge_rule']!r}"
            )
        if spec["fail_trees"] not in ("drop", "keep"):
            raise TxturesError(f"fail_trees must be drop or keep, not {spec['fail_trees']!r}")
        spec["updates"] = [int(u) for u in spec["updates"]]
        spec["trainer_version"] = LVPCFG_VERSION
        return spec

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=4, memory_gb=48, time_hours=8, ngpus=1)

    def build(self, resolved, directory: Path, context: RunContext, dependencies) -> None:
        from txtures.gcg import mark_preterminals, remove_operators
        from txtures.gcg.treebank import read_trees
        from txtures.lvpcfg.em import index_trees
        from txtures.lvpcfg.trainer import train
        from txtures.lvpcfg.trees import training_tree

        lines = [
            mark_preterminals(remove_operators(line))
            for line in read_trees(dependencies["treebank"])
            if resolved["fail_trees"] == "keep" or not line.startswith("(FAIL ")
        ]
        trees = [t for t in (training_tree(line) for line in lines) if t is not None]
        categories = category_numbers(trees)
        corpus = index_trees(trees, categories)
        grammar, lexicon, split_trees, log = train(
            corpus,
            categories,
            cycles=resolved["cycles"],
            seed=resolved["seed"],
            merge_rule=resolved["merge_rule"],
            updates=tuple(resolved["updates"]),
            device=_device(context),
        )
        save(
            directory,
            categories,
            grammar,
            lexicon,
            split_trees,
            {"trees": len(trees), **{"phases": log.phases}},
        )

    def load(self, resolved, directory: Path, context: RunContext, dependencies) -> TrainedGrammar:
        return load(directory)


def category_numbers(trees) -> dict[str, int]:
    """Number categories in order of first appearance (preorder), `ROOT` first."""
    numbers = {"ROOT": 0}
    for tree in trees:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                continue
            numbers.setdefault(node.label, len(numbers))
            stack.extend(reversed(node.children))
    return numbers


def _device(context: RunContext) -> str:
    import torch

    if context.device not in ("auto", None):
        return context.device
    return "cuda" if torch.cuda.is_available() else "cpu"


# Storage


def _ragged(table: Mapping[tuple, np.ndarray]) -> dict[str, np.ndarray]:
    keys = sorted(table)
    values = [np.asarray(table[k], dtype=np.float64) for k in keys]
    return {
        "keys": np.array(keys, dtype=np.int64).reshape(len(keys), -1),
        "values": np.concatenate([v.ravel() for v in values]) if values else np.zeros(0),
        "offsets": np.cumsum([0] + [v.size for v in values]),
    }


def _unragged(data, prefix: str, shape) -> dict[tuple, np.ndarray]:
    keys, values, offsets = data[prefix + "keys"], data[prefix + "values"], data[prefix + "offsets"]
    out = {}
    for i, key in enumerate(keys):
        key = tuple(int(k) for k in key)
        out[key] = values[offsets[i] : offsets[i + 1]].reshape(shape(key))
    return out


def save(directory: Path, categories, grammar, lexicon, split_trees, training) -> None:
    write_json(
        directory / "grammar.json",
        {
            "categories": categories,
            "substates": {str(c): n for c, n in grammar.substates.items()},
            "split_trees": {str(c): t for c, t in split_trees.items()},
            "smoothed": grammar.smoother is not None,
        },
    )
    arrays = {}
    for name, table in (("binary_", grammar.binary), ("unary_", grammar.unary)):
        arrays.update({name + k: v for k, v in _ragged(table).items()})
    np.savez_compressed(directory / "rules.npz", **arrays)
    words: list[str] = sorted(
        {w for m in lexicon.word_tag.values() for w in m} | set(lexicon.word_count)
    )
    sigs = sorted({s for m in lexicon.unseen_word_tag.values() for s in m})
    word_ids = {w: i for i, w in enumerate(words)}
    sig_ids = {s: i for i, s in enumerate(sigs)}
    lex = {}
    lex.update(
        {
            "seen_" + k: v
            for k, v in _ragged(
                {(t, word_ids[w]): v for t, m in lexicon.word_tag.items() for w, v in m.items()}
            ).items()
        }
    )
    lex.update(
        {
            "unseen_" + k: v
            for k, v in _ragged(
                {
                    (t, sig_ids[s]): v
                    for t, m in lexicon.unseen_word_tag.items()
                    for s, v in m.items()
                }
            ).items()
        }
    )
    lex.update(
        {"tag_" + k: v for k, v in _ragged({(t,): v for t, v in lexicon.tag_count.items()}).items()}
    )
    lex.update(
        {
            "unseentag_" + k: v
            for k, v in _ragged({(t,): v for t, v in lexicon.unseen_tag_count.items()}).items()
        }
    )
    lex["word_count"] = np.array([lexicon.word_count.get(w, 0.0) for w in words])
    np.savez_compressed(directory / "lexicon.npz", **lex)
    write_json(
        directory / "lexicon.json",
        {
            "words": words,
            "signatures": sigs,
            "total": lexicon.total,
            "total_unseen": lexicon.total_unseen,
        },
    )
    write_json(directory / "training.json", training)


def load(directory: Path) -> TrainedGrammar:
    from txtures.lvpcfg import splitmerge as sm
    from txtures.lvpcfg.em import Grammar
    from txtures.lvpcfg.lexicon import Lexicon

    meta = json.loads((directory / "grammar.json").read_text(encoding="utf-8"))
    substates = {int(c): n for c, n in meta["substates"].items()}
    split_trees = {int(c): t for c, t in meta["split_trees"].items()}
    rules = np.load(directory / "rules.npz")
    binary = _unragged(
        rules, "binary_", lambda k: (substates[k[0]], substates[k[1]], substates[k[2]])
    )
    unary = _unragged(rules, "unary_", lambda k: (substates[k[0]], substates[k[1]]))
    grammar_smoother = sm.smoothers(split_trees, 0.01) if meta["smoothed"] else None
    lexicon_smoother = sm.smoothers(split_trees, 0.1) if meta["smoothed"] else None
    grammar = Grammar(substates, binary, unary, grammar_smoother)
    lex = np.load(directory / "lexicon.npz")
    info = json.loads((directory / "lexicon.json").read_text(encoding="utf-8"))
    words, sigs = info["words"], info["signatures"]
    lexicon = Lexicon(substates, lexicon_smoother)
    for (t, w), v in _unragged(lex, "seen_", lambda k: (substates[k[0]],)).items():
        lexicon.word_tag.setdefault(t, {})[words[w]] = v
    for (t, s), v in _unragged(lex, "unseen_", lambda k: (substates[k[0]],)).items():
        lexicon.unseen_word_tag.setdefault(t, {})[sigs[s]] = v
    lexicon.tag_count = {
        k[0]: v for k, v in _unragged(lex, "tag_", lambda k: (substates[k[0]],)).items()
    }
    lexicon.unseen_tag_count = {
        k[0]: v for k, v in _unragged(lex, "unseentag_", lambda k: (substates[k[0]],)).items()
    }
    lexicon.word_count = {w: float(c) for w, c in zip(words, lex["word_count"], strict=True)}
    lexicon.total, lexicon.total_unseen = info["total"], info["total_unseen"]
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    return TrainedGrammar(meta["categories"], grammar, lexicon, split_trees, training)
