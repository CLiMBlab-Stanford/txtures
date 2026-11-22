"""The `lvpcfg_grammar` resource: a latent-variable PCFG trained on a treebank.

It depends on a `gcg_treebank`. Its trees go through ModelBlocks' `nol` and
`prtrm` steps (operators and morphology tags removed, preterminals marked)
and Berkeley's preparation (`prsr.training_tree`), then `prsr.train`.

Spec keys: `treebank` (the `gcg_treebank` spec), `cycles` (3), `seed` (2),
`merge_rule` (`negligible` or `berkeley`), `fail_trees` (`drop`, the
default, or `keep`, as ModelBlocks trains), and `updates` (`[49, 19, 9]`, the
EM updates of the split, merge, and smooth phases).

Entries hold the files of `prsr.Grammar.save`. The identity includes
`prsr.TRAIN_VERSION`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import prsr
from prsr import Grammar, Pcfg

from txtures.errors import TxturesError
from txtures.modules.base import Resource, RunContext, Workload

DEFAULTS = {
    "cycles": 3,
    "seed": 2,
    "merge_rule": "negligible",
    "fail_trees": "drop",
    "updates": [49, 19, 9],
}


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
        spec["trainer_version"] = prsr.TRAIN_VERSION
        return spec

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=4, memory_gb=48, time_hours=8, ngpus=1)

    def build(self, resolved, directory: Path, context: RunContext, dependencies) -> None:
        from prsr.gcg import training_lines

        from txtures.gcg.treebank import read_trees

        lines = training_lines(read_trees(dependencies["treebank"]), resolved["fail_trees"])
        trees = [t for t in map(prsr.training_tree, lines) if t is not None]
        grammar = prsr.train(
            trees,
            cycles=resolved["cycles"],
            seed=resolved["seed"],
            merge_rule=resolved["merge_rule"],
            updates=tuple(resolved["updates"]),
            device=_device(context),
            progress=lambda message: print(message, flush=True),
        )
        grammar.save(directory)

    def load(self, resolved, directory: Path, context: RunContext, dependencies) -> Grammar:
        return Grammar.load(directory)


def _device(context: RunContext) -> str:
    import torch

    if context.device not in ("auto", None):
        return context.device
    return "cuda" if torch.cuda.is_available() else "cpu"


# The compiled left-corner model


@dataclass
class LoadedSynprocModel:
    """A loaded `synproc_model`: the exported PCFG and its compiled left-corner factors."""

    pcfg: Any  # prsr.leftcorner.Pcfg, before folding unaries
    compiled: Any  # prsr.leftcorner.Compiled


class SynprocModel(Resource):
    """synproc's depth-bounded left-corner model of an `lvpcfg_grammar`.

    Spec keys: `grammar` (the `lvpcfg_grammar` spec). The resource stores
    the grammar exported as Berkeley's text dump would hold it
    (`prsr.Grammar.pcfg`): binary rules, the Viterbi unary
    closure, the root's rules, and the lexicon, about 1% of the size of
    ModelBlocks' text model. Loading folds the unaries and compiles the
    left-corner factors (`leftcorner.compile_model`), a few seconds on a GPU.
    """

    kind = "synproc_model"
    version = 1

    def dependencies(self) -> Mapping[str, Resource]:
        return {"grammar": LvpcfgGrammar(self.spec.get("grammar", {}))}

    def resolve(self, context: RunContext) -> dict[str, Any]:
        from prsr import leftcorner as lc

        return {
            "max_depth": lc.MAX_DEPTH,
            "iterations": lc.ITERATIONS,
            "parse_version": prsr.PARSE_VERSION,
        }

    def workload(self, resolved: Mapping[str, Any]) -> Workload:
        return Workload(slurm=True, ncores=2, memory_gb=16, time_hours=1, ngpus=0)

    def build(self, resolved, directory: Path, context: RunContext, dependencies) -> None:
        dependencies["grammar"].pcfg().save(directory)

    def load(
        self, resolved, directory: Path, context: RunContext, dependencies
    ) -> LoadedSynprocModel:
        from prsr import leftcorner as lc

        pcfg = Pcfg.load(directory)
        return LoadedSynprocModel(pcfg, lc.compile_model(pcfg.without_unaries(), _device(context)))
