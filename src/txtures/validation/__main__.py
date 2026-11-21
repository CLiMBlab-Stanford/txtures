"""`python -m txtures.validation COMMAND`: benchmarks against ModelBlocks and the Berkeley parser."""

from __future__ import annotations

import argparse
import os
import sys

from txtures.errors import TxturesError
from txtures.validation import __doc__ as overview
from txtures.validation import paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m txtures.validation",
        description=overview,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str, *arguments: tuple[str, dict]):
        sub = commands.add_parser(name, help=help)
        for argument, options in arguments:
            sub.add_argument(argument, **options)
        return sub

    section = ("section", {"help": "a two-digit WSJ section"})
    grammar_dump = (
        "grammar",
        {"help": "a Berkeley text dump in the work directory, such as seed2.3sm"},
    )
    command("gcg15-reference", "ModelBlocks' gcg15 pipeline on one WSJ section", section)
    command(
        "gcg15-compare",
        "txtures.gcg against the reference stages of one section",
        section,
        ("--processes", {"type": int, "default": 8}),
    )
    command("berkeley-build", "clone the Berkeley parser and compile the Java harnesses")
    command("berkeley-prepare", "the trainer's input from the gcg15 references of WSJ 02-21")
    command(
        "berkeley-train",
        "train a Berkeley grammar as ModelBlocks does",
        ("seed", {"type": int}),
        ("--cycles", {"type": int, "default": 3}),
    )
    command(
        "berkeley-dump",
        "run a harness on a saved grammar",
        ("mode", {"choices": ["onestep", "merge", "split", "nodes"]}),
        ("grammar", {"help": "a saved grammar file, such as seed2.3sm.gr_2_splitting.gr"}),
    )
    command(
        "compare-onestep",
        "an EM iteration against Java's",
        ("grammar", {"help": "a dumped grammar file name"}),
    )
    command(
        "compare-split-merge",
        "a split or merge against Java's",
        ("mode", {"choices": ["split", "merge"]}),
        ("grammar", {"help": "a dumped grammar file name"}),
    )
    command("modelblocks-build", "compile ModelBlocks' compiler and parser in the work directory")
    command(
        "modelblocks-compile",
        "ModelBlocks' chain from a Berkeley dump to a synproc model",
        grammar_dump,
    )
    command(
        "compare-folding",
        "reading and unary folding against ModelBlocks' x-ccu and x-cc",
        grammar_dump,
    )
    command("compare-compiled", "every synproc table entry against ModelBlocks'", grammar_dump)
    command("compare-export", "exporting a trained grammar against Berkeley's dump", grammar_dump)
    command("sentences", "write test sentences from WSJ", section, ("count", {"type": int}))
    command(
        "gold-sentences",
        "write non-FAIL test sentences and their gold gcg15 trees",
        section,
        ("count", {"type": int}),
    )
    command(
        "modelblocks-merge",
        "join chunked synproc outputs",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name"}),
        ("parts", {"type": int}),
    )
    command(
        "modelblocks-parse",
        "ModelBlocks' synproc on test sentences",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name, such as wsj23-50"}),
        ("--beam", {"type": int, "default": 5000}),
        ("--threads", {"type": int, "default": 8}),
        ("--part", {"type": int, "help": "parse only this chunk (default: SLURM_ARRAY_TASK_ID)"}),
        ("--parts", {"type": int, "help": "the number of chunks"}),
    )
    command(
        "parse",
        "txtures' GPU parser on test sentences",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name"}),
        ("--beam", {"type": int, "default": 5000}),
        ("--merge", {"choices": ["once", "synproc"], "default": "once"}),
        ("--max-sentences", {"type": int, "default": 32}),
    )
    command(
        "debug-beams",
        "txtures' lexical entries and beams per word, in synproc_debug's format",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name"}),
        ("--beam", {"type": int, "default": 5000}),
        ("--merge", {"choices": ["once", "synproc"], "default": "synproc"}),
    )
    command(
        "compare-verbose",
        "txtures' lexical entries and beams against synproc_debug's",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name"}),
        ("--merge", {"choices": ["once", "synproc"], "default": "synproc"}),
    )
    command(
        "score",
        "bracket F1 (ModelBlocks' evalb) and bits per word of a parse against gold trees",
        ("sentences", {"help": "a .linetoks name with a .gold file"}),
        ("parse", {"help": "a synproc .tokdecs or txtures .json file in the work directory"}),
    )
    command(
        "compare-parse",
        "per-word surprisal against ModelBlocks'",
        grammar_dump,
        ("sentences", {"help": "a .linetoks name"}),
        ("--merge", {"choices": ["once", "synproc"], "default": "once"}),
    )
    args = parser.parse_args(argv)
    try:
        where = paths()
        name = args.command
        if name.startswith("gcg15"):
            from txtures.validation import gcg15

            if name == "gcg15-reference":
                gcg15.reference(where, args.section)
            else:
                return 1 if gcg15.compare(where, args.section, args.processes) else 0
        elif name.startswith("berkeley") or name in ("compare-onestep", "compare-split-merge"):
            from txtures.validation import berkeley

            {
                "berkeley-build": lambda: berkeley.build(where),
                "berkeley-prepare": lambda: berkeley.prepare(where),
                "berkeley-train": lambda: berkeley.train(where, args.seed, args.cycles),
                "berkeley-dump": lambda: berkeley.dump(where, args.mode, args.grammar),
                "compare-onestep": lambda: berkeley.compare_onestep(where, args.grammar),
                "compare-split-merge": lambda: berkeley.compare_split_merge(
                    where, args.mode, args.grammar
                ),
            }[name]()
        elif name == "score":
            from txtures.validation import scoring

            scoring.score(where, args.sentences, args.parse)
        elif name in (
            "sentences",
            "gold-sentences",
            "parse",
            "compare-parse",
            "debug-beams",
            "compare-verbose",
        ):
            from txtures.validation import parsing

            if name == "sentences":
                print(parsing.sentences(where, args.section, args.count))
            elif name == "gold-sentences":
                print(parsing.gold_sentences(where, args.section, args.count))
            elif name == "parse":
                parsing.parse(
                    where, args.grammar, args.sentences, args.beam, args.merge, args.max_sentences
                )
            elif name == "debug-beams":
                parsing.debug_beams(where, args.grammar, args.sentences, args.beam, args.merge)
            elif name == "compare-verbose":
                parsing.compare_verbose(where, args.grammar, args.sentences, args.merge)
            else:
                parsing.compare(where, args.grammar, args.sentences, args.merge)
        else:
            from txtures.validation import modelblocks

            {
                "modelblocks-build": lambda: modelblocks.build(where),
                "modelblocks-compile": lambda: modelblocks.compile_reference(where, args.grammar),
                "compare-folding": lambda: modelblocks.compare_folding(where, args.grammar),
                "compare-compiled": lambda: modelblocks.compare_compiled(where, args.grammar),
                "compare-export": lambda: modelblocks.compare_export(where, args.grammar),
                "modelblocks-parse": lambda: modelblocks.parse_reference(
                    where,
                    args.grammar,
                    args.sentences,
                    args.beam,
                    args.threads,
                    args.part
                    if args.part is not None
                    else (int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.parts else None),
                    args.parts,
                ),
                "modelblocks-merge": lambda: modelblocks.merge_parts(
                    where, args.grammar, args.sentences, args.parts
                ),
            }[name]()
    except TxturesError as error:
        print(f"txtures.validation: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
