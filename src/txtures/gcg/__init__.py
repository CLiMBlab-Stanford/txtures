# This file is part of txtures. Parts are translated from ModelBlocks
# (resource-linetrees/scripts/editabletrees2linetrees.pl and killUnaries.pl,
# resource-gcg/scripts/makefailtrees.py, and the gcg15 rule of
# resource-gcg/Makefile), Copyright 2009, ModelBlocks developers, and were
# modified for txtures in 2026.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""Reannotate Penn Treebank trees into generalized categorial grammar (gcg15).

This is a port of the ModelBlocks pipeline that makes `%.gcg15.senttrees`
from Penn Treebank `.mrg` files, and it reproduces that pipeline's output
exactly:

1. `ptb_lines`: one tree per line (`editabletrees2linetrees.pl`).
2. `txtures.gcg.fixes.fix`: repairs of treebank errors (`make-trees-fixed.pl`).
3. `txtures.gcg.rules.annotate`: the gcg15 reannotation (`annotate-gcg15.pl`).
4. `kill_unaries`: collapse unary branches, keeping lexical chains, and
   delete empty categories (`killUnaries.pl -l`).
5. `fail_tree`: replace a tree whose reannotation failed, recognizable by a
   category with no lowercase letter, by a right-branching tree of `FAIL`
   nodes over its words (`makefailtrees.py`).
6. Remove `-f` tags (the original treebank category) and move each `-l` tag
   (the attachment operator) to the end of its category.

`reannotate` runs steps 2-6 on one line. Lines that are not trees, such as
the `!ARTICLE` delimiters between files, pass through steps 2 and 3 as they
do in ModelBlocks.
"""

from __future__ import annotations

import re

from txtures.gcg import fixes, rules
from txtures.gcg.perl import PerlString

__all__ = ["ARTICLE", "GCG15_VERSION", "fail_tree", "kill_unaries", "ptb_lines", "reannotate"]

GCG15_VERSION = 1
"""Increment whenever `reannotate` would give any tree a different result.

It is part of the identity of every `gcg_treebank`. Changes that keep the
output identical, such as speedups or rewrites checked against all of WSJ,
leave it alone.
"""

ARTICLE = "!ARTICLE"
"""The line ModelBlocks writes before the trees of each treebank file."""


def ptb_lines(text: str) -> list[str]:
    """The trees of a `.mrg` file, one per line, without the outer bracket."""
    lines, tree, depth = [], "", 0
    for raw in text.split("\n"):
        line = re.sub(r"^\s*", "", raw)
        if line == "" or line.startswith("*"):
            continue
        tree += line + " "
        depth += line.count("(") - line.count(")")
        if depth == 0:
            tree = re.sub(r"^\s*\((?:TOP)?\s*\((.*)\s*\)\s*\)\s*$", r"(\1)", tree)
            tree = re.sub(r" *\)", ")", tree)
            tree = re.sub(r" +", " ", tree)
            lines.append(re.sub(r" $", "", tree))
            tree = ""
    return lines


def kill_unaries(text: str) -> str:
    """`killUnaries.pl -l`: collapse unary branches above lexical chains, drop empties."""
    line = PerlString(text)
    line.sub(r"\~", "!tilde!", g=True)
    line.sub(r"[\[\]]", "!", g=True)
    while line.search(r"\([^\(\)]*\)"):
        line.sub(r"\(([^\(\)]*)\)", r"\~\1\~")
        # kill the lower category of each unary branch that is not a preterminal
        line.sub(r"\~([^ ]*) +<([^ ]*) ([^>]*\[[^>]*)> *\~", r"\~\1 \3\~")
        # kill '-NONE-' subtrees
        line.sub(r" \~-NONE- .*\~", "")
        line.sub(r" \~[^ ]* *\~", "")
        while line.sub(r"(\~[^\~]*)<([^<>]*)>", r"\1\[\2\]"):
            pass
        line.sub(r"\~(.*)\~", r"<\1>")
    line.sub("<", "[")
    line.sub(">", "]")
    line.sub(r"\[", "(", g=True)
    line.sub(r"\]", ")", g=True)
    line.sub("!tilde!", "~", g=True)
    return line.text


def fail_tree(line: str) -> str:
    """`makefailtrees.py`: a failed reannotation becomes right-branching `FAIL` nodes."""
    if not re.search(r".*\([^a-z]+ .*", line):
        return line
    words = [found[1:-1] for found in re.findall(r" [^\) ]+\)", line)]

    def build(words: list[str]) -> str:
        if len(words) == 1:
            return f"(FAIL {words[0]})"
        return f"(FAIL {build(words[:1])} {build(words[1:])})"

    return build(words) + "\n"


def reannotate(line: str) -> str:
    """The gcg15 tree for one tree line from `ptb_lines` (steps 2-6), without a newline."""
    text = PerlString(line + "\n", fixes.VARIABLES)
    fixes.fix(text)
    text = PerlString(text.text, rules.VARIABLES)
    rules.annotate(text)
    result = fail_tree(kill_unaries(text.text))
    result = re.sub(r"\(([^ ]*)-f[^ ]*", r"(\1", result)
    result = re.sub(r"\(([^ ]*)-l([A-Z])([^ ]*)", r"(\1\3-l\2", result)
    return result.rstrip("\n")
