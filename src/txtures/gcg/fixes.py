# This file is part of txtures. It is translated from ModelBlocks
# (resource-gcg/scripts/make-trees-fixed.pl),
# Copyright 2009, ModelBlocks developers, and was modified for txtures in 2026:
# the Perl rules are kept verbatim and their control flow is rewritten in Python.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""The treebank repairs of `make-trees-fixed.pl`, applied to one tree line."""

from txtures.gcg.perl import PerlString

VARIABLES = {}


def fix(tree: PerlString) -> None:
    """Repair one PTB tree line, in place, as `make-trees-fixed.pl` does in gcg mode."""

    ## remove extra spaces
    tree.sub("  +", " ", g=True)
    tree.sub(r" \( ", r" \(", g=True)
    tree.sub(r" \)", r"\)", g=True)
    tree.sub(r"\)\(", r"\) \(", g=True)
    ## change honorifics to adj
    tree.sub(r"\(NNP (Mrs\.|Mr\.|Ms\.|Dr\.|Prof\.|Sir|Madam|Rev\.)\)", r"\(JJ \1\)", g=True)
    ## conjoin disjoint trees with root node
    tree.sub(r"^(\(*[^S!\( ].*)$", r"\(S \1\)")
    tree.sub(r"^([^!].*[^\)]\))$", r"\(S \1\)")
    ## conjoin erroneously split punctuation
    tree.sub(r"\(\. \?\) \(\. \?\)", r"\(\? \?\)", g=True)
    ## fix auto tagged complementizers
    tree.sub(r"\(SBAR([^ ]*) *\((DT|RB) that", r"\(SBAR\1 \(IN that", g=True)
    #### post-coling12 fixes
    if not False:
        # correct wrong type of empty category
        tree.sub(r"\((WH[^ ]*-([0-9]+) )(?!.* \*T\*-\2)(.* )\*-\2\)", r"\(\1\3\*T\*-\2\)")
        # line 200
        tree.sub(
            r"\(NP-SBJ \(PRP he\) \) \(VP \(VBZ says\)  \(NP \(-NONE- \*-2\) \)\)",
            r"\(NP-SBJ \(PRP he\) \) \(VP \(VBZ says\)  \(NP \(-NONE- \*T\*-2\) \)\)",
        )
        # line 1628 from 02to21
        tree.sub(
            r"\(WHNP \(WDT which\) \) \(S  \(NP-SBJ-1 \(DT the\)",
            r"\(WHNP-1 \(WDT which\) \) \(S  \(NP-SBJ \(DT the\)",
        )
        # line 3670
        tree.sub(
            r"\(VBN entwined\)  \(PP-CLR \(-NONE- \*-2\) \)\)",
            r"\(VBN entwined\)  \(PP-CLR \(-NONE- \*T\*-2\) \)\)",
        )
        # line 5642
        tree.sub(
            r" Bush\) \) \(VP \(VBD allowed\)  \(S  \(NP-SBJ \(-NONE- \*-1",
            r" Bush\) \) \(VP \(VBD allowed\)  \(S  \(NP-SBJ \(-NONE- \*T\*-1",
        )
        # line 6447
        tree.sub(
            r"WHNP-1 \(WP What\) \) \(S  \(NP-SBJ \(-NONE- \*-1",
            r"WHNP-1 \(WP What\) \) \(S  \(NP-SBJ \(-NONE- \*T\*-1",
        )
        # line 7650
        tree.sub(
            r"-NONE- \*-3\) \) \(VP \(VBZ is\)  \(VP \(VBN caused",
            r"-NONE- \*T\*-3\) \) \(VP \(VBZ is\)  \(VP \(VBN caused",
        )
        # line 7725
        tree.sub(
            r" interviewer\) \) \(NP \(-NONE- \*-1\) \) \(NP-TMP \(NN yesterday",
            r" interviewer\) \) \(NP \(-NONE- \*T\*-1\) \) \(NP-TMP \(NN yesterday",
        )
        # line 8354
        tree.sub(
            r"PRP it\) \) \(VP \(VBD missed\)  \(NP \(-NONE- \*-3",
            r"PRP it\) \) \(VP \(VBD missed\)  \(NP \(-NONE- \*T\*-3",
        )
        # line 8567
        tree.sub(
            r"PRP it\) \) \(VP \(VBZ is\) \(RB n't\)  \(NP-PRD \(-NONE- \*-1",
            r"PRP it\) \) \(VP \(VBZ is\) \(RB n't\)  \(NP-PRD \(-NONE- \*T\*-1",
        )
        # line 10593 (of wsj02to21)
        tree.sub(
            r"(\(NP \(NP \(JJ grilled.* pudding\)\)) \(, ,\) \(PP (.* tea)\)",
            r"\(NP \1\) \(, ,\) \2",
        )
        # line 10700
        tree.sub(
            r"PRP he\) \) \(VP \(VBD was\)  \(NP-LOC-PRD \(-NONE- \*-1",
            r"PRP he\) \) \(VP \(VBD was\)  \(NP-LOC-PRD \(-NONE- \*T\*-1",
        )
        # line 13344
        tree.sub(
            r"VBN syndicated\) \(NN program\) \) \(S  \(NP-SBJ-1 \(-NONE- \*-3",
            r"VBN syndicated\) \(NN program\) \) \(S  \(NP-SBJ-1 \(-NONE- \*T\*-3",
        )
        # line 15562
        tree.sub(
            r"VBD screamed\)  \(NP \(-NONE- \*-1\) \)\) \(NP-SBJ \(DT the",
            r"VBD screamed\)  \(NP \(-NONE- \*T\*-1\) \)\) \(NP-SBJ \(DT the",
        )
        # line 32942
        tree.sub(
            r"DT that\) \(NN roster\) \) \(NP-PRD \(-NONE- \*-1",
            r"DT that\) \(NN roster\) \) \(NP-PRD \(-NONE- \*T\*-1",
        )
        # line 34448
        tree.sub(
            r"WDT that\) \) \(S  \(NP-SBJ \(-NONE- \*-1\) \) \(VP \(VBZ has",
            r"WDT that\) \) \(S  \(NP-SBJ \(-NONE- \*T\*-1\) \) \(VP \(VBZ has",
        )
        # line 37777
        tree.sub(
            r"WDT which\) \) \(S  \(NP-SBJ-2 \(-NONE- \*-1\) \) \(PP \(IN like",
            r"WDT which\) \) \(S  \(NP-SBJ-2 \(-NONE- \*T\*-1\) \) \(PP \(IN like",
        )
    #### normalize spacing to de-clutter rules
    # never more than one space
    tree.sub(" +", " ", g=True)
    # no spaces before close parens
    tree.sub(r" +\)", r"\)", g=True)
    if not False:
        ## detected from morph project...
        tree.sub(r"\(VBG preferred\)", r"\(VBN preferred\)")
        # line 4032
        tree.sub(r"\(VP=3 \(VBD positioned", r"\(VP=3 \(VBN positioned")
        # line 7473
        tree.sub("diversifed", "diversified")
        # line 12564
        tree.sub("decribe", "describe")
        # line 12564
        tree.sub("establsh", "establish")
        # line 25545
        tree.sub("syndciate", "syndicate")
        # line 21115
        tree.sub("contribued", "contributed")
        # line 32739
        tree.sub(r"\(VB[ND] repurchase\)", r"\(VB repurchase\)")
        # line 18270 & ...
        tree.sub(r"\(VB[ND] build\)", r"\(VB build\)")
        # line 35473
        tree.sub(r"\(VBN dead\)", r"\(JJ dead\)")
        # line 30656
        tree.sub(
            r"\(PP \(IN by\) \(NP-LGS-3 \(NN Moody\) \(POS 's\)\)\)\) \(CC and\) \(VP \(ADJP-PRD=1 \(JJ SP1-plus\)\) \(PP \(IN by\) \(NP-LGS=3 \(NNP S&P\)\)",
            r"\(PP-3 \(IN by\) \(NP-LGS \(NN Moody\) \(POS 's\)\)\)\) \(CC and\) \(VP \(ADJP-PRD=1 \(JJ SP1-plus\)\) \(PP=3 \(IN by\) \(NP-LGS \(NNP S&P\)\)",
        )
        # line 1409&al:
        tree.sub(
            r"\(IN for\) (\(RB now\)|\(ADVP[^ ]* \(RB now\)\))", r"\(IN for\) \(NP \(RB now\)\)"
        )
        # line 10016:
        tree.sub(r"\(IN for\) \(RB good\)", r"\(IN for\) \(NP good\)")
    ## misc empty category madness...
    # line 640 (incl 01):
    tree.sub(r"ADVP\|PRT", "ADVP")
    # line 16932 (incl 01):
    tree.sub(r"PRT\|ADVP", "ADVP")
    # line 722&al: some annotator is a fan of matt johnson
    tree.sub(r"\(DT the\) \(DT the\)", r"\(DT the\)")
    # line 1187&al: sentence segmenter error
    tree.sub(r"\(NP \(DT the\)\)\)", r"\(NP \(DT the\) \(NNP U.S.\)\)\)")
    # line 1574: tokenizer
    tree.sub(r"\(RB close\) \(NP", r"\(VB close\) \(NP")
    # line 3780:
    tree.sub(r"\(VP \(NNP Face", r"\(VP \(VBP Face")
    #  #line 4371:
    #  s/\(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBG alternating\) \(NP \(NP \(NNS feats\)\)/\(NP \(NP \(VBG alternating\) \(NP \(NP \(NNS feats\)\)/;
    # line 5393:
    tree.sub(r"\(VP \(POS 's\)", r"\(VP \(VBZ 's\)")
    # line 5823:
    tree.sub(
        r"\(IN for\) \(S-NOM \(NP-SBJ \(NNP Congress\)\) \(VP \(TO to\)",
        r"\(IN for\) \(S \(NP-SBJ \(NNP Congress\)\) \(VP \(TO to\)",
    )
    # line 7440:
    tree.sub(r"\(VP \(JJ own\)", r"\(VP \(VBP own\)")
    # line 8535:
    tree.sub(r"\(VP \(NNP Let\)", r"\(VP \(VB Let\)")
    # line 8803:
    tree.sub(r"\(VP \(NNP Meet\)", r"\(VP \(VB Meet\)")
    # line 8935:
    tree.sub(r"\(VP \(JJ oversold\)", r"\(VP \(VBN oversold\)")
    # line 11592:
    tree.sub(
        r"\(IN as\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBZ is\)",
        r"\(IN as\) \(S \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBZ is\)",
    )
    # line 13002: conj scoping error
    tree.sub(
        r"\(VP \(VBD agreed\) \(PP-CLR \(IN on\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP (\(VP \(VBG denying\) \(NP \(DT the\) \(NN injunction\)\)\)) \(CC and\) (\(VP \(VBD did\).*work\))\)\)\)",
        r"\(VP \(VP \(VBD agreed\) \(PP-CLR \(IN on\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \1\)\)\) \(CC and\) \2",
    )
    # line 17656:
    tree.sub(
        r"\(PP-CLR \(TO to\) \(S-NOM \(NP-SBJ \(NNP Helmuth\)",
        r"\(PP-CLR \(TO to\) \(S \(NP-SBJ \(NNP Helmuth\)",
    )
    # line 6440&al: S-NOM applied too liberally
    tree.sub(r"(\(PP[^ ]* \((?:IN|RB) [Aa]s\)) \(S-NOM ", r"\1 \(S ", g=True)
    tree.sub(
        r"(\(PP[^ ]* \((?:IN|RB) (?:[Ww]ith|[Bb]efore|[Aa]fter|[Ll]ike|[Uu]ntil|[Ff]or)\)) \(S-NOM (?!(\([^\(\)]*\([^\(\)]*\)\) )?\(NP-SBJ[^ ]* \(-NONE-|\(NP-SBJ[^\)]*California|\(NP-SBJ[^\)]*B\.A\.T|\(NP-SBJ[^\)]*things)",
        r"\1 \(S ",
        g=True,
    )
    # line 17057:
    tree.sub(r"\(VP \(JJ unhindered\)", r"\(VP \(VBN unhindered\)")
    # line 32010:
    ### FAILED: s/\(S (\(\S \(NP-SBJ \(-NONE- \*\)\) \(VP \(TO to\) \(VP \(VB press\) \(NP \(DT a\) \(JJ strict\) \(JJ seven-point\))/\(S-4 \1/;
    # line 34322:
    # line 38511:
    # line 20481:
    tree.sub(
        r"\(S-NOM (\(NP-SBJ[^ ]* \(-NONE- \*\)\) \(VP (\(ADVP \(RB just\)\) )?\(VBN? )", r"\(S \1"
    )
    # line 10 (incl 01):
    tree.sub(
        r"(\(WHADVP-3 \(-NONE- 0\) \) \(S  \(NP-SBJ \(-NONE- )\*(\) \) \(RB not\)  \(VP \(TO to\)  .*) \(ADVP \(-NONE- \*T\*-3\) \)",
        r"\1\*T\*-3\2",
    )
    # line 1281 (incl 01):
    tree.sub(
        r"\(NP \(NNP American\) \(NNP City\) \)\)\) \(NP-TMP \(-NONE- \*T\*-1\) \)",
        r"\(NP \(NNP American\) \(NNP City\) \)\)\)",
    )
    # line 14650 NOTE: this is an annotation bug, but it seems to hurt not help
    # s/\(DT a\) \(WDT which/\(WDT which/;
    # line 15834,22399 (incl 01):  NOTE: this is not a bug, but a reannotation!
    tree.sub(
        r"\(WHNP-2 *\(-NONE- *0\) *\) *\(IN *as\)(.*)\(S *\(NP-SBJ *\(-NONE- *\*-1\) *\) *\(NP-PRD *\(-NONE- *\*T\*-2\) *\) *\)",
        r"\(WHNP-2 \(IN as\)\)\1\(NP \(-NONE- \*\)\) \(NP \(-NONE- \*T\*-2\)\)",
    )
    # line 16659 (02to21):
    tree.sub(
        r"\(WHADVP-2 \(WRB how\) \(NP \(NP \(JJ much\)\)",
        r"\(WHNP-2 \(WHNP \(WHNP \(WRB how\) \(JJ much\)\)",
    )
    # line 19318 (incl 01):  NOTE: this is not a bug, but a reannotation!
    tree.sub(
        r"\(SBARQ *\(WHADVP *\(WRB *why\) *\) *\(FRAG *\(SQ *\(-NONE- *\*\?\*\) *\) *\) *\)",
        r"\(SBARQ \(WRB why\)\)",
    )
    # line 4914,27175 (incl 01):
    tree.sub(
        r"\(SBAR *\(WHNP-5 *\(-NONE- *0\) *\) *\(S(?:-PRP)? *\(-NONE- *\*ICH\*-4\) *\) *\)(?: *\(NP *\(-NONE- *\*ICH\*-1\) *\))?(.*)\(-NONE- *\*T\*-5\)",
        r"\1\(-NONE- \*-5\)",
    )
    # line 20028 (incl 01):
    tree.sub(
        r"\(VBN *called\) *\(S *\(NP-SBJ *\(-NONE- *\*-2\) *\) *\(NP-PRD *\(-NONE- *\*\?\*\) *\) *\)",
        r"\(VBN called\) \(NP \(-NONE- 0\)\)",
    )
    # line 21405 (incl 01):
    tree.sub(
        r"\(SBAR *\(WHADVP *\(-NONE- *0\) *\) *\(S *\(-NONE- *\*T\*-1\) *\) *\)",
        r"\(S \(-NONE- \*T\*-1\)\)",
    )
    # line 23454 (02to21):
    tree.sub(r"\(PP \(IN of\) \(W?H?NP \(W", r"\(WHPP \(IN of)\ \(WHNP \(W")
    # line 30323 (incl 01):
    tree.sub(
        r"\(WHNP-2  \(WHNP \(-NONE- 0\) \) \(SBAR \(-NONE- \*ICH\*-4\) \)\)",
        r"\(WHNP-2 \(-NONE- 0\)\)",
    )
    # line 31306 (incl 01):
    tree.sub(
        r"\(VBN nicknamed\)  \(S  \(NP-SBJ \(-NONE- \*-1\) \) \(NP-PRD \(-NONE- \*ICH\*-2\) \)\)",
        r"\(VBN nicknamed\) \(NP \(-NONE- 0\)\)",
    )
    # line 39118 (incl 01):
    tree.sub(
        r"\(SBAR *\(WHADVP-1 *\(-NONE- *0\) *\) *\(IN *for\)(?!.*\*T\*-1)", r"\(SBAR \(IN for\)"
    )
    # line 34404 (incl 01):
    tree.sub(r"(\(DT The\) \(NN sense\))  \(SBAR \(-NONE- \*EXP\*-1\) \)", r"\1")
    # line 24441 (incl 01):
    tree.sub(r"\(PP *\(NP *\(VBG", r"\(PP \(VP \(VBG")
    # line 32473 (02to21):
    tree.sub(r"\(WRB how\) \(RB much\)", r"\(WRB how\) \(JJ much\)")
    # line 39525 (incl 01):
    tree.sub(
        r"\(NP *\(VB buy\) *\(CC and\) *\(VB sell\) *\)",
        r"\(VB \(VB buy\) \(CC and\) \(VB sell\)\)",
    )
    ## trace errors...
    # line 962:
    #  s/\(SBAR-PRD \(WHNP \(WP what\)\)/!!!!!!!FOUND!!!!!!!/;
    tree.sub(
        r"\(SBAR-PRD \(WHNP \(WP what\)\) \(S \(NP-SBJ-3 \(PRP you\)\) \(VP \(VBP 're\) \(VP \(VBG going\) \(S \(NP-SBJ \(-NONE- \*-3\)\) \(VP \(TO to\) \(VP \(VB find\) ",
        r" \(SBAR-PRD \(WHNP-6 \(WP what\)\) \(S \(NP-SBJ-3 \(PRP you\)\) \(VP \(VBP 're\) \(VP \(VBG going\) \(S \(NP-SBJ \(-NONE- \*-3\)\) \(VP \(TO to\) \(VP \(VB find\) \(NP \(-NONE- \*T\*-6\)\) ",
    )
    # line 2556:
    tree.sub(
        r"\(WHNP \(WP what\)\) \(S \(NP-SBJ \(DT the\) \(-LRB- -LRB-\) \(NNP Federal\) \(NNP Reserve\) \(-RRB- -RRB-\)\) \(VP \(MD will\) \(VP \(VB do\)",
        r"\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(DT the\) \(-LRB- -LRB-\) \(NNP Federal\) \(NNP Reserve\) \(-RRB- -RRB-\)\) \(VP \(MD will\) \(VP \(VB do\) \(NP \(-NONE- \*T\*-6\)\)",
    )
    # line 2965:
    tree.sub(
        r"\(WHNP \(WP What\)\) \(S \(NP-SBJ-1 \(PRP I\)\) \(VP \(VBP 'm\) \(VP \(VBG trying\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB say\)",
        r"\(WHNP-6 \(WP What\)\) \(S \(NP-SBJ-1 \(PRP I\)\) \(VP \(VBP 'm\) \(VP \(VBG trying\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB say\) \(NP \(-NONE- \*T\*-6\)\)",
    )
    # line 18338:
    tree.sub(
        r"\(WHNP \(WP what\)\) \(S \(NP-SBJ \(PRP they\)\) \(VP \(VBD had\) \(VP \(VBN hoped\) \(SBAR \(-NONE- 0\) \(S \(-NONE- \*\?\*\)\)\)",
        r"\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(PRP they\)\) \(VP \(VBD had\) \(VP \(VBN hoped\) \(SBAR \(-NONE- \*T\*-6\)\)",
    )
    # line 28261:
    tree.sub(
        r"\(WHNP \(WP what\)\) \(S \(NP-SBJ \(NNS banks\)\) \(VP \(VBP charge\) \(NP \(DT each\) \(JJ other\)\)",
        r"\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(NNS banks\)\) \(VP \(VBP charge\) \(NP \(DT each\) \(JJ other\)\) \(NP \(-NONE- \*T\*-6\)\)",
    )
    # line 28680:
    tree.sub(
        r"\(WHNP \(WP what\)\) \(S \(NP-SBJ-1 \(NNP Sony\)\) \(VP \(VBD was\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB offer\)",
        r"\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ-1 \(NNP Sony\)\) \(VP \(VBD was\) \(S \(NP-SBJ \(-NONE- *-1\)\) \(VP \(TO to\) \(VP \(VB offer\) \(NP \(-NONE- \*T\*-6\)\)",
    )
    # line 1628:
    tree.sub(
        r"\(WHNP \(WDT which\) \) \(S \(NP-SBJ-1 \(DT the\) \(JJ Soviet\) \(NN leader\) \)",
        r"\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(DT the\) \(JJ Soviet\) \(NN leader\) \)",
    )
    # line 3642:
    tree.sub(
        r"\(WHNP \(IN that\) \) \(S \(NP-SBJ-1 \(DT the\) \(NNS Japanese\) \) \(VP \(MD might\) \(VP \(VB want\) \(S \(NP-SBJ \(-NONE- \*-1\) \) \(VP \(TO to\) \(VP \(VB buy\)",
        r"\(WHNP-2 \(IN that\) \) \(S \(NP-SBJ-1 \(DT the\) \(NNS Japanese\) \) \(VP \(MD might\) \(VP \(VB want\) \(S \(NP-SBJ \(-NONE- \*-1\) \) \(VP \(TO to\) \(VP \(VB buy\) \(NP \(-NONE- \*T\*-2\) \)",
    )
    # line 5642:
    tree.sub(
        r"\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(NNP Bush\) \) \(VP \(VBD allowed\) \(S \(NP-SBJ \(-NONE- \*-1\)",
        r"\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(NNP Bush\) \) \(VP \(VBD allowed\) \(S \(NP-SBJ \(-NONE- \*T\*-1\)",
    )
    # line 7797:
    tree.sub(
        r"\(NP \(DT The\) \(JJ same\) \(NN evening\) \) \(SBAR \(WHNP-4 \(IN that\) \)",
        r"\(NP \(DT The\) \(JJ same\) \(NN evening\) \) \(SBAR \(WHPP-4 \(IN that\) \)",
    )
    # line 17204:
    tree.sub(
        r"\(WHNP-3 \(WDT which\) \) \(S \(NP-SBJ \(JJ many\) \(NNS investors\) \) \(VP \(VBP say\) \(SBAR \(-NONE- 0\) \(S \(NP-SBJ-1 \(-NONE- \*-3\)",
        r"\(WHNP-3 \(WDT which\) \) \(S \(NP-SBJ \(JJ many\) \(NNS investors\) \) \(VP \(VBP say\) \(SBAR \(-NONE- 0\) \(S \(NP-SBJ-1 \(-NONE- \*T\*-3\)",
    )
    # line 17563:
    tree.sub(
        r"\(WHNP \(WP what\) \) \(S \(NP-SBJ \(JJ other\) \(JJ prosecutorial\) \(NNS abuses\) \) \(VP \(MD may\) \(VP \(VB have\) \(VP \(VBN occurred\)",
        r"\(WHNP-1 \(WP what\) \(JJ other\) \(JJ prosecutorial\) \(NNS abuses\) \) \(S \(NP-SBJ \(-NONE- \*T\*-1\) \) \(VP \(MD may\) \(VP \(VB have\) \(VP \(VBN occurred\)",
    )
    # line 25018:
    tree.sub(
        r"\(NP \(NP \(DT a\) \(NN year\) \) \(SBAR \(WHNP-1 \(IN that\) \)",
        r"\(NP \(NP \(DT a\) \(NN year\) \) \(SBAR \(WHPP-1 \(IN that\) \)",
    )
    # line 27938:
    tree.sub(
        r"\(WHNP \(WP what\) \) \(S \(NP-SBJ \(DT some\) \(NNS economists\) \) \(VP \(VBP call\) ",
        r"\(WHNP-1 \(WP what\) \) \(S \(NP-SBJ \(DT some\) \(NNS economists\) \) \(VP \(VBP call\) \(NP \(-NONE- \*T\*-1\) \) ",
    )
    # line 29529:
    tree.sub(
        r"\(WHNP \(WDT which\) \) \(S \(NP-SBJ \(PRP it\) \) \(VP \(MD will\) \(VP \(VB occupy\) ",
        r"\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(PRP it\) \) \(VP \(MD will\) \(VP \(VB occupy\) \(NP \(-NONE- \*T\*-1\) \) ",
    )
    # line 34726:
    # s/\(WHADJP \(RB exactly\) \(WP what\)\) \(NN impact\)/\(WHNP \(RB exactly\) \(WDT what\) \(NN impact\)\)/;
    # line 36352:
    tree.sub(
        r"\(WHNP \(IN that\) \) \(S \(NP-SBJ \(DT the\) \(NN government\) \) \(VP \(VBD feared\) \(SBAR \(WHNP-1 \(-NONE- 0\) \)",
        r"\(WHNP-1 \(IN that\) \) \(S \(NP-SBJ \(DT the\) \(NN government\) \) \(VP \(VBD feared\) \(SBAR \(WHNP \(-NONE- 0\) \)",
    )
    # line 36736:
    tree.sub(
        r"\(WHNP \(IN that\) \) \(S \(NP-SBJ \(PRP he\) \) \(VP \(VBD estimated\) \(SBAR \(WHNP-2 \(-NONE- 0\) \)",
        r"\(WHNP-2 \(IN that\) \) \(S \(NP-SBJ \(PRP he\) \) \(VP \(VBD estimated\) \(SBAR \(WHNP \(-NONE- 0\) \)",
    )
    # line 37010:
    tree.sub(
        r"\(WHNP \(IN that\) \) \(S \(NP-SBJ \(DT some\) \(NNS analysts\) \) \(VP \(VBP say\) \(SBAR \(WHNP-1 \(-NONE- 0\) \) ",
        r"\(WHNP-1 \(IN that\) \) \(S \(NP-SBJ \(DT some\) \(NNS analysts\) \) \(VP \(VBP say\) \(SBAR \(WHNP \(-NONE- 0\) \) ",
    )
    # line 37777:
    tree.sub(
        r"\(SBAR \(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ-2 \(-NONE- \*-1\) \)",
        r"\(SBAR \(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ-2 \(-NONE- \*T\*-1\) \)",
    )
    # line 39576:
    tree.sub(
        r"\(WHNP \(WDT which\) \) \(S \(NP-SBJ \(PRP they\) \) \(VP \(VBP sell\) ",
        r"\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(PRP they\) \) \(VP \(VBP sell\) \(NP \(-NONE- \*T\*-1\) \)",
    )
    ## false VBN -> should be VBD...
    # line 41:
    tree.sub(
        r"\(, ,\) \(VP \(VBN reached\) \(NP \(-NONE- \*T\*-",
        r"\(, ,\) \(VP \(VBD reached\) \(NP \(-NONE- \*T\*-",
    )
    # line 4257:
    tree.sub(r"2\) \) \(VP \(VBN forced", r"2\) \) \(VP \(VBN forced")
    # line 6124:
    tree.sub(
        r"\(VP \(VBN suspended\) \(NP \(NN stock-index\)",
        r"\(VP \(VBD suspended\) \(NP \(NN stock-index\)",
    )
    # line 6815:
    tree.sub(
        r"\(VP \(VBN involved\) \(NP \(JJ critical\)", r"\(VP \(VBD involved\) \(NP \(JJ critical\)"
    )
    # line 7676:
    tree.sub(r"\(NN week\) \) \(VP \(VBN succeeded\)", r"\(NN week\) \) \(VP \(VBD succeeded\)")
    # line 10076:
    tree.sub(
        r"\(NNS completions\) \) \(VP \(VBN lagged\)", r"\(NNS completions\) \) \(VP \(VBD lagged\)"
    )
    # line 10572:
    tree.sub(r"\(RB ago\) \) \(VBN acquired\)", r"\(RB ago\) \) \(VBD acquired\)")
    # line 11963:
    tree.sub(r"\(, ,\) \) \(VP \(VBN announced\)", r"\(, ,\) \) \(VP \(VBD announced\)")
    # line 12946:
    tree.sub(r"3\) \) \(VP \(VBN generated", r"3\) \) \(VP \(VBD generated")
    # line 13609:
    tree.sub(r"NNS researchers\) \) \(VP \(VBN agreed", r"NNS researchers\) \) \(VP \(VBD agreed")
    # line 15066:
    tree.sub(r"1\) \) \(VP \(VBN started", r"1\) \) \(VP \(VBD started")
    # line 15591, 28900:
    tree.sub(r"1\) \) \(VP \(VBN changed", r"1\) \) \(VP \(VBD changed")
    # line 15961:
    tree.sub(r"1\) \) \(VP \(VBN opened", r"1\) \) \(VP \(VBD opened")
    # line 18772:
    tree.sub(r"NNP KKR\) \) \(VP \(VBN restructured", r"NNP KKR\) \) \(VP \(VBD restructured")
    # line 19265:
    tree.sub(r"1\) \) \(VP \(VBN included", r"1\) \) \(VP \(VBD included")
    # line 19680:
    tree.sub(r"VBN become\) \(NP-PRD \(JJ overnight", r"VBD become\) \(NP-PRD \(JJ overnight")
    # line 19748:
    tree.sub(r"NN bank\) \) \(VP \(VBN disclosed\)", r"NN bank\) \) \(VP \(VBD disclosed\)")
    # line 21201:
    tree.sub(r"NP Goodson\) \) \(VP \(VBN bought", r"NP Goodson\) \) \(VP \(VBD bought")
    # line 23901:
    tree.sub(r"2\) \) \(VP \(VBN opened", r"2\) \) \(VP \(VBD opened")
    # line 24689:
    tree.sub(r"1\) \) \(VP \(VBN traded", r"1\) \) \(VP \(VBD traded")
    # line 24700:
    tree.sub(r"3\) \) \(VP \(VBN stretched", r"3\) \) \(VP \(VBD stretched")
    # line 26286:
    tree.sub(r"VBN set\) \(NP \(NNS plans", r"VBD set\) \(NP \(NNS plans")
    # line 28510:
    tree.sub(r"1\) \) \(VP \(VBN made", r"1\) \) \(VP \(VBD made")
    # line 32752:
    tree.sub(r"NNS eggs\) \) \(VP \(VBN come", r"NNS eggs\) \) \(VP \(VBD come")
    # line 33883:
    tree.sub(r"NNP II\) \) \(VP \(VBN ended", r"NNP II\) \) \(VP \(VBD ended")
    # line 38619:
    tree.sub(r"1\) \) \(VP \(VBN set\) \(PRT \(RP off", r"1\) \) \(VP \(VBD set\) \(PRT \(RP off")
    # line 38773:
    tree.sub(r"NN company\) \) \(VP \(VBN valued", r"NN company\) \) \(VP \(VBD valued")
    # line 38814:
    tree.sub(r"1\) \) \(VP \(VBN induced", r"1\) \) \(VP \(VBD induced")
    # line 39070:
    tree.sub(r"NN market\) \) \(VP \(VBN stabilized", r"NN market\) \) \(VP \(VBD stabilized")
    # line 39264:
    tree.sub(r"PRP it\) \) \(VP \(VBN opened", r"PRP it\) \) \(VP \(VBD opened")
    # line 39579:
    tree.sub(r"NN analyst\) \) \(VP \(VBN estimated", r"NN analyst\) \) \(VP \(VBD estimated")
    # line 39619:
    tree.sub(r"RB finally\) \) \(VP \(VBN opened", r"RB finally\) \) \(VP \(VBD opened")
    # line 39623:
    tree.sub(r"NN market\) \) \(VP \(VBN opened", r"NN market\) \) \(VP \(VBD opened")
    # line 39779:
    tree.sub(r"RB never\) \) \(VBN reopened", r"RB never\) \) \(VBD reopened")
    # line ...: weird 'have issues forecasts' error -> should be 'issued' (would get caught by annotCats)
    tree.sub(r"\(VP *\(NNS issues", r"\(VP \(VBN issued")
    # false NN -> should be VB/VBZ
    tree.sub(r"\(VP  *\(NNS  *([^\)]*)\)", r"\(VP \(VBZ \1\)", g=True)
    tree.sub(r"\(VP  *\(NN  *([^\)]*)\)", r"\(VP \(VB \1\)", g=True)
    # false VBD -> should be VBN...
    tree.sub(
        r"\(VBD ((?!received)[^\)]*)\) *\(NP *\(-NONE- \*\) \)",
        r"\(VBN \1\) \(NP \(-NONE- \*\) \)",
        g=True,
    )
    ## for each constituent...
    while tree.search(r"\([^\(\)]*\)"):
        ## convert outer parens to braces...
        tree.sub(r"\(([^\(\)]*)\)", r"{\1}")
        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
        # line 3828, 19200 (incl 01): false (NP ... PRP POS) ... -> should be (NP ...) (VP VBZ ...)
        tree.sub(
            r"{(.*) <(NP[^>]*\[PRP[^\]]*\]) *\[POS *'s\] *> *(<VP.*)}",
            r"\(\1 <\2> \{VP <VBZ 's> \3\}\)",
        )
        #    # line 2247, ...: false POS 's following letter -> should be NNS (also wrong, but avoids later problems)
        #    s/{((?!NP).*<NP[^>]*\[(?:NN|PRP)[^\]]*\] *)\[POS *'s\]( *>.*)}/{\1 \[NNS 's\]\2}/;
        #    #s/{(NP.*<(?:NN|PRP)[^ ]* [A-Z]>) *<POS *'s> *}/{\1 <NNS 's>}/;
        #    # ???? does not seem to exist!
        #    # line 24513:
        tree.sub(
            "{(SBAR) *<(WHNP-1 [^>]*)> *<(S) ([^>]*)> <(VP [^>]*)> *}", r"{\1 <\2> <\3 \4 [\5]>}"
        )
        # line 6603, 18358 (incl 01): false NP at beginning of VP -> should be VP
        tree.sub(r"\((VP *)\{NP( <VBZ (?:kills|blames)>.*)\}", r"\(\1\{VP\2}")
        ## conj scoping errors...
        # line 3911 (incl 01):
        tree.sub(
            r"(<VB have>[^\(]*){VP *(<VP *\[VBN made\].*) +(<CC.*<VP *\[VB discard\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 5940 (incl 01):
        tree.sub(
            r"(<VB resume>[^\(]*){NP *(<NP *\[NN growth\].*) +(<CC.*)<NP *(\[VB service\].*)}",
            r"{VP \1\2\} \3<VP \4",
        )
        # line 10683 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN purchased\].*) +(<CC.*<VP *\[VBZ plans\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 10860 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN started\].*) +(<CC.*<VP *\[VBZ has\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 18311 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN met\].*) +(<CC.*<VP *\[VBZ is\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 29504 (incl 01):
        tree.sub(
            r"(<VBP have>[^\(]*){VP *(<VP *\[VBN plunged\].*) +(<CC.*<VP *\[PP[^>]*\[VBP are\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 29706 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN reacted\].*) +(<CC.*<VP *\[ADV[^>]*\[VBZ is\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 29951 (incl 01):
        tree.sub(
            r"(<VBP have>[^\(]*){VP *(<VP *\[VBN outgrown\].*) +(<CC.*<VP *\[VBP have\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 30304 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN lost\].*) +(<CC.*<VP *\[VBZ has\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 31705 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(<VP *\[VBN licensed\].*) +(<CC.*<VP *\[VBZ plans\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 41333 (incl 01):
        tree.sub(
            r"(<VBP have>[^\(]*){VP *(<VP *\[VBN read\].*) +(<CC.*<VP *\[VBP estimate\].*)}",
            r"{VP \1\2\} \3",
        )
        # line 41381 (incl 01):
        tree.sub(
            r"(<VBZ has>[^\(]*){VP *(.*<VP *\[VB hurt\].*) +(<CC.*<VP *\[VBZ has\].*)}",
            r"\(VP {VP \1\2\} \3\)",
        )
        # line 40563 (incl 01):
        tree.sub(
            r"{VP *(<VBD had> *<VP *\[VBN silted\][^>]*>)( *<CC.*<VP *\[ADV[^>]*\[VBD was\].*)}",
            r"\(VP {VP \1\}\2\)",
        )
        # line 10807 (incl 01):
        tree.sub("{(NP[^ ]*) *<(NP[^>]*)> *<VBZ *'s> (.*)}", r"{\1 <\2 \[POS 's\]> \3}")
        ####################
        ## convert inner angles (if any) to bracks...
        while tree.sub("({[^{}]*)<([^<>]*)>", r"\1\[\2\]"):
            pass
        ## convert outer braces to angles...
        tree.sub("{(.*)}", r"<\1>")
    ## finish up...
    tree.sub("<", "[")
    tree.sub(">", "]")
    ## translate to parens again...
    tree.sub(r"\[", r"\(", g=True)
    tree.sub(r"\]", r"\)", g=True)
    ## for each constituent...
    while tree.search(r"\([^\(\)]*\)"):
        ## convert outer parens to braces...
        tree.sub(r"\(([^\(\)]*)\)", r"{\1}")
        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
        # false VBZ 's anywhere in NP -> should be POS
        while tree.sub("{(NP.*)<VBZ *'s>((?! *<DT).*)}", r"{\1<POS 's>\2}"):
            pass
        # false POS 's not at end of NP sections off beginning
        tree.sub("{(NP[^ ]*) (.*<.*<POS *'s>)(.*<.*)}", r"\(\1 \{NP \2\}\3\)")
        # false VBZ non-'s at end of NP -> should be NNS
        tree.sub("{(NP.*)<VBZ ([^>]*)> *}", r"{\1<NNS \2>}")
        # false VB anywhere immediately beneath NP -> should be NN
        while tree.sub("{(NP.*)<VB ([^>]*)>(.*)}", r"{\1<NN \2>\3}"):
            pass
        # false VBG conjoined in NP -> should be NN
        tree.sub(
            "{(NP.*)<VBG ([^>]*)> *<CC ([^>]*)> *<VBG ([^>]*)> *}", r"{\1<NN \2> <CC \3> <NN \4>}"
        )
        # false VBG at end of NP -> should be NN
        tree.sub("{(NP.*)<VBG ([^>]*)> *}", r"{\1<NN \2>}")
        # false S -> should be S-ADV
        tree.sub("{(S *<)S( [^>]*> *<,[^>]*> .*<NP.*)}", r"{\1S-ADV\2}")
        # false VBD -> should be VBN...
        tree.sub(r"({NP[^\}]*<VP[^\]]*)VBD", r"\1VBN")
        # missing -NONE- for passive (e.g. 'based on')
        #    s/{(PP[^ ]* <VBN [^ ]*>) (<PP.*)}/{\1 <NP <-NONE- *>> \2}/;
        # false VBN post-modifying NP without -NONE- -> should be with -NONE-...
        if False:
            tree.sub(
                r"(\(NP[^\(\)]*\{VP *<VBN[^>]*> *(?!( *<PRT[^>]*>|( *<[^>]*>)*| *<PP[^ ]* *\[(RB|RP|IN)[^\]]*\])?[^>\]]*\[-NONE- \*))",
                r"\1 <NP \[-NONE- \*\]>",
            )
        # false VBN in root S -> should be VBD (dependent on S-ADV above)...
        tree.sub(
            r"(^[^\)>\]]*\{S .*<NP[^>]*>(?! *<VP[^>\]]*MD).*<VP[^>\]]*)VBN(?! married|[^\]]*\] *\[NP *\[-NONE- *\*)(.*\})",
            r"\1VBD\2",
        )
        if not False:
            # false RB yet -> should be CC...
            tree.sub(
                r"{([^ ]*) <([^- ]*(?![A-Z]))([^>]*)>((?: <[^ ]* (?:``|,|'')>)*) <[^\]>]* yet[\]]*> <\2([^>]*)>}",
                r"{\1 <\2\3>\4 <CC yet> <\2\5>}",
            )
            # false SBAR WH without a trace gets trace added...
            tree.sub(
                "{(SBAR[^ ]*) <(WHADVP(?:(?!-[0-9]+)[^ ])*) ([^>]*)> <(S[^ ]*) ([^>]*)>}",
                r"{\1 <\2-0000 \3> <\4 \5 \[ADVP \[-NONE- \*T\*-0000\]\]>}",
            )
            # false extraposed SBAR too deep in constit gets pushed up...
            tree.sub(r"(\[-NONE- \*ICH\*(-[0-9]+)\].*){(.*) <(SBAR[^ ]*\2.*)>}", r"\1\{\3\} <\4>")
            # never happens
            # s/{(SBAR[^ ]*) <(WHADVP[^ ]*)(-[0-9]+(?![0-9]))([^>]*)> <(S[^ ]*)(?![^>]*\[-NONE- \*T\*\3\][^>]*)([^>]*)>}/{\1-mody <\2\3\4> <\1 \[-NONE- \*T\*\3\]\2>}/;
        ####################
        ## convert inner angles (if any) to bracks...
        while tree.sub("({[^{}]*)<([^<>]*)>", r"\1\[\2\]"):
            pass
        ## convert outer braces to angles...
        tree.sub("{(.*)}", r"<\1>")
    ## finish up...
    tree.sub("<", "[")
    tree.sub(">", "]")
    ## translate to parens again...
    tree.sub(r"\[", r"\(", g=True)
    tree.sub(r"\]", r"\)", g=True)
    # false VBD -> should be VBN (dependent on VBZ->POS change above)...
    tree.sub(r"(( be| being| been| is| was|VBZ 's| are| were| 're)\)[^\)]*)VBD", r"\1VBN", g=True)
    # false VBD -> should be VBN (dependent on VBZ->POS change above)...
    tree.sub(r"(( have| 've| having| has|VBZ 's| had| 'd)\)[^\)]*)VBD", r"\1VBN", g=True)
    ## for each constituent...
    while tree.search(r"\([^\(\)]*\)"):
        ## convert outer parens to braces...
        tree.sub(r"\(([^\(\)]*)\)", r"{\1}")
        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
        # false VBD in conj following initial VBN -> should be VBN, for every subsequent conjunct
        while tree.sub(
            r"{(VP[^ ]* *<VP[^>\]]*VBN[^>]*>(?= .*<CC[^>]*>.*\}).*<VP[^>\]]*)VBD(.*)}", r"{\1VBN\2}"
        ):
            pass
        # false VBN in conj following initial VBD -> should be VBD, for every subsequent conjunct
        while tree.sub(
            r"{(VP[^ ]* *<VP[^>\]]*VBD[^>]*>(?= .*<CC[^>]*>.*\}).*<VP[^>\]]*)VBN(.*)}", r"{\1VBD\2}"
        ):
            pass
        ####################
        ## convert inner angles (if any) to bracks...
        while tree.sub("({[^{}]*)<([^<>]*)>", r"\1\[\2\]"):
            pass
        ## convert outer braces to angles...
        tree.sub("{(.*)}", r"<\1>")
    ## finish up...
    tree.sub("<", "[")
    tree.sub(">", "]")
    ## translate to parens again...
    tree.sub(r"\[", r"\(", g=True)
    tree.sub(r"\]", r"\)", g=True)
    ## VBN .. by without (-NONE- *) gets it...
    tree.sub(
        r"\(VBN (?!traveled|forced|pressured|said)([^ \)]*)\)([^ ]* *\(PP \(IN by)",
        r"\(VBN \1\) \(NP \(-NONE- \*\)\)\2",
        g=True,
    )
    ## output...
