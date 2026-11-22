# GCG treebank

`gcg_treebank` is a resource holding sections of the Wall Street Journal
Penn Treebank reannotated into generalized categorial grammar (GCG), version
gcg15. It is the training data for the incremental parsers that txtures is
gaining. No module uses it yet.

## Providing the treebank

The Penn Treebank is licensed by the Linguistic Data Consortium (Treebank-3,
LDC99T42). txtures cannot ship it or download it, so each site points txtures
at its own copy:

```bash
txtures site set corpora.ptb3=/data/PTB3
```

The directory may be Treebank-3's root or its `treebank_3` directory; either
way, the trees are read from `parsed/mrg/wsj/NN/*.mrg`. A request may name
another copy instead, with `treebank` set to its absolute path.

The resource is identified by the contents of the files it reads, not by
their path, so moving or copying the treebank rebuilds nothing, and a copy
with different files is a different resource. Its trees are derived from
the treebank and fall under the same license: they stay in the resource
store and should not be copied out of the site.

## Building it

```bash
txtures resources build --kind gcg_treebank                       # sections 02-21
txtures resources build --kind gcg_treebank --spec sections=[23]  # section 23
```

| Spec key | Default | Meaning |
| --- | --- | --- |
| `treebank` | `ptb3` | `ptb3` for the `corpora.ptb3` setting, or the absolute path of a copy |
| `sections` | 02 to 21 | the WSJ sections, as numbers |

Where Slurm is set up, the build runs as a job without GPUs, with 16 cores;
on 16 cores, sections 02-21 take a few minutes. The resource holds
`trees.txt.gz`, one tree per line, with a `!ARTICLE` line before the trees
of each treebank file, and `properties.json`, with the number of trees and
of `FAIL` trees.

## The reannotation

The trees are made exactly as ModelBlocks makes `%.gcg15.senttrees`:

1. Each tree of a `.mrg` file is put on one line.
2. Known errors of the treebank are repaired.
3. The gcg15 rules reannotate each tree: they propagate head categories
   up, then relabel each constituent top-down with a GCG category, its
   attachment operator (`-lA` argument, `-lM` modifier, `-lC` conjunct, and
   others), and its gaps, and finally remove the temporary annotations.
4. Unary branches above lexical chains are collapsed and empty categories
   removed.
5. A tree whose reannotation failed (some category has no lowercase
   letter, such as an untranslated `NP`) becomes a right-branching tree of
   `FAIL` nodes over its words. About 7% of WSJ trees fail this way.
6. The original treebank categories (`-f` tags) are removed and each
   attachment operator is moved to the end of its category.

These steps run in the [prsr](https://prsr.readthedocs.io) package
(`prsr.gcg`), a Python port of the
ModelBlocks scripts
`editabletrees2linetrees.pl`, `make-trees-fixed.pl`, `annotate-gcg15.pl`,
`killUnaries.pl`, and `makefailtrees.py`, with the rules of the Perl
scripts kept verbatim. On all 49,208 trees of WSJ sections 00-24 the port
gives the same output as ModelBlocks at every step.

```python
from prsr.gcg import ptb_lines, reannotate

for line in ptb_lines(open("wsj_0001.mrg").read()):
    print(reannotate(line))
```
