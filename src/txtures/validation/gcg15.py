"""The gcg15 reannotation against ModelBlocks' Perl pipeline, section by section."""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from prsr.gcg import fail_tree, fixes, kill_unaries, rules
from prsr.gcg.perl import PerlString

from txtures.validation import Paths, run

STAGES = (("fixed", ".fixed.senttrees"), ("annotated", ".annotated"), ("final", ".gcg15.senttrees"))


def reference(paths: Paths, section: str) -> None:
    """ModelBlocks' `%.gcg15.senttrees` for one WSJ section, keeping each stage.

    The steps are those of `resource-treebank/Makefile` (`genmodel/wsj%.senttrees`)
    and `resource-gcg/Makefile` (`%.fixed.senttrees`, `%.senttrees`).
    """
    mb, out = paths.modelblocks, paths.gcg15
    base = out / f"wsj{section}"
    run(
        f"""
        rm -f {base}.senttrees
        for f in {paths.wsj / section}/*.mrg; do
          echo '!ARTICLE' >> {base}.senttrees
          perl {mb}/resource-linetrees/scripts/editabletrees2linetrees.pl < $f >> {base}.senttrees
        done
        perl {mb}/resource-gcg/scripts/make-trees-fixed.pl < {base}.senttrees > {base}.fixed.senttrees
        perl {mb}/resource-gcg/scripts/annotate-gcg15.pl < {base}.fixed.senttrees > {base}.annotated 2>/dev/null
        perl {mb}/resource-linetrees/scripts/killUnaries.pl -l < {base}.annotated \\
          | python3 {mb}/resource-gcg/scripts/makefailtrees.py \\
          | perl -pe 's/\\(([^ ]*)-f[^ ]*/\\(\\1/g' \\
          | perl -pe 's/\\(([^ ]*)-l([A-Z])([^ ]*)/\\(\\1\\3-l\\2/g;' > {base}.gcg15.senttrees
        """
    )


def stages(line: str) -> tuple[str, str, str]:
    """txtures' output for one line after the fixes, the annotation, and the end."""
    text = PerlString(line + "\n", fixes.VARIABLES)
    fixes.fix(text)
    fixed = text.text
    text = PerlString(fixed, rules.VARIABLES)
    rules.annotate(text)
    annotated = text.text
    final = fail_tree(kill_unaries(annotated))
    final = re.sub(r"\(([^ ]*)-f[^ ]*", r"(\1", final)
    final = re.sub(r"\(([^ ]*)-l([A-Z])([^ ]*)", r"(\1\3-l\2", final)
    return fixed.rstrip("\n"), annotated.rstrip("\n"), final.rstrip("\n")


def _check(job):
    line, references = job
    for (name, _), mine, theirs in zip(STAGES, stages(line), references, strict=True):
        if mine != theirs:
            return name, line, mine, theirs
    return None


def compare(paths: Paths, section: str, processes: int = 8) -> int:
    """Compare txtures with the reference stages of one section; print and return the mismatches."""
    base = paths.gcg15 / f"wsj{section}"
    source = Path(f"{base}.senttrees").read_text(encoding="utf-8").splitlines()
    references = [
        Path(f"{base}{suffix}").read_text(encoding="utf-8").splitlines() for _, suffix in STAGES
    ]
    jobs = [(line, [r[i] for r in references]) for i, line in enumerate(source)]
    mismatches = 0
    with ProcessPoolExecutor(processes) as pool:
        for found in pool.map(_check, jobs, chunksize=16):
            if found is not None:
                mismatches += 1
                if mismatches <= 3:
                    name, line, mine, theirs = found
                    print(
                        f"{name} differs for {line[:120]}\n  txtures {mine[:200]}\n  ref     {theirs[:200]}"
                    )
    print(f"WSJ {section}: {len(source)} lines, {mismatches} mismatches")
    return mismatches
