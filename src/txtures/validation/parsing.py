"""The GPU synproc parser against ModelBlocks' `synproc`, on treebank sentences."""

from __future__ import annotations

import json
import math
import time

import torch

from txtures.gcg import ptb_lines
from txtures.lvpcfg import leftcorner as lc
from txtures.lvpcfg.parser import Parser
from txtures.lvpcfg.trees import parse as parse_tree
from txtures.validation import Paths


def sentences(paths: Paths, section: str, count: int) -> str:
    """Write the first `count` sentences of a WSJ section as `.linetoks` (words, empties removed)."""
    found: list[str] = []
    for path in sorted((paths.wsj / section).glob("*.mrg")):
        for line in ptb_lines(path.read_text(encoding="utf-8")):
            words: list[str] = []
            stack = [parse_tree("(" + line + ")")]
            while stack:
                node = stack.pop()
                if node.is_preterminal:
                    if node.label != "-NONE-":
                        words.append(node.children[0].label)
                else:
                    stack.extend(reversed(node.children))
            found.append(" ".join(words))
            if len(found) == count:
                name = f"wsj{section}-{count}"
                (paths.mb / f"{name}.linetoks").write_text(
                    "\n".join(found) + "\n", encoding="utf-8"
                )
                return name
    raise ValueError(f"WSJ {section} has fewer than {count} sentences")


def parse(paths: Paths, grammar: str, name: str, beam: int = 5000) -> None:
    """Parse a `.linetoks` file with txtures on the GPU, from a Berkeley dump; time each sentence."""
    source = paths.berkeley / grammar
    start = time.time()
    pcfg = lc.Pcfg.from_berkeley_text(open(f"{source}.grammar"), open(f"{source}.lexicon"))
    model = lc.compile_model(pcfg.without_unaries(), "cuda")
    parser = Parser(model, beam=beam)
    torch.cuda.synchronize()
    print(f"load and compile {time.time() - start:.1f}s", flush=True)
    text = (paths.mb / f"{name}.linetoks").read_text(encoding="utf-8").splitlines()
    results = []
    for i, words in enumerate(line.split() for line in text):
        start = time.time()
        [result] = parser.parse([words])
        torch.cuda.synchronize()
        print(f"sentence {i}: {len(words)} words, {time.time() - start:.2f}s", flush=True)
        results.append(
            {
                "words": words,
                "surprisal": result["surprisal"],
                "preterminals": [model.symbols[s[0]] if s else None for s in result["states"]],
            }
        )
    out = paths.mb / f"{name}.{grammar}.txtures.json"
    out.write_text(json.dumps(results), encoding="utf-8")
    print(f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak; wrote {out}")


def read_tokdecs(path, lengths: list[int]) -> list[list[tuple[str, str, float]]]:
    """synproc's output (`-c`): a header, then `word hidden-state surprisal` per word."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("word pos"):
            continue
        fields = line.split(" ")
        rows.append((fields[0], fields[1] if len(fields) > 2 else "", float(fields[-1])))
    out, position = [], 0
    for n in lengths:
        out.append(rows[position : position + n])
        position += n
    return out


def compare(paths: Paths, grammar: str, name: str) -> None:
    """Per-word surprisal of txtures and ModelBlocks on the same sentences."""
    mine = json.loads((paths.mb / f"{name}.{grammar}.txtures.json").read_text(encoding="utf-8"))
    theirs = read_tokdecs(paths.mb / f"{name}.{grammar}.tokdecs", [len(s["words"]) for s in mine])
    worst, words, failed = 0.0, 0, 0
    for sentence, reference in zip(mine, theirs, strict=True):
        for word, a, (ref_word, _, b) in zip(
            sentence["words"], sentence["surprisal"], reference, strict=True
        ):
            assert word == ref_word, (word, ref_word)
            words += 1
            if math.isnan(b):
                failed += 1
                continue
            worst = max(worst, abs(a - b))
    print(
        f"{len(mine)} sentences, {words} words: max |surprisal difference| {worst:.3g} bits; "
        f"{failed} words where ModelBlocks failed"
    )
