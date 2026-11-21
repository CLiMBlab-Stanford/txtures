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


def gold_sentences(paths: Paths, section: str, count: int) -> str:
    """Write the first `count` sentences of a section whose gold gcg15 tree is not a FAIL tree.

    `NAME.linetoks` holds their words and `NAME.gold` their gold gcg15 trees
    (from `gcg15-reference SECTION`), one per line.
    """
    trees = [
        line
        for line in (paths.gcg15 / f"wsj{section}.gcg15.senttrees")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("(") and not line.startswith("(FAIL ")
    ][:count]
    if len(trees) < count:
        raise ValueError(f"WSJ {section} has fewer than {count} non-FAIL sentences")
    words = [" ".join(parse_tree("(" + tree + ")").leaves()) for tree in trees]
    name = f"wsj{section}-gold{count}"
    (paths.mb / f"{name}.linetoks").write_text("\n".join(words) + "\n", encoding="utf-8")
    (paths.mb / f"{name}.gold").write_text("\n".join(trees) + "\n", encoding="utf-8")
    return name


def load_grammar(paths: Paths, grammar: str) -> lc.Pcfg:
    """A grammar as a `Pcfg`: a Berkeley text dump in the Berkeley work directory, or
    `lvpcfg:IDENTITY` (a prefix is enough) for an `lvpcfg_grammar` that txtures trained."""
    if not grammar.startswith("lvpcfg:"):
        source = paths.berkeley / grammar
        return lc.Pcfg.from_berkeley_text(open(f"{source}.grammar"), open(f"{source}.lexicon"))
    from txtures import api
    from txtures.lvpcfg.resource import load

    prefix = grammar.split(":", 1)[1]
    store = api.settings().resource_store / "lvpcfg_grammar"
    found = [p for p in store.glob(f"{prefix[:2]}/{prefix}*") if (p / "manifest.json").exists()]
    if len(found) != 1:
        raise ValueError(f"{len(found)} lvpcfg_grammar entries match {prefix!r} in {store}")
    trained = load(found[0])
    return lc.pcfg_from_trained(trained.categories, trained.grammar, trained.lexicon)


def parse(
    paths: Paths,
    grammar: str,
    name: str,
    beam: int = 5000,
    merge: str = "once",
    max_sentences: int = 32,
    device: str = "cuda",
) -> None:
    """Parse a `.linetoks` file with txtures on the GPU, from a Berkeley dump, and time it.

    All sentences go through one continuously batched run with at most
    `max_sentences` active at a time; loading and compiling are timed apart.
    `device` is a torch device, such as `cuda` or `cpu`.
    """
    cuda = device.startswith("cuda")

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize()

    start = time.time()
    pcfg = load_grammar(paths, grammar)
    model = lc.compile_model(pcfg.without_unaries(), device)
    parser = Parser(model, beam=beam, lexicon_merge=merge)
    synchronize()
    load = time.time() - start
    text = (paths.mb / f"{name}.linetoks").read_text(encoding="utf-8").splitlines()
    sentences = [line.split() for line in text]
    start = time.time()
    found = parser.parse(sentences, max_sentences=max_sentences)
    synchronize()
    elapsed = time.time() - start
    words = sum(len(s) for s in sentences)
    print(
        f"load and compile {load:.1f}s; {len(sentences)} sentences, {words} words in {elapsed:.1f}s "
        f"({elapsed / words:.4f} s per word, {max_sentences} sentences at a time); "
        f"on {device}"
        + (f"; {torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak" if cuda else ""),
        flush=True,
    )
    results = [
        {
            "words": words_,
            "surprisal": result["surprisal"],
            "preterminals": [model.symbols[s[0]] if s else None for s in result["states"]],
            "states": result["states"],
        }
        for words_, result in zip(sentences, found, strict=True)
    ]
    suffix = "" if cuda else f".{device}"
    out = paths.mb / f"{name}.{grammar.replace(':', '-')}.txtures-{merge}{suffix}.json"
    out.write_text(json.dumps({"symbols": model.symbols, "sentences": results}), encoding="utf-8")
    print(f"wrote {out}")


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


def compare(paths: Paths, grammar: str, name: str, merge: str = "once") -> None:
    """Per-word surprisal and preterminals of txtures and ModelBlocks on the same sentences."""
    mine = json.loads(
        (paths.mb / f"{name}.{grammar}.txtures-{merge}.json").read_text(encoding="utf-8")
    )["sentences"]
    theirs = read_tokdecs(
        paths.mb / f"{name}.{grammar.split('.')[0]}.tokdecs", [len(s["words"]) for s in mine]
    )
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


def debug_beams(
    paths: Paths, grammar: str, name: str, beam: int = 5000, merge: str = "synproc"
) -> None:
    """Write txtures' lexical entries and full beams per word in synproc_debug's format.

    For each word: `WORD:w`, its `LEX tag prob` lines, then `BEAM (w)` with one
    `logprob tag f j A/B;A/B` line per entry, best first.
    """
    source = paths.berkeley / grammar
    pcfg = lc.Pcfg.from_berkeley_text(open(f"{source}.grammar"), open(f"{source}.lexicon"))
    model = lc.compile_model(pcfg.without_unaries(), "cuda")
    parser = Parser(model, beam=beam, lexicon_merge=merge)
    sym = model.symbols
    lines = []
    for words in (paths.mb / f"{name}.linetoks").read_text(encoding="utf-8").split("\n"):
        words = words.split()
        if not words:
            continue
        history: list = []
        original_step = parser.step

        def recording_step(beams, ws, last, history=history, original_step=original_step):
            new, done = original_step(beams, ws, last)
            history.append(new)
            return new, done

        parser.step = recording_step
        parser.parse([words])
        parser.step = original_step
        for word, beams in zip(words, history, strict=True):
            lines.append(f"WORD:{word}")
            tags, probs = parser.word_entries(word)
            for t, p in zip(tags.tolist(), probs.tolist(), strict=True):
                lines.append(f"LEX {sym[t]} {p!r}")
            lines.append(f"BEAM ({word})")
            order = torch.argsort(beams.score, descending=True)
            for i in order.tolist():
                size = int(beams.size[i])
                store = ";".join(
                    f"{sym[int(beams.store[i, k, 0])]}/{sym[int(beams.store[i, k, 1])]}"
                    for k in range(size)
                )
                lines.append(
                    f"{float(beams.score[i])!r} {sym[int(beams.pretrm[i])]} {int(beams.fork[i])} {int(beams.join[i])} {store}"
                )
    out = paths.mb / f"{name}.{grammar}.txtures-{merge}.verbose"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def read_verbose(path) -> list[dict]:
    """synproc_debug's output: per word occurrence, its lexical entries and beam."""
    words, current, section = [], None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("WORD:"):
            current = {"word": line[5:], "lex": {}, "beam": {}}
            words.append(current)
            section = "lex"
        elif line.startswith("BEAM ("):
            section = "beam"
        elif current is None or not line.strip():
            continue
        elif section == "lex" and line.startswith("LEX "):
            _, tag, p = line.split(" ")
            current["lex"][tag] = float(p)
        elif section == "beam":
            fields = line.split(" ")
            if len(fields) < 4 or fields[0] in ("word",):
                continue
            try:
                score = float(fields[0])
            except ValueError:
                continue
            store = fields[4] if len(fields) > 4 and not fields[4].startswith("me:") else ""
            current["beam"][" ".join(fields[1:4] + [store])] = score
    return words


def compare_verbose(paths: Paths, grammar: str, name: str, merge: str = "synproc") -> None:
    """Lexical entries and beams of txtures against synproc_debug's, word by word."""
    theirs = read_verbose(paths.mb / f"{name}.{grammar.split('.')[0]}.verbose")
    mine = read_verbose(paths.mb / f"{name}.{grammar}.txtures-{merge}.verbose")
    for a, b in zip(mine, theirs, strict=True):
        assert a["word"] == b["word"]
        lex_tags = set(a["lex"]) ^ set(b["lex"])
        lex_diff = max(
            (
                abs(a["lex"][t] - b["lex"][t]) / b["lex"][t]
                for t in set(a["lex"]) & set(b["lex"])
                if b["lex"][t]
            ),
            default=0.0,
        )
        shared = set(a["beam"]) & set(b["beam"])
        score_diff = max((abs(a["beam"][k] - b["beam"][k]) for k in shared), default=0.0)
        print(
            f"{a['word']:12} lex: {len(a['lex'])} vs {len(b['lex'])} tags, {len(lex_tags)} differ, "
            f"rel {lex_diff:.1e}; beam: {len(a['beam'])} vs {len(b['beam'])}, shared {len(shared)}, "
            f"max |log p diff| {score_diff:.2e}"
        )
