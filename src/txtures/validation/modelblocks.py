"""The left-corner compilation and the synproc parser against ModelBlocks' programs."""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from txtures.lvpcfg import leftcorner as lc
from txtures.validation import DATA, Paths, dumps, run

PROGRAMS = {
    "ccmodel2synproccptmodel": "resource-incrsem/src/ccmodel2synproccptmodel.cpp",
    "synproc": "resource-incrsem/src/synproc.cpp",
}


def build(paths: Paths) -> None:
    """Compile ModelBlocks' compiler and parser in the work directory.

    Two local changes, both in parsing numbers: `stod` becomes `strtod` in
    `Delimited<double>` and `Double`, because `stod` throws on the subnormal
    probabilities `ccu2cc.py` can write; and armadillo, which `Delimited.hpp`
    includes only for vector types these programs do not use, is replaced
    by a stub (`txtures/validation/data/stub/armadillo`).
    """
    mb, out = paths.modelblocks, paths.mb
    patched = out / "patched"
    patched.mkdir(exist_ok=True)
    fix = "sed 's/val(stod(ps))/val(strtod(ps, nullptr))/'"
    run(f"{fix} {mb}/resource-logreg/include/Delimited.hpp > {patched}/Delimited.hpp")
    includes = (
        f"-I{DATA / 'stub'} -I{patched} -I{mb}/resource-incrsem/include -I{mb}/resource-logreg/include "
        f"-I{mb}/resource-incrsem/src -I{mb}/resource-rvtl"
    )
    run(f"gcc -O2 -w -o {out / 'evalb'} {mb}/resource-linetrees/src/evalb.c")
    for program, source in PROGRAMS.items():
        run(f"{fix} {mb}/{source} > {patched}/{program}.cpp")
        run(
            f"g++ {includes} -O3 -DNDEBUG -fpermissive -std=c++17 -w {patched}/{program}.cpp -lm -lpthread -o {out / program}"
        )


def compile_reference(paths: Paths, grammar: str) -> None:
    """ModelBlocks' chain from a Berkeley text dump to a `.synprocmodel`, timed per stage
    (`resource-lcparse/Makefile` `%.x-ccu.model`, `%.x-cc.model`; `resource-incrsem/Makefile`
    `%.synprocmodel`). `grammar` names a dump in the Berkeley work directory, such as `seed2.3sm`."""
    scripts = paths.modelblocks / "resource-lcparse" / "scripts"
    source = paths.berkeley / grammar
    run(
        f"""
        cd {paths.mb}
        time ( sed 's/[+]/-/g' < {source}.grammar | python3 {scripts}/berkgrammar2ckygr.py > {grammar}.x-ccu.model
               sed 's/[+]/-/g' < {source}.lexicon | python3 {scripts}/berklexicon2ckylex.py >> {grammar}.x-ccu.model )
        time ( python3 {scripts}/ccu2cc.py < {grammar}.x-ccu.model > {grammar}.x-cc.model )
        time ( sed 's/^Cr/R/' < {grammar}.x-cc.model | ./ccmodel2synproccptmodel > {grammar}.synprocmodel )
        """
    )


def parse_reference(
    paths: Paths,
    grammar: str,
    sentences: str,
    beam: int = 5000,
    threads: int = 8,
    part: int | None = None,
    parts: int | None = None,
) -> None:
    """ModelBlocks' synproc with its standard flags (`-c -u -b5000`) on a `.linetoks` file.

    With `part` and `parts`, only the part-th of `parts` contiguous chunks is
    parsed (for Slurm arrays); `merge_parts` joins their outputs.
    """
    name = sentences
    if parts is not None:
        lines = (paths.mb / f"{sentences}.linetoks").read_text(encoding="utf-8").splitlines()
        size = -(-len(lines) // parts)
        name = f"{sentences}.part{part:03d}"
        chunk = lines[part * size : (part + 1) * size]
        (paths.mb / f"{name}.linetoks").write_text("\n".join(chunk) + "\n", encoding="utf-8")
    run(
        f"cd {paths.mb} && /usr/bin/time -v ./synproc -c -u -b{beam} -p{threads} {grammar}.synprocmodel "
        f"< {name}.linetoks > {name}.{grammar}.tokdecs 2> {name}.{grammar}.stderr"
    )


def merge_parts(paths: Paths, grammar: str, sentences: str, parts: int) -> None:
    """Join the chunked synproc outputs of `parse_reference` in order (one header kept)."""
    out = []
    for part in range(parts):
        lines = (
            (paths.mb / f"{sentences}.part{part:03d}.{grammar}.tokdecs")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        out += [line for line in lines if not line.startswith("word pos")]
    (paths.mb / f"{sentences}.{grammar}.tokdecs").write_text(
        "word pos f j store totsurp\n" + "\n".join(out) + "\n", encoding="utf-8"
    )


def _read_ccu(path) -> tuple[dict, dict, dict, dict]:
    cc, x, cr, cu = defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
    for line in path.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("CC "):
            lhs, rhs = line[3:].split(" : ")
            pair, value = rhs.rsplit(" = ", 1)
            left, right = pair.split(" ")
            cc[lhs][(left, right)] = float(value)
        elif line.startswith("X "):
            lhs, rhs = line[2:].split(" : ", 1)
            word, value = rhs.rsplit(" = ", 1)
            x[lhs][word] = float(value)
        elif line.startswith("Cr : "):
            child, value = line[5:].rsplit(" = ", 1)
            cr[child] = float(value)
        elif line.startswith("Cu "):
            lhs, rhs = line[3:].split(" : ")
            child, value = rhs.rsplit(" = ", 1)
            cu[lhs][child] = float(value)
    return cc, x, cr, cu


def _with_terminals(pcfg: lc.Pcfg) -> dict:
    table = dumps.flat(pcfg.binary)
    table.update({(c, (lc.TERMINAL, lc.TERMINAL)): v for c, v in pcfg.terminal.items()})
    return table


def compare_folding(paths: Paths, grammar: str) -> None:
    """Reading the dump and folding unaries, against ModelBlocks' `x-ccu` and `x-cc` models."""
    source = paths.berkeley / grammar
    pcfg = lc.Pcfg.from_berkeley_text(open(f"{source}.grammar"), open(f"{source}.lexicon"))
    cc, x, cr, cu = _read_ccu(paths.mb / f"{grammar}.x-ccu.model")
    dumps.compare(_with_terminals(pcfg), dumps.flat(cc), "x-ccu binary and terminal")
    dumps.compare(dumps.flat(pcfg.lexicon), dumps.flat(x), "x-ccu lexicon")
    dumps.compare(dumps.flat(pcfg.unary), dumps.flat(cu), "x-ccu unary")
    dumps.compare(pcfg.root, cr, "x-ccu root")
    folded = pcfg.without_unaries()
    cc, x, cr, _ = _read_ccu(paths.mb / f"{grammar}.x-cc.model")
    dumps.compare(_with_terminals(folded), dumps.flat(cc), "x-cc binary and terminal")
    dumps.compare(dumps.flat(folded.lexicon), dumps.flat(x), "x-cc lexicon")
    dumps.compare(folded.root, cr, "x-cc root")


def compare_compiled(paths: Paths, grammar: str) -> None:
    """Every F, P, W, J, A, and B entry of ModelBlocks' `.synprocmodel`, from txtures' factors."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = paths.berkeley / grammar
    start = time.time()
    pcfg = lc.Pcfg.from_berkeley_text(
        open(f"{source}.grammar"), open(f"{source}.lexicon")
    ).without_unaries()
    model = lc.compile_model(pcfg, device)
    if device == "cuda":
        torch.cuda.synchronize()
    print(
        f"read, folded, and compiled on {device} in {time.time() - start:.1f}s; {len(model.symbols)} symbols"
    )
    ix, depths = model.index, lc.MAX_DEPTH
    plus, left_corner = model.plus.cpu().numpy(), model.left_corner.cpu().numpy()
    f = {d: [x.cpu().numpy() for x in lc.f_table(model, d)] for d in range(depths + 1)}
    tg = {d: lc._terminal_gamma(model, d, lc.LEFT).cpu().numpy() for d in range(depths + 2)}
    j = {d: [x.cpu().numpy() for x in lc.j_table(model, d)] for d in range(depths + 1)}
    b = {
        (d, s): lc.b_conditional(model, d, s)
        for d in range(depths + 2)
        for s in (0, 1)
        if lc._valid(d, s)
    }
    stats = defaultdict(lambda: [0, 0.0, 0])

    def note(kind, mine, ref):
        entry = stats[kind]
        entry[0] += 1
        if mine is None:
            entry[2] += 1
        else:
            entry[1] = max(entry[1], abs(mine - ref) / max(abs(ref), 1e-300))

    for line in open(paths.mb / f"{grammar}.synprocmodel", encoding="utf-8"):
        kind = line[0]
        lhs, rhs = line[2:].rstrip("\n").split(" : ", 1)
        out, value = rhs.rsplit(" = ", 1)
        value, fields = float(value), lhs.split(" ")
        try:
            if kind == "F":
                d, bb = int(fields[0]), ix[fields[1]]
                f0, f1 = f[d][0][bb], f[d][1][bb]
                note("F", (f1 if out == "1" else f0) / (f0 + f1), value)
            elif kind == "P":
                d, fork, bb = int(fields[0]), int(fields[1]), ix[fields[2]]
                if fork == 1:
                    note("P", plus[d][bb, ix[out]] * tg[d + 1][ix[out]] / f[d][1][bb], value)
                else:
                    note("P", 1.0 if out == fields[2] else 0.0, value)
            elif kind == "W":
                note("W", pcfg.lexicon[fields[0]][out], value)
            elif kind == "J":
                dd, bb, c0 = int(fields[0]), ix[fields[1]], ix[fields[2]]
                j0, j1 = j[dd - 1][0][bb, c0], j[dd - 1][1][bb, c0]
                note("J", (j1 if out == "1" else j0) / (j0 + j1), value)
            elif kind == "A":
                dd, join, bb = int(fields[0]), int(fields[1]), ix[fields[2]]
                if join == 0:
                    c0, a = ix[fields[3]], ix[out]
                    note(
                        "A",
                        plus[dd - 1][bb, a] * left_corner[dd][a, c0] / j[dd - 1][0][bb, c0],
                        value,
                    )
                else:
                    note("A", 1.0 if out == fields[2] else 0.0, value)
            elif kind == "B":
                dd, join, a, c0 = int(fields[0]), int(fields[1]), ix[fields[2]], ix[fields[3]]
                note("B", b[(dd, 0 if join == 0 else 1)].get((a, c0), {}).get(ix[out]), value)
        except KeyError:
            note(kind, None, value)
    for kind, (count, worst, missing) in sorted(stats.items()):
        print(
            f"{kind}: {count} entries, max relative difference {worst:.2e}, missing in txtures {missing}"
        )


def compare_export(paths: Paths, grammar: str) -> None:
    """`leftcorner.pcfg_from_trained` on a dumped Java grammar, against Berkeley's own text dump."""
    from txtures.validation.berkeley import INPUT

    state = dumps.load(paths.berkeley / "dumps" / f"{grammar}.gr.before")
    corpus = dumps.training_corpus(paths.berkeley / f"{INPUT}.senttrees", state.categories)
    trained = {node[1] for tree in corpus.nodes for node in tree if node[0] == "unary"}
    mine = lc.pcfg_from_trained(state.categories, state.grammar(trained), state.lexicon())
    source = paths.berkeley / grammar
    ref = lc.Pcfg.from_berkeley_text(open(f"{source}.grammar"), open(f"{source}.lexicon"))
    dumps.compare(dumps.flat(mine.binary), dumps.flat(ref.binary), "binary")
    dumps.compare(dumps.flat(mine.unary), dumps.flat(ref.unary), "unary (Viterbi closure)")
    dumps.compare(mine.root, ref.root, "root")
    dumps.compare(dumps.flat(mine.lexicon), dumps.flat(ref.lexicon), "lexicon")
    dumps.compare(mine.terminal, ref.terminal, "terminal")
