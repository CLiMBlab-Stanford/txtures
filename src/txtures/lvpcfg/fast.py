"""EM iterations with the grammar and lexicon kept as tensors, for training on a GPU.

`txtures.lvpcfg.em` and `txtures.lvpcfg.lexicon` define the computation and
are checked against Berkeley's trainer; this module performs the same
computation on tensors, and `test_lvpcfg` checks it against them:

- lexical scores for every (tag, word) entry at once (`Lexicon.score`);
- lexical counts by scatter-adding token posteriors, then the rare-word
  tying and pruning (`Lexicon.train_token`, `tie_rare_words`, `prune`);
- the grammar M-step by scatter-adding parent totals (`em.normalize`), with
  smoothing as a matrix product over parent substates.

Sums are taken in a different order than Berkeley's, so results differ at
the level of rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from txtures.lvpcfg.em import Corpus, Grammar, prepare
from txtures.lvpcfg.inside import expected_counts
from txtures.lvpcfg.lexicon import PRUNE, RARE, SMOOTH_IN_UNKNOWNS, SMOOTHING, Lexicon


@dataclass
class Tokens:
    """The corpus's tokens and lexicon entries as index arrays (built once)."""

    entry: torch.Tensor  # token -> (tag, word) entry
    unseen_entry: torch.Tensor  # token -> (tag, signature) entry
    tag: torch.Tensor  # token -> tag
    word: torch.Tensor  # token -> word in `vocabulary`
    signature: torch.Tensor  # token -> signature in `vocabulary`
    entry_tag: torch.Tensor
    entry_word: torch.Tensor  # in `vocabulary`
    unseen_tag: torch.Tensor
    entries: list[tuple[int, str]]
    unseen_entries: list[tuple[int, str]]
    vocabulary: list[str]  # words and signatures


def tokens(corpus: Corpus, device: str) -> Tokens:
    key = ("tokens", str(device))
    if key in corpus._cache:
        return corpus._cache[key]
    vocabulary: dict[str, int] = {}
    entries: dict[tuple[int, str], int] = {}
    unseen: dict[tuple[int, str], int] = {}
    arrays = {k: [] for k in ("entry", "unseen", "tag", "word", "sig")}
    for (word, tag, _), sig in zip(corpus.tokens, corpus.signatures, strict=True):
        arrays["entry"].append(entries.setdefault((tag, word), len(entries)))
        arrays["unseen"].append(unseen.setdefault((tag, sig), len(unseen)))
        arrays["tag"].append(tag)
        arrays["word"].append(vocabulary.setdefault(word, len(vocabulary)))
        arrays["sig"].append(vocabulary.setdefault(sig, len(vocabulary)))

    def t(values) -> torch.Tensor:
        return torch.as_tensor(np.array(values, dtype=np.int64), device=device)

    found = Tokens(
        t(arrays["entry"]),
        t(arrays["unseen"]),
        t(arrays["tag"]),
        t(arrays["word"]),
        t(arrays["sig"]),
        t([tag for tag, _ in entries]),
        t([vocabulary[word] for _, word in entries]),
        t([tag for tag, _ in unseen]),
        list(entries),
        list(unseen),
        list(vocabulary),
    )
    corpus._cache[key] = found
    return found


@dataclass
class State:
    """A grammar and lexicon as padded tensors (`S` substates at most)."""

    substates: dict[int, int]
    binary: torch.Tensor  # [rules, S, S, S]
    unary: torch.Tensor  # [rules, S, S]
    binary_parent: torch.Tensor
    unary_parent: torch.Tensor
    unary_self: torch.Tensor  # unary rules from a category to itself
    binary_left: torch.Tensor
    binary_right: torch.Tensor
    unary_child: torch.Tensor
    seen: torch.Tensor  # [entries, S]: c_TW
    unseen: torch.Tensor  # [unseen entries, S]: signature counts
    tag_count: torch.Tensor  # [categories, S]
    unseen_tag_count: torch.Tensor  # [categories, S]
    word_count: torch.Tensor  # [vocabulary]
    total: float
    total_unseen: float
    grammar_smoother: torch.Tensor | None  # [categories, S, S]
    lexicon_smoother: torch.Tensor | None
    device: str = "cpu"  # as given to `to_state`, which keys the corpus's caches


def smoother_tensor(table, categories: int, size: int, device) -> torch.Tensor | None:
    if table is None:
        return None
    out = np.zeros((categories, size, size))
    for c, matrix in table.items():
        out[c, : matrix.shape[0], : matrix.shape[1]] = matrix
    return torch.as_tensor(out, device=device)


def to_state(corpus: Corpus, grammar: Grammar, lexicon: Lexicon, device: str = "cpu") -> State:
    """Copy a dictionary grammar and lexicon into tensors."""
    prepared = prepare(corpus, grammar, lexicon, True, device)
    tok = tokens(corpus, device)
    size = max(grammar.substates.values())
    categories = max(grammar.substates) + 1

    def rows(table, keys, n_of) -> np.ndarray:
        out = np.zeros((len(keys), size))
        for i, (tag, item) in enumerate(keys):
            values = table.get(tag, {}).get(item)
            if values is not None:
                out[i, : len(values)] = values
        return out

    def per_tag(table) -> np.ndarray:
        out = np.zeros((categories, size))
        for tag, values in table.items():
            out[tag, : len(values)] = values
        return out

    binary_keys = sorted(prepared.binary_index, key=prepared.binary_index.get)
    unary_keys = sorted(prepared.unary_index, key=prepared.unary_index.get)

    def t(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), device=device)

    return State(
        dict(grammar.substates),
        prepared.binary,
        prepared.unary,
        t(np.array([k[0] for k in binary_keys], dtype=np.int64)),
        t(np.array([k[0] for k in unary_keys], dtype=np.int64)),
        t(np.array([k[0] == k[1] for k in unary_keys], dtype=bool)),
        t(np.array([k[1] for k in binary_keys], dtype=np.int64)),
        t(np.array([k[2] for k in binary_keys], dtype=np.int64)),
        t(np.array([k[1] for k in unary_keys], dtype=np.int64)),
        t(rows(lexicon.word_tag, tok.entries, None)),
        t(rows(lexicon.unseen_word_tag, tok.unseen_entries, None)),
        t(per_tag(lexicon.tag_count)),
        t(per_tag(lexicon.unseen_tag_count)),
        t(np.array([lexicon.word_count.get(w, 0.0) for w in tok.vocabulary])),
        float(lexicon.total),
        float(lexicon.total_unseen),
        smoother_tensor(grammar.smoother, categories, size, device),
        smoother_tensor(lexicon.smoother, categories, size, device),
        str(device),
    )


def to_dicts(
    corpus: Corpus, state: State, grammar: Grammar, lexicon: Lexicon
) -> tuple[Grammar, Lexicon]:
    """Copy a state back into dictionaries shaped like `grammar` and `lexicon`."""
    prepared = prepare(corpus, grammar, lexicon, True, state.device)
    tok = tokens(corpus, state.device)
    sub = state.substates
    binary = state.binary.cpu().numpy()
    unary = state.unary.cpu().numpy()
    new_grammar = Grammar(
        sub,
        {
            k: binary[i, : sub[k[0]], : sub[k[1]], : sub[k[2]]].copy()
            for k, i in prepared.binary_index.items()
        },
        {k: unary[i, : sub[k[0]], : sub[k[1]]].copy() for k, i in prepared.unary_index.items()},
        grammar.smoother,
    )
    seen, unseen = state.seen.cpu().numpy(), state.unseen.cpu().numpy()
    tag_count, unseen_tag = state.tag_count.cpu().numpy(), state.unseen_tag_count.cpu().numpy()
    new_lexicon = Lexicon(sub, lexicon.smoother)
    for i, (tag, word) in enumerate(tok.entries):
        new_lexicon.word_tag.setdefault(tag, {})[word] = seen[i, : sub[tag]].copy()
    for i, (tag, sig) in enumerate(tok.unseen_entries):
        new_lexicon.unseen_word_tag.setdefault(tag, {})[sig] = unseen[i, : sub[tag]].copy()
    tags = sorted({tag for tag, _ in tok.entries})
    new_lexicon.tag_count = {tag: tag_count[tag, : sub[tag]].copy() for tag in tags}
    new_lexicon.unseen_tag_count = {tag: unseen_tag[tag, : sub[tag]].copy() for tag in tags}
    counts = state.word_count.cpu().numpy()
    new_lexicon.word_count = {w: float(c) for w, c in zip(tok.vocabulary, counts, strict=True)}
    new_lexicon.total, new_lexicon.total_unseen = state.total, state.total_unseen
    return new_grammar, new_lexicon


def lexical_scores(state: State, tok: Tokens, no_smoothing: bool) -> torch.Tensor:
    """`Lexicon.score` for every entry: P(word | tag substate), `[entries, S]`."""
    c_w = state.word_count[tok.entry_word].unsqueeze(1)
    c_t = state.tag_count[tok.entry_tag]
    if state.total_unseen == 0:
        p_t_u = torch.ones_like(c_t)
    else:
        p_t_u = state.unseen_tag_count[tok.entry_tag] / state.total_unseen
    if no_smoothing:
        weight = torch.full_like(c_w, 0.0001)
    else:
        weight = torch.where(c_w > SMOOTH_IN_UNKNOWNS, 0.0001, SMOOTHING[1]).to(c_w.dtype)
    p_t_w = (state.seen + weight * p_t_u) / (c_w + weight)
    p_w_t = p_t_w * (c_w / state.total) / (c_t / state.total)
    skip = (c_t == 0) | (p_t_w == 0)
    scores = torch.where(skip, torch.zeros_like(p_w_t), torch.where(p_w_t == 0, 1e-87, p_w_t))
    if state.lexicon_smoother is not None:
        scores = torch.einsum("eij,ej->ei", state.lexicon_smoother[tok.entry_tag], scores)
    return scores


def normalize(
    state: State, binary: torch.Tensor, unary: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`em.normalize`: divide by parent-substate totals and prune."""
    categories = state.tag_count.shape[0]
    totals = binary.new_zeros(categories, binary.shape[1])
    totals.index_add_(0, state.binary_parent, binary.sum(dim=(2, 3)))
    unary_sums = unary.sum(dim=2) * (~state.unary_self).unsqueeze(1)
    totals.index_add_(0, state.unary_parent, unary_sums)

    def divide(counts: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        total = totals[parents].reshape(counts.shape[:2] + (1,) * (counts.dim() - 2))
        safe = torch.where(total != 0, total, torch.ones_like(total))
        values = torch.where(total != 0, counts / safe, counts)
        return torch.where((total != 0) & (values < PRUNE), torch.zeros_like(values), values)

    return divide(binary, state.binary_parent), divide(unary, state.unary_parent)


def iterate(
    corpus: Corpus,
    state: State,
    rare: int = RARE,
    lexicon_only: bool = False,
    grammar_smoother: torch.Tensor | None = None,
) -> tuple[float, State]:
    """One EM iteration on tensors (`em.iterate`)."""
    device = state.device
    tok = tokens(corpus, device)
    encoded = corpus._cache[("encoding", str(device))][0]
    scores = lexical_scores(state, tok, no_smoothing=True)
    total, binary_counts, unary_counts, token_counts = expected_counts(
        encoded, state.binary, state.unary, scores[tok.entry]
    )
    # Lexicon M-step.
    weights = token_counts.sum(dim=1)
    rare_token = state.word_count[tok.word] < rare + 0.5
    seen = torch.zeros_like(state.seen).index_add_(0, tok.entry, token_counts)
    unseen = torch.zeros_like(state.unseen).index_add_(
        0, tok.unseen_entry[rare_token], token_counts[rare_token]
    )
    tag_count = torch.zeros_like(state.tag_count).index_add_(0, tok.tag, token_counts)
    unseen_tag_count = torch.zeros_like(state.unseen_tag_count).index_add_(
        0, tok.tag[rare_token], token_counts[rare_token]
    )
    word_count = torch.zeros_like(state.word_count).index_add_(0, tok.word, weights)
    word_count.index_add_(0, tok.signature[rare_token], weights[rare_token])
    # Rare words take their tag's unseen-word distribution.
    unseen_totals = unseen_tag_count.sum(dim=1)
    tied = (word_count[tok.entry_word] < rare + 0.5) & (unseen_totals[tok.entry_tag] != 0)
    safe = torch.where(unseen_totals != 0, unseen_totals, torch.ones_like(unseen_totals))
    distribution = unseen_tag_count[tok.entry_tag] / safe[tok.entry_tag].unsqueeze(1)
    seen = torch.where(tied.unsqueeze(1), distribution * seen.sum(dim=1, keepdim=True), seen)
    seen = torch.where(seen < PRUNE, torch.zeros_like(seen), seen)
    new = replace(
        state,
        seen=seen,
        unseen=unseen,
        tag_count=tag_count,
        unseen_tag_count=unseen_tag_count,
        word_count=word_count,
        total=float(weights.sum()),
        total_unseen=float(weights[rare_token].sum()),
    )
    if lexicon_only:
        return float(total), new
    binary, unary = normalize(state, binary_counts, unary_counts)
    if state.grammar_smoother is not None:
        smoother_b = state.grammar_smoother[state.binary_parent]
        smoother_u = state.grammar_smoother[state.unary_parent]
        binary, unary = normalize(
            state,
            torch.einsum("rap,rpbc->rabc", smoother_b, binary),
            torch.einsum("rap,rpb->rab", smoother_u, unary),
        )
    return float(total), replace(new, binary=binary, unary=unary)


# Split and merge on tensors (`splitmerge.split`, `merge_statistics`, `merge`)

ROOT_CATEGORY = 0


def _valid(substates: dict[int, int], size: int, device) -> torch.Tensor:
    """`[categories, size]`: which substates each category has."""
    counts = torch.zeros(max(substates) + 1, dtype=torch.long, device=device)
    for c, n in substates.items():
        counts[c] = n
    return torch.arange(size, device=device).unsqueeze(0) < counts.unsqueeze(1)


def split(state: State, generator: torch.Generator, randomness: float = 1.0) -> State:
    """`splitmerge.split` on tensors: substate k becomes 2k and 2k + 1 (not ROOT's).

    A binary weight `w[a, b, c]` becomes, for new parent substates `2a + p`,
    `w / 4 + s_i εL[p, a, b, c] + s_j εR[p, a, b, c, i]` at children
    `(2b + i, 2c + j)`, with `s = (+1, -1)` and `ε = w / 4 · r / 100 · (u - 0.5)`;
    a unary weight becomes `w / 2 + s_i ε[p, a, b]`. ROOT, which is never a
    child, keeps one substate: its second copy is dropped.
    """
    substates = {c: n if c == ROOT_CATEGORY else 2 * n for c, n in state.substates.items()}
    old = state.binary.shape[1]
    size = 2 * old
    device = state.device
    sign = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

    def noise(shape) -> torch.Tensor:
        u = torch.rand(shape, generator=generator, dtype=torch.float64)
        return (u - 0.5).to(device) * randomness / 100

    r = state.binary.shape[0]
    w = state.binary / 4
    e_left = w.unsqueeze(1) * noise((r, 2, old, old, old))  # [r, p, a, b, c]
    e_right = w.unsqueeze(1).unsqueeze(-1) * noise((r, 2, old, old, old, 2))  # [..., i]
    # dimensions [r, a, p, b, i, c, j]
    base = w[:, :, None, :, None, :, None]
    left = e_left.permute(0, 2, 1, 3, 4)[:, :, :, :, None, :, None] * sign.view(1, 1, 1, 1, 2, 1, 1)
    right = e_right.permute(0, 2, 1, 3, 5, 4)[..., None] * sign
    binary = (base + left + right).reshape(r, size, size, size)
    u = state.unary.shape[0]
    wu = state.unary / 2
    e = wu.unsqueeze(1) * noise((u, 2, old, old))  # [u, p, a, b]
    unary = (wu[:, :, None, :, None] + e.permute(0, 2, 1, 3)[..., None] * sign).reshape(
        u, size, size
    )
    valid = _valid(substates, size, device)
    binary = binary * valid[state.binary_parent][:, :, None, None]
    unary = unary * valid[state.unary_parent][:, :, None]

    def halve(values: torch.Tensor) -> torch.Tensor:
        return values.repeat_interleave(2, dim=1) / 2

    return replace(
        state,
        substates=substates,
        binary=binary,
        unary=unary,
        seen=halve(state.seen),
        unseen=halve(state.unseen),
        tag_count=halve(state.tag_count) * valid,
        unseen_tag_count=halve(state.unseen_tag_count) * valid,
        grammar_smoother=None,
        lexicon_smoother=None,
    )


def merge_statistics(corpus: Corpus, state: State) -> tuple[dict, dict]:
    """`splitmerge.merge_statistics` on tensors: merge weights and losses by category."""
    from txtures.lvpcfg.inside import node_posteriors

    device = state.device
    tok = tokens(corpus, device)
    encoded, *_, categories = corpus._cache[("encoding", str(device))]
    cat = torch.as_tensor(categories, device=device)
    scores = lexical_scores(state, tok, no_smoothing=False)
    totals, inside, outside, posterior = node_posteriors(
        encoded, state.binary, state.unary, scores[tok.entry]
    )
    sizes = torch.as_tensor([len(t) for t in corpus.nodes], device=device)
    keep = torch.repeat_interleave(torch.isfinite(totals), sizes)
    size = state.binary.shape[1]
    n_categories = state.tag_count.shape[0]
    valid = _valid(state.substates, size, device)
    sums = posterior.new_zeros(n_categories, size).index_add_(0, cat[keep], posterior[keep])
    totals_c = (sums * valid).sum(dim=1, keepdim=True)
    weights = sums / torch.where(totals_c != 0, totals_c, torch.ones_like(totals_c))
    w = weights[cat]
    i_s, o_s = inside * valid[cat], outside * valid[cat]
    separated = (i_s * o_s).sum(dim=1, keepdim=True)
    ii, ij, oi, oj = i_s[:, 0::2], i_s[:, 1::2], o_s[:, 0::2], o_s[:, 1::2]
    wi, wj = w[:, 0::2], w[:, 1::2]
    ws = torch.where(wi + wj != 0, wi + wj, torch.ones_like(wi))
    merged = (ii * wi / ws + ij * wj / ws) * (oi + oj)
    combined = separated - ii * oi - ij * oj + merged
    ok = (merged != 0) & (separated != 0) & valid[cat][:, 1::2] & keep.unsqueeze(1)
    ratio = torch.where(ok, separated / torch.where(ok, combined, torch.ones_like(combined)), 1.0)
    contributions = torch.log(ratio)
    deltas = contributions.new_zeros(n_categories, size // 2).index_add_(0, cat, contributions)
    weights_np, deltas_np = weights.cpu().numpy(), deltas.cpu().numpy()
    by_weight = {c: weights_np[c, :n] for c, n in state.substates.items()}
    by_delta = {c: deltas_np[c, : n // 2] for c, n in state.substates.items() if n // 2 > 0}
    return by_weight, by_delta


def merge(corpus: Corpus, state: State, pairs: dict, weights: dict) -> State:
    """`splitmerge.merge` on tensors: merge the chosen sibling pairs."""
    device = state.device
    old = state.binary.shape[1]
    new_substates, maps = {}, {}
    for c, n in state.substates.items():
        merged = set(pairs.get(c, ()))
        target, count = [], 0
        for s in range(n):
            if s % 2 == 1 and s // 2 in merged:
                target.append(target[-1])
            else:
                target.append(count)
                count += 1
        maps[c] = target
        new_substates[c] = count
    new_substates[ROOT_CATEGORY] = 1
    size = max(new_substates.values())
    n_categories = state.tag_count.shape[0]
    sums = np.zeros((n_categories, old, size))
    averages = np.zeros((n_categories, size, old))
    for c, target in maps.items():
        w = weights[c]
        for s, t in enumerate(target):
            sums[c, s, t] = 1.0
            members = [m for m, x in enumerate(target) if x == t]
            averages[c, t, s] = 1.0 if len(members) == 1 else w[s] / w[members].sum()
    sums_t = torch.as_tensor(sums, device=device)
    averages_t = torch.as_tensor(averages, device=device)
    binary = torch.einsum(
        "rAa,rabc,rbB,rcC->rABC",
        averages_t[state.binary_parent],
        state.binary,
        sums_t[state.binary_left],
        sums_t[state.binary_right],
    )
    unary = torch.einsum(
        "rAa,rab,rbB->rAB", averages_t[state.unary_parent], state.unary, sums_t[state.unary_child]
    )
    entry_tags = tokens(corpus, device).entry_tag
    return replace(
        state,
        substates=new_substates,
        binary=binary,
        unary=unary,
        seen=torch.einsum("ea,eaB->eB", state.seen, sums_t[entry_tags]),
        # Berkeley does not merge the signature counts; the next pass replaces them.
        unseen=state.unseen[:, :size],
        tag_count=torch.einsum("ta,taB->tB", state.tag_count, sums_t),
        unseen_tag_count=state.unseen_tag_count[:, :size],
    )
