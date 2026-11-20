# This file is part of txtures. The parser follows ModelBlocks' synproc
# (resource-incrsem/src/synproc.cpp), Copyright 2009, ModelBlocks developers,
# reimplemented for txtures in 2026.
#
# txtures is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See LICENSE.
"""synproc on the GPU: batched beam search over many sentences at once.

The semantics are those of `txtures.lvpcfg.synproc.reference_parse`. Beam
entries of all sentences live in one flat set of tensors, each tagged with
its sentence, so sentences of different lengths cost nothing once they end.
Each word is one step:

1. Apex and base. With join 0 their product is
   `plus[d-1][b, parent] · gamma_left[d][r] / J0[d-1][b, c0]` for each rule
   `r = parent -> c0 c1` whose left child is the entry's `c0`; with join 1 it
   is `gamma_right[d][r]` renormalized over the rules `b -> c0 ·`. The
   candidates are pruned, per sentence, to the best `working` partial
   scores.
2. Preterminal, fork, and join for the word, for every survivor.
3. Entries with the same state keep their best score (Viterbi), and each
   sentence keeps its best `beam`.

Every factor is at most 1, so a partial score bounds its completions. If
some sentence's best pruned partial beats its last kept entry, the step is
redone with twice the working size, so the beam is the exact top-`beam`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from txtures.lvpcfg import leftcorner as lc
from txtures.lvpcfg.synproc import berkeley_signature

NEG = -math.inf


@dataclass
class Beams:
    """The live beam entries of all sentences (flat)."""

    sentence: torch.Tensor  # [M]
    pretrm: torch.Tensor  # [M]
    fork: torch.Tensor  # [M]
    join: torch.Tensor  # [M]
    store: torch.Tensor  # [M, slots, 2] apex and base, -1 beyond the size
    size: torch.Tensor  # [M]
    score: torch.Tensor  # [M] log probability
    back: torch.Tensor  # [M] index of the entry in the previous beams it came from


class Parser:
    """synproc's beam search with a compiled left-corner model, on one device."""

    def __init__(self, model: lc.Compiled, beam: int = 5000, working: int | None = None):
        self.model = model
        self.beam = beam
        self.working = working or 2 * beam
        device = model.plus.device
        self.device = device
        n = len(model.symbols)
        self.n = n
        self.root = model.index[lc.ROOT]
        self.slots = lc.MAX_DEPTH + 2
        depths = lc.MAX_DEPTH + 1
        self.plus = model.plus
        self.j0 = torch.stack([lc.j_table(model, d)[0] for d in range(depths)])
        self.j1 = torch.stack([lc.j_table(model, d)[1] for d in range(depths)])
        f = [lc.f_table(model, d) for d in range(depths)]
        self.f0 = torch.stack([x[0] for x in f])
        self.f1 = torch.stack([x[1] for x in f])
        self.terminal = torch.stack(
            [lc._terminal_gamma(model, d, lc.LEFT) for d in range(depths + 1)]
        )
        parent, left, right = model.rule_parent, model.rule_left, model.rule_right
        # rules by left child (join 0)
        order = torch.argsort(left, stable=True)
        self.by_left = order
        counts = torch.bincount(left, minlength=n)
        self.left_start = torch.cumsum(counts, 0) - counts
        self.left_count = counts
        # rules by (parent, left child) (join 1)
        pair = parent * n + left
        order = torch.argsort(pair, stable=True)
        self.by_pair = order
        keys, counts = torch.unique_consecutive(pair[order], return_counts=True)
        self.pair_keys = keys
        self.pair_start = torch.cumsum(counts, 0) - counts
        self.pair_count = counts
        self.parent, self.left, self.right = parent, left, right
        self.gamma = model.gamma
        # gamma_right renormalized over the rules sharing (parent, left)
        totals = torch.zeros(model.gamma.shape[0], len(keys), dtype=torch.float64, device=device)
        position = torch.searchsorted(keys, pair)
        for d in range(model.gamma.shape[0]):
            totals[d].index_add_(0, position, model.gamma[d, lc.RIGHT])
        self.gamma_right_norm = model.gamma[:, lc.RIGHT] / torch.where(
            totals[:, position] > 0, totals[:, position], torch.ones_like(totals[:, position])
        )
        self.first_any = model.first.sum(dim=2) > 0  # [depths, N]
        self._lexicon: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._entries: dict[str, dict[int, float]] = {}
        for tag, words in model.lexicon.items():
            if tag in model.index:
                for word, p in words.items():
                    self._entries.setdefault(word, {})[model.index[tag]] = p

    # Lexicon

    def word_entries(self, word: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A word's preterminals and P(word | preterminal), with its signature's mass added."""
        if word not in self._lexicon:
            known = self._entries.get(word)
            signature = self._entries.get(berkeley_signature(word), {})
            merged = dict(signature) if known is None else dict(known)
            if known is not None:
                for tag, p in signature.items():
                    merged[tag] = merged.get(tag, 0.0) + p
            tags = torch.tensor(list(merged), dtype=torch.long, device=self.device)
            probs = torch.tensor(list(merged.values()), dtype=torch.float64, device=self.device)
            self._lexicon[word] = (tags, probs)
        return self._lexicon[word]

    # Store helpers

    def _slot(self, store, index, field):
        """`store[m, index[m], field]`, or the root where `index` < 0."""
        safe = index.clamp(min=0)
        value = store[torch.arange(len(index), device=self.device), safe, field]
        return torch.where(index >= 0, value, torch.full_like(value, self.root))

    def _lchild(self, store, size, fork, pretrm):
        apex = self._slot(store, size - 1, 0)
        return torch.where(fork == 1, pretrm, apex)

    # One word

    def _apex_base(self, beams: Beams):
        """Stage 1 scores: candidate (entry, apex, base, depth) with their partial scores.

        Only indices and scores are built here; stores are built for survivors.
        An apex of -1 marks the empty store of the start, -2 an apex taken from
        the store (join 1).
        """
        dev = self.device
        depth = beams.size + beams.fork - beams.join
        ancestor = self._slot(beams.store, beams.size - 2 + beams.fork, 1)
        lchild = self._lchild(beams.store, beams.size, beams.fork, beams.pretrm)
        pieces = []
        start = depth <= 0
        if start.any():
            idx = torch.nonzero(start).squeeze(1)
            minus = torch.full_like(idx, -1)
            pieces.append(
                (idx, minus, minus, torch.zeros(len(idx), dtype=torch.float64, device=dev))
            )
        zero = (beams.join == 0) & (depth >= 1) & (depth <= lc.MAX_DEPTH + 1)
        if zero.any():
            idx = torch.nonzero(zero).squeeze(1)
            rule, entry = _expand(self.left_start[lchild[idx]], self.left_count[lchild[idx]], idx)
            rule = self.by_left[rule]
            d, b = depth[entry], ancestor[entry]
            weight = (
                self.plus[d - 1, b, self.parent[rule]]
                * self.gamma[d, lc.LEFT, rule]
                / self.j0[d - 1, b, lchild[entry]]
            )
            pieces.append((entry, self.parent[rule], self.right[rule], torch.log(weight)))
        one = (beams.join == 1) & (depth >= 0) & (depth <= lc.MAX_DEPTH)
        if one.any():
            idx = torch.nonzero(one).squeeze(1)
            idx = idx[self.first_any[depth[idx], ancestor[idx]]]
            key = ancestor[idx] * self.n + lchild[idx]
            position = torch.searchsorted(self.pair_keys, key).clamp(max=len(self.pair_keys) - 1)
            found = self.pair_keys[position] == key
            idx, position = idx[found], position[found]
            rule, entry = _expand(self.pair_start[position], self.pair_count[position], idx)
            rule = self.by_pair[rule]
            weight = self.gamma_right_norm[depth[entry], rule]
            pieces.append((entry, torch.full_like(entry, -2), self.right[rule], torch.log(weight)))
        if not pieces:
            return None
        entry, apex, base, log_weight = (torch.cat(x) for x in zip(*pieces, strict=True))
        keep = torch.isfinite(log_weight)
        entry, apex, base, log_weight = entry[keep], apex[keep], base[keep], log_weight[keep]
        return entry, apex, base, beams.score[entry] + log_weight

    def _stores(self, beams: Beams, entry, apex, base):
        """The new store of each surviving stage-1 candidate, its size, and a shared id."""
        dev = self.device
        depth = beams.size[entry] + beams.fork[entry] - beams.join[entry]
        new_size = torch.where(apex == -1, torch.zeros_like(entry), depth)
        store = beams.store[entry].clone()
        rows = torch.arange(len(entry), device=dev)
        slot = (new_size - 1).clamp(min=0)
        apex = torch.where(apex == -2, store[rows, slot, 0], apex)
        positions = torch.arange(self.slots, device=dev).unsqueeze(0)
        store = torch.where((positions >= new_size.unsqueeze(1)).unsqueeze(2), -1, store)
        filled = new_size > 0
        store[rows[filled], slot[filled], 0] = apex[filled]
        store[rows[filled], slot[filled], 1] = base[filled]
        unique, store_id = torch.unique(store.reshape(len(entry), -1), dim=0, return_inverse=True)
        return store, new_size, store_id, unique.reshape(-1, self.slots, 2)

    def step(self, beams: Beams, words: list[str], last: torch.Tensor) -> tuple[Beams, bool]:
        """Advance every active sentence by one word (`words[s]` for sentence s)."""
        stage = self._apex_base(beams)
        if stage is None:
            return _empty(self.slots, self.device), True
        entry, apex, base, partial = stage
        sentence = beams.sentence[entry]
        working = self.working
        while True:
            kept, pruned_best = _top_per_sentence(sentence, partial, working)
            store, size, store_id, _ = self._stores(beams, entry[kept], apex[kept], base[kept])
            new = self._word(
                beams,
                entry[kept],
                store,
                size,
                store_id,
                partial[kept],
                sentence[kept],
                words,
                last,
            )
            threshold = _kth_per_sentence(new.sentence, new.score, self.beam)
            if all(bound <= threshold.get(s, NEG) for s, bound in pruned_best.items()):
                return new, True
            working *= 2

    def _word(
        self, beams, entry, store, size, store_id, score, sentence, words, last, chunk=1 << 22
    ):
        """Stage 2: preterminal, fork, and join for every stage-1 survivor, in chunks.

        Each chunk's candidates are folded into a running beam per sentence,
        keyed by (sentence, store, preterminal, fork, join) and kept by best
        score. Dropping states below a sentence's current `beam`-th score is
        exact, since that score only rises as chunks are added.
        """
        dev = self.device
        rows = torch.arange(len(entry), device=dev)
        brink = self._slot(store, size - 1, 1)
        anc0 = self._slot(store, size - 2, 1)
        top_apex = self._slot(store, size - 1, 0)
        running = None
        for s in torch.unique(sentence).tolist():
            tags, pw = self.word_entries(words[s])
            if len(tags) == 0:
                continue
            mine = rows[sentence == s]
            step = max(1, chunk // len(tags))
            for begin in range(0, len(mine), step):
                part = mine[begin : begin + step]
                found = self._candidates(
                    part, tags, pw, size, brink, anc0, top_apex, score, bool(last[s])
                )
                if found is None:
                    continue
                e, tag, fork, join, total = found
                candidate = (sentence[e], store_id[e], tag, fork, join, total, e)
                running = _fold(running, candidate, self.beam)
        if running is None:
            return _empty(self.slots, dev)
        sent, sid, tag, fork, join, total, e = running
        return Beams(sent, tag, fork, join, store[e], size[e], total, entry[e])

    def _candidates(self, part, tags, pw, size, brink, anc0, top_apex, score, last):
        """Scores of every (survivor, preterminal, fork, join) for one sentence's chunk."""
        m, t = len(part), len(tags)
        e = part.repeat_interleave(t)
        tag = tags.repeat(m)
        p_w = pw.repeat(m)
        d, b = size[e], brink[e]
        ok = d <= lc.MAX_DEPTH
        dd = d.clamp(max=lc.MAX_DEPTH)
        total_f = self.f0[dd, b] + self.f1[dd, b]
        found = []
        for fork in (0, 1):
            if fork == 0:
                p_f = self.f0[dd, b] / total_f
                p_p = (tag == b).to(torch.float64)
                anc, lch = anc0[e], top_apex[e]
            else:
                p_f = self.f1[dd, b] / total_f
                term = self.terminal[(dd + 1).clamp(max=lc.MAX_DEPTH + 1), tag]
                p_p = self.plus[dd, b, tag] * term / self.f1[dd, b]
                anc, lch = b, tag
            join_depth = d + fork
            jd = (join_depth - 1).clamp(min=0, max=lc.MAX_DEPTH)
            j_ok = (join_depth >= 1) & (join_depth <= lc.MAX_DEPTH + 1)
            j0, j1 = self.j0[jd, anc, lch], self.j1[jd, anc, lch]
            total_j = j0 + j1
            for join, j in ((0, j0), (1, j1)):
                final_size = d + fork - join
                allowed = final_size == 0 if last else final_size > 0
                prob = p_w * p_f * p_p * j / total_j
                good = ok & j_ok & allowed & (total_f > 0) & (total_j > 0) & (prob > 0)
                if good.any():
                    g = torch.nonzero(good).squeeze(1)
                    found.append(
                        (
                            e[g],
                            tag[g],
                            torch.full_like(g, fork),
                            torch.full_like(g, join),
                            score[e[g]] + torch.log(prob[g]),
                        )
                    )
        if not found:
            return None
        return tuple(torch.cat(x) for x in zip(*found, strict=True))

    # Sentences

    def parse(self, sentences: list[list[str]]) -> list[dict]:
        """Parse sentences together; per sentence, per-word surprisal (bits) and the best states."""
        dev = self.device
        count = len(sentences)
        start = Beams(
            torch.arange(count, device=dev),
            torch.full((count,), -1, device=dev),
            torch.zeros(count, dtype=torch.long, device=dev),
            torch.ones(count, dtype=torch.long, device=dev),
            torch.full((count, self.slots, 2), -1, dtype=torch.long, device=dev),
            torch.ones(count, dtype=torch.long, device=dev),
            torch.zeros(count, dtype=torch.float64, device=dev),
            torch.full((count,), -1, device=dev),
        )
        lengths = torch.tensor([len(s) for s in sentences], device=dev)
        history = [start]
        surprisal = [[] for _ in sentences]
        beams = start
        t = 0
        while True:
            active = lengths[beams.sentence] > t if len(beams.sentence) else beams.sentence.bool()
            if not active.any():
                break
            selected = torch.nonzero(active).squeeze(1)
            current = _select(beams, active)
            words = [s[t] if t < len(s) else "" for s in sentences]
            last = lengths == t + 1
            new, _ = self.step(current, words, last)
            # surprisal: log2 of the previous beam's total over the current one's
            for s in torch.unique(current.sentence).tolist():
                before = current.score[current.sentence == s]
                after = new.score[new.sentence == s]
                if len(after) == 0:
                    surprisal[s].append(math.nan)
                    continue
                reference = before.max()
                surprisal[s].append(
                    float(
                        (
                            torch.logsumexp(before - reference, 0)
                            - torch.logsumexp(after - reference, 0)
                        )
                        / math.log(2)
                    )
                )
            new.back = selected[new.back]  # an index into the previous beams
            history.append(new)
            beams = new
            t += 1
        return [
            {"surprisal": surprisal[s], "states": _best_path(history, s, self.model)}
            for s in range(count)
        ]


def _empty(slots: int, device) -> Beams:
    z = torch.zeros(0, dtype=torch.long, device=device)
    return Beams(
        z,
        z,
        z,
        z,
        torch.zeros(0, slots, 2, dtype=torch.long, device=device),
        z,
        torch.zeros(0, dtype=torch.float64, device=device),
        z,
    )


def _select(beams: Beams, mask: torch.Tensor) -> Beams:
    return Beams(
        *(
            x[mask]
            for x in (
                beams.sentence,
                beams.pretrm,
                beams.fork,
                beams.join,
                beams.store,
                beams.size,
                beams.score,
                beams.back,
            )
        )
    )


def _expand(starts: torch.Tensor, counts: torch.Tensor, owners: torch.Tensor):
    """Ragged ranges: positions `starts[i] .. starts[i] + counts[i]` and their owners."""
    owner = torch.repeat_interleave(owners, counts)
    first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    offset = torch.arange(int(counts.sum()), device=counts.device) - first
    return torch.repeat_interleave(starts, counts) + offset, owner


def _fold(running, candidate, beam: int):
    """Merge candidates into the running beam: best per state, then each sentence's `beam` best."""
    if running is not None:
        candidate = tuple(torch.cat([a, b]) for a, b in zip(running, candidate, strict=True))
    sent, sid, tag, fork, join, total, e = candidate
    order = torch.argsort(total, descending=True, stable=True)
    sent, sid, tag, fork, join, total, e = (
        x[order] for x in (sent, sid, tag, fork, join, total, e)
    )
    key = torch.stack([sent, sid, tag, fork, join], dim=1)
    unique, inverse = torch.unique(key, dim=0, return_inverse=True)
    first = torch.full((len(unique),), len(total), dtype=torch.long, device=total.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(total), device=total.device), reduce="amin")
    sent, sid, tag, fork, join, total, e = (
        x[first] for x in (sent, sid, tag, fork, join, total, e)
    )
    kept, _ = _top_per_sentence(sent, total, beam)
    return tuple(x[kept] for x in (sent, sid, tag, fork, join, total, e))


def _top_per_sentence(sentence: torch.Tensor, score: torch.Tensor, k: int):
    """Indices of each sentence's `k` best scores, and each sentence's best score not kept."""
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(sentence[order], stable=True)]
    sorted_sentence = sentence[order]
    starts = torch.searchsorted(sorted_sentence, sorted_sentence, right=False)
    rank = torch.arange(len(order), device=score.device) - starts
    kept = order[rank < k]
    pruned = order[rank == k]
    best = {
        int(s): float(v)
        for s, v in zip(sentence[pruned].tolist(), score[pruned].tolist(), strict=True)
    }
    return kept, best


def _kth_per_sentence(sentence: torch.Tensor, score: torch.Tensor, k: int) -> dict[int, float]:
    """Each sentence's `k`-th best score, or -inf where it has fewer than `k`."""
    kept, _ = _top_per_sentence(sentence, score, k)
    out: dict[int, float] = {}
    counts: dict[int, int] = {}
    for s, v in zip(sentence[kept].tolist(), score[kept].tolist(), strict=True):
        counts[s] = counts.get(s, 0) + 1
        out[s] = min(out.get(s, math.inf), v)
    return {s: (v if counts[s] >= k else NEG) for s, v in out.items()}


def _best_path(history, s: int, model: lc.Compiled) -> list[tuple]:
    path = []
    index = None
    for t in range(len(history) - 1, 0, -1):
        beams = history[t]
        if index is None:
            mine = torch.nonzero(beams.sentence == s).squeeze(1)
            if len(mine) == 0:
                continue
            index = int(mine[torch.argmax(beams.score[mine])])
        store = beams.store[index]
        size = int(beams.size[index])
        path.append(
            (
                int(beams.pretrm[index]),
                int(beams.fork[index]),
                int(beams.join[index]),
                tuple((int(store[k, 0]), int(store[k, 1])) for k in range(size)),
            )
        )
        index = int(beams.back[index])
    return list(reversed(path))
