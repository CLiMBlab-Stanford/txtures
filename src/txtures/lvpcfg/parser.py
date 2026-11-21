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
from txtures.lvpcfg.synproc import MergedLexicon

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

    def __init__(
        self,
        model: lc.Compiled,
        beam: int = 5000,
        working: int | None = None,
        lexicon_merge: str = "once",
    ):
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
        self.merged = MergedLexicon(model.lexicon, model.index, lexicon_merge)

    # Lexicon

    def word_entries(self, word: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A word's preterminals and P(word | preterminal), unknown-word mass merged in."""
        if word not in self._lexicon:
            merged = self.merged.entries(word)
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
        # Bound each candidate's completion by its best word, fork, and preterminal.
        depth = beams.size[entry] + beams.fork[entry] - beams.join[entry]
        new_size = torch.where(apex == -1, torch.zeros_like(depth), depth)
        brink = torch.where(apex == -1, torch.full_like(base, self.root), base)
        bounded = partial + self._completion_bound(sentence, new_size, brink, words)
        alive = torch.isfinite(bounded)
        entry, apex, base, partial, sentence, bounded = (
            x[alive] for x in (entry, apex, base, partial, sentence, bounded)
        )
        if len(sentence) == 0:
            return _empty(self.slots, self.device), True
        # Each sentence has its own working size; a sentence whose pruning could
        # have changed its beam is redone alone, with twice the room.
        dev = self.device
        work = torch.full((int(sentence.max()) + 1,), self.working, dtype=torch.long, device=dev)
        todo = torch.ones_like(work, dtype=torch.bool)
        done: list[Beams] = []
        while True:
            subset = torch.nonzero(todo[sentence]).squeeze(1)
            kept, pruned_best = _top_per_sentence(sentence[subset], bounded[subset], work)
            kept = subset[kept]
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
            violators = [s for s, bound in pruned_best.items() if bound > threshold.get(s, NEG)]
            redo = torch.zeros_like(todo)
            if violators:
                redo[torch.tensor(violators, device=dev)] = True
            done.append(_select(new, ~redo[new.sentence]) if len(new.sentence) else new)
            if not violators:
                return _concat_all(done, self.slots, dev), True
            work[redo] *= 2
            todo = redo

    def _completion_bound(self, sentence, size, brink, words) -> torch.Tensor:
        """log of an upper bound on P(word, fork, preterminal | new store) for each candidate.

        It is the best over the word's preterminals and both forks; the join's
        probability, at most 1, is left out. It depends only on the new store's
        size and base, so it is computed once per distinct pair.
        """
        dev = self.device
        out = torch.full(size.shape, NEG, dtype=torch.float64, device=dev)
        for s in torch.unique(sentence).tolist():
            tags, pw = self.word_entries(words[s])
            if len(tags) == 0:
                continue
            mine = torch.nonzero(sentence == s).squeeze(1)
            unique, inverse = torch.unique(size[mine] * self.n + brink[mine], return_inverse=True)
            d, b = unique // self.n, unique % self.n
            ok = d <= lc.MAX_DEPTH
            dd = d.clamp(max=lc.MAX_DEPTH)
            f0, f1 = self.f0[dd, b], self.f1[dd, b]
            total = f0 + f1
            lexical = torch.zeros(self.n, dtype=torch.float64, device=dev)
            lexical[tags] = pw
            term = self.terminal[(dd + 1).clamp(max=lc.MAX_DEPTH + 1)][:, tags]
            fork1 = (self.plus[dd, b][:, tags] * term * pw).max(dim=1).values
            fork0 = f0 * lexical[b]
            best = torch.maximum(fork0, fork1) / torch.where(
                total > 0, total, torch.ones_like(total)
            )
            usable = ok & (total > 0) & (best > 0)
            best = torch.where(usable, torch.log(best), torch.full_like(best, NEG))
            out[mine] = best[inverse]
        return out

    def _word(
        self, beams, entry, store, size, store_id, score, sentence, words, last, chunk=1 << 25
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
                running = _fold(running, candidate, self.beam, self.n)
        if running is None:
            return _empty(self.slots, dev)
        sent, sid, tag, fork, join, total, e = running
        return Beams(sent, tag, fork, join, store[e], size[e], total, entry[e])

    def _candidates(self, part, tags, pw, size, brink, anc0, top_apex, score, last):
        """Scores of every (survivor, preterminal, fork, join) for one sentence's chunk.

        Only candidates with a nonzero probability are built: with fork 0 the
        preterminal must be the store's base, and with fork 1 only
        preterminals the base can reach get a join.
        """
        dev = self.device
        d, b = size[part], brink[part]
        ok = d <= lc.MAX_DEPTH
        dd = d.clamp(max=lc.MAX_DEPTH)
        f0, f1 = self.f0[dd, b], self.f1[dd, b]
        total_f = torch.where(ok & (f0 + f1 > 0), f0 + f1, torch.zeros_like(f0))
        lexical = torch.zeros(self.n, dtype=torch.float64, device=dev)
        lexical[tags] = pw
        pieces = []
        # fork 0: the preterminal is the base itself
        keep = (total_f > 0) & (f0 > 0) & (lexical[b] > 0)
        rows = torch.nonzero(keep).squeeze(1)
        if len(rows):
            weight = f0[rows] / total_f[rows] * lexical[b[rows]]
            pieces.append((rows, b[rows], 0, weight, anc0[part[rows]], top_apex[part[rows]]))
        # fork 1: preterminals reachable from the base
        term = self.terminal[(dd + 1).clamp(max=lc.MAX_DEPTH + 1)][:, tags]
        weight = (
            self.plus[dd, b][:, tags]
            * term
            * pw
            / torch.where(total_f > 0, total_f, torch.ones_like(total_f)).unsqueeze(1)
        )
        weight = torch.where((total_f > 0).unsqueeze(1), weight, torch.zeros_like(weight))
        rows, columns = torch.nonzero(weight > 0, as_tuple=True)
        if len(rows):
            pieces.append((rows, tags[columns], 1, weight[rows, columns], b[rows], tags[columns]))
        found = []
        for rows, tag, fork, weight, anc, lch in pieces:
            join_depth = d[rows] + fork
            jd = (join_depth - 1).clamp(min=0, max=lc.MAX_DEPTH)
            j_ok = (join_depth >= 1) & (join_depth <= lc.MAX_DEPTH + 1)
            j0, j1 = self.j0[jd, anc, lch], self.j1[jd, anc, lch]
            total_j = j0 + j1
            safe = torch.where(total_j > 0, total_j, torch.ones_like(total_j))
            for join, j in ((0, j0), (1, j1)):
                final_size = join_depth - join
                allowed = final_size == 0 if last else final_size > 0
                prob = weight * j / safe
                good = j_ok & allowed & (total_j > 0) & (prob > 0)
                g = torch.nonzero(good).squeeze(1)
                if len(g):
                    e = part[rows[g]]
                    found.append(
                        (
                            e,
                            tag[g],
                            torch.full_like(g, fork),
                            torch.full_like(g, join),
                            score[e] + torch.log(prob[g]),
                        )
                    )
        if not found:
            return None
        return tuple(torch.cat(x) for x in zip(*found, strict=True))

    # Sentences

    def parse(
        self, sentences: list[list[str]], max_sentences: int = 32, window: int = 4096
    ) -> list[dict]:
        """Parse sentences; per sentence, per-word surprisal (bits) and the best states."""
        return [result for _, result in self.parse_stream(sentences, max_sentences, window)]

    def parse_stream(self, sentences, max_sentences: int = 32, window: int = 4096):
        """Yield `(index, result)` for each sentence, in input order, parsing many at once.

        Sentences are admitted as others finish, up to `max_sentences` at a
        time, each advancing through its own words; finished results wait in
        a buffer until every earlier sentence is done. Admission never runs
        more than `window` sentences ahead of the oldest unfinished one.
        """
        dev = self.device
        sentences = list(sentences)
        count = len(sentences)
        position = [0] * count  # next word of each sentence
        surprisal: dict[int, list[float]] = {}
        started: dict[int, int] = {}  # sentence -> step at which it was admitted
        finished: dict[int, dict] = {}
        history: dict[int, Beams] = {}
        beams = _empty(self.slots, dev)
        queue, step, emitted = 0, 0, 0
        while emitted < count:
            # admit sentences, empty ones finishing at once
            active = set(beams.sentence.unique().tolist()) if len(beams.sentence) else set()
            new = []
            while (
                queue < count
                and len(active) + len(new) < max_sentences
                and queue < emitted + window
            ):
                if not sentences[queue]:
                    finished[queue] = {"surprisal": [], "states": []}
                else:
                    new.append(queue)
                queue += 1
            if new:
                beams = _concat(beams, self._start(new, step))
                for s in new:
                    started[s], surprisal[s] = step, []
            while emitted < count and emitted in finished:
                yield emitted, finished.pop(emitted)
                emitted += 1
            if not len(beams.sentence):
                continue
            history[step] = beams
            words = {s: sentences[s][position[s]] for s in beams.sentence.unique().tolist()}
            last = {s: position[s] + 1 == len(sentences[s]) for s in words}
            new_beams, _ = self.step(beams, _Lookup(words), _Lookup(last))
            self._surprisal(beams, new_beams, words, surprisal)
            step += 1
            history[step] = new_beams
            done = [s for s in words if last[s] or not (new_beams.sentence == s).any()]
            for s in words:
                position[s] += 1
            for s in done:
                finished[s] = {
                    "surprisal": surprisal.pop(s) + [math.nan] * (len(sentences[s]) - position[s]),
                    "states": _best_path(history, s, started.pop(s), step, len(sentences[s])),
                }
            keep = torch.ones(len(new_beams.sentence), dtype=torch.bool, device=dev)
            for s in done:
                keep &= new_beams.sentence != s
            beams = _select(new_beams, keep)
            oldest = min(started.values(), default=step)
            for k in [k for k in history if k < oldest]:
                del history[k]
            # the next step's backpointers index `beams`, a selection of new_beams
            history[step] = beams
            while emitted < count and emitted in finished:
                yield emitted, finished.pop(emitted)
                emitted += 1

    def _start(self, new: list[int], step: int) -> Beams:
        dev, n = self.device, len(new)
        return Beams(
            torch.tensor(new, dtype=torch.long, device=dev),
            torch.full((n,), -1, dtype=torch.long, device=dev),
            torch.zeros(n, dtype=torch.long, device=dev),
            torch.ones(n, dtype=torch.long, device=dev),
            torch.full((n, self.slots, 2), -1, dtype=torch.long, device=dev),
            torch.ones(n, dtype=torch.long, device=dev),
            torch.zeros(n, dtype=torch.float64, device=dev),
            torch.full((n,), -1, dtype=torch.long, device=dev),
        )

    def _surprisal(self, before: Beams, after: Beams, words, surprisal) -> None:
        """log2 of each sentence's previous beam total over its current one, from its best."""
        sentences = torch.tensor(sorted(words), dtype=torch.long, device=self.device)
        index_b = torch.searchsorted(sentences, before.sentence)
        best = torch.full((len(sentences),), NEG, dtype=torch.float64, device=self.device)
        best.scatter_reduce_(0, index_b, before.score, reduce="amax")
        prev = torch.zeros_like(best).index_add_(
            0, index_b, torch.exp(before.score - best[index_b])
        )
        curr = torch.zeros_like(best)
        if len(after.sentence):
            index_a = torch.searchsorted(sentences, after.sentence)
            curr.index_add_(0, index_a, torch.exp(after.score - best[index_a]))
        values = ((torch.log(prev) - torch.log(curr)) / math.log(2)).tolist()
        for s, v in zip(sentences.tolist(), values, strict=True):
            surprisal[s].append(v if math.isfinite(v) else math.nan)


class _Lookup:
    """`words[s]` and `last[s]` for the step methods, from a dict."""

    def __init__(self, table: dict):
        self.table = table

    def __getitem__(self, key):
        return self.table[int(key)]


def _concat(a: Beams, b: Beams) -> Beams:
    return Beams(
        *(
            torch.cat([x, y])
            for x, y in zip(
                (a.sentence, a.pretrm, a.fork, a.join, a.store, a.size, a.score, a.back),
                (b.sentence, b.pretrm, b.fork, b.join, b.store, b.size, b.score, b.back),
                strict=True,
            )
        )
    )


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


def _fold(running, candidate, beam: int, symbols: int):
    """Merge candidates into the running beam: best per state, then each sentence's `beam` best."""
    if running is not None:
        candidate = tuple(torch.cat([a, b]) for a, b in zip(running, candidate, strict=True))
    sent, sid, tag, fork, join, total, e = candidate
    order = torch.argsort(total, descending=True, stable=True)
    sent, sid, tag, fork, join, total, e = (
        x[order] for x in (sent, sid, tag, fork, join, total, e)
    )
    # one int64 per state: sentence (14 bits), store (24), preterminal, fork, join
    tag_bits = max(1, (symbols - 1).bit_length())
    if 14 + 24 + tag_bits + 2 > 63:
        raise ValueError(f"{symbols} symbols are too many to pack a state key")
    key = (((sent << 24 | sid) << tag_bits | tag) << 2) | (fork << 1) | join
    unique, inverse = torch.unique(key, return_inverse=True)
    first = torch.full((len(unique),), len(total), dtype=torch.long, device=total.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(total), device=total.device), reduce="amin")
    sent, sid, tag, fork, join, total, e = (
        x[first] for x in (sent, sid, tag, fork, join, total, e)
    )
    kept, _ = _top_per_sentence(sent, total, beam)
    return tuple(x[kept] for x in (sent, sid, tag, fork, join, total, e))


def _top_per_sentence(sentence: torch.Tensor, score: torch.Tensor, k):
    """Indices of each sentence's `k` best scores, and each sentence's best score not kept.

    `k` is a number, or a tensor giving each sentence's (indexed by sentence).
    """
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(sentence[order], stable=True)]
    sorted_sentence = sentence[order]
    starts = torch.searchsorted(sorted_sentence, sorted_sentence, right=False)
    rank = torch.arange(len(order), device=score.device) - starts
    limit = k[sorted_sentence] if isinstance(k, torch.Tensor) else k
    kept = order[rank < limit]
    pruned = order[rank == limit]
    best = {
        int(s): float(v)
        for s, v in zip(sentence[pruned].tolist(), score[pruned].tolist(), strict=True)
    }
    return kept, best


def _kth_per_sentence(sentence: torch.Tensor, score: torch.Tensor, k: int) -> dict[int, float]:
    """Each sentence's `k`-th best score, or -inf where it has fewer than `k`."""
    if len(sentence) == 0:
        return {}
    order = torch.argsort(score, descending=True, stable=True)
    order = order[torch.argsort(sentence[order], stable=True)]
    sorted_sentence = sentence[order]
    starts = torch.searchsorted(sorted_sentence, sorted_sentence, right=False)
    rank = torch.arange(len(order), device=score.device) - starts
    at = order[rank == k - 1]
    found = dict(zip(sentence[at].tolist(), score[at].tolist(), strict=True))
    return {s: found.get(s, NEG) for s in torch.unique(sentence).tolist()}


def _concat_all(parts: list[Beams], slots: int, device) -> Beams:
    out = _empty(slots, device)
    for part in parts:
        out = _concat(out, part)
    return out


def _best_path(history: dict, s: int, start: int, end: int, length: int) -> list:
    """The best states of sentence `s`, admitted at step `start`, last advanced to step `end`."""
    path: list = []
    beams = history[end]
    mine = torch.nonzero(beams.sentence == s).squeeze(1)
    if len(mine) == 0:
        return [None] * length
    index = int(mine[torch.argmax(beams.score[mine])])
    for k in range(end, start, -1):
        beams = history[k]
        store = beams.store[index]
        size = int(beams.size[index])
        path.append(
            (
                int(beams.pretrm[index]),
                int(beams.fork[index]),
                int(beams.join[index]),
                tuple((int(store[j, 0]), int(store[j, 1])) for j in range(size)),
            )
        )
        index = int(beams.back[index])
    return list(reversed(path))
