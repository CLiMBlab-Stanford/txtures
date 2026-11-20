"""Inside scores of fixed trees under a latent-variable PCFG, batched for the GPU.

Training sees only the treebank's trees, so each node's category is known
and only its substate is latent. For a node of category A with substates
a, the inside score is

- `I[a] = P(word | A_a)` at a preterminal,
- `I[a] = Σ_b U[a, b] I_child[b]` at a unary node, and
- `I[a] = Σ_{b,c} R[a, b, c] I_left[b] I_right[c]` at a binary node,

where `U` and `R` are the weights of the node's rule. The tree's
likelihood is the root's inside score (the root has one substate). The
expected count of each rule and lexical entry is its weight times the
derivative of the log-likelihood with respect to it, so one backward pass
replaces the outside pass.

Nodes are processed one height at a time, every tree at once. Each node's
inside vector is divided by its sum, and the logarithms of these factors
are added up, so long sentences do not underflow. Categories have up to
`substates` substates; unused ones are padded with zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

LEXICAL, UNARY, BINARY = 0, 1, 2


@dataclass
class Level:
    """The nodes of one height and kind: their rule or lexical index, and their children.

    Nodes are kept in a buffer ordered by height and kind, so each level
    fills the contiguous slice `start:stop`; `left` and `right` are buffer
    positions.
    """

    nodes: torch.Tensor  # node indices (in the order given to `encode`)
    kind: int
    start: int
    stop: int
    index: torch.Tensor  # rule (UNARY, BINARY) or lexical entry (LEXICAL) of each node
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None


@dataclass
class Encoded:
    """Trees as arrays: each node's height, kind, rule or lexical index, and children."""

    levels: list[Level]
    roots: torch.Tensor  # buffer positions of the roots
    nodes: int
    position: torch.Tensor  # buffer position of each node


def encode(
    trees: Sequence[Sequence[tuple[int, int, int, int]]], device: str | torch.device = "cpu"
) -> Encoded:
    """Encode trees given as node lists in post-order.

    Each node is `(kind, index, left, right)`: `left` and `right` are the
    positions of its children within the same tree (`-1` when absent), and
    the last node is the root.
    """
    kinds, indices, lefts, rights, heights, roots = [], [], [], [], [], []
    offset = 0
    for tree in trees:
        local_heights: list[int] = []
        for kind, index, left, right in tree:
            if kind == LEXICAL:
                height = 0
            elif kind == UNARY:
                height = local_heights[left] + 1
            else:
                height = max(local_heights[left], local_heights[right]) + 1
            local_heights.append(height)
            kinds.append(kind)
            indices.append(index)
            lefts.append(left + offset if left >= 0 else -1)
            rights.append(right + offset if right >= 0 else -1)
        heights += local_heights
        offset += len(tree)
        roots.append(offset - 1)
    kinds_a, heights_a = np.array(kinds), np.array(heights)
    indices_a, lefts_a, rights_a = np.array(indices), np.array(lefts), np.array(rights)
    order = np.lexsort((kinds_a, heights_a)) if len(heights_a) else np.zeros(0, dtype=int)
    position = np.empty_like(order)
    position[order] = np.arange(len(order))

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    levels = []
    keys = np.stack([heights_a[order], kinds_a[order]], axis=1) if len(order) else np.zeros((0, 2))
    boundaries = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
    for start, stop in zip(
        np.concatenate([[0], boundaries]), np.concatenate([boundaries, [len(order)]]), strict=True
    ):
        if stop <= start:
            continue
        chosen = order[start:stop]
        kind = int(kinds_a[chosen[0]])
        levels.append(
            Level(
                tensor(chosen),
                kind,
                int(start),
                int(stop),
                tensor(indices_a[chosen]),
                tensor(position[lefts_a[chosen]]) if kind != LEXICAL else None,
                tensor(position[rights_a[chosen]]) if kind == BINARY else None,
            )
        )
    return Encoded(levels, tensor(position[np.array(roots, dtype=int)]), offset, tensor(position))


def log_likelihoods(
    encoded: Encoded,
    binary: torch.Tensor,
    unary: torch.Tensor,
    lexical: torch.Tensor,
    node_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """The log-likelihood of each tree.

    `binary` is `[rules, S, S, S]` (parent, left, right substate), `unary`
    `[rules, S, S]` (parent, child), and `lexical` `[entries, S]`: the
    probability of each lexical entry's word given each substate of its tag.
    `node_offsets` (`[nodes, S]`, all zeros) is added to each node's inside
    scores; the derivative of the log-likelihood with respect to it is each
    node's outside scores, up to a factor per node.
    """
    substates = lexical.shape[1]
    inside = lexical.new_zeros(encoded.nodes, substates)
    log_scale = lexical.new_zeros(encoded.nodes)
    for level in encoded.levels:
        if level.kind == LEXICAL:
            scores = lexical[level.index]
            below = lexical.new_zeros(len(level.nodes))
        elif level.kind == UNARY:
            scores = torch.einsum("nab,nb->na", unary[level.index], inside[level.left])
            below = log_scale[level.left]
        else:
            scores = torch.einsum(
                "nabc,nb,nc->na", binary[level.index], inside[level.left], inside[level.right]
            )
            below = log_scale[level.left] + log_scale[level.right]
        if node_offsets is not None:
            scores = scores + node_offsets[level.nodes]
        total = scores.sum(dim=1)
        inside[level.start : level.stop] = scores / total.unsqueeze(1)
        log_scale[level.start : level.stop] = below + torch.log(total)
    return log_scale[encoded.roots] + torch.log(inside[encoded.roots, 0])


def expected_counts(
    encoded: Encoded,
    binary: torch.Tensor,
    unary: torch.Tensor,
    lexical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The total log-likelihood and the expected counts of every weight.

    A weight's expected count is the posterior expected number of times it
    is used, summed over the trees: the weight times the derivative of the
    total log-likelihood with respect to it.
    """
    params = [p.detach().requires_grad_(True) for p in (binary, unary, lexical)]
    total = log_likelihoods(encoded, *params).sum()
    grads = torch.autograd.grad(total, params, allow_unused=True)
    counts = [
        (g if g is not None else torch.zeros_like(p)) * p.detach()
        for g, p in zip(grads, params, strict=True)
    ]
    return total.detach(), counts[0], counts[1], counts[2]


def node_posteriors(
    encoded: Encoded,
    binary: torch.Tensor,
    unary: torch.Tensor,
    lexical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each tree's log-likelihood, and each node's inside, outside, and posterior scores.

    Inside scores are normalized to sum to one at each node; outside scores
    are correct up to a factor per node, which leaves ratios such as
    Berkeley's merge losses unchanged; posteriors sum to one at each node.
    The outside scores are exact even where a substate's inside score is 0.
    """
    offsets = lexical.new_zeros(encoded.nodes, lexical.shape[1]).requires_grad_(True)
    totals = log_likelihoods(encoded, binary.detach(), unary.detach(), lexical.detach(), None)
    shifted = log_likelihoods(encoded, binary.detach(), unary.detach(), lexical.detach(), offsets)
    (outside,) = torch.autograd.grad(shifted.sum(), [offsets])
    with torch.no_grad():
        inside = _normalized_inside(encoded, binary, unary, lexical)
        posterior = inside * outside
        sums = posterior.sum(dim=1, keepdim=True)
        posterior = torch.where(sums > 0, posterior / sums, posterior)
    return totals.detach(), inside, outside, posterior


def _normalized_inside(encoded, binary, unary, lexical) -> torch.Tensor:
    substates = lexical.shape[1]
    inside = lexical.new_zeros(encoded.nodes, substates)
    for level in encoded.levels:
        if level.kind == LEXICAL:
            scores = lexical[level.index]
        elif level.kind == UNARY:
            scores = torch.einsum("nab,nb->na", unary[level.index], inside[level.left])
        else:
            scores = torch.einsum(
                "nabc,nb,nc->na", binary[level.index], inside[level.left], inside[level.right]
            )
        inside[level.start : level.stop] = scores / scores.sum(dim=1, keepdim=True)
    return inside[encoded.position]
