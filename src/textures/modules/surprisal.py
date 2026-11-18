"""Region surprisal aggregated from language-model token scores."""

from __future__ import annotations

from typing import Literal

from textures.modules.base import Module, ModuleConfig


class SurprisalConfig(ModuleConfig):
    """Parameters for aggregating token scores into region surprisal."""

    correction: Literal["word_boundary", "none"] = "word_boundary"
    """Whether to apply the word-boundary correction to each region."""

    units: Literal["bits", "nats"] = "bits"
    """Logarithm base of the reported surprisal: 2 for bits, e for nats."""


class Surprisal(Module):
    """Compute the surprisal of each region from `lm_scores` output."""

    name = "surprisal"
    contract_version = 1
    Config = SurprisalConfig
    requires = ("lm_scores",)
    features = ("surprisal",)
    sweeps = {
        "correction": ("correction",),
        "units": ("units",),
    }
