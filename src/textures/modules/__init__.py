"""Modules available in textures and the default registry that holds them."""

from textures.modules.base import Module, ModuleConfig, Registry
from textures.modules.lm_scores import LMScores
from textures.modules.surprisal import Surprisal

REGISTRY = Registry([LMScores, Surprisal])

__all__ = ["REGISTRY", "Module", "ModuleConfig", "Registry"]
