"""Modules available in textures and the default registry that holds them."""

from textures.modules.base import Item, Module, ModuleConfig, Registry, Resource, RunContext
from textures.modules.lm_scores import LMScores
from textures.modules.surprisal import Surprisal

REGISTRY = Registry([LMScores, Surprisal])

__all__ = ["REGISTRY", "Item", "Module", "ModuleConfig", "Registry", "Resource", "RunContext"]
