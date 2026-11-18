"""Modules available in txtures and the default registry that holds them."""

from txtures.modules.base import Item, Module, ModuleConfig, Registry, Resource, RunContext
from txtures.modules.lm_scores import LMScores
from txtures.modules.surprisal import Surprisal

REGISTRY = Registry([LMScores, Surprisal])

__all__ = ["REGISTRY", "Item", "Module", "ModuleConfig", "Registry", "Resource", "RunContext"]
