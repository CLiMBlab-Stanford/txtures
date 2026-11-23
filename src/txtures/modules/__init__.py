"""Modules available in txtures and the default registry that holds them."""

from txtures.modules.base import (
    Item,
    Module,
    ModuleConfig,
    Registry,
    Resource,
    RunContext,
    Workload,
)
from txtures.modules.dependency_locality import DependencyLocality
from txtures.modules.lm_scores import LMScores
from txtures.modules.named_entity import NamedEntity
from txtures.modules.ngram import NgramSurprisal, UnigramSurprisal
from txtures.modules.parse import Parse
from txtures.modules.pos import Pos
from txtures.modules.sentences import Sentences
from txtures.modules.surprisal import Surprisal
from txtures.modules.word_length import WordLength

REGISTRY = Registry(
    [
        LMScores,
        Surprisal,
        UnigramSurprisal,
        NgramSurprisal,
        WordLength,
        Sentences,
        Parse,
        Pos,
        NamedEntity,
        DependencyLocality,
    ]
)

__all__ = [
    "REGISTRY",
    "Item",
    "Module",
    "ModuleConfig",
    "Registry",
    "Resource",
    "RunContext",
    "Workload",
    "resource_class",
]


def resource_class(kind: str) -> type[Resource]:
    """Return the resource class of `kind`, among every `Resource` subclass loaded.

    Raises:
        KeyError: No loaded resource class has this kind.
    """
    import txtures.gcg.treebank  # noqa: F401 - defines the gcg_treebank resource
    import txtures.lvpcfg.resource  # noqa: F401 - defines lvpcfg_grammar and synproc_model
    import txtures.tools  # noqa: F401 - defines the tools_image resource

    pending, found = list(Resource.__subclasses__()), {}
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if isinstance(getattr(cls, "kind", None), str):
            found.setdefault(cls.kind, cls)
    return found[kind]
