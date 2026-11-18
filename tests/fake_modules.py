"""Stand-in modules that exercise the engine without language models.

`fake_scores` is persisted, uses a resource, and scores each region by its
length times a scale. `fake_feature` is not persisted and turns those scores
into two feature columns. `CALLS` counts how many documents each module has
processed, so tests can tell computation from reuse.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from txtures.modules import Item, Module, ModuleConfig, Registry, Resource, RunContext

CALLS: Counter[str] = Counter()
ORDER: list[str] = []
"""Texts of the documents `fake_scores` computed, in order."""
_CALLS_LOCK = threading.Lock()
STATE = {"resolved_revision": "rev1", "delay": 0.0, "fail_on": None}


def count(name: str, n: int) -> None:
    with _CALLS_LOCK:
        CALLS[name] += n


def reset() -> None:
    CALLS.clear()
    ORDER.clear()
    STATE.update(resolved_revision="rev1", delay=0.0, fail_on=None)


class Weights(Resource):
    kind = "fake_weights"
    version = 1

    def resolve(self, context: RunContext):
        resolved = dict(self.spec)
        if resolved.get("revision") is None:
            resolved["revision"] = STATE["resolved_revision"]
        return resolved

    def build(self, resolved, directory: Path, context: RunContext) -> None:
        count("build", 1)
        (directory / "weights.json").write_text(json.dumps({"scale": resolved["scale"]}))

    def load(self, resolved, directory: Path, context: RunContext):
        count("load", 1)
        return json.loads((directory / "weights.json").read_text())["scale"]


class FakeScoresConfig(ModuleConfig):
    model: str = "m"
    revision: str | None = None
    scale: int = 1


class FakeScores(Module):
    name = "fake_scores"
    contract_version = 1
    Config = FakeScoresConfig
    sweeps = {"model": ("model", "revision"), "scale": ("scale",)}
    batch_size = 2

    def resources(self, config):
        return {
            "weights": Weights(
                {"model": config.model, "revision": config.revision, "scale": config.scale}
            )
        }

    def notes(self, config):
        return ["revision is not pinned"] if config.revision is None else []

    def run_batch(self, config, items: list[Item], context):
        results = []
        for item in items:
            text = item.document.text
            if STATE["fail_on"] and STATE["fail_on"] in text:
                raise RuntimeError(f"failed on {text!r}")
            if STATE["delay"]:
                time.sleep(STATE["delay"])
            scale = item.resources["weights"]
            lengths = [end - start for start, end in item.document.regions]
            results.append({"values": np.array(lengths, dtype=np.float64) * scale})
        with _CALLS_LOCK:
            ORDER.extend(item.document.text for item in items)
        count("fake_scores", len(items))
        return results


class FakeFeatureConfig(ModuleConfig):
    offset: int = 0
    mode: Literal["plain", "half"] = "plain"


class FakeFeature(Module):
    name = "fake_feature"
    contract_version = 1
    Config = FakeFeatureConfig
    requires = ("fake_scores",)
    features = ("length", "double")
    sweeps = {"offset": ("offset",)}
    persist = False

    def check(self, config, corpus):
        if config.mode == "half" and any(len(d.regions) % 2 for d in corpus.documents):
            return ["half mode needs an even number of regions per document"]
        return []

    def run(self, config, item: Item, context):
        count("fake_feature", 1)
        return item.upstream["fake_scores"]["values"] + config.offset

    def columns(self, config, document, result):
        return {"length": result, "double": result * 2}


def registry() -> Registry:
    return Registry([FakeScores, FakeFeature])
