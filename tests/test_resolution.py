from typing import Literal

import pytest
from conftest import write

from textures.definitions import Definitions
from textures.definitions.resolution import derive_label
from textures.errors import DefinitionError
from textures.modules import Module, ModuleConfig, Registry
from textures.modules.lm_scores import LMScoresConfig


def problems(error: pytest.ExceptionInfo[DefinitionError]) -> str:
    return "\n".join(str(problem) for problem in error.value.problems)


def column_names(definitions: Definitions, workflow: str = "main", **kwargs) -> list[str]:
    return [column.name for column in definitions.expand(workflow, **kwargs).columns()]


# Flattening and defaults


def test_packaged_main_uses_schema_defaults():
    expansion = Definitions.packaged().expand()
    (lm,) = expansion.variants["lm_scores"]
    assert lm.config == LMScoresConfig()
    assert [column.name for column in expansion.columns()] == ["surprisal"]


def test_digest_ignores_how_defaults_are_written(store):
    write(store / "configs" / "lm_scores" / "explicit.yml", "model: gpt2\ndtype: float32\n")
    write(store / "workflows" / "explicit.yml", "lm_scores: explicit\n")
    definitions = Definitions.load(store)
    main = definitions.expand("main").variants["surprisal"][0]
    explicit = definitions.expand("explicit").variants["surprisal"][0]
    assert main.digest == explicit.digest


def test_store_main_partially_overrides_packaged_defaults(store):
    write(store / "configs" / "lm_scores" / "main.yml", "dtype: bfloat16\n")
    write(store / "configs" / "lm_scores" / "derived.yml", "extends: main\nmodel: pythia\n")
    write(store / "configs" / "lm_scores" / "plain.yml", "model: pythia\n")
    definitions = Definitions.load(store)
    assert definitions.config("lm_scores", "main").variants[0].config.dtype == "bfloat16"
    derived = definitions.config("lm_scores", "derived").variants[0].config
    assert (derived.model, derived.dtype) == ("pythia", "bfloat16")
    plain = definitions.config("lm_scores", "plain").variants[0].config
    assert (plain.model, plain.dtype) == ("pythia", "float32")


def test_extends_chains_apply_overrides_in_order(store):
    configs = store / "configs" / "lm_scores"
    write(configs / "a.yml", "model: a\nwindow: 512\n")
    write(configs / "b.yml", "extends: a\nmodel: b\n")
    write(configs / "c.yml", "extends: b\nstride: 128\n")
    config = Definitions.load(store).config("lm_scores", "c").variants[0].config
    assert (config.model, config.window, config.stride) == ("b", 512, 128)


def test_extends_cycle_is_reported(store):
    configs = store / "configs" / "lm_scores"
    write(configs / "a.yml", "extends: b\n")
    write(configs / "b.yml", "extends: a\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "cycle: a -> b -> a" in problems(error)


def test_missing_parent_is_reported(store):
    write(store / "configs" / "lm_scores" / "a.yml", "extends: nothere\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "no lm_scores config named 'nothere'" in problems(error)


def test_schema_errors_name_the_field(store):
    write(store / "configs" / "lm_scores" / "a.yml", "dtype: float8\nwindow: 4\nstride: 4\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    text = problems(error)
    assert "dtype:" in text


def test_unknown_config_key_is_rejected(store):
    write(store / "configs" / "surprisal" / "a.yml", "unit: bits\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("surprisal", "a")
    assert "unit: Extra inputs are not permitted" in problems(error)


# Sweeps


@pytest.mark.parametrize(
    ("key", "value", "label"),
    [
        ("model", "EleutherAI/pythia-410m", "pythia410m"),
        ("model", "gpt2", "gpt2"),
        ("model", "org/1b-model", "model1bmodel"),
        ("window", 512, "window512"),
        ("rate", 0.5, "rate05"),
        ("flag", True, "flagTrue"),
    ],
)
def test_derived_labels(key, value, label):
    assert derive_label(key, value) == label


def test_list_sweep_matches_hand_written_configs(store):
    configs = store / "configs" / "lm_scores"
    write(configs / "swept.yml", "model: [gpt2, EleutherAI/pythia-410m]\n")
    write(configs / "gpt2.yml", "model: gpt2\n")
    write(configs / "pythia.yml", "model: EleutherAI/pythia-410m\n")
    write(store / "workflows" / "swept.yml", "lm_scores: swept\n")
    write(store / "workflows" / "listed.yml", "lm_scores: [gpt2, pythia]\n")
    definitions = Definitions.load(store)
    swept = definitions.expand("swept")
    listed = definitions.expand("listed")
    assert [v.digest for v in swept.variants["surprisal"]] == [
        v.digest for v in listed.variants["surprisal"]
    ]
    assert [c.name for c in swept.columns()] == ["surprisal_gpt2", "surprisal_pythia410m"]
    assert [c.name for c in listed.columns()] == ["surprisal_gpt2", "surprisal_pythia"]


def test_labeled_sweep_entries_may_set_their_group(store):
    write(
        store / "configs" / "lm_scores" / "families.yml",
        "model:\n  small: gpt2\n  pythia: {model: EleutherAI/pythia-410m, revision: abc123}\n",
    )
    resolved = Definitions.load(store).config("lm_scores", "families")
    configs = [variant.config for variant in resolved.variants]
    assert [(c.model, c.revision) for c in configs] == [
        ("gpt2", None),
        ("EleutherAI/pythia-410m", "abc123"),
    ]
    assert [variant.labels[0].value for variant in resolved.variants] == ["small", "pythia"]


def test_group_field_cannot_be_fixed_alongside_its_sweep(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: [gpt2, pythia]\nrevision: abc\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "revision cannot be set alongside a model sweep" in problems(error)


def test_sweep_entries_may_not_set_fields_outside_their_group(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model:\n  a: {model: gpt2, dtype: float16}\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "model entries may set only model, revision; got dtype" in problems(error)


def test_unsweepable_list_is_explained(store):
    write(store / "configs" / "lm_scores" / "a.yml", "dtype: [float32, float16]\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "dtype cannot be swept (sweepable parameters: model, window, context)" in problems(error)


def test_repeated_derived_labels_ask_for_explicit_labels(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: [a/gpt2, b/gpt2]\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    assert "repeats label gpt2; give explicit labels" in problems(error)


def test_invalid_variant_names_its_labels(store):
    write(
        store / "configs" / "lm_scores" / "a.yml",
        "window:\n  big: {window: 1024, stride: 256}\n  small: {window: 128, stride: 256}\n",
    )
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).config("lm_scores", "a")
    text = problems(error)
    assert "stride must be smaller than window (variant small)" in text
    assert "variant big" not in text


def test_single_value_sweep_adds_no_label(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: [gpt2]\n")
    write(store / "workflows" / "a.yml", "lm_scores: a\n")
    assert column_names(Definitions.load(store), "a") == ["surprisal"]


def test_sweeps_expand_as_product_and_propagate_downstream(store):
    write(
        store / "configs" / "lm_scores" / "a.yml",
        "context: [document, sentence]\nmodel: [x/m1, x/m2]\n",
    )
    write(store / "configs" / "surprisal" / "b.yml", "correction: [word_boundary, none]\n")
    write(store / "workflows" / "w.yml", "lm_scores: a\nsurprisal: b\n")
    expansion = Definitions.load(store).expand("w")
    assert len(expansion.variants["lm_scores"]) == 4
    assert [column.name for column in expansion.columns()] == [
        "surprisal_m1_document_wordboundary",
        "surprisal_m1_document_none",
        "surprisal_m1_sentence_wordboundary",
        "surprisal_m1_sentence_none",
        "surprisal_m2_document_wordboundary",
        "surprisal_m2_document_none",
        "surprisal_m2_sentence_wordboundary",
        "surprisal_m2_sentence_none",
    ]
    for variant in expansion.variants["surprisal"]:
        assert variant.upstream["lm_scores"].digest == variant.lineage["lm_scores"]


def test_config_id_labels_precede_sweep_labels(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: gpt2\n")
    write(store / "configs" / "lm_scores" / "b.yml", "model: pythia\nwindow: [512, 1024]\n")
    write(store / "workflows" / "w.yml", "lm_scores: [a, b]\n")
    assert column_names(Definitions.load(store), "w") == [
        "surprisal_a",
        "surprisal_b_window512",
        "surprisal_b_window1024",
    ]


def test_max_variants_limit_is_enforced(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: [a, b, c]\n")
    write(store / "workflows" / "w.yml", "lm_scores: a\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).expand("w", max_variants=2)
    assert "lm_scores expands to 3 variants" in problems(error)


def test_column_collision_detection(store):
    write(store / "configs" / "lm_scores" / "a.yml", "model: {x: gpt2, y: pythia}\n")
    write(store / "configs" / "lm_scores" / "a_x.yml", "model: other\n")
    write(store / "workflows" / "w.yml", "lm_scores: [a, a_x]\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).expand("w")
    assert "two variants produce column 'surprisal_a_x'" in problems(error)


# Workflows


def test_workflow_without_extends_starts_from_packaged_main(store):
    write(store / "workflows" / "bare.yml", "surprisal: main\n")
    definitions = Definitions.load(store)
    assert definitions.workflow("bare") == {"lm_scores": ("main",), "surprisal": ("main",)}


def test_workflow_rejects_unknown_modules_and_bad_values(store):
    write(store / "workflows" / "w.yml", "parser: main\nsurprisal: [main, main]\nlm_scores: 3\n")
    with pytest.raises(DefinitionError) as error:
        Definitions.load(store).workflow("w")
    text = problems(error)
    assert "unknown module 'parser'" in text
    assert "surprisal lists a config ID twice" in text
    assert "lm_scores must name a config ID" in text


def test_requesting_a_module_without_features_is_rejected():
    with pytest.raises(DefinitionError) as error:
        Definitions.packaged().expand(modules=["lm_scores"])
    assert "module 'lm_scores' has no feature columns" in problems(error)


# Shared ancestors


class Source(Module):
    class Config(ModuleConfig):
        value: Literal["a", "b", "c"] = "a"

    name = "source"
    contract_version = 1
    sweeps = {"value": ("value",)}


class Left(Module):
    class Config(ModuleConfig):
        pass

    name = "left"
    contract_version = 1
    requires = ("source",)


class Right(Module):
    class Config(ModuleConfig):
        pass

    name = "right"
    contract_version = 1
    requires = ("source",)


class Join(Module):
    class Config(ModuleConfig):
        pass

    name = "join"
    contract_version = 1
    requires = ("left", "right")
    features = ("joined",)


def test_shared_ancestors_are_not_mixed(store):
    registry = Registry([Source, Left, Right, Join])
    write(store / "configs" / "source" / "swept.yml", "value: [a, b]\n")
    write(store / "workflows" / "main.yml", "source: swept\nleft: main\nright: main\njoin: main\n")
    definitions = Definitions.load(store, registry)
    expansion = definitions.expand("main")
    joins = expansion.variants["join"]
    assert len(joins) == 2
    for variant in joins:
        left, right = variant.upstream["left"], variant.upstream["right"]
        assert left.lineage["source"] == right.lineage["source"]
    assert [c.name for c in expansion.columns(registry)] == ["joined_a", "joined_b"]


def test_registry_rejects_inconsistent_modules():
    class Bad(Module):
        class Config(ModuleConfig):
            model: str = "x"

        name = "bad"
        contract_version = 1
        sweeps = {"model": ("revision",)}

    with pytest.raises(ValueError, match="must list 'model' first"):
        Registry([Bad])
