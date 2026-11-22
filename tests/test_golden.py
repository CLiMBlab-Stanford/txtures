"""Recompute golden outputs and compare them with the recorded ones (see `golden.py`)."""

import json

import golden
import pytest


def _marks(name):
    marks = [pytest.mark.heavy]
    if name in golden.TOOLS:
        marks.append(pytest.mark.tools)
    return marks


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=_marks(name)) if name in golden.HEAVY else name
        for name in golden.TARGETS
    ],
)
def test_golden_outputs(name):
    recorded_path = golden.path(name)
    assert recorded_path.exists(), (
        f"no golden values for {name}; record them with `python tests/golden.py --update {name}`"
    )
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    current = golden.compute(name)
    bump = golden.TARGETS[name].bump
    if recorded["version"] != current["version"]:
        pytest.fail(
            f"{bump} is {current['version']} but the golden values were recorded under "
            f"{recorded['version']}; regenerate them with "
            f"`python tests/golden.py --update {name}`"
        )
    if recorded.get("inputs") != current.get("inputs"):
        pytest.fail(
            f"{name}'s external inputs changed (recorded {recorded.get('inputs')}, now "
            f"{current.get('inputs')}), as when a dependency release brings new models; "
            f"regenerate with `python tests/golden.py --update {name}`"
        )
    changed = golden.differences(recorded["values"], current["values"])
    assert not changed, (
        f"{name} computes different values under the same version. If the change is "
        f"deliberate, increment {bump} and regenerate with "
        f"`python tests/golden.py --update {name}`.\n  " + "\n  ".join(changed[:20])
    )


def test_update_refuses_changed_values_under_the_same_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(golden, "GOLDEN", tmp_path)
    values = {"x": 1.0}
    target = golden.Target(lambda: 3, lambda: values, "the version")
    monkeypatch.setitem(golden.TARGETS, "fake", target)
    assert golden.update(["fake"]) == 0
    values["x"] = 2.0
    assert golden.update(["fake"]) == 1
    assert "increment the version first" in capsys.readouterr().err
    monkeypatch.setitem(golden.TARGETS, "fake", golden.Target(lambda: 4, lambda: values, "v"))
    assert golden.update(["fake"]) == 0
    assert json.loads((tmp_path / "fake.json").read_text())["values"] == {"x": 2.0}


def test_differences_compare_floats_within_the_tolerance():
    assert golden.differences({"a": [1.0, None]}, {"a": [1.0 + 1e-9, None]}) == []
    assert golden.differences([1.0], [1.001]) == ["[0]: 1.0 != 1.001"]
    assert golden.differences({"a": 1}, {"b": 1}) == [
        "/a: present in only one",
        "/b: present in only one",
    ]
