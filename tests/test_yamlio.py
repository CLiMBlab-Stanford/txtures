import pytest
from conftest import write

from txtures.yamlio import YamlError, load_mapping


def test_empty_file_reads_as_empty_mapping(tmp_path):
    assert load_mapping(write(tmp_path / "a.yml", "")) == {}


def test_duplicate_keys_are_rejected(tmp_path):
    path = write(tmp_path / "a.yml", "model: gpt2\nmodel: pythia\n")
    with pytest.raises(YamlError, match="duplicate key 'model'"):
        load_mapping(path)


def test_nested_duplicate_keys_are_rejected(tmp_path):
    path = write(tmp_path / "a.yml", "model:\n  a: x\n  a: y\n")
    with pytest.raises(YamlError, match="duplicate key 'a'"):
        load_mapping(path)


@pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", "1: a\n"])
def test_top_level_must_be_mapping_with_string_keys(tmp_path, text):
    with pytest.raises(YamlError, match="expected a mapping"):
        load_mapping(write(tmp_path / "a.yml", text))
