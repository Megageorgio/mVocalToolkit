import pytest

from mvocaltoolkit.training.dictionaries import read_entries
from mvocaltoolkit.training.dataset import SourceSpec, build_dataset
from conftest import make_wav


def test_text_dictionary_import_handles_bom_comments_and_spaces(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("\ufeff# comment\na\ta\nword k a\n", encoding="utf-8")
    assert read_entries(path) == [("a", ["a"]), ("word", ["k", "a"])]


def test_multilingual_openutau_dictionary_uses_selected_language(tmp_path):
    path = tmp_path / "dsdict.yaml"
    path.write_text("entries:\n- {grapheme: a, phonemes: [ja/a]}\n- {grapheme: enword, phonemes: [en/a]}\n- {grapheme: SP, phonemes: [SP]}\n", encoding="utf-8")
    assert read_entries(path, "ja") == [("a", ["a"]), ("SP", ["SP"])]


def test_yaml_dictionary_is_converted_in_the_prepared_snapshot(tmp_path):
    source = tmp_path / "source"
    for index in range(2):
        make_wav(source / f"clip{index}.wav", rate=16000 + index * 100)
        (source / f"clip{index}.lab").write_text("0 1000000 SP\n1000000 9000000 a\n9000000 10000000 SP\n", encoding="utf-8")
    dictionary = tmp_path / "dsdict.yaml"
    dictionary.write_text("entries:\n- {grapheme: a, phonemes: [ja/a]}\n", encoding="utf-8")
    result = build_dataset([SourceSpec(str(source), "singer", "ja")], tmp_path / "out", {"ja": str(dictionary)})
    assert not any(p.get("code") == "unknown_phonemes" for p in result.problems)
    assert (tmp_path / "out" / "dictionary-ja.txt").read_text(encoding="utf-8") == "a\ta\n"
    assert "ja/a" in dictionary.read_text(encoding="utf-8")  # original dictionary is unchanged


@pytest.mark.parametrize("content", ["symbols: []", "entries: [bad]", "entries: [{grapheme: a}]", "entries: []"])
def test_malformed_dictionary_has_an_actionable_error(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_entries(path, "ja")
