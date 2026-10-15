"""Dataset checks must catch expensive failures before a trainer is installed or started."""
import json
import shutil
import struct
import wave

import pytest
from conftest import make_wav

from mvocaltoolkit.labels import Label
from mvocaltoolkit.training.dataset import SourceSpec, build_dataset, validate_report
from mvocaltoolkit.training.doctor import inspect_recording


def _source(root, name="a", rate=16000):
    make_wav(root / f"{name}.wav", rate=rate)
    (root / f"{name}.lab").write_text("0 1000000 SP\n1000000 9000000 a\n9000000 10000000 SP\n", encoding="utf-8")


def test_overlap_gap_nonfinite_and_audio_bounds(tmp_path):
    audio = make_wav(tmp_path / "clip.wav")
    for segments, code in [([(0, .8, "a"), (.5, 1, "SP")], "overlap"),
                           ([(0, .4, "a"), (.6, 1, "SP")], "label_gap"),
                           ([(0, 1.2, "a")], "labels_past_audio"),
                           ([(0, .5, "a")], "unlabelled_tail"),
                           ([(0, float("nan"), "a")], "invalid_interval")]:
        findings, _ = inspect_recording(audio, Label.from_segments(segments))
        assert any(p["code"] == code and p["severity"] == "error" for p in findings)


def test_clipping_low_level_and_dc_offset(tmp_path):
    for value, codes in [(32767, {"clipping", "dc_offset"}), (0, {"low_level"})]:
        path = tmp_path / f"{value}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 44100, 44100, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<h", value) * 44100)
        findings, metadata = inspect_recording(path, Label.from_segments([(0, 1, "SP")]))
        assert codes <= {p["code"] for p in findings}
        assert metadata["sample_rate"] == 44100


def test_excludes_bad_labels_and_duplicates_without_validation_leakage(tmp_path):
    src = tmp_path / "src"
    _source(src)
    shutil.copy2(src / "a.wav", src / "duplicate.wav")
    shutil.copy2(src / "a.lab", src / "duplicate.lab")
    _source(src, "b", rate=17000)
    _source(src, "bad", rate=18000)
    (src / "bad.lab").write_text("0 8000000 a\n5000000 10000000 SP\n", encoding="utf-8")
    result = build_dataset([SourceSpec(str(src), "singer", "ja")], tmp_path / "out")
    assert result.files == 2
    assert len(result.speakers[0]["test_prefixes"]) == 1
    assert any(p["code"] == "duplicate" and p["excluded"] for p in result.problems)
    assert any(p["code"] == "overlap" and p["excluded"] for p in result.problems)
    assert result.coverage == {"ja": {"a": 2}}
    assert len({r["sha256"] for r in result.recordings}) == 2


def test_one_recording_has_blocking_validation_finding(tmp_path):
    _source(tmp_path / "src")
    result = build_dataset([SourceSpec(str(tmp_path / "src"), "one", "ja")], tmp_path / "out")
    assert any(p["code"] == "validation" and p["severity"] == "error" for p in result.problems)
    assert result.speakers[0]["test_prefixes"] == []


def test_same_speaker_can_have_two_languages_and_dictionary_snapshots(tmp_path):
    for lang, rate in [("ja", 16000), ("en", 18000)]:
        _source(tmp_path / lang, rate=rate)
        _source(tmp_path / lang, "b", rate=rate + 100)
    dictionary = tmp_path / "custom.txt"
    dictionary.write_text("a\ta\nmissing\tb\n", encoding="utf-8")
    result = build_dataset([SourceSpec(str(tmp_path / lang), "singer", lang, license="CC BY") for lang in ("ja", "en")],
                           tmp_path / "out", dictionaries={"ja": str(dictionary)})
    assert len(result.speakers) == 2 and result.files == 4
    assert (tmp_path / "out" / "singer" / "ja" / "transcriptions.csv").is_file()
    assert (tmp_path / "out" / "singer" / "en" / "transcriptions.csv").is_file()
    dictionary.write_text("changed\tz\n", encoding="utf-8")
    assert (tmp_path / "out" / "dictionary-ja.txt").read_text(encoding="utf-8").startswith("a\ta")
    assert all(r["license"] == "CC BY" for r in result.recordings)
    saved = json.loads((tmp_path / "out" / "dataset.json").read_text(encoding="utf-8"))
    assert saved["coverage"] == {"ja": {"a": 2}, "en": {"a": 2}}


def test_changed_audio_same_size_is_copied_again(tmp_path):
    src = tmp_path / "src"
    _source(src)
    source = SourceSpec(str(src), "singer", "ja")
    build_dataset([source], tmp_path / "out")
    path = src / "a.wav"
    data = bytearray(path.read_bytes())
    data[-2:] = b"\x00\x00"
    path.write_bytes(data)
    build_dataset([source], tmp_path / "out")
    assert (tmp_path / "out" / "singer" / "wavs" / "a.wav").read_bytes() == data


@pytest.mark.parametrize("speaker", ["../outside", "..", "bad/name"])
def test_rejects_speaker_path_escape(tmp_path, speaker):
    _source(tmp_path / "src")
    with pytest.raises(ValueError, match="Invalid"):
        build_dataset([SourceSpec(str(tmp_path / "src"), speaker, "ja")], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_output_inside_source(tmp_path):
    _source(tmp_path / "src")
    with pytest.raises(ValueError, match="overlap"):
        build_dataset([SourceSpec(str(tmp_path / "src"))], tmp_path / "src" / "prepared")


def test_report_gate_rejects_missing_data_and_pitch_without_notes(tmp_path):
    path = tmp_path / "dataset.json"
    with pytest.raises(ValueError, match="missing"):
        validate_report(str(path))
    path.write_text(json.dumps({"files": 2, "speakers": [{"files": 2, "note_files": 0}], "problems": []}), encoding="utf-8")
    validate_report(str(path))  # acoustic training does not require notes
    with pytest.raises(ValueError, match="pitch training needs notes"):
        validate_report(str(path), needs_notes=True)
    path.write_text(json.dumps({"files": 2, "problems": [{"problem": "unknown phones", "severity": "error"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown phones"):
        validate_report(str(path))
    path.write_text(json.dumps({"files": 2, "problems": [{"problem": "bad file", "severity": "error", "excluded": True}]}), encoding="utf-8")
    validate_report(str(path))  # excluded recordings do not block the remaining valid data


def test_explicit_extra_phonemes_are_allowed_with_a_custom_dictionary(tmp_path):
    src = tmp_path / "src"
    _source(src)
    _source(src, "b", rate=17000)
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("b\tb\n", encoding="utf-8")
    result = build_dataset([SourceSpec(str(src), "main", "ja")], tmp_path / "out", {"ja": str(dictionary)}, extra_phonemes=["a"])
    assert not any(p.get("code") == "unknown_phonemes" for p in result.problems)
