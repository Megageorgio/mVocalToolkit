import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from conftest import make_wav
from mvocaltoolkit.api_models import RecordingPreparationRequest
from mvocaltoolkit.formats.ds_csv import label_from_row, row_from_label
from mvocaltoolkit.labels import Label
from mvocaltoolkit.models.store import InstalledModel
from mvocaltoolkit.training.dataset import SourceSpec, build_dataset, _label_for, _label_index
from mvocaltoolkit.training.openutau import VoicebankSpec, write_profile_dictionaries
from mvocaltoolkit.training.phonemizers import PROFILES, convert_label
from mvocaltoolkit.training.preparation import add_groups, cut_points, prepare_recordings, preserved_f0, segment_bounds
from mvocaltoolkit.training.preparation_check import model_language_ids, preparation_check


def test_multi13_canonical_is_not_applied_twice_and_preserves_groups():
    original = label_from_row(dict(ph_seq="de/e de/eh ru/ax en/i SP", ph_dur=".1 .2 .3 .2 .2", ph_num="2 1 1 1"))
    targets = {"de": "de-marzipan", "ru": "ru-hhskt", "en": "en-arpa-plus"}
    native, _ = convert_label(original, "ru", targets, alphabet="wfl-multi13-canonical", prefix=True)
    assert [p.text for p in native.phones] == ["de/eh", "de/e", "ru/ax", "en/iy", "SP"]
    same, _ = convert_label(native, "ru", targets, alphabet="wfl-native", prefix=True)
    assert same.phones == native.phones
    assert row_from_label("x", native)["ph_num"] == "2 1 1 1"
    assert original.phones[0].text == "de/e"


def test_russian_reductions_and_explicit_custom_split_reported():
    label = Label.from_segments([(0, .4, "ru/ax"), (.4, 1, "ru/py")])
    converted, report = convert_label(label, "ru", {"ru": "ru-openutau"}, alphabet="wfl-native", prefix=True)
    assert [p.text for p in converted.phones] == ["ru/a", "ru/pp"]
    assert report["lossy"]
    original = Label.from_segments([(0, 1, "unknown")])
    with pytest.raises(ValueError, match="Unmapped"):
        convert_label(original, "ru", {"ru": "ru-hhskt"})
    split, report = convert_label(original, "ru", {"ru": "ru-hhskt"}, overrides={"unknown": ["j", "a"]})
    assert [(p.start, p.end, p.text) for p in split.phones] == [(0, .5, "j"), (.5, 1, "a")]
    assert report["lossy"] and not report["unknown"]


def test_groups_do_not_disable_pause_segmentation_and_fades_stay_in_silence():
    label = Label.from_segments([(0, 1, "SP"), (1, 9, "ru/a"), (9, 10, "SP"), (10, 22, "ru/i"), (22, 23, "SP")])
    add_groups(label, PROFILES["ru-hhskt"]["vowels"], [])
    assert cut_points(label, 20, .25) == [0, 9.5, 23]
    bounds = segment_bounds(label, 0, 9.5, .1, 100)
    assert bounds["start"] == .9 and bounds["end"] == 9.1
    assert bounds["fade_in"] == pytest.approx(.1) and bounds["fade_out"] == pytest.approx(.1)
    sung = segment_bounds(label, 1, 9, .1, 100)
    assert sung == dict(start=1, end=9, fade_in=0, fade_out=0)


def test_existing_curve_is_cropped_without_modifying_source(tmp_path):
    wav = make_wav(tmp_path / "x.wav")
    ds = wav.with_suffix(".ds")
    ds.write_text(json.dumps([dict(f0_seq="0 100 200 300", f0_timestep=".25")]))
    contents = ds.read_bytes()
    values, hop = preserved_f0(wav, {}, .5, 1)
    assert values == [200, 300] and hop == .25
    assert ds.read_bytes() == contents


def test_same_basename_in_another_session_cannot_supply_missing_labels(tmp_path):
    a = make_wav(tmp_path / "session-a/clip.wav")
    b = make_wav(tmp_path / "session-b/clip.wav")
    a.with_suffix(".lab").write_text("0 10000000 a\n")
    a.with_suffix(".ds").write_text(json.dumps([dict(ph_seq="a", ph_dur="1", f0_seq="440 440", f0_timestep=".5")]))
    index = _label_index(tmp_path)
    assert _label_for(a, {}, index).phones[0].text == "a"
    assert _label_for(b, {}, index) is None
    # A root CSV keyed only by basename is also ambiguous.
    assert _label_for(b, {"clip": Label.from_segments([(0, 1, "wrong")])}, index) is None


def test_raw_wav_calls_wfl_with_verified_language_ids_and_keeps_originals(tmp_path):
    source = tmp_path / "source"
    original = make_wav(source / "clip.wav")
    original_bytes = original.read_bytes()
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "langs.txt").write_text("en,7\nru,2\n")
    model = InstalledModel(id="multi13-test", engine="wfl_asr", path=str(model_root), languages=["en", "ru"], layout={"generation": 2})
    assert model_language_ids(model) == {"en": 7, "ru": 2}
    class Models:
        catalog = SimpleNamespace(get=lambda key: None)
        def get_installed(self, key): return model
        async def require(self, *args): return model
    tk = SimpleNamespace(models=Models())
    class Job:
        def check_cancelled(self): pass
        def progress(self, *args, **kwargs): pass
    calls = []
    async def call(tk, job, engine, method, params):
        calls.append(engine)
        if engine == "wfl_asr_v2":
            assert params["lang_id"] == 2 and params["extra_lang_ids"] == [7]
            assert params["language_prefix"] == "always"
            return [{"ok": True, "phones": [[0, .5, "ru/ax", .9], [.5, 1, "en/iy", .8]]}]
        assert engine == "audio"
        for piece in params["slices"]:
            dest = Path(piece["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(original_bytes)
        return {"files": [p["path"] for p in params["slices"]]}
    req = RecordingPreparationRequest(folder=str(source), out_dir=str(tmp_path / "out"), language="ru", label_mode="missing",
        label_model=model.id, extra_languages=["en"], phonemizer_profiles={"ru": "ru-hhskt", "en": "en-arpa-plus"}, f0_method="none")
    check = preparation_check(tk, req)
    assert check["ok"] and check["pending_auto_labels"] == 1 and calls == []
    result = asyncio.run(prepare_recordings(tk, Job(), req, call))
    assert result["count"] == 1 and calls == ["wfl_asr_v2", "audio"]
    assert original.read_bytes() == original_bytes and not original.with_suffix(".lab").exists()
    assert (tmp_path / "out/original-labels/clip.json").is_file()
    assert "ru/ax en/iy" in (tmp_path / "out/transcriptions.csv").read_text()
    # Mixed labels require separate, unprefixed dictionaries for DiffSinger.
    built = build_dataset([SourceSpec(folder=str(tmp_path / "out"), speaker="s", language="ru")], tmp_path / "dataset",
        phonemizer_profiles=req.phonemizer_profiles)
    assert built.phonemes == {"ru": ["ax"], "en": ["iy"]}
    assert (tmp_path / "dataset/dictionary-en.txt").read_text() == "iy\tiy\n"
    assert json.loads((tmp_path / "dataset/dataset.json").read_text())["phonemizer_profiles"] == req.phonemizer_profiles
    invalid = req.model_copy(update={"extra_languages": ["not-a-language"]})
    with pytest.raises(ValueError, match="does not declare"):
        preparation_check(tk, invalid.model_copy(update={"out_dir": str(tmp_path / "unused")}))


def test_bank_writes_hhskt_and_stock_remappers_to_same_trained_inventory(tmp_path):
    phones = ["SP", "AP"] + ["ru/" + p for p in PROFILES["ru-hhskt"]["vowels"] + PROFILES["ru-hhskt"]["consonants"]]
    spec = VoicebankSpec(name="s", acoustic="unused", out_dir=str(tmp_path), phonemizer_profiles={"ru": "ru-hhskt"})
    made, report = write_profile_dictionaries(tmp_path, {}, phones, spec)
    assert {p.name for p in made} == {"dsdict-ru-hhskt.yaml", "dsdict-ru.yaml"}
    custom = yaml.safe_load((tmp_path / "dsdict-ru-hhskt.yaml").read_text())
    stock = yaml.safe_load((tmp_path / "dsdict-ru.yaml").read_text())
    assert {s["symbol"]: s["type"] for s in custom["symbols"]}["ru/x"] == "vowel"
    assert {r["from"]: r["to"] for r in stock["replacements"]}["bb"] == "ru/by"
    assert all(r["to"] in phones for r in stock["replacements"])
    assert any(r["reduced_vowel_folding"] for r in report)
