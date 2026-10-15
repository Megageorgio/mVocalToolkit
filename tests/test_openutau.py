"""The OpenUtau voicebank assembly and its linter, on fake export folders."""
import json

import pytest
import yaml

from mvocaltoolkit.training.openutau import VoicebankSpec, assemble, lint


def _export(folder, name, kind, phonemes, languages, speakers=(), dictionaries=None):
    folder.mkdir(parents=True)
    (folder / f"{name}.phonemes.json").write_text(json.dumps(phonemes), encoding="utf-8")
    (folder / f"{name}.languages.json").write_text(json.dumps(languages), encoding="utf-8")
    cfg = {"phonemes": f"{name}.phonemes.json", "languages": f"{name}.languages.json",
           "use_lang_id": len(languages) > 1, "hidden_size": 256, "sample_rate": 44100, "hop_size": 512}
    if kind == "acoustic":
        (folder / f"{name}.onnx").write_bytes(b"onnx")
        cfg.update({"acoustic": f"{name}.onnx", "vocoder": "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
                    "use_key_shift_embed": True, "use_speed_embed": True, "mel_base": "e"})
    else:
        for part in ("linguistic", "dur", "pitch", "variance"):
            (folder / f"{name}.{part}.onnx").write_bytes(b"onnx")
        cfg.update({"linguistic": f"{name}.linguistic.onnx", "dur": f"{name}.dur.onnx", "pitch": f"{name}.pitch.onnx",
                    "variance": f"{name}.variance.onnx", "predict_dur": True, "predict_pitch": True,
                    "predict_energy": True, "use_expr": True, "use_note_rest": True})
    if speakers:
        cfg["speakers"] = [f"{name}.{s}" for s in speakers]
        for s in speakers:
            (folder / f"{name}.{s}.emb").write_bytes(b"emb")
    (folder / "dsconfig.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for lang, entries in (dictionaries or {}).items():
        (folder / f"dictionary-{lang}.txt").write_text("\n".join(f"{g}\t{p}" for g, p in entries), encoding="utf-8")


def test_single_language_bank(tmp_path):
    phonemes = ["AP", "SP", "a", "k", "n"]
    _export(tmp_path / "aco", "acoustic", "acoustic", phonemes, {"ja": 0}, dictionaries={"ja": [("ka", "k a"), ("n", "n")]})
    _export(tmp_path / "var", "variance", "variance", phonemes, {"ja": 0})
    result = assemble(VoicebankSpec(name="Test Singer", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"),
                                    variance=[str(tmp_path / "var")], licenses=["CC BY 4.0"]))
    assert result.problems == []
    assert result.models == ["acoustic", "dur", "pitch", "variance"]
    assert result.languages == ["ja"] and result.speakers == []
    bank = tmp_path / "out" / "Test Singer"
    assert (bank / "character.txt").read_text(encoding="utf-8") == "name=Test Singer\n"
    character = yaml.safe_load((bank / "character.yaml").read_text(encoding="utf-8"))
    assert character["singer_type"] == "diffsinger" and "subbanks" not in character
    main = yaml.safe_load((bank / "dsconfig.yaml").read_text(encoding="utf-8"))
    assert main["acoustic"] == "dsmain/acoustic.onnx" and main["vocoder"].startswith("pc_nsf_hifigan")
    assert main["use_key_shift_embed"] is True and main["mel_base"] == "e"
    dur = yaml.safe_load((bank / "dsdur" / "dsconfig.yaml").read_text(encoding="utf-8"))
    assert dur["linguistic"] == "linguistic.onnx" and dur["dur"] == "dur.onnx" and dur["phonemes"] == "phonemes.json"
    pitch = yaml.safe_load((bank / "dspitch" / "dsconfig.yaml").read_text(encoding="utf-8"))
    assert pitch["phonemes"] == "phonemes.json" and pitch["use_expr"] is True and pitch["pitch"] == "pitch.onnx"
    assert (bank / "dsdur" / "linguistic.onnx").is_file() and (bank / "dsvariance" / "variance.onnx").is_file()
    dsdict = yaml.safe_load((bank / "dsdict.yaml").read_text(encoding="utf-8"))
    symbols = {s["symbol"]: s["type"] for s in dsdict["symbols"]}
    assert symbols["a"] == "vowel" and symbols["k"] == "stop" and symbols["n"] == "nasal" and symbols["SP"] == "vowel"
    assert {"grapheme": "ka", "phonemes": ["k", "a"]} in dsdict["entries"]
    assert (bank / "dsdur" / "dsdict.yaml").is_file() and (bank / "dspitch" / "dsdict.yaml").is_file()
    assert "CC BY 4.0" in (bank / "readme.txt").read_text(encoding="utf-8")
    report = json.loads((bank / "report.json").read_text(encoding="utf-8"))
    assert report["licenses"] == ["CC BY 4.0"] and report["problems"] == []
    assert "report.json" in result.files
    assert lint(bank) == []


def test_hhskt_default_and_compatible_dictionaries_reach_every_variance_head(tmp_path):
    from mvocaltoolkit.training.phonemizers import PROFILES
    p = PROFILES["ru-hhskt"]
    phones = ["SP", "AP", *p["vowels"], *p["consonants"]]
    _export(tmp_path / "aco", "acoustic", "acoustic", phones, {"ru": 0}, dictionaries={"ru": [("да", "d a")]})
    _export(tmp_path / "var", "variance", "variance", phones, {"ru": 0})
    result = assemble(VoicebankSpec(name="Hhskt", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"),
        variance=[str(tmp_path / "var")], phonemizer_profiles={"ru": "ru-hhskt"}, default_phonemizer_profile="ru-hhskt"))
    bank = tmp_path / "out/Hhskt"
    assert yaml.safe_load((bank / "character.yaml").read_text())["default_phonemizer"] == p["phonemizer"]
    for head in (bank, bank / "dsdur", bank / "dspitch", bank / "dsvariance"):
        assert (head / "dsdict-ru-hhskt.yaml").is_file() and (head / "dsdict-ru.yaml").is_file()
    assert "phonemizer-report.json" in result.files
    assert result.problems == []
    # Later project changes must not reinterpret an already trained alphabet.
    (tmp_path / "aco/dataset-report.json").write_text(json.dumps({"phonemizer_profiles": {"ru": "ru-hhskt"}}))
    with pytest.raises(ValueError, match="training snapshot"):
        assemble(VoicebankSpec(name="Conflict", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"),
            phonemizer_profiles={"ru": "ru-openutau"}))
    assert not (tmp_path / "out/Conflict").exists()


def test_custom_vocoder_keeps_its_data_and_weights_license(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja": 0}, dictionaries={"ja": [("a", "a")]})
    vocoder = tmp_path / "voc"
    vocoder.mkdir()
    (vocoder / "vocoder.onnx").write_bytes(b"onnx")
    (vocoder / "vocoder.yaml").write_text("name: vocoder\nmodel: vocoder.onnx\n", encoding="utf-8")
    (vocoder / "dataset-report.json").write_text(json.dumps({"license": "CC BY-NC-SA 4.0", "files": 5,
        "recordings": [{"speaker": "source", "license": "CC BY"}]}), encoding="utf-8")
    result = assemble(VoicebankSpec(name="Custom", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"), vocoder=str(vocoder)))
    assert result.problems == []
    bank = tmp_path / "out/Custom"
    assert (bank / "dsvocoder/vocoder.onnx").exists()
    report = json.loads((bank / "report.json").read_text(encoding="utf-8"))
    assert "CC BY-NC-SA 4.0" in report["licenses"] and "source: CC BY" in report["licenses"]
    assert any(item["files"] == 5 for item in report["training_data"])


def test_custom_vocoder_uses_yaml_selection_and_reports_mel_mismatch(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja":0}, dictionaries={"ja":[("a","a")]})
    aco_yaml = tmp_path / "aco/dsconfig.yaml"
    cfg = yaml.safe_load(aco_yaml.read_text())
    cfg.update(num_mel_bins=128, mel_scale="slaney", fft_size=2048)
    aco_yaml.write_text(yaml.safe_dump(cfg))
    vocoder = tmp_path / "voc"
    (vocoder / "graphs").mkdir(parents=True)
    (vocoder / "wrong.onnx").write_bytes(b"unselected")
    (vocoder / "graphs/chosen.onnx").write_bytes(b"selected")
    (vocoder / "vocoder.yaml").write_text(yaml.safe_dump(dict(model="graphs/chosen.onnx", num_mel_bins=80, mel_scale="htk", fft_size=1024)))
    result = assemble(VoicebankSpec(name="Custom", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"), vocoder=str(vocoder)))
    bank = tmp_path / "out/Custom"
    assert (bank / "dsvocoder/graphs/chosen.onnx").read_bytes() == b"selected"
    assert not (bank / "dsvocoder/wrong.onnx").exists()
    assert yaml.safe_load((bank / "dsvocoder/vocoder.yaml").read_text())["model"] == "graphs/chosen.onnx"
    assert all(any(key in p["problem"] for p in result.problems) for key in ("num_mel_bins", "mel_scale", "fft_size"))


def test_invalid_vocoder_reference_is_rejected_before_writing_a_bank(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja":0})
    vocoder = tmp_path / "voc"
    vocoder.mkdir()
    (tmp_path / "outside.onnx").write_bytes(b"original")
    (vocoder / "vocoder.yaml").write_text("model: ../outside.onnx\n")
    with pytest.raises(ValueError, match="inside its folder"):
        assemble(VoicebankSpec(name="Custom", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"), vocoder=str(vocoder)))
    assert not (tmp_path / "out").exists()
    assert (tmp_path / "outside.onnx").read_bytes() == b"original"


def test_multilingual_speakers_and_linter(tmp_path):
    phonemes = ["AP", "SP", "en/a", "en/k", "xx/a", "xx/k", "xx/n"]
    langs = {"en": 0, "xx": 1}
    _export(tmp_path / "aco", "acoustic", "acoustic", phonemes, langs, speakers=("alice", "bob"),
            dictionaries={"xx": [("ka", "k a"), ("n", "n")], "en": [("a", "a"), ("ka", "k a")]})
    _export(tmp_path / "var", "variance", "variance", phonemes, langs, speakers=("alice", "bob"))
    result = assemble(VoicebankSpec(name="Duo", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"),
                                    variance=[str(tmp_path / "var")], vowels={"xx": ["a"], "en": ["a"]}))
    assert result.problems == []
    assert result.speakers == ["alice", "bob"] and result.languages == ["en", "xx"]
    bank = tmp_path / "out" / "Duo"
    character = yaml.safe_load((bank / "character.yaml").read_text(encoding="utf-8"))
    assert [s["suffix"] for s in character["subbanks"]] == ["alice", "bob"]
    assert (bank / "embeds" / "alice.emb").is_file() and (bank / "dsdur" / "embeds" / "bob.emb").is_file()
    main = yaml.safe_load((bank / "dsconfig.yaml").read_text(encoding="utf-8"))
    assert main["speakers"] == ["embeds/alice", "embeds/bob"]
    xx = yaml.safe_load((bank / "dsdict-xx.yaml").read_text(encoding="utf-8"))
    assert {"grapheme": "ka", "phonemes": ["xx/k", "xx/a"]} in xx["entries"]
    assert {"from": "k", "to": "xx/k"} in xx["replacements"]  # a language without a built-in OpenUtau phonemizer
    assert {"symbol": "xx/n", "type": "nasal"} in xx["symbols"] and {"symbol": "en/a", "type": "vowel"} in xx["symbols"]
    en = yaml.safe_load((bank / "dsdict-en.yaml").read_text(encoding="utf-8"))
    assert "replacements" not in en  # en has a built-in OpenUtau phonemizer that tags itself
    assert (bank / "dsdict.yaml").is_file()
    # the linter: a model reference removed and a mismatched use_lang_id
    (bank / "dsdur" / "dur.onnx").unlink()
    cfg = yaml.safe_load((bank / "dspitch" / "dsconfig.yaml").read_text(encoding="utf-8"))
    cfg["use_lang_id"] = False
    (bank / "dspitch" / "dsconfig.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    problems = [p["problem"] for p in lint(bank)]
    assert any("dur: dur.onnx not found" in p for p in problems)
    assert any("use_lang_id differs" in p for p in problems)


def test_mismatched_exports_are_reported(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja": 0}, speakers=("x",), dictionaries={"ja": [("a", "a")]})
    _export(tmp_path / "var", "variance", "variance", ["AP", "SP", "a", "b"], {"ja": 0}, speakers=("y",))
    result = assemble(VoicebankSpec(name="Odd", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"), variance=[str(tmp_path / "var")]))
    problems = " ".join(p["problem"] for p in result.problems)
    assert "speakers differ" in problems and "phoneme set differs" in problems


def test_separate_variance_runs_keep_their_own_encoders(tmp_path):
    phones = ["AP", "SP", "a"]
    _export(tmp_path / "aco", "acoustic", "acoustic", phones, {"ja": 0}, dictionaries={"ja": [("a", "a")]})
    for name, keep in (("duration", "dur"), ("pitch", "pitch")):
        folder = tmp_path / name
        _export(folder, "variance", "variance", phones, {"ja": 0})
        (folder / "variance.linguistic.onnx").write_bytes(name.encode())
        for part in {"dur", "pitch", "variance"} - {keep}:
            (folder / f"variance.{part}.onnx").unlink()
    result = assemble(VoicebankSpec(name="Separate", out_dir=str(tmp_path / "out"),
                                    acoustic=str(tmp_path / "aco"),
                                    variance=[str(tmp_path / "duration"), str(tmp_path / "pitch")]))
    assert result.problems == []
    bank = tmp_path / "out" / "Separate"
    assert (bank / "dsdur" / "linguistic.onnx").read_bytes() == b"duration"
    assert (bank / "dspitch" / "linguistic.onnx").read_bytes() == b"pitch"


def test_multilingual_model_with_one_dictionary_keeps_language_tags(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "ja/a", "en/a"],
            {"ja": 0, "en": 1}, dictionaries={"ja": [("a", "a")]})
    result = assemble(VoicebankSpec(name="Tagged", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco")))
    bank = tmp_path / "out" / "Tagged"
    dictionary = yaml.safe_load((bank / "dsdict-ja.yaml").read_text(encoding="utf-8"))
    assert {"grapheme": "a", "phonemes": ["ja/a"]} in dictionary["entries"]
    assert any("no dictionary-en.txt" in p["problem"] for p in result.problems)


def test_missing_encoder_is_reported_for_its_own_component(tmp_path):
    phones = ["AP", "SP", "a"]
    _export(tmp_path / "aco", "acoustic", "acoustic", phones, {"ja": 0}, dictionaries={"ja": [("a", "a")]})
    _export(tmp_path / "var", "variance", "variance", phones, {"ja": 0})
    (tmp_path / "var" / "variance.linguistic.onnx").unlink()
    result = assemble(VoicebankSpec(name="Missing", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"),
                                    variance=[str(tmp_path / "var")]))
    for component in ("dsdur", "dspitch", "dsvariance"):
        assert any(component in p["file"] and "linguistic.onnx not found" in p["problem"] for p in result.problems)


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "bad/name", "bad\\name", "CON", "nul.txt", "Singer.", "line\nbreak"])
def test_invalid_bank_names_do_not_touch_output(tmp_path, name):
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        assemble(VoicebankSpec(name=name, out_dir=str(out), acoustic="missing", overwrite=True))
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_cannot_overwrite_source_export(tmp_path):
    folder = tmp_path / "aco"
    _export(folder, "acoustic", "acoustic", ["AP", "SP", "a"], {"ja": 0})
    with pytest.raises(ValueError, match="overlap"):
        assemble(VoicebankSpec(name="aco", out_dir=str(tmp_path), acoustic=str(folder), overwrite=True))
    assert (folder / "acoustic.onnx").read_bytes() == b"onnx"


def test_existing_bank_is_preserved_and_empty_folder_is_usable(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja": 0}, dictionaries={"ja": [("a", "a")]})
    bank = tmp_path / "out" / "Singer"
    bank.mkdir(parents=True)
    spec = VoicebankSpec(name="Singer", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco"))
    assert assemble(spec).problems == []
    with pytest.raises(FileExistsError):
        assemble(spec)
    assert (bank / "character.txt").read_text(encoding="utf-8") == "name=Singer\n"


def test_export_data_snapshot_carries_donor_license_into_bank(tmp_path):
    _export(tmp_path / "aco", "acoustic", "acoustic", ["AP", "SP", "a"], {"ja": 0}, dictionaries={"ja": [("a", "a")]})
    report = {"files": 10, "seconds": 600, "speakers": [{"speaker": "donor", "role": "donor", "language": "ja", "seconds": 600}],
              "recordings": [{"speaker": "donor", "license": "CC BY 4.0"}]}
    (tmp_path / "aco" / "dataset-report.json").write_text(json.dumps(report), encoding="utf-8")
    result = assemble(VoicebankSpec(name="Snapshot", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "aco")))
    bank = tmp_path / "out" / "Snapshot"
    saved = json.loads((bank / "report.json").read_text(encoding="utf-8"))
    assert saved["licenses"] == ["donor: CC BY 4.0"]
    assert saved["training_data"][0]["seconds"] == 600
    assert "donor (donor, ja): 10.0 minutes" in (bank / "readme.txt").read_text(encoding="utf-8")
    assert result.problems == []


def test_select_heads_independently_from_exports_and_keep_each_encoder(tmp_path):
    phones = ["AP", "SP", "a"]
    _export(tmp_path / "a", "acoustic", "acoustic", phones, {"xx": 0}, dictionaries={"xx": [("a", "a")]})
    for name in ("duration", "pitch"):
        _export(tmp_path / name, "variance", "variance", phones, {"xx": 0})
        (tmp_path / name / "variance.linguistic.onnx").write_bytes(name.encode())
    result = assemble(VoicebankSpec(name="Mixed", out_dir=str(tmp_path / "out"), acoustic=str(tmp_path / "a"),
        components={"dur": str(tmp_path / "duration"), "pitch": str(tmp_path / "pitch")}, phoneme_types={"a": "vowel"}))
    bank = tmp_path / "out/Mixed"
    assert result.problems == [] and result.models == ["acoustic", "dur", "pitch"]
    assert (bank / "dsdur/linguistic.onnx").read_bytes() == b"duration"
    assert (bank / "dspitch/linguistic.onnx").read_bytes() == b"pitch"
    assert not (bank / "dsvariance").exists()
