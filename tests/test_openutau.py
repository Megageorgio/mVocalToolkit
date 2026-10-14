"""The OpenUtau voicebank assembly and its linter, on fake export folders."""
import json

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
    assert dur["linguistic"] == "../dsmain/linguistic.onnx" and dur["dur"] == "dur.onnx" and dur["phonemes"] == "phonemes.json"
    pitch = yaml.safe_load((bank / "dspitch" / "dsconfig.yaml").read_text(encoding="utf-8"))
    assert pitch["phonemes"] == "../dsdur/phonemes.json" and pitch["use_expr"] is True and pitch["pitch"] == "pitch.onnx"
    assert (bank / "dsmain" / "linguistic.onnx").is_file() and (bank / "dsvariance" / "variance.onnx").is_file()
    dsdict = yaml.safe_load((bank / "dsdict.yaml").read_text(encoding="utf-8"))
    symbols = {s["symbol"]: s["type"] for s in dsdict["symbols"]}
    assert symbols["a"] == "vowel" and symbols["k"] == "stop" and symbols["n"] == "nasal" and symbols["SP"] == "vowel"
    assert {"grapheme": "ka", "phonemes": ["k", "a"]} in dsdict["entries"]
    assert (bank / "dsdur" / "dsdict.yaml").is_file() and (bank / "dspitch" / "dsdict.yaml").is_file()
    assert "CC BY 4.0" in (bank / "readme.txt").read_text(encoding="utf-8")
    assert lint(bank) == []


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
