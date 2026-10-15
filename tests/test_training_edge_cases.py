"""Regression cases for local preflight, vocabulary typing and batch recovery."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from mvocaltoolkit.training.resources import resource_options, measured_limits, reduce_after_oom
from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus
from mvocaltoolkit.training.speech_preview import checked_speech_preview
from mvocaltoolkit.training.test_phrases import inventory, build_phrase, PhraseRequest, note_midi


def test_unlimited_validation_is_capped_even_at_minimum_training_batch(tmp_path):
    settings = dict(resources=dict(oom_retries=1), overrides=dict(max_val_batch_frames=-1, max_val_batch_size=0))
    run = Run(RunInfo(id="r", dir=str(tmp_path), spec=RunSpec(kind="diffsinger_acoustic", config=settings), status=RunStatus.failed, error_kind="oom"))
    (tmp_path / "PREPARED").touch()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(dict(max_batch_frames=1000, max_batch_size=1, max_val_batch_frames=-1, max_val_batch_size=0)))
    result = reduce_after_oom(run)
    assert result["validation_limits"] == dict(max_val_batch_frames=1000, max_val_batch_size=1)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    for key, value in result["validation_limits"].items():
        assert saved[key] == run.info.spec.config["overrides"][key] == run.info.spec.config["overlay"][key] == value
    assert reduce_after_oom(run) is None


def test_memory_sizing_caps_unlimited_validation_but_preserves_smaller_limits():
    config = dict(datasets=[dict(raw_data_dir="data", speaker="s", language="ru")], dictionaries={"ru": "dict"},
        resources=dict(auto_batch=True), overrides=dict(max_val_batch_frames=-1, max_val_batch_size=1))
    limits, report = measured_limits("diffsinger_acoustic", config, dict(gpus=[dict(index=0, memory_free_mb=4096)]), "cuda:0")
    assert report["measured"] and limits["max_val_batch_frames"] == limits["max_batch_frames"]
    assert limits["max_val_batch_size"] == 1


@pytest.mark.parametrize("options", ["false", [], dict(auto_batch="false"), dict(oom_retries=1.5), dict(oom_retries=True),
    dict(min_batch_frames=False), dict(memory_fraction=True), dict(memory_fraction=float("nan"))])
def test_resource_options_do_not_coerce_wrong_types_into_enabled_features(options):
    with pytest.raises(ValueError): resource_options(dict(resources=options))


def test_language_dictionary_types_win_and_disabled_profile_stays_disabled(tmp_path):
    (tmp_path / "phonemes.json").write_text(json.dumps(["SP", "a", "p", "x"]))
    (tmp_path / "dsdict-ru.yaml").write_text(yaml.safe_dump(dict(symbols=[dict(symbol="a", type="vowel"), dict(symbol="p", type="stop")])))
    (tmp_path / "dsdict.yaml").write_text(yaml.safe_dump(dict(symbols=[dict(symbol="a", type="stop"), dict(symbol="p", type="vowel"), dict(symbol="x")])))
    (tmp_path / "dataset-report.json").write_text(json.dumps(dict(phonemizer_profiles=dict(ru=""))))
    result = inventory(tmp_path, "ru")
    assert result["vowels"] == ["a"] and result["consonants"] == ["p"]
    assert result["unclassified"] == ["x"] and result["classification_profile"] is None


@pytest.mark.parametrize("filename, contents", [
    ("dsconfig.yaml", "[one, two]"), ("phonemes.json", '"a"'), ("phonemes.json", '["a", "a"]'),
    ("phonemes.json", '{"a": 0, "p": 0}'), ("phonemes.json", '{"a": "zero"}'),
    ("dsdict.yaml", "symbols: [wrong]"), ("dataset-report.json", "[]"),
    ("dataset-report.json", '{"phonemizer_profiles": ["ru"]}'),
])
def test_invalid_vocabularies_and_metadata_fail_with_actionable_validation(tmp_path, filename, contents):
    (tmp_path / "phonemes.json").write_text('["a", "p"]')
    (tmp_path / filename).write_text(contents)
    with pytest.raises(ValueError): inventory(tmp_path, "ru")


def test_explicit_empty_consonants_allow_retyping_a_phone_as_vowel(tmp_path):
    (tmp_path / "phonemes.json").write_text('["SP", "p"]')
    result = build_phrase(PhraseRequest(model_dir=str(tmp_path), mode="v", vowels=["p"], consonants=[]), tmp_path / "phrase")
    assert result["syllables"] == ["p"] and float(result["segment"]["ph_dur"]) == .35
    with pytest.raises(ValueError, match="Notes must"): note_midi("C" + "9" * 400)


def test_external_configuration_resolves_its_relative_vocabulary(tmp_path):
    checkpoint, settings = tmp_path / "checkpoint", tmp_path / "settings"
    checkpoint.mkdir(); settings.mkdir()
    (settings / "tokens.json").write_text('["a", "p"]')
    config = settings / "config.yaml"
    config.write_text("phonemes: tokens.json\n")
    result = inventory(checkpoint, "ru", str(config))
    assert result["vocabulary"] == str(settings / "tokens.json") and result["phonemes"] == ["a", "p"]


def test_relative_run_report_keeps_its_explicit_alphabet_choice(tmp_path):
    (tmp_path / "phonemes.json").write_text('["a", "p"]')
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/data.json").write_text(json.dumps(dict(phonemizer_profiles=dict(ru=""))))
    (tmp_path / "run.json").write_text(json.dumps(dict(spec=dict(config=dict(data_report="metadata/data.json")))))
    result = inventory(tmp_path, "ru")
    assert result["classification_profile"] is None and result["unclassified"] == ["a", "p"]


def test_empty_lora_base_does_not_resolve_to_the_current_directory(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "lora_config.json").write_text(json.dumps(dict(base_model="", lora_config={})))
    request = dict(kind="voxcpm_voice", model_dir=str(root), text="Test", out_dir="unmanaged")
    with pytest.raises(ValueError, match="local VoxCPM base"): checked_speech_preview(request)
    base = tmp_path / "base"
    base.mkdir()
    result = checked_speech_preview({**request, "base_model": str(base)})
    assert result["base_model"] == str(base.resolve()) and "out_dir" not in result


@pytest.mark.parametrize("options", [dict(text="  "), dict(steps=True), dict(max_tokens=0), dict(guidance=float("nan")), dict(device="cuda:-1")])
def test_invalid_speech_requests_fail_before_engine_submission(tmp_path, options):
    request = dict(kind="qwen_tts_voice", model_dir=str(tmp_path), text="Test")
    request.update(options)
    with pytest.raises(ValueError): checked_speech_preview(request)


def test_existing_speech_output_is_rejected_before_torch_import(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "src/mvocaltoolkit/engines/_runtime/mvt_speech_preview.py"
    spec = importlib.util.spec_from_file_location("speech_output_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package, output = tmp_path / "package", tmp_path / "output"
    package.mkdir(); output.mkdir()
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="already exists"):
        module.render("qwen_tts_voice", str(package), str(output), "Test")
