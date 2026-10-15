from collections import Counter
import json

import pytest

from mvocaltoolkit.training.recording_plan import RecordingPlanRequest, recording_plan


def report(tmp_path):
    dictionary = tmp_path / "dictionary-ru.txt"
    dictionary.write_text("one\ta p\ntwo\ti t\n", encoding="utf-8")
    data = dict(dictionaries=dict(ru=dictionary.name), phonemizer_profiles=dict(ru="ru-hhskt"),
        coverage=dict(ru=dict(a=20, i=20, p=30, t=10)),
        speaker_coverage=dict(first=dict(ru=dict(a=12, i=0, p=0, t=10)), second=dict(ru=dict(a=8, i=20, p=30, t=0))))
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def request(path, **options):
    return RecordingPlanRequest(report=str(path), language="ru", speaker="first", include_profile=False, **options)


def test_plans_use_individual_coverage_and_account_for_repetitions_without_modifying_data(tmp_path):
    path = report(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = recording_plan(request(path))
    assert {row["phone"] for row in result["needed"]} == {"i", "p"}
    assert result["rows"] and not result["remaining"]
    planned = Counter()
    for row in result["rows"]:
        for phone in row["phonemes"]: planned[phone] += row["repetitions"]
    assert dict(planned) == result["planned_occurrences"]
    assert all(result["counts"].get(p, 0) + planned[p] >= 5 for p in ("i", "p"))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original
    assert recording_plan(request(path)) == result
    combined = recording_plan(request(path).model_copy(update=dict(speaker="")))
    assert not combined["needed"] and not combined["rows"]


def test_explicit_targets_keep_well_covered_anchor_vowels_and_soft_consonants(tmp_path):
    path = report(tmp_path)
    result = recording_plan(request(path, targets=["ru/py"], consonants=["py"], vowels=["a"], patterns=["cv", "vc"]))
    assert {row["phone"] for row in result["needed"]} == {"py"}
    assert {tuple(row["phonemes"]) for row in result["rows"]} == {("a", "py"), ("py", "a")}
    assert "p" not in result["planned_occurrences"] and not result["remaining"]


def test_limits_and_unknown_types_report_remaining_deficits_honestly(tmp_path):
    path = report(tmp_path)
    limited = recording_plan(request(path, limit=1))
    assert len(limited["rows"]) == 1 and limited["remaining"]
    custom = recording_plan(request(path, targets=["zz"], vowels=[], consonants=[]))
    assert custom["unclassified"] == ["zz"] and custom["rows"][0]["pattern"] == "phone"
    assert custom["remaining"] == {"zz": 2}


def test_profile_opt_out_and_selected_language_are_respected(tmp_path):
    path = report(tmp_path)
    data = json.loads(path.read_text())
    data["phonemizer_profiles"]["ru"] = ""
    path.write_text(json.dumps(data))
    result = recording_plan(request(path).model_copy(update=dict(include_profile=True)))
    assert result["profile"] is None and result["unclassified"] == ["i", "p"]
    with pytest.raises(ValueError, match="singer/language"): recording_plan(request(path).model_copy(update=dict(speaker="missing")))
    with pytest.raises(ValueError, match="selected language"): recording_plan(request(path).model_copy(update=dict(language="en")))
    with pytest.raises(ValueError, match="selected language"): recording_plan(request(path, targets=["en/p"]))


def test_legacy_profile_inference_does_not_expand_the_alphabet_without_selection(tmp_path):
    path = report(tmp_path)
    data = json.loads(path.read_text())
    data.pop("phonemizer_profiles")
    path.write_text(json.dumps(data))
    result = recording_plan(request(path).model_copy(update=dict(include_profile=True)))
    assert result["profile_inferred"] and {p["phone"] for p in result["needed"]} == {"i", "p"}
    selected = recording_plan(request(path).model_copy(update=dict(include_profile=True, profile="ru-hhskt")))
    assert not selected["profile_inferred"] and len(selected["needed"]) > len(result["needed"])


def test_cvc_counts_repeated_consonants_and_rejects_ambiguous_types(tmp_path):
    path = report(tmp_path)
    result = recording_plan(request(path, targets=["p"], patterns=["cvc"]))
    assert result["rows"][0]["phonemes"].count("p") == 2 and result["planned_occurrences"]["p"] == 6
    assert not result["remaining"]
    with pytest.raises(ValueError, match="overlap"): recording_plan(request(path, vowels=["a"], consonants=["a"]))


def test_api_returns_plan_without_queueing_engines_or_creating_outputs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home, Settings
    from mvocaltoolkit.toolkit import Toolkit
    tk = Toolkit(Home(tmp_path / "home"), Settings(catalog_url="", auto_install_engines=False, auto_download_models=False))
    path = report(tmp_path)
    async def forbidden(*args, **kwargs): raise AssertionError("No engine should be called")
    monkeypatch.setattr(tk.engines, "call", forbidden)
    with TestClient(create_app(tk)) as client:
        result = client.post("/training/dataset/recording-plan", json=request(path).model_dump())
        assert result.status_code == 200 and result.json()["rows"]
        assert not tk.jobs.jobs
        invalid = client.post("/training/dataset/recording-plan", json={**request(path).model_dump(), "patterns": []})
        assert invalid.status_code == 400 and not tk.jobs.jobs
