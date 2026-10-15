import asyncio
import json

import pytest
from conftest import make_wav

from mvocaltoolkit import formats
from mvocaltoolkit.api_models import RecordingPreparationRequest
from mvocaltoolkit.formats.ds_csv import label_from_row, row_from_label
from mvocaltoolkit.labels import Label
from mvocaltoolkit.training.dataset import SourceSpec, build_dataset
from mvocaltoolkit.training.preparation import add_groups, crop_label, cut_points, ds_segment, notes_for_label, prepare_recordings


def test_pause_cuts_preserve_word_and_phone_timing():
    label = Label.from_segments([(0, 9, "a"), (9, 10, "SP"), (10, 22, "i"), (22, 23, "SP"), (23, 30, "u")])
    points = cut_points(label, 20, .25)
    assert points == [0, 9.5, 22.5, 30]
    assert cut_points(Label.from_segments([(0, 30, "a")]), 20, .25) == [0, 30]
    parts = [crop_label(label, a, b) for a, b in zip(points, points[1:])]
    assert sum(p.end_time() for p in parts) == 30
    assert parts[0].phones[-1].text == parts[1].phones[0].text == "SP"
    assert parts[1].phones[1].start == .5


def test_notes_and_groups_fill_rests_and_set_slurs():
    label = Label.from_segments([(0, .2, "k"), (.2, .5, "a"), (.5, .8, "i"), (.8, 1, "SP")])
    assert add_groups(label, ["a", "i"], ["r", "l"])
    notes = notes_for_label(label, [{"start": .1, "end": .7, "pitch": 60}, {"start": .7, "end": .9, "pitch": 62}])
    label.tiers["notes"] = notes
    row = row_from_label("clip", label)
    assert row["ph_num"] == "1 1 1 1"
    assert sum(float(d) for d in row["note_dur"].split()) == pytest.approx(1)
    assert row["note_slur"].split().count("1") == 3
    assert notes[0].text == "rest"
    ds = ds_segment(row, [0, 440, 440])
    imported = formats.loads(json.dumps([ds]), "ds")
    assert row_from_label("clip", imported)["ph_num"] == row["ph_num"]
    assert imported.tiers["notes"][1].text == "C4"


def test_ds_next_to_wavs_is_rescued_into_dataset_and_keeps_f0(tmp_path):
    source = tmp_path / "source"
    for i in range(2):
        audio = make_wav(source / "wavs" / f"clip{i}.wav", rate=16000 + i)
        audio.with_suffix(".lab").write_text("0 10000000 a\n")
        audio.with_suffix(".ds").write_text(json.dumps([{"ph_seq": "a", "ph_dur": "1", "ph_num": "1", "note_seq": "C4", "note_dur": "1", "note_slur": "0", "f0_seq": "440 440", "f0_timestep": "0.5"}]))
    result = build_dataset([SourceSpec(folder=str(source), speaker="s", language="xx")], tmp_path / "prepared")
    assert result.files == 2 and result.speakers[0]["note_files"] == 2
    assert json.loads((tmp_path / "prepared/s/ds/clip0.ds").read_text())[0]["f0_seq"] == "440 440"
    assert "C4" in (tmp_path / "prepared/s/transcriptions.csv").read_text()
    assert not any(p.get("code") == "ds_mismatch" for p in result.problems)


def test_ph_num_without_words_survives_csv_and_invalid_groups_are_rejected():
    row = {"ph_seq": "k a r i", "ph_dur": ".1 .2 .1 .2", "ph_num": "2 2"}
    assert row_from_label("x", label_from_row(row))["ph_num"] == "2 2"
    for invalid in ("1", "0 4", "-1 5"):
        with pytest.raises(ValueError):
            label_from_row({**row, "ph_num": invalid})


def test_preparation_builds_csv_ds_copy_and_does_not_modify_input(tmp_path):
    source = tmp_path / "source"
    original = make_wav(source / "a.wav")
    original.with_suffix(".lab").write_text("0 10000000 a\n")
    original_bytes = original.read_bytes()
    class Job:
        def check_cancelled(self): pass
        def progress(self, *args, **kwargs): pass
    async def call(tk, job, engine, method, params):
        assert engine == "audio"
        for part in params["slices"]:
            from pathlib import Path
            target = Path(part["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original_bytes)
        return {"files": [s["path"] for s in params["slices"]]}
    req = RecordingPreparationRequest(folder=str(source), out_dir=str(tmp_path / "out"), f0_method="none")
    result = asyncio.run(prepare_recordings(None, Job(), req, call))
    assert result["count"] == 1 and original.read_bytes() == original_bytes
    assert len(list((tmp_path / "out/ds").glob("*.ds"))) == 1
    assert (tmp_path / "out/transcriptions.csv").exists()
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(prepare_recordings(None, Job(), req.model_copy(update={"out_dir": str(source / "bad")}), call))
