import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from mvocaltoolkit.training.recipes import FAMILIES, check, scan, settings, snapshot


def sources(tmp_path):
    from mvocaltoolkit import formats
    from mvocaltoolkit.labels import Label
    folder=tmp_path/"audio";folder.mkdir()
    for i in range(4):
        path=folder/f"sample{i}.wav"
        with wave.open(str(path),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes([i+1,0])*16000)
        path.with_suffix(".lab").write_text("0 5000000 a\n5000000 10000000 b\n",encoding="utf-8")
        path.with_suffix(".txt").write_text(f"A sample number {i}",encoding="utf-8")
    return [{"folder":str(folder),"speaker":"singer","language":"ru","license":"CC0"}]


def teacher(tmp_path):
    path=tmp_path/"teacher";path.mkdir()
    for name in ("config.yaml","phonemes.txt","langs.txt","best.pt"):(path/name).write_text("fixture")
    return str(path)


@pytest.mark.parametrize("kind",list(FAMILIES))
def test_every_recipe_checks_and_snapshots_without_torch_or_install(tmp_path,kind):
    config={"sources":sources(tmp_path)}
    if kind=="mcfl_student":config["teacher"]=teacher(tmp_path)
    if kind=="game_notes":
        from mvocaltoolkit.formats.ds_csv import dumps_rows
        rows=[{"name":f"sample{i}","ph_seq":"a b","ph_dur":"0.5 0.5","note_seq":"C4 D4","note_dur":"0.5 0.5","note_slur":"0 0"} for i in range(4)]
        Path(config["sources"][0]["folder"],"transcriptions.csv").write_text(dumps_rows(rows),encoding="utf-8")
    result=check(kind,config)
    assert result["files"]==4 and result["train_files"]==3 and result["validation_files"]==1
    assert not any(f["severity"]=="error" for f in result["findings"])
    first=snapshot(kind,config,tmp_path/"snapshots");second=snapshot(kind,config,tmp_path/"snapshots")
    assert first==second and "records" not in first["data_report"]
    data=json.loads(Path(first["prepared_file"]).read_text(encoding="utf-8"))
    train={r["sha256"] for r in data["records"] if not r["validation"]}
    validation={r["sha256"] for r in data["records"] if r["validation"]}
    assert not train&validation and len(train|validation)==4


@pytest.mark.parametrize("change",[{"learning_rate":float("nan")},{"steps":0},{"save_interval":101},
    {"validation_fraction":.9},{"prepare_only":"false"},{"overrides":[]},{"batch_size":True},{"seed":-1}])
def test_bad_recipe_knobs_fail_before_install(change):
    with pytest.raises(ValueError):settings("tifa_aligner",change)


def test_recipe_duplicate_and_language_split_guard(tmp_path):
    config={"sources":sources(tmp_path)}
    folder=Path(config["sources"][0]["folder"])
    (folder/"sample3.wav").write_bytes((folder/"sample0.wav").read_bytes())
    result=scan("sofa_aligner",config)
    assert result["files"]==3 and any(f["code"]=="duplicate" for f in result["findings"])
    for name in ("sample1.wav","sample2.wav"):(folder/name).unlink()
    with pytest.raises(ValueError,match="recording findings"):snapshot("sofa_aligner",config,tmp_path/"snapshots")


def test_mcfl_requires_a_real_teacher_and_offline_speech_needs_local_weights():
    with pytest.raises(ValueError,match="teacher folder"):settings("mcfl_student",{})
    with pytest.raises(ValueError,match="downloads are disabled"):settings("qwen_tts_voice",{"allow_download":False})


def test_sofa_batches_preserve_short_corpora_and_final_bucket(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"src/mvocaltoolkit/engines/_runtime"))
    from mvt_lightning import AudioBatches
    batches=AudioBatches([2,7,1,9,3],2)
    assert len(batches)==3
    assert sorted(i for batch in batches for i in batch)==list(range(5))
    assert list(AudioBatches([],1))==[]


def test_bad_game_notes_block_preflight(tmp_path):
    config={"sources":sources(tmp_path)}
    from mvocaltoolkit.formats.ds_csv import dumps_rows
    rows=[{"name":f"sample{i}","ph_seq":"a b","ph_dur":"0.5 0.5","note_seq":"C99 D4","note_dur":"0.5 0.5","note_slur":"0 0"} for i in range(4)]
    Path(config["sources"][0]["folder"],"transcriptions.csv").write_text(dumps_rows(rows),encoding="utf-8")
    assert any(f["code"]=="invalid_recording" for f in check("game_notes",config)["findings"])


def test_speech_overrides_and_unknown_student_size_are_rejected(tmp_path):
    with pytest.raises(ValueError,match="named settings"):
        settings("qwen_tts_voice",{"overrides":{"model":{}}})
    with pytest.raises(ValueError,match="MCFL size"):
        settings("mcfl_student",{"teacher":teacher(tmp_path),"size":"misspelled"})


def test_recipe_api_derives_engine_and_never_installs_during_preflight(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from mvocaltoolkit.toolkit import Toolkit
    from mvocaltoolkit.settings import Home,Settings
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.training.runs import RunInfo
    tk=Toolkit(Home(tmp_path/"home"),Settings(catalog_url=""))
    config={"sources":sources(tmp_path)}
    async def forbidden(*args,**kwargs):raise AssertionError("Preflight tried to install an engine")
    monkeypatch.setattr(tk.engines,"call",forbidden)
    queued=[]
    def start(spec):
        queued.append(spec)
        return SimpleNamespace(info=RunInfo(id="example",spec=spec,dir=str(tmp_path/"run")))
    monkeypatch.setattr(tk.training,"start",start)
    with TestClient(create_app(tk)) as client:
        assert len(client.get("/training/recipes").json())==8
        assert client.post("/training/recipes/tifa_aligner/check",json=config).status_code==200
        response=client.post("/training/runs",json={"kind":"tifa_aligner","device":"cpu","config":config})
        assert response.status_code==200,response.text
        assert queued[0].engine_name()=="tifa_train" and Path(queued[0].config["prepared_file"]).exists()
        assert client.post("/training/runs",json={"kind":"tifa_aligner","engine":"rvc","config":config}).status_code==400
