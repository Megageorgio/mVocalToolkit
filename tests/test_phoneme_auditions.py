import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mvocaltoolkit import formats
from mvocaltoolkit.settings import Home
from mvocaltoolkit.training.test_phrases import PhraseRequest, inventory, build_phrase
from mvocaltoolkit.training.resources import measured_limits, reduce_after_oom
from mvocaltoolkit.training.phoneme_preview import suggest_mappings
from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus
from mvocaltoolkit.training.manager import TrainingManager


def bank(tmp_path):
    root = tmp_path / "bank"; root.mkdir()
    (root / "phonemes.json").write_text(json.dumps(["SP", "AP", "ru/a", "ru/i", "ru/p", "ru/t", "ru/k", "en/ae"]))
    (root / "dsconfig.yaml").write_text("phonemes: phonemes.json\n")
    (root / "dsdict.yaml").write_text(yaml.safe_dump(dict(symbols=[dict(symbol=p, type="vowel" if p in {"SP","AP","ru/a","ru/i"} else "stop") for p in ["SP","AP","ru/a","ru/i","ru/p","ru/t","ru/k"]])))
    return root


@pytest.mark.parametrize("mode", ["cv", "vc", "v", "cvc", "manual"])
def test_phoneme_phrases_are_trained_grouped_timed_and_deterministic(tmp_path, mode):
    root = bank(tmp_path)
    info = inventory(root, "ru")
    assert "en/ae" not in info["phonemes"] and "SP" not in info["vowels"]
    req = PhraseRequest(model_dir=str(root), mode=mode, phonemes="p a | SP t i | AP a", notes="C4 F#4 64", limit=6)
    result = build_phrase(req, tmp_path / "phrase")
    segment = result["segment"]
    assert set(segment["ph_seq"].split()) <= set(info["phonemes"])
    assert sum(map(int,segment["ph_num"].split())) == len(segment["ph_seq"].split())
    assert abs(sum(map(float,segment["ph_dur"].split())) - sum(map(float,segment["note_dur"].split()))) < 1e-6
    assert abs(len(segment["f0_seq"].split()) * .01 - result["seconds"]) < .011
    assert formats.read(Path(result["path"])).phones
    if mode in {"cv", "vc", "cvc"}: assert len(set(result["syllables"])) == 6
    second = build_phrase(req, tmp_path / "again")
    assert second["segment"] == segment
    with pytest.raises(FileExistsError): build_phrase(req, tmp_path / "phrase")
    with pytest.raises(ValueError, match="Untrained"):
        build_phrase(req.model_copy(update={"mode":"manual","phonemes":"unknown"}), tmp_path / "bad")


def test_ipa_mapping_keeps_ambiguities_and_phonetic_distinctions(tmp_path):
    result = suggest_mappings([dict(phone="ru/py",ipa="pʲ"),dict(phone="ru/p",ipa="p"),dict(phone="x",ipa="x")],
        [dict(phone="p_soft",ipa="pʲ"),dict(phone="p1",ipa="p"),dict(phone="p2",ipa="p")])
    assert result["mappings"] == {"ru/py":["p_soft"]}
    assert result["ambiguous"] == {"ru/p":["p1","p2"]} and result["missing"] == ["x"]


def test_legacy_stock_russian_checkpoint_without_profile_can_generate_syllables(tmp_path):
    (tmp_path/"phonemes.json").write_text(json.dumps(["SP","AP","aa","ii","b","d"]))
    info=inventory(tmp_path,"ru")
    assert info["classification_profile"]=="ru-openutau" and info["vowels"]==["aa","ii"]
    result=build_phrase(PhraseRequest(model_dir=str(tmp_path),limit=4),tmp_path.parent/(tmp_path.name+"-phrase"))
    assert len(result["syllables"])==4 and all("aa" in s or "ii" in s for s in result["syllables"])


def test_memory_measurement_and_bounded_oom_recovery_keep_checkpoint_bytes(tmp_path):
    cfg = dict(datasets=[dict(raw_data_dir="data",speaker="s",language="ru")], dictionaries={"ru":"dict"}, preset={"gpu_gb":8},
        resources={"auto_batch":True,"oom_retries":2}, overrides={"max_batch_frames":12000,"max_batch_size":8})
    limits, report = measured_limits("diffsinger_acoustic",cfg,{"gpus":[dict(index=0,memory_free_mb=4096)]},"cuda:0")
    assert 1000 <= limits["max_batch_frames"] < 12000 and report["measured"]
    assert measured_limits("diffsinger_acoustic",cfg,{"gpus":[]},"cpu")[0] == {}
    run = Run(RunInfo(id="r",dir=str(tmp_path),spec=RunSpec(kind="diffsinger_acoustic",config=cfg),status=RunStatus.failed,error_kind="oom"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(dict(max_batch_frames=12000,max_batch_size=8,max_val_batch_frames=24000)))
    (tmp_path / "PREPARED").touch(); checkpoint=tmp_path/"model_ckpt_steps_1.ckpt"; checkpoint.write_bytes(b"untouched")
    first=reduce_after_oom(run); second=reduce_after_oom(run)
    assert first["limits"] == dict(max_batch_frames=6000,max_batch_size=4)
    assert second["limits"] == dict(max_batch_frames=3000,max_batch_size=2)
    assert reduce_after_oom(run) is None and checkpoint.read_bytes() == b"untouched"
    assert yaml.safe_load((tmp_path/"config.yaml").read_text())["max_val_batch_frames"] == 3000


def test_oom_follower_requeues_only_after_failed_process_exits(tmp_path):
    run = Run(RunInfo(id="r",dir=str(tmp_path),spec=RunSpec(kind="diffsinger_acoustic",config={"resources":{"oom_retries":1}}),status=RunStatus.training,pid=123))
    (tmp_path/"config.yaml").write_text("max_batch_frames: 8000\nmax_batch_size: 4\n")
    (tmp_path/"PREPARED").touch()
    (tmp_path/"events.jsonl").write_text(json.dumps(dict(event="end",status="failed",error="out of memory",error_kind="oom"))+"\n")
    run.process=SimpleNamespace(poll=lambda:0)
    manager=TrainingManager(SimpleNamespace(home=Home(tmp_path/"home")))
    asyncio.run(manager._follow_loop(run))
    assert run.info.status == RunStatus.queued and run.info.pid is None and run.info.resource_retries == 1


def test_early_stop_counts_validation_checks_and_preserves_patience(tmp_path):
    path=Path(__file__).parents[1]/"src/mvocaltoolkit/engines/_runtime/mvt_early_stop.py"
    spec=importlib.util.spec_from_file_location("early_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    options=dict(patience=2,min_delta=.1,min_steps=10)
    early=module.EarlyStop(tmp_path,options)
    assert not early.update(5,1) and not early.update(10,1)
    assert not early.update(20,.95) and not early.update(20,.95)
    early=module.EarlyStop(tmp_path,options)
    assert not early.update(25,float("nan")) and early.update(30,.94)
    assert not module.EarlyStop(tmp_path,dict(patience=0)).update(100,10)


def test_phrase_and_speech_api_use_managed_outputs_and_matching_engines(tmp_path,monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from mvocaltoolkit.settings import Settings
    from mvocaltoolkit.toolkit import Toolkit
    from mvocaltoolkit.server.app import create_app
    root=bank(tmp_path)
    tk=Toolkit(Home(tmp_path/"home"),Settings(catalog_url="",auto_install_engines=False,auto_download_models=False))
    calls=[]
    async def call(engine,method,params,**kwargs):
        calls.append((engine,method,params));return dict(files=[],out_dir=params["out_dir"])
    monkeypatch.setattr(tk.engines,"call",call)
    with TestClient(create_app(tk)) as client:
        req=dict(model_dir=str(root),language="ru",mode="manual",phonemes="p a | t i")
        response=client.post("/training/test-phrase/create",json=req)
        assert response.status_code==200 and Path(response.json()["path"]).is_relative_to(tk.home.outputs)
        assert Path(response.json()["path"]).is_file() and not calls
        assert client.post("/training/test-phrase/create",json={**req,"phonemes":"unknown"}).status_code==400
        before = len(tk.jobs.jobs)
        for options in [dict(max_tokens=0), dict(text="   "), dict(steps=True)]:
            invalid = dict(kind="qwen_tts_voice", model_dir=str(root), text="Test")
            invalid.update(options)
            assert client.post("/training/speech/render", json=invalid).status_code == 400
        assert not calls and len(tk.jobs.jobs) == before
        for kind,engine in [("qwen_tts_voice","qwen_tts"),("voxcpm_voice","voxcpm")]:
            response=client.post("/training/speech/render",json=dict(kind=kind,model_dir=str(root),text="Test",out_dir="unmanaged"))
            assert response.status_code==200
            identifier=response.json()["id"]
            for _ in range(100):
                info=client.get(f"/jobs/{identifier}").json()
                if info["status"] in {"done","failed"}:break
                time.sleep(.01)
            assert info["status"]=="done",info
            assert calls[-1][0:2]==(engine,"render_speech")
            assert Path(calls[-1][2]["out_dir"]).is_relative_to(tk.home.outputs)
