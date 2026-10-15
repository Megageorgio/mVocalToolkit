"""Tiny local speech-package contracts, without pretrained networks or downloads."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
sf=pytest.importorskip("soundfile")
pytest.importorskip("soxr")
safetensors=pytest.importorskip("safetensors.torch")
yaml=pytest.importorskip("yaml")


def load(name,monkeypatch):
    runtime=Path(__file__).parents[1]/"src/mvocaltoolkit/engines/_runtime"
    monkeypatch.syspath_prepend(str(runtime))
    spec=importlib.util.spec_from_file_location(name,runtime/(name+".py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_qwen_export_preserves_distinct_speaker_embeddings_and_reserved_ids(tmp_path,monkeypatch):
    export=load("mvt_train_export",monkeypatch)
    run,base,source,out=[tmp_path/name for name in ("run","base","source","export")]
    for folder in (run,base,source):folder.mkdir()
    config=dict(talker_config=dict(vocab_size=3075,codec_bos_id=3001,spk_id={},spk_is_dialect={}))
    (base/"config.json").write_text(json.dumps(config))
    (run/"config.yaml").write_text(yaml.safe_dump(dict(base_model=str(base),speaker_name="voice")))
    (run/"input-manifest.json").write_text(json.dumps(dict(records=[dict(license="CC0")])))
    weights={"talker.model.codec_embedding.weight":torch.zeros(3075,4)}
    torch.save(dict(step=1,model=weights,metadata=dict(speaker_embeddings={"first":torch.ones(4),"second":torch.full((4,),2.)})),run/"model_ckpt_steps_1.ckpt")
    export.export(run,source,"qwen_tts_voice",out,1)
    result=json.loads((out/"config.json").read_text()); ids=result["talker_config"]["spk_id"]
    assert ids=={"first":3000,"second":3002}
    written=safetensors.load_file(str(out/"model.safetensors"))["talker.model.codec_embedding.weight"]
    assert torch.all(written[3000]==1) and torch.all(written[3002]==2) and torch.all(written[3001]==0)
    original=torch.load(run/"model_ckpt_steps_1.ckpt",weights_only=True)["model"]["talker.model.codec_embedding.weight"]
    assert torch.all(original==0)


@pytest.mark.parametrize("kind",["qwen_tts_voice","voxcpm_voice"])
def test_speech_preparation_uses_only_same_speaker_training_references(tmp_path,monkeypatch,kind):
    import numpy as np
    data=load("mvt_train_data",monkeypatch)
    base,tokenizer,source=[tmp_path/name for name in ("base","tokenizer","source")]
    for p in (base,tokenizer,source):p.mkdir()
    run=tmp_path/"run";run.mkdir()
    records=[]
    import hashlib
    for speaker in ("first","second"):
        for val in (False,True):
            path=tmp_path/f"{speaker}-{val}.wav"
            sf.write(path,np.sin(np.arange(16000)/(20 if speaker=="first" else 30))*.1,16000)
            records.append(dict(audio=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),speaker=speaker,language="ru",validation=val,text=speaker+str(val)))
    settings=dict(seed=42,base_model=str(base),allow_download=False,tokenizer=str(tokenizer))
    (run/"input-manifest.json").write_text(json.dumps(dict(settings=settings,records=records)))
    fake=SimpleNamespace(from_pretrained=lambda *a,**k:SimpleNamespace(encode=lambda p:SimpleNamespace(audio_codes=[torch.zeros((3,16),dtype=torch.int64)])))
    monkeypatch.setitem(sys.modules,"qwen_tts",SimpleNamespace(Qwen3TTSTokenizer=fake))
    data.prepare(run,source,kind,"cpu")
    train=[json.loads(line) for line in (run/"train.jsonl").read_text().splitlines()]
    validation=[json.loads(line) for line in (run/"validation.jsonl").read_text().splitlines()]
    for record in validation:
        expected=next(r for r in train if r["speaker"]==record["speaker"])
        assert record["ref_audio"]==expected["audio"] and record["ref_audio"]!=record["audio"]
    assert validation[0]["ref_audio"]!=validation[1]["ref_audio"]


@pytest.mark.parametrize("kind",["qwen_tts_voice","voxcpm_voice"])
def test_speech_preview_native_call_contract_and_real_wave_output(tmp_path,monkeypatch,kind):
    import numpy as np
    preview=load("mvt_speech_preview",monkeypatch)
    root=tmp_path/"package";root.mkdir()
    called=[]
    if kind=="qwen_tts_voice":
        class Model:
            def get_supported_speakers(self):return ["first","second"]
            def generate_custom_voice(self,**kwargs):called.append(kwargs);return [np.zeros(240)],24000
        monkeypatch.setitem(sys.modules,"qwen_tts",SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=lambda *a,**k:Model())))
    else:
        reference=root/"ref.wav";sf.write(reference,np.zeros(240),24000)
        (root/"speaker-references.json").write_text(json.dumps(dict(second=dict(audio="ref.wav",text="reference"))))
        class Model:
            tts_model=SimpleNamespace(sample_rate=24000)
            def __init__(self,*a,**k):pass
            def generate(self,**kwargs):called.append(kwargs);return np.zeros(240)
        monkeypatch.setitem(sys.modules,"voxcpm",SimpleNamespace(VoxCPM=Model))
    result=preview.render(kind,str(root),str(tmp_path/"render"),"Test text",speaker="second",device="cpu")
    audio,rate=sf.read(result["files"][0])
    assert rate==24000 and len(audio)==240 and called[0]["text"]=="Test text"
    if kind=="qwen_tts_voice":assert called[0]["speaker"]=="second"
    else:assert called[0]["reference_wav_path"]==str(root/"ref.wav")


def test_rvc_heldout_comparison_is_bounded_and_preserves_training_rng_and_mode(tmp_path):
    path=Path(__file__).parents[1]/"src/mvocaltoolkit/engines/rvc/heldout.py"
    spec=importlib.util.spec_from_file_location("heldout_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    wave=torch.sin(torch.arange(4096).float()/20)[None,None]
    class Model(torch.nn.Module):
        def infer(self,*args):return (wave.clone(),)
    model=Model().train()
    data=(torch.zeros(1,10,768),torch.tensor([10]),torch.ones(1,10,dtype=torch.int64),torch.ones(1,10),None,None,wave,torch.tensor([4096]),torch.tensor([0]))
    before=torch.get_rng_state().clone()
    metrics,files=module.evaluate(model,[data]*10,tmp_path,"cpu",40000,42)
    assert metrics["chunks"]==8 and metrics["heldout_log_spectral_l1"]==0
    assert model.training and torch.equal(before,torch.get_rng_state())
    assert len(files)==2 and all(Path(p).is_file() for p in files)
