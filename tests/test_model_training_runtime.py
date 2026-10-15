"""Optional native Torch checks; the base server test environment stays torch-free."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
sys.path.insert(0,str(Path(__file__).parents[1]/"src/mvocaltoolkit/engines/_runtime"))
from mvt_train_common import Stopped
from mvt_torch_loop import fit
from mvt_qwen_train import codec_loss


def test_accumulation_stop_and_optimizer_resume_match_uninterrupted_training(tmp_path):
    from torch.utils.data import DataLoader,TensorDataset
    cfg=dict(steps=3,epochs=3,accumulate=2,save_interval=1,patience=0,seed=42,resolved_device="cpu")
    loader=DataLoader(TensorDataset(torch.tensor([[.1],[.2],[.3]])),batch_size=1)
    def execute(folder,stop=False,resume=False):
        folder.mkdir(exist_ok=True)
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(.5)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        if stop:(folder/"STOP").touch()
        def loss(batch):return model(batch[0]).square().mean(),{}
        try:fit(folder,cfg,model,optimizer,None,loader,loss,lambda:(1.0,{}),resume=resume)
        except Stopped:assert stop
        return model
    continuous=execute(tmp_path/"continuous")
    interrupted=tmp_path/"resumed"
    execute(interrupted,stop=True)
    saved=torch.load(interrupted/"model_ckpt_steps_1.ckpt",weights_only=False)
    assert saved["batch"]==2 and saved["epoch"]==0
    (interrupted/"STOP").unlink()
    resumed=execute(interrupted,resume=True)
    torch.testing.assert_close(resumed.weight,continuous.weight,rtol=0,atol=0)
    state=torch.load(interrupted/"model_ckpt_steps_3.ckpt",weights_only=False)
    assert state["optimizer"]["state"][0]["step"]==3
    events=[json.loads(line) for line in (interrupted/"events.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events if event["event"]=="start"]==[0,1]


def test_qwen_projects_text_and_predicts_codec_before_seeing_target():
    class Talker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=torch.nn.Module()
            self.model.text_embedding=torch.nn.Embedding(8,6)
            self.model.codec_embedding=torch.nn.Embedding(8,4)
            self.text_projection=torch.nn.Linear(6,4)
            self.head=torch.nn.Linear(4,8)
            self.code_predictor=SimpleNamespace(get_input_embeddings=lambda:[self.model.codec_embedding]*15)
        def forward(self,inputs_embeds,**kwargs):
            hidden=inputs_embeds.cumsum(1)
            return SimpleNamespace(logits=self.head(hidden),hidden_states=((hidden,),))
        def forward_sub_talker_finetune(self,targets,hidden):
            self.first_target_hidden=hidden[0].detach().clone()
            return None,hidden.square().mean()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.talker=Talker();self.device="cpu";self.dtype=torch.float32
            self.speaker_encoder=torch.nn.Linear(3,4)
    model=Model()
    batch=dict(input_ids=torch.zeros(1,10,2,dtype=torch.long),ref_mels=torch.ones(1,3),
        text_embedding_mask=torch.ones(1,10,1,dtype=torch.bool),codec_embedding_mask=torch.ones(1,10,1,dtype=torch.bool),
        codec_ids=torch.zeros(1,10,16,dtype=torch.long),codec_mask=torch.tensor([[False]*8+[True,False]]),
        attention_mask=torch.ones(1,10,dtype=torch.long),codec_0_labels=torch.tensor([[-100]*8+[1,2]]))
    loss,_=codec_loss(model,batch);loss.backward()
    assert model.talker.text_projection.weight.grad.abs().sum()>0
    first=model.talker.first_target_hidden.clone()
    batch["codec_ids"][0,8,:]=7
    codec_loss(model,batch)
    torch.testing.assert_close(first,model.talker.first_target_hidden,rtol=0,atol=0)


def test_sofa_finetuning_matches_phoneme_names_when_ids_change():
    from mvt_sofa_base import transfer
    def model(vocab):return SimpleNamespace(backbone=torch.nn.Linear(2,2),head=torch.nn.Linear(2,4),vocab=vocab)
    base=model({"SP":0,"a":1,0:"SP",1:"a","<vocab_size>":2});target=model({"a":0,"SP":1,0:"a",1:"SP","<vocab_size>":2})
    with torch.no_grad():
        base.head.weight.copy_(torch.arange(8.).reshape(4,2));base.head.bias.copy_(torch.arange(4.))
    transfer(target,base)
    torch.testing.assert_close(target.head.weight,base.head.weight[[1,0,2,3]])
    torch.testing.assert_close(target.head.bias,base.head.bias[[1,0,2,3]])


@pytest.mark.parametrize("kind,lora",[("qwen_tts_voice",False),("voxcpm_voice",False),("voxcpm_voice",True)])
def test_speech_export_preserves_required_resources_and_selected_weights(tmp_path,kind,lora):
    safe=pytest.importorskip("safetensors.torch")
    import yaml
    from mvt_train_export import export
    run=tmp_path/"run";run.mkdir()
    base=tmp_path/"base";base.mkdir()
    (base/"config.json").write_text(json.dumps({"talker_config":{},"architecture":"voxcpm2"}))
    (base/"tokenizer.json").write_text("{}")
    (base/"audiovae.safetensors").write_bytes(b"audio decoder")
    (base/"model-00001-of-00002.safetensors").write_bytes(b"stale base weights")
    (base/"model.safetensors.index.json").write_text("{}")
    tokenizer=base/"speech_tokenizer";tokenizer.mkdir()
    (tokenizer/"model.safetensors").write_bytes(b"speech tokenizer weights")
    cfg=dict(base_model=str(base),base_model_origin="openbmb/VoxCPM2",lora=lora,lora_rank=2,speaker_name="voice",tokenizer=str(tokenizer))
    (run/"config.yaml").write_text(yaml.safe_dump(cfg))
    (run/"input-manifest.json").write_text(json.dumps({"records":[{"license":"CC0"}]}))
    weights={"talker.model.codec_embedding.weight":torch.zeros(3001,4),"speaker_encoder.weight":torch.ones(4,4)} if kind=="qwen_tts_voice" else {"layer.lora_A" if lora else "layer.weight":torch.ones(2,4)}
    torch.save({"model":weights,"metadata":{"speaker_embedding":torch.arange(4.)}},run/"model_ckpt_steps_1.ckpt")
    out=tmp_path/"export"
    export(run,tmp_path/"source",kind,out,1)
    assert (out/"training-provenance.json").is_file()
    if kind=="qwen_tts_voice":
        state=safe.load_file(str(out/"model.safetensors"))
        assert "speaker_encoder.weight" not in state
        torch.testing.assert_close(state["talker.model.codec_embedding.weight"][3000],torch.arange(4.))
        assert (out/"speech_tokenizer/model.safetensors").read_bytes()==b"speech tokenizer weights"
        assert not (out/"model-00001-of-00002.safetensors").exists()
        assert not (out/"model.safetensors.index.json").exists()
        assert json.loads((out/"config.json").read_text())["talker_config"]["spk_id"]=={"voice":3000}
    elif lora:
        assert json.loads((out/"lora_config.json").read_text())["base_model"]=="openbmb/VoxCPM2"
        assert set(safe.load_file(str(out/"lora_weights.safetensors")))=={"layer.lora_A"}
    else:
        assert (out/"audiovae.safetensors").read_bytes()==b"audio decoder"
        assert (out/"tokenizer.json").is_file()
        assert set(safe.load_file(str(out/"model.safetensors")))=={"layer.weight"}
