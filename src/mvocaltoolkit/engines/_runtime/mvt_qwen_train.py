"""Single-speaker Qwen3-TTS fine-tuning using the official dataset and model."""
import json
import sys

from mvt_train_common import device, seed
from mvt_torch_loop import fit


def codec_loss(model, batch):
    """Shift next-token labels once; supervise code prediction at the matching hidden position."""
    import torch
    import torch.nn.functional as F
    b = {k:v.to(model.device) for k,v in batch.items()}
    model.speaker_encoder.eval()
    with torch.no_grad(): speaker = model.speaker_encoder(b["ref_mels"].to(model.dtype))
    text = model.talker.text_projection(model.talker.model.text_embedding(b["input_ids"][:,:,0])) * b["text_embedding_mask"]
    codec = model.talker.model.codec_embedding(b["input_ids"][:,:,1]) * b["codec_embedding_mask"]
    codec[:,6] = speaker
    embeds = text + codec
    # Residual codec embeddings at an input position belong only to that position's already-known token.
    for level in range(1,16):
        embeds = embeds + model.talker.code_predictor.get_input_embeddings()[level-1](b["codec_ids"][:,:,level]) * b["codec_mask"].unsqueeze(-1)
    output = model.talker(inputs_embeds=embeds[:,:-1], attention_mask=b["attention_mask"][:,:-1], output_hidden_states=True)
    main = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), b["codec_0_labels"][:,1:].reshape(-1), ignore_index=-100)
    hidden = output.hidden_states[0][-1]
    # Code sequence at t is conditioned on hidden(t-1), never hidden(t) that has seen that sequence.
    target_mask = b["codec_mask"][:,1:]
    targets = b["codec_ids"][:,1:][target_mask]
    _, residual = model.talker.forward_sub_talker_finetune(targets, hidden[target_mask])
    return main + .3 * residual, {"codec":float(main.detach()), "residual":float(residual.detach())}


def train(run, source, kind, requested, resume):
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoConfig
    sys.path.insert(0,str(source / "finetuning"))
    from dataset import TTSDataset
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg["resolved_device"] = device(requested); seed(cfg["seed"])
    wrapper = Qwen3TTSModel.from_pretrained(cfg["base_model"], device_map=cfg["resolved_device"], torch_dtype=torch.float32, attn_implementation="sdpa")
    model = wrapper.model
    if model.speaker_encoder is None:
        raise ValueError("Choose a Qwen3-TTS 12Hz Base model with a speaker encoder")
    if model.config.talker_config.num_code_groups != 16:
        raise ValueError("This recipe requires a Qwen3-TTS 12Hz model with 16 codec groups")
    model.speaker_encoder.requires_grad_(False).eval()
    config = AutoConfig.from_pretrained(cfg["base_model"])
    def dataset(name):
        values = [json.loads(line) for line in (run / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        return TTSDataset(values, wrapper.processor, config)
    train_ds, validation_ds = dataset("train.jsonl"), dataset("validation.jsonl")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], collate_fn=train_ds.collate_fn, num_workers=0)
    validation_loader = DataLoader(validation_ds, batch_size=1, collate_fn=validation_ds.collate_fn, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=.01)
    reference = train_ds[0]["ref_mel"].to(cfg["resolved_device"])
    with torch.no_grad(): speaker = model.speaker_encoder(reference.to(model.dtype)).mean(dim=0).cpu()
    def loss(batch): return codec_loss(model,batch)
    def validate():
        losses = [float(codec_loss(model,batch)[0]) for batch in validation_loader]
        return sum(losses)/len(losses), {"metric":"Held-out codec cross entropy", "recordings":len(losses)}
    fit(run,cfg,model,optimizer,None,loader,loss,validate,
        metadata={"kind":kind,"speaker_embedding":speaker,"speaker_name":cfg["speaker_name"],"base_model":cfg["base_model"]}, resume=resume)
