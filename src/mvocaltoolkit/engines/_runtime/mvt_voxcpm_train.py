"""Adapter for the pinned official VoxCPM full/LoRA training recipe."""
import json
import math
import os

from mvt_train_common import Stopped, atomic_torch, chosen, device, emit, listing, numbers, restore_rng, rng_state, saved, seed


def adapt(source):
    for interval in ("valid_interval","save_interval"):
        anchor=f"step % {interval} == 0 or step == num_iters - 1"
        if source.count(anchor) != 1: raise ValueError("Unsupported VoxCPM interval contract")
        source=source.replace(anchor,f"(step + 1) % {interval} == 0 or step == num_iters - 1")
    anchor = "            scheduler.step()"
    if source.count(anchor) != 1: raise ValueError("Unsupported VoxCPM trainer contract")
    source = source.replace(anchor, anchor + "\n            _mvt_step(locals())")
    anchor = "                    accelerator.backward(total_loss)"
    if source.count(anchor) != 1: raise ValueError("Unsupported VoxCPM backward contract")
    source = source.replace(anchor, "                    if not torch.isfinite(total_loss): raise ValueError('Non-finite training loss')\n" + anchor)
    anchor = "            accelerator.step(optimizer)"
    if source.count(anchor) != 1: raise ValueError("Unsupported VoxCPM optimizer contract")
    source = source.replace(anchor,"            if not torch.isfinite(grad_norm): raise ValueError('Non-finite gradients')\n"+anchor)
    anchor = "    with tracker.live():"
    if source.count(anchor) != 1: raise ValueError("Unsupported VoxCPM loop contract")
    source = source.replace(anchor, "    num_iters = _mvt_limit(len(train_loader), grad_accum_steps, num_iters)\n" + anchor)
    anchor = "    train_iter = iter(train_loader)"
    if source.count(anchor) != 2: raise ValueError("Unsupported VoxCPM data iterator contract")
    # Only the outer iterator is initialized here; the indented second occurrence is its epoch reset.
    source = source.replace("\n" + anchor + "\n", "\n    data_epoch = start_step * grad_accum_steps // max(1, len(train_loader))\n"
        "    if hasattr(getattr(train_loader, 'sampler', None), 'set_epoch'): train_loader.sampler.set_epoch(data_epoch)\n"
        + anchor + "\n    for _ in range(start_step * grad_accum_steps % max(1, len(train_loader))): next(train_iter)\n", 1)
    return source


def train(run, source, kind, requested, resume):
    import torch
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    resolved = device(requested); seed(cfg["seed"])
    if resolved == "cpu": torch.cuda.is_available = lambda:False
    else: os.environ["LOCAL_RANK"] = resolved.split(":")[1]
    state = torch.load(chosen(run), map_location="cpu", weights_only=False) if resume and listing(run) else None
    current = {"step":state["step"] if state else 0, "best":state.get("best",math.inf) if state else math.inf,
               "bad":state.get("bad",0) if state else 0}
    class Finished(Exception): pass
    def checkpoint(model, optimizer, scheduler, *args, best=False, **kwargs):
        unwrapped = model.module if hasattr(model,"module") else model
        weights = {k:v for k,v in unwrapped.state_dict().items() if ("lora_" in k if cfg["lora"] else not k.startswith("audio_vae."))}
        step = current["step"]
        payload = {"format":"voxcpm-v1","model":weights,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                   "scaler":current["accelerator"].scaler.state_dict() if hasattr(current["accelerator"].scaler,"state_dict") else None,
                   "step":step,"rng":rng_state(),"best":current["best"],"bad":current["bad"],
                   "metadata":{"base_model":cfg["base_model"],"lora":cfg["lora"],"lora_rank":cfg["lora_rank"]}}
        atomic_torch(payload,run/f"model_ckpt_steps_{step}.ckpt");saved(run,step,best)
    def load(model, optimizer, scheduler, *args, **kwargs):
        if not state:return 0
        unwrapped=model.module if hasattr(model,"module") else model
        expected={k for k in unwrapped.state_dict() if ("lora_" in k if cfg["lora"] else not k.startswith("audio_vae."))}
        if expected != set(state["model"]): raise ValueError("The VoxCPM checkpoint does not match the selected base and LoRA settings")
        unwrapped.load_state_dict(state["model"],strict=False)
        optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);restore_rng(state.get("rng"))
        if state.get("scaler") is not None:current["accelerator"].scaler.load_state_dict(state["scaler"])
        return state["step"]
    def step(values):
        current["step"]=values["step"]+1
        current["training"]=(values["model"],values["optimizer"],values["scheduler"])
        losses=numbers(values["loss_dict"])
        emit(run,"step",step=current["step"],max_steps=cfg["steps"],losses={"total_loss":sum(losses.values()),**losses},
             lr=values["optimizer"].param_groups[0]["lr"],gpu_mb=torch.cuda.memory_allocated()/2**20 if resolved.startswith("cuda") else 0)
        if (run/"STOP").exists():
            checkpoint(values["model"],values["optimizer"],values["scheduler"]);raise Stopped()
    script=source/"scripts/train_voxcpm_finetune.py"
    def limit(batches,accumulate,steps):
        if batches == 0: raise ValueError("No training batches; reduce batch size or add recordings")
        return min(steps,max(1,math.ceil(cfg["epochs"]*batches/accumulate)))
    scope={"__name__":"mvt_voxcpm_native","__file__":str(script),"_mvt_step":step,
           "_mvt_limit":limit}
    exec(compile(adapt(script.read_text(encoding="utf-8")),str(script),"exec"),scope)
    scope.update(save_checkpoint=checkpoint,load_checkpoint=load)
    original_accelerator=scope["Accelerator"]
    def accelerator(amp=True):
        current["accelerator"]=original_accelerator(amp=amp,seed=cfg["seed"])
        return current["accelerator"]
    scope["Accelerator"]=accelerator
    original_metrics=scope["TrainingTracker"].log_metrics
    def metrics(self,values,split):
        original_metrics(self,values,split)
        if split != "val": return
        score=float(values["loss/total"])
        if not math.isfinite(score): raise ValueError("Non-finite validation loss")
        better=score < current["best"]
        current["bad"]=0 if better else current["bad"]+1
        if better: current["best"]=score
        (run/"validation.json").write_text(json.dumps({"step":current["step"],"score":score,"metrics":numbers(values),"max_batches":10}),encoding="utf-8")
        emit(run,"validation",step=current["step"],val_loss=score,metrics=numbers(values))
        checkpoint(*current["training"],best=better)
        if cfg["patience"] and current["bad"] >= cfg["patience"]: raise Finished()
    scope["TrainingTracker"].log_metrics=metrics
    original_audio=scope["SummaryWriter"].add_audio
    def audio(self,tag,samples,global_step=None,sample_rate=44100,**kwargs):
        original_audio(self,tag,samples,global_step=global_step,sample_rate=sample_rate,**kwargs)
        if "generated_audio" not in tag: return
        import re
        import numpy as np
        import soundfile as sf
        folder=run/"samples";folder.mkdir(exist_ok=True)
        name=re.sub(r"[^A-Za-z0-9_-]","_",tag)
        values=samples.detach().cpu().numpy() if hasattr(samples,"detach") else np.asarray(samples)
        path=folder/f"{current['step']}-{name}.wav"
        sf.write(path,values.squeeze(),sample_rate)
        emit(run,"samples",step=current["step"],files=[str(path)])
    scope["SummaryWriter"].add_audio=audio
    emit(run,"start",step=current["step"],max_steps=cfg["steps"])
    try:
        scope["train"](pretrained_path=cfg["base_model"],train_manifest=str(run/"train.jsonl"),val_manifest=str(run/"validation.jsonl"),
        sample_rate=16000,out_sample_rate=48000,batch_size=cfg["batch_size"],grad_accum_steps=cfg["accumulate"],num_workers=0,
        preprocessing_num_workers=1,num_iters=cfg["steps"],log_interval=1,valid_interval=cfg["save_interval"],save_interval=cfg["save_interval"],
        learning_rate=cfg["learning_rate"],warmup_steps=min(100,cfg["steps"]//10),max_steps=cfg["steps"],max_batch_tokens=8192,
        save_path=str(run/"native"),tensorboard=str(run/"tensorboard"),
        lora={"enable_lm":True,"enable_dit":True,"enable_proj":False,"r":cfg["lora_rank"],"alpha":cfg["lora_rank"],"dropout":0.0} if cfg["lora"] else None)
    except Finished: pass
