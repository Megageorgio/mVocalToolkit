"""Run pinned Lightning recipes with shared progress, stop and checkpoint handling."""
import json
import math
import os
import runpy
import sys
from pathlib import Path

from mvt_train_common import Stopped, chosen, device, emit, listing, merge, numbers, saved, seed


class AudioBatches:
    """Keep every recording, including the last bucket that the pinned SOFA sampler omits."""
    def __init__(self, lengths, batch_size):
        self.batch_size = batch_size
        self.indices = sorted(range(len(lengths)), key=lambda i:lengths[i])
    def __len__(self): return (len(self.indices) + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        import random
        batches = [self.indices[i:i+self.batch_size] for i in range(0,len(self.indices),self.batch_size)]
        random.shuffle(batches)
        return iter(batches)


def train(run, source, kind, requested, resume):
    import lightning.pytorch as pl
    import torch
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    resolved = device(requested); seed(cfg["seed"])
    os.chdir(source)
    # Native recipes sometimes provide prefetch/persistent-worker arguments even with zero workers.
    original_loader = torch.utils.data.DataLoader.__init__
    def loader(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) == 0:
            kwargs.update(prefetch_factor=None, persistent_workers=False)
        original_loader(self, *args, **kwargs)
    torch.utils.data.DataLoader.__init__ = loader
    checkpoint = chosen(run) if resume and listing(run) else None
    class Events(pl.Callback):
        last = -1
        stopped = False
        bad = 0
        def save(self, trainer, best=False):
            step = trainer.global_step
            target = run / f"model_ckpt_steps_{step}.ckpt"; temporary = target.with_suffix(".tmp")
            trainer.save_checkpoint(temporary)
            os.replace(temporary, target); saved(run, step, best)
        def on_train_start(self, trainer, module):
            emit(run, "start", step=trainer.global_step, max_steps=cfg["steps"])
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step == self.last or step == 0: return
            self.last = step
            loss=outputs.get("loss") if isinstance(outputs,dict) else outputs
            if loss is not None and not math.isfinite(float(loss.detach() if hasattr(loss,"detach") else loss)):
                raise ValueError("Non-finite training loss; check data and learning rate")
            values = numbers(outputs if isinstance(outputs, dict) else {"loss": outputs})
            values.update(numbers({str(k):v for k,v in trainer.callback_metrics.items() if "train" in str(k)}))
            if "loss" in values: values["total_loss"] = values["loss"]
            opt = trainer.optimizers[0]
            emit(run, "step", step=step, max_steps=cfg["steps"], epoch=trainer.current_epoch + 1, losses=values,
                 lr=opt.param_groups[0]["lr"], gpu_mb=torch.cuda.memory_allocated()/2**20 if resolved.startswith("cuda") else 0)
            if (run / "STOP").exists():
                self.save(trainer); self.stopped = True; trainer.should_stop = True
        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking: return
            for key,value in trainer.callback_metrics.items():
                if "loss" in str(key).lower() and getattr(value,"numel",lambda:1)()==1 and not math.isfinite(float(value)):
                    raise ValueError("Non-finite validation loss")
            values = numbers(trainer.callback_metrics)
            candidates = [(k,v) for k,v in values.items() if ("val" in k.lower() or "valid" in k.lower()) and "loss" in k.lower()]
            if not candidates: candidates = [(k,v) for k,v in values.items() if k in {"total_loss","loss"}]
            if not candidates: return
            score = min(candidates, key=lambda kv: ("total" not in kv[0], len(kv[0])))[1]
            (run / "validation.json").write_text(json.dumps({"step": trainer.global_step, "score": score, "metrics": values}), encoding="utf-8")
            emit(run, "validation", step=trainer.global_step, val_loss=score, metrics=values)
            best_file = run / "best-score.json"
            previous = json.loads(best_file.read_text())["score"] if best_file.exists() else float("inf")
            better = score < previous
            self.bad = 0 if better else self.bad + 1
            if better: best_file.write_text(json.dumps({"score":score}), encoding="utf-8")
            self.save(trainer, better)
            if cfg["patience"] and self.bad >= cfg["patience"]: trainer.should_stop = True
        def state_dict(self): return {"bad":self.bad}
        def load_state_dict(self,state_dict): self.bad = state_dict.get("bad",0)
        def on_train_end(self, trainer, module):
            self.save(trainer)
    events = Events()
    original_init, original_fit = pl.Trainer.__init__, pl.Trainer.fit
    def init(self, *args, **kwargs):
        callbacks = [c for c in kwargs.get("callbacks", []) if not isinstance(c, pl.callbacks.ModelCheckpoint)]
        kwargs.update(callbacks=[*callbacks, events], enable_checkpointing=False, devices=1 if resolved == "cpu" else [int(resolved.split(":")[1])],
                      accelerator="cpu" if resolved == "cpu" else "gpu", precision="32-true", strategy="auto",
                      max_steps=cfg["steps"], max_epochs=cfg["epochs"], num_sanity_val_steps=0,
                      default_root_dir=str(run), accumulate_grad_batches=cfg["accumulate"],
                      logger=pl.loggers.TensorBoardLogger(str(run / "tensorboard"), name=""))
        original_init(self, *args, **kwargs)
    def start(self, *args, **kwargs):
        if checkpoint: kwargs["ckpt_path"] = str(checkpoint)
        return original_fit(self, *args, **kwargs)
    pl.Trainer.__init__, pl.Trainer.fit = init, start
    if kind in {"tifa_aligner", "game_notes"}:
        sys.argv = ["train.py", "--config", str(run / "native.yaml"), "--work-dir", str(run), "--exp-name", "native"]
        if checkpoint: sys.argv += ["--resume-from", str(checkpoint)]
    else:
        import dataset
        dataset.WeightedBinningAudioBatchSampler = lambda types,lengths,*args,**kwargs:AudioBatches(lengths,cfg["batch_size"])
        from modules.task.forced_alignment import LitForcedAlignmentTask
        from mvt_sofa_base import transfer
        LitForcedAlignmentTask.load_pretrained=transfer
        native = yaml.safe_load((source / "configs/train_config.yaml").read_text())
        if cfg.get("sofa_model_config"):native["model"]=cfg["sofa_model_config"]
        merge(native, cfg["overrides"])
        native.update(model_name="native", dataloader_workers=0, batch_max_length=cfg["batch_size"] * 10,
                      val_check_interval=cfg["save_interval"], random_seed=cfg["seed"], data_augmentation_size=0)
        native["optimizer_config"]["total_steps"] = cfg["steps"]
        native["optimizer_config"]["lr"] = {"backbone":cfg["learning_rate"], "head":cfg["learning_rate"]}
        path = run / "native.yaml"; path.write_text(yaml.safe_dump(native), encoding="utf-8")
        os.chdir(run)
        sys.argv = ["train.py", "--config_path", str(path), "--data_folder", str(run / "raw")]
        if cfg["base_model"] and not checkpoint:
            base = Path(cfg.get("sofa_base_checkpoint",cfg["base_model"]))
            if base.is_dir(): base = next(iter(base.glob("*.ckpt")), base / "model.ckpt")
            sys.argv += ["--pretrained_model_path", str(base)]
    try:
        runpy.run_path(str(source / "train.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None): raise
    if events.last < 1 and checkpoint is None:
        raise ValueError("The native trainer completed without any optimizer updates; check the prepared data")
    if events.stopped: raise Stopped()
