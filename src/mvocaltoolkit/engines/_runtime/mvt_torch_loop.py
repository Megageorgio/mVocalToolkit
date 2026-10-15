"""Finite-step training loop with recording-disjoint evaluation and resumable optimizer state."""
import json
import math
from pathlib import Path

from mvt_train_common import Stopped, atomic_torch, chosen, emit, listing, numbers, restore_rng, rng_state, saved


def fit(run, cfg, model, optimizer, scheduler, loader, loss_fn, validate, *, metadata=None, ema=None, resume=False):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(run / "tensorboard")) as writer:
        return _fit(run,cfg,model,optimizer,scheduler,loader,loss_fn,validate,metadata=metadata,ema=ema,resume=resume,writer=writer)


def _fit(run, cfg, model, optimizer, scheduler, loader, loss_fn, validate, *, metadata, ema, resume, writer):
    import torch
    if len(loader) == 0: raise ValueError("No training batches; reduce batch size or add recordings")
    step, start_epoch, offset, best, bad = 0, 0, 0, math.inf, 0
    if resume and listing(run):
        state = torch.load(chosen(run), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"): scheduler.load_state_dict(state["scheduler"])
        if ema and state.get("ema"): ema.load_state_dict(state["ema"], cfg["resolved_device"])
        step, start_epoch, offset, best, bad = state["step"], state["epoch"], state["batch"], state["best"], state["bad"]
        restore_rng(state.get("rng"))
    emit(run, "start", step=step, max_steps=cfg["steps"])
    epoch, batch = start_epoch, offset

    def checkpoint(is_best=False):
        state = {"format": "mvt-torch-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None, "ema": ema.state_dict() if ema else None,
                 "step": step, "epoch": epoch, "batch": batch, "rng": rng_state(), "best": best, "bad": bad,
                 "metadata": metadata or {}}
        atomic_torch(state, run / f"model_ckpt_steps_{step}.ckpt"); saved(run, step, is_best)

    for epoch_index in range(start_epoch, cfg["epochs"]):
        epoch = epoch_index
        if hasattr(loader.sampler, "epoch"): loader.sampler.epoch = epoch
        if hasattr(loader.dataset, "seed"): loader.dataset.seed = cfg["seed"] + epoch
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for index, item in enumerate(loader):
            if epoch_index == start_epoch and index < offset: continue
            if step >= cfg["steps"]: return
            model.train()
            loss, parts = loss_fn(item)
            if not torch.isfinite(loss): raise ValueError("Non-finite training loss; check data and lower learning rate")
            # Flush a shorter final accumulation group using its actual number of micro-batches.
            group_size = min(cfg["accumulate"], len(loader) - index + accumulated)
            (loss / group_size).backward()
            accumulated += 1
            if accumulated < group_size: continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm): raise ValueError("Non-finite gradients; check data and learning rate")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True); accumulated = 0
            if scheduler: scheduler.step()
            if ema: ema.update(model)
            step += 1; batch = index + 1
            if batch == len(loader): epoch, batch = epoch_index + 1, 0
            emit(run, "step", step=step, max_steps=cfg["steps"], epoch=epoch_index + 1,
                 losses={"total_loss": float(loss.detach()), **numbers(parts)}, lr=optimizer.param_groups[0]["lr"],
                 gpu_mb=torch.cuda.memory_allocated() / 2**20 if cfg["resolved_device"].startswith("cuda") else 0)
            for key,value in {"total_loss":float(loss.detach()),**numbers(parts)}.items(): writer.add_scalar("train/"+key,value,step)
            writer.add_scalar("train/learning_rate",optimizer.param_groups[0]["lr"],step)
            if (run / "STOP").exists():
                checkpoint(); raise Stopped()
            if step % cfg["save_interval"] == 0 or step == cfg["steps"] or epoch >= cfg["epochs"]:
                model.eval()
                if ema: ema.swap(model)
                with torch.no_grad(): score, report = validate()
                if ema: ema.swap(model)
                if not math.isfinite(score): raise ValueError("Validation did not produce a finite score")
                improved = score < best
                if improved: best, bad = score, 0
                else: bad += 1
                (run / "validation.json").write_text(json.dumps({"step": step, "score": score, **report}, ensure_ascii=False), encoding="utf-8")
                emit(run, "validation", step=step, val_loss=score, metrics={"validation.score": score})
                writer.add_scalar("validation/score",score,step);writer.flush()
                checkpoint(improved)
                if cfg["patience"] and bad >= cfg["patience"]: return
            if step >= cfg["steps"]: return
        offset = 0
    if step and (not listing(run) or listing(run)[-1]["step"] != step): checkpoint()
