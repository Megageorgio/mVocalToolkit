"""RNN-T training with checkpoints, reproducible resume and held-out pronunciation metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torchaudio

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sofa"))
sys.path.insert(0, str(HERE.parent / "diffsinger"))
from g2p_model import _instantiate
from checkpoint_store import listing, retain_best


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, token in enumerate(a):
        new = [i + 1]
        for j, target in enumerate(b):
            new.append(min(new[-1] + 1, row[j + 1] + 1, row[j] + (token != target)))
        row = new
    return row[-1]


@torch.no_grad()
def evaluate(model, values, stopped=lambda: False):
    model.eval()
    errors, edits, phones, wrong = [], 0, 0, 0
    for word, references in values:
        if stopped():
            return None
        prediction = model.predict_str(word)
        reference = min(references, key=lambda r: (distance(prediction, r), len(r)))
        count = distance(prediction, reference)
        edits += count
        phones += len(reference)
        wrong += count != 0
        if count and len(errors) < 200:
            errors.append({"word": word, "prediction": prediction, "references": references, "edits": count})
    return {"per": edits / max(1, phones), "wer": wrong / max(1, len(values)), "words": len(values), "errors": errors}


def train(run: Path, device: str, resume: bool):
    data = json.loads((run / "dictionary-data.json").read_text(encoding="utf-8"))
    random.seed(data["seed"])
    torch.manual_seed(data["seed"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _instantiate(data["model"]).to(device)
    if data.get("base_weights"):
        state = torch.load(run / "base-model.ptsd", map_location="cpu", weights_only=True)
        if "state_dict" in state: state = state["state_dict"]
        target = model.state_dict()
        for key, weights in target.items():
            source = state.get(key)
            if source is None:
                raise ValueError(f"Missing base-model parameter: {key}")
            if source.shape == weights.shape:
                target[key] = source
            elif key in {"encoder.emb.weight", "decoder.emb.weight", "decoder.joint.project_layer.weight", "decoder.joint.project_layer.bias"} and source.shape[1:] == weights.shape[1:] and source.shape[0] <= weights.shape[0]:
                weights[:source.shape[0]].copy_(source)
            else:
                raise ValueError(f"Incompatible base-model parameter: {key}")
        model.load_state_dict(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=data["learning_rate"], betas=(.9, .98))
    rng = random.Random(data["seed"])
    step, epoch, offset, best, stale, order = 0, 0, 0, float("inf"), 0, []
    checkpoints = listing(run)
    if resume and checkpoints:
        state = torch.load(checkpoints[-1]["path"], map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, offset, best, stale, order = (state[k] for k in ("step", "epoch", "offset", "best", "stale", "order"))
        rng.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    def emit(event, **values):
        with (run / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "time": time.time(), **values}) + "\n")
    def stopped():
        return (run / "STOP").exists()
    def save():
        target = run / f"model_ckpt_steps_{step}.ckpt"
        temporary = target.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "epoch": epoch, "offset": offset, "best": best, "stale": stale, "order": order,
                    "random_state": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}, temporary)
        temporary.replace(target)
        emit("checkpoint", step=step, path=str(target), size=target.stat().st_size)
        return target
    chars = {v: i for i, v in enumerate(data["model"]["encoder"]["graphemes"])}
    phones = {v: i for i, v in enumerate(data["model"]["decoder"]["phonemes"])}
    pad = torch.nn.utils.rnn.pad_sequence
    emit("start", max_steps=data["max_updates"], resumed=resume)
    for warning in data["warnings"]:
        emit("warning", kind="dictionary", text=warning)
    while epoch < data["epochs"] and step < data["max_updates"]:
        if not order:
            order = list(range(len(data["train"])))
            rng.shuffle(order)
        model.train()
        while offset < len(order) and step < data["max_updates"]:
            if stopped():
                save(); emit("end", status="stopped"); return
            items = [data["train"][n] for n in order[offset:offset + data["batch_size"]]]
            src = [torch.tensor([chars[c] for c in w], dtype=torch.int32) for w, _ in items]
            tgt = [torch.tensor([phones[p] for p in pron], dtype=torch.int32) for _, pron in items]
            lengths = torch.tensor([len(s) for s in src], dtype=torch.int32, device=device)
            target_lengths = torch.tensor([len(t) for t in tgt], dtype=torch.int32, device=device)
            src_batch = pad(src, batch_first=True, padding_value=1).to(device)
            targets = pad(tgt, batch_first=True, padding_value=0).to(device)
            decoder_input = torch.cat((torch.full((len(items), 1), 2, dtype=torch.int32, device=device), targets), dim=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(model.encoder.emb(src_batch), lengths.cpu(), batch_first=True, enforce_sorted=False)
            memory, _ = model.encoder.lstm(packed)
            memory, _ = torch.nn.utils.rnn.pad_packed_sequence(memory, batch_first=True)
            memory = model.encoder.fc(memory)
            decoded, _ = model.decoder.lstm(model.decoder.emb(decoder_input))
            logits = model.decoder.joint(memory, model.decoder.fc(decoded))
            loss = torchaudio.functional.rnnt_loss(logits.float(), targets, lengths, target_lengths, blank=2, clamp=1.0)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite RNN-T loss; reduce learning rate or inspect the dictionary")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            offset += len(items)
            emit("step", step=step, epoch=epoch + 1, losses={"total_loss": float(loss.detach())}, lr=optimizer.param_groups[0]["lr"])
        metrics = evaluate(model, data["validation"], stopped)
        if metrics is None:
            save(); emit("end", status="stopped"); return
        improved = metrics["per"] < best
        best, stale = min(best, metrics["per"]), 0 if improved else stale + 1
        epoch += 1
        offset, order = 0, []
        emit("validation", step=step, val_loss=metrics["per"], metrics={"phoneme_error_rate": metrics["per"], "word_error_rate": metrics["wer"]})
        validation_text = json.dumps({"step": step, **metrics}, ensure_ascii=False, indent=1)
        temporary_report = run / "validation.json.tmp"
        temporary_report.write_text(validation_text, encoding="utf-8")
        temporary_report.replace(run / "validation.json")
        (run / "validations").mkdir(exist_ok=True)
        (run / "validations" / f"{step}.json").write_text(validation_text, encoding="utf-8")
        checkpoint = save()
        if improved:
            retain_best(run, step)
        # Keep recent checkpoints and the best/manual retained copies.
        recent = sorted(run.glob("model_ckpt_steps_*.ckpt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for old in recent[:-3]:
            old.unlink()
            emit("checkpoint_removed", step=int(old.stem.rsplit("_", 1)[1]))
        if data["patience"] and stale >= data["patience"]:
            emit("warning", kind="early_stop", text="Held-out phoneme error did not improve; early stopping")
            break
    emit("end", status="done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default=os.environ.get("MVT_DEVICE", "auto"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    folder = Path(args.run_dir)
    try:
        train(folder, args.device, args.resume)
    except Exception as exc:
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "end", "time": time.time(), "status": "failed", "error": str(exc)}) + "\n")
        raise
