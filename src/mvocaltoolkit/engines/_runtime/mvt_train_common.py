"""Shared run events and checkpoint storage for optional trainers."""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diffsinger"))
from checkpoint_store import listing, retain_best


class Stopped(Exception):
    pass


def emit(run, event, **data):
    with (Path(run) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "time": time.time(), **data}, allow_nan=False) + "\n")


def atomic_torch(value, path):
    import torch
    path = Path(path); tmp = path.with_suffix(".tmp")
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def saved(run, step, best=False):
    path = Path(run) / f"model_ckpt_steps_{step}.ckpt"
    emit(run, "checkpoint", step=step, path=str(path), size=path.stat().st_size)
    if best:
        retain_best(Path(run), step)
    for old in listing(Path(run))[:-3]:
        if not old["starred"] and Path(old["path"]).parent == Path(run) and not (Path(run) / "best" / Path(old["path"]).name).exists():
            Path(old["path"]).unlink(missing_ok=True)
            emit(run, "checkpoint_removed", step=old["step"])


def chosen(run, step=None):
    found = [c for c in listing(Path(run)) if step is None or c["step"] == step]
    if not found:
        raise ValueError("No saved checkpoint at this step")
    return Path(found[-1]["path"])


def device(requested=None):
    import torch
    value = requested or os.environ.get("MVT_DEVICE", "auto")
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cuda": value = "cuda:0"
    if value == "mps" or value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("This trainer requires an available CPU or CUDA device")
    if value.startswith("cuda"): torch.cuda.set_device(torch.device(value))
    return value


def seed(value):
    import numpy as np
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(data):
    import numpy as np
    import torch
    if not data: return
    random.setstate(data["python"]); torch.set_rng_state(data["torch"])
    n = data["numpy"]; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    if data.get("cuda") and torch.cuda.is_available(): torch.cuda.set_rng_state_all(data["cuda"])


def numbers(values):
    result = {}
    for key, value in values.items():
        try:
            if hasattr(value, "numel") and value.numel() != 1: continue
            n = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            if math.isfinite(n): result[str(key)] = n
        except (TypeError, ValueError):
            pass
    return result


def load_config(source, file):
    import yaml
    config = yaml.safe_load(Path(file).read_text(encoding="utf-8"))
    bases = config.pop("bases", [])
    if isinstance(bases, str): bases = [bases]
    result = {}
    for base in bases: merge(result, load_config(source, Path(source) / base))
    merge(result, config)
    return result


def merge(target, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict): merge(target[key], value)
        else: target[key] = value
    return target


def resolve_model(value, allow_download=True):
    path = Path(value).expanduser()
    if path.is_dir(): return path.resolve()
    if not allow_download: raise ValueError("Choose a local model: automatic downloads are disabled")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(value)).resolve()
