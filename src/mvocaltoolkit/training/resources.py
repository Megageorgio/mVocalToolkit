"""Opt-in batch sizing and bounded OOM recovery without architecture changes."""
import math
import os
import yaml


def resource_options(config):
    value = config.get("resources")
    if value is None: value = {}
    if not isinstance(value, dict): raise ValueError("resources must be an object")
    allowed = {"auto_batch", "oom_retries", "memory_fraction", "min_batch_frames", "min_batch_size"}
    if set(value) - allowed: raise ValueError("Unknown resource option")
    result = dict(auto_batch=value.get("auto_batch", False), oom_retries=value.get("oom_retries", 0),
        memory_fraction=value.get("memory_fraction", .7), min_batch_frames=value.get("min_batch_frames", 1000), min_batch_size=value.get("min_batch_size", 1))
    if not isinstance(result["auto_batch"], bool): raise ValueError("auto_batch must be true or false")
    for key in ("oom_retries", "min_batch_frames", "min_batch_size"):
        if isinstance(result[key], bool) or not isinstance(result[key], int): raise ValueError(f"{key} must be a whole number")
    fraction = result["memory_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
        raise ValueError("memory_fraction must be a finite number")
    if not 0 <= result["oom_retries"] <= 10 or not .1 <= result["memory_fraction"] <= .95 or result["min_batch_frames"] < 1 or result["min_batch_size"] < 1:
        raise ValueError("Invalid batch recovery limits or memory fraction")
    return result


def _validation_caps(config, training):
    limits = {}
    for key in ("max_val_batch_frames", "max_val_batch_size"):
        cap = training[key.replace("max_val_", "max_")]
        old = int(config.get(key) or 0)
        # Native configurations use nonpositive values for unlimited validation.
        limits[key] = min(old, cap) if old > 0 else cap
    return limits


def measured_limits(kind, config, environment, device):
    from .diffsinger import build_overlay
    options = resource_options(config)
    overlay = build_overlay(kind, {**config, "device": device})
    gpus = environment.get("gpus") or []
    index = int(device.split(":")[1]) if device.startswith("cuda:") else 0
    gpu = next((g for g in gpus if g["index"] == index), None)
    if device in {"cpu", "mps"} or not gpu:
        return {}, dict(device=device, measured=False, reason="No selected CUDA device; existing batch limits retained")
    free = float(gpu["memory_free_mb"])
    if not math.isfinite(free) or free <= 0: raise ValueError("CUDA device reports no free memory")
    reference = float((config.get("preset") or {}).get("gpu_gb", 8)) * 1024
    ratio = min(1., free * options["memory_fraction"] / max(1, reference))
    limits = {key: max(options[minimum], min(int(overlay[key]), int(overlay[key] * ratio))) for key, minimum in
        [("max_batch_frames", "min_batch_frames"), ("max_batch_size", "min_batch_size")]}
    # A user-set low cap must never be increased by an automatic minimum.
    limits = {key: min(value, int(overlay[key])) for key, value in limits.items()}
    limits.update(_validation_caps(overlay, limits))
    return limits, dict(device=index, measured=True, free_mb=free, memory_fraction=options["memory_fraction"], limits=limits)


def reduce_after_oom(run):
    info = run.info
    options = resource_options(info.spec.config)
    if not info.spec.kind.startswith("diffsinger_") or info.status.value != "failed" or info.error_kind != "oom" or info.resource_retries >= options["oom_retries"]:
        return None
    path = run.dir / "config.yaml"
    if not path.is_file() or not (run.dir / "PREPARED").is_file(): return None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    old = {key: int(cfg.get(key) or minimum) for key, minimum in [("max_batch_frames", options["min_batch_frames"]), ("max_batch_size", options["min_batch_size"])]}
    limits = {key: min(old[key], max(options[minimum], old[key] // 2)) for key, minimum in [("max_batch_frames", "min_batch_frames"), ("max_batch_size", "min_batch_size")]}
    validation = _validation_caps(cfg, limits)
    if limits == old and all(cfg.get(key) == value for key, value in validation.items()): return None
    cfg.update(limits)
    # Validation can allocate more than training; cap both paths.
    cfg.update(validation)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"); os.replace(tmp, path)
    info.config_written = cfg
    info.resource_retries += 1
    info.spec.config["overrides"] = {**info.spec.config.get("overrides", {}), **limits, **validation}
    info.spec.config["overlay"] = {**info.spec.config.get("overlay", {}), **limits, **validation}
    return dict(kind="oom_retry", attempt=info.resource_retries, previous=old, limits=limits, validation_limits=validation,
        text=f"OOM recovery {info.resource_retries}/{options['oom_retries']}: batch limits {limits}; restarting from the last saved checkpoint")
