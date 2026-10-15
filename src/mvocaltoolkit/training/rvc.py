"""Torch-free RVC settings, recording checks and immutable input manifests."""
from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from pathlib import Path

from .vocoder import audio_info


def settings(config: dict) -> dict:
    def integer(key, default, low, high):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be a whole number between {low} and {high}")
        return value
    rate = integer("sample_rate", 40000, 32000, 48000)
    if rate not in {32000, 40000, 48000}:
        raise ValueError("sample_rate must be 32000, 40000 or 48000")
    epochs = integer("epochs", 100, 1, 10000)
    result = {"sample_rate": rate, "epochs": epochs, "batch_size": integer("batch_size", 2, 1, 64),
              "save_every_epoch": integer("save_every_epoch", min(5, epochs), 1, epochs),
              "max_steps": integer("max_steps", 20, 0, 10_000_000),
              "index_vectors": integer("index_vectors", 20000, 0, 100000),
              "seed": integer("seed", 1234, 0, 2**31 - 1)}
    for key, default, low, high in (("learning_rate", .0001, 0, .001), ("chunk_seconds", 3.7, 1, 8), ("overlap_seconds", .3, 0, 1)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {key}")
        result[key] = value
    if result["overlap_seconds"] >= result["chunk_seconds"]:
        raise ValueError("Overlap must be shorter than a chunk")
    if config.get("pitch_method", "harvest") != "harvest":
        raise ValueError("This trainer supports Harvest pitch extraction")
    if not isinstance(config.get("prepare_only", False), bool):
        raise ValueError("prepare_only must be true or false")
    result.update(pitch_method="harvest", prepare_only=config.get("prepare_only", False))
    result["pretrained"] = config.get("pretrained", True)
    if not isinstance(result["pretrained"], bool):
        raise ValueError("pretrained must be true or false")
    for key in ("base_generator", "base_discriminator"):
        result[key] = str(config.get(key) or "").strip()
        if result[key] and not Path(result[key]).is_file():
            raise ValueError(f"{key} file does not exist")
    if bool(result["base_generator"]) != bool(result["base_discriminator"]):
        raise ValueError("Choose both generator and discriminator training checkpoints")
    if result["base_generator"] and not result["pretrained"]:
        raise ValueError("Enable pretrained initialization to use local base weights")
    result["embedder"] = str(config.get("embedder") or "").strip()
    if result["embedder"]:
        folder = Path(result["embedder"])
        if not all((folder / n).is_file() for n in ("config.json", "pytorch_model.bin")):
            raise ValueError("The ContentVec folder needs config.json and pytorch_model.bin")
    return result


def scan_sources(config: dict) -> dict:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Select folders with recordings of one voice")
    speakers = {str(s.get("speaker") or "") for s in sources if isinstance(s, dict)}
    if len(speakers) > 1:
        raise ValueError("Select exactly one speaker for an RVC model")
    recordings, problems, seen = [], [], set()
    for source in sources:
        if not isinstance(source, dict) or not str(source.get("folder") or "").strip():
            raise ValueError("Each source needs a recording folder")
        folder = Path(source["folder"]).expanduser().resolve()
        if not folder.is_dir():
            problems.append({"code": "missing_folder", "severity": "error", "file": str(folder)})
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"} or any(p.startswith(".") for p in path.relative_to(folder).parts):
                continue
            try:
                rate, channels, seconds = audio_info(path)
                if rate < 16000 or channels not in {1, 2} or seconds < .8:
                    raise ValueError("Use mono/stereo audio, at least 16 kHz and 0.8 seconds")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in seen:
                    problems.append({"code": "duplicate", "severity": "note", "file": str(path)})
                    continue
                seen.add(digest)
                recordings.append({"file": str(path), "sha256": digest, "seconds": seconds, "sample_rate": rate,
                                   "channels": channels, "speaker": source.get("speaker", ""), "license": source.get("license", "")})
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                problems.append({"code": "invalid_audio", "severity": "error", "file": str(path), "detail": str(exc)})
    seconds = sum(r["seconds"] for r in recordings)
    if not recordings:
        problems.append({"code": "no_audio", "severity": "error"})
    if seconds < 600:
        problems.append({"code": "little_audio", "severity": "warning"})
    problems.append({"code": "training_preview", "severity": "note"})
    return {"files": len(recordings), "seconds": seconds, "speaker": next(iter(speakers), ""),
            "recordings": recordings, "problems": problems}


def check(config: dict) -> dict:
    return {"settings": settings(config), **scan_sources(config)}


def snapshot(config: dict, folder: Path) -> dict:
    data = check(config)
    if any(p["severity"] == "error" for p in data["problems"]):
        raise ValueError("Fix recording errors before starting RVC training")
    # Hash local base files as well: replacing them while a run waits in the queue must not silently change it.
    data["base_hashes"] = {}
    for key in ("base_generator", "base_discriminator"):
        if data["settings"][key]:
            with Path(data["settings"][key]).open("rb") as stream:
                data["base_hashes"][key] = hashlib.file_digest(stream, "sha256").hexdigest()
    content = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}.json"
    if not target.exists():
        import uuid
        tmp = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(content)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    return {"prepared_file": str(target), "prepared_sha256": digest,
            "data_report": {k: v for k, v in data.items() if k not in {"recordings", "base_hashes"}},
            "assets": resolve_assets(data["settings"])}


def resolve_assets(config: dict) -> dict:
    rate = config.get("sample_rate", 40000) // 1000
    assets = {"embedder": config.get("embedder") or "rvc-contentvec"}
    if config.get("base_generator"):
        assets.update(generator=config["base_generator"], discriminator=config["base_discriminator"])
    elif config.get("pretrained", True):
        assets.update(generator=f"rvc-hifigan-g-{rate}k", discriminator=f"rvc-hifigan-d-{rate}k")
    return assets
