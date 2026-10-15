"""Small, torch-free checks and presets for SingingVocoders PC-NSF-HiFiGAN fine-tuning."""
from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any

BASE_MODEL = "singingvocoders-pc-nsf-hifigan-2025.02"
LICENSE = "CC BY-NC-SA 4.0"


def audio_info(path: Path) -> tuple[int, int, float]:
    if path.suffix.lower() == ".flac":
        with path.open("rb") as stream:
            if stream.read(4) != b"fLaC":
                raise ValueError("Invalid FLAC header")
            header = stream.read(4)
            if len(header) != 4 or header[0] & 0x7f != 0 or int.from_bytes(header[1:], "big") != 34:
                raise ValueError("FLAC STREAMINFO is missing")
            data = stream.read(34)
            if len(data) != 34:
                raise ValueError("Truncated FLAC STREAMINFO")
            packed = int.from_bytes(data[10:18], "big")
            rate, channels = packed >> 44, ((packed >> 41) & 7) + 1
            frames = packed & ((1 << 36) - 1)
            if not rate or not frames:
                raise ValueError("FLAC duration is unknown")
            return rate, channels, frames / rate
    with wave.open(str(path)) as stream:
        return stream.getframerate(), stream.getnchannels(), stream.getnframes() / stream.getframerate()


def scan_sources(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("sources") or []
    if not isinstance(sources, list) or not sources:
        raise ValueError("Add at least one folder of clean recordings")
    recordings, problems, seen = [], [], set()
    for source in sources:
        if not isinstance(source, dict) or not str(source.get("folder") or "").strip():
            raise ValueError("Each audio source needs a non-empty folder path")
        folder = Path(str(source.get("folder") or "")).expanduser().resolve()
        if not folder.is_dir():
            problems.append({"file": str(folder), "code": "missing_folder", "severity": "error", "problem": "Recording folder does not exist"})
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"} or any(p.startswith(".") for p in path.relative_to(folder).parts):
                continue
            try:
                rate, channels, seconds = audio_info(path)
                if rate != 44100:
                    problems.append({"file": str(path), "code": "sample_rate", "sample_rate": rate, "severity": "error",
                                     "problem": f"Use original 44100 Hz recordings (this file: {rate} Hz); upsampling does not restore bandwidth"})
                    continue
                if channels not in {1, 2} or seconds < 0.5:
                    raise ValueError("Use mono/stereo recordings at least 0.5 seconds long")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in seen:
                    problems.append({"file": str(path), "code": "duplicate", "severity": "note", "problem": "Exact duplicate excluded"})
                    continue
                seen.add(digest)
                recordings.append({"file": str(path), "sha256": digest, "seconds": seconds, "channels": channels,
                                   "speaker": source.get("speaker", ""), "license": source.get("license", "")})
                if seconds > 30:
                    problems.append({"file": str(path), "code": "long", "severity": "warning", "problem": "Long recording; short clean phrases are easier to validate"})
                if channels == 2:
                    problems.append({"file": str(path), "code": "stereo", "severity": "note", "problem": "Stereo will be mixed to mono"})
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                problems.append({"file": str(path), "code": "invalid_audio", "severity": "error", "problem": str(exc)})
    if len(recordings) < 2:
        problems.append({"code": "no_split", "severity": "error", "problem": "At least two distinct recordings are required for separate training and validation"})
    seconds = sum(r["seconds"] for r in recordings)
    if seconds < 600:
        problems.append({"code": "little_audio", "severity": "warning", "problem": "Under 10 minutes of audio: suitable for a setup check, limited for fine-tuning"})
    return {"files": len(recordings), "seconds": seconds, "recordings": recordings, "problems": problems, "license": LICENSE}


def build_overlay(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("accept_license") is not True:
        raise ValueError("Read and accept the pretrained vocoder's CC BY-NC-SA 4.0 terms before fine-tuning")
    preset = config.get("preset") or {}
    gpu = int(preset.get("gpu_gb", 8))
    goal = preset.get("goal", "check")
    if goal not in {"check", "full"}:
        raise ValueError("Vocoder goal must be check or full")
    def integer(key, default, low, high):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be a whole number between {low} and {high}")
        return value
    steps = integer("steps", 100 if goal == "check" else 2000, 1, 1_000_000)
    interval = integer("validation_interval", min(100 if goal == "check" else 250, steps), 1, steps)
    batch = integer("batch_size", 10 if gpu >= 24 else 6 if gpu >= 16 else 4 if gpu >= 12 else 2 if gpu >= 8 else 1, 1, 64)
    crop = integer("crop_frames", 48 if gpu >= 24 else 32 if gpu >= 16 else 20, 8, 256)
    lr = config.get("learning_rate", 0.00001)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("learning_rate must be between 0 and 0.001")
    device = str(config.get("device", "auto"))
    overlay = {"max_updates": steps * 2, "val_check_interval": interval, "batch_size": batch, "crop_mel_frames": crop,
               "ds_workers": 0, "dataloader_prefetch_factor": 2, "num_ckpt_keep": 5,
               "permanent_ckpt_start": 0, "permanent_ckpt_interval": 0, "num_sanity_val_steps": 0,
               "num_valid_plots": 3, "log_interval": 20, "pl_trainer_devices": 1, "pl_trainer_precision": "32-true",
               "audio_sample_rate": 44100, "audio_num_mel_bins": 128, "hop_size": 512, "fft_size": 2048,
               "win_size": 2048, "fmin": 40, "fmax": 16000, "pe": "parselmouth", "seed": 42,
               "discriminate_optimizer_args": {"lr": lr}, "generater_optimizer_args": {"lr": lr}}
    if device == "cpu":
        overlay["pl_trainer_accelerator"] = "cpu"
    elif device.startswith("cuda"):
        overlay["pl_trainer_accelerator"] = "gpu"
        if ":" in device:
            overlay["pl_trainer_devices"] = [int(device.split(":")[1])]
    return overlay


def resolve_assets(config: dict[str, Any]) -> dict[str, str]:
    return {"base": str(config.get("base_model") or BASE_MODEL)}
