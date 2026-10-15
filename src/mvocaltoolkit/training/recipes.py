"""Preflight and frozen manifests for labelling, note and speech model training."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
import wave
from pathlib import Path

from .dataset import _csv_rows, _label_for, _label_index
from .vocoder import audio_info

FAMILIES = {
    "tifa_aligner": {"engine": "tifa_train", "input": "phones", "title": "TIFA", "lr": .0002},
    "sofa_aligner": {"engine": "sofa_train", "input": "phones", "title": "SOFA", "lr": .0002},
    "wfl_asr_teacher": {"engine": "label_train", "input": "phones", "title": "WFL-ASR", "lr": .0001},
    "mcfl_student": {"engine": "label_train", "input": "phones", "title": "MCFL", "lr": .002},
    "refiner_boundary": {"engine": "refiner_train", "input": "phones", "title": "mRefinerModel", "lr": .001},
    "game_notes": {"engine": "game_train", "input": "notes", "title": "GAME", "lr": .0002},
    "qwen_tts_voice": {"engine": "qwen_tts", "input": "text", "title": "Qwen3-TTS", "lr": .00002},
    "voxcpm_voice": {"engine": "voxcpm", "input": "text", "title": "VoxCPM2", "lr": .0001},
}
RECIPE_ENGINES = {f["engine"] for f in FAMILIES.values()}
SILENCE = {"SP", "AP", "sil", "pau", "br", ""}


def settings(kind: str, config: dict) -> dict:
    if kind not in FAMILIES:
        raise ValueError("Unknown training family")
    family = FAMILIES[kind]
    def integer(key, default, low, high):
        v = config.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
            raise ValueError(f"{key}: use a whole number from {low} to {high}")
        return v
    result = {"kind": kind, "epochs": integer("epochs", 20, 1, 10000),
              "steps": integer("steps", 100, 1, 10_000_000), "batch_size": integer("batch_size", 1, 1, 256),
              "seed": integer("seed", 42, 0, 2**31 - 1), "patience": integer("patience", 5, 0, 10000),
              "accumulate": integer("accumulate", 1, 1, 128), "workers": 0,
              "size": str(config.get("size", "small")), "speaker_name": str(config.get("speaker_name") or "voice")}
    result["save_interval"] = integer("save_interval", min(50, result["steps"]), 1, result["steps"])
    for key, default, low, high in (("learning_rate", family["lr"], 0, .1), ("validation_fraction", .1, .01, .5)):
        v = config.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low < v <= high:
            raise ValueError(f"Invalid {key}")
        result[key] = float(v)
    for key in ("prepare_only", "allow_download", "lora"):
        v = config.get(key, key != "prepare_only")
        if not isinstance(v, bool):
            raise ValueError(f"{key} must be true or false")
        result[key] = v
    result["lora_rank"] = integer("lora_rank", 16, 1, 256)
    for key in ("base_model", "teacher", "encoder", "tokenizer"):
        result[key] = str(config.get(key) or "").strip()
    if kind == "mcfl_student":
        teacher = Path(result["teacher"])
        if not result["teacher"] or not all((teacher/name).is_file() for name in ("config.yaml", "phonemes.txt", "langs.txt")) or not any(teacher.glob("*.pt")) and not any(teacher.glob("*.ckpt")):
            raise ValueError("MCFL needs an existing WFL-ASR v2 teacher folder; train/export WFL first")
    if result["base_model"] and kind not in {"qwen_tts_voice", "voxcpm_voice"} and not Path(result["base_model"]).exists():
        raise ValueError("The base model path does not exist")
    if family["input"] == "text":
        result["base_model"] = result["base_model"] or ("Qwen/Qwen3-TTS-12Hz-0.6B-Base" if kind == "qwen_tts_voice" else "openbmb/VoxCPM2")
        if not result["allow_download"] and not Path(result["base_model"]).is_dir():
            raise ValueError("Automatic downloads are disabled; choose a local base model folder")
        if not re.fullmatch(r"[\w.-]{1,80}", result["speaker_name"], re.UNICODE):
            raise ValueError("Speaker name: use 1–80 letters, digits, underscores or dots")
    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("Advanced overrides must be a JSON object")
    encoded = json.dumps(overrides, allow_nan=False)
    if len(encoded) > 1_048_576:
        raise ValueError("Overrides exceed 1 MB")
    result["overrides"] = overrides
    if family["input"] == "text" and overrides:
        raise ValueError("Speech trainers use the named settings; native overrides are not supported")
    if kind == "mcfl_student" and result["size"] not in {"small", "medium", "large"}:
        raise ValueError("MCFL size must be small, medium or large")
    dictionaries = config.get("dictionaries", {}) if kind == "tifa_aligner" else {}
    if not isinstance(dictionaries, dict) or any(not isinstance(k,str) or not isinstance(v,str) or not Path(v).is_file() for k,v in dictionaries.items()):
        raise ValueError("Project dictionaries must point to existing files")
    result["dictionaries"] = dictionaries
    return result


def _intervals(tier, seconds):
    output, cursor = [], 0.0
    for p in tier:
        start, end = float(p.start), float(p.end)
        if not math.isfinite(start + end) or start < cursor - .002 or start < 0 or end <= start or end > seconds + .03:
            raise ValueError("Labels overlap or exceed the audio; repair their timings")
        if start > cursor + .002:
            output.append([cursor, start, "SP"])
        output.append([max(cursor, start), min(end, seconds), p.text])
        cursor = end
    if seconds > cursor + .002:
        output.append([cursor, seconds, "SP"])
    return output


def scan(kind: str, config: dict) -> dict:
    cfg = settings(kind, config)
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Select recording folders")
    records, findings, seen = [], [], set()
    for source in sources:
        if not isinstance(source, dict) or not str(source.get("folder") or "").strip():
            raise ValueError("Each source needs a folder")
        folder = Path(source["folder"]).expanduser().resolve()
        lang = str(source.get("language") or "und")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", lang):
            raise ValueError("Use a short language code without path separators")
        if not folder.is_dir():
            findings.append({"severity": "error", "code": "missing_folder", "file": str(folder)})
            continue
        index = _label_index(folder)
        csv_rows = _csv_rows(folder)
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"} or any(p.startswith(".") for p in path.relative_to(folder).parts):
                continue
            try:
                rate, channels, seconds = audio_info(path)
                if seconds < .2 or seconds > 120 or channels not in {1, 2} or rate < 8000:
                    raise ValueError("Use mono/stereo recordings of 0.2–120 seconds, at least 8 kHz")
                label = _label_for(path, csv_rows, index) if FAMILIES[kind]["input"] != "text" else None
                record = {"audio": str(path), "language": lang, "speaker": str(source.get("speaker") or "voice"),
                          "license": str(source.get("license") or ""), "seconds": seconds}
                if FAMILIES[kind]["input"] == "text":
                    transcript = path.with_suffix(".txt")
                    if not transcript.is_file():
                        findings.append({"severity":"error","code":"missing_transcript","file":str(path),"detail":transcript.name})
                        continue
                    text = transcript.read_text(encoding="utf-8-sig").strip()
                    if not text or len(text) > 10000:
                        raise ValueError("A non-empty matching .txt transcript of at most 10000 characters is required")
                    record["text"] = text
                elif label is None:
                    raise ValueError("Matching timed labels are required")
                elif kind == "game_notes":
                    notes_tier = label.tiers.get("notes", [])
                    if not notes_tier or not any(n.text.lower() not in {"rest", "sp", "r", ""} for n in notes_tier):
                        raise ValueError("GAME needs note timings with at least one non-rest note (.ds or CSV)")
                    notes = _intervals(notes_tier, seconds)
                    normalized = []
                    for a, b, note in notes:
                        if note.lower() in {"sp", "r", "rest", ""}:
                            note = "rest"
                        else:
                            if not re.fullmatch(r"[A-Ga-g][#b]?-?\d+", note):
                                raise ValueError("Use note names such as C4, F#4, Bb3 or rest")
                            match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", note)
                            letter, accidental, octave = match.groups()
                            midi = (int(octave) + 1) * 12 + {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[letter.upper()] + {"":0,"#":1,"b":-1}[accidental]
                            if not 0 <= midi <= 127: raise ValueError("Notes must be within MIDI 0–127")
                            note = letter.upper() + accidental + octave
                        normalized.append([a,b,note])
                    record["notes"] = normalized
                else:
                    if not label.phones or not any(p.text not in SILENCE for p in label.phones):
                        raise ValueError("Phoneme labels are empty or contain only silence")
                    record["phones"] = _intervals(label.phones, seconds)
                    if kind == "refiner_boundary" and len(record["phones"]) < 2:
                        raise ValueError("A boundary refiner needs internal phoneme boundaries")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in seen:
                    findings.append({"severity": "note", "code": "duplicate", "file": str(path)})
                    continue
                seen.add(digest); record["sha256"] = digest; records.append(record)
            except (OSError, ValueError, EOFError, wave.Error) as exc:
                findings.append({"severity": "error", "code": "invalid_recording", "file": str(path), "detail": str(exc)})
    # Whole identical recordings cannot leak between train and validation, even across source folders/languages.
    for lang in sorted({r["language"] for r in records}):
        group = sorted([r for r in records if r["language"] == lang], key=lambda r: hashlib.sha256(f"{cfg['seed']}:{r['sha256']}".encode()).hexdigest())
        count = max(1, round(len(group) * cfg["validation_fraction"]))
        if len(group) < 2:
            findings.append({"severity": "error", "code": "no_split", "detail": lang})
        for i, record in enumerate(group):
            record["validation"] = i < count
    seconds = sum(r["seconds"] for r in records)
    if not records:
        findings.append({"severity": "error", "code": "no_audio"})
    if 0 < seconds < 600:
        findings.append({"severity": "warning", "code": "little_audio"})
    if FAMILIES[kind]["input"] == "text" and len({r["speaker"] for r in records}) > 1:
        findings.append({"severity": "error", "code": "one_speaker"})
    dictionaries = {}
    if kind == "tifa_aligner":
        from .dictionaries import read_entries
        for language, path in cfg["dictionaries"].items():
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", language): raise ValueError("Invalid dictionary language code")
            entries = read_entries(Path(path))
            dictionaries[language] = "\n".join(f"{word}\t{' '.join(phones)}" for word,phones in entries) + "\n"
    return {"settings": cfg, "files": len(records), "seconds": seconds, "dictionaries":dictionaries,
            "train_files": sum(not r["validation"] for r in records), "validation_files": sum(r["validation"] for r in records),
            "findings": findings, "records": records}


def check(kind, config):
    return {k: v for k, v in scan(kind, config).items() if k not in {"records","dictionaries"}}


def snapshot(kind: str, config: dict, folder: Path) -> dict:
    data = scan(kind, config)
    if any(f["severity"] == "error" for f in data["findings"]):
        raise ValueError("Fix recording findings before training")
    content = json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(content).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.json"
    if not path.exists():
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(content); os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return {"prepared_file": str(path), "prepared_sha256": digest, "assets": {},
            "data_report": {k: v for k, v in data.items() if k not in {"records","dictionaries"}}}
