"""Non-destructive recording preparation: pause cuts, notes and DiffSinger .ds files."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .. import formats
from ..formats import ds_csv
from ..labels import Interval, Label, fill_gaps
from .dataset import AUDIO, _csv_rows, _label_for, _label_index, _ds_for
from .doctor import inspect_recording
from .phonemizers import convert_label, vowel_inventory
from .preparation_check import preparation_check, validate_model, model_language_ids

SILENCES = {"sp", "sil", "pau", "<sp>", "silence", ""}


def add_groups(label: Label, vowels: list[str], liquids: list[str]) -> bool:
    """Fill absent ph_num groups using an explicit language inventory, never a first-letter guess."""
    if label.words:
        return True
    if not vowels:
        label.tiers["words"] = [Interval(start=p.start, end=p.end, text="_") for p in label.phones]
        return False
    phones = label.phones
    def is_vowel(phone):
        return phone.text in vowels or phone.text.split("/", 1)[-1] in vowels
    def is_liquid(phone):
        return phone.text in liquids or phone.text.split("/", 1)[-1] in liquids
    boundaries = {0, len(phones)}
    for i, phone in enumerate(phones):
        if phone.text.lower() in SILENCES or phone.text in {"AP", "br"}:
            boundaries.update((i, i + 1))
        elif is_vowel(phone):
            start = i
            if i > 1 and is_liquid(phones[i - 1]) and not is_vowel(phones[i - 2]) and phones[i - 2].text.lower() not in SILENCES:
                start -= 1
            boundaries.add(start)
    limits = sorted(boundaries)
    label.tiers["words"] = [Interval(start=phones[a].start, end=phones[b - 1].end,
        text="SP" if all(p.text.lower() in SILENCES for p in phones[a:b]) else "_") for a, b in zip(limits, limits[1:])]
    return True


def cut_points(label: Label, max_length: float, min_silence: float) -> list[float]:
    """Cut only inside pauses outside words. An unbroken phrase may exceed the target."""
    end = label.end_time()
    words = label.tiers.get("words", [])
    pauses = [(p.start + p.end) / 2 for p in label.phones
              if p.text.lower() in SILENCES and p.duration >= min_silence
              and not any(w.start < (p.start + p.end) / 2 < w.end
                          and any(q.text.lower() not in SILENCES and q.start < w.end and q.end > w.start for q in label.phones)
                          for w in words)]
    points = [0.0]
    while end - points[-1] > max_length:
        nearby = [p for p in pauses if points[-1] + min_silence < p <= points[-1] + max_length]
        if not nearby:
            nearby = [p for p in pauses if p > points[-1] + min_silence]
            if not nearby:
                break
            points.append(nearby[0])
        else:
            points.append(nearby[-1])
    return points + [end]


def crop_label(label: Label, start: float, end: float) -> Label:
    tiers = {tier: [Interval(start=max(p.start, start) - start, end=min(p.end, end) - start,
                             text=p.text, confidence=p.confidence, data=dict(p.data or {}))
                    for p in entries if p.end > start + 1e-7 and p.start < end - 1e-7]
             for tier, entries in label.tiers.items()}
    return Label(tiers=tiers)


def notes_for_label(label: Label, estimated: list[dict]) -> list[Interval]:
    """Fill rests, snap pitches and split at word boundaries; slurs stay within a word."""
    end = label.end_time()
    raw = []
    cursor = 0.0
    for item in sorted(estimated, key=lambda n: float(n["start"])):
        start, stop = max(cursor, float(item["start"])), min(end, float(item["end"]))
        pitch = float(item["pitch"])
        if not all(math.isfinite(v) for v in (start, stop, pitch)) or stop <= start:
            continue
        if start > cursor:
            raw.append((cursor, start, "rest"))
        midi = round(pitch)
        name = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[midi % 12] + str(midi // 12 - 1)
        raw.append((start, stop, name))
        cursor = stop
    if cursor < end:
        raw.append((cursor, end, "rest"))
    return align_notes(label, raw)


def align_notes(label: Label, raw: list[tuple[float, float, str]]) -> list[Interval]:
    words = label.tiers.get("words") or label.phones
    result = []
    for word in words:
        within = []
        for start, stop, name in raw:
            a, b = max(start, word.start), min(stop, word.end)
            if b - a < 1e-7:
                continue
            if within and within[-1].text == name:
                within[-1].end = b
            else:
                within.append(Interval(start=a, end=b, text=name, data={"slur": int(bool(within))}))
        result.extend(within)
    return result


def ds_segment(row: dict, f0: list[float] | None = None, hop: float = 0.01) -> dict:
    ds_csv.label_from_row(row)  # rejects mismatched and non-finite durations
    result = {"offset": 0.0, "text": row.get("word_seq", ""),
              **{k: str(row[k]) for k in ("ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur", "note_glide") if row.get(k)}}
    if "ph_num" not in result:
        result["ph_num"] = " ".join("1" for _ in str(row["ph_seq"]).split())
    if f0 is not None:
        if not f0 or any(not math.isfinite(v) or v < 0 for v in f0):
            raise ValueError("f0 must be a non-empty finite curve")
        result.update(f0_timestep=str(hop), f0_seq=" ".join(f"{v:.3f}" for v in f0))
    return result


def segment_bounds(label: Label, start: float, end: float, edge_silence: float | None, fade_ms: float) -> dict:
    """Cap only labelled silence; fades never reach a sung phone or breath."""
    first = next((p for p in label.phones if p.end > start + 1e-7), None)
    last = next((p for p in reversed(label.phones) if p.start < end - 1e-7), None)
    if edge_silence is not None:
        if first and first.text.lower() in SILENCES:
            start = min(end, max(start, first.end - edge_silence))
        if last and last.text.lower() in SILENCES:
            end = max(start, min(end, last.start + edge_silence))
    if end <= start:
        return {}
    fade = fade_ms / 1000
    return dict(start=start, end=end,
        fade_in=min(fade, max(0, first.end - start), (end - start) / 2) if first and first.text.lower() in SILENCES else 0,
        fade_out=min(fade, max(0, end - last.start), (end - start) / 2) if last and last.text.lower() in SILENCES else 0)


def preserved_f0(audio: Path, index: dict, start: float, end: float) -> tuple[list[float] | None, float]:
    """Read a matching DS curve without changing its sample grid or source file."""
    path = _ds_for(audio, index)
    if path is None:
        return None, .01
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = entries if isinstance(entries, list) else [entries]
    if len(entries) != 1:
        return None, .01
    item = entries[0]
    if not item.get("f0_seq"):
        return None, .01
    hop = float(item.get("f0_timestep", .01))
    if not math.isfinite(hop) or hop <= 0:
        raise ValueError("Source DS has an invalid f0_timestep")
    offset = float(item.get("offset", 0))
    if not math.isfinite(offset):
        raise ValueError("Source DS has an invalid offset")
    values = [float(v) for v in str(item["f0_seq"]).split()]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Source DS has an invalid f0 curve")
    if start < offset - 1e-5 or end > offset + len(values) * hop + .002:
        return None, hop
    # Preserve exactly when no cut is made; otherwise sample the original grid
    # at the segment's grid times (nearest neighbour, reported as preserved).
    return [values[min(len(values) - 1, max(0, round((start + n * hop - offset) / hop)))]
            for n in range(max(1, math.ceil((end - start) / hop)))], hop


async def prepare_recordings(tk, job, req, call) -> dict[str, Any]:
    check = preparation_check(tk, req)
    if not check["ok"]:
        raise ValueError("Preparation check failed: " + "; ".join(p["problem"] for p in check["problems"] if p["severity"] == "error"))
    root, out = Path(req.folder).expanduser().resolve(), Path(req.out_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Recording folder does not exist")
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Preparation output and source folders must not overlap")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation output must be empty")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO)
    if not files:
        raise ValueError("No WAV or FLAC recordings found")
    out.mkdir(parents=True, exist_ok=True)
    index, csv_cache = _label_index(root), {}
    rows, made, problems, conversions = [], [], [], []
    label_model = None
    if check["pending_auto_labels"]:
        label_model = await tk.models.require("wfl_asr", req.label_model)
        validate_model(label_model, req)
        language_ids = model_language_ids(label_model)
    model = None
    if req.estimate_notes:
        model = await tk.models.require("game", req.note_model)
    method, model_path = req.f0_method, None
    if req.write_ds and method not in {"none", "preserve"}:
        from ..pipelines.extra import _pitch_model
        method, model_path, _ = await _pitch_model(tk, job, method)
    for i, audio in enumerate(files):
        job.check_cancelled()
        job.progress(i / len(files), stage="prepare_recordings", message=audio.name)
        csv_cache.setdefault(audio.parent, _csv_rows(audio.parent) or _csv_rows(root))
        label = _label_for(audio, csv_cache[audio.parent], index)
        automatic = req.label_mode == "replace" or not label or not label.phones
        if automatic and label_model is not None:
            generation = int(label_model.layout.get("generation", 1))
            res = await call(tk, job, "wfl_asr_v2" if generation >= 2 else "wfl_asr", "segment", {
                "model": {"path": label_model.path, "layout": label_model.layout},
                "items": [{"audio": str(audio), "name": audio.stem, "phonemes": None}],
                "lang_id": language_ids[req.language],
                "extra_lang_ids": [language_ids[c] for c in dict.fromkeys(req.extra_languages) if c != req.language],
                "language_prefix": "always", "decoder": "viterbi", "confidence": True})
            if len(res) != 1 or not res[0].get("ok", True) or not res[0].get("phones"):
                raise ValueError(f"WFL labelling failed for {audio.name}: {res}")
            label = Label(tiers={"phones": [Interval(start=s, end=e, text=t,
                confidence=rest[0] if rest else None) for s, e, t, *rest in res[0]["phones"]]})
            raw = out / "original-labels" / audio.relative_to(root).with_suffix(".json")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(label.model_dump_json(indent=1), encoding="utf-8")
        if not label or not label.phones:
            problems.append({"file": str(audio), "problem": "no phoneme labels", "code": "no_labels"})
            continue
        findings, metadata = inspect_recording(audio, label)
        errors = [p for p in findings if p.get("severity") == "error"]
        if errors:
            problems.extend(errors)
            continue
        if automatic:
            label.tiers["phones"] = fill_gaps(label.phones, metadata.get("seconds", label.end_time()))
        prefix_languages = req.language_prefix or bool(req.extra_languages)
        label, conversion = convert_label(label, req.language, req.phonemizer_profiles,
            alphabet="wfl-native" if automatic else req.source_alphabet,
            prefix=prefix_languages, overrides=req.phoneme_mappings, unknown_policy="keep")
        conversions.append({"file": str(audio.relative_to(root)), "automatic": automatic,
                            "model": label_model.id if automatic and label_model else None, **conversion})
        for field in ("unknown", "lossy", "extensions"):
            if conversion[field]:
                problems.append({"file": str(audio), "code": field, "problem": str(conversion[field])})
        if conversion["unknown"] and req.unknown_phonemes == "error":
            (out / "preparation-report.json").write_text(json.dumps(dict(count=len(made), files=made, problems=problems,
                conversions=conversions, out_dir=str(out), status="blocked_unknown_phonemes"), ensure_ascii=False, indent=1), encoding="utf-8")
            raise ValueError("Unmapped WFL phonemes: " + ", ".join(conversion["unknown"]) + f". See {out / 'preparation-report.json'}")
        vowels = req.vowels or vowel_inventory(req.phonemizer_profiles, prefix_languages)
        if not add_groups(label, vowels, req.liquids):
            problems.append({"file": str(audio), "problem": "no word groups: provide vowel/liquid inventory for ph_num; fallback is one phoneme per group", "code": "missing_word_groups"})
        points = cut_points(label, req.max_length, req.min_silence) if req.segment else [0.0, label.end_time()]
        prefix = audio.stem + "-" + hashlib.sha256(str(audio.relative_to(root)).encode()).hexdigest()[:8]
        slices = [{**bounds, "path": str(out / "wavs" / f"{prefix}-{j + 1:03d}.wav")}
                  for j, (a, b) in enumerate(zip(points, points[1:]))
                  if (bounds := segment_bounds(label, a, b, req.edge_silence, req.fade_ms))]
        await call(tk, job, "audio", "slice_audio", {"audio": str(audio), "slices": slices})
        for part in slices:
            job.check_cancelled()
            target = Path(part["path"])
            piece = crop_label(label, part["start"], part["end"])
            if req.segment and part["end"] - part["start"] > req.max_length + .001:
                problems.append({"file": str(target), "problem": "no safe pause within target duration", "code": "long_phrase"})
            if model is not None:
                res = await call(tk, job, "game", "extract", {"model": {"path": model.path, "layout": model.layout},
                    "audio": str(target), "output_dir": str(out / "notes"), "language": req.language or None,
                    "output_formats": ["csv"], "round_pitch": True})
                if not res.get("ok", True):
                    raise ValueError(res.get("error", "Note estimation failed"))
                piece.tiers["notes"] = notes_for_label(piece, res["notes"])
            elif piece.tiers.get("notes"):
                piece.tiers["notes"] = align_notes(piece, [(n.start, n.end, n.text) for n in piece.tiers["notes"]])
            formats.write(piece, target.with_suffix(".lab"), "htk")
            # TextGrid preserves words and notes for later imports.
            formats.write(piece, target.with_suffix(".TextGrid"), "textgrid")
            row = ds_csv.row_from_label(target.stem, piece)
            rows.append(row)
            if req.write_ds:
                curve, hop = None, .01
                if method == "preserve":
                    curve, hop = preserved_f0(audio, index, part["start"], part["end"])
                    if curve is None:
                        problems.append({"file": str(audio), "code": "missing_f0", "problem": "No single-segment source DS curve to preserve; output DS has no f0"})
                elif method != "none":
                    res = await call(tk, job, "pitch", "pitch", {"items": [{"audio": str(target)}],
                        "method": method, "model_path": model_path, "hop": .01, "f0_min": 50.0, "f0_max": 1600.0})
                    if not res[0].get("ok", True):
                        raise ValueError(res[0].get("error", "Pitch extraction failed"))
                    curve = res[0]["f0"]
                dest = out / "ds" / (target.stem + ".ds")
                dest.parent.mkdir(exist_ok=True)
                dest.write_text(json.dumps([ds_segment(row, curve, hop)], ensure_ascii=False, indent=1), encoding="utf-8")
            made.append(str(target))
    (out / "transcriptions.csv").write_text(ds_csv.dumps_rows(rows), encoding="utf-8")
    result = {"out_dir": str(out), "files": made, "count": len(made), "problems": problems,
              "notes": req.estimate_notes, "f0_method": method if req.write_ds else None,
              "phonemizer_profiles": req.phonemizer_profiles, "conversions": conversions,
              "label_model": label_model.id if label_model else None, "primary_language": req.language,
              "extra_languages": req.extra_languages}
    (out / "preparation-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if not made:
        raise ValueError("No labelled recordings could be prepared; check source labels")
    return result
