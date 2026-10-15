"""Deterministic phoneme-only audition phrases, without a text frontend."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import yaml
from pydantic import BaseModel, Field

from .phonemizers import PROFILES, default_profile


class PhraseRequest(BaseModel):
    model_config = {"extra": "forbid"}
    model_dir: str = ""
    run_id: str = ""
    config: str = ""
    language: str = "ru"
    mode: str = "cv"
    phonemes: str = ""
    vowels: list[str] | None = None
    consonants: list[str] | None = None
    notes: str = "C4"
    vowel_seconds: float = Field(.35, ge=.02, le=3)
    consonant_seconds: float = Field(.08, ge=.01, le=1)
    pause_seconds: float = Field(.12, ge=0, le=2)
    limit: int = Field(16, ge=1, le=64)


def inventory(folder: Path, language: str, cfg_path: str = "") -> dict:
    folder = folder.expanduser().resolve()
    config_file = Path(cfg_path).expanduser().resolve() if cfg_path else next((folder / name for name in ("dsconfig.yaml", "config.yaml") if (folder / name).is_file()), None)
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8-sig")) or {} if config_file else {}
    candidates = [folder / str(cfg.get("phonemes", "phonemes.json")), folder / "phonemes.json", folder / "binary/phonemes.json"]
    if cfg.get("binary_data_dir"):
        binary = Path(cfg["binary_data_dir"])
        candidates.append((binary if binary.is_absolute() else config_file.parent / binary) / "phonemes.json")
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise ValueError("No phonemes.json found; prepare the run or select an exported bank")
    data = json.loads(found.read_text(encoding="utf-8-sig"))
    phones = list(data) if isinstance(data, list) else sorted(data, key=data.get)
    if not phones or any(not isinstance(p, str) or not p or any(c.isspace() for c in p) for p in phones):
        raise ValueError("Invalid phoneme vocabulary")
    phones = [p for p in phones if not p.startswith("<") and ("/" not in p or p.startswith(language + "/"))]
    vowels, consonants = set(), set()
    for name in (f"dsdict-{language}.yaml", "dsdict.yaml"):
        path = folder / name
        if not path.is_file(): continue
        dictionary = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for row in dictionary.get("symbols", []):
            if row.get("symbol") in phones and row.get("symbol") not in {"SP", "AP"}:
                (vowels if row.get("type") == "vowel" else consonants).add(row["symbol"])
    profiles = {}
    run_file = folder / "run.json"
    if run_file.is_file():
        run = json.loads(run_file.read_text(encoding="utf-8-sig"))
        settings = run.get("spec", {}).get("config", {})
        profiles.update(settings.get("phonemizer_profiles") or {})
        report = settings.get("data_report")
        if isinstance(report, str) and Path(report).is_file():
            profiles.update(json.loads(Path(report).read_text(encoding="utf-8-sig")).get("phonemizer_profiles") or {})
    for name in ("report.json", "dataset-report.json", "dataset.json"):
        path = folder / name
        if path.is_file(): profiles.update(json.loads(path.read_text(encoding="utf-8-sig")).get("phonemizer_profiles") or {})
    selected = PROFILES.get(profiles.get(language) or default_profile(language), {})
    if language not in profiles:
        # Checkpoints may predate alphabet snapshots. Select the best inventory fit;
        # this only classifies trained symbols and never renames a phone.
        candidates = [p for p in PROFILES.values() if p["language"] == language]
        short_phones = {p.split("/", 1)[-1] for p in phones} - {"SP", "AP"}
        if candidates:
            selected = max(candidates, key=lambda p: (len(short_phones & set(p["vowels"] + p["consonants"])), p["id"] == selected.get("id")))
    for phone in phones:
        if phone in vowels or phone in consonants: continue
        short = phone.split("/", 1)[-1]
        if short in selected.get("vowels", []): vowels.add(phone)
        elif short in selected.get("consonants", []): consonants.add(phone)
    return dict(phonemes=phones, vowels=[p for p in phones if p in vowels], consonants=[p for p in phones if p in consonants and p not in {"SP", "AP"}],
        unclassified=[p for p in phones if p not in vowels | consonants | {"SP", "AP"}], vocabulary=str(found), language=language,
        classification_profile=selected.get("id"))


def note_midi(note):
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", note)
    if match:
        key, acc, octave = match.groups()
        number = (int(octave) + 1) * 12 + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[key.upper()] + {"": 0, "#": 1, "b": -1}[acc]
    else:
        number = float(note)
    if not math.isfinite(number) or not 1 <= number <= 127:
        raise ValueError("Notes must be C4/F#4/Bb3 or MIDI numbers 1–127")
    return number


def build_phrase(req: PhraseRequest, out: Path) -> dict:
    if req.mode not in {"cv", "vc", "v", "cvc", "manual"}:
        raise ValueError("Phrase mode must be cv, vc, v, cvc or manual")
    info = inventory(Path(req.model_dir), req.language, req.config)
    vocabulary = set(info["phonemes"])
    def resolve(token):
        if token in vocabulary: return token
        prefixed = req.language + "/" + token
        if prefixed in vocabulary: return prefixed
        raise ValueError(f"Untrained phoneme: {token}; choose symbols from this model's vocabulary")
    vowels = [resolve(p) for p in (info["vowels"] if req.vowels is None else req.vowels)]
    consonants = [resolve(p) for p in (info["consonants"] if req.consonants is None else req.consonants)]
    if set(vowels) & set(consonants) or {"SP", "AP"} & set(vowels + consonants):
        raise ValueError("Vowels/consonants must be disjoint and cannot contain SP/AP")
    if req.mode == "manual":
        groups = [[resolve(p) for p in line.split()] for line in re.split(r"[|\n]+", req.phonemes) if line.strip()]
        if not groups or sum(map(len, groups)) > 512:
            raise ValueError("Enter 1–512 trained phonemes; use | or new lines between syllables")
    else:
        if not vowels or (req.mode != "v" and not consonants):
            raise ValueError("Specify vowel/consonant lists for this custom phoneme set")
        # Rotate vowels between consonants so a small sample covers both inventories.
        groups = []
        for i in range(min(req.limit, len(vowels) * (len(consonants) if req.mode != "v" else 1))):
            c = consonants[i % len(consonants)] if consonants else None
            v = vowels[i % len(vowels)] if req.mode == "v" else vowels[(i // len(consonants) + i % len(consonants)) % len(vowels)]
            groups.append([v] if req.mode == "v" else [c, v] if req.mode == "cv" else [v, c] if req.mode == "vc" else [c, v, c])
    notes = req.notes.split()
    if not notes: raise ValueError("Specify at least one note")
    midis = [note_midi(note) for note in notes]
    phones, durations, counts, note_seq, note_dur, f0 = [], [], [], [], [], []
    for index, group in enumerate(groups):
        # Pauses and breaths have their own note/group, even in manually entered sequences.
        pieces, current = [], []
        for phone in group:
            if phone in {"SP", "AP"}:
                if current: pieces.append(current); current = []
                pieces.append([phone])
            else: current.append(phone)
        if current: pieces.append(current)
        for piece in pieces:
            rest = piece[0] in {"SP", "AP"}
            seconds = [req.pause_seconds if rest else req.vowel_seconds if p in vowels or p not in consonants else req.consonant_seconds for p in piece]
            seconds = [max(.01, s) for s in seconds]
            phones.extend(piece); durations.extend(seconds); counts.append(len(piece))
            note_seq.append("rest" if rest else notes[index % len(notes)]); note_dur.append(sum(seconds))
        if index < len(groups) - 1 and req.pause_seconds and "SP" in vocabulary:
            phones.append("SP"); durations.append(req.pause_seconds); counts.append(1); note_seq.append("rest"); note_dur.append(req.pause_seconds)
    total = sum(durations)
    if total > 120: raise ValueError("Test phrases must not exceed two minutes")
    # Frame boundaries use cumulative rounding, keeping notes and f0 on one timeline.
    elapsed, previous = 0., 0
    for note, seconds in zip(note_seq, note_dur):
        elapsed += seconds; end = round(elapsed / .01)
        hz = 0 if note == "rest" else 440 * 2 ** ((midis[notes.index(note)] - 69) / 12)
        f0.extend([round(hz, 4)] * (end - previous)); previous = end
    fmt = lambda values: " ".join(str(round(v, 6)) if isinstance(v, float) else str(v) for v in values)
    segment = dict(offset=0, ph_seq=" ".join(phones), ph_dur=fmt(durations), ph_num=fmt(counts), note_seq=" ".join(note_seq), note_dur=fmt(note_dur), note_slur=fmt([0] * len(counts)), f0_seq=fmt(f0), f0_timestep=.01)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "test-phrase.ds"
    path.write_text(json.dumps([segment], ensure_ascii=False, indent=1), encoding="utf-8")
    result = dict(path=str(path), seconds=total, syllables=[" ".join(group) for group in groups], segment=segment, inventory=info)
    (out / "phrase-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
