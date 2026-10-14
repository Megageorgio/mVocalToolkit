"""Building a DiffSinger raw dataset (wavs/ + transcriptions.csv per speaker) from labelled recordings.

Sources are folders with recordings and labels the toolkit reads (.lab, .TextGrid, Audacity .txt, .json, a
transcriptions.csv or .ds files next to the audio). The output:

    <out>/<speaker>/wavs/<name>.wav      the recordings (copied; .flac stays .flac)
    <out>/<speaker>/transcriptions.csv   name, ph_seq, ph_dur (+ ph_num, note_seq, note_dur, note_slur when known)
    <out>/dictionary-<lang>.txt          every phoneme that occurs (except AP and SP), one "ph<TAB>ph" line each,
                                         unless a dictionary was given
    <out>/dataset.json                   what went in: speakers, languages, files, phonemes, minutes, problems
"""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import formats
from ..formats import ds_csv
from ..labels import Label

AUDIO = (".wav", ".flac")
LABELS = (".lab", ".TextGrid", ".textgrid", ".txt", ".json")
GLOBAL_PHONEMES = {"AP", "SP"}
# other labellers' names for silence and breath: DiffSinger reserves SP and AP
RENAMES = {"sil": "SP", "pau": "SP", "br": "AP", "bre": "AP"}


@dataclass
class SourceSpec:
    folder: str
    speaker: str = ""
    language: str = ""
    recursive: bool = True


@dataclass
class BuildResult:
    out_dir: str
    speakers: list[dict[str, Any]] = field(default_factory=list)
    dictionaries: dict[str, str] = field(default_factory=dict)
    phonemes: dict[str, list[str]] = field(default_factory=dict)
    problems: list[dict[str, Any]] = field(default_factory=list)
    files: int = 0
    seconds: float = 0.0


def _audio_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:  # noqa: BLE001
            return None
    return None


def _label_index(root: Path) -> dict[str, list[Path]]:
    """Label files below the source root by stem: labels often live in a folder of their own (lab/ next to wav/)."""
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in LABELS or path.suffix.lower() in (".lab", ".textgrid", ".json", ".ds")):
            index.setdefault(path.stem, []).append(path)
    return index


def _label_for(audio: Path, csv_rows: dict[str, Label], index: dict[str, list[Path]] | None = None) -> Label | None:
    if audio.stem in csv_rows:
        return csv_rows[audio.stem]
    candidates = [audio.with_suffix(ext) for ext in LABELS]
    for other in (index or {}).get(audio.stem, []):
        if other.parent != audio.parent and other not in candidates:
            candidates.append(other)
    for candidate in candidates:
        ext = candidate.suffix if candidate.suffix in LABELS else candidate.suffix.lower()
        if candidate.is_file():
            if ext == ".txt":
                # Audacity labels only: a lyrics .txt has no numbers
                try:
                    first = candidate.read_text(encoding="utf-8", errors="replace").strip().split("\n", 1)[0]
                    float(first.split("\t")[0])
                except (ValueError, IndexError):
                    continue
            try:
                return formats.read(candidate)
            except Exception:  # noqa: BLE001
                return None
    ds = audio.with_suffix(".ds")
    if ds.is_file():
        try:
            return formats.read(ds)
        except Exception:  # noqa: BLE001
            return None
    return None


def _csv_rows(folder: Path) -> dict[str, Label]:
    path = folder / "transcriptions.csv"
    if not path.is_file():
        return {}
    try:
        rows = ds_csv.loads_rows(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, Label] = {}
    for row in rows:
        try:
            out[str(row.get("name"))] = ds_csv.label_from_row(row)
        except Exception:  # noqa: BLE001
            continue
    return out


def _stable_choice(names: list[str], count: int) -> list[str]:
    """`count` names chosen by a hash of the name: the same validation files every time."""
    if len(names) <= count:
        return list(names)
    ranked = sorted(names, key=lambda n: hashlib.sha1(n.encode("utf-8")).hexdigest())
    return sorted(ranked[:count])


def build_dataset(sources: list[SourceSpec], out_dir: Path, dictionaries: dict[str, str] | None = None,
                  validation_files: int = 3, progress: Any = None) -> BuildResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BuildResult(out_dir=str(out_dir))
    phonemes_by_lang: dict[str, set[str]] = {}
    speakers: dict[str, dict[str, Any]] = {}
    all_audio: list[tuple[SourceSpec, Path, Path]] = []
    for source in sources:
        root = Path(source.folder).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Source folder not found: {root}")
        files = sorted(p for p in (root.rglob("*") if source.recursive else root.iterdir())
                       if p.is_file() and p.suffix.lower() in AUDIO and not any(part.startswith(".") for part in p.relative_to(root).parts))
        for audio in files:
            all_audio.append((source, root, audio))
    total = max(1, len(all_audio))
    csv_cache: dict[Path, dict[str, Label]] = {}
    label_index: dict[Path, dict[str, list[Path]]] = {}
    seen_names: dict[str, set[str]] = {}
    for index, (source, root, audio) in enumerate(all_audio):
        if progress is not None:
            progress(index / total, f"{audio.name}")
        speaker = source.speaker or root.name
        language = source.language or (next(iter(dictionaries)) if dictionaries else "xx")
        folder = audio.parent
        if folder not in csv_cache:
            csv_cache[folder] = _csv_rows(folder)
        if root not in label_index:
            label_index[root] = _label_index(root)
        label = _label_for(audio, csv_cache[folder], label_index[root])
        if label is None or not label.phones:
            result.problems.append({"file": str(audio), "problem": "no labels"})
            continue
        phones = label.phones
        for iv in phones:
            iv.text = RENAMES.get(iv.text, iv.text)
        bad = [iv for iv in phones if iv.end - iv.start <= 0 or not iv.text.strip()]
        if bad:
            result.problems.append({"file": str(audio), "problem": f"{len(bad)} empty or zero-length phonemes"})
            continue
        entry = speakers.setdefault(speaker, {"speaker": speaker, "language": language, "names": [], "seconds": 0.0,
                                              "raw_data_dir": str(out_dir / speaker)})
        name = audio.stem
        names = seen_names.setdefault(speaker, set())
        if name in names:
            k = 2
            while f"{name}_{k}" in names:
                k += 1
            name = f"{name}_{k}"
        names.add(name)
        wavs = out_dir / speaker / "wavs"
        wavs.mkdir(parents=True, exist_ok=True)
        target = wavs / (name + audio.suffix.lower())
        if not target.exists() or target.stat().st_size != audio.stat().st_size:
            shutil.copy2(audio, target)
        row = ds_csv.row_from_label(name, label, with_words="words" in label.tiers, with_notes="notes" in label.tiers)
        entry.setdefault("rows", []).append(row)
        entry["names"].append(name)
        seconds = _audio_seconds(audio) or label.end_time()
        entry["seconds"] += seconds
        result.seconds += seconds
        result.files += 1
        for iv in phones:
            if iv.text not in GLOBAL_PHONEMES:
                phonemes_by_lang.setdefault(language, set()).add(iv.text)
    for speaker, entry in speakers.items():
        rows = entry.pop("rows", [])
        (out_dir / speaker / "transcriptions.csv").write_text(ds_csv.dumps_rows(rows), encoding="utf-8")
        entry["test_prefixes"] = _stable_choice(entry["names"], validation_files)
        entry["files"] = len(entry["names"])
        entry.pop("names")
        result.speakers.append(entry)
    for language, found in phonemes_by_lang.items():
        result.phonemes[language] = sorted(found)
        if dictionaries and dictionaries.get(language):
            result.dictionaries[language] = str(dictionaries[language])
            known = read_dictionary_phonemes(Path(dictionaries[language]))
            missing = sorted(found - known)
            if missing:
                result.problems.append({"language": language, "problem": "phonemes not in the dictionary",
                                        "phonemes": missing})
        else:
            path = out_dir / f"dictionary-{language}.txt"
            path.write_text("".join(f"{ph}\t{ph}\n" for ph in sorted(found)), encoding="utf-8")
            result.dictionaries[language] = str(path)
    (out_dir / "dataset.json").write_text(json.dumps({
        "speakers": result.speakers, "dictionaries": result.dictionaries, "phonemes": result.phonemes,
        "problems": result.problems, "files": result.files, "seconds": result.seconds,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if progress is not None:
        progress(1.0, "Dataset written")
    return result


def read_dictionary_phonemes(path: Path) -> set[str]:
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            found.update(parts[1].split())
    return found


def spec_from_result(result: BuildResult, kind: str = "diffsinger_acoustic", preset: dict[str, Any] | None = None,
                     **extra: Any) -> dict[str, Any]:
    """A run spec config (see training/diffsinger.py) for the dataset just built."""
    return {
        "datasets": [{"raw_data_dir": s["raw_data_dir"], "speaker": s["speaker"], "language": s["language"],
                      "test_prefixes": s["test_prefixes"]} for s in result.speakers],
        "dictionaries": dict(result.dictionaries),
        "preset": preset or {"goal": "full"},
        **extra,
    }
