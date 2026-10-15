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
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import formats
from ..formats import ds_csv
from ..labels import Label
from .doctor import inspect_recording
from .dictionaries import read_entries

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
    role: str = "main"
    license: str = ""


@dataclass
class BuildResult:
    out_dir: str
    speakers: list[dict[str, Any]] = field(default_factory=list)
    dictionaries: dict[str, str] = field(default_factory=dict)
    phonemes: dict[str, list[str]] = field(default_factory=dict)
    problems: list[dict[str, Any]] = field(default_factory=list)
    files: int = 0
    seconds: float = 0.0
    recordings: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, dict[str, int]] = field(default_factory=dict)


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


def _ds_for(audio: Path, index: dict[str, list[Path]] | None = None) -> Path | None:
    adjacent = audio.with_suffix(".ds")
    if adjacent.is_file():
        return adjacent
    choices = [p for p in (index or {}).get(audio.stem, []) if p.suffix.lower() == ".ds"]
    return choices[0] if len(choices) == 1 else None


def _label_for(audio: Path, csv_rows: dict[str, Label], index: dict[str, list[Path]] | None = None) -> Label | None:
    label = _base_label_for(audio, csv_rows, index)
    ds = _ds_for(audio, index)
    if ds is not None:
        try:
            extra = formats.read(ds)
            if label is None:
                return extra
            if (len(extra.phones) == len(label.phones) and
                    all(a.text == b.text and abs(a.start - b.start) < .002 and abs(a.end - b.end) < .002
                        for a, b in zip(label.phones, extra.phones))):
                for tier in ("notes", "words"):
                    if extra.tiers.get(tier) and not label.tiers.get(tier):
                        label.tiers[tier] = extra.tiers[tier]
        except (OSError, ValueError, TypeError):
            pass
    return label


def _base_label_for(audio: Path, csv_rows: dict[str, Label], index: dict[str, list[Path]] | None = None) -> Label | None:
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
                  validation_files: int = 3, progress: Any = None, extra_phonemes: list[str] | None = None) -> BuildResult:
    out_dir = Path(out_dir)
    if validation_files < 1:
        raise ValueError("At least one validation file is required")
    result = BuildResult(out_dir=str(out_dir))
    phonemes_by_lang: dict[str, set[str]] = {}
    speakers: dict[tuple[str, str], dict[str, Any]] = {}
    speaker_languages: dict[str, set[str]] = {}
    all_audio: list[tuple[SourceSpec, Path, Path]] = []
    for source in sources:
        root = Path(source.folder).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Source folder not found: {root}")
        if out_dir.resolve() == root.resolve() or root.resolve() in out_dir.resolve().parents or out_dir.resolve() in root.resolve().parents:
            raise ValueError("Dataset output and source folders must not overlap")
        speaker = source.speaker or root.name
        language = source.language or (next(iter(dictionaries)) if dictionaries else "xx")
        for component in (speaker, language):
            if not component or component in {".", ".."} or component.endswith((".", " ")) or re.search(r'[\\/:*?"<>|\x00-\x1f]', component):
                raise ValueError(f"Invalid speaker or language folder name: {component!r}")
        speaker_languages.setdefault(speaker, set()).add(language)
        files = sorted(p for p in (root.rglob("*") if source.recursive else root.iterdir())
                       if p.is_file() and p.suffix.lower() in AUDIO and not any(part.startswith(".") for part in p.relative_to(root).parts))
        for audio in files:
            all_audio.append((source, root, audio))
    out_dir.mkdir(parents=True, exist_ok=True)
    total = max(1, len(all_audio))
    csv_cache: dict[Path, dict[str, Label]] = {}
    label_index: dict[Path, dict[str, list[Path]]] = {}
    seen_names: dict[tuple[str, str], set[str]] = {}
    seen_audio: dict[str, str] = {}
    counts_by_lang: dict[str, Counter] = {}
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
        rows = csv_cache[folder] or _csv_rows(root)
        label = _label_for(audio, rows, label_index[root])
        if label is None or not label.phones:
            result.problems.append({"file": str(audio), "problem": "no labels", "code": "no_labels", "severity": "warning", "excluded": True})
            continue
        label = label.model_copy(deep=True)
        phones = label.phones
        for iv in phones:
            iv.text = RENAMES.get(iv.text, iv.text)
        bad = [iv for iv in phones if iv.end - iv.start <= 0 or not iv.text.strip()]
        if bad:
            result.problems.append({"file": str(audio), "problem": f"{len(bad)} empty or zero-length phonemes", "code": "invalid_labels", "severity": "warning", "excluded": True})
            continue
        findings, metadata = inspect_recording(audio, label)
        invalid = any(p["severity"] == "error" for p in findings)
        result.problems.extend([{**p, "excluded": invalid} for p in findings])
        if invalid:
            continue
        with audio.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen_audio:
            result.problems.append({"file": str(audio), "problem": "duplicate audio excluded", "code": "duplicate", "severity": "warning", "excluded": True, "original": seen_audio[digest]})
            continue
        seen_audio[digest] = str(audio)
        raw_dir = out_dir / speaker
        if len(speaker_languages[speaker]) > 1:
            raw_dir /= language
        entry = speakers.setdefault((speaker, language), {"speaker": speaker, "language": language, "names": [], "seconds": 0.0,
                                                        "role": source.role, "raw_data_dir": str(raw_dir), "note_files": 0, "group_files": 0})
        name = audio.stem
        names = seen_names.setdefault((speaker, language), set())
        if name in names:
            k = 2
            while f"{name}_{k}" in names:
                k += 1
            name = f"{name}_{k}"
        names.add(name)
        wavs = raw_dir / "wavs"
        wavs.mkdir(parents=True, exist_ok=True)
        target = wavs / (name + audio.suffix.lower())
        unchanged = False
        if target.exists() and target.stat().st_size == audio.stat().st_size:
            with target.open("rb") as stream:
                unchanged = hashlib.file_digest(stream, "sha256").hexdigest() == digest
        if not unchanged:
            shutil.copy2(audio, target)
        ds = _ds_for(audio, label_index[root])
        if ds is not None:
            try:
                ds_label = formats.read(ds)
                if len(ds_label.phones) != len(label.phones) or any(
                        RENAMES.get(a.text, a.text) != b.text or abs(a.duration - b.duration) > .002
                        for a, b in zip(ds_label.phones, label.phones)):
                    raise ValueError("phonemes or timing differ from the selected labels")
                contents = json.loads(ds.read_text(encoding="utf-8-sig"))
                if not isinstance(contents, list) or len(contents) != 1:
                    raise ValueError("training requires one segment per recording")
                contents[0]["ph_seq"] = " ".join(p.text for p in label.phones)
                destination = raw_dir / "ds" / (name + ".ds")
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(json.dumps(contents, ensure_ascii=False, indent=1), encoding="utf-8")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                result.problems.append({"file": str(ds), "problem": f".ds not copied: {exc}",
                                        "code": "ds_mismatch", "severity": "warning"})
        row = ds_csv.row_from_label(name, label, with_words="words" in label.tiers, with_notes="notes" in label.tiers)
        entry.setdefault("rows", []).append(row)
        entry["names"].append(name)
        entry["note_files"] += bool(label.tiers.get("notes")) and any(n.text.lower() not in {"rest", "r", "sp", "ap"} for n in label.tiers["notes"])
        entry["group_files"] += bool(row.get("ph_num"))
        seconds = _audio_seconds(audio) or label.end_time()
        entry["seconds"] += seconds
        result.seconds += seconds
        result.files += 1
        result.recordings.append({"file": str(audio), "prepared": str(target), "speaker": speaker, "language": language,
                                  "role": source.role, "license": source.license, "sha256": digest, **metadata})
        for iv in phones:
            if iv.text not in GLOBAL_PHONEMES:
                phonemes_by_lang.setdefault(language, set()).add(iv.text)
                counts_by_lang.setdefault(language, Counter())[iv.text] += 1
    for (speaker, language), entry in speakers.items():
        rows = entry.pop("rows", [])
        (Path(entry["raw_data_dir"]) / "transcriptions.csv").write_text(ds_csv.dumps_rows(rows), encoding="utf-8")
        # Always leave at least one recording for training, even in a tiny check run.
        entry["test_prefixes"] = _stable_choice(entry["names"], min(validation_files, max(0, len(rows) - 1)))
        entry["files"] = len(entry["names"])
        entry.pop("names")
        result.speakers.append(entry)
        if not entry["test_prefixes"]:
            result.problems.append({"speaker": speaker, "language": language, "code": "validation", "severity": "error", "problem": "need at least two distinct labelled recordings for a training/validation split"})
        elif len(entry["test_prefixes"]) < 3:
            result.problems.append({"speaker": speaker, "language": language, "code": "small_validation", "severity": "warning", "problem": "fewer than three validation recordings"})
        threshold = 600 if entry["role"] == "donor" else 1800
        if entry["seconds"] < threshold:
            result.problems.append({"speaker": speaker, "language": language, "code": "little_data", "severity": "warning", "problem": f"less than {threshold // 60} minutes for this {entry['role']} speaker"})
    for language, found in phonemes_by_lang.items():
        result.phonemes[language] = sorted(found)
        if dictionaries and dictionaries.get(language):
            source_dictionary = Path(dictionaries[language])
            snapshot = out_dir / f"dictionary-{language}.txt"
            entries = read_entries(source_dictionary, language)
            snapshot.write_text("".join(word + "\t" + " ".join(phones) + "\n" for word, phones in entries), encoding="utf-8")
            result.dictionaries[language] = str(snapshot)
            known = read_dictionary_phonemes(snapshot)
            missing = sorted(found - known - set(extra_phonemes or []))
            if missing:
                result.problems.append({"language": language, "problem": "phonemes not in the dictionary",
                                        "phonemes": missing, "code": "unknown_phonemes", "severity": "error"})
            uncovered = sorted(known - found - GLOBAL_PHONEMES)
            if uncovered:
                result.problems.append({"language": language, "problem": "dictionary phonemes missing from recordings", "phonemes": uncovered, "code": "missing_phonemes", "severity": "warning"})
        else:
            path = out_dir / f"dictionary-{language}.txt"
            path.write_text("".join(f"{ph}\t{ph}\n" for ph in sorted(found)), encoding="utf-8")
            result.dictionaries[language] = str(path)
        result.coverage[language] = dict(sorted(counts_by_lang.get(language, {}).items()))
        rare = sorted(p for p, n in result.coverage[language].items() if n < 5)
        if rare:
            result.problems.append({"language": language, "problem": "phonemes with fewer than five occurrences", "phonemes": rare, "code": "rare_phonemes", "severity": "warning"})
    if not result.files:
        result.problems.append({"problem": "no usable labelled recordings", "code": "no_data", "severity": "error"})
    (out_dir / "dataset.json").write_text(json.dumps({
        "speakers": result.speakers, "dictionaries": result.dictionaries, "phonemes": result.phonemes,
        "problems": result.problems, "files": result.files, "seconds": result.seconds,
        "recordings": result.recordings, "coverage": result.coverage,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if progress is not None:
        progress(1.0, "Dataset written")
    return result


def read_dictionary_phonemes(path: Path) -> set[str]:
    return {phoneme for _, phones in read_entries(path) for phoneme in phones}


def validate_report(path: str, needs_notes: bool = False, needs_groups: bool = False) -> None:
    """Apply the same preflight gate for CLI/API clients as for the desktop wizard."""
    if not path:
        return  # Older and hand-written specs may not carry a doctor report.
    report_path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"Dataset report is missing: {report_path}; rebuild the dataset")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    blockers = [p["problem"] for p in report.get("problems", []) if p.get("severity") == "error" and not p.get("excluded")]
    if not report.get("files"):
        blockers.append("no usable labelled recordings")
    if needs_notes and any(s.get("note_files", 0) < s.get("files", 0) for s in report.get("speakers", [])):
        blockers.append("pitch training needs notes in every recording")
    if needs_groups and any(s.get("group_files", s.get("files", 0)) < s.get("files", 0) for s in report.get("speakers", [])):
        blockers.append("duration training needs ph_num groups in every recording; prepare recordings first")
    if blockers:
        raise ValueError("Dataset check blocks training: " + "; ".join(dict.fromkeys(blockers)))


def spec_from_result(result: BuildResult, kind: str = "diffsinger_acoustic", preset: dict[str, Any] | None = None,
                     **extra: Any) -> dict[str, Any]:
    """A run spec config (see training/diffsinger.py) for the dataset just built."""
    return {
        "datasets": [{"raw_data_dir": s["raw_data_dir"], "speaker": s["speaker"], "language": s["language"],
                      "test_prefixes": s["test_prefixes"]} for s in result.speakers],
        "dictionaries": dict(result.dictionaries),
        "preset": preset or {"goal": "full"},
        "data_report": str(Path(result.out_dir) / "dataset.json"),
        **extra,
    }
