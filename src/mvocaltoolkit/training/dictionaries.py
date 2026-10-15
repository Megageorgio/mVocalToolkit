"""Import text pronunciation dictionaries and OpenUtau dsdict YAML into DiffSinger's text format."""
from __future__ import annotations

from pathlib import Path
import unicodedata
import uuid
from collections import Counter

import yaml


def read_entries(path: Path, language: str | None = None) -> list[tuple[str, list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[tuple[str, list[str]]] = []
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid dictionary YAML: {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise ValueError(f"Expected an OpenUtau dictionary with an entries list: {path}")
        for entry in data["entries"]:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid dictionary entry in {path}")
            word, phones = entry.get("grapheme"), entry.get("phonemes")
            if isinstance(phones, str):
                phones = phones.split()
            if not isinstance(word, str) or not isinstance(phones, list) or not phones or not all(isinstance(p, str) and p and not any(c.isspace() for c in p) for p in phones):
                raise ValueError(f"Each dictionary entry needs a grapheme and phonemes: {path}")
            if language:
                # A multilingual dsdict can be selected for each language without leaking another language's IDs.
                if any("/" in phone and not phone.startswith(language + "/") for phone in phones):
                    continue
                phones = [phone.removeprefix(language + "/") for phone in phones]
            entries.append((word, phones))
    else:
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith(("#", ";;;")):
                continue
            parts = line.split("\t", 1) if "\t" in line else line.split(maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].split():
                raise ValueError(f"Expected word followed by phonemes in {path}: {line[:80]}")
            entries.append((parts[0].strip(), parts[1].split()))
    if not entries:
        raise ValueError(f"No pronunciation entries for {language or 'the selected language'} in {path}")
    if any(not word.strip() or any(c in word for c in "\t\r\n") for word, _ in entries):
        raise ValueError(f"Dictionary graphemes must not contain tabs or newlines: {path}")
    return entries


def normalize_entries(values: list[dict]) -> list[tuple[str, list[str]]]:
    if not isinstance(values, list) or not values or len(values) > 200_000:
        raise ValueError("Provide between 1 and 200000 pronunciation entries")
    result = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("Each pronunciation entry must be an object")
        word = unicodedata.normalize("NFC", str(entry.get("word") or "").strip())
        phones = entry.get("phones")
        if isinstance(phones, str):
            phones = phones.split()
        if (not word or any(c in word for c in "\t\r\n") or not isinstance(phones, list) or not phones
                or any(not isinstance(p, str) or not p or any(c.isspace() for c in p) for p in phones)):
            raise ValueError("Every entry needs a word and space-separated phonemes, without tabs or newlines")
        result.append((word, [unicodedata.normalize("NFC", p) for p in phones]))
    return result


def describe_entries(entries: list[tuple[str, list[str]]]) -> dict:
    words = Counter(word for word, _ in entries)
    return {"entries": [{"word": word, "phones": phones} for word, phones in entries],
            "words": len(words), "duplicates": [word for word, count in words.items() if count > 1],
            "phonemes": dict(sorted(Counter(phone for _, phones in entries for phone in phones).items()))}


def save_copy(values: list[dict], out_dir: Path) -> dict:
    entries = normalize_entries(values)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ("dictionary-" + uuid.uuid4().hex[:12] + ".txt")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for word, phones in entries:
            stream.write(word + "\t" + " ".join(phones) + "\n")
    return {"path": str(target.resolve()), "words": len({word for word, _ in entries}), "entries": len(entries)}
