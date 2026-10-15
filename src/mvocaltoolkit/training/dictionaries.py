"""Import text pronunciation dictionaries and OpenUtau dsdict YAML into DiffSinger's text format."""
from __future__ import annotations

from pathlib import Path

import yaml


def read_entries(path: Path, language: str | None = None) -> list[tuple[str, list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[tuple[str, list[str]]] = []
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
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
