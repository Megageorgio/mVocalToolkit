"""Pronunciation dictionaries (SOFA / DiffSinger style: "word<TAB>ph1 ph2 ..." per line)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class Dictionary:
    def __init__(self, entries: dict[str, list[str]]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> "Dictionary":
        return _load(str(Path(path).resolve()), Path(path).stat().st_mtime)

    def lookup(self, word: str) -> list[str] | None:
        for variant in _variants(word):
            if variant in self.entries:
                return self.entries[variant]
        return None

    def __contains__(self, word: str) -> bool:
        return self.lookup(word) is not None

    def phonemes(self) -> set[str]:
        return {ph for phs in self.entries.values() for ph in phs}

    def missing(self, words: list[str]) -> list[str]:
        result: list[str] = []
        for word in words:
            if word not in ("SP", "AP") and self.lookup(word) is None and word not in result:
                result.append(word)
        return result


def _variants(word: str):
    yield word
    lowered = word.lower()
    if lowered != word:
        yield lowered
    if "ё" in lowered:
        yield lowered.replace("ё", "е")
    if "е" in lowered:
        # some Russian dictionaries spell ё explicitly
        yield lowered.replace("е", "ё")


@lru_cache(maxsize=16)
def _load(path: str, _mtime: float) -> Dictionary:
    entries: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" in line:
                word, phones = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                word, phones = parts
            word = word.strip()
            if word and word not in entries:
                entries[word] = phones.split()
    return Dictionary(entries)
