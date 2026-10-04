"""DiffSinger transcriptions.csv.

Columns: name, ph_seq, ph_dur, and optionally ph_num, word_seq, word_dur, note_seq, note_dur, note_slur...
Durations are in seconds, sequences are space separated. One row per audio file.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from ..labels import Interval, Label

BASE_COLUMNS = ["name", "ph_seq", "ph_dur"]


def row_from_label(name: str, label: Label, with_words: bool = True, with_notes: bool = True) -> dict[str, str]:
    row: dict[str, str] = {"name": name}
    phones = label.tiers.get("phones", [])
    row["ph_seq"] = " ".join(iv.text for iv in phones)
    row["ph_dur"] = " ".join(_fmt(iv.duration) for iv in phones)
    words = label.tiers.get("words")
    if with_words and words:
        row["word_seq"] = " ".join(iv.text for iv in words)
        row["word_dur"] = " ".join(_fmt(iv.duration) for iv in words)
        row["ph_num"] = " ".join(str(n) for n in _ph_num(phones, words))
    notes = label.tiers.get("notes")
    if with_notes and notes:
        row["note_seq"] = " ".join(iv.text for iv in notes)
        row["note_dur"] = " ".join(_fmt(iv.duration) for iv in notes)
        row["note_slur"] = " ".join(str(int(bool((iv.data or {}).get("slur", 0)))) for iv in notes)
    return row


def _ph_num(phones: list[Interval], words: list[Interval]) -> list[int]:
    """Number of phonemes inside each word (by time)."""
    counts = []
    for word in words:
        n = sum(1 for ph in phones if ph.start >= word.start - 1e-4 and ph.end <= word.end + 1e-4)
        counts.append(max(n, 1))
    return counts


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def dumps_rows(rows: list[dict[str, str]]) -> str:
    columns = list(BASE_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buffer.getvalue()


def loads_rows(text: str) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(text)))


def label_from_row(row: dict[str, Any]) -> Label:
    label = Label()
    label.tiers["phones"] = _intervals(row.get("ph_seq", ""), row.get("ph_dur", ""))
    if row.get("word_seq") and row.get("word_dur"):
        label.tiers["words"] = _intervals(row["word_seq"], row["word_dur"])
    if row.get("note_seq") and row.get("note_dur"):
        label.tiers["notes"] = _intervals(row["note_seq"], row["note_dur"])
    return label


def _intervals(seq: str, dur: str) -> list[Interval]:
    texts = seq.split()
    durations = [float(x) for x in dur.split()]
    result = []
    t = 0.0
    for text, d in zip(texts, durations):
        result.append(Interval(start=t, end=t + d, text=text))
        t += d
    return result
