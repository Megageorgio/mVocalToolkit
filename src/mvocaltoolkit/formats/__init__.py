"""Label file formats.

Supported:
    htk       HTK / NNSVS / Sinsy .lab ("start end phoneme", time in 100ns units)
    textgrid  Praat TextGrid (long and short text format)
    ds_csv    DiffSinger transcriptions.csv (name, ph_seq, ph_dur[, word_seq, word_dur, note_seq...])
    json      the toolkit's own JSON representation of Label
    audacity  Audacity label track ("start\\tend\\ttext", seconds)
"""

from __future__ import annotations

from pathlib import Path

from ..labels import Label
from . import audacity, ds_csv, htk, json_format, textgrid

FORMATS = {
    "htk": htk,
    "textgrid": textgrid,
    "json": json_format,
    "audacity": audacity,
}
# ds_csv works on many files at once, see ds_csv.write_rows / read_rows

ALIASES = {
    "lab": "htk",
    "nnsvs": "htk",
    "sinsy": "htk",
    "htk_lab": "htk",
    "praat": "textgrid",
    "tg": "textgrid",
    "diffsinger": "ds_csv",
    "csv": "ds_csv",
    "transcriptions": "ds_csv",
    "txt": "audacity",
}

EXTENSIONS = {
    "htk": ".lab",
    "textgrid": ".TextGrid",
    "json": ".json",
    "audacity": ".txt",
    "ds_csv": ".csv",
}


def normalize_format(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in FORMATS and key != "ds_csv":
        raise ValueError(f"Unknown label format: {name}")
    return key


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".lab":
        return "htk"
    if suffix == ".textgrid":
        return "textgrid"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "ds_csv"
    if suffix == ".txt":
        return "audacity"
    raise ValueError(f"Can't detect label format of {path}")


def dumps(label: Label, fmt: str, tier: str = "phones") -> str:
    fmt = normalize_format(fmt)
    if fmt == "ds_csv":
        return ds_csv.dumps_rows([ds_csv.row_from_label("item", label)])
    module = FORMATS[fmt]
    return module.dumps(label, tier=tier)


def loads(text: str, fmt: str, tier: str = "phones") -> Label:
    fmt = normalize_format(fmt)
    if fmt == "ds_csv":
        rows = ds_csv.loads_rows(text)
        if not rows:
            raise ValueError("Empty transcriptions.csv")
        return ds_csv.label_from_row(rows[0])
    module = FORMATS[fmt]
    return module.loads(text, tier=tier)


def read(path: Path, fmt: str | None = None) -> Label:
    fmt = normalize_format(fmt) if fmt else detect_format(path)
    return loads(path.read_text(encoding="utf-8-sig"), fmt)


def write(label: Label, path: Path, fmt: str, tier: str = "phones") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(label, fmt, tier=tier), encoding="utf-8")
    return path
