"""DiffSinger .ds phrases; temporal tiers for training-data import."""
import json
import math

from ..labels import Interval, Label
from . import ds_csv


def loads(text: str, tier: str = "phones") -> Label:
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data or any(not isinstance(s, dict) for s in data):
        raise ValueError("Expected a non-empty list of .ds segments")
    result = Label()
    for segment in data:
        offset = float(segment.get("offset", 0))
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Invalid .ds segment offset")
        label = ds_csv.label_from_row(segment)
        if not label.phones:
            raise ValueError("Training .ds needs ph_seq and ph_dur")
        if segment.get("ph_num") and not label.words:
            counts = [int(n) for n in str(segment["ph_num"]).split()]
            if any(n < 1 for n in counts) or sum(counts) != len(label.phones):
                raise ValueError("ph_num must group all phonemes exactly once")
            texts = str(segment.get("text", "")).split()
            pos = 0
            for i, count in enumerate(counts):
                group = label.phones[pos:pos + count]
                label.words.append(Interval(start=group[0].start, end=group[-1].end,
                                            text=texts[i] if len(texts) == len(counts) else "_"))
                pos += count
        for name, entries in label.tiers.items():
            result.tiers.setdefault(name, []).extend(Interval(start=p.start + offset, end=p.end + offset, text=p.text, data=p.data) for p in entries)
    for entries in result.tiers.values():
        entries.sort(key=lambda p: p.start)
    return result


def dumps(label: Label, tier: str = "phones") -> str:
    row = ds_csv.row_from_label("item", label)
    row.pop("name", None)
    return json.dumps([{"offset": 0.0, **row}], ensure_ascii=False, indent=1)
