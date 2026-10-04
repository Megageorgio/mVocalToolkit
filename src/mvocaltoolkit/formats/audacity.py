"""Audacity label track: "start<TAB>end<TAB>text", times in seconds."""

from __future__ import annotations

from ..labels import Interval, Label


def dumps(label: Label, tier: str = "phones") -> str:
    lines = [f"{iv.start:.6f}\t{iv.end:.6f}\t{iv.text}" for iv in label.tiers.get(tier, [])]
    return "\n".join(lines) + ("\n" if lines else "")


def loads(text: str, tier: str = "phones") -> Label:
    intervals = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0].startswith("\\"):
            continue  # skip spectral selection lines
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        intervals.append(Interval(start=start, end=end, text=parts[2] if len(parts) > 2 else ""))
    return Label(tiers={tier: intervals})
