"""HTK label (.lab): one interval per line, "start end text", times in 100 ns units.

Lines with only a phoneme (no times) are accepted too; they get zero times.
"""

from __future__ import annotations

from ..labels import Interval, Label

UNIT = 10_000_000


def dumps(label: Label, tier: str = "phones") -> str:
    lines = []
    for iv in label.tiers.get(tier, []):
        lines.append(f"{round(iv.start * UNIT)} {round(iv.end * UNIT)} {iv.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def loads(text: str, tier: str = "phones") -> Label:
    intervals = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and _is_number(parts[0]) and _is_number(parts[1]):
            intervals.append(Interval(start=int(parts[0]) / UNIT, end=int(parts[1]) / UNIT, text=parts[2]))
        else:
            intervals.append(Interval(start=0.0, end=0.0, text=line))
    return Label(tiers={tier: intervals})


def _is_number(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False
