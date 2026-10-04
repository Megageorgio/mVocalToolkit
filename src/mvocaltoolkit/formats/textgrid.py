"""Praat TextGrid (interval tiers only). Writes the long text format, reads both long and short formats."""

from __future__ import annotations

import re

from ..labels import Interval, Label


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def dumps(label: Label, tier: str | None = None) -> str:
    tiers = label.tiers if tier is None or tier == "phones" else {tier: label.tiers.get(tier, [])}
    # Praat convention used by most SVS tools: words first, then phones
    order = sorted(tiers.keys(), key=lambda name: {"words": 0, "phones": 1}.get(name, 2))
    xmax = label.end_time()
    out = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax}",
        "tiers? <exists>",
        f"size = {len(order)}",
        "item []:",
    ]
    for index, name in enumerate(order, start=1):
        intervals = tiers[name]
        out += [
            f"    item [{index}]:",
            '        class = "IntervalTier"',
            f"        name = {_quote(name)}",
            "        xmin = 0",
            f"        xmax = {xmax}",
            f"        intervals: size = {len(intervals)}",
        ]
        for i, iv in enumerate(intervals, start=1):
            out += [
                f"        intervals [{i}]:",
                f"            xmin = {iv.start}",
                f"            xmax = {iv.end}",
                f"            text = {_quote(iv.text)}",
            ]
    return "\n".join(out) + "\n"


_TOKEN = re.compile(r'"((?:[^"]|"")*)"|([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(<exists>|<absent>)')


def loads(text: str, tier: str = "phones") -> Label:
    """Parses both long and short TextGrid formats by reading the sequence of values."""
    tokens: list[str | float] = []
    # drop "item [1]:" / "intervals [2]:" indices of the long format (they aren't values)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    for match in _TOKEN.finditer(text):
        if match.group(1) is not None:
            tokens.append(match.group(1).replace('""', '"'))
        elif match.group(2) is not None:
            tokens.append(float(match.group(2)))
        else:
            tokens.append(match.group(3))
    # tokens: "ooTextFile", "TextGrid", xmin, xmax, <exists>, size, then tiers
    pos = 0

    def take():
        nonlocal pos
        value = tokens[pos]
        pos += 1
        return value

    take(), take()  # file type, object class
    take(), take()  # xmin, xmax
    if pos < len(tokens) and tokens[pos] in ("<exists>", "<absent>"):
        take()
    size = int(take())  # type: ignore[arg-type]
    label = Label()
    for _ in range(size):
        tier_class = take()
        name = str(take())
        take(), take()  # xmin, xmax
        count = int(take())  # type: ignore[arg-type]
        intervals = []
        if tier_class == "IntervalTier":
            for _ in range(count):
                start = float(take())  # type: ignore[arg-type]
                end = float(take())  # type: ignore[arg-type]
                value = str(take())
                intervals.append(Interval(start=start, end=end, text=value))
        else:  # TextTier (points): time, mark
            for _ in range(count):
                t = float(take())  # type: ignore[arg-type]
                value = str(take())
                intervals.append(Interval(start=t, end=t, text=value))
        label.tiers[_tier_name(name)] = intervals
    return label


def _tier_name(name: str) -> str:
    lowered = name.lower()
    if lowered in ("phones", "phone", "phonemes", "phoneme", "ph"):
        return "phones"
    if lowered in ("words", "word"):
        return "words"
    return name
