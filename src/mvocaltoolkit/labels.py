"""Label model shared by all engines and formats.

A label is a set of named tiers ("phones", "words", "notes"...), each tier is a list of intervals in seconds.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

SILENCE = "SP"
BREATH = "AP"


class Interval(BaseModel):
    start: float
    end: float
    text: str
    confidence: float | None = None
    # free-form extra data (e.g. MIDI pitch for notes)
    data: dict[str, Any] | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class Label(BaseModel):
    tiers: dict[str, list[Interval]] = Field(default_factory=dict)
    duration: float | None = None
    # overall confidence reported by the engine
    confidence: float | None = None

    @property
    def phones(self) -> list[Interval]:
        return self.tiers.setdefault("phones", [])

    @property
    def words(self) -> list[Interval]:
        return self.tiers.setdefault("words", [])

    def end_time(self) -> float:
        ends = [iv.end for tier in self.tiers.values() for iv in tier]
        if self.duration is not None:
            ends.append(self.duration)
        return max(ends, default=0.0)

    @classmethod
    def from_segments(cls, segments: list[tuple[float, float, str]], tier: str = "phones") -> "Label":
        return cls(tiers={tier: [Interval(start=s, end=e, text=t) for s, e, t in segments]})


def fill_gaps(intervals: list[Interval], duration: float | None = None, filler: str = SILENCE) -> list[Interval]:
    """Inserts `filler` intervals into gaps so that the tier covers [0, duration] without holes."""
    result: list[Interval] = []
    cursor = 0.0
    eps = 1e-6
    for iv in sorted(intervals, key=lambda x: x.start):
        if iv.start > cursor + eps:
            result.append(Interval(start=cursor, end=iv.start, text=filler))
        result.append(iv)
        cursor = max(cursor, iv.end)
    if duration is not None and duration > cursor + eps:
        result.append(Interval(start=cursor, end=duration, text=filler))
    return result
