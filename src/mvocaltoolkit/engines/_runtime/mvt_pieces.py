"""Long recordings aligned again by segments cut at pauses: finding clear silences, choosing the cuts from a first
pass, and filling the gaps of the joined result with SP. Shared by the aligners' workers."""

from __future__ import annotations

from typing import Any

SILENCE = "SP"
BREATH = "AP"


def with_silence(rows: list[list[Any]], duration: float, min_gap: float = 0.015) -> list[list[Any]]:
    """Gaps (empty intervals and uncovered time) become SP, as in the other aligners' labels.
    A gap of a frame or so (left by an omitted zero-width phoneme) goes to the interval before it."""
    out: list[list[Any]] = []
    t = 0.0
    for start, end, text in rows:
        if out and t + 1e-6 < start < t + min_gap:
            out[-1][1] = start
        elif start > t + 1e-6:
            out.append([t, start, SILENCE])
        out.append([start, end, text or SILENCE])
        t = end
    if duration > t + 1e-6:
        out.append([t, duration, SILENCE])
    merged: list[list[Any]] = []
    for row in out:
        if merged and row[2] == SILENCE and merged[-1][2] == SILENCE:
            merged[-1][1] = row[1]
        else:
            merged.append(row)
    return merged


def silences(audio, sr: int, min_silence: float, below_db: float = 30.0) -> list[tuple[float, float]]:
    """Clear silences: at least min_silence long and below_db quieter than the loud parts of the file."""
    import numpy as np  # noqa: PLC0415

    hop, win = int(sr * 0.01), int(sr * 0.03)
    if len(audio) < win:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    loud = np.percentile(rms, 95)
    if loud <= 0:
        return []
    quiet = rms < loud * 10 ** (-below_db / 20)
    out, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.01 >= min_silence:
                out.append((start * 0.01 + 0.015, i * 0.01 + 0.015))
            start = None
    return out


def cut_points(duration: float, silences: list[tuple[float, float]], phones: list[list[Any]],
               max_length: float, min_piece: float = 1.0, min_breath: float = 0.15) -> list[float]:
    """Cuts between phrases: in the middle of clear silences where the first pass has no sound (only SP),
    and at the start of breaths (AP, the breath goes to the next segment). Segments of up to max_length seconds
    when the pauses allow it."""
    sounds = [(s, e) for s, e, p in phones if p != SILENCE]
    candidates = [c for c in ((a + b) / 2 for a, b in silences)
                  if not any(s - 0.02 < c < e + 0.02 for s, e in sounds)]
    candidates = sorted(candidates + [s for s, e, p in phones if p == BREATH and e - s >= min_breath])
    cuts, start = [], 0.0
    while duration - start > max_length:
        within = [c for c in candidates if start + min_piece < c <= start + max_length]
        later = [c for c in candidates if c > start + max_length and duration - c > min_piece]
        cut = max(within) if within else (later[0] if later else None)
        if cut is None:
            break
        cuts.append(cut)
        start = cut
    return cuts
