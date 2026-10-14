"""An oto.ini entry sung as a note: the pure helpers of the vocoder worker (no WORLD or vocoder needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ENGINES = Path(__file__).resolve().parent.parent / "src" / "mvocaltoolkit" / "engines"
sys.path.insert(0, str(ENGINES / "_runtime"))
_spec = importlib.util.spec_from_file_location("vocoder_worker", ENGINES / "vocoder" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)  # type: ignore[union-attr]


def test_note_names():
    assert worker.tone_name(60) == "C4"
    assert worker.tone_name(61) == "C#4"
    assert worker.tone_name(69) == "A4"
    assert worker.tone_name(47) == "B2"


def test_entry_span_follows_the_cutoff_rules():
    # negative cutoff: length from offset; values kept inside the sound
    assert worker.oto_span(100, 250, -700, 200, 50, 1000.0) == (100, 800, 250, 200, 50)
    # positive cutoff: from the end of the file
    assert worker.oto_span(100, 50, 300, 80, -10, 1000.0) == (100, 700, 50, 80, -10)
    # consonant and preutterance past the end are cut to it
    assert worker.oto_span(900, 500, 0, 400, 20, 1000.0) == (900, 1000.0, 100.0, 100.0, 20)


def test_two_notes_join_like_utau():
    # one note: sound from the offset, preutterance + length long, fading in over the overlap
    assert worker.note_layout(150, 50, 600, 1) == [(0.0, 750, 50, 30.0)]
    # two: the second sound starts a preutterance before its note, the first fades out over the overlap
    first, second = worker.note_layout(150, 50, 600, 2)
    assert first == (0.0, 650, 50, 50)
    assert second == (600, 750, 50, 30.0)


def test_resampler_gets_the_utau_command_line():
    args = worker.resampler_args("r.exe", "in.wav", "out.wav", 60, "g-5", 100, 523, 150, -700)
    exe = args.index("r.exe")
    assert args[exe:] == ["r.exe", "in.wav", "out.wav", "C4", "100", "g-5", "100", "550", "150", "-700", "100", "0",
                          "!120", "AA#107#"]
