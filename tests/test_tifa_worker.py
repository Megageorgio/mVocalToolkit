"""Pure helpers of the TIFA worker: optional breaths and cutting long files at pauses (no TIFA needed)."""

from __future__ import annotations

import sys
from pathlib import Path

ENGINES = Path(__file__).resolve().parent.parent / "src" / "mvocaltoolkit" / "engines"
sys.path.insert(0, str(ENGINES / "_runtime"))
sys.path.insert(0, str(ENGINES / "tifa"))

import worker  # noqa: E402

BREATH = worker._breath_word()


def test_optional_breaths_in_text_words_and_phonemes():
    text, pfml = worker._source({"text": "раз, два три."}, "auto", None, "ru", breaths=True)
    assert pfml and text.count(BREATH) == 3  # start, after "раз," and after "три."
    assert text.startswith(BREATH) and "два три." in text
    words, pfml = worker._source({"words": ["раз", "SP", "два"]}, "auto", None, "ru", breaths=True)
    assert pfml and words.count(BREATH) == 2
    phones, _ = worker._source({"phonemes": ["a", "SP", "SP", "b"]}, "none", None, "ru", breaths=True)
    assert phones.count(BREATH) == 2 and phones.count("<word") == 4
    # without the option nothing changes
    assert worker._source({"text": "раз, два"}, "auto", None, "ru") == ("раз, два", False)
    assert BREATH not in worker._source({"words": ["раз", "SP", "два"]}, "auto", None, "ru")[0]


def test_unheard_breaths_are_dropped():
    rows = [[0.0, 0.5, "a"], [0.5, 0.52, "AP"], [0.52, 1.0, "b"], [1.0, 1.4, "AP"]]
    assert worker._drop_unheard_breaths(rows) == [[0.0, 0.52, "a"], [0.52, 1.0, "b"], [1.0, 1.4, "AP"]]
    assert worker._drop_unheard_breaths(rows, join=False) == [[0.0, 0.5, "a"], [0.52, 1.0, "b"], [1.0, 1.4, "AP"]]


def test_cut_points_only_at_pauses():
    phones = [[0, 9.5, "a"], [9.5, 10.5, "SP"], [10.5, 19.0, "b"], [19.0, 19.4, "AP"], [19.4, 40, "c"]]
    # a quiet spot inside "c" (30 s) is not a pause: no cut there
    silences = [(9.6, 10.4), (29.8, 30.2)]
    assert worker._cut_points(40, silences, phones, max_length=25) == [19.0]
    assert worker._cut_points(40, silences, phones, max_length=15) == [10.0, 19.0]
    assert worker._cut_points(20, silences, phones, max_length=25) == []


def test_silences_found_in_audio():
    try:
        import numpy as np
    except ImportError:  # numpy lives in the engine environments, not in the core
        return

    sr = 16000
    t = np.arange(sr * 3) / sr
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio[sr:int(sr * 1.5)] = 0  # 0.5 s of silence
    found = worker._silences(audio, sr, min_silence=0.3)
    assert len(found) == 1 and abs(found[0][0] - 1.0) < 0.05 and abs(found[0][1] - 1.5) < 0.05


def test_texts_follow_their_words():
    old = [[0, 1, "ne"], [1, 2, "ko"], [2, 3, "SP"], [3, 4, "AP"], [4, 5, "a"]]
    new = [[0, 1.1, "ne"], [1.1, 2.2, "ko"], [4.1, 5, "a"]]
    texts = [[0, 2, "猫"], [4, 5, "あ"]]
    assert worker._move_texts(texts, old, new) == [[0, 2.2, "猫"], [4.1, 5, "あ"]]
    assert worker._move_texts(texts, old, new[:2]) is None
