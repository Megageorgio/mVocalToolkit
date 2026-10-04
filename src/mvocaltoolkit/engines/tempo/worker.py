"""Tempo estimation worker (DeepRhythm)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mvt_engine as rt

_predictor = None
CLIP_SECONDS = 8


def _get_predictor():
    global _predictor
    if _predictor is None:
        from deeprhythm import DeepRhythmPredictor  # noqa: PLC0415

        device = rt.device()
        try:
            _predictor = DeepRhythmPredictor(device=device)
        except TypeError:
            _predictor = DeepRhythmPredictor()
    return _predictor


@rt.method()
def tempo(items: list[dict[str, Any]], model: str = "deeprhythm") -> list[dict[str, Any]]:
    predictor = _get_predictor()
    results = []
    for index, item in enumerate(items):
        rt.progress(index / len(items), f"Estimating tempo of {Path(item['audio']).name}")
        try:
            import librosa  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            # DeepRhythm analyses 8-second clips and fails on shorter audio: short files are looped
            audio, sr = librosa.load(item["audio"], sr=22050, mono=True)
            if len(audio) == 0:
                raise ValueError("empty audio")
            clip = CLIP_SECONDS * sr
            if len(audio) < clip:
                audio = np.tile(audio, int(np.ceil(clip / len(audio))))[:clip]
            result = predictor.predict_from_audio(audio, sr, include_confidence=True)
            bpm, confidence = (result[0], result[1]) if isinstance(result, tuple) else (result, None)
            results.append({"ok": True, "name": item.get("name"), "bpm": float(bpm),
                            "confidence": float(confidence) if confidence is not None else None})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "name": item.get("name"), "error": f"{type(e).__name__}: {e}"})
    return results


if __name__ == "__main__":
    rt.run()
