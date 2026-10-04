"""Tempo estimation worker (DeepRhythm)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mvt_engine as rt

_predictor = None


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
            try:
                bpm, confidence = predictor.predict(item["audio"], include_confidence=True)
            except TypeError:
                bpm, confidence = predictor.predict(item["audio"]), None
            results.append({"ok": True, "name": item.get("name"), "bpm": float(bpm),
                            "confidence": float(confidence) if confidence is not None else None})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "name": item.get("name"), "error": f"{type(e).__name__}: {e}"})
    return results


if __name__ == "__main__":
    rt.run()
