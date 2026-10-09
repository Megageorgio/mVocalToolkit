"""Boundary refiner engine worker (mRefinerModel): moves the boundaries of given segments, keeps the phonemes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mvt_engine as rt

_cache: dict[str, Any] = {}


def _load(model: dict[str, Any], device: str):
    from mrefinermodel import Refiner  # noqa: PLC0415  (mRefinerModel)

    root = Path(model["path"])
    layout = model.get("layout", {})
    key = f"{root}|{device}"
    if key not in _cache:
        _cache.clear()
        rt.free_memory()
        _cache[key] = Refiner(str(root), checkpoint=layout.get("checkpoint"), device=device)
    return _cache[key]


@rt.method()
def refine(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    mode: str = "auto",
    phone_map: dict[str, str] | None = None,
    min_confidence: float | None = None,
    max_shift_ms: float | None = None,
    **_ignored: Any,
) -> list[dict[str, Any]]:
    import soundfile as sf  # noqa: PLC0415

    refiner = _load(model, rt.device())
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / max(1, len(items)), f"Refining {name}")
        try:
            audio, sr = sf.read(item["audio"], dtype="float32")
            segments = [(float(s), float(e), str(p)) for s, e, p, *_ in item.get("phones") or []]
            out, info = refiner.refine(audio, sr, segments, mode=mode, phone_map=phone_map or None,
                                       min_confidence=min_confidence, max_shift_ms=max_shift_ms, return_info=True)
            results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in out],
                            "info": {k: (float(v) if isinstance(v, float) else v) for k, v in info.items()}})
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Refined")
    return results


if __name__ == "__main__":
    rt.run()
