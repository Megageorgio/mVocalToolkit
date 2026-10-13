"""MCFL engine worker: an exported MCFL package (model.onnx, labels.txt, langs.txt, ...) run with its own
mcfl_infer.py when it has one, else with the copy next to this worker (github.com/Megageorgio/MCFL, MIT)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import mvt_engine as rt

_models: dict[str, Any] = {}


def _runtime(root: Path):
    """The mcfl_infer module of the package (it matches the model), else the bundled one."""
    own = root / "mcfl_infer.py"
    if own.is_file():
        spec = importlib.util.spec_from_file_location(f"mcfl_infer_{abs(hash(str(root)))}", own)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    import mcfl_infer  # noqa: PLC0415

    return mcfl_infer


def _load(path: str) -> Any:
    if path not in _models:
        root = Path(path)
        threads = max(1, min(4, os.cpu_count() or 1))
        _models[path] = _runtime(root).MCFL(str(root), threads=threads)
    return _models[path]


@rt.method()
def segment(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    language: str | None = None,
    silence_threshold: float | None = None,
    min_silence_duration: float | None = None,
    **_ignored: Any,  # options of the WFL-ASR engines
) -> list[dict[str, Any]]:
    m = _load(model["path"])
    langs = list(m.langs)
    code = (language or "").lower().split("-")[0].split("_")[0]
    if code not in m.langs:
        code = langs[0]
        rt.log(f"MCFL: language {language!r} is not in the model ({', '.join(langs)}), using {code}")
    # the request's silence options over the package's own (config.json); the Viterbi bias stays the package's,
    # picked for this model when it was exported (the WFL-ASR default of the request doesn't suit it)
    for key, value in (("silence_threshold", silence_threshold), ("min_silence_sec", min_silence_duration)):
        if value is not None:
            m.cfg[key] = value
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / len(items), f"Segmenting {name}")
        try:
            segments = m.label_file(item["audio"], lang=code)
            results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in segments]})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()
