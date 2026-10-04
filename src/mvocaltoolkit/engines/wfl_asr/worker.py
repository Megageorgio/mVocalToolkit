"""WFL-ASR engine worker (runs with the WFL-ASR source on sys.path)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import mvt_engine as rt
import yaml


def _prepare_config(model_root: Path, layout: dict[str, Any], work: Path) -> Path:
    """The model's config points to its training folder; rewrite paths to the downloaded model folder."""
    config_path = model_root / layout["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    phonemes = next(iter(sorted(model_root.rglob("phonemes.txt"))), None)
    save_dir = phonemes.parent if phonemes else model_root
    config.setdefault("output", {})["save_dir"] = str(save_dir)
    patched = work / "config.yaml"
    patched.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return patched


@rt.method()
def segment(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    lang_id: int | None = None,
    sample: bool = False,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    from infer import infer_audio  # noqa: PLC0415  (WFL-ASR)

    device = rt.device()
    root = Path(model["path"])
    layout = model.get("layout", {})
    results = []
    with tempfile.TemporaryDirectory(prefix="mvt-wfl-") as tmp:
        work = Path(tmp)
        config = _prepare_config(root, layout, work)
        if confidence_threshold is None:
            confidence_threshold = (yaml.safe_load(config.read_text(encoding="utf-8")).get("postprocess") or {}).get(
                "confidence_threshold", 0.0
            )
        for index, item in enumerate(items):
            name = item.get("name") or Path(item["audio"]).stem
            rt.progress(index / len(items), f"Segmenting {name}")
            try:
                # isolated copy: WFL-ASR reads <name>.txt next to the audio as a forced phoneme list
                # and writes caches next to it
                audio = work / f"{index}{Path(item['audio']).suffix.lower() or '.wav'}"
                shutil.copyfile(item["audio"], audio)
                if item.get("phonemes"):
                    audio.with_suffix(".txt").write_text(" ".join(item["phonemes"]), encoding="utf-8")
                segments = infer_audio(
                    audio_path=str(audio),
                    config_path=str(config),
                    checkpoint_path=str(root / layout["checkpoint"]),
                    output_lab_path=None,
                    device=device,
                    lang_id=lang_id,
                    sample=sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    confidence_threshold=confidence_threshold or 0.0,
                )
                results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in segments]})
            except Exception as e:  # noqa: BLE001
                if rt.is_oom(e):
                    rt.free_memory()
                results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()
